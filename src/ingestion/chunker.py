"""Split documents into retrievable chunks.

Two strategies live here, and building both is the point:

`chunk_fixed` is the naive baseline — cut every N tokens, ignore structure.
`chunk_structural` walks the DocBook tree instead, breaking on section
boundaries, keeping atomic blocks whole, and prefixing each chunk with its
heading path.

Phase 5 runs retrieval over both and reports the delta. Without the baseline
there is no way to show that the structural version earned its complexity —
and if it turns out it did not, that is a result worth reporting too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import tiktoken
from lxml import etree

from src.ingestion.loader import Document

# Defaults from the plan. Not tuned yet: tuning before Phase 5 exists would be
# guessing, since there is nothing to measure "better" against.
TARGET_TOKENS = 512
OVERLAP_TOKENS = 64

# 512 is not a preference. bge-small-en-v1.5 declares
# "max_position_embeddings": 512 in its config, and text beyond that is
# silently truncated at embed time — no error, just content that was never
# indexed. Chunks may exceed it only when an atomic block forces it, and those
# get flagged so Phase 5 can measure the damage.
EMBED_MODEL_LIMIT = 512

# Elements that carry code. Splitting one leaves an unusable fragment: half a
# CREATE TABLE statement helps nobody. They are small in practice — the median
# programlisting is 28 tokens — so keeping them whole costs little.
CODE_BLOCKS = frozenset({"programlisting", "screen", "synopsis", "literallayout"})

# Tables split badly for a different reason: a row without its header row loses
# the meaning of every column. 25% of tables here exceed the embedding limit,
# which is the known cost of this rule.
TABLE_BLOCKS = frozenset({"table", "informaltable"})

ATOMIC_BLOCKS = CODE_BLOCKS | TABLE_BLOCKS

# A <variablelist> is a container, not a unit. The parameter list in config.sgml
# is 5,798 tokens across 22 entries — treating it as atomic would build one
# chunk the embedder reads 9% of. Each <varlistentry> is the real unit: a
# parameter name plus its description, median 50 tokens, and exactly what a
# question like "what is the default value of shared_buffers" should retrieve.
LIST_ITEMS = frozenset({"varlistentry", "listitem"})

# Elements that sit inside a sentence rather than starting a new block. Taken
# from what the corpus actually nests inside <para>: <literal> alone appears
# 22,533 times. Treating these as blocks split sentences into fragments that
# were then joined by blank lines — "You can create a" / "view" / "over the
# query" — which is unreadable in a retrieved chunk and pollutes the embedding.
INLINE_ELEMENTS = frozenset(
    {
        "literal",
        "type",
        "command",
        "function",
        "structfield",
        "replaceable",
        "productname",
        "option",
        "returnvalue",
        "parameter",
        "varname",
        "application",
        "filename",
        "acronym",
        "structname",
        "quote",
        "firstterm",
        "symbol",
        "emphasis",
        "envar",
        "glossterm",
        "citation",
        "systemitem",
        "token",
        "classname",
        "methodname",
        "objectname",
        "database",
        "email",
        "guilabel",
        "guimenu",
        "keycap",
        "sgmltag",
        "abbrev",
        "phrase",
        "subscript",
        "superscript",
        "trademark",
        "wordasword",
        "foreignphrase",
        "errorname",
        "errorcode",
        "constant",
    }
)

# Cross-references. They carry no text of their own — the published docs
# render them as a section number — so they leave a gap in the sentence:
# "Refer back to the queries in ." Dropping the surrounding space is closer to
# readable than leaving the stray punctuation adrift.
XREF_ELEMENTS = frozenset({"xref", "link", "ulink", "olink"})

# A list entry only earns its own chunk if it carries enough context to be
# worth retrieving alone. The shared_buffers entry (285 tokens) does; an
# acronyms.sgml entry — "AM / Access Method", 4 tokens — does not, and giving
# each one a chunk buried the index in fragments. Below this, entries pack with
# their neighbours like ordinary prose.
MIN_STANDALONE_TOKENS = 80

# Section elements, in both shapes this corpus uses: chapters nest sect1..sect4,
# while the 220 command reference pages nest refsect1..refsect3.
SECTION_ELEMENTS = frozenset(
    {f"sect{i}" for i in range(1, 6)}
    | {f"refsect{i}" for i in range(1, 4)}
    | {"simplesect", "refsynopsisdiv", "refnamediv"}
)

# Elements dropped during extraction; see loader for why.
SKIP_ELEMENTS = frozenset({"indexterm", "comment", "remark", "refmeta", "title"})

_HEADING_SEPARATOR = " > "


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


@dataclass
class Chunk:
    """One retrievable unit.

    `text` is what gets embedded and searched: the heading path followed by the
    body. That prefix is the single highest-leverage trick in the pipeline — a
    chunk reading "The default is typically 128 megabytes" answers a question
    about shared_buffers but contains none of its words, so neither BM25 nor a
    dense retriever can find it. Prefixing makes the context explicit:

        Server Configuration > Resource Consumption > shared_buffers
        The default is typically 128 megabytes.

    `body` keeps the text without the prefix, because Phase 6's grounding gate
    checks generated claims against source content and should not be able to
    "support" a claim using a heading we synthesized.
    """

    chunk_id: str
    doc_path: str
    doc_title: str
    doc_type: str
    heading_path: tuple[str, ...]
    body: str
    chunk_index: int
    token_count: int
    has_code: bool
    has_table: bool
    # True when an atomic block pushed this chunk past the embedding limit.
    # Phase 5 uses it to quantify what the never-split rule cost.
    oversized: bool = False
    strategy: str = "structural"

    @property
    def heading(self) -> str:
        return _HEADING_SEPARATOR.join(self.heading_path)

    @property
    def text(self) -> str:
        """What actually gets indexed."""
        return f"{self.heading}\n\n{self.body}" if self.heading_path else self.body

    def to_metadata(self) -> dict[str, str | int | bool]:
        """Flat metadata for the vector store.

        Chroma only accepts scalars, so the heading tuple is joined here rather
        than at the call site.
        """
        return {
            "doc_path": self.doc_path,
            "doc_title": self.doc_title,
            "doc_type": self.doc_type,
            "heading_path": self.heading,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "has_code": self.has_code,
            "has_table": self.has_table,
            "oversized": self.oversized,
            "strategy": self.strategy,
        }


# --- Naive baseline --------------------------------------------------------


def chunk_fixed(
    doc: Document,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """Cut the flattened text every `target_tokens`, ignoring all structure.

    This is what most RAG projects ship. It exists here as the control: Phase 5
    compares it against `chunk_structural` so the structural version's extra
    complexity is justified with a number rather than an assertion.
    """
    encoder = _encoder()
    token_ids = encoder.encode(doc.content, disallowed_special=())
    if not token_ids:
        return []

    stride = max(1, target_tokens - overlap_tokens)
    chunks: list[Chunk] = []

    for index, start in enumerate(range(0, len(token_ids), stride)):
        window = token_ids[start : start + target_tokens]
        if not window:
            break
        body = encoder.decode(window).strip()
        if not body:
            continue

        chunks.append(
            Chunk(
                chunk_id=f"{doc.path}::fixed::{index}",
                doc_path=doc.path,
                doc_title=doc.title,
                doc_type=doc.doc_type,
                heading_path=(),  # the whole point: no structure is known
                body=body,
                chunk_index=index,
                token_count=len(window),
                has_code=False,  # cannot tell from flat text
                has_table=False,
                strategy="fixed",
            )
        )

        # The final window is shorter than the stride; stop rather than emit a
        # duplicate tail.
        if start + target_tokens >= len(token_ids):
            break

    return chunks


# --- Structure-aware -------------------------------------------------------


@dataclass
class _Block:
    """A piece of a section, with two independent properties.

    `atomic` means "never cut this in half" — a code block or a table.
    `standalone` means "do not pack this with neighbours" — a parameter entry,
    which is a complete unit on its own.

    They are not the same thing, and conflating them is a trap. A 28-token
    <programlisting> must not be split, but it should still sit in the same
    chunk as the paragraph explaining it; isolating every atomic block drove
    the median chunk down to 40 tokens, roughly one sentence, which is too
    little context to retrieve or generate from.
    """

    text: str
    tokens: int
    atomic: bool
    standalone: bool = False
    is_code: bool = False
    is_table: bool = False


@dataclass
class _Section:
    """A leaf section: its heading path and the blocks directly inside it."""

    heading_path: tuple[str, ...]
    blocks: list[_Block] = field(default_factory=list)


def _local(element: etree._Element) -> str | None:
    return etree.QName(element).localname if isinstance(element.tag, str) else None


def _element_title(element: etree._Element) -> str:
    """The heading for a section element, in whichever shape it uses."""
    for xpath in ("title", "refmeta/refentrytitle", "refnamediv/refname"):
        found = element.find(xpath)
        if found is not None:
            text = " ".join("".join(found.itertext()).split())
            if text:
                return text
    return ""


def _render(element: etree._Element) -> str:
    """Readable text for one block.

    Code keeps its indentation verbatim; prose gets its source line wrapping
    collapsed. Same rules as the loader, applied per block here.
    """
    tag = _local(element)
    raw = "".join(element.itertext())
    if tag in CODE_BLOCKS:
        return raw.strip("\n")
    return re.sub(r"\s+", " ", raw).strip()


def _contains_code(element: etree._Element) -> bool:
    return any(
        _local(d) in CODE_BLOCKS for d in element.iter() if isinstance(d.tag, str)
    )


def _is_block_level(tag: str | None) -> bool:
    """Whether an element starts its own block rather than sitting in a sentence.

    DocBook mixes the two freely inside a <para>: <literal> and <xref> are
    inline, <programlisting> and <table> are not. Anything unrecognised is
    treated as block-level so nested structure is still walked into — a wrong
    guess there costs a paragraph break, while the reverse would swallow a code
    sample into a sentence.
    """
    if tag is None:
        return False
    return tag not in INLINE_ELEMENTS and tag not in XREF_ELEMENTS


def _collect_blocks(element: etree._Element) -> list[_Block]:
    """Flatten one section's own content into blocks.

    Nested sections are skipped — `_walk_sections` visits them separately so
    they get their own heading path. Atomic blocks are emitted whole. List
    containers are descended into so each entry becomes its own block.
    """
    blocks: list[_Block] = []

    def emit(
        text: str,
        *,
        atomic: bool,
        standalone: bool = False,
        code: bool = False,
        table: bool = False,
    ) -> None:
        # Prose keeps DocBook's ~70-column source wrapping until here; code was
        # already rendered verbatim and must not be touched.
        if atomic:
            text = text.strip()
        else:
            text = re.sub(r"\s+", " ", text)
            # An empty <xref/> leaves a space before the punctuation that
            # followed it: "Refer back to the queries in ."
            text = re.sub(r"\s+([,.;:)])", r"\1", text).strip()
        if text:
            blocks.append(
                _Block(
                    text=text,
                    tokens=count_tokens(text),
                    atomic=atomic,
                    standalone=standalone,
                    is_code=code,
                    is_table=table,
                )
            )

    def visit(node: etree._Element, *, is_root: bool = False) -> None:
        if not isinstance(node.tag, str):
            return  # comments and processing instructions carry no content

        tag = _local(node)

        if tag in SKIP_ELEMENTS:
            return
        # Only the section we were asked about is entered; its subsections
        # belong to their own heading path.
        if tag in SECTION_ELEMENTS and not is_root:
            return

        if tag in ATOMIC_BLOCKS:
            emit(
                _render(node),
                atomic=True,
                code=tag in CODE_BLOCKS,
                table=tag in TABLE_BLOCKS,
            )
            return

        # A list entry is one semantic unit: a parameter name plus its
        # description. Held together unless it alone exceeds the target, in
        # which case it is broken down like ordinary content.
        if tag in LIST_ITEMS and not is_root:
            text = _render_children(node)
            if text and (tokens := count_tokens(text)) <= TARGET_TOKENS:
                # Substantial entries stand alone, so a question about
                # shared_buffers retrieves a chunk about shared_buffers rather
                # than one that is half work_mem. Short entries pack normally.
                emit(
                    text,
                    atomic=True,
                    standalone=tokens >= MIN_STANDALONE_TOKENS,
                    code=_contains_code(node),
                )
                return
            # too large to keep whole — fall through and recurse

        # Walk the children, accumulating inline text into one run and
        # breaking only at block-level elements.
        #
        # The distinction matters: a paragraph mixes <literal> and <xref>,
        # which belong mid-sentence, with <programlisting>, which does not.
        # Treating every child alike split sentences into fragments joined by
        # blank lines — "You can create a" / "view" / "over the query".
        run: list[str] = []

        def flush_run() -> None:
            if run:
                emit(" ".join(run), atomic=False)
                run.clear()

        if node.text and node.text.strip():
            run.append(node.text)

        for child in node:
            if not isinstance(child.tag, str):
                continue
            child_tag = _local(child)
            if child_tag in SKIP_ELEMENTS:
                pass
            elif _is_block_level(child_tag):
                flush_run()
                visit(child)
            else:
                # Inline: its text belongs in the sentence being built. An
                # <xref/> is empty, so nothing is appended and the sentence
                # closes over the gap.
                inline = "".join(child.itertext())
                if inline:
                    run.append(inline)
            if child.tail and child.tail.strip():
                run.append(child.tail)

        flush_run()

    def _render_children(node: etree._Element) -> str:
        parts = [
            rendered
            for child in node
            if isinstance(child.tag, str) and _local(child) not in SKIP_ELEMENTS
            if (rendered := _render(child))
        ]
        return "\n".join(parts) if parts else _render(node)

    visit(element, is_root=True)
    return blocks


def _walk_sections(element: etree._Element, path: tuple[str, ...]) -> list[_Section]:
    """Build the section tree, depth first.

    Content sitting directly under a node becomes that node's own section;
    nested sections become their own, each carrying the full heading path down
    from the document root. That path is what gets prefixed onto every chunk.
    """
    title = _element_title(element)
    current_path = (*path, title) if title else path

    sections: list[_Section] = []
    own_blocks = _collect_blocks(element)
    if own_blocks:
        sections.append(_Section(heading_path=current_path, blocks=own_blocks))

    for child in element:
        if _local(child) in SECTION_ELEMENTS:
            sections.extend(_walk_sections(child, current_path))

    return sections


def _pack(
    blocks: list[_Block],
    target_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, int, bool, bool, bool]]:
    """Greedily fill chunks up to `target_tokens`, never splitting an atomic block.

    Returns (text, tokens, has_code, has_table, oversized) per chunk.
    """
    packed: list[tuple[str, int, bool, bool, bool]] = []
    buffer: list[_Block] = []
    buffered_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffered_tokens
        if not buffer:
            return
        text = "\n\n".join(b.text for b in buffer).strip()
        tokens = buffered_tokens
        packed.append(
            (
                text,
                tokens,
                any(b.is_code for b in buffer),
                any(b.is_table for b in buffer),
                tokens > EMBED_MODEL_LIMIT,
            )
        )
        # Carry a tail of the previous chunk forward, so a sentence that lands
        # on a boundary still appears whole somewhere. Only non-atomic blocks
        # are carried: repeating a whole code block wastes the budget.
        carry: list[_Block] = []
        carried = 0
        for block in reversed(buffer):
            if block.atomic or carried + block.tokens > overlap_tokens:
                break
            carry.insert(0, block)
            carried += block.tokens
        buffer = carry
        buffered_tokens = carried

    for block in blocks:
        # Two reasons to give a block its own chunk, and only two:
        #
        #   standalone — a parameter entry is a complete unit; packing it with
        #                the next parameter blurs what the chunk is about.
        #   oversized  — it does not fit, and splitting it would leave an
        #                unusable fragment. The plan calls this correct.
        #
        # Everything else atomic still gets packed with its neighbours: a code
        # sample belongs with the paragraph that introduces it.
        if block.standalone or (block.atomic and block.tokens > target_tokens):
            flush()
            buffer = []
            buffered_tokens = 0
            packed.append(
                (
                    block.text,
                    block.tokens,
                    block.is_code,
                    block.is_table,
                    block.tokens > EMBED_MODEL_LIMIT,
                )
            )
            continue

        if buffered_tokens + block.tokens > target_tokens and buffer:
            flush()

        buffer.append(block)
        buffered_tokens += block.tokens

    flush()
    return packed


def chunk_structural(
    doc: Document,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split a document along its own structure.

    Falls back to fixed-size chunking when there is no parsed tree to walk.
    """
    if doc.raw_xml is None:
        return chunk_fixed(doc, target_tokens, overlap_tokens)

    sections = _walk_sections(doc.raw_xml, ())
    if not sections:
        return chunk_fixed(doc, target_tokens, overlap_tokens)

    chunks: list[Chunk] = []
    index = 0
    for section in sections:
        for text, tokens, has_code, has_table, oversized in _pack(
            section.blocks, target_tokens, overlap_tokens
        ):
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.path}::structural::{index}",
                    doc_path=doc.path,
                    doc_title=doc.title,
                    doc_type=doc.doc_type,
                    heading_path=section.heading_path,
                    body=text,
                    chunk_index=index,
                    token_count=tokens,
                    has_code=has_code,
                    has_table=has_table,
                    oversized=oversized,
                )
            )
            index += 1

    return chunks or chunk_fixed(doc, target_tokens, overlap_tokens)


def chunk_corpus(
    docs: list[Document],
    strategy: str = "structural",
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk every document with the named strategy."""
    if strategy not in ("structural", "fixed"):
        raise ValueError(f"unknown strategy: {strategy!r}")

    splitter = chunk_structural if strategy == "structural" else chunk_fixed
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(splitter(doc, target_tokens, overlap_tokens))
    return chunks
