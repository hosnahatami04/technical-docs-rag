"""Read the raw PostgreSQL DocBook corpus into Document objects.

The corpus ships as DocBook XML with a .sgml extension (PostgreSQL moved off
real SGML in 2017 but kept the filenames). It is well-formed XML apart from a
handful of custom entities, which we substitute before parsing.

One source file becomes one Document. That granularity is deliberate: the
evaluation labels gold answers by source file (`gold_doc_paths`), so the unit
the loader emits is the unit Hit@k is scored against. Splitting into retrievable
pieces is Phase 2's job, not this module's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

# --- Custom entities -------------------------------------------------------
# An XML parser only knows the five built-ins (&lt; &gt; &amp; &quot; &apos;).
# Everything else PostgreSQL defines in postgres.sgml's DTD subset, which we do
# not load. Substituting them textually is simpler and more predictable than
# wiring up an entity resolver.
_ENTITIES: dict[str, str] = {
    "&commit_baseurl;": "https://git.postgresql.org/pg/commitdiff/",
    "&version;": "17.0",
    "&majorversion;": "17",
    # Typographic characters -> their plain-text equivalents.
    "&mdash;": "—",
    "&ndash;": "–",
    "&nbsp;": " ",
    "&zwsp;": "",  # zero-width space: pure layout, carries no meaning
    "&bull;": "•",
    "&sect;": "§",
    "&rarr;": "→",
    "&larr;": "←",
    "&copy;": "©",
    "&reg;": "®",
    "&trade;": "™",
    "&hellip;": "…",
    "&ldquo;": "“",
    "&rdquo;": "”",
    "&lsquo;": "‘",
    "&rsquo;": "’",
    # Accented letters appearing in author names and prose.
    "&aacute;": "á",
    "&eacute;": "é",
    "&iacute;": "í",
    "&oacute;": "ó",
    "&uacute;": "ú",
    "&ouml;": "ö",
    "&uuml;": "ü",
    "&auml;": "ä",
    "&ccedil;": "ç",
    "&ntilde;": "ñ",
    "&ocirc;": "ô",
    "&ecirc;": "ê",
    "&egrave;": "è",
    "&agrave;": "à",
    "&szlig;": "ß",
    "&aring;": "å",
    "&oslash;": "ø",
    "&aelig;": "æ",
}

# Any &name; we did not list above. Dropping the markup but keeping the name is
# better than leaving the raw entity in, which would break the parser.
_UNKNOWN_ENTITY = re.compile(r"&(?!(?:lt|gt|amp|quot|apos|#)\b)([A-Za-z][\w.-]*);")

# Elements that hold code rather than prose. Phase 2 must never split these;
# Phase 1 only needs to know they exist so it can preserve their whitespace.
CODE_ELEMENTS = frozenset({"programlisting", "screen", "synopsis", "literallayout"})

# Elements whose text is navigational or indexing metadata, not content.
# Index terms in particular repeat words already in the prose, which would skew
# both BM25 term frequencies and the corpus statistics. <refmeta> holds man-page
# bookkeeping ("CREATE TABLE 7 SQL - Language Statements") that reads as content
# but is not.
_DROP_ELEMENTS = frozenset({"indexterm", "comment", "remark", "refmeta"})

# Root elements that mark a file as a real document rather than a fragment.
_DOC_ROOTS = frozenset(
    {"chapter", "sect1", "refentry", "appendix", "part", "preface", "article"}
)

# Elements that should start a new line when flattened to text.
_BLOCK_OPEN = frozenset({"title", "para", "listitem", "row", "term", "entry"})
_BLOCK_CLOSE = frozenset({"title", "para", "listitem", "row"})


@dataclass
class Document:
    """One source file from the corpus.

    `raw_xml` is kept alongside the flattened `content` on purpose: Phase 2
    needs the element tree to find heading boundaries and atomic blocks, and
    that structure cannot be recovered from flat text.
    """

    path: str  # relative to the corpus root — the key gold labels use
    title: str
    content: str
    doc_type: str
    raw_xml: etree._Element | None = field(default=None, repr=False, compare=False)

    @property
    def token_estimate(self) -> int:
        """Rough token count. ~4 characters per token holds well for English
        technical prose; corpus_stats.py does the exact count when it matters.
        """
        return len(self.content) // 4


def classify(path: Path, root_tag: str) -> str:
    """Label a document so Phase 5 can break results down by kind.

    The hypothesis worth testing later is that BM25 wins on `config` and
    `sql_command` (dense identifier lookups like max_wal_size) while dense
    retrieval wins on `guide` (conceptual prose). That comparison is only
    possible if the label is recorded now.

    Classification leans on structure first and filenames second. The corpus
    puts all 219 SQL/client command references under ref/ as <refentry>, which
    is a far more reliable signal than any name prefix — `pgbuffercache.sgml`
    is a contrib extension chapter, not a command, despite the `pg` prefix.
    """
    name = path.stem

    # Structural signals, in order of reliability.
    if root_tag == "refentry":
        return "sql_command"
    if root_tag == "appendix":
        return "appendix"

    if name == "config" or name.startswith("runtime-config"):
        return "config"
    if name.startswith(("functions-", "datatype", "typeconv")):
        return "function_ref"
    if name.startswith(("catalog", "view-", "information_schema")):
        return "catalog"
    if name.startswith(
        (
            "pgbuffercache",
            "pgcrypto",
            "pgstat",
            "pgtrgm",
            "pgrowlocks",
            "pgfreespacemap",
            "pgprewarm",
            "pgsurgery",
            "pgvisibility",
            "pgwalinspect",
            "contrib",
        )
    ):
        return "extension"
    return "guide"


def _substitute_entities(text: str) -> str:
    """Make the text parseable as plain XML."""
    for entity, replacement in _ENTITIES.items():
        text = text.replace(entity, replacement)
    # Whatever is left is unknown to us and unknown to the parser. Keep the
    # word, drop the & and ; so parsing can continue.
    return _UNKNOWN_ENTITY.sub(r"\1", text)


# Marks the boundaries of a verbatim block while the text is being assembled.
# U+0000 cannot occur in valid XML character data, so it can never collide with
# real content — which a whitespace heuristic would. DocBook indents prose by
# two or three spaces for source readability, so "is this line indented?" tells
# you nothing about whether it is code.
_CODE_SENTINEL = "\x00"


def _extract_text(element: etree._Element) -> str:
    """Flatten an element tree to readable text.

    Two rules that a naive ''.join(itertext()) would get wrong:

    1. Code blocks keep their internal whitespace exactly. Indentation is
       meaning in a SQL example, and collapsing it makes the sample useless to
       anyone reading a retrieved chunk.
    2. Prose gets its whitespace collapsed. DocBook wraps lines at ~70 columns
       for source readability; those newlines are an artifact of the file
       format, not of the sentence.
    """
    parts: list[str] = []

    def walk(node: etree._Element) -> None:
        tag = etree.QName(node).localname if isinstance(node.tag, str) else None

        if tag in _DROP_ELEMENTS:
            # Skip the element and its subtree, but keep the text that follows
            # it — an <indexterm> often sits mid-sentence.
            if node.tail:
                parts.append(node.tail)
            return

        if tag in CODE_ELEMENTS:
            code = "".join(node.itertext())
            # Fenced in sentinels so the whitespace pass below can find this
            # block exactly, instead of guessing at it from indentation.
            parts.append(
                "\n\n" + _CODE_SENTINEL + code.strip("\n") + _CODE_SENTINEL + "\n\n"
            )
            if node.tail:
                parts.append(node.tail)
            return

        if tag in _BLOCK_OPEN:
            parts.append("\n")
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
        if tag in _BLOCK_CLOSE:
            parts.append("\n")
        if node.tail:
            parts.append(node.tail)

    walk(element)
    raw = "".join(parts)

    # Split on the sentinels. Because they were emitted in matched pairs, the
    # odd-numbered segments are exactly the verbatim blocks and the
    # even-numbered ones are exactly the prose.
    segments = raw.split(_CODE_SENTINEL)
    rebuilt: list[str] = []
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            rebuilt.append("\n\n" + segment.strip("\n") + "\n\n")  # code: verbatim
        else:
            # Prose: DocBook hard-wraps source lines at ~70 columns, so its
            # newlines are a file-format artifact. Collapse runs of whitespace,
            # but keep paragraph breaks.
            paragraphs = (re.sub(r"\s+", " ", p).strip() for p in segment.split("\n\n"))
            rebuilt.append("\n\n".join(p for p in paragraphs if p))

    text = "".join(rebuilt)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _find_title(root: etree._Element) -> str | None:
    """Pull the human-readable title out of a document root.

    DocBook does not put it in the same place for every root element: chapters
    and sections use <title>, while a <refentry> (all 220 command reference
    pages) uses <refmeta><refentrytitle> and <refnamediv><refname>. Reading only
    <title> would leave every SQL command page titled after its filename —
    "Create_Table" instead of "CREATE TABLE".
    """
    for xpath in ("refmeta/refentrytitle", "refnamediv/refname", "title"):
        el = root.find(xpath)
        if el is not None:
            text = " ".join("".join(el.itertext()).split())
            if text:
                return text
    return None


def load_document(file_path: Path, corpus_root: Path) -> Document | None:
    """Parse one .sgml file. Returns None if it holds no usable content.

    Never raises on bad input. A corpus of 386 files will contain something
    unexpected, and one malformed file must not abort the whole ingest.
    """
    try:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if not raw.strip():
        return None

    cleaned = _substitute_entities(raw)

    # recover=True is the safety net: substitution handles the entities we know
    # about, and the parser salvages what it can from anything we missed.
    parser = etree.XMLParser(recover=True, resolve_entities=False, huge_tree=True)
    try:
        root = etree.fromstring(cleaned.encode("utf-8"), parser)
    except etree.XMLSyntaxError:
        return None

    if root is None:
        return None

    root_tag = etree.QName(root).localname
    if root_tag not in _DOC_ROOTS:
        # Fragments like version.sgml or standalone entity files.
        return None

    title = (
        _find_title(root) or file_path.stem.replace("_", " ").replace("-", " ").upper()
    )

    content = _extract_text(root)
    if len(content) < 100:
        # Stub or placeholder — indexing it only adds noise.
        return None

    return Document(
        path=file_path.relative_to(corpus_root).as_posix(),
        title=title,
        content=content,
        doc_type=classify(file_path, root_tag),
        raw_xml=root,
    )


def load_corpus(corpus_root: str | Path) -> list[Document]:
    """Load every document under `corpus_root`, sorted by path.

    Sorted order matters: it makes chunk IDs stable across runs, which is what
    lets the committed results in results/ be compared run to run.
    """
    root = Path(corpus_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Corpus not found at {root}. Run: bash data/download.sh"
        )

    docs: list[Document] = []
    for file_path in sorted(root.rglob("*.sgml")):
        doc = load_document(file_path, root)
        if doc is not None:
            docs.append(doc)
    return docs
