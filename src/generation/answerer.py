"""Build an answer from retrieved chunks.

The generator's job here is narrower than it sounds: it does not answer from
knowledge, it rewrites what the retrieved passages already say. Everything that
makes the answer trustworthy — whether the context could answer at all, whether
each claim is supported — is handled by the two gates around it, not here.

Keeping that boundary sharp is the point. An answerer that also decides when to
refuse is an answerer whose refusals cannot be measured separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.generation.llm import Completion, OllamaClient
from src.retrieval.base import ScoredChunk

# Chunks are numbered [1]..[k] in the prompt rather than cited by chunk_id.
# A real id looks like "config.sgml::structural::142": it costs tokens, means
# nothing to a reader, and a model asked to reproduce it will eventually
# mistype one. The numbers map back to the chunk list positionally.
CITATION = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "You answer questions about PostgreSQL using only the numbered sources "
    "provided. You never use knowledge from outside those sources."
)

# The instructions are explicit about citation and about not inventing, because
# both failures are silent: an uncited answer still reads fine, and a plausible
# invented sentence is exactly what the grounding gate exists to catch.
ANSWER_PROMPT = """\
Answer the question using only the sources below.

Rules:
- Use only information stated in the sources. Do not add anything else.
- Cite the source for each fact with its number in square brackets, like [1].
- If the sources do not contain the answer, reply exactly: INSUFFICIENT
- Answer in two or three sentences. Do not repeat the question.

Sources:
{context}

Question: {question}

Answer:"""

# What the model says when the context cannot support an answer. This is a
# fallback, not the abstention mechanism: the answerability gate decides that
# before generation, and is measured on its own.
INSUFFICIENT_MARKER = "INSUFFICIENT"


@dataclass
class Answer:
    """A generated answer and everything needed to judge it."""

    question: str
    text: str
    chunks: list[ScoredChunk]
    cited_indexes: tuple[int, ...] = ()
    refused: bool = False
    completion: Completion | None = field(default=None, repr=False)

    @property
    def cited_chunks(self) -> list[ScoredChunk]:
        """The chunks the answer actually pointed at.

        Phase 6's grounding gate checks claims against these rather than against
        everything retrieved: an answer citing [1] should be supported by source
        1, and letting it borrow support from an uncited source would make the
        citation meaningless.
        """
        return [
            self.chunks[i - 1] for i in self.cited_indexes if 1 <= i <= len(self.chunks)
        ]

    @property
    def has_citations(self) -> bool:
        return bool(self.cited_indexes)


def format_context(chunks: list[ScoredChunk], max_chars: int = 1800) -> str:
    """Render retrieved chunks as a numbered source list.

    The heading path is included because it is often where the answer's subject
    is named — a chunk body may say "the default is typically 128 megabytes"
    while only the heading says "shared_buffers".

    Oversized chunks are truncated. A single 3,940-token table would otherwise
    consume the whole context window and push the other four sources out.
    """
    parts: list[str] = []
    for index, result in enumerate(chunks, start=1):
        body = result.chunk.body
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + " ...[truncated]"
        heading = result.heading or result.chunk.doc_title
        parts.append(f"[{index}] {heading}\n{body}")
    return "\n\n".join(parts)


def parse_citations(text: str, n_sources: int) -> tuple[int, ...]:
    """Pull the source numbers out of an answer, in order, deduplicated.

    Numbers outside the range are dropped: a model citing [7] when five sources
    were given has hallucinated a citation, and treating it as real would let
    an unsupported claim look attributed.
    """
    seen: list[int] = []
    for match in CITATION.finditer(text):
        index = int(match.group(1))
        if 1 <= index <= n_sources and index not in seen:
            seen.append(index)
    return tuple(seen)


class Answerer:
    """Generates an answer from retrieved chunks."""

    def __init__(
        self, client: OllamaClient | None = None, *, max_tokens: int = 300
    ) -> None:
        self.client = client or OllamaClient()
        self.max_tokens = max_tokens

    def answer(self, question: str, chunks: list[ScoredChunk]) -> Answer:
        if not chunks:
            return Answer(
                question=question,
                text="I don't have enough information to answer this.",
                chunks=[],
                refused=True,
            )

        prompt = ANSWER_PROMPT.format(context=format_context(chunks), question=question)
        completion = self.client.generate(
            prompt, system=SYSTEM_PROMPT, max_tokens=self.max_tokens
        )

        text = completion.text.strip()
        refused = text.upper().startswith(INSUFFICIENT_MARKER)
        if refused:
            text = "I don't have enough information to answer this."

        return Answer(
            question=question,
            text=text,
            chunks=chunks,
            cited_indexes=() if refused else parse_citations(text, len(chunks)),
            refused=refused,
            completion=completion,
        )
