"""The interface all three retrievers implement.

The point of this file is that the evaluation harness in Phase 5 must not know
which retriever it is running. If BM25 and the dense retriever return different
shapes, every comparison becomes a special case and the ablation table stops
being a fair test.

So: one `search(query, k) -> list[ScoredChunk]`, one score field, one ordering
guarantee. Anything a specific retriever wants to expose beyond that goes in
`debug`, where the harness can ignore it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.ingestion.chunker import Chunk


@dataclass
class ScoredChunk:
    """A chunk with the score and rank one retriever assigned it.

    `score` means something different in each retriever — BM25 returns an
    unbounded relevance sum, the dense retriever returns cosine similarity in
    [-1, 1], RRF returns a small reciprocal-rank total. They are deliberately
    not normalized against each other: comparing raw scores across retrievers
    is meaningless, and pretending otherwise is exactly the mistake RRF exists
    to avoid.

    `rank` is what stays comparable. It starts at 1, because Hit@1 and MRR are
    defined over human ranks, and an off-by-one here would quietly shift every
    metric in the report.
    """

    chunk: Chunk
    score: float
    rank: int
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def doc_path(self) -> str:
        """The field file-level Hit@k is scored against."""
        return self.chunk.doc_path

    @property
    def heading(self) -> str:
        """The field section-level Hit@k is scored against."""
        return self.chunk.heading


class Retriever(ABC):
    """Base class for BM25, dense and hybrid retrieval."""

    name: str = "retriever"

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """Return the k best chunks for `query`, best first.

        Contract every implementation must hold, because Phase 5 relies on all
        of it:

        - at most k results, fewer only when the index holds fewer chunks
        - sorted by descending score
        - ranks 1..n with no gaps
        - no duplicate chunk_id
        - deterministic: the same query returns the same order every time,
          otherwise a committed results file cannot be reproduced
        """

    @abstractmethod
    def __len__(self) -> int:
        """Number of chunks indexed."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, chunks={len(self)})"


def rank_results(
    scored: list[tuple[Chunk, float]],
    k: int,
    *,
    debug: dict[str, dict[str, Any]] | None = None,
) -> list[ScoredChunk]:
    """Turn (chunk, score) pairs into a ranked, truncated result list.

    Shared by all three retrievers so the ordering rules live in one place.

    The sort key breaks ties on chunk_id rather than leaving them to Python's
    stable sort over whatever order the index happened to produce. Ties are
    common in BM25 — any two chunks matching the same single rare term score
    identically — and without a deterministic tiebreak the committed results
    would drift between runs on the same data.
    """
    ordered = sorted(scored, key=lambda pair: (-pair[1], pair[0].chunk_id))
    results: list[ScoredChunk] = []
    for index, (chunk, score) in enumerate(ordered[:k], start=1):
        results.append(
            ScoredChunk(
                chunk=chunk,
                score=float(score),
                rank=index,
                debug=(debug or {}).get(chunk.chunk_id, {}),
            )
        )
    return results
