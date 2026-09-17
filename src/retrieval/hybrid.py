"""Hybrid retrieval by Reciprocal Rank Fusion.

BM25 and the dense retriever fail in opposite directions: one cannot match a
paraphrase, the other cannot guarantee an exact identifier. Fusing them is
meant to cover both. Whether it actually does is a question for Phase 5, and if
the answer is no, that gets reported.

The difficulty in combining two retrievers is that their scores are not
comparable. BM25 returns an unbounded sum that depends on corpus statistics;
cosine similarity is bounded in [-1, 1]. Adding them means whichever happens to
produce larger numbers dominates, and rescaling them requires picking a
normalization whose choice changes the answer.

RRF sidesteps this entirely by throwing the scores away and keeping only the
ranks:

    score(d) = sum over retrievers of  1 / (k + rank_of_d_in_that_retriever)

Rank 1 contributes 1/61, rank 2 contributes 1/62, and a document missing from a
retriever's list contributes nothing. No normalization, no weighting, no
parameter that has to be tuned per corpus. That is why the plan picks it over
weighted score blending.
"""

from __future__ import annotations

from typing import Any

from src.ingestion.chunker import Chunk
from src.retrieval.base import Retriever, ScoredChunk, rank_results

# The constant from Cormack, Clarke & Buettcher (2009), which introduced RRF.
# Its job is to flatten the curve: without it, rank 1 would score 1.0 and rank
# 2 only 0.5, so a single retriever's top hit could never be outvoted. At k=60
# the gap between 1/61 and 1/62 is small, which lets agreement between
# retrievers matter more than any one retriever's confidence.
RRF_K = 60

# Retrieve deep from each retriever, fuse, then keep a short list. The depth
# matters: a document ranked 15th by BM25 and 12th by dense is a strong
# consensus candidate that neither would have surfaced in its own top 5.
FUSION_DEPTH = 20


class HybridRetriever(Retriever):
    """Fuses several retrievers' rankings with RRF."""

    name = "hybrid"

    def __init__(
        self,
        retrievers: list[Retriever],
        *,
        rrf_k: int = RRF_K,
        fusion_depth: int = FUSION_DEPTH,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("HybridRetriever needs at least two retrievers to fuse")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        if fusion_depth < 1:
            raise ValueError("fusion_depth must be at least 1")

        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.fusion_depth = fusion_depth

    def __len__(self) -> int:
        return max(len(retriever) for retriever in self.retrievers)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        debug: dict[str, dict[str, Any]] = {}

        for retriever in self.retrievers:
            for result in retriever.search(query, k=self.fusion_depth):
                chunk_id = result.chunk_id
                chunks.setdefault(chunk_id, result.chunk)
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (
                    self.rrf_k + result.rank
                )
                # Recording each retriever's rank is what makes a per-category
                # disagreement in Phase 5 explainable rather than just visible.
                entry = debug.setdefault(chunk_id, {"ranks": {}})
                entry["ranks"][retriever.name] = result.rank

        for chunk_id, entry in debug.items():
            entry["retrievers_agreeing"] = len(entry["ranks"])
            entry["rrf_score"] = fused[chunk_id]

        scored = [(chunks[chunk_id], score) for chunk_id, score in fused.items()]
        return rank_results(scored, k, debug=debug)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], *, rrf_k: int = RRF_K
) -> list[tuple[str, float]]:
    """Fuse ranked id lists directly, best first.

    Extracted from the retriever so the fusion arithmetic can be tested against
    a hand-computed expectation without building an index — which is what the
    plan asks for.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
