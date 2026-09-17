"""Retrieval metrics: Hit@k and MRR, scored at two levels.

Every metric here answers one question: did the retriever find the passage that
answers the question, and how high did it rank?

Two levels, because the corpus forces it. All 374 configuration parameters live
in one 54k-token config.sgml, and 23 of the 55 answerable questions point at
that file. File-level Hit@k counts a hit as soon as any chunk from config.sgml
appears — including one about locale settings — which is close to free.
Section-level scoring asks whether the retriever found the right *passage*.

Reporting both is deliberate. File-level is comparable to how most published
RAG numbers are computed, so an outside reader can place ours. Section-level is
the one that reflects whether retrieval actually works. Where the two diverge is
itself a result.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import mean

from src.eval.questions import Question
from src.retrieval.base import ScoredChunk

# The k values the plan asks for. Hit@5 matters because Phase 6 hands roughly
# five chunks to the generator: a gold passage below rank 5 is one the answer
# can never be built from.
DEFAULT_K_VALUES = (1, 3, 5)

# How deep to retrieve. Larger than any k we score, so MRR can see a gold
# passage that lands outside the top 5 rather than scoring it a flat zero.
RETRIEVAL_DEPTH = 20


class MatchLevel:
    """The two granularities a result can be judged at."""

    FILE = "file"
    SECTION = "section"


def matches_file(question: Question, result: ScoredChunk) -> bool:
    return result.doc_path in question.gold_doc_paths


def matches_section(question: Question, result: ScoredChunk) -> bool:
    """Whether the result is the gold passage, not merely the gold file.

    A prefix match counts. The chunker may split one labeled section into
    several chunks, each carrying a heading path that extends the label —
    "Server Configuration > Write Ahead Log > Checkpoints > Notes" is still
    inside the labeled section, and calling it a miss would penalize the
    chunker for being more granular than the label.
    """
    if not question.gold_headings:
        return False
    return any(
        result.heading == heading or result.heading.startswith(heading + " > ")
        for heading in question.gold_headings
    )


def first_relevant_rank(
    question: Question, results: Sequence[ScoredChunk], level: str
) -> int | None:
    """Rank of the first correct result, or None if there is none.

    Ranks are 1-based, matching ScoredChunk.rank, because MRR is defined over
    human ranks — an off-by-one here would shift every number in the report.
    """
    match = matches_file if level == MatchLevel.FILE else matches_section
    for result in results:
        if match(question, result):
            return result.rank
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    """1.0 if a correct result appeared within the top k, else 0.0."""
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    """1/rank of the first correct result, 0.0 if there was none.

    The reciprocal is what makes MRR care about position: rank 1 scores 1.0,
    rank 2 scores 0.5, rank 10 scores 0.1. Hit@k cannot distinguish rank 1 from
    rank 3 — both are hits — and this is what fills that gap.
    """
    return 1.0 / rank if rank is not None else 0.0


@dataclass
class QuestionResult:
    """What one retriever did on one question."""

    question_id: str
    category: str
    answerable: bool
    retriever: str
    file_rank: int | None
    section_rank: int | None
    top_score: float
    retrieved_paths: tuple[str, ...] = ()
    retrieved_headings: tuple[str, ...] = ()

    def hit(self, k: int, level: str = MatchLevel.SECTION) -> float:
        rank = self.file_rank if level == MatchLevel.FILE else self.section_rank
        return hit_at_k(rank, k)

    def rr(self, level: str = MatchLevel.SECTION) -> float:
        rank = self.file_rank if level == MatchLevel.FILE else self.section_rank
        return reciprocal_rank(rank)


def evaluate_question(
    question: Question, results: Sequence[ScoredChunk], retriever_name: str
) -> QuestionResult:
    """Score one retriever's output for one question."""
    return QuestionResult(
        question_id=question.id,
        category=question.category,
        answerable=question.answerable,
        retriever=retriever_name,
        file_rank=first_relevant_rank(question, results, MatchLevel.FILE),
        section_rank=first_relevant_rank(question, results, MatchLevel.SECTION),
        top_score=results[0].score if results else 0.0,
        retrieved_paths=tuple(r.doc_path for r in results),
        retrieved_headings=tuple(r.heading for r in results),
    )


@dataclass
class MetricSummary:
    """Aggregated metrics over a set of questions.

    Only answerable questions are counted. An unanswerable one has no gold
    passage, so every retriever scores zero on it, and including them would
    drag all three down by the same amount — changing the absolute numbers
    while telling you nothing about the comparison. Abstention is measured
    separately in Phase 6, where it belongs.
    """

    retriever: str
    level: str
    n_questions: int
    hits: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    category: str | None = None

    def as_row(self, k_values: Sequence[int] = DEFAULT_K_VALUES) -> dict[str, object]:
        row: dict[str, object] = {"retriever": self.retriever, "n": self.n_questions}
        for k in k_values:
            row[f"hit@{k}"] = self.hits.get(k, 0.0)
        row["mrr"] = self.mrr
        return row


def summarize(
    results: Iterable[QuestionResult],
    *,
    retriever: str,
    level: str = MatchLevel.SECTION,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    category: str | None = None,
) -> MetricSummary:
    """Average Hit@k and MRR over the answerable questions in `results`."""
    scored = [r for r in results if r.answerable]
    if category is not None:
        scored = [r for r in scored if r.category == category]

    if not scored:
        return MetricSummary(
            retriever=retriever,
            level=level,
            n_questions=0,
            hits=dict.fromkeys(k_values, 0.0),
            mrr=0.0,
            category=category,
        )

    return MetricSummary(
        retriever=retriever,
        level=level,
        n_questions=len(scored),
        hits={k: mean(r.hit(k, level) for r in scored) for k in k_values},
        mrr=mean(r.rr(level) for r in scored),
        category=category,
    )


def compare_retrievers(
    summaries: Sequence[MetricSummary], metric: str = "mrr"
) -> tuple[str, float]:
    """Return the best retriever by one metric, and its margin over second place.

    The margin is what decides whether a difference is worth reporting as a
    finding. On 55 questions, one question is worth 1.8 percentage points, so a
    gap of 0.01 is a single question changing rank — noise, not a result.
    """
    if not summaries:
        return ("none", 0.0)

    def value(summary: MetricSummary) -> float:
        if metric == "mrr":
            return summary.mrr
        if metric.startswith("hit@"):
            return summary.hits.get(int(metric[4:]), 0.0)
        raise ValueError(f"unknown metric: {metric!r}")

    ordered = sorted(summaries, key=value, reverse=True)
    best = ordered[0]
    margin = value(best) - value(ordered[1]) if len(ordered) > 1 else value(best)
    return (best.retriever, margin)
