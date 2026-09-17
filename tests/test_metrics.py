"""Tests for the retrieval metrics.

These matter more than most tests in this repo. Every number in the evaluation
report comes through this module, and a metric that is subtly wrong produces a
report that looks entirely reasonable and is entirely false — there is no
failing test, no exception, just numbers nobody can check.

So the tests here are arithmetic: given a known rank, assert the exact value.
"""

from __future__ import annotations

import pytest

from src.eval.metrics import (
    MatchLevel,
    QuestionResult,
    compare_retrievers,
    evaluate_question,
    first_relevant_rank,
    hit_at_k,
    matches_file,
    matches_section,
    reciprocal_rank,
    summarize,
)
from src.eval.questions import Question
from src.ingestion.chunker import Chunk
from src.retrieval.base import ScoredChunk


def make_question(
    *,
    qid: str = "q1",
    category: str = "factual",
    answerable: bool = True,
    paths: tuple[str, ...] = ("config.sgml",),
    headings: tuple[str, ...] = (
        "Server Configuration > Write Ahead Log > Checkpoints",
    ),
) -> Question:
    return Question(
        id=qid,
        question="What is the default value of max_wal_size?",
        category=category,
        answerable=answerable,
        gold_doc_paths=paths,
        gold_headings=headings,
        gold_answer="1 GB." if answerable else "",
        key_claims=("max_wal_size defaults to 1 GB",) if answerable else (),
        unanswerable_reason="" if answerable else "not in this version",
    )


def make_result(rank: int, path: str, heading: str) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"{path}::{rank}",
        doc_path=path,
        doc_title="Doc",
        doc_type="config",
        heading_path=tuple(heading.split(" > ")) if heading else (),
        body="body",
        chunk_index=rank,
        token_count=10,
        has_code=False,
        has_table=False,
    )
    return ScoredChunk(chunk=chunk, score=1.0 / rank, rank=rank)


GOLD_HEADING = "Server Configuration > Write Ahead Log > Checkpoints"
OTHER_HEADING = "Server Configuration > Locale and Formatting"


# --- Matching ---------------------------------------------------------------


def test_file_match_is_by_document_path() -> None:
    question = make_question()
    assert matches_file(question, make_result(1, "config.sgml", OTHER_HEADING))
    assert not matches_file(question, make_result(1, "wal.sgml", GOLD_HEADING))


def test_section_match_requires_the_right_heading() -> None:
    # The whole reason section-level scoring exists: a chunk from the right
    # file but the wrong section is not an answer.
    question = make_question()
    assert matches_section(question, make_result(1, "config.sgml", GOLD_HEADING))
    assert not matches_section(question, make_result(1, "config.sgml", OTHER_HEADING))


def test_section_match_accepts_a_deeper_heading() -> None:
    # The chunker may split one labeled section into several chunks whose
    # heading paths extend the label. Those are still inside the section, and
    # calling them misses would penalize the chunker for being more granular
    # than the label.
    question = make_question()
    deeper = make_result(1, "config.sgml", GOLD_HEADING + " > Notes")
    assert matches_section(question, deeper)


def test_section_match_rejects_a_sibling_with_a_shared_prefix() -> None:
    # "Checkpoints" must not match "CheckpointsAndRecovery". Requiring the
    # separator is what stops a prefix check from over-matching.
    question = make_question()
    sibling = make_result(1, "config.sgml", GOLD_HEADING + "AndRecovery")
    assert not matches_section(question, sibling)


def test_section_match_is_false_when_no_gold_heading_is_labeled() -> None:
    question = make_question(headings=())
    assert not matches_section(question, make_result(1, "config.sgml", GOLD_HEADING))


# --- Ranks ------------------------------------------------------------------


def test_first_relevant_rank_finds_the_earliest_match() -> None:
    question = make_question()
    results = [
        make_result(1, "wal.sgml", OTHER_HEADING),
        make_result(2, "config.sgml", GOLD_HEADING),
        make_result(3, "config.sgml", GOLD_HEADING),
    ]
    assert first_relevant_rank(question, results, MatchLevel.SECTION) == 2


def test_first_relevant_rank_is_none_when_nothing_matches() -> None:
    question = make_question()
    results = [make_result(1, "wal.sgml", OTHER_HEADING)]
    assert first_relevant_rank(question, results, MatchLevel.SECTION) is None


def test_file_and_section_ranks_can_differ() -> None:
    # The common case on this corpus: the right file at rank 1, the right
    # passage at rank 3.
    question = make_question()
    results = [
        make_result(1, "config.sgml", OTHER_HEADING),
        make_result(2, "wal.sgml", OTHER_HEADING),
        make_result(3, "config.sgml", GOLD_HEADING),
    ]
    assert first_relevant_rank(question, results, MatchLevel.FILE) == 1
    assert first_relevant_rank(question, results, MatchLevel.SECTION) == 3


# --- Hit@k ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rank", "k", "expected"),
    [
        (1, 1, 1.0),
        (1, 3, 1.0),
        (3, 3, 1.0),  # inclusive: rank 3 is within top 3
        (4, 3, 0.0),
        (2, 1, 0.0),
        (None, 5, 0.0),
    ],
)
def test_hit_at_k(rank: int | None, k: int, expected: float) -> None:
    assert hit_at_k(rank, k) == expected


# --- MRR --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(1, 1.0), (2, 0.5), (3, 1 / 3), (10, 0.1), (None, 0.0)],
)
def test_reciprocal_rank(rank: int | None, expected: float) -> None:
    assert reciprocal_rank(rank) == pytest.approx(expected)


def test_mrr_distinguishes_ranks_that_hit_at_k_cannot() -> None:
    # Both are Hit@3 = 1.0, but one is clearly better. This is the gap MRR
    # exists to fill.
    assert hit_at_k(1, 3) == hit_at_k(3, 3)
    assert reciprocal_rank(1) > reciprocal_rank(3)


# --- Aggregation ------------------------------------------------------------


def make_qr(
    qid: str,
    *,
    section_rank: int | None,
    file_rank: int | None = None,
    category: str = "factual",
    answerable: bool = True,
) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        category=category,
        answerable=answerable,
        retriever="test",
        file_rank=file_rank if file_rank is not None else section_rank,
        section_rank=section_rank,
        top_score=1.0,
    )


def test_summarize_averages_over_questions() -> None:
    results = [
        make_qr("a", section_rank=1),
        make_qr("b", section_rank=3),
        make_qr("c", section_rank=None),
    ]
    summary = summarize(results, retriever="test")
    assert summary.n_questions == 3
    assert summary.hits[1] == pytest.approx(1 / 3)
    assert summary.hits[3] == pytest.approx(2 / 3)
    assert summary.mrr == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)


def test_summarize_excludes_unanswerable_questions() -> None:
    # Every retriever scores zero on an unanswerable question, so including
    # them would move all three numbers by the same amount while telling you
    # nothing about the comparison.
    results = [
        make_qr("a", section_rank=1),
        make_qr("u", section_rank=None, answerable=False, category="unanswerable"),
    ]
    summary = summarize(results, retriever="test")
    assert summary.n_questions == 1
    assert summary.mrr == pytest.approx(1.0)


def test_summarize_filters_by_category() -> None:
    results = [
        make_qr("a", section_rank=1, category="factual"),
        make_qr("b", section_rank=None, category="conceptual"),
    ]
    factual = summarize(results, retriever="test", category="factual")
    assert factual.n_questions == 1
    assert factual.mrr == pytest.approx(1.0)


def test_summarize_handles_an_empty_set() -> None:
    summary = summarize([], retriever="test")
    assert summary.n_questions == 0
    assert summary.mrr == 0.0
    assert summary.hits[3] == 0.0


def test_summarize_can_score_at_file_level() -> None:
    results = [make_qr("a", section_rank=None, file_rank=1)]
    section = summarize(results, retriever="test", level=MatchLevel.SECTION)
    file_level = summarize(results, retriever="test", level=MatchLevel.FILE)
    assert section.mrr == 0.0
    assert file_level.mrr == pytest.approx(1.0)


# --- Comparison -------------------------------------------------------------


def test_compare_retrievers_returns_winner_and_margin() -> None:
    a = summarize([make_qr("x", section_rank=1)], retriever="a")
    b = summarize([make_qr("x", section_rank=2)], retriever="b")
    winner, margin = compare_retrievers([a, b])
    assert winner == "a"
    assert margin == pytest.approx(0.5)


def test_compare_retrievers_can_use_hit_at_k() -> None:
    a = summarize([make_qr("x", section_rank=5)], retriever="a")
    b = summarize([make_qr("x", section_rank=1)], retriever="b")
    winner, _ = compare_retrievers([a, b], metric="hit@1")
    assert winner == "b"


def test_compare_retrievers_rejects_an_unknown_metric() -> None:
    a = summarize([make_qr("x", section_rank=1)], retriever="a")
    with pytest.raises(ValueError, match="unknown metric"):
        compare_retrievers([a], metric="ndcg")


def test_compare_retrievers_handles_no_summaries() -> None:
    assert compare_retrievers([]) == ("none", 0.0)


# --- End to end -------------------------------------------------------------


def test_evaluate_question_records_both_levels() -> None:
    question = make_question()
    results = [
        make_result(1, "config.sgml", OTHER_HEADING),
        make_result(2, "config.sgml", GOLD_HEADING),
    ]
    scored = evaluate_question(question, results, "bm25")
    assert scored.retriever == "bm25"
    assert scored.file_rank == 1
    assert scored.section_rank == 2
    assert scored.hit(1, MatchLevel.FILE) == 1.0
    assert scored.hit(1, MatchLevel.SECTION) == 0.0


def test_evaluate_question_handles_empty_results() -> None:
    scored = evaluate_question(make_question(), [], "bm25")
    assert scored.file_rank is None
    assert scored.section_rank is None
    assert scored.top_score == 0.0
    assert scored.rr() == 0.0


def test_evaluate_question_keeps_what_was_retrieved() -> None:
    # The report quotes averages; the raw record is what lets a reader check a
    # single number rather than trust the summary.
    results = [make_result(1, "config.sgml", GOLD_HEADING)]
    scored = evaluate_question(make_question(), results, "bm25")
    assert scored.retrieved_paths == ("config.sgml",)
    assert scored.retrieved_headings == (GOLD_HEADING,)
