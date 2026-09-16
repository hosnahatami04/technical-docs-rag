"""Tests for the question set and its loader.

Two kinds of test live here. Most check the loader's contract using synthetic
questions, so a failure points at the validation logic. A few at the end check
the committed question set itself — those are the ones that would catch a
labeling mistake introduced while editing questions.json by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.questions import (
    CATEGORIES,
    TOTAL_QUESTIONS,
    QuestionSetError,
    load_questions,
    parse_question,
    summarize,
    validate_against_corpus,
    validate_set,
)


def answerable(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "q-1",
        "question": "What is the default value of max_wal_size?",
        "category": "factual",
        "answerable": True,
        "gold_doc_paths": ["config.sgml"],
        "gold_headings": ["Server Configuration > Write Ahead Log > Checkpoints"],
        "gold_answer": "1 GB.",
        "key_claims": ["max_wal_size defaults to 1 GB"],
    }
    base.update(overrides)
    return base


def unanswerable(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "u-1",
        "question": "What is the default value of io_method?",
        "category": "unanswerable",
        "answerable": False,
        "unanswerable_reason": "io_method is a PostgreSQL 18 parameter.",
        "absent_terms": ["io_method"],
    }
    base.update(overrides)
    return base


# --- Parsing a single question ---------------------------------------------


def test_parses_an_answerable_question() -> None:
    question = parse_question(answerable())
    assert question.id == "q-1"
    assert question.answerable is True
    assert question.gold_doc_paths == ("config.sgml",)
    assert question.key_claims == ("max_wal_size defaults to 1 GB",)


def test_parses_an_unanswerable_question() -> None:
    question = parse_question(unanswerable())
    assert question.answerable is False
    assert question.gold_doc_paths == ()
    assert question.absent_terms == ("io_method",)


@pytest.mark.parametrize("field", ["id", "question", "category", "answerable"])
def test_missing_required_field_is_rejected(field: str) -> None:
    raw = answerable()
    del raw[field]
    with pytest.raises(QuestionSetError, match="missing required field"):
        parse_question(raw)


def test_unknown_category_is_rejected() -> None:
    with pytest.raises(QuestionSetError, match="unknown category"):
        parse_question(answerable(category="trivia"))


def test_empty_question_text_is_rejected() -> None:
    with pytest.raises(QuestionSetError, match="question text is empty"):
        parse_question(answerable(question="   "))


def test_non_boolean_answerable_is_rejected() -> None:
    # JSON "true" as a string would otherwise pass truthiness checks silently.
    with pytest.raises(QuestionSetError, match="must be a boolean"):
        parse_question(answerable(answerable="true"))


def test_gold_paths_must_be_a_list() -> None:
    with pytest.raises(QuestionSetError, match="must be a list"):
        parse_question(answerable(gold_doc_paths="config.sgml"))


def test_empty_string_in_a_list_is_rejected() -> None:
    with pytest.raises(QuestionSetError, match="empty or non-string"):
        parse_question(answerable(gold_doc_paths=["config.sgml", ""]))


# --- The two shapes a question may take ------------------------------------


def test_answerable_question_needs_a_gold_path() -> None:
    with pytest.raises(QuestionSetError, match="no gold path"):
        parse_question(answerable(gold_doc_paths=[]))


def test_answerable_question_needs_key_claims() -> None:
    # Without key claims there is nothing for Phase 6's grounding gate to score
    # against, and they cannot be written later without re-reading the source.
    with pytest.raises(QuestionSetError, match="no key claims"):
        parse_question(answerable(key_claims=[]))


def test_answerable_question_needs_a_gold_answer() -> None:
    with pytest.raises(QuestionSetError, match="no answer"):
        parse_question(answerable(gold_answer=""))


def test_unanswerable_question_must_not_have_gold_paths() -> None:
    # A gold path on an unanswerable question means one of the two labels is
    # wrong, and abstention scoring would silently use the broken one.
    with pytest.raises(QuestionSetError, match="must have no gold paths"):
        parse_question(unanswerable(gold_doc_paths=["config.sgml"]))


def test_unanswerable_question_must_not_have_key_claims() -> None:
    with pytest.raises(QuestionSetError, match="must have no key claims"):
        parse_question(unanswerable(key_claims=["something"]))


def test_unanswerable_question_needs_a_reason() -> None:
    # The reason is what makes the label checkable rather than asserted.
    with pytest.raises(QuestionSetError, match="no reason"):
        parse_question(unanswerable(unanswerable_reason=""))


def test_answerable_question_must_not_carry_an_unanswerable_reason() -> None:
    with pytest.raises(QuestionSetError, match="should not carry"):
        parse_question(answerable(unanswerable_reason="because"))


def test_answerable_question_must_not_carry_absent_terms() -> None:
    with pytest.raises(QuestionSetError, match="should not carry absent_terms"):
        parse_question(answerable(absent_terms=["io_method"]))


def test_category_and_answerable_flag_must_agree() -> None:
    with pytest.raises(QuestionSetError, match="cannot be answerable"):
        parse_question(answerable(category="unanswerable"))
    with pytest.raises(QuestionSetError, match="only category 'unanswerable'"):
        parse_question(unanswerable(category="factual"))


def test_multi_section_must_span_more_than_one_place() -> None:
    raw = answerable(
        category="multi_section",
        gold_doc_paths=["config.sgml"],
        gold_headings=["Server Configuration > Resource Consumption > Memory"],
    )
    with pytest.raises(QuestionSetError, match="more than one"):
        parse_question(raw)


def test_multi_section_may_span_two_headings_in_one_document() -> None:
    # config.sgml is 54k tokens; two of its sections are as far apart as two
    # separate files, which is what the plan's "distant parts" means.
    question = parse_question(
        answerable(
            category="multi_section",
            gold_doc_paths=["config.sgml"],
            gold_headings=[
                "Server Configuration > Resource Consumption > Memory",
                "Server Configuration > Query Planning > Planner Cost Constants",
            ],
        )
    )
    assert question.category == "multi_section"
    assert question.is_multi_document is False


# --- The set as a whole -----------------------------------------------------


def test_duplicate_ids_are_rejected() -> None:
    questions = [parse_question(answerable()), parse_question(answerable())]
    with pytest.raises(QuestionSetError, match="duplicate question ids"):
        validate_set(questions, strict_counts=False)


def test_duplicate_question_text_is_rejected() -> None:
    # Two ids asking the same thing inflate whichever retriever happens to win
    # on it, and waste a slot in a set sized for balance.
    questions = [
        parse_question(answerable(id="a")),
        parse_question(answerable(id="b")),
    ]
    with pytest.raises(QuestionSetError, match="duplicate question text"):
        validate_set(questions, strict_counts=False)


def test_wrong_total_is_rejected() -> None:
    with pytest.raises(QuestionSetError, match=f"expected {TOTAL_QUESTIONS}"):
        validate_set([parse_question(answerable())])


def test_gold_paths_are_checked_against_the_corpus() -> None:
    # A mistyped path becomes a question no retriever can score on, which reads
    # as a retrieval failure rather than a labeling bug.
    questions = [parse_question(answerable(gold_doc_paths=["typo.sgml"]))]
    with pytest.raises(QuestionSetError, match="not found in corpus"):
        validate_against_corpus(questions, {"config.sgml"})


def test_summarize_counts_by_category() -> None:
    questions = [
        parse_question(answerable(id="a")),
        parse_question(answerable(id="b", question="Another factual question?")),
        parse_question(unanswerable(id="c")),
    ]
    summary = summarize(questions)
    assert summary["factual"].total == 2
    assert summary["factual"].answerable == 2
    assert summary["unanswerable"].total == 1
    assert summary["unanswerable"].answerable == 0
    assert summary["factual"].gold_paths == {"config.sgml"}


# --- Loading from disk ------------------------------------------------------


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_questions("nope/questions.json")


def test_top_level_must_be_a_list(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"id": "x"}), encoding="utf-8")
    with pytest.raises(QuestionSetError, match="must be a list"):
        load_questions(path, strict_counts=False)


# --- The committed question set --------------------------------------------


def test_committed_set_loads_and_validates() -> None:
    assert len(load_questions()) == TOTAL_QUESTIONS


def test_committed_set_matches_the_planned_category_counts() -> None:
    questions = load_questions()
    for category, expected in CATEGORIES.items():
        actual = sum(1 for q in questions if q.category == category)
        assert actual == expected, f"{category}: {actual} != {expected}"


def test_committed_set_has_fifteen_unanswerable_questions() -> None:
    # Without these, abstention cannot be measured at all, which is the single
    # thing the plan says separates a demo from a system.
    questions = load_questions()
    assert sum(1 for q in questions if not q.answerable) == 15


def test_every_unanswerable_question_explains_itself() -> None:
    for question in load_questions():
        if not question.answerable:
            assert len(question.unanswerable_reason) > 40, question.id


def test_every_answerable_question_has_at_least_two_key_claims() -> None:
    # One claim makes groundedness a coin flip; two or more make the score mean
    # something.
    for question in load_questions():
        if question.answerable:
            assert len(question.key_claims) >= 2, question.id


def test_gold_headings_are_recorded_for_configuration_questions() -> None:
    # All 374 configuration parameters live in one 54k-token file, so
    # file-level Hit@k is nearly free for them. Section-level scoring is the
    # reason these labels exist.
    questions = load_questions()
    config_questions = [q for q in questions if "config.sgml" in q.gold_doc_paths]
    assert len(config_questions) > 10
    assert all(q.gold_headings for q in config_questions)


def test_question_ids_follow_their_category() -> None:
    prefixes = {
        "factual": "fact-",
        "procedural": "proc-",
        "conceptual": "conc-",
        "multi_section": "multi-",
        "unanswerable": "unans-",
    }
    for question in load_questions():
        assert question.id.startswith(prefixes[question.category]), question.id
