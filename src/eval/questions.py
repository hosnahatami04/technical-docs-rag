"""Load and validate the labeled question set.

The question set is the measuring instrument for everything that follows, so it
is loaded through a schema rather than read as raw JSON. A typo in a
`gold_doc_paths` entry would otherwise show up much later as a retriever that
mysteriously scores zero on one question.

Validation is strict on purpose: every gold path must exist in the corpus, every
answerable question must have at least one gold path and at least one key claim,
and every unanswerable one must have neither.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

QUESTIONS_PATH = Path("questions/questions.json")

# The five categories from the plan, with the counts it specifies.
CATEGORIES: dict[str, int] = {
    "factual": 20,
    "procedural": 15,
    "conceptual": 10,
    "multi_section": 10,
    "unanswerable": 15,
}

TOTAL_QUESTIONS = sum(CATEGORIES.values())


class QuestionSetError(ValueError):
    """Raised when the question set does not satisfy its own contract."""


@dataclass(frozen=True)
class Question:
    """One labeled evaluation question.

    `gold_doc_paths` is what Hit@k is scored against. `gold_headings` is the
    stricter variant: all 374 configuration parameters live in a single
    54k-token config.sgml, so every configuration question would score a
    file-level hit almost by default. Scoring the heading path as well
    distinguishes "found the right file" from "found the right passage".

    `key_claims` are the facts a correct answer must contain. Phase 6 scores
    groundedness against them, and they are useless unless written now, while
    the source passage is in front of you.
    """

    id: str
    question: str
    category: str
    answerable: bool
    gold_doc_paths: tuple[str, ...] = ()
    gold_headings: tuple[str, ...] = ()
    gold_answer: str = ""
    key_claims: tuple[str, ...] = ()
    # Why an unanswerable question is unanswerable. Recorded so a reader can
    # check the claim rather than trust it — and so a future PostgreSQL release
    # that adds the feature can be spotted.
    unanswerable_reason: str = ""
    # Terms that must not appear anywhere in the corpus for this question to be
    # genuinely unanswerable, checked by verify_questions. Empty when the
    # question rests on a fact never being stated rather than a term never
    # appearing — that kind cannot be checked by string search.
    absent_terms: tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_multi_document(self) -> bool:
        return len(self.gold_doc_paths) > 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QuestionSetError(message)


def _as_tuple(value: object, field_name: str, question_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    _require(
        isinstance(value, list),
        f"{question_id}: {field_name} must be a list, got {type(value).__name__}",
    )
    assert isinstance(value, list)
    for item in value:
        _require(
            isinstance(item, str) and item.strip() != "",
            f"{question_id}: {field_name} contains an empty or non-string entry",
        )
    return tuple(value)


def parse_question(raw: dict[str, object]) -> Question:
    """Build one Question, failing loudly on anything malformed."""
    question_id = str(raw.get("id", "<missing id>"))

    for required in ("id", "question", "category", "answerable"):
        _require(required in raw, f"{question_id}: missing required field {required!r}")

    category = str(raw["category"])
    _require(
        category in CATEGORIES,
        f"{question_id}: unknown category {category!r}, expected one of "
        f"{sorted(CATEGORIES)}",
    )

    answerable = raw["answerable"]
    _require(
        isinstance(answerable, bool),
        f"{question_id}: answerable must be a boolean",
    )
    assert isinstance(answerable, bool)

    question_text = str(raw["question"]).strip()
    _require(question_text != "", f"{question_id}: question text is empty")

    gold_paths = _as_tuple(raw.get("gold_doc_paths"), "gold_doc_paths", question_id)
    gold_headings = _as_tuple(raw.get("gold_headings"), "gold_headings", question_id)
    key_claims = _as_tuple(raw.get("key_claims"), "key_claims", question_id)
    absent_terms = _as_tuple(raw.get("absent_terms"), "absent_terms", question_id)
    gold_answer = str(raw.get("gold_answer", "")).strip()
    reason = str(raw.get("unanswerable_reason", "")).strip()

    # The two shapes a question can take, enforced so neither can drift.
    if answerable:
        _require(
            category != "unanswerable",
            f"{question_id}: category 'unanswerable' cannot be answerable",
        )
        _require(
            gold_paths != (), f"{question_id}: answerable question has no gold path"
        )
        _require(gold_answer != "", f"{question_id}: answerable question has no answer")
        _require(
            key_claims != (), f"{question_id}: answerable question has no key claims"
        )
        _require(
            reason == "",
            f"{question_id}: answerable question should not carry an unanswerable_reason",
        )
        _require(
            absent_terms == (),
            f"{question_id}: answerable question should not carry absent_terms",
        )
    else:
        _require(
            category == "unanswerable",
            f"{question_id}: only category 'unanswerable' may be unanswerable",
        )
        _require(
            gold_paths == (),
            f"{question_id}: unanswerable question must have no gold paths",
        )
        _require(
            key_claims == (),
            f"{question_id}: unanswerable question must have no key claims",
        )
        # The reason is what makes an unanswerable question checkable rather
        # than an assertion.
        _require(reason != "", f"{question_id}: unanswerable question has no reason")

    if category == "multi_section":
        # The plan defines this category as "requires combining two distant
        # parts of the docs" — distant sections, not necessarily distinct
        # files. config.sgml alone is 54k tokens, so its Memory section and its
        # Query Planning section are as far apart as two separate documents.
        _require(
            len(gold_paths) > 1 or len(gold_headings) > 1,
            f"{question_id}: multi_section question must span more than one "
            f"document or more than one heading path",
        )

    return Question(
        id=question_id,
        question=question_text,
        category=category,
        answerable=answerable,
        gold_doc_paths=gold_paths,
        gold_headings=gold_headings,
        gold_answer=gold_answer,
        key_claims=key_claims,
        unanswerable_reason=reason,
        absent_terms=absent_terms,
        notes=str(raw.get("notes", "")).strip(),
    )


def validate_set(questions: list[Question], *, strict_counts: bool = True) -> None:
    """Check properties of the set as a whole, not of single questions."""
    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    _require(not duplicates, f"duplicate question ids: {sorted(duplicates)}")

    texts = [q.question.lower().strip() for q in questions]
    dup_text = {t for t in texts if texts.count(t) > 1}
    _require(not dup_text, f"duplicate question text: {sorted(dup_text)[:3]}")

    if not strict_counts:
        return

    _require(
        len(questions) == TOTAL_QUESTIONS,
        f"expected {TOTAL_QUESTIONS} questions, found {len(questions)}",
    )

    for category, expected in CATEGORIES.items():
        actual = sum(1 for q in questions if q.category == category)
        _require(
            actual == expected,
            f"category {category!r}: expected {expected} questions, found {actual}",
        )

    # Abstention cannot be measured without unanswerable questions, which is the
    # whole reason 15 of them exist.
    unanswerable = sum(1 for q in questions if not q.answerable)
    _require(
        unanswerable == CATEGORIES["unanswerable"],
        f"expected {CATEGORIES['unanswerable']} unanswerable questions, "
        f"found {unanswerable}",
    )


def validate_against_corpus(
    questions: Iterable[Question], corpus_paths: set[str]
) -> None:
    """Check every gold path names a document that actually exists.

    A mistyped path silently becomes a question no retriever can ever score on,
    which reads as a retrieval failure rather than a labeling bug.
    """
    missing: list[str] = []
    for question in questions:
        for path in question.gold_doc_paths:
            if path not in corpus_paths:
                missing.append(f"{question.id} -> {path}")
    _require(not missing, "gold paths not found in corpus: " + ", ".join(missing))


def load_questions(
    path: str | Path = QUESTIONS_PATH, *, strict_counts: bool = True
) -> list[Question]:
    """Load, parse and validate the question set."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Question set not found at {file_path}")

    raw = json.loads(file_path.read_text(encoding="utf-8"))
    _require(isinstance(raw, list), f"{file_path}: top level must be a list")

    questions = [parse_question(item) for item in raw]
    validate_set(questions, strict_counts=strict_counts)
    return questions


@dataclass
class CategoryBreakdown:
    """Per-category counts, for the eval report in Phase 5."""

    category: str
    total: int = 0
    answerable: int = 0
    multi_document: int = 0
    gold_paths: set[str] = field(default_factory=set)


def summarize(questions: Iterable[Question]) -> dict[str, CategoryBreakdown]:
    summary: dict[str, CategoryBreakdown] = {}
    for question in questions:
        entry = summary.setdefault(
            question.category, CategoryBreakdown(category=question.category)
        )
        entry.total += 1
        entry.answerable += int(question.answerable)
        entry.multi_document += int(question.is_multi_document)
        entry.gold_paths.update(question.gold_doc_paths)
    return summary
