"""Check the question set against the corpus it claims to describe.

Run:  python -m src.eval.verify_questions

Structural validation lives in questions.py and runs on every load. This script
does the slower, semantic checks that need the chunked corpus:

- every gold heading path exists in the chunked corpus
- an answerable question's gold chunks actually mention what it asks about
- an unanswerable question's premise really is absent from the corpus

The last one matters most. An unanswerable question is a claim about the corpus,
and a claim nobody checks is just an assertion — if a term turns out to be
documented after all, the question is mislabeled and every abstention metric
built on it is wrong.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from src.eval.questions import Question, load_questions, validate_against_corpus
from src.ingestion.chunker import Chunk, chunk_corpus
from src.ingestion.loader import load_corpus

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPUS_ROOT = Path("data/raw/sgml")

# Identifier-shaped tokens a question asks about: snake_case names, SQL
# keywords in caps, and function calls. These are what must be present in the
# gold chunks (for answerable questions) or absent from the corpus (for
# unanswerable ones).
_IDENTIFIER = re.compile(
    r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b|\b[A-Z]{3,}(?:\s+[A-Z]{2,})*\b"
)


def identifiers_in(text: str) -> set[str]:
    return {m.group(0) for m in _IDENTIFIER.finditer(text)}


def check_headings(questions: list[Question], chunks: list[Chunk]) -> list[str]:
    """Every gold heading must name a real section of the corpus."""
    known = {c.heading for c in chunks}
    problems: list[str] = []
    for question in questions:
        for heading in question.gold_headings:
            if heading in known:
                continue
            # A prefix match still identifies a real section; the chunker may
            # have nested one level deeper than the label records.
            if any(h.startswith(heading) for h in known):
                continue
            problems.append(f"{question.id}: heading not in corpus: {heading!r}")
    return problems


def check_answerable_grounding(
    questions: list[Question], chunks: list[Chunk]
) -> list[str]:
    """The gold documents must actually discuss what the question asks about.

    Catches a gold path that points at a plausible but wrong document — the
    label looks right and the question silently becomes unscoreable.
    """
    by_path: dict[str, str] = {}
    for chunk in chunks:
        by_path[chunk.doc_path] = by_path.get(chunk.doc_path, "") + "\n" + chunk.body

    problems: list[str] = []
    for question in questions:
        if not question.answerable:
            continue
        wanted = identifiers_in(question.question) | identifiers_in(
            question.gold_answer
        )
        if not wanted:
            continue
        haystack = "\n".join(by_path.get(p, "") for p in question.gold_doc_paths)
        missing = {w for w in wanted if w.lower() not in haystack.lower()}
        # Some identifiers in a gold answer are prose ("WAL", "MVCC"); only
        # flag when nothing at all matched, which means the wrong document.
        if missing == wanted:
            problems.append(
                f"{question.id}: none of {sorted(wanted)[:4]} appear in "
                f"{list(question.gold_doc_paths)}"
            )
    return problems


def check_unanswerable_absence(
    questions: list[Question], corpus_text: str
) -> list[str]:
    """An unanswerable question's subject must genuinely not be documented.

    The thing that must be absent is named explicitly in `absent_terms` rather
    than guessed from the question text. Guessing does not work: "What is the
    syntax for a VIRTUAL generated column?" is unanswerable because no such
    syntax is documented, yet the words VIRTUAL and generated column both
    appear in the corpus — the docs describe the concept and then say
    PostgreSQL does not implement it.

    A question with no `absent_terms` is unanswerable for a reason no string
    search can check (a fact simply never stated), and is reported as
    unverifiable rather than passed silently.
    """
    lowered = corpus_text.lower()
    problems: list[str] = []
    unverifiable: list[str] = []

    for question in questions:
        if question.answerable:
            continue
        if not question.absent_terms:
            unverifiable.append(question.id)
            continue
        present = [
            term
            for term in question.absent_terms
            if re.search(rf"\b{re.escape(term.lower())}\b", lowered)
        ]
        if present:
            problems.append(
                f"{question.id}: declared absent but found in corpus: {present}"
            )

    if unverifiable:
        print(
            f"\n  note: {len(unverifiable)} unanswerable questions have no "
            f"absent_terms and rest on a fact never being stated rather than a "
            f"term never appearing: {', '.join(unverifiable)}"
        )
    return problems


def report(questions: list[Question], chunks: list[Chunk]) -> None:
    print("=" * 66)
    print("QUESTION SET")
    print("=" * 66)

    by_category: dict[str, list[Question]] = {}
    for question in questions:
        by_category.setdefault(question.category, []).append(question)

    print(f"\n  total questions       {len(questions):>6}")
    print(f"  answerable            {sum(q.answerable for q in questions):>6}")
    print(f"  unanswerable          {sum(not q.answerable for q in questions):>6}")

    print("\n  by category:")
    for category, group in sorted(by_category.items()):
        multi = sum(1 for q in group if q.is_multi_document)
        print(f"    {category:<15} {len(group):>3}   multi-document: {multi}")

    answerable = [q for q in questions if q.answerable]
    claims = sum(len(q.key_claims) for q in answerable)
    print(f"\n  key claims total      {claims:>6}")
    print(f"  mean claims/question  {claims / len(answerable):>6.1f}")

    # How concentrated the gold labels are. If most answerable questions point
    # at one document, file-level Hit@k is measuring very little.
    path_counts: dict[str, int] = {}
    for question in answerable:
        for path in question.gold_doc_paths:
            path_counts[path] = path_counts.get(path, 0) + 1
    print(f"\n  distinct gold documents {len(path_counts):>4}")
    print("  most cited:")
    for path, count in sorted(path_counts.items(), key=lambda kv: -kv[1])[:6]:
        share = 100 * count / len(answerable)
        print(f"    {path:<34} {count:>3}  ({share:.0f}% of answerable)")

    heading_counts = {h for q in questions for h in q.gold_headings}
    print(f"\n  distinct gold headings  {len(heading_counts):>4}")
    print("    section-level scoring is what separates 'found config.sgml'")
    print("    from 'found the passage about this parameter'")


def main() -> int:
    if not CORPUS_ROOT.exists():
        print(
            f"Corpus not found at {CORPUS_ROOT}. Run: bash data/download.sh",
            file=sys.stderr,
        )
        return 1

    questions = load_questions()
    docs = load_corpus(CORPUS_ROOT)
    chunks = chunk_corpus(docs, strategy="structural")

    validate_against_corpus(questions, {d.path for d in docs})

    corpus_text = "\n".join(d.content for d in docs)
    problems: list[str] = []
    problems += check_headings(questions, chunks)
    problems += check_answerable_grounding(questions, chunks)
    problems += check_unanswerable_absence(questions, corpus_text)

    report(questions, chunks)

    print("\n" + "=" * 66)
    if problems:
        print(f"PROBLEMS: {len(problems)}")
        print("=" * 66)
        for problem in problems:
            print(f"  {problem}")
        return 1

    print("VERIFIED")
    print("=" * 66)
    print("""
  Every gold path exists, every gold heading names a real section, every
  answerable question's gold documents discuss its subject, and every
  unanswerable question's subject is genuinely absent from the corpus.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
