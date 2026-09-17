"""Show the three retrievers side by side on a handful of questions.

Run:  python -m src.retrieval.compare

This is a qualitative tool, not an evaluation. It prints what each retriever
returns for a few questions chosen to expose their different failure modes, so
the behaviour is visible before Phase 5 reduces it all to four numbers.

The real measurement — Hit@k and MRR over all 70 labeled questions — is Phase
5's job. Reading this output is how you form the hypothesis that those numbers
then confirm or refute.
"""

from __future__ import annotations

import sys

from src.eval.questions import Question, load_questions
from src.ingestion.indexer import build_retrievers, load_chunks
from src.retrieval.base import Retriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Chosen to contrast, not to sample. Each is a case where the retrievers are
# expected to disagree, and the disagreement is the interesting part.
PROBE_IDS = [
    "fact-001",  # names the identifier outright — BM25's best case
    "fact-006",  # asks about SCRAM iterations without naming scram_iterations
    "fact-009",  # "rotate log files" must reach log_rotation_age
    "conc-003",  # pure concept, no identifier anywhere — dense's best case
    "unans-006",  # a parameter one character from a real one
]


def _hit(question: Question, result_path: str, result_heading: str) -> str:
    """Mark a result against the gold labels, at both levels."""
    if not question.answerable:
        return "  "
    file_hit = result_path in question.gold_doc_paths
    section_hit = any(
        result_heading == heading or result_heading.startswith(heading)
        for heading in question.gold_headings
    )
    if section_hit:
        return "★★"  # right passage
    if file_hit:
        return "★ "  # right file, wrong passage
    return "  "


def show(question: Question, retrievers: dict[str, Retriever], k: int = 3) -> None:
    print("\n" + "=" * 74)
    print(f"{question.id}  [{question.category}]")
    print(f"  {question.question}")
    if question.answerable:
        print(f"  gold: {', '.join(question.gold_doc_paths)}")
        for heading in question.gold_headings:
            print(f"        {heading}")
    else:
        print(f"  gold: (unanswerable) {question.unanswerable_reason[:60]}")
    print("=" * 74)

    for name, retriever in retrievers.items():
        print(f"\n  {name}")
        results = retriever.search(question.question, k=k)
        if not results:
            print("    (no results)")
            continue
        for result in results:
            mark = _hit(question, result.doc_path, result.heading)
            agreement = ""
            if "retrievers_agreeing" in result.debug:
                ranks = result.debug["ranks"]
                agreement = "  " + " ".join(
                    f"{n}={r}" for n, r in sorted(ranks.items())
                )
            print(
                f"    {mark} {result.rank}. [{result.score:7.4f}] "
                f"{result.doc_path:<22} {result.heading[:40]}{agreement}"
            )


def main() -> int:
    print("Loading corpus and indexes ...")
    chunks = load_chunks()
    retrievers = build_retrievers(chunks)
    questions = {q.id: q for q in load_questions()}

    print(f"  {len(chunks):,} chunks indexed")
    print("\n  ★★ = retrieved the gold passage")
    print("  ★  = retrieved the gold file, but the wrong passage")

    for probe_id in PROBE_IDS:
        question = questions.get(probe_id)
        if question is not None:
            show(question, retrievers)

    print("\n" + "=" * 74)
    print("""
  What to look for:

  - fact-001 names max_wal_size outright. BM25 should win it outright; if the
    dense retriever also finds it, the embedding is doing more than expected.
  - fact-006 and fact-009 ask for something without naming it. BM25 has no
    literal term to match, so this is where dense retrieval should pull ahead.
  - conc-003 has no identifier at all.
  - unans-006 names a parameter that does not exist. Every retriever will
    return something confident, which is the whole reason Phase 6 needs an
    answerability gate — retrieval alone cannot abstain.

  Numbers, not impressions, come from Phase 5.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
