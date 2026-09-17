"""Investigate the disagreements behind the headline numbers.

Run:  python -m src.eval.analyze

The report says which retriever won. This says why — which questions each one
got that the others missed, and what those questions have in common.

It exists because of what the first run showed: the hybrid does not beat dense
retrieval on this corpus. The plan is explicit that a result like that gets
reported plainly and investigated rather than quietly buried, and an
investigation needs the per-question record, not the averages.
"""

from __future__ import annotations

import sys
from collections import Counter

from src.eval.metrics import MatchLevel, QuestionResult
from src.eval.questions import load_questions
from src.eval.runner import RESULTS_DIR, load_raw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _rank(result: QuestionResult, level: str) -> int | None:
    return result.file_rank if level == MatchLevel.FILE else result.section_rank


def by_id(results: list[QuestionResult]) -> dict[str, QuestionResult]:
    return {r.question_id: r for r in results}


def exclusive_wins(
    a: list[QuestionResult], b: list[QuestionResult], level: str, k: int = 3
) -> list[str]:
    """Questions `a` gets within top k that `b` does not."""
    b_by_id = by_id(b)
    wins = []
    for result in a:
        if not result.answerable:
            continue
        other = b_by_id.get(result.question_id)
        if other is None:
            continue
        a_rank, b_rank = _rank(result, level), _rank(other, level)
        if a_rank is not None and a_rank <= k and (b_rank is None or b_rank > k):
            wins.append(result.question_id)
    return wins


def rank_changes(
    source: list[QuestionResult], fused: list[QuestionResult], level: str
) -> tuple[list[str], list[str]]:
    """Questions the fusion moved up, and questions it moved down.

    This is the heart of the hybrid investigation. RRF cannot invent a good
    result: it can only reorder what its inputs gave it. When it loses, it is
    because averaging two rankings pulled a correct top hit down more often than
    it pushed one up.
    """
    fused_by_id = by_id(fused)
    improved, degraded = [], []
    for result in source:
        if not result.answerable:
            continue
        after = fused_by_id.get(result.question_id)
        if after is None:
            continue
        before_rank, after_rank = _rank(result, level), _rank(after, level)
        if before_rank is None and after_rank is not None:
            improved.append(result.question_id)
        elif before_rank is not None and after_rank is None:
            degraded.append(result.question_id)
        elif before_rank is not None and after_rank is not None:
            if after_rank < before_rank:
                improved.append(result.question_id)
            elif after_rank > before_rank:
                degraded.append(result.question_id)
    return improved, degraded


def main() -> int:
    raw_path = RESULTS_DIR / "eval_raw.json"
    if not raw_path.exists():
        print(
            f"No results at {raw_path}. Run: python -m src.eval.runner", file=sys.stderr
        )
        return 1

    raw = load_raw(raw_path)
    structural = raw["structural"]
    questions = {q.id: q for q in load_questions()}
    level = MatchLevel.SECTION

    print("=" * 70)
    print("WHY THE NUMBERS CAME OUT THAT WAY")
    print("=" * 70)

    # --- who gets what the others miss -------------------------------------
    print("\nExclusive wins at Hit@3, section level")
    print("-" * 70)
    for name, other in (("bm25", "dense"), ("dense", "bm25")):
        wins = exclusive_wins(structural[name], structural[other], level)
        categories = Counter(questions[q].category for q in wins)
        print(f"\n  {name} gets {len(wins)} that {other} misses")
        if categories:
            print(f"    by category: {dict(categories)}")
        for qid in wins[:5]:
            print(f"    {qid}: {questions[qid].question[:58]}")

    # --- what fusion did ----------------------------------------------------
    print("\n" + "-" * 70)
    print("What RRF fusion actually did")
    print("-" * 70)
    for name in ("bm25", "dense"):
        improved, degraded = rank_changes(structural[name], structural["hybrid"], level)
        net = len(improved) - len(degraded)
        print(f"\n  versus {name}:")
        print(f"    fusion improved the rank on  {len(improved):>3} questions")
        print(f"    fusion worsened the rank on  {len(degraded):>3} questions")
        print(f"    net                          {net:>+3}")

    # --- where the gold passage was, when it was missed ---------------------
    print("\n" + "-" * 70)
    print("Misses: where was the gold passage?")
    print("-" * 70)
    for name in ("bm25", "dense", "hybrid"):
        results = [r for r in structural[name] if r.answerable]
        missed_entirely = [r for r in results if r.section_rank is None]
        deep = [r for r in results if r.section_rank is not None and r.section_rank > 5]
        wrong_passage = [
            r for r in results if r.section_rank is None and r.file_rank is not None
        ]
        print(f"\n  {name}")
        print(f"    not in the top 20 at all      {len(missed_entirely):>3}")
        print(f"      of those, right file found  {len(wrong_passage):>3}")
        print(f"    found but below rank 5        {len(deep):>3}")

    # --- the questions nobody gets ------------------------------------------
    print("\n" + "-" * 70)
    print("Questions no retriever answers at section level")
    print("-" * 70)
    universal = [
        qid
        for qid in (r.question_id for r in structural["bm25"] if r.answerable)
        if all(
            by_id(structural[name])[qid].section_rank is None
            for name in ("bm25", "dense", "hybrid")
        )
    ]
    print(f"\n  {len(universal)} questions:")
    for qid in universal:
        question = questions[qid]
        print(f"    {qid} [{question.category}] {question.question[:52]}")
    if universal:
        print("\n  These are where the next improvement is, if there is one.")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
