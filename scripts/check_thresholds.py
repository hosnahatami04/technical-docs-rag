"""Fail the build when a committed result falls below its recorded floor.

Run:  python scripts/check_thresholds.py

The plan asks CI to fail if Hit@3 drops below the recorded value or
groundedness falls under threshold. A CI runner cannot reproduce those numbers
— no corpus, no embedding index, no GPU — so this checks what is committed.

That is still a real gate. The numbers in results/ are what the project quotes,
so a change that degrades retrieval and regenerates them fails here, and so
does editing them by hand to look better. What it cannot catch is a change that
degrades retrieval without regenerating the results; `make eval` is what closes
that gap, and it is the developer's job to run it.

Thresholds sit a little below the values measured on 2026-09-17, so ordinary
run-to-run movement does not fail the build while a real regression does.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Deliberately standard library only. This script imports nothing from src/,
# which means the CI job that runs it needs no dependencies — no torch, no
# tiktoken, no embedding model. It reads the committed JSON and does the
# arithmetic itself.
#
# The duplication with src/eval/metrics.py is the price, and it is worth
# paying: a gate that needs 500MB of PyTorch installed to compute an average
# is a gate that gets skipped.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"

# One question is worth 1/55 = 0.018 on retrieval and 1/15 = 0.067 on
# abstention. The floors are set roughly two questions below what was measured,
# so a single question changing rank is not a build failure but a genuine
# regression is.


@dataclass(frozen=True)
class Threshold:
    label: str
    measured: float
    floor: float

    def check(self, actual: float) -> tuple[bool, str]:
        ok = actual >= self.floor
        arrow = "ok" if ok else "FAIL"
        return ok, (
            f"  [{arrow}] {self.label:<44} {actual:.3f}  "
            f"(floor {self.floor:.3f}, measured {self.measured:.3f})"
        )


RETRIEVAL_THRESHOLDS = {
    # (strategy, retriever, level, metric): Threshold
    ("structural", "hybrid", "section", "hit@3"): Threshold(
        "hybrid Hit@3, section level", 0.655, 0.600
    ),
    ("structural", "hybrid", "section", "hit@5"): Threshold(
        "hybrid Hit@5, section level", 0.800, 0.745
    ),
    ("structural", "dense", "section", "mrr"): Threshold(
        "dense MRR, section level", 0.613, 0.560
    ),
    ("structural", "hybrid", "file", "hit@3"): Threshold(
        "hybrid Hit@3, file level", 0.873, 0.818
    ),
}

GENERATION_THRESHOLDS = {
    "correct_abstention": Threshold("correct abstention, both gates", 0.867, 0.733),
    "groundedness_pre_gate": Threshold("groundedness before the gate", 0.921, 0.850),
    "answerability_f1": Threshold("answerability F1, both gates", 0.871, 0.800),
}


def _fail(message: str) -> int:
    print(f"\nFAILED: {message}", file=sys.stderr)
    return 1


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def check_retrieval(failures: list[str]) -> None:
    path = RESULTS / "eval_raw.json"
    if not path.exists():
        failures.append(f"{path} is missing. Run: make eval")
        return

    raw = json.loads(path.read_text(encoding="utf-8"))["strategies"]
    print("\nRetrieval (results/eval_raw.json)")

    for (strategy, retriever, level, metric), threshold in RETRIEVAL_THRESHOLDS.items():
        rows = raw.get(strategy, {}).get(retriever)
        if not rows:
            failures.append(f"missing results for {strategy}/{retriever}")
            continue

        # Only answerable questions are scored; the rest have no gold passage,
        # so every retriever scores zero on them. Same rule as src/eval/metrics.
        scored = [r for r in rows if r["answerable"]]
        rank_field = "file_rank" if level == "file" else "section_rank"
        ranks = [r[rank_field] for r in scored]

        if metric == "mrr":
            actual = _mean([1.0 / rank if rank else 0.0 for rank in ranks])
        else:
            k = int(metric.removeprefix("hit@"))
            actual = _mean([1.0 if rank and rank <= k else 0.0 for rank in ranks])

        ok, line = threshold.check(actual)
        print(line)
        if not ok:
            failures.append(threshold.label)


def check_generation(failures: list[str]) -> None:
    path = RESULTS / "generation_raw.json"
    if not path.exists():
        failures.append(f"{path} is missing. Run: make generate")
        return

    configurations = json.loads(path.read_text(encoding="utf-8"))["configurations"]
    rows = configurations.get("both_gates")
    if not rows:
        failures.append("generation results have no 'both_gates' configuration")
        return

    print("\nGeneration (results/generation_raw.json)")

    unanswerable = [r for r in rows if not r["answerable"]]
    answerable = [r for r in rows if r["answerable"]]
    # Groundedness before the gate: every answer the model wrote, including the
    # ones the gate then rejected. The post-gate number cannot fall below the
    # threshold by construction, so it would be a useless thing to gate on.
    scored = [r["grounding_score"] for r in rows if r["grounding_score"] is not None]

    # Answerability as a binary classification, positive = "answerable".
    true_positive = sum(1 for r in answerable if not r["abstained"])
    false_negative = sum(1 for r in answerable if r["abstained"])
    false_positive = sum(1 for r in unanswerable if not r["abstained"])
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )

    actuals = {
        "correct_abstention": _mean(
            [1.0 if r["abstained"] else 0.0 for r in unanswerable]
        ),
        "groundedness_pre_gate": _mean(scored),
        "answerability_f1": (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        ),
    }
    for key, threshold in GENERATION_THRESHOLDS.items():
        ok, line = threshold.check(actuals[key])
        print(line)
        if not ok:
            failures.append(threshold.label)


def check_reports_exist(failures: list[str]) -> None:
    """The generated reports must be committed alongside the raw records.

    They are what a reader actually opens, and a raw file updated without its
    report means the prose and the numbers disagree.
    """
    print("\nGenerated reports")
    for name in (
        "evaluation.md",
        "generation.md",
        "corpus_stats.txt",
        "chunk_stats.txt",
    ):
        path = RESULTS / name
        exists = path.exists()
        print(f"  [{'ok' if exists else 'FAIL'}] {name}")
        if not exists:
            failures.append(f"{path} is missing")


def main() -> int:
    if not RESULTS.exists():
        return _fail(f"{RESULTS}/ does not exist.")

    print("=" * 68)
    print("QUALITY GATE")
    print("  Checks the committed results against their recorded floors.")
    print("=" * 68)

    failures: list[str] = []
    try:
        check_retrieval(failures)
        check_generation(failures)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        # A results file that does not have the shape this script expects is a
        # failure, not something to report as "all checks passed".
        return _fail(f"results file is malformed: {type(exc).__name__}: {exc}")
    check_reports_exist(failures)

    print("\n" + "=" * 68)
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        print("=" * 68)
        print(
            "\nIf this is an intentional change, regenerate the results with "
            "`make eval` and `make generate`, then update the thresholds in "
            "this file with the new measured values."
        )
        return 1

    print("All checks passed.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
