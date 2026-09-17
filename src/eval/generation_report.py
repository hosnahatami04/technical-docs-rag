"""Turn the generation evaluation record into a Markdown report.

Run:  python -m src.eval.generation_report

Reads results/generation_raw.json and writes results/generation.md.

What this report has to make visible, and what a single quality score would
hide: which gate stopped what. The plan's example sentence is
"the answerability gate catches 13 of 15 unanswerable questions; the grounding
gate catches 4 hallucinated claims that slipped past it" — two numbers that only
exist because the gates are separate.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from src.eval.generation_metrics import (
    GenerationResult,
    GenerationSummary,
    summarize_generation,
)
from src.eval.generation_runner import RAW_PATH, load_generation_raw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path("results")

CONFIG_LABELS = {
    "both_gates": "Both gates",
    "no_gates": "No gates",
    "answerability_only": "Answerability gate only",
    "threshold_gate": "Threshold gate (instead of LLM)",
}

# 15 unanswerable questions, so one is worth 6.7 percentage points. That is a
# coarse instrument, and the report says so rather than reporting a two-point
# difference as a result.
ONE_UNANSWERABLE = 1 / 15


def _pct(value: float) -> str:
    return f"{value:.3f}"


def _pct_or_dash(value: float, measured: bool) -> str:
    """Render an unmeasured metric as a dash rather than 0.000.

    A configuration with the grounding gate disabled has no groundedness
    score at all, and printing 0.000 reads as "nothing was supported" — the
    opposite of what it means.
    """
    return _pct(value) if measured else "—"


def _summary_table(summaries: Sequence[GenerationSummary]) -> list[str]:
    rows = [
        "| Configuration | Correct abstention | Wrong abstention | "
        "Answered unanswerable | Groundedness (pre-gate) | Groundedness "
        "(delivered) | Answerability F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        rows.append(
            f"| {CONFIG_LABELS.get(s.label, s.label)} | "
            f"{_pct(s.correct_abstention_rate)} | "
            f"{_pct(s.wrong_abstention_rate)} | "
            f"{s.answered_unanswerable} / {s.n_unanswerable} | "
            f"{_pct_or_dash(s.groundedness_pre_gate, s.total_claims > 0)} | "
            f"{_pct_or_dash(s.groundedness, s.total_claims > 0)} | "
            f"{_pct(s.answerability_f1)} |"
        )
    return rows


def _gate_attribution(results: list[GenerationResult]) -> list[str]:
    """Which stage stopped each abstention, and whether it was right to."""
    lines: list[str] = []
    stopped = Counter(r.stopped_by for r in results if r.abstained)

    lines.append(
        "| Stopped by | Total | Correctly (unanswerable) | Wrongly (answerable) |"
    )
    lines.append("|---|---|---|---|")
    for stage in ("answerability", "generator", "grounding", "retrieval"):
        total = stopped.get(stage, 0)
        if not total:
            continue
        correct = sum(
            1
            for r in results
            if r.abstained and r.stopped_by == stage and not r.answerable
        )
        wrong = total - correct
        lines.append(f"| {stage} | {total} | {correct} | {wrong} |")
    return lines


def build_report(configurations: dict[str, list[GenerationResult]]) -> str:
    lines: list[str] = []
    add = lines.append

    primary = configurations.get("both_gates", next(iter(configurations.values())))
    summaries = [
        summarize_generation(results, label=label)
        for label, results in configurations.items()
    ]
    main = summarize_generation(primary, label="both_gates")

    add("# Generation and gate evaluation")
    add("")
    add(
        f"{main.n_questions} questions through the full pipeline: retrieve, "
        f"answerability gate, generate, grounding gate. "
        f"{main.n_answerable} are answerable and {main.n_unanswerable} are not. "
        f"Raw record: `results/generation_raw.json`."
    )
    add("")

    # --- what the metrics mean ---------------------------------------------
    add("## How to read these numbers")
    add("")
    add(
        "**Correct abstention** — of the questions the corpus genuinely cannot "
        "answer, how many did the system refuse? This is the headline number, "
        "and on its own it is misleading: a system that refuses everything "
        "scores 1.000."
    )
    add("")
    add(
        "**Wrong abstention** — of the answerable questions, how many did it "
        "refuse anyway? This is what stops the first number from being gamed. "
        "The two are read together."
    )
    add("")
    add(
        "**Groundedness** — the fraction of sentences in a generated answer "
        "that the cited sources actually support. Reported twice, and the "
        "difference matters:"
    )
    add("")
    add(
        "- *pre-gate* averages over every answer the model wrote, including "
        "the ones the grounding gate then rejected. This measures the "
        "generator."
    )
    add(
        "- *delivered* averages over the answers that survived the gate. With "
        "the gate enabled this is near-tautological — anything below the 0.8 "
        "threshold was converted into an abstention, so the survivors cannot "
        "score below it. A value of 1.000 here is a property of the gate, not "
        "evidence the model is faithful."
    )
    add("")
    add(
        "Reporting only the second number would be the mistake this project "
        "exists to avoid: it looks like a perfect score and is guaranteed by "
        "construction."
    )
    add("")
    add(
        "**Answerability F1** — treating 'this question is answerable' as a "
        "binary classification. F1 balances refusing too much against refusing "
        "too little."
    )
    add("")
    add(
        f"With {main.n_unanswerable} unanswerable questions, one question is "
        f"worth {ONE_UNANSWERABLE:.3f}. This is a coarse instrument and "
        f"differences of one or two questions should not be read as results."
    )
    add("")

    # --- the ablation ------------------------------------------------------
    add("## What each gate contributes")
    add("")
    add(
        "Each configuration changes one thing, so the effect of each gate can "
        "be attributed rather than assumed."
    )
    add("")
    lines.extend(_summary_table(summaries))
    add("")

    no_gates = next((s for s in summaries if s.label == "no_gates"), None)
    if no_gates:
        delta = main.correct_abstention_rate - no_gates.correct_abstention_rate
        add(
            f"Without gates, the system correctly refuses "
            f"{_pct(no_gates.correct_abstention_rate)} of unanswerable "
            f"questions; with both gates, {_pct(main.correct_abstention_rate)} "
            f"— a difference of {delta:+.3f}, about "
            f"{abs(delta) / ONE_UNANSWERABLE:.0f} questions."
        )
        add("")
        add(
            "The ungated number is not zero because the generator sometimes "
            "declines on its own when the sources plainly do not answer. That "
            "is not a substitute for a gate: it is unmeasured, unswitchable, "
            "and cannot be reported separately."
        )
        add("")

    # --- gate attribution ---------------------------------------------------
    add("## Which gate stopped what")
    add("")
    add(
        "The reason the gates are separate modules. A single quality check "
        "could report that the system abstained, but not whether it abstained "
        "because nothing could answer the question or because the answer it "
        "wrote was unsupported."
    )
    add("")
    lines.extend(_gate_attribution(primary))
    add("")

    grounding_catches = sum(
        1 for r in primary if r.abstained and r.stopped_by == "grounding"
    )
    answerability_catches = sum(
        1
        for r in primary
        if r.abstained and r.stopped_by == "answerability" and not r.answerable
    )
    add(
        f"In the plan's terms: the answerability gate catches "
        f"{answerability_catches} of {main.n_unanswerable} unanswerable "
        f"questions, and the grounding gate catches {grounding_catches} "
        f"answers whose claims the sources did not support."
    )
    add("")

    # --- groundedness detail ------------------------------------------------
    add("## Groundedness")
    add("")
    add(f"- {main.total_claims} claims checked across every answer written")
    add(f"- {main.unsupported_claims} were not supported by the cited sources")
    add(
        f"- groundedness before the gate: {_pct(main.groundedness_pre_gate)} "
        f"— what the generator produced"
    )
    add(
        f"- groundedness of delivered answers: {_pct(main.groundedness)} "
        f"— what a user receives, bounded below by the gate's threshold"
    )
    add(
        f"- {main.n_grounded_answers} answers passed the 0.8 threshold and "
        f"were returned"
    )
    add("")

    unsupported_examples = [
        (r.question_id, claim) for r in primary for claim in r.unsupported_claims
    ][:5]
    if unsupported_examples:
        add("Claims the gate rejected:")
        add("")
        for qid, claim in unsupported_examples:
            add(f"- `{qid}` — {claim}")
        add("")

    # --- per category -------------------------------------------------------
    add("## By question category")
    add("")
    add("| Category | n | Abstained | Groundedness |")
    add("|---|---|---|---|")
    categories = sorted({r.category for r in primary})
    for category in categories:
        rows = [r for r in primary if r.category == category]
        answered = [
            r for r in rows if not r.abstained and r.grounding_score is not None
        ]
        grounded = (
            sum(r.grounding_score or 0 for r in answered) / len(answered)
            if answered
            else 0.0
        )
        add(
            f"| {category} | {len(rows)} | "
            f"{sum(1 for r in rows if r.abstained)} | "
            f"{_pct(grounded)} |"
        )
    add("")

    # --- limits -------------------------------------------------------------
    add("## What these numbers do not show")
    add("")
    add(
        "- **The judge is the model being judged.** Both gates use the same "
        "model that writes the answers. A claim it believes it can support, it "
        "will call supported. An independent judge would be a stronger test."
    )
    add(
        "- **Groundedness is not correctness.** A sentence faithfully copied "
        "from a source the retriever should not have returned scores 1.000."
    )
    add(
        f"- **Sample size.** {main.n_unanswerable} unanswerable questions is "
        f"enough to see whether abstention works at all, not enough to "
        f"distinguish two configurations that differ by one question."
    )
    add("")

    return "\n".join(lines) + "\n"


def main() -> int:
    if not RAW_PATH.exists():
        print(f"No results at {RAW_PATH}.", file=sys.stderr)
        print("Run: python -m src.eval.generation_runner", file=sys.stderr)
        return 1

    configurations = load_generation_raw()
    report = build_report(configurations)

    out_path = RESULTS_DIR / "generation.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
