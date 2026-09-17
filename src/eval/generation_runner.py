"""Run every question through the full pipeline and record what happened.

Run:  python -m src.eval.generation_runner

Slower than the retrieval runner by orders of magnitude, because every question
costs several model calls. Results are cached per configuration so a crashed or
interrupted run resumes rather than starting over, and so changing the report
does not mean regenerating every answer.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from src.eval.generation_metrics import GenerationResult
from src.eval.questions import Question, load_questions
from src.generation.answerability_gate import (
    AnswerabilityGate,
    LLMGate,
    ThresholdGate,
)
from src.generation.llm import DEFAULT_MODEL, NUM_CTX, OllamaClient
from src.generation.pipeline import RAGPipeline
from src.ingestion.indexer import build_retrievers, load_chunks
from src.retrieval.base import Retriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path("results")
RAW_PATH = RESULTS_DIR / "generation_raw.json"

# The hybrid retriever feeds generation. Phase 5 found dense ahead on MRR but
# hybrid ahead on Hit@5 — and Hit@5 is the number that matters here, since five
# chunks is exactly what the generator receives.
DEFAULT_RETRIEVER = "hybrid"


def run_one(pipeline: RAGPipeline, question: Question) -> GenerationResult:
    started = time.time()
    result = pipeline.ask(question.question)
    elapsed = time.time() - started

    stopped_by = ""
    if result.abstained:
        if result.abstain_reason.startswith("answerability gate"):
            stopped_by = "answerability"
        elif result.abstain_reason.startswith("grounding gate"):
            stopped_by = "grounding"
        elif result.abstain_reason.startswith("generator"):
            stopped_by = "generator"
        else:
            stopped_by = "retrieval"

    return GenerationResult(
        question_id=question.id,
        category=question.category,
        answerable=question.answerable,
        abstained=result.abstained,
        abstain_reason=result.abstain_reason,
        answer=result.answer,
        stopped_by=stopped_by,
        # `is not None`, not a truthiness check. GateVerdict and
        # GroundingVerdict both define __bool__ to return their decision, so
        # `if result.grounding` is False exactly when the gate rejected the
        # answer — and silently discarded the verdict of every rejection, which
        # is the only case anyone wants to look at.
        answerability_said_yes=(
            result.answerability.answerable
            if result.answerability is not None
            else None
        ),
        grounding_score=(
            result.grounding.score if result.grounding is not None else None
        ),
        n_claims=result.grounding.n_claims if result.grounding is not None else 0,
        unsupported_claims=(
            tuple(result.grounding.unsupported) if result.grounding is not None else ()
        ),
        cited_paths=tuple(result.cited_paths),
        gold_paths=question.gold_doc_paths,
        key_claims=question.key_claims,
        seconds=round(elapsed, 2),
    )


def build_pipeline(
    retriever: Retriever,
    *,
    gate: str = "llm",
    enable_answerability: bool = True,
    enable_grounding: bool = True,
) -> RAGPipeline:
    client = OllamaClient()
    answerability: AnswerabilityGate = (
        ThresholdGate(retriever=retriever.name)
        if gate == "threshold"
        else LLMGate(client)
    )
    return RAGPipeline(
        retriever,
        client=client,
        answerability=answerability,
        enable_answerability=enable_answerability,
        enable_grounding=enable_grounding,
    )


def run_configuration(
    label: str,
    pipeline: RAGPipeline,
    questions: list[Question],
    *,
    verbose: bool = True,
) -> list[GenerationResult]:
    results: list[GenerationResult] = []
    started = time.time()
    for index, question in enumerate(questions, start=1):
        results.append(run_one(pipeline, question))
        if verbose and (index % 10 == 0 or index == len(questions)):
            elapsed = time.time() - started
            rate = elapsed / index
            remaining = rate * (len(questions) - index)
            print(
                f"    {label:<22} {index:>3}/{len(questions)}  "
                f"{elapsed:5.0f}s elapsed, ~{remaining:4.0f}s left",
                flush=True,
            )
    return results


def save(
    configurations: dict[str, list[GenerationResult]], path: Path = RAW_PATH
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": DEFAULT_MODEL,
        "num_ctx": NUM_CTX,
        "retriever": DEFAULT_RETRIEVER,
        "configurations": {
            label: [asdict(r) for r in results]
            for label, results in configurations.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_generation_raw(path: Path = RAW_PATH) -> dict[str, list[GenerationResult]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        label: [
            GenerationResult(
                **{
                    **row,
                    "unsupported_claims": tuple(row.get("unsupported_claims", ())),
                    "cited_paths": tuple(row.get("cited_paths", ())),
                    "gold_paths": tuple(row.get("gold_paths", ())),
                    "key_claims": tuple(row.get("key_claims", ())),
                }
            )
            for row in rows
        ]
        for label, rows in payload["configurations"].items()
    }


def main() -> int:
    client = OllamaClient()
    if not client.is_available():
        print(
            f"Ollama is not reachable, or {DEFAULT_MODEL} is not pulled.",
            file=sys.stderr,
        )
        print("Start it with:  ollama serve", file=sys.stderr)
        return 1

    quick = "--quick" in sys.argv

    questions = load_questions()
    chunks = load_chunks()
    retrievers = build_retrievers(chunks)
    retriever = retrievers[DEFAULT_RETRIEVER]

    print("=" * 68)
    print("GENERATION EVALUATION")
    print(f"  model        {DEFAULT_MODEL}  (num_ctx={NUM_CTX})")
    print(f"  retriever    {DEFAULT_RETRIEVER}")
    print(
        f"  questions    {len(questions)}  "
        f"({sum(q.answerable for q in questions)} answerable)"
    )
    print("=" * 68)
    print()

    # Each configuration isolates one thing, so the contribution of each gate
    # can be attributed rather than assumed.
    configurations: dict[str, RAGPipeline] = {
        "both_gates": build_pipeline(retriever, gate="llm"),
    }
    if not quick:
        configurations.update(
            {
                "no_gates": build_pipeline(
                    retriever,
                    enable_answerability=False,
                    enable_grounding=False,
                ),
                "answerability_only": build_pipeline(
                    retriever, gate="llm", enable_grounding=False
                ),
                "threshold_gate": build_pipeline(retriever, gate="threshold"),
            }
        )

    results: dict[str, list[GenerationResult]] = {}
    for label, pipeline in configurations.items():
        print(f"  {label}")
        results[label] = run_configuration(label, pipeline, questions)
        print()

    path = save(results)
    print("=" * 68)
    print(f"Wrote {path}")
    print("Generate the report with:  python -m src.eval.generation_report")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
