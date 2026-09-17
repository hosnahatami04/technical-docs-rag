"""Ask one question through the full pipeline and show every stage.

Run:  python -m src.generation.demo "how do I rebuild an index concurrently?"

With no argument it runs a short set chosen to show both outcomes — a question
the system answers and one it refuses.

This is for looking at behaviour, not measuring it. The numbers come from
`make generate`; this is how you see what produced them.
"""

from __future__ import annotations

import sys

from src.generation.llm import DEFAULT_MODEL, NUM_CTX, OllamaClient
from src.generation.pipeline import PipelineResult, RAGPipeline
from src.ingestion.indexer import build_retrievers, load_chunks

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# One answerable, one not. The second is a PostgreSQL 18 parameter, so every
# retriever returns confident results for it and only a gate can catch it.
DEFAULT_QUESTIONS = [
    "What is the default value of max_wal_size?",
    "What is the default value of autovacuum_vacuum_max_threshold?",
]


def show(result: PipelineResult) -> None:
    print("\n" + "=" * 70)
    print(f"Q: {result.question}")
    print("=" * 70)

    print("\n  retrieved:")
    for chunk in result.chunks:
        print(
            f"    {chunk.rank}. [{chunk.score:7.4f}] {chunk.doc_path:<22} "
            f"{chunk.heading[:40]}"
        )

    if result.answerability is not None:
        verdict = "PASS" if result.answerability.answerable else "STOP"
        print(f"\n  answerability gate: {verdict}")
        print(f"    {result.answerability.reason[:100]}")

    if result.grounding is not None:
        verdict = "PASS" if result.grounding.grounded else "STOP"
        print(
            f"\n  grounding gate: {verdict}  "
            f"({result.grounding.score:.0%} of {result.grounding.n_claims} claims)"
        )
        for claim in result.grounding.unsupported:
            print(f"    unsupported: {claim[:80]}")

    print()
    if result.abstained:
        print(f"  ABSTAINED — {result.abstain_reason[:90]}")
        print(f"  {result.answer}")
    else:
        print(f"  ANSWER: {result.answer}")
        if result.cited_paths:
            print(f"  cites: {', '.join(result.cited_paths)}")


def main() -> int:
    client = OllamaClient()
    if not client.is_available():
        print(
            f"Ollama is not reachable, or {DEFAULT_MODEL} is not pulled.",
            file=sys.stderr,
        )
        print("Start it with:  ollama serve", file=sys.stderr)
        return 1

    questions = sys.argv[1:] or DEFAULT_QUESTIONS

    print(f"model: {DEFAULT_MODEL} (num_ctx={NUM_CTX})")
    print("loading corpus and indexes ...")
    retrievers = build_retrievers(load_chunks())
    pipeline = RAGPipeline(retrievers["hybrid"], client=client)

    for question in questions:
        show(pipeline.ask(question))

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
