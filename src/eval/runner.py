"""Run every question through every retriever and record what happened.

Run:  python -m src.eval.runner

Separating this from report.py is deliberate: running is slow and writes a
machine-readable file, reporting is instant and reads it. That split means the
report can be regenerated or reformatted without re-running retrieval, and the
raw per-question record stays available for anyone who wants to check a number
rather than trust the summary.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.eval.metrics import RETRIEVAL_DEPTH, QuestionResult, evaluate_question
from src.eval.questions import Question, load_questions
from src.ingestion.chunker import Chunk, chunk_corpus
from src.ingestion.loader import load_corpus
from src.retrieval.base import Retriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DEFAULT_INDEX_DIR, DenseRetriever
from src.retrieval.hybrid import HybridRetriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPUS_ROOT = Path("data/raw/sgml")
RESULTS_DIR = Path("results")

# Nothing in this pipeline samples, but the seed is set anyway: sentence
# transformers and numpy both have code paths that consult the global RNG, and
# a number that cannot be reproduced is not a measurement.
SEED = 42

# Retrieval runs to depth 20 so MRR can see a gold passage below rank 5, but
# only the top 5 results are written to disk. Ranks in the record are the real
# ones; the stored list is there so a reader can check a number by hand.
PERSIST_DEPTH = 5


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_retriever(
    retriever: Retriever,
    questions: list[Question],
    *,
    depth: int = RETRIEVAL_DEPTH,
) -> list[QuestionResult]:
    """Run one retriever over every question, in the order given.

    Questions are processed in their file order rather than sorted or shuffled,
    so two runs produce byte-identical output.
    """
    return [
        evaluate_question(
            question, retriever.search(question.question, k=depth), retriever.name
        )
        for question in questions
    ]


def build_all(
    chunks: list[Chunk],
    *,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    rebuild: bool = False,
) -> dict[str, Retriever]:
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, index_dir=index_dir, rebuild=rebuild)
    return {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever([bm25, dense])}


def run_all(
    questions: list[Question],
    *,
    strategy: str = "structural",
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    rebuild: bool = False,
    verbose: bool = True,
) -> dict[str, list[QuestionResult]]:
    """Every question through every retriever, for one chunking strategy."""
    set_seed()

    docs = load_corpus(CORPUS_ROOT)
    chunks = chunk_corpus(docs, strategy=strategy)
    if verbose:
        print(f"  {strategy:<12} {len(chunks):>6,} chunks")

    # Both ablation arms share one index directory. The collection name is
    # already a hash of the chunk ids and their text, so the two strategies
    # cannot collide — and sharing means the structural index built by
    # `make index` is reused rather than re-embedded, which is seven minutes
    # of CPU either way.
    retrievers = build_all(chunks, index_dir=index_dir, rebuild=rebuild)

    results: dict[str, list[QuestionResult]] = {}
    for name, retriever in retrievers.items():
        start = time.time()
        results[name] = run_retriever(retriever, questions)
        if verbose:
            elapsed = time.time() - start
            per_query = elapsed / len(questions) * 1000
            print(f"    {name:<8} {elapsed:>6.1f}s  ({per_query:.0f}ms/query)")

    return results


def save(
    results: dict[str, dict[str, list[QuestionResult]]],
    path: Path = RESULTS_DIR / "eval_raw.json",
) -> Path:
    """Write the per-question record, keyed by strategy then retriever.

    Committed so a reader can check any single number in the report against
    what the retriever actually returned, instead of taking the summary on
    trust.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def row(result: QuestionResult) -> dict[str, object]:
        data = asdict(result)
        # Ranks are computed over the full depth of 20, but only the top 5 are
        # stored. That is enough to check any Hit@k or MRR number in the report
        # by hand, and keeps the committed file readable rather than an
        # 800KB wall of headings.
        data["retrieved_paths"] = list(result.retrieved_paths[:PERSIST_DEPTH])
        data["retrieved_headings"] = list(result.retrieved_headings[:PERSIST_DEPTH])
        return data

    payload = {
        "seed": SEED,
        "retrieval_depth": RETRIEVAL_DEPTH,
        "persisted_results_per_question": PERSIST_DEPTH,
        "strategies": {
            strategy: {
                name: [row(result) for result in per_retriever]
                for name, per_retriever in by_retriever.items()
            }
            for strategy, by_retriever in results.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_raw(
    path: Path = RESULTS_DIR / "eval_raw.json",
) -> dict[str, dict[str, list[QuestionResult]]]:
    """Read back what save() wrote."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        strategy: {
            name: [
                QuestionResult(
                    **{
                        **row,
                        "retrieved_paths": tuple(row.get("retrieved_paths", ())),
                        "retrieved_headings": tuple(row.get("retrieved_headings", ())),
                    }
                )
                for row in rows
            ]
            for name, rows in by_retriever.items()
        }
        for strategy, by_retriever in payload["strategies"].items()
    }


def main() -> int:
    if not CORPUS_ROOT.exists():
        print(f"Corpus not found at {CORPUS_ROOT}.", file=sys.stderr)
        print("Run: bash data/download.sh", file=sys.stderr)
        return 1

    rebuild = "--rebuild" in sys.argv
    # The chunking ablation doubles the work: the fixed-chunk corpus needs its
    # own embeddings, which is another few minutes on CPU.
    strategies = ["structural"] if "--quick" in sys.argv else ["structural", "fixed"]

    questions = load_questions()

    print("=" * 66)
    print("EVALUATION RUN")
    print(f"  questions        {len(questions)}")
    print(f"  answerable       {sum(q.answerable for q in questions)}")
    print(f"  retrieval depth  {RETRIEVAL_DEPTH}")
    print(f"  seed             {SEED}")
    print("=" * 66)
    print()

    all_results: dict[str, dict[str, list[QuestionResult]]] = {}
    for strategy in strategies:
        all_results[strategy] = run_all(questions, strategy=strategy, rebuild=rebuild)
        print()

    path = save(all_results)
    print("=" * 66)
    print(f"Wrote {path}")
    print("Generate the report with:  python -m src.eval.report")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
