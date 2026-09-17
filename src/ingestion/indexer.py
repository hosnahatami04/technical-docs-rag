"""Build and cache the retrieval indexes.

Run:  python -m src.ingestion.indexer

One command turns the corpus into all three retrievers. The dense index is the
expensive part — embedding 6,406 chunks on CPU takes minutes — so it is
persisted to disk and reused. The cache key is a hash of the chunk ids, their
text, and the model name, because a stale index is worse than no index: it
would quietly evaluate new chunks against old embeddings and report the numbers
as if nothing had changed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from src.ingestion.chunker import Chunk, chunk_corpus
from src.ingestion.loader import load_corpus
from src.retrieval.base import Retriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DEFAULT_INDEX_DIR, DenseRetriever
from src.retrieval.hybrid import HybridRetriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPUS_ROOT = Path("data/raw/sgml")


def load_chunks(strategy: str = "structural") -> list[Chunk]:
    return chunk_corpus(load_corpus(CORPUS_ROOT), strategy=strategy)


def build_retrievers(
    chunks: list[Chunk],
    *,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    rebuild: bool = False,
    show_progress: bool = False,
) -> dict[str, Retriever]:
    """Build all three retrievers over the same chunks.

    Returned as a dict keyed by name so the Phase 5 harness can iterate over
    them without knowing what any of them is.
    """
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(
        chunks,
        index_dir=index_dir,
        rebuild=rebuild,
        show_progress=show_progress,
    )
    hybrid = HybridRetriever([bm25, dense])
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}


def main() -> int:
    if not CORPUS_ROOT.exists():
        print(f"Corpus not found at {CORPUS_ROOT}.", file=sys.stderr)
        print("Run: bash data/download.sh", file=sys.stderr)
        return 1

    rebuild = "--rebuild" in sys.argv

    print("=" * 66)
    print("BUILDING RETRIEVAL INDEXES")
    print("=" * 66)

    start = time.time()
    chunks = load_chunks()
    print(f"\n  chunks            {len(chunks):>8,}   ({time.time() - start:.1f}s)")

    start = time.time()
    bm25 = BM25Retriever(chunks)
    print(f"  bm25 index        {len(bm25):>8,}   ({time.time() - start:.1f}s)")

    print("\n  dense index (embedding is the slow step)")
    start = time.time()
    dense = DenseRetriever(chunks, rebuild=rebuild, show_progress=True)
    elapsed = time.time() - start
    cached = elapsed < 5 and not rebuild
    print(
        f"  dense index       {len(dense):>8,}   ({elapsed:.1f}s"
        f"{', cached' if cached else ''})"
    )
    print(f"  collection        {dense._collection_name}")

    hybrid = HybridRetriever([bm25, dense])
    print(f"  hybrid            {len(hybrid):>8,}   (fuses the two above)")

    # A single query through all three, as a smoke test that the indexes are
    # not merely built but usable.
    probe = "What is the default value of max_wal_size?"
    print("\n" + "-" * 66)
    print(f"  probe: {probe}")
    print("-" * 66)
    for name, retriever in (("bm25", bm25), ("dense", dense), ("hybrid", hybrid)):
        results = retriever.search(probe, k=1)
        if results:
            top = results[0]
            print(
                f"    {name:<8} [{top.score:7.4f}] {top.doc_path:<18} "
                f"{top.heading[:34]}"
            )
        else:
            print(f"    {name:<8} no results")

    print("\n" + "=" * 66)
    print(f"Indexes ready in {DEFAULT_INDEX_DIR}")
    print("Rebuild with:  python -m src.ingestion.indexer --rebuild")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
