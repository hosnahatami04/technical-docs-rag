"""Compare the two chunking strategies over the whole corpus.

Run:  python -m src.ingestion.chunk_stats

This is descriptive, not evaluative. It reports what each strategy produces —
counts, size distributions, how many chunks exceed the embedding limit — but it
cannot say which retrieves better. That needs the labeled question set (Phase 3)
and the metrics harness (Phase 5). Committing these numbers now gives that
comparison a baseline to be measured against.
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path

from src.ingestion.chunker import (
    EMBED_MODEL_LIMIT,
    OVERLAP_TOKENS,
    TARGET_TOKENS,
    Chunk,
    chunk_corpus,
)
from src.ingestion.loader import load_corpus

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPUS_ROOT = Path("data/raw/sgml")


def _percentiles(values: list[int]) -> dict[int, int]:
    ordered = sorted(values)
    return {
        p: ordered[min(len(ordered) - 1, int(len(ordered) * p / 100))]
        for p in (10, 25, 50, 75, 90, 99)
    }


def describe(name: str, chunks: list[Chunk]) -> None:
    tokens = [c.token_count for c in chunks]
    oversized = [c for c in chunks if c.token_count > EMBED_MODEL_LIMIT]

    print(f"\n{name}")
    print("-" * 66)
    print(f"  chunks                {len(chunks):>10,}")
    print(f"  total tokens          {sum(tokens):>10,}")
    print(f"  mean tokens/chunk     {statistics.mean(tokens):>10,.0f}")
    print(f"  median tokens/chunk   {statistics.median(tokens):>10,.0f}")
    print(f"  max tokens/chunk      {max(tokens):>10,}")

    pcts = _percentiles(tokens)
    print(
        "  percentiles           " + "  ".join(f"p{p}={v:,}" for p, v in pcts.items())
    )

    # Chunks past the embedding limit are the cost of never splitting an atomic
    # block. The model reads only the first 512 tokens of these, silently.
    share = 100 * len(oversized) / len(chunks)
    print(
        f"\n  over the {EMBED_MODEL_LIMIT}-token embedding limit: "
        f"{len(oversized):,} ({share:.1f}%)"
    )
    if oversized:
        kinds = Counter(
            "table" if c.has_table else "code" if c.has_code else "prose"
            for c in oversized
        )
        print("    by kind: " + ", ".join(f"{k}={v}" for k, v in kinds.most_common()))
        truncated = sum(c.token_count - EMBED_MODEL_LIMIT for c in oversized)
        print(f"    tokens the embedder will not see: {truncated:,}")

    # Very small chunks carry too little context to retrieve or generate from.
    tiny = sum(1 for t in tokens if t < 50)
    print(f"  under 50 tokens:        {tiny:,} ({100 * tiny / len(chunks):.1f}%)")

    with_headings = sum(1 for c in chunks if c.heading_path)
    print(
        f"  carrying a heading path: {with_headings:,} "
        f"({100 * with_headings / len(chunks):.0f}%)"
    )

    print("\n  by document type:")
    by_type = Counter(c.doc_type for c in chunks)
    for doc_type, count in by_type.most_common():
        print(f"    {doc_type:<14} {count:>7,}")


def main() -> int:
    if not CORPUS_ROOT.exists():
        print(f"Corpus not found at {CORPUS_ROOT}.", file=sys.stderr)
        print("Run: bash data/download.sh", file=sys.stderr)
        return 1

    docs = load_corpus(CORPUS_ROOT)

    print("=" * 66)
    print("CHUNKING COMPARISON")
    print(f"  documents         {len(docs):,}")
    print(f"  target tokens     {TARGET_TOKENS}")
    print(f"  overlap tokens    {OVERLAP_TOKENS}")
    print("=" * 66)

    fixed = chunk_corpus(docs, strategy="fixed")
    structural = chunk_corpus(docs, strategy="structural")

    describe("FIXED — naive baseline, cut every N tokens", fixed)
    describe("STRUCTURAL — section-aware, atomic blocks kept whole", structural)

    print("\n" + "=" * 66)
    print("WHAT THIS DOES AND DOES NOT SHOW")
    print("=" * 66)
    ratio = len(structural) / len(fixed)
    print(f"""
  Structural produces {ratio:.1f}x the chunks of fixed, and they are smaller:
  it breaks at section boundaries rather than filling to a token budget.

  Whether that helps retrieval is not answerable here. Smaller chunks give
  sharper embeddings but less context; heading prefixes add signal but also
  add tokens. The labeled question set (Phase 3) and the metrics harness
  (Phase 5) decide it, and that is where the tuning belongs.

  What can be said now: {sum(1 for c in structural if c.heading_path):,} of
  {len(structural):,} structural chunks carry an explicit heading path, and
  none of the fixed ones do.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
