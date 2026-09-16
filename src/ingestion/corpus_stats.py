"""Print statistics about the loaded corpus.

Run:  python -m src.ingestion.corpus_stats

These numbers feed two decisions. They go in the README so a reader knows what
the retrieval numbers were measured over, and they tell us whether the planned
512-token chunk size is sensible for documents of this shape — a corpus whose
average document is 80k tokens splits into a very different number of chunks
than one averaging 2k, and that changes embedding time and index size.
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path

import tiktoken
from lxml import etree

from src.ingestion.loader import CODE_ELEMENTS, Document, load_corpus

CORPUS_ROOT = Path("data/raw/sgml")

# Document titles contain em-dashes and accented characters. A Windows console
# defaults to cp1252 and raises UnicodeEncodeError on those, so force UTF-8 on
# our own streams rather than mangling the data to suit the terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# cl100k_base is a general-purpose BPE tokenizer. It is not the tokenizer the
# embedding model uses, so these counts are indicative rather than exact — good
# enough to size chunks against, and far faster than loading a transformer.
_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text, disallowed_special=()))


def max_section_depth(doc: Document) -> int:
    """Deepest nesting level of sect1/sect2/... in the document.

    Phase 2 prepends the heading path to every chunk, so deep nesting means
    long prefixes eating into the token budget.
    """
    if doc.raw_xml is None:
        return 0
    depth = 0
    for element in doc.raw_xml.iter():
        if not isinstance(element.tag, str):
            continue
        tag = etree.QName(element).localname
        if tag.startswith("sect") and tag[4:].isdigit():
            depth = max(depth, int(tag[4:]))
    return depth


def count_code_blocks(doc: Document) -> int:
    """Atomic blocks Phase 2 must not split."""
    if doc.raw_xml is None:
        return 0
    return sum(
        1
        for el in doc.raw_xml.iter()
        if isinstance(el.tag, str) and etree.QName(el).localname in CODE_ELEMENTS
    )


def _histogram(label: str, counts: Counter, total: int) -> None:
    print(f"\n{label}")
    width = max((len(str(k)) for k in counts), default=0)
    for key, n in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        bar = "#" * max(1, round(40 * n / total))
        print(f"  {key!s:<{width}}  {n:5d}  {100 * n / total:5.1f}%  {bar}")


def main() -> int:
    if not CORPUS_ROOT.exists():
        print(f"Corpus not found at {CORPUS_ROOT}.", file=sys.stderr)
        print("Run: bash data/download.sh", file=sys.stderr)
        return 1

    docs = load_corpus(CORPUS_ROOT)
    if not docs:
        print("Corpus loaded but contained no documents.", file=sys.stderr)
        return 1

    token_counts = [count_tokens(d.content) for d in docs]
    total_tokens = sum(token_counts)

    version = Path("data/raw/CORPUS_VERSION")
    print("=" * 66)
    print("CORPUS STATISTICS")
    if version.exists():
        for line in version.read_text().strip().splitlines():
            print(f"  {line}")
    print("=" * 66)

    print(f"\nDocuments loaded      {len(docs):>10,}")
    print(f"Total tokens          {total_tokens:>10,}")
    print(f"Mean tokens/doc       {statistics.mean(token_counts):>10,.0f}")
    print(f"Median tokens/doc     {statistics.median(token_counts):>10,.0f}")
    print(f"Min tokens/doc        {min(token_counts):>10,}")
    print(f"Max tokens/doc        {max(token_counts):>10,}")

    # The median matters more than the mean here: a handful of very large
    # documents (config.sgml is one) drag the mean far above what a typical
    # document looks like.
    ordered = sorted(token_counts)
    for pct in (50, 75, 90, 95, 99):
        idx = min(len(ordered) - 1, int(len(ordered) * pct / 100))
        print(f"p{pct:<2d} tokens/doc         {ordered[idx]:>10,}")

    _histogram("Documents by type:", Counter(d.doc_type for d in docs), len(docs))
    _histogram(
        "Max section depth (sect1..sectN):",
        Counter(max_section_depth(d) for d in docs),
        len(docs),
    )

    code_blocks = [count_code_blocks(d) for d in docs]
    print(f"\nAtomic code blocks    {sum(code_blocks):>10,}")
    print(
        f"Docs containing code  {sum(1 for c in code_blocks if c):>10,}"
        f"  ({100 * sum(1 for c in code_blocks if c) / len(docs):.0f}%)"
    )

    print("\nLargest documents:")
    for tokens, doc in sorted(
        zip(token_counts, docs, strict=True), key=lambda p: -p[0]
    )[:8]:
        print(f"  {tokens:>8,}  {doc.path:<34s} {doc.title[:28]}")

    # What this implies for Phase 2.
    target, overlap = 512, 64
    stride = target - overlap
    est_chunks = sum(max(1, -(-t // stride)) for t in token_counts)
    print("\n" + "-" * 66)
    print(f"At {target}-token chunks with {overlap} overlap (stride {stride}):")
    print(f"  estimated chunks    {est_chunks:>10,}")
    print(f"  mean chunks/doc     {est_chunks / len(docs):>10,.1f}")
    print("-" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
