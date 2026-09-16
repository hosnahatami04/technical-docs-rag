# RAG over Technical Documentation

Retrieval-augmented question answering over the PostgreSQL documentation, built
to measure rather than assert. Three retrievers are implemented and compared on
the same labeled question set, and two independent gates decide whether an
answer is grounded and whether the question was answerable at all.

> Status: **Phase 1 of 7 complete** (corpus ingestion). Retrieval numbers are
> not available yet — this README will carry the real ones, not placeholders.

## Why this repo exists

Anyone can wire up a vector store and call it RAG. The contribution here is the
comparison: BM25 alone, dense embeddings alone, and a hybrid of both, measured
side by side. If the hybrid does not win, that gets reported too.

## Corpus

| | |
|---|---|
| Source | PostgreSQL official documentation (DocBook XML) |
| Version | 17.0, pinned to git tag `REL_17_0` |
| Documents | 379 loaded from 386 source files |
| Total tokens | 1,344,576 |
| Median / mean tokens per document | 983 / 3,548 |
| Largest document | `func.sgml`, 114,264 tokens |
| Atomic code blocks | 3,917 across 87% of documents |

The corpus is pinned by git tag and verified by SHA-256 checksum on download, so
the numbers reported here can be reproduced rather than taken on trust.
Full statistics: [`results/corpus_stats.txt`](results/corpus_stats.txt).

Two properties of this corpus shape later phases:

- **The length distribution is heavily skewed.** Half the documents are under
  1,000 tokens, but eight exceed 24,000. `func.sgml` alone will produce roughly
  8% of all chunks.
- **Structure comes in two shapes.** 220 SQL command pages are `<refentry>`
  documents using `<refsect1>`; the remaining 159 are chapters using
  `<sect1>`/`<sect2>`. The chunker has to handle both.

## Quick start

```bash
pip install -r requirements.txt   # or: make install
bash data/download.sh             # or: make corpus   (~27MB, pinned)
make stats                        # corpus statistics
make check                        # lint + tests
```

## Project layout

```
src/ingestion/   loader.py, corpus_stats.py        Phase 1  ✅
                 chunker.py, indexer.py            Phase 2
src/retrieval/   bm25.py, dense.py, hybrid.py      Phase 4
src/generation/  answerer.py, *_gate.py            Phase 6
src/eval/        metrics.py, runner.py, report.py  Phase 5
questions/       70 labeled Q&A pairs              Phase 3
results/         evaluation outputs, committed
```

## Ownership

Personal project built against public PostgreSQL documentation. No employer
data or proprietary content is involved.
