# technical-docs-rag

Answers questions about the PostgreSQL docs, and says "I don't know" when the answer isn't there.

## What this is

I built this to find out whether a RAG system can be made to admit what it
doesn't know, and to measure that rather than assume it.

The corpus is the PostgreSQL 17.0 documentation — 379 documents, about 1.3
million tokens. You ask a question in plain English, it finds the relevant
passages, writes an answer, and cites where each fact came from.

The part I care about is the refusals. Ask it about a parameter that doesn't
exist in this version and it won't invent one, even though the search comes
back full of very similar parameters that look like a good match. Two separate
checks stand in the way: one decides before writing whether the passages can
actually answer the question, the other checks afterwards that every sentence
in the answer is backed by a source.

Everything runs locally. Nothing is sent anywhere.

## How well it works

70 hand-written questions, 15 of which have no answer in the docs.

**Finding the right passage** — three retrievers, measured on the same questions:

| | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| BM25 (keyword) | 0.382 | 0.564 | 0.709 | 0.517 |
| Embeddings | 0.491 | 0.636 | 0.782 | **0.613** |
| Both, fused | 0.455 | **0.655** | **0.800** | 0.585 |

**Refusing when it should:**

- caught 13 of the 15 unanswerable questions
- 92% of the sentences it wrote were backed by a source
- but it also refused 11 questions it could have answered

That last number is the cost of being careful, and I'd rather report it than
hide it. Full numbers and the reasoning behind them are in
[`results/`](results/).

## Install (to use it)

You need Python 3.12, about 2GB of free disk, and
[Ollama](https://ollama.com) running.

```bash
git clone https://github.com/hosnahatami04/technical-docs-rag
cd technical-docs-rag

pip install -r requirements.txt
bash data/download.sh          # fetches the docs, pinned to a git tag
make index                     # builds the search index, ~7 min on CPU
```

Pull the model if you don't have it:

```bash
ollama pull qwen2.5:7b-instruct
```

Ask it something:

```bash
make ask
```

Or run the API:

```bash
make serve
```

```bash
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the default value of max_wal_size?"}'
```

The response includes the answer, the citations, and both checks' verdicts —
so you can see why it answered the way it did. Interactive docs are at
`localhost:8000/docs`.

### With Docker

```bash
bash data/download.sh          # still needed — the corpus isn't in the image
docker build -t technical-docs-rag .
docker run -p 8000:8000 \
  -v "$PWD/data:/app/data" \
  --add-host=host.docker.internal:host-gateway \
  technical-docs-rag
```

The embedding model is baked into the image so it starts offline. The corpus
isn't, because it's fetched at a pinned tag and I'd rather the tag stay the
source of truth.

## Install (to work on it)

Same as above, plus:

```bash
make check     # lint, tests, and the quality gate
make test      # just the tests
make eval      # re-run retrieval over all 70 questions, ~1 min
make generate  # re-run the full pipeline, ~45 min
```

`make help` lists everything.

The tests don't need the corpus, the model, or a network — they build their own
fixtures. That's deliberate: CI has none of those.

If you change anything that affects retrieval or generation, re-run `make eval`
and `make generate` and commit the updated `results/`. CI checks those numbers
against floors in `scripts/check_thresholds.py` and fails if they drop.

Some things worth knowing before you dig in:

- `data/raw/` and `data/indexes/` are gitignored. The download script rebuilds
  the first, `make index` rebuilds the second.
- The index is cached by a hash of the chunks and the model name, so changing
  either rebuilds it and you can't accidentally evaluate new chunks against old
  embeddings.
- `num_ctx=4096` in `src/generation/llm.py` isn't arbitrary. At the model's
  default context the weights don't fit in 8GB of VRAM and it runs ~4x slower.

## Contributing

It's a personal project, but issues and PRs are welcome.

If you send one:

- `make check` should pass
- add a test for whatever you changed
- if you touched retrieval or generation, include the regenerated `results/`
- commit messages that say *why*, not just what

I'm particularly interested in anything that reduces the 11 wrong refusals
without losing the 13 correct ones.

## Known issues

**It refuses too often.** 11 of 55 answerable questions get turned away, mostly
procedural ones where the answer is spread across a section rather than stated
in one sentence. The gate prompt is in
`src/generation/answerability_gate.py` and has already been through three
versions.

**The judge is the model being judged.** Both checks use the same model that
writes the answers, so it's grading its own work. A separate model would be a
stronger test.

**Fusing the two retrievers didn't help.** Embeddings alone beat the fusion on
MRR. It does win on Hit@5, which is what matters for the generator, so it's
still the default — but it's not the improvement I expected. The analysis is
in [`results/evaluation.md`](results/evaluation.md).

**One question nothing can find.** "How do I find which queries are currently
running?" — the answer is in `monitoring.sgml` and no retriever surfaces it.

**Slow.** About 15 seconds a question, because each one is three model calls.
That's the cost of the checks being real rather than claimed.

**70 questions isn't many.** Enough to see whether something works, not enough
to tell apart two configurations that differ by one question.

## License and ownership

Personal project, built against the public PostgreSQL documentation. No
employer data or proprietary content involved.
