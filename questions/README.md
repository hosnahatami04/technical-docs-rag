# How these questions were designed

70 hand-labeled questions over the PostgreSQL 17.0 documentation, written
before any retriever existed.

That ordering is deliberate. An evaluation written after the system is built
tends to be an evaluation the system passes: you reach for questions you have
already seen work, and you stop when the numbers look reasonable. Writing the
questions first means the retrievers are measured against a target that was
fixed before anyone knew which way the numbers would fall.

## Category balance

| Category | Count | What it tests |
|---|---|---|
| `factual` | 20 | Exact lookup — a default value, a flag's effect |
| `procedural` | 15 | A sequence of steps spread across a section |
| `conceptual` | 10 | Explanation rather than lookup |
| `multi_section` | 10 | Combining two distant parts of the docs |
| `unanswerable` | 15 | Questions the corpus genuinely does not answer |
| **Total** | **70** | 55 answerable + 15 not |

The weighting favours factual lookup because that is where the retrievers are
expected to disagree most. BM25 matches `max_wal_size` literally; a dense
retriever matches the *meaning* of a question that never spells the identifier
out. Twenty questions of that shape give the comparison enough resolution to
show a real difference rather than noise.

Conceptual questions are the smallest answerable group at ten. They are the
hardest to label objectively — "what is MVCC" has no single correct passage —
so a larger set would add variance without adding signal.

## Label format

```json
{
  "id": "fact-001",
  "question": "What is the default value of max_wal_size in PostgreSQL?",
  "category": "factual",
  "answerable": true,
  "gold_doc_paths": ["config.sgml"],
  "gold_headings": ["Server Configuration > Write Ahead Log > Checkpoints"],
  "gold_answer": "The default value of max_wal_size is 1 GB.",
  "key_claims": [
    "max_wal_size defaults to 1 GB",
    "max_wal_size sets the maximum size the WAL is allowed to grow between automatic checkpoints"
  ]
}
```

`gold_headings` is not in the original plan, and it is here because of
something the corpus survey turned up: **all 374 configuration parameters live
in a single 54,000-token `config.sgml`.** Twenty-three of the 55 answerable
questions — 42% — point at that one file. Scored at file level, every one of
them is a hit as soon as any chunk from `config.sgml` appears in the results,
which is close to free.

Scoring the heading path as well separates two different things:

- **file-level Hit@k** — did it find the right document? (comparable to how
  most RAG evaluations report, so the numbers mean something to an outside
  reader)
- **section-level Hit@k** — did it find the right *passage*? (the number that
  actually reflects whether retrieval works)

Phase 5 reports both. Where they diverge is itself a finding.

`key_claims` are the facts a correct answer must contain. Phase 6 scores
groundedness against them. They have to be written now, while the source
passage is open — reconstructing them later means re-reading 55 sections, and
in practice means writing claims that match whatever the system happened to
produce.

## Writing the unanswerable ones

This is the part that took longest. *"What is the airspeed velocity of a
swallow?"* tests nothing: any retriever returns garbage, any generator refuses,
and the abstention score is free. A useful unanswerable question has to look
exactly like an answerable one.

Four patterns, all verified against the corpus rather than assumed:

**1. Real features from a later release** (9 questions)
`io_method`, `io_workers`, `io_uring` support, `uuidv7`, OAuth authentication,
`WITHOUT OVERLAPS`, `autovacuum_vacuum_max_threshold`,
`vacuum_max_eager_scan_fail_rate`, `enable_self_join_removal` — all genuine
PostgreSQL 18 additions, all absent from 17.0. These are the strongest tests because the surrounding topic is
documented in depth, so retrieval returns something confident and plausible.

**2. Invented names that sound real** (2 questions)
`checkpoint_spread_factor` is designed to sit next to the real
`checkpoint_completion_target`, which does control the spreading behaviour the
invented name implies. `log_connections_verbose` is phrased as though the
parameter is known to exist.

**3. A concept described but not implemented** (1 question)
PostgreSQL 17.0 explains what virtual generated columns are, then states it
implements only stored ones. Asking for the *syntax* has no answer, but the
conceptual passage is a strong distractor. Deliberately paired with
`fact-015`, which asks the answerable version of the same topic.

**4. A documented topic with an undocumented specific** (3 questions)
"Which version first supported parallel query?" — parallel query is covered
thoroughly; the introducing release is not stated. "What is the maximum number
of rows a table can hold?" — the limits appendix gives table size, column
count and field size, and lists row count as unlimited. "How do I configure the
number of parallel workers used by pg_dump?" presupposes a server setting that
does not exist, while `pg_dump` itself is documented at length. These test
whether abstention depends on topical similarity or on actually checking the
claim.

Every unanswerable question records an `unanswerable_reason`, and where the
claim is a term never appearing, an `absent_terms` list that
`verify_questions.py` checks against the whole corpus. A label nobody verifies
is an assertion, and every abstention metric would be built on it.

Eleven of the fifteen carry `absent_terms`. The remaining four — patterns 3
and 4 — are unanswerable for a reason no string search can confirm, because
the subject is discussed but the specific fact is never stated. Those are
reported as unverifiable rather than passed silently.

## Verification

```bash
python -m src.eval.verify_questions
```

Checks that every gold path names a real document, every gold heading names a
real section (all 66 heading entries match exactly, not by prefix), every
answerable question's
gold documents actually discuss its subject, and every declared-absent term is
genuinely absent.

Three labeling errors were caught this way rather than by review:

- `indexes.sgml` did not exist — the file is `indices.sgml`
- `REINDEX > Parameters > Rebuilding Indexes Concurrently` was wrong; the
  section is under `Notes`
- a `multi_section` question citing two distant headings within `config.sgml`
  was rejected by a validator rule that assumed multi-section meant
  multi-*file*. The rule was wrong, not the question.

## Distribution of gold documents

24 distinct documents across 55 answerable questions, and 66 heading entries
naming 48 distinct sections. The concentration is real and worth stating plainly:

| Document | Questions | Share |
|---|---|---|
| `config.sgml` | 23 | 42% |
| `high-availability.sgml` | 4 | 7% |
| `backup.sgml` | 4 | 7% |
| everything else | ≤3 each | — |

This mirrors the corpus, where configuration is genuinely one enormous
document. It is not a flaw in the question set so much as a property of the
material — and the reason section-level scoring exists.

## What this set cannot measure

- **Ranking beyond the gold passage.** A retriever that returns the right
  chunk at rank 1 and nonsense at ranks 2–5 scores the same as one that returns
  five relevant chunks. MRR partly covers this; nothing here covers precision.
- **Answer quality beyond the key claims.** A fluent answer containing all the
  key claims scores full marks even if it is badly written.
- **Questions nobody thought to ask.** 70 hand-written questions reflect the
  author's model of what users ask. Real query logs would look different.
