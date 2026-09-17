# Generation and gate evaluation

70 questions through the full pipeline: retrieve, answerability gate, generate, grounding gate. 55 are answerable and 15 are not. Raw record: `results/generation_raw.json`.

## How to read these numbers

**Correct abstention** — of the questions the corpus genuinely cannot answer, how many did the system refuse? This is the headline number, and on its own it is misleading: a system that refuses everything scores 1.000.

**Wrong abstention** — of the answerable questions, how many did it refuse anyway? This is what stops the first number from being gamed. The two are read together.

**Groundedness** — the fraction of sentences in a generated answer that the cited sources actually support. Reported twice, and the difference matters:

- *pre-gate* averages over every answer the model wrote, including the ones the grounding gate then rejected. This measures the generator.
- *delivered* averages over the answers that survived the gate. With the gate enabled this is near-tautological — anything below the 0.8 threshold was converted into an abstention, so the survivors cannot score below it. A value of 1.000 here is a property of the gate, not evidence the model is faithful.

Reporting only the second number would be the mistake this project exists to avoid: it looks like a perfect score and is guaranteed by construction.

**Answerability F1** — treating 'this question is answerable' as a binary classification. F1 balances refusing too much against refusing too little.

With 15 unanswerable questions, one question is worth 0.067. This is a coarse instrument and differences of one or two questions should not be read as results.

## What each gate contributes

Each configuration changes one thing, so the effect of each gate can be attributed rather than assumed.

| Configuration | Correct abstention | Wrong abstention | Answered unanswerable | Groundedness (pre-gate) | Groundedness (delivered) | Answerability F1 |
|---|---|---|---|---|---|---|
| Both gates | 0.867 | 0.200 | 2 / 15 | 0.921 | 1.000 | 0.871 |
| No gates | 0.800 | 0.000 | 3 / 15 | — | — | 0.973 |
| Answerability gate only | 0.867 | 0.073 | 2 / 15 | — | — | 0.944 |
| Threshold gate (instead of LLM) | 0.867 | 0.182 | 2 / 15 | 0.889 | 1.000 | 0.882 |

Without gates, the system correctly refuses 0.800 of unanswerable questions; with both gates, 0.867 — a difference of +0.067, about 1 questions.

The ungated number is not zero because the generator sometimes declines on its own when the sources plainly do not answer. That is not a substitute for a gate: it is unmeasured, unswitchable, and cannot be reported separately.

## Which gate stopped what

The reason the gates are separate modules. A single quality check could report that the system abstained, but not whether it abstained because nothing could answer the question or because the answer it wrote was unsupported.

| Stopped by | Total | Correctly (unanswerable) | Wrongly (answerable) |
|---|---|---|---|
| answerability | 17 | 13 | 4 |
| grounding | 7 | 0 | 7 |

In the plan's terms: the answerability gate catches 13 of 15 unanswerable questions, and the grounding gate catches 7 answers whose claims the sources did not support.

## Groundedness

- 96 claims checked across every answer written
- 13 were not supported by the cited sources
- groundedness before the gate: 0.921 — what the generator produced
- groundedness of delivered answers: 1.000 — what a user receives, bounded below by the gate's threshold
- 46 answers passed the 0.8 threshold and were returned

Claims the gate rejected:

- `proc-002` — Alternatively, you can take a base backup using the low-level API by executing the `SELECT pg_backup_start(label => 'label', fast => false);` command as a user with appropriate permissions.
- `proc-003` — To configure continuous WAL archiving, you need to set `archive_mode` to `on` or `always`, and define an `archive_command` that specifies how to archive WAL files.
- `proc-007` — To load data from a CSV file into a table using the COPY command, you first ensure that the file_fdw extension is installed and create a foreign server pointing to your CSV file.
- `proc-007` — Then, use the CREATE FOREIGN TABLE command to define the table structure and specify the CSV file name and format.
- `proc-007` — Finally, execute the COPY command with the appropriate options to import the data.

## By question category

| Category | n | Abstained | Groundedness |
|---|---|---|---|
| conceptual | 10 | 0 | 1.000 |
| factual | 20 | 2 | 1.000 |
| multi_section | 10 | 3 | 1.000 |
| procedural | 15 | 6 | 1.000 |
| unanswerable | 15 | 13 | 1.000 |

## What these numbers do not show

- **The judge is the model being judged.** Both gates use the same model that writes the answers. A claim it believes it can support, it will call supported. An independent judge would be a stronger test.
- **Groundedness is not correctness.** A sentence faithfully copied from a source the retriever should not have returned scores 1.000.
- **Sample size.** 15 unanswerable questions is enough to see whether abstention works at all, not enough to distinguish two configurations that differ by one question.

