"""Decide, before generating, whether the retrieved context can answer at all.

Phase 4 established why this is needed with a number: asked about
`autovacuum_vacuum_max_threshold`, a parameter that does not exist in
PostgreSQL 17, all three retrievers returned confident results and the dense
retriever scored its top hit 0.86. Retrieval always returns the nearest thing.
It cannot report that nothing was actually near.

So something else has to make that call, and it has to run *before* generation:
once a model has five plausible passages in front of it, writing a plausible
answer is the path of least resistance.

Two implementations, because the plan asks for the comparison:

`ThresholdGate` reads the retrieval score. Free, instant, and entirely
mechanical — it never sees the question.

`LLMGate` asks the model. Slower, but it can notice that a passage about
`autovacuum_vacuum_threshold` does not answer a question about
`autovacuum_vacuum_max_threshold`, which no score can.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.generation.answerer import format_context
from src.generation.llm import LLMError, OllamaClient
from src.retrieval.base import ScoredChunk


@dataclass
class GateVerdict:
    """One gate's decision, with the reason it gave.

    `reason` is not decoration: a gate that refuses without explanation cannot
    be debugged, and the API returns this to the caller.
    """

    answerable: bool
    confidence: float
    reason: str
    gate: str

    def __bool__(self) -> bool:
        return self.answerable


class AnswerabilityGate(ABC):
    """Runs before generation. True means 'the context can support an answer'."""

    name: str = "gate"

    @abstractmethod
    def check(self, question: str, chunks: list[ScoredChunk]) -> GateVerdict: ...


# --- Threshold ---------------------------------------------------------------

# Retrieval scores are not comparable across retrievers: BM25 returns an
# unbounded sum, cosine similarity sits in [-1, 1], RRF returns roughly 1/61 per
# agreeing retriever. So each gets its own threshold.
#
# These are the F1-optimal values, swept over every score in Phase 5's run —
# not guesses. Sweeping them exposed why this gate is the weak baseline:
#
#   retriever  best threshold  F1     always-say-yes F1
#   bm25             9.4856    0.887        0.880
#   dense            0.7306    0.885        0.880
#   hybrid           0.0164    0.880        0.880
#
# The best achievable threshold barely beats answering "yes" to everything, and
# for the hybrid it does not beat it at all. The distributions overlap almost
# entirely: unanswerable questions score a median hybrid top-score of 0.0320
# against 0.0323 for answerable ones.
#
# That is not a tuning failure, it is what the score measures. A question about
# a parameter that does not exist retrieves the real parameters around it, and
# those are genuinely similar — high similarity is the correct answer to
# "how close is the nearest passage", which is a different question from
# "does any passage answer this".
DEFAULT_THRESHOLDS: dict[str, float] = {
    "bm25": 9.4856,
    "dense": 0.7306,
    "hybrid": 0.0164,
}


class ThresholdGate(AnswerabilityGate):
    """Abstain when the best retrieval score falls below a threshold.

    The cheap baseline, and worth measuring precisely because it is cheap: if it
    performs comparably to the LLM gate, the LLM gate is not earning its
    latency.

    Its structural weakness is that it never reads the question. A question
    about a parameter that does not exist still retrieves the real parameters
    around it, and those are genuinely similar — so the score stays high and the
    gate lets it through.
    """

    name = "threshold"

    def __init__(self, threshold: float = 0.75, retriever: str | None = None) -> None:
        self.threshold = (
            DEFAULT_THRESHOLDS.get(retriever, threshold)
            if retriever is not None
            else threshold
        )

    def check(self, question: str, chunks: list[ScoredChunk]) -> GateVerdict:
        if not chunks:
            return GateVerdict(
                answerable=False,
                confidence=1.0,
                reason="No chunks were retrieved.",
                gate=self.name,
            )

        top = chunks[0].score
        passed = top >= self.threshold
        # Distance from the threshold, scaled so the number is readable rather
        # than claiming to be a probability.
        confidence = min(1.0, abs(top - self.threshold) / max(self.threshold, 1e-6))
        return GateVerdict(
            answerable=passed,
            confidence=round(confidence, 3),
            reason=(
                f"Top retrieval score {top:.4f} "
                f"{'meets' if passed else 'falls below'} threshold {self.threshold:.4f}."
            ),
            gate=self.name,
        )


# --- LLM ---------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You judge whether documentation excerpts contain what is needed to answer a "
    "question. The information may be spread across sentences and need rewording; "
    "that still counts. What does not count is a different parameter, a different "
    "feature, or the right topic with the specific fact never stated."
)

# This prompt took three attempts, and both failures are worth recording
# because they fail in opposite directions.
#
# v1 listed abstract rules for when to answer NO ("the sources describe the
# setting but never state its default"). It rejected fact-001, whose top source
# says in plain words "The default is 1 GB." Given rules phrased as patterns,
# the model matched patterns instead of reading.
#
# v2 asked it to quote the sentence that answers the question. That fixed
# fact-001 and scored 8/8 on a hand-picked sample — but over the full run it
# wrongly refused 23 of 55 answerable questions, 15 of them procedural or
# multi-section. Those questions have no single answering sentence: "how do I
# load data from a CSV file" is answered by a section, not a line. Asked for a
# sentence, the model correctly reported there wasn't one, and refused.
#
# v3 asks what actually matters — whether an answer can be *built* from the
# sources — while still requiring evidence be quoted. The lesson is that a
# hand-picked sample confirmed a prompt the full run then falsified.
JUDGE_PROMPT = """\
Sources:
{context}

Question: {question}

Task: decide whether an answer to this question can be written using only the
sources above.

Reply YES if the sources contain the information, even if it is spread across
several sentences and would need rewording. Quote a few words showing where it
is.

Reply NO only if the information is genuinely absent: the sources describe a
different parameter, a different feature, or the right topic without ever
stating what was asked.

Format:
YES - <a few words from the sources>
NO - <what the sources discuss instead>

Reply with one line only."""

_VERDICT = re.compile(r"^\s*(YES|NO)\b", re.IGNORECASE)


class LLMGate(AnswerabilityGate):
    """Ask the model whether the context answers the question."""

    name = "llm"

    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient()

    def check(self, question: str, chunks: list[ScoredChunk]) -> GateVerdict:
        if not chunks:
            return GateVerdict(
                answerable=False,
                confidence=1.0,
                reason="No chunks were retrieved.",
                gate=self.name,
            )

        prompt = JUDGE_PROMPT.format(
            context=format_context(chunks, max_chars=1200), question=question
        )
        try:
            completion = self.client.generate(
                prompt, system=JUDGE_SYSTEM, max_tokens=80
            )
        except LLMError as exc:
            # A gate that cannot run must not silently approve. Failing closed
            # costs recall; failing open costs exactly the hallucinations the
            # gate exists to prevent.
            return GateVerdict(
                answerable=False,
                confidence=0.0,
                reason=f"Gate could not run: {exc}",
                gate=self.name,
            )

        text = completion.text.strip()
        match = _VERDICT.match(text)
        if match is None:
            # No parseable verdict is not a yes.
            return GateVerdict(
                answerable=False,
                confidence=0.0,
                reason=f"Unparseable verdict: {text[:120]!r}",
                gate=self.name,
            )

        answerable = match.group(1).upper() == "YES"
        reason = text[match.end() :].strip(" .:-") or text
        return GateVerdict(
            answerable=answerable,
            confidence=1.0,
            reason=reason[:200],
            gate=self.name,
        )
