"""Check, after generating, whether every claim in the answer is supported.

The answerability gate asks whether an answer was possible. This asks whether
the answer that came out is actually in the sources — a different failure, and
one that survives the first gate. An answer can be built from genuinely
relevant passages and still contain a sentence the passages never said.

That is not hypothetical here. The first call made to this model in this
project produced:

    "VACUUM in PostgreSQL is used to reclaim space from deleted rows and
     remove deadlocks in the database"

The first half is correct and in the docs. The second half is invented — VACUUM
has nothing to do with deadlocks — and it reads exactly as confidently as the
first. Nothing upstream catches it.

The plan asks for sentence-level claim checking rather than fine-grained
decomposition, which is the right call: a sentence is a unit a reader can check
by eye, and splitting further multiplies model calls for a granularity nobody
consumes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.generation.answerer import Answer
from src.generation.llm import LLMError, OllamaClient
from src.retrieval.base import ScoredChunk

# Sentence splitting that does not break on the abbreviations and identifiers
# this corpus is full of: "e.g.", "128MB.", "postgresql.conf". Splitting on
# every period would turn one claim into three fragments, each unsupported on
# its own.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_CITATION = re.compile(r"\s*\[\d+\]")

# Below this, a sentence is a fragment rather than a claim — "Yes." or "See
# below." Checking those costs a model call and yields noise.
#
# Set at 15 rather than something safer-sounding like 25, because a threshold
# that high drops real claims: "The default is 1 GB." is 20 characters and is
# exactly the kind of sentence this gate exists to verify. A short unchecked
# claim is a hallucination that escapes entirely, which is a worse failure than
# spending one model call on a fragment.
MIN_CLAIM_CHARS = 15

# The fraction of claims that must be supported for the answer to pass. 0.8
# allows one unsupported sentence in five; anything stricter fails answers over
# phrasing rather than substance, since a model rewording a source does not
# always keep its words.
DEFAULT_THRESHOLD = 0.8

VERIFY_SYSTEM = (
    "You check whether a statement is supported by source text. You answer only "
    "SUPPORTED or NOT_SUPPORTED."
)

VERIFY_PROMPT = """\
Source text:
{source}

Statement: {claim}

Is the statement supported by the source text above?

Answer SUPPORTED only if the source states this, or states something that
directly implies it. Answer NOT_SUPPORTED if the source is about the topic but
does not state this, or if the statement adds anything the source does not say.

Reply with exactly one word: SUPPORTED or NOT_SUPPORTED."""

_VERDICT = re.compile(r"\b(NOT_SUPPORTED|SUPPORTED)\b", re.IGNORECASE)


@dataclass
class ClaimCheck:
    """One sentence and whether the sources back it."""

    claim: str
    supported: bool
    reason: str = ""


@dataclass
class GroundingVerdict:
    """The result of checking every claim in one answer."""

    grounded: bool
    score: float
    claims: list[ClaimCheck] = field(default_factory=list)
    threshold: float = DEFAULT_THRESHOLD

    @property
    def unsupported(self) -> list[str]:
        """The sentences that failed — what a reviewer actually wants to see."""
        return [c.claim for c in self.claims if not c.supported]

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    def __bool__(self) -> bool:
        return self.grounded


def split_claims(text: str) -> list[str]:
    """Split an answer into checkable sentences.

    Citation markers are stripped: "[1]" is not part of the claim, and leaving
    it in would ask the model to verify a piece of formatting.
    """
    cleaned = _CITATION.sub("", text).strip()
    if not cleaned:
        return []
    sentences = [s.strip() for s in _SENTENCE_END.split(cleaned)]
    return [s for s in sentences if len(s) >= MIN_CLAIM_CHARS]


class GroundingGate:
    """Verifies each sentence of an answer against the cited sources."""

    name = "grounding"

    def __init__(
        self,
        client: OllamaClient | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        max_source_chars: int = 2400,
    ) -> None:
        self.client = client or OllamaClient()
        self.threshold = threshold
        self.max_source_chars = max_source_chars

    def _source_text(self, chunks: list[ScoredChunk]) -> str:
        parts = [f"{r.heading}\n{r.chunk.body}" for r in chunks]
        joined = "\n\n".join(parts)
        return joined[: self.max_source_chars]

    def verify_claim(self, claim: str, source: str) -> ClaimCheck:
        prompt = VERIFY_PROMPT.format(source=source, claim=claim)
        try:
            completion = self.client.generate(
                prompt, system=VERIFY_SYSTEM, max_tokens=12
            )
        except LLMError as exc:
            # Same reasoning as the answerability gate: a check that cannot run
            # must not pass by default.
            return ClaimCheck(
                claim=claim, supported=False, reason=f"check failed: {exc}"
            )

        match = _VERDICT.search(completion.text)
        if match is None:
            return ClaimCheck(
                claim=claim,
                supported=False,
                reason=f"unparseable verdict: {completion.text[:60]!r}",
            )
        # NOT_SUPPORTED is matched first in the alternation, so a reply
        # containing it is never read as SUPPORTED by substring.
        supported = match.group(1).upper() == "SUPPORTED"
        return ClaimCheck(claim=claim, supported=supported, reason=completion.text[:60])

    def check(self, answer: Answer) -> GroundingVerdict:
        """Score one answer.

        Claims are checked against the chunks the answer cited, falling back to
        everything retrieved when it cited nothing. Checking against cited
        sources is what gives the citation meaning — an answer citing [1] should
        not be able to borrow support from an uncited [4].
        """
        if answer.refused:
            # A refusal makes no claims, so it is vacuously grounded. Scoring it
            # zero would punish the system for behaving correctly.
            return GroundingVerdict(
                grounded=True, score=1.0, claims=[], threshold=self.threshold
            )

        claims = split_claims(answer.text)
        if not claims:
            return GroundingVerdict(
                grounded=True, score=1.0, claims=[], threshold=self.threshold
            )

        chunks = answer.cited_chunks or answer.chunks
        source = self._source_text(chunks)
        checks = [self.verify_claim(claim, source) for claim in claims]

        supported = sum(1 for c in checks if c.supported)
        score = supported / len(checks)
        return GroundingVerdict(
            grounded=score >= self.threshold,
            score=round(score, 3),
            claims=checks,
            threshold=self.threshold,
        )
