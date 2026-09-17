"""The full question-answering path: retrieve, gate, generate, gate.

This is where the plan's "never silently generate" requirement lives. Two
independent gates sit around generation, and either one can stop it:

    question
       ↓  retrieve k chunks
       ↓  answerability gate  ──── fails ──→  abstain, return what was retrieved
       ↓  generate
       ↓  grounding gate      ──── fails ──→  abstain, return what was retrieved
       ↓  answer + citations + both verdicts

Keeping them separate is the point. Merged into one "is this good?" check, you
could report that the system abstained 14 times but not whether it abstained
because nothing could answer the question or because the answer it wrote was
unsupported. Those are different failures with different fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.generation.answerability_gate import (
    AnswerabilityGate,
    GateVerdict,
    LLMGate,
)
from src.generation.answerer import Answer, Answerer
from src.generation.grounding_gate import GroundingGate, GroundingVerdict
from src.generation.llm import OllamaClient
from src.retrieval.base import Retriever, ScoredChunk

ABSTENTION_TEXT = "I don't have enough information to answer this."

# Five chunks to the generator, which is what Phase 5's Hit@5 measures — the
# number was chosen so the retrieval metric and the generation input describe
# the same thing.
DEFAULT_K = 5


@dataclass
class PipelineResult:
    """Everything one question produced, including why it was refused."""

    question: str
    answer: str
    abstained: bool
    abstain_reason: str = ""
    chunks: list[ScoredChunk] = field(default_factory=list)
    citations: tuple[int, ...] = ()
    answerability: GateVerdict | None = None
    grounding: GroundingVerdict | None = None
    generated: Answer | None = field(default=None, repr=False)

    @property
    def cited_paths(self) -> list[str]:
        return [
            self.chunks[i - 1].doc_path
            for i in self.citations
            if 1 <= i <= len(self.chunks)
        ]

    def to_dict(self) -> dict[str, object]:
        """Flat shape for the API and for the evaluation record."""
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "citations": [
                {
                    "index": i,
                    "doc_path": self.chunks[i - 1].doc_path,
                    "heading": self.chunks[i - 1].heading,
                }
                for i in self.citations
                if 1 <= i <= len(self.chunks)
            ],
            "retrieved": [
                {
                    "rank": r.rank,
                    "score": round(r.score, 6),
                    "doc_path": r.doc_path,
                    "heading": r.heading,
                }
                for r in self.chunks
            ],
            "answerability_gate": (
                {
                    "gate": self.answerability.gate,
                    "answerable": self.answerability.answerable,
                    "reason": self.answerability.reason,
                }
                if self.answerability is not None
                else None
            ),
            "grounding_gate": (
                {
                    "grounded": self.grounding.grounded,
                    "score": self.grounding.score,
                    "n_claims": self.grounding.n_claims,
                    "unsupported": self.grounding.unsupported,
                }
                if self.grounding is not None
                else None
            ),
        }


class RAGPipeline:
    """Retrieval, both gates, and generation in between."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        client: OllamaClient | None = None,
        answerability: AnswerabilityGate | None = None,
        grounding: GroundingGate | None = None,
        k: int = DEFAULT_K,
        enable_answerability: bool = True,
        enable_grounding: bool = True,
    ) -> None:
        shared = client or OllamaClient()
        self.retriever = retriever
        self.answerer = Answerer(shared)
        # The plan asks for the gates to be separately switchable, so their
        # individual contribution can be measured rather than assumed.
        self.answerability = answerability or LLMGate(shared)
        self.grounding = grounding or GroundingGate(shared)
        self.k = k
        self.enable_answerability = enable_answerability
        self.enable_grounding = enable_grounding

    def ask(self, question: str) -> PipelineResult:
        chunks = self.retriever.search(question, k=self.k)

        if not chunks:
            return PipelineResult(
                question=question,
                answer=ABSTENTION_TEXT,
                abstained=True,
                abstain_reason="Nothing was retrieved for this question.",
            )

        # --- gate one, before generation --------------------------------
        answerability: GateVerdict | None = None
        if self.enable_answerability:
            answerability = self.answerability.check(question, chunks)
            if not answerability.answerable:
                # The plan is explicit that an abstention still returns what was
                # retrieved: a user who disagrees with the refusal can look at
                # the sources and judge for themselves.
                return PipelineResult(
                    question=question,
                    answer=ABSTENTION_TEXT,
                    abstained=True,
                    abstain_reason=f"answerability gate: {answerability.reason}",
                    chunks=chunks,
                    answerability=answerability,
                )

        generated = self.answerer.answer(question, chunks)

        if generated.refused:
            return PipelineResult(
                question=question,
                answer=ABSTENTION_TEXT,
                abstained=True,
                abstain_reason="generator declined: the sources did not contain the answer",
                chunks=chunks,
                answerability=answerability,
                generated=generated,
            )

        # --- gate two, after generation ---------------------------------
        grounding: GroundingVerdict | None = None
        if self.enable_grounding:
            grounding = self.grounding.check(generated)
            if not grounding.grounded:
                return PipelineResult(
                    question=question,
                    answer=ABSTENTION_TEXT,
                    abstained=True,
                    abstain_reason=(
                        f"grounding gate: only {grounding.score:.0%} of claims were "
                        f"supported by the sources"
                    ),
                    chunks=chunks,
                    answerability=answerability,
                    grounding=grounding,
                    generated=generated,
                )

        return PipelineResult(
            question=question,
            answer=generated.text,
            abstained=False,
            chunks=chunks,
            citations=generated.cited_indexes,
            answerability=answerability,
            grounding=grounding,
            generated=generated,
        )
