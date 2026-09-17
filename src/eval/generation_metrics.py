"""Metrics for the generation layer: abstention, answerability F1, groundedness.

Phase 5 measured whether retrieval found the right passage. These measure what
happens after: whether the system knows when it cannot answer, and whether the
answers it does give are supported by the sources.

The distinction the plan cares about is that the two gates are scored
separately. A single "was the output good" number cannot tell you that the
answerability gate caught 13 of 15 unanswerable questions while the grounding
gate caught 4 hallucinated claims that got past it — and that sentence is the
whole point of building two gates instead of one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import mean


@dataclass
class GenerationResult:
    """What the pipeline produced for one question, scored against its label."""

    question_id: str
    category: str
    answerable: bool
    abstained: bool
    abstain_reason: str = ""
    answer: str = ""
    # Which gate stopped it, when one did. Kept distinct so the two can be
    # credited separately.
    stopped_by: str = ""
    answerability_said_yes: bool | None = None
    grounding_score: float | None = None
    n_claims: int = 0
    unsupported_claims: tuple[str, ...] = ()
    cited_paths: tuple[str, ...] = ()
    gold_paths: tuple[str, ...] = ()
    key_claims: tuple[str, ...] = ()
    seconds: float = 0.0

    @property
    def correct_abstention(self) -> bool:
        """Abstained on a question that genuinely has no answer."""
        return self.abstained and not self.answerable

    @property
    def wrong_abstention(self) -> bool:
        """Refused a question the corpus could have answered.

        This is the cost side of abstention, and reporting it matters: a system
        that refuses everything scores a perfect correct-abstention rate.
        """
        return self.abstained and self.answerable

    @property
    def answered_unanswerable(self) -> bool:
        """The failure the gates exist to prevent."""
        return not self.abstained and not self.answerable

    @property
    def cited_correctly(self) -> bool:
        """Cited at least one of the gold documents."""
        return bool(set(self.cited_paths) & set(self.gold_paths))


@dataclass
class ConfusionMatrix:
    """Answerability as a binary classification, with F1.

    Positive = "this question is answerable". Recorded as counts rather than
    only the F1 so a reader can see which way the errors fall: 15 false
    positives and 0 false negatives is a very different system from the reverse,
    at the same F1.
    """

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )
        return (self.true_positive + self.true_negative) / total if total else 0.0


def answerability_confusion(results: Iterable[GenerationResult]) -> ConfusionMatrix:
    """Score the system's answer/abstain decision against the labels."""
    matrix = ConfusionMatrix()
    for result in results:
        predicted_answerable = not result.abstained
        if result.answerable and predicted_answerable:
            matrix.true_positive += 1
        elif result.answerable and not predicted_answerable:
            matrix.false_negative += 1
        elif not result.answerable and predicted_answerable:
            matrix.false_positive += 1
        else:
            matrix.true_negative += 1
    return matrix


def gate_confusion(results: Iterable[GenerationResult]) -> ConfusionMatrix:
    """Score the answerability gate alone, ignoring later stages.

    Separate from `answerability_confusion` because the system can abstain for
    reasons the gate had nothing to do with — the generator declining, or the
    grounding gate rejecting the answer. Crediting the gate for those would
    overstate it.
    """
    matrix = ConfusionMatrix()
    for result in results:
        if result.answerability_said_yes is None:
            continue
        predicted = result.answerability_said_yes
        if result.answerable and predicted:
            matrix.true_positive += 1
        elif result.answerable and not predicted:
            matrix.false_negative += 1
        elif not result.answerable and predicted:
            matrix.false_positive += 1
        else:
            matrix.true_negative += 1
    return matrix


@dataclass
class GenerationSummary:
    """Aggregate numbers for one configuration."""

    label: str
    n_questions: int
    n_answerable: int
    n_unanswerable: int

    correct_abstention_rate: float = 0.0
    wrong_abstention_rate: float = 0.0
    answered_unanswerable: int = 0

    # Groundedness of the answers the system actually returned. With the
    # grounding gate enabled this is near-tautological — every answer below the
    # threshold was converted into an abstention, so the surviving answers
    # cannot score below it. Reported anyway, because it is what a user
    # receives, but `groundedness_pre_gate` is the number that measures the
    # generator.
    groundedness: float = 0.0
    # Groundedness across every answer the model wrote, including the ones the
    # gate then rejected. This is the honest measure of how often the generator
    # invents, and the only one of the two that can move.
    groundedness_pre_gate: float = 0.0
    n_grounded_answers: int = 0
    total_claims: int = 0
    unsupported_claims: int = 0

    answerability_f1: float = 0.0
    gate_f1: float = 0.0
    citation_rate: float = 0.0
    mean_seconds: float = 0.0

    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    gate_only: ConfusionMatrix = field(default_factory=ConfusionMatrix)


def summarize_generation(
    results: Sequence[GenerationResult], label: str = "default"
) -> GenerationSummary:
    """Aggregate one run into the numbers the report quotes."""
    if not results:
        return GenerationSummary(
            label=label, n_questions=0, n_answerable=0, n_unanswerable=0
        )

    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    confusion = answerability_confusion(results)
    gate_only = gate_confusion(results)

    # Two populations, and the difference between them is the point.
    #
    # `answered` is what the user receives. With the gate on, every answer that
    # scored below threshold became an abstention, so this set cannot score
    # below the threshold — 1.000 here is a property of the gate, not evidence
    # the generator is faithful.
    #
    # `all_scored` includes the answers the gate rejected. That is the number
    # that measures the generator, and the only one that can move.
    answered = [r for r in results if not r.abstained and r.grounding_score is not None]
    all_scored = [r for r in results if r.grounding_score is not None]
    grounded_scores = [
        r.grounding_score for r in answered if r.grounding_score is not None
    ]
    pre_gate_scores = [
        r.grounding_score for r in all_scored if r.grounding_score is not None
    ]

    return GenerationSummary(
        label=label,
        n_questions=len(results),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        correct_abstention_rate=(
            mean(r.correct_abstention for r in unanswerable) if unanswerable else 0.0
        ),
        wrong_abstention_rate=(
            mean(r.wrong_abstention for r in answerable) if answerable else 0.0
        ),
        answered_unanswerable=sum(1 for r in results if r.answered_unanswerable),
        groundedness=mean(grounded_scores) if grounded_scores else 0.0,
        groundedness_pre_gate=mean(pre_gate_scores) if pre_gate_scores else 0.0,
        n_grounded_answers=sum(1 for r in answered if (r.grounding_score or 0) >= 0.8),
        # Counted over every scored answer, not just the surviving ones —
        # otherwise the unsupported claims the gate caught vanish from the
        # report that exists to describe them.
        total_claims=sum(r.n_claims for r in all_scored),
        unsupported_claims=sum(len(r.unsupported_claims) for r in all_scored),
        answerability_f1=confusion.f1,
        gate_f1=gate_only.f1,
        citation_rate=(mean(r.cited_correctly for r in answered) if answered else 0.0),
        mean_seconds=mean(r.seconds for r in results),
        confusion=confusion,
        gate_only=gate_only,
    )
