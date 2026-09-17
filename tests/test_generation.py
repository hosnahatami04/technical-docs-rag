"""Tests for the generation layer and both gates.

Nothing here calls Ollama. The model is replaced by a fake that returns scripted
responses, because what needs testing is the wiring — prompt construction,
verdict parsing, the abstention path, the metric arithmetic — not whether
qwen2.5 is a good judge. That question is answered by the evaluation run, not
by a unit test.

It also keeps the suite runnable in CI, where there is no GPU and no model.
"""

from __future__ import annotations

import pytest

from src.eval.generation_metrics import (
    GenerationResult,
    answerability_confusion,
    gate_confusion,
    summarize_generation,
)
from src.generation.answerability_gate import LLMGate, ThresholdGate
from src.generation.answerer import (
    Answer,
    Answerer,
    format_context,
    parse_citations,
)
from src.generation.grounding_gate import GroundingGate, split_claims
from src.generation.llm import Completion, LLMError
from src.generation.pipeline import ABSTENTION_TEXT, RAGPipeline
from src.ingestion.chunker import Chunk
from src.retrieval.base import Retriever, ScoredChunk


class FakeClient:
    """Returns scripted replies, in order, and records what it was asked."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.prompts: list[str] = []
        self.model = "fake"
        self.num_ctx = 4096

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, **_: object) -> Completion:
        self.prompts.append(prompt)
        text = self.replies.pop(0) if self.replies else "OK"
        return Completion(
            text=text,
            prompt_tokens=10,
            output_tokens=10,
            eval_seconds=0.1,
            total_seconds=0.1,
        )


class BrokenClient(FakeClient):
    def generate(self, prompt: str, **_: object) -> Completion:
        raise LLMError("ollama is down")


def make_result(
    index: int, *, body: str = "The default is 1 GB.", score: float = 0.9
) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"c{index}",
        doc_path="config.sgml",
        doc_title="Server Configuration",
        doc_type="config",
        heading_path=("Server Configuration", "Write Ahead Log", "Checkpoints"),
        body=body,
        chunk_index=index,
        token_count=20,
        has_code=False,
        has_table=False,
    )
    return ScoredChunk(chunk=chunk, score=score, rank=index)


@pytest.fixture
def results() -> list[ScoredChunk]:
    return [make_result(i, score=1.0 - i * 0.1) for i in range(1, 4)]


class FakeRetriever(Retriever):
    name = "fake"

    def __init__(self, results: list[ScoredChunk]) -> None:
        self._results = results

    def __len__(self) -> int:
        return len(self._results)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        return self._results[:k]


# --- Context formatting -----------------------------------------------------


def test_context_numbers_sources_from_one(results: list[ScoredChunk]) -> None:
    # Citations refer to these numbers, and a model told [0] would cite [0].
    context = format_context(results)
    assert context.startswith("[1]")
    assert "[2]" in context


def test_context_includes_the_heading_path(results: list[ScoredChunk]) -> None:
    # Often the only place the answer's subject is named: a body may say "the
    # default is 128 megabytes" while only the heading says shared_buffers.
    assert "Write Ahead Log" in format_context(results)


def test_context_truncates_an_oversized_chunk() -> None:
    # A single 3,940-token table would otherwise consume the window and push
    # the other four sources out.
    huge = make_result(1, body="x" * 5000)
    context = format_context([huge], max_chars=100)
    assert "[truncated]" in context
    assert len(context) < 400


# --- Citation parsing -------------------------------------------------------


def test_parses_citations_in_order() -> None:
    assert parse_citations("Foo [2]. Bar [1].", 3) == (2, 1)


def test_deduplicates_repeated_citations() -> None:
    assert parse_citations("Foo [1]. Bar [1]. Baz [2].", 3) == (1, 2)


def test_drops_out_of_range_citations() -> None:
    # A model citing [7] when five sources were given has invented the
    # citation; treating it as real would let an unsupported claim look sourced.
    assert parse_citations("Foo [7]. Bar [1].", 5) == (1,)


def test_no_citations_yields_empty() -> None:
    assert parse_citations("Just prose.", 5) == ()


# --- Answerer ---------------------------------------------------------------


def test_answerer_returns_the_model_text(results: list[ScoredChunk]) -> None:
    answerer = Answerer(FakeClient(["The default is 1 GB [1]."]))
    answer = answerer.answer("What is it?", results)
    assert answer.text == "The default is 1 GB [1]."
    assert answer.cited_indexes == (1,)
    assert answer.refused is False


def test_answerer_recognizes_the_insufficient_marker(
    results: list[ScoredChunk],
) -> None:
    answerer = Answerer(FakeClient(["INSUFFICIENT"]))
    answer = answerer.answer("What is it?", results)
    assert answer.refused is True
    assert answer.cited_indexes == ()
    assert "don't have enough information" in answer.text


def test_answerer_abstains_when_nothing_was_retrieved() -> None:
    client = FakeClient(["should not be called"])
    answer = Answerer(client).answer("What is it?", [])
    assert answer.refused is True
    assert client.prompts == []  # the model was never asked


def test_cited_chunks_resolve_to_the_right_sources(
    results: list[ScoredChunk],
) -> None:
    answer = Answer(question="q", text="Foo [2].", chunks=results, cited_indexes=(2,))
    assert answer.cited_chunks == [results[1]]


def test_cited_chunks_ignore_out_of_range_indexes(
    results: list[ScoredChunk],
) -> None:
    answer = Answer(question="q", text="", chunks=results, cited_indexes=(99,))
    assert answer.cited_chunks == []


# --- Threshold gate ---------------------------------------------------------


def test_threshold_gate_passes_a_high_score(results: list[ScoredChunk]) -> None:
    verdict = ThresholdGate(threshold=0.5).check("q", results)
    assert verdict.answerable is True
    assert "0.5" in verdict.reason or "meets" in verdict.reason


def test_threshold_gate_rejects_a_low_score(results: list[ScoredChunk]) -> None:
    assert ThresholdGate(threshold=0.99).check("q", results).answerable is False


def test_threshold_gate_rejects_empty_retrieval() -> None:
    assert ThresholdGate().check("q", []).answerable is False


def test_threshold_gate_uses_a_per_retriever_default() -> None:
    # BM25 scores are unbounded sums; cosine sits in [-1, 1]. One threshold
    # cannot serve both.
    bm25 = ThresholdGate(retriever="bm25")
    dense = ThresholdGate(retriever="dense")
    assert bm25.threshold > dense.threshold


def test_threshold_gate_never_reads_the_question(
    results: list[ScoredChunk],
) -> None:
    # Its structural weakness, asserted so the limitation is documented in the
    # suite rather than only in prose.
    gate = ThresholdGate(threshold=0.5)
    a = gate.check("What is the default value of max_wal_size?", results)
    b = gate.check("What is the airspeed velocity of a swallow?", results)
    assert a.answerable == b.answerable


# --- LLM gate ---------------------------------------------------------------


def test_llm_gate_reads_a_yes(results: list[ScoredChunk]) -> None:
    gate = LLMGate(FakeClient(['YES - "The default is 1 GB."']))
    verdict = gate.check("What is it?", results)
    assert verdict.answerable is True
    assert "1 GB" in verdict.reason


def test_llm_gate_reads_a_no(results: list[ScoredChunk]) -> None:
    gate = LLMGate(FakeClient(["NO - the sources discuss a different parameter"]))
    assert gate.check("What is it?", results).answerable is False


def test_llm_gate_asks_for_quoted_evidence(results: list[ScoredChunk]) -> None:
    # Requiring evidence is what stopped the gate judging abstractly and
    # rejecting a question whose source plainly stated the answer.
    client = FakeClient(["YES - quoted"])
    LLMGate(client).check("What is it?", results)
    assert "quote" in client.prompts[0].lower()


def test_llm_gate_allows_an_answer_spread_across_sentences(
    results: list[ScoredChunk],
) -> None:
    # The correction the full run forced. An earlier prompt asked for the one
    # sentence that answers the question, which wrongly refused 23 of 55
    # answerable questions — "how do I load data from a CSV file" is answered
    # by a section, not a line, so the model correctly reported there was no
    # such sentence and refused.
    client = FakeClient(["YES - evidence"])
    LLMGate(client).check("How do I load a CSV file?", results)
    prompt = client.prompts[0].lower()
    assert "spread across" in prompt
    assert "rewording" in prompt


def test_llm_gate_fails_closed_when_the_model_is_down(
    results: list[ScoredChunk],
) -> None:
    # A gate that cannot run must not approve. Failing open costs exactly the
    # hallucinations the gate exists to prevent.
    verdict = LLMGate(BrokenClient()).check("q", results)
    assert verdict.answerable is False
    assert "could not run" in verdict.reason


def test_llm_gate_rejects_an_unparseable_verdict(
    results: list[ScoredChunk],
) -> None:
    verdict = LLMGate(FakeClient(["I think maybe possibly"])).check("q", results)
    assert verdict.answerable is False
    assert "Unparseable" in verdict.reason


# --- Claim splitting --------------------------------------------------------


def test_splits_on_sentence_boundaries() -> None:
    claims = split_claims(
        "The default value is 1 GB. Increasing it lengthens crash recovery."
    )
    assert len(claims) == 2


def test_citation_markers_are_stripped_from_claims() -> None:
    # "[1]" is formatting, not a claim; asking the model to verify it is noise.
    claims = split_claims("The default value of max_wal_size is 1 GB [1].")
    assert "[1]" not in claims[0]


def test_does_not_split_on_abbreviations_or_identifiers() -> None:
    # This corpus is full of "postgresql.conf" and "e.g." — splitting on every
    # period turns one claim into unsupported fragments.
    claims = split_claims(
        "Set the value in postgresql.conf to change how the server behaves."
    )
    assert len(claims) == 1


def test_short_fragments_are_not_claims() -> None:
    assert split_claims("Yes. OK.") == []


def test_a_short_but_complete_claim_is_still_checked() -> None:
    # "The default is 1 GB." is 20 characters and is exactly the kind of
    # sentence this gate exists to verify. An earlier 25-character minimum
    # dropped it, which would let a short hallucination through unchecked.
    assert split_claims("The default is 1 GB.") == ["The default is 1 GB."]


def test_empty_text_yields_no_claims() -> None:
    assert split_claims("") == []


# --- Grounding gate ---------------------------------------------------------


def test_grounding_passes_a_supported_answer(results: list[ScoredChunk]) -> None:
    gate = GroundingGate(FakeClient(["SUPPORTED"]), threshold=0.8)
    answer = Answer(
        question="q",
        text="The default value of max_wal_size is 1 GB.",
        chunks=results,
        cited_indexes=(1,),
    )
    verdict = gate.check(answer)
    assert verdict.grounded is True
    assert verdict.score == 1.0


def test_grounding_catches_an_invented_claim(results: list[ScoredChunk]) -> None:
    # The real failure this gate exists for: the first call made to this model
    # in the project claimed VACUUM "removes deadlocks", which it does not.
    gate = GroundingGate(FakeClient(["SUPPORTED", "NOT_SUPPORTED"]), threshold=0.8)
    answer = Answer(
        question="q",
        text=(
            "The default value of max_wal_size is 1 GB. "
            "It also automatically removes deadlocks from the database."
        ),
        chunks=results,
        cited_indexes=(1,),
    )
    verdict = gate.check(answer)
    assert verdict.grounded is False
    assert verdict.score == 0.5
    assert len(verdict.unsupported) == 1
    assert "deadlocks" in verdict.unsupported[0]


def test_not_supported_is_not_read_as_supported(
    results: list[ScoredChunk],
) -> None:
    # NOT_SUPPORTED contains SUPPORTED as a substring; a naive check inverts
    # every verdict.
    gate = GroundingGate(FakeClient(["NOT_SUPPORTED"]))
    answer = Answer(
        question="q",
        text="A claim long enough to be checked properly.",
        chunks=results,
        cited_indexes=(1,),
    )
    assert gate.check(answer).score == 0.0


def test_grounding_treats_a_refusal_as_vacuously_grounded(
    results: list[ScoredChunk],
) -> None:
    # A refusal makes no claims. Scoring it zero would punish correct behavior.
    gate = GroundingGate(FakeClient([]))
    answer = Answer(question="q", text=ABSTENTION_TEXT, chunks=results, refused=True)
    verdict = gate.check(answer)
    assert verdict.grounded is True
    assert verdict.n_claims == 0


def test_grounding_checks_against_cited_sources_only(
    results: list[ScoredChunk],
) -> None:
    # An answer citing [1] should not borrow support from an uncited [3],
    # otherwise the citation means nothing.
    client = FakeClient(["SUPPORTED"])
    gate = GroundingGate(client)
    answer = Answer(
        question="q",
        text="A claim long enough to be checked by the gate.",
        chunks=[
            make_result(1, body="CITED SOURCE TEXT"),
            make_result(2, body="UNCITED SOURCE TEXT"),
        ],
        cited_indexes=(1,),
    )
    gate.check(answer)
    assert "CITED SOURCE TEXT" in client.prompts[0]
    assert "UNCITED SOURCE TEXT" not in client.prompts[0]


def test_grounding_fails_closed_when_the_model_is_down(
    results: list[ScoredChunk],
) -> None:
    gate = GroundingGate(BrokenClient())
    answer = Answer(
        question="q",
        text="A claim long enough to be checked properly.",
        chunks=results,
        cited_indexes=(1,),
    )
    assert gate.check(answer).grounded is False


# --- Pipeline ---------------------------------------------------------------


def test_pipeline_answers_when_both_gates_pass(
    results: list[ScoredChunk],
) -> None:
    client = FakeClient(
        [
            'YES - "The default is 1 GB."',  # answerability
            "The default is 1 GB [1].",  # generation
            "SUPPORTED",  # grounding
        ]
    )
    pipeline = RAGPipeline(FakeRetriever(results), client=client)
    result = pipeline.ask("What is it?")
    assert result.abstained is False
    assert result.answer == "The default is 1 GB [1]."
    assert result.citations == (1,)


def test_pipeline_abstains_when_the_first_gate_fails(
    results: list[ScoredChunk],
) -> None:
    client = FakeClient(["NO - unrelated sources"])
    pipeline = RAGPipeline(FakeRetriever(results), client=client)
    result = pipeline.ask("What is it?")
    assert result.abstained is True
    assert result.abstain_reason.startswith("answerability gate")
    # Generation never ran: only the gate consumed a prompt.
    assert len(client.prompts) == 1


def test_pipeline_abstains_when_grounding_fails(
    results: list[ScoredChunk],
) -> None:
    client = FakeClient(
        [
            'YES - "quoted"',
            "The default is 1 GB. It also removes deadlocks from the database.",
            "SUPPORTED",
            "NOT_SUPPORTED",
        ]
    )
    pipeline = RAGPipeline(FakeRetriever(results), client=client)
    result = pipeline.ask("What is it?")
    assert result.abstained is True
    assert result.abstain_reason.startswith("grounding gate")


def test_abstention_still_returns_what_was_retrieved(
    results: list[ScoredChunk],
) -> None:
    # The plan is explicit: a user who disagrees with a refusal can look at the
    # sources and judge for themselves.
    client = FakeClient(["NO - unrelated"])
    pipeline = RAGPipeline(FakeRetriever(results), client=client)
    result = pipeline.ask("q")
    assert result.abstained is True
    assert len(result.chunks) == 3


def test_gates_are_independently_switchable(
    results: list[ScoredChunk],
) -> None:
    # Without this, the contribution of each gate cannot be attributed.
    client = FakeClient(["The default is 1 GB [1]."])
    pipeline = RAGPipeline(
        FakeRetriever(results),
        client=client,
        enable_answerability=False,
        enable_grounding=False,
    )
    result = pipeline.ask("q")
    assert result.abstained is False
    assert len(client.prompts) == 1  # generation only


def test_pipeline_abstains_when_retrieval_is_empty() -> None:
    client = FakeClient([])
    pipeline = RAGPipeline(FakeRetriever([]), client=client)
    result = pipeline.ask("q")
    assert result.abstained is True
    assert client.prompts == []


def test_a_failing_gate_verdict_is_still_recorded(
    results: list[ScoredChunk],
) -> None:
    # A bug that cost a full evaluation run. GateVerdict and GroundingVerdict
    # define __bool__ to return their decision, so `if result.grounding` is
    # False exactly when the gate rejected the answer — and every consumer
    # written with a truthiness check silently dropped the verdict of every
    # rejection, which is the only case anyone wants to look at.
    client = FakeClient(
        [
            'YES - "quoted"',
            "The default is 1 GB. It also removes deadlocks from the database.",
            "SUPPORTED",
            "NOT_SUPPORTED",
        ]
    )
    result = RAGPipeline(FakeRetriever(results), client=client).ask("q")

    assert result.abstained is True
    assert result.grounding is not None
    assert bool(result.grounding) is False  # the trap
    assert result.grounding.score == 0.5

    payload = result.to_dict()
    assert payload["grounding_gate"] is not None
    assert payload["grounding_gate"]["grounded"] is False
    assert payload["grounding_gate"]["unsupported"]


def test_a_failing_answerability_verdict_is_still_recorded(
    results: list[ScoredChunk],
) -> None:
    result = RAGPipeline(
        FakeRetriever(results), client=FakeClient(["NO - unrelated sources"])
    ).ask("q")
    assert result.answerability is not None
    assert bool(result.answerability) is False
    assert result.to_dict()["answerability_gate"]["answerable"] is False


def test_result_serializes_both_gate_verdicts(
    results: list[ScoredChunk],
) -> None:
    # The API returns these; the plan calls them what makes it interesting.
    client = FakeClient(['YES - "q"', "The default is 1 GB [1].", "SUPPORTED"])
    result = RAGPipeline(FakeRetriever(results), client=client).ask("q")
    payload = result.to_dict()
    assert payload["answerability_gate"]["answerable"] is True
    assert payload["grounding_gate"]["grounded"] is True
    assert payload["citations"][0]["doc_path"] == "config.sgml"


# --- Metrics ----------------------------------------------------------------


def make_gr(
    qid: str, *, answerable: bool, abstained: bool, **kwargs: object
) -> GenerationResult:
    return GenerationResult(
        question_id=qid,
        category="factual",
        answerable=answerable,
        abstained=abstained,
        **kwargs,  # type: ignore[arg-type]
    )


def test_correct_abstention_requires_an_unanswerable_question() -> None:
    assert make_gr("a", answerable=False, abstained=True).correct_abstention
    assert not make_gr("b", answerable=True, abstained=True).correct_abstention


def test_wrong_abstention_is_tracked_separately() -> None:
    # A system that refuses everything has a perfect correct-abstention rate;
    # this is the number that exposes it.
    assert make_gr("a", answerable=True, abstained=True).wrong_abstention


def test_answered_unanswerable_is_the_failure_the_gates_prevent() -> None:
    assert make_gr("a", answerable=False, abstained=False).answered_unanswerable


def test_confusion_matrix_counts_all_four_cells() -> None:
    matrix = answerability_confusion(
        [
            make_gr("a", answerable=True, abstained=False),  # TP
            make_gr("b", answerable=True, abstained=True),  # FN
            make_gr("c", answerable=False, abstained=False),  # FP
            make_gr("d", answerable=False, abstained=True),  # TN
        ]
    )
    assert (matrix.true_positive, matrix.false_negative) == (1, 1)
    assert (matrix.false_positive, matrix.true_negative) == (1, 1)
    assert matrix.precision == 0.5
    assert matrix.recall == 0.5
    assert matrix.f1 == pytest.approx(0.5)


def test_perfect_classification_scores_f1_of_one() -> None:
    matrix = answerability_confusion(
        [
            make_gr("a", answerable=True, abstained=False),
            make_gr("b", answerable=False, abstained=True),
        ]
    )
    assert matrix.f1 == pytest.approx(1.0)


def test_gate_confusion_scores_the_gate_not_the_system() -> None:
    # The system can abstain for reasons the gate had nothing to do with.
    # Crediting the gate for those would overstate it.
    results = [
        make_gr(
            "a",
            answerable=True,
            abstained=True,
            answerability_said_yes=True,
            stopped_by="grounding",
        ),
    ]
    assert answerability_confusion(results).false_negative == 1
    assert gate_confusion(results).true_positive == 1


def test_summary_excludes_abstentions_from_groundedness() -> None:
    # An abstention makes no claims and would score 1.0, rewarding refusal.
    summary = summarize_generation(
        [
            make_gr(
                "a", answerable=True, abstained=False, grounding_score=0.5, n_claims=2
            ),
            make_gr("b", answerable=False, abstained=True),
        ]
    )
    assert summary.groundedness == pytest.approx(0.5)


def test_summary_handles_an_empty_run() -> None:
    summary = summarize_generation([])
    assert summary.n_questions == 0
    assert summary.groundedness == 0.0
