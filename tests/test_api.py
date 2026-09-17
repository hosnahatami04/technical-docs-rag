"""Tests for the HTTP layer.

The pipeline is replaced with a fake. What needs testing here is the API's own
behaviour — validation, status codes, the shape of the response, and what
happens when a dependency is missing — not the retrieval and generation that
Phases 4 to 6 already measured.

That substitution is also what makes these tests runnable in CI, where there is
no corpus, no index, and no model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import api
from src.generation.answerability_gate import GateVerdict
from src.generation.grounding_gate import GroundingVerdict
from src.generation.pipeline import ABSTENTION_TEXT, PipelineResult
from src.ingestion.chunker import Chunk
from src.retrieval.base import Retriever, ScoredChunk


def make_chunk(index: int = 1) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"c{index}",
        doc_path="config.sgml",
        doc_title="Server Configuration",
        doc_type="config",
        heading_path=("Server Configuration", "Write Ahead Log", "Checkpoints"),
        body="The default is 1 GB.",
        chunk_index=index,
        token_count=20,
        has_code=False,
        has_table=False,
    )
    return ScoredChunk(chunk=chunk, score=0.9, rank=index)


class FakeRetriever(Retriever):
    name = "fake"

    def __len__(self) -> int:
        return 42

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        return [make_chunk(i) for i in range(1, min(k, 3) + 1)]


class FakePipeline:
    """Returns a scripted PipelineResult and records the questions it saw."""

    def __init__(self, result: PipelineResult | None = None) -> None:
        self.result = result
        self.asked: list[str] = []

    def ask(self, question: str) -> PipelineResult:
        self.asked.append(question)
        if self.result is not None:
            return self.result
        chunks = [make_chunk(1), make_chunk(2)]
        return PipelineResult(
            question=question,
            answer="The default value of max_wal_size is 1 GB [1].",
            abstained=False,
            chunks=chunks,
            citations=(1,),
            answerability=GateVerdict(
                answerable=True,
                confidence=1.0,
                reason='"The default is 1 GB."',
                gate="llm",
            ),
            grounding=GroundingVerdict(grounded=True, score=1.0, claims=[]),
        )


class BrokenPipeline:
    def ask(self, question: str) -> PipelineResult:
        raise RuntimeError("ollama exploded")


class FakeClient:
    def __init__(self, available: bool = True) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakePath:
    """Stands in for CORPUS_ROOT so a test can control whether it exists.

    The corpus is gitignored, so on a runner it never does and on a developer
    machine it usually does. A health check that reads the real filesystem
    would therefore give different answers in the two places.
    """

    def __init__(self, exists: bool) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists

    def __str__(self) -> str:
        return "data/raw/sgml"


@pytest.fixture
def ready_state(monkeypatch: pytest.MonkeyPatch) -> FakePipeline:
    """A service that has finished starting up, with a fake pipeline behind it."""
    pipeline = FakePipeline()
    retriever = FakeRetriever()
    api._state.update(
        {
            "chunks": [make_chunk(i).chunk for i in range(1, 43)],
            "retrievers": {"bm25": retriever, "dense": retriever, "hybrid": retriever},
            "pipelines": {"bm25": pipeline, "dense": pipeline, "hybrid": pipeline},
            "client": FakeClient(available=True),
            "ready": True,
            "error": None,
            "startup_seconds": 1.23,
        }
    )
    # The app's lifespan would rebuild real state over the fake one.
    monkeypatch.setattr(api, "_build_state", lambda: None)

    # health() reads CORPUS_ROOT.exists() from the real filesystem, which is
    # the one thing this fixture cannot fake by assigning to _state. Left
    # alone, these tests pass on a developer machine that has run
    # `bash data/download.sh` and fail in CI, where data/raw/ is gitignored and
    # never present — which is exactly what happened.
    monkeypatch.setattr(api, "CORPUS_ROOT", FakePath(exists=True))
    return pipeline


@pytest.fixture
def client(ready_state: FakePipeline) -> TestClient:
    return TestClient(api.app)


# --- meta --------------------------------------------------------------------


def test_index_lists_the_endpoints(client: TestClient) -> None:
    body = client.get("/").json()
    assert "POST /ask" in body["endpoints"]
    assert "GET /health" in body["endpoints"]


def test_health_reports_ok_when_everything_is_up(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] == 42
    assert body["model_available"] is True
    assert sorted(body["retrievers"]) == ["bm25", "dense", "hybrid"]


def test_health_is_degraded_when_the_model_is_unreachable(
    client: TestClient,
) -> None:
    # Retrieval still works without the model; only generation needs it.
    # Reporting "degraded" rather than "down" says exactly that, and the detail
    # names the command that fixes it.
    api._state["client"] = FakeClient(available=False)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_available"] is False
    assert "ollama serve" in body["detail"]


def test_health_is_degraded_when_the_corpus_is_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The in-memory index keeps serving, but a restart would fail — so this is
    # degraded rather than ok, and the detail has to say why.
    monkeypatch.setattr(api, "CORPUS_ROOT", FakePath(exists=False))
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["corpus_present"] is False
    assert "Corpus missing" in body["detail"]


def test_health_reports_starting_before_indexes_are_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A service that refuses to boot can only be diagnosed from container logs.
    # This one starts anyway so /health can say what is wrong.
    monkeypatch.setattr(api, "_build_state", lambda: None)
    api._state.update(
        {
            "ready": False,
            "error": "Corpus not found",
            "client": None,
            "startup_seconds": 0.4,
            "chunks": None,
            "retrievers": {},
        }
    )
    body = TestClient(api.app).get("/health").json()
    assert body["status"] == "starting"
    assert "Corpus not found" in body["detail"]


# --- /ask: the happy path ----------------------------------------------------


def test_ask_returns_the_answer_and_citations(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "What is max_wal_size?"}).json()
    assert body["abstained"] is False
    assert "1 GB" in body["answer"]
    assert body["citations"][0]["doc_path"] == "config.sgml"
    assert body["citations"][0]["heading"].startswith("Server Configuration")


def test_ask_returns_both_gate_verdicts(client: TestClient) -> None:
    # The plan calls these what make the API interesting: the caller sees why
    # the system answered, not just what it said.
    body = client.post("/ask", json={"question": "What is max_wal_size?"}).json()
    assert body["answerability_gate"]["answerable"] is True
    assert "1 GB" in body["answerability_gate"]["reason"]
    assert body["grounding_gate"]["grounded"] is True


def test_ask_returns_what_was_retrieved(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "What is max_wal_size?"}).json()
    assert len(body["retrieved"]) == 2
    assert body["retrieved"][0]["rank"] == 1
    assert body["retrieved"][0]["doc_path"] == "config.sgml"


def test_ask_reports_which_retriever_ran(client: TestClient) -> None:
    body = client.post(
        "/ask", json={"question": "What is max_wal_size?", "retriever": "bm25"}
    ).json()
    assert body["retriever"] == "bm25"


def test_ask_strips_whitespace_from_the_question(
    client: TestClient, ready_state: FakePipeline
) -> None:
    client.post("/ask", json={"question": "  What is max_wal_size?  "})
    assert ready_state.asked == ["What is max_wal_size?"]


# --- /ask: abstention --------------------------------------------------------


def test_abstention_returns_200_with_the_reason(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A refusal is a successful answer to a question that has none — not an
    # error. Returning 4xx would make callers treat working behaviour as a bug.
    refusal = PipelineResult(
        question="q",
        answer=ABSTENTION_TEXT,
        abstained=True,
        abstain_reason="answerability gate: the sources describe a different parameter",
        chunks=[make_chunk(1)],
        answerability=GateVerdict(
            answerable=False,
            confidence=1.0,
            reason="the sources describe a different parameter",
            gate="llm",
        ),
    )
    api._state["pipelines"]["hybrid"] = FakePipeline(refusal)

    response = client.post("/ask", json={"question": "What is a fake parameter?"})
    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert "answerability gate" in body["abstain_reason"]


def test_abstention_still_returns_the_sources(
    client: TestClient,
) -> None:
    # What lets a caller disagree with a refusal: they can read what was
    # retrieved and judge for themselves.
    refusal = PipelineResult(
        question="q",
        answer=ABSTENTION_TEXT,
        abstained=True,
        abstain_reason="answerability gate: no",
        chunks=[make_chunk(1), make_chunk(2)],
        answerability=GateVerdict(
            answerable=False, confidence=1.0, reason="no", gate="llm"
        ),
    )
    api._state["pipelines"]["hybrid"] = FakePipeline(refusal)
    body = client.post("/ask", json={"question": "A question?"}).json()
    assert len(body["retrieved"]) == 2


def test_a_failing_grounding_verdict_reaches_the_response(
    client: TestClient,
) -> None:
    # GroundingVerdict defines __bool__ to return its decision, so a truthiness
    # check in to_dict() dropped exactly the rejections a caller wants to see.
    # This is the API-level guard against that bug returning.
    rejected = PipelineResult(
        question="q",
        answer=ABSTENTION_TEXT,
        abstained=True,
        abstain_reason="grounding gate: only 50% of claims were supported",
        chunks=[make_chunk(1)],
        grounding=GroundingVerdict(grounded=False, score=0.5, claims=[]),
    )
    api._state["pipelines"]["hybrid"] = FakePipeline(rejected)
    body = client.post("/ask", json={"question": "A question?"}).json()
    assert body["grounding_gate"] is not None
    assert body["grounding_gate"]["grounded"] is False
    assert body["grounding_gate"]["score"] == 0.5


# --- /ask: validation --------------------------------------------------------


def test_empty_question_is_rejected(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_whitespace_only_question_is_rejected(client: TestClient) -> None:
    # Passes the min_length check but is not a question.
    assert client.post("/ask", json={"question": "     "}).status_code == 422


def test_missing_question_is_rejected(client: TestClient) -> None:
    assert client.post("/ask", json={}).status_code == 422


def test_overlong_question_is_rejected(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "x" * 2000}).status_code == 422


def test_unknown_retriever_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/ask", json={"question": "A question?", "retriever": "magic"}
    )
    assert response.status_code == 422


def test_k_outside_the_allowed_range_is_rejected(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "Q?", "k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "Q?", "k": 99}).status_code == 422


# --- /ask: gate switches -----------------------------------------------------


def test_gates_can_be_disabled_per_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reproduces the Phase 6 ablation against a live service rather than
    # asking a reader to trust the committed numbers.
    #
    # A non-default configuration builds a fresh pipeline rather than using a
    # cached one, so this path constructs a real RAGPipeline over the fake
    # retriever. Patching the class keeps the test on the API's own behaviour —
    # that the flags reach the pipeline — without pulling in generation.
    built: dict[str, object] = {}

    class SpyPipeline(FakePipeline):
        def __init__(self, retriever, **kwargs):
            super().__init__()
            built.update(kwargs)

    monkeypatch.setattr(api, "RAGPipeline", SpyPipeline)

    response = client.post(
        "/ask",
        json={
            "question": "A question?",
            "answerability_gate": False,
            "grounding_gate": False,
        },
    )
    assert response.status_code == 200
    assert built["enable_answerability"] is False
    assert built["enable_grounding"] is False


def test_a_non_default_k_reaches_the_pipeline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: dict[str, object] = {}

    class SpyPipeline(FakePipeline):
        def __init__(self, retriever, **kwargs):
            super().__init__()
            built.update(kwargs)

    monkeypatch.setattr(api, "RAGPipeline", SpyPipeline)
    client.post("/ask", json={"question": "A question?", "k": 10})
    assert built["k"] == 10


def test_the_default_configuration_reuses_the_cached_pipeline(
    client: TestClient, ready_state: FakePipeline
) -> None:
    # Building indexes per request would make the API unusable; the cached
    # pipeline is what stops that. Asserting it is reused keeps a future
    # refactor from quietly reintroducing per-request construction.
    client.post("/ask", json={"question": "A question?"})
    assert ready_state.asked == ["A question?"]


# --- failure modes -----------------------------------------------------------


def test_ask_returns_503_before_startup_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "_build_state", lambda: None)
    api._state.update({"ready": False, "error": "Corpus not found"})
    response = TestClient(api.app).post("/ask", json={"question": "A question?"})
    assert response.status_code == 503
    assert "Corpus not found" in response.json()["detail"]


def test_a_pipeline_failure_becomes_a_502(client: TestClient) -> None:
    # 502 rather than 500: the failure is in a dependency this service calls,
    # and the distinction tells an operator where to look.
    api._state["pipelines"]["hybrid"] = BrokenPipeline()
    response = client.post("/ask", json={"question": "A question?"})
    assert response.status_code == 502
    assert "ollama exploded" in response.json()["detail"]


# --- schema ------------------------------------------------------------------


def test_openapi_schema_is_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/ask" in schema["paths"]
    assert "/health" in schema["paths"]


def test_ask_documents_an_unanswerable_example(client: TestClient) -> None:
    # The examples are documentation: they show a caller the case the gates
    # exist for, which a bare schema cannot.
    schema = client.get("/openapi.json").json()
    body = schema["paths"]["/ask"]["post"]["requestBody"]
    examples = body["content"]["application/json"]["examples"]
    assert "unanswerable" in examples
    assert "autovacuum_vacuum_max_threshold" in str(examples["unanswerable"])
