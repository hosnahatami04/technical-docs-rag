"""HTTP interface to the RAG pipeline.

Run:  uvicorn src.api:app --reload

Two endpoints matter. `POST /ask` answers a question; `GET /health` reports
whether the service can actually serve one.

What makes the response worth reading is not the answer — it is everything
beside it. Both gate verdicts, the retrieval scores, and the passages that were
considered all come back, so a caller can see *why* the system answered or
refused rather than being asked to trust it. A refusal in particular returns
the sources it rejected, which is what lets someone disagree with it.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.generation.llm import DEFAULT_MODEL, NUM_CTX, OllamaClient
from src.generation.pipeline import DEFAULT_K, RAGPipeline
from src.ingestion.indexer import CORPUS_ROOT, build_retrievers, load_chunks
from src.retrieval.base import Retriever

logger = logging.getLogger("rag.api")

# Loading the corpus, chunking it, and attaching the indexes takes a few
# seconds. Doing it per request would make the API unusable, so it happens once
# at startup and lives in this module-level state for the process lifetime.
_state: dict[str, Any] = {
    "chunks": None,
    "retrievers": {},
    "pipelines": {},
    "client": None,
    "ready": False,
    "error": None,
    "startup_seconds": 0.0,
}

# The plan's default. Phase 5 found dense ahead on MRR but hybrid ahead on
# Hit@5, and Hit@5 is what matters when five chunks go to the generator.
DEFAULT_RETRIEVER = "hybrid"

RetrieverName = Literal["bm25", "dense", "hybrid"]


# --- request and response models --------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="A question about the PostgreSQL documentation.",
        examples=["What is the default value of max_wal_size?"],
    )
    retriever: RetrieverName = Field(
        DEFAULT_RETRIEVER,
        description="Which retriever to use. All three are built at startup.",
    )
    k: int = Field(
        DEFAULT_K,
        ge=1,
        le=20,
        description="How many passages to retrieve and hand to the generator.",
    )
    # Exposed so a caller can reproduce the Phase 6 ablation against a live
    # service, and so the cost of each gate is visible rather than asserted.
    answerability_gate: bool = Field(
        True, description="Run the answerability gate before generating."
    )
    grounding_gate: bool = Field(
        True, description="Run the grounding gate after generating."
    )


class Citation(BaseModel):
    index: int
    doc_path: str
    heading: str


class RetrievedChunk(BaseModel):
    rank: int
    score: float
    doc_path: str
    heading: str


class AnswerabilityVerdict(BaseModel):
    gate: str
    answerable: bool
    reason: str


class GroundingVerdictModel(BaseModel):
    grounded: bool
    score: float
    n_claims: int
    unsupported: list[str]


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    abstain_reason: str = ""
    citations: list[Citation] = []
    retrieved: list[RetrievedChunk] = []
    answerability_gate: AnswerabilityVerdict | None = None
    grounding_gate: GroundingVerdictModel | None = None
    retriever: str
    seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "starting"]
    corpus_present: bool
    chunks_indexed: int
    retrievers: list[str]
    model: str
    model_available: bool
    num_ctx: int
    startup_seconds: float
    detail: str = ""


# --- lifespan ----------------------------------------------------------------


def _build_state() -> None:
    """Load the corpus and build every retriever. Called once, at startup."""
    started = time.time()
    try:
        chunks = load_chunks()
        retrievers = build_retrievers(chunks)
        client = OllamaClient()

        _state["chunks"] = chunks
        _state["retrievers"] = retrievers
        _state["client"] = client
        # One pipeline per retriever, built up front. Constructing one per
        # request would be cheap, but keeping them here makes it obvious that
        # the expensive parts — indexes and the model client — are shared.
        _state["pipelines"] = {
            name: RAGPipeline(retriever, client=client)
            for name, retriever in retrievers.items()
        }
        _state["ready"] = True
        _state["error"] = None
    except Exception as exc:
        # Starting anyway is deliberate: /health must be able to say what is
        # wrong. A service that refuses to boot can only be diagnosed from
        # container logs.
        _state["ready"] = False
        _state["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Startup failed")
    finally:
        _state["startup_seconds"] = round(time.time() - started, 2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _build_state()
    if _state["ready"]:
        logger.info(
            "Ready: %d chunks, retrievers=%s, %.1fs",
            len(_state["chunks"]),
            sorted(_state["retrievers"]),
            _state["startup_seconds"],
        )
    yield
    _state.clear()


app = FastAPI(
    title="RAG over PostgreSQL documentation",
    description=(
        "Question answering over the PostgreSQL 17.0 documentation, with two "
        "independent gates. Every response carries both gate verdicts and the "
        "passages that were considered, so a caller can see why the system "
        "answered or refused."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --- endpoints ---------------------------------------------------------------


@app.get("/", tags=["meta"])
def index() -> dict[str, Any]:
    return {
        "service": "technical-docs-rag",
        "corpus": "PostgreSQL 17.0 documentation",
        "endpoints": {
            "POST /ask": "Answer a question, with both gate verdicts.",
            "GET /health": "Readiness, index size, and model availability.",
            "GET /docs": "Interactive API documentation.",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Whether the service can actually serve a request.

    Deliberately more than a liveness probe. The three things that break this
    service in practice are a missing corpus, a missing index, and an
    unreachable model — and each fails differently. Reporting them separately
    is the difference between "it's broken" and knowing what to restart.
    """
    client: OllamaClient | None = _state.get("client")
    model_available = client.is_available() if client else False
    corpus_present = CORPUS_ROOT.exists()

    if not _state["ready"]:
        return HealthResponse(
            status="starting",
            corpus_present=corpus_present,
            chunks_indexed=0,
            retrievers=[],
            model=DEFAULT_MODEL,
            model_available=model_available,
            num_ctx=NUM_CTX,
            startup_seconds=_state["startup_seconds"],
            detail=_state["error"] or "Indexes are still building.",
        )

    # Retrieval works without the model; only generation needs it. Reporting
    # "degraded" rather than "down" says exactly that.
    detail = ""
    status: Literal["ok", "degraded"] = "ok"
    if not model_available:
        status = "degraded"
        detail = (
            f"Retrieval is available but {DEFAULT_MODEL} is not reachable. "
            f"Start it with `ollama serve`; /ask will fail until then."
        )
    if not corpus_present:
        status = "degraded"
        detail = (detail + " ").strip() + (
            f" Corpus missing from {CORPUS_ROOT}; the in-memory index is still "
            f"serving, but a restart would fail."
        ).strip()

    return HealthResponse(
        status=status,
        corpus_present=corpus_present,
        chunks_indexed=len(_state["chunks"]),
        retrievers=sorted(_state["retrievers"]),
        model=DEFAULT_MODEL,
        model_available=model_available,
        num_ctx=NUM_CTX,
        startup_seconds=_state["startup_seconds"],
        detail=detail,
    )


def _pipeline_for(request: AskRequest) -> RAGPipeline:
    """The pipeline for this request, configured per its gate flags."""
    retrievers: dict[str, Retriever] = _state["retrievers"]
    retriever = retrievers.get(request.retriever)
    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown retriever {request.retriever!r}. "
            f"Available: {sorted(retrievers)}.",
        )

    # The cached pipeline covers the default configuration. Anything else gets
    # a fresh one, which is cheap: it reuses the same indexes and client.
    if request.answerability_gate and request.grounding_gate and request.k == DEFAULT_K:
        return _state["pipelines"][request.retriever]

    return RAGPipeline(
        retriever,
        client=_state["client"],
        k=request.k,
        enable_answerability=request.answerability_gate,
        enable_grounding=request.grounding_gate,
    )


@app.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(
    request: Annotated[
        AskRequest,
        Body(
            openapi_examples={
                "answerable": {
                    "summary": "A question the docs answer",
                    "value": {"question": "What is the default value of max_wal_size?"},
                },
                "unanswerable": {
                    "summary": "A parameter that does not exist in 17.0",
                    "description": (
                        "Retrieval returns confident results for this — the real "
                        "autovacuum parameters are genuinely similar. The "
                        "answerability gate is what catches it."
                    ),
                    "value": {
                        "question": (
                            "What is the default value of "
                            "autovacuum_vacuum_max_threshold?"
                        )
                    },
                },
                "no_gates": {
                    "summary": "The same question with both gates off",
                    "description": "Reproduces the Phase 6 ablation live.",
                    "value": {
                        "question": (
                            "What is the default value of "
                            "autovacuum_vacuum_max_threshold?"
                        ),
                        "answerability_gate": False,
                        "grounding_gate": False,
                    },
                },
            }
        ),
    ],
) -> AskResponse:
    """Answer a question, and report how the answer was reached.

    Expect several seconds per call: the answerability gate, generation, and
    one grounding check per claim are separate model calls. That cost is the
    feature — it is what makes the verdicts in the response real rather than
    asserted.
    """
    if not _state["ready"]:
        raise HTTPException(
            status_code=503,
            detail=_state["error"] or "Indexes are still building. Try again shortly.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    pipeline = _pipeline_for(request)

    started = time.time()
    try:
        result = pipeline.ask(question)
    except Exception as exc:
        logger.exception("Pipeline failed for question %r", question)
        raise HTTPException(
            status_code=502,
            detail=f"The pipeline failed: {type(exc).__name__}: {exc}",
        ) from exc
    elapsed = time.time() - started

    payload = result.to_dict()
    return AskResponse(
        question=question,
        answer=payload["answer"],
        abstained=payload["abstained"],
        abstain_reason=payload["abstain_reason"],
        citations=[Citation(**c) for c in payload["citations"]],
        retrieved=[RetrievedChunk(**r) for r in payload["retrieved"]],
        answerability_gate=(
            AnswerabilityVerdict(**payload["answerability_gate"])
            if payload["answerability_gate"] is not None
            else None
        ),
        grounding_gate=(
            GroundingVerdictModel(**payload["grounding_gate"])
            if payload["grounding_gate"] is not None
            else None
        ),
        retriever=request.retriever,
        seconds=round(elapsed, 2),
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Return JSON for unexpected errors rather than an HTML traceback page."""
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {type(exc).__name__}"},
    )
