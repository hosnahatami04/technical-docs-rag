# Multi-stage, so the runtime image carries the installed packages and the
# embedding model but not the build toolchain that produced them.
#
# The model is baked in on purpose. sentence-transformers downloads
# bge-small-en-v1.5 on first use, which would mean the container needs network
# access to answer its first question and fails in an air-gapped deployment.
# Copying it into the image costs ~130MB and makes startup deterministic.

# ---- build stage -------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# lxml needs libxml2/libxslt headers to build; they are not needed at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Dependencies are copied and installed before the source, so editing a Python
# file does not invalidate the layer that took minutes to build.
COPY requirements.txt .

# The CPU-only torch wheel is ~200MB against ~2.5GB for the CUDA build, and
# nothing in this image uses a GPU: the embedding model runs on CPU and
# generation happens in a separate Ollama process.
RUN pip install --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Pull the embedding model into the image. The name is duplicated from
# src/retrieval/dense.py rather than imported, because importing it here would
# mean copying the source into the build stage for one string.
ENV HF_HOME=/opt/hf
RUN PYTHONPATH=/install/lib/python3.12/site-packages \
    python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

# ---- runtime stage -----------------------------------------------------------
FROM python:3.12-slim AS runtime

# Runtime needs the shared libraries, not the headers.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libxml2 \
        libxslt1.1 \
        curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY --from=builder /opt/hf /opt/hf

# HF_HUB_OFFLINE stops sentence-transformers reaching for the network at all:
# the model is already in the image, and a silent download attempt on first use
# is exactly what baking it in was meant to prevent.
ENV HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Ollama runs on the host, not in this container. On Docker Desktop the name
# below resolves to the host; on Linux, run with
#   --add-host=host.docker.internal:host-gateway
# or override with -e OLLAMA_HOST=http://<host>:11434
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    RAG_MODEL=qwen2.5:7b-instruct \
    RAG_NUM_CTX=4096

WORKDIR /app

COPY src/ ./src/
COPY questions/ ./questions/
COPY results/ ./results/
COPY data/download.sh ./data/download.sh

# Nothing here needs root.
RUN useradd --create-home --uid 1000 rag \
 && mkdir -p /app/data/raw /app/data/indexes \
 && chown -R rag:rag /app
USER rag

EXPOSE 8000

# /health reports readiness rather than mere liveness: it distinguishes a
# missing corpus from an unreachable model, so an orchestrator restarting on
# failure restarts for a reason.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# The corpus is not baked in — it is 13MB of documentation that
# data/download.sh fetches at a pinned git tag, and freezing it into the image
# would make the image the source of truth instead of the tag. Mount it:
#
#   bash data/download.sh          # on the host, once
#   docker run -p 8000:8000 \
#     -v "$PWD/data:/app/data" \
#     technical-docs-rag
#
# The same mount persists the Chroma index, so the ~7 minutes of embedding
# happens once rather than on every container start.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
