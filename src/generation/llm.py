"""Minimal Ollama client.

One place that knows how to talk to the model, so the answerer and both gates
cannot drift apart on temperature, context size, or error handling.

No SDK: the Ollama HTTP API is two endpoints and a JSON body, and a dependency
here would be more code to pin than to write.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# The model actually installed on this machine. The plan specifies qwen2.5:3b,
# but 7b-instruct was already present and is the better model for the same VRAM
# budget once the context window is sized correctly — see NUM_CTX below.
DEFAULT_MODEL = os.environ.get("RAG_MODEL", "qwen2.5:7b-instruct")

# This number is the difference between the model running on the GPU and
# running half on the CPU, and it was measured rather than guessed.
#
# qwen2.5 declares a 32,768-token context, and Ollama reserves KV cache for all
# of it: the model loads at 8.68GB, which does not fit in this machine's 8GB of
# VRAM, so 13% spills to system RAM and every generated token has to cross that
# boundary.
#
# At num_ctx=4096 the same model loads at 4.92GB and sits entirely on the GPU.
# Nothing about the weights changed; only the cache reservation did.
#
# 4096 is also simply enough: 5 chunks x 512 tokens + prompt + answer is about
# 3,200 tokens. Larger windows measured slower here, since a bigger KV cache is
# more work per token even when it fits.
NUM_CTX = int(os.environ.get("RAG_NUM_CTX", "4096"))

# Deterministic by construction. A seed alone is not enough — at any
# temperature above zero the sampler still picks among candidates, and a report
# whose numbers move between runs is not a measurement.
TEMPERATURE = 0.0
SEED = 42

# Generous, because the first call after a cold start includes loading ~5GB of
# weights onto the GPU. Steady-state generation is far faster.
REQUEST_TIMEOUT = 300


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or returns nothing usable."""


@dataclass
class Completion:
    """One model response, with the timing Ollama reports separately.

    `eval_duration` covers generation only, excluding the model load. Measuring
    with wall-clock time instead made the model look 17x slower than it is —
    3 tokens/sec versus the 51.6 it actually sustains once loaded.
    """

    text: str
    prompt_tokens: int
    output_tokens: int
    eval_seconds: float
    total_seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.eval_seconds if self.eval_seconds else 0.0


class OllamaClient:
    """Talks to a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str = DEFAULT_HOST,
        num_ctx: int = NUM_CTX,
        temperature: float = TEMPERATURE,
        seed: int = SEED,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed

    def is_available(self) -> bool:
        """Whether the server is up and the model is pulled.

        Checked before a run rather than discovered 40 questions in.
        """
        try:
            with urllib.request.urlopen(
                f"{self.host}/api/tags", timeout=10
            ) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return False
        installed = {m.get("name", "") for m in payload.get("models", [])}
        return self.model in installed

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> Completion:
        """Send one prompt and return the completion."""
        body: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        if system:
            body["system"] = system
        if stop:
            body["options"]["stop"] = stop  # type: ignore[index]

        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                payload = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.host}. Is `ollama serve` running? ({exc})"
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise LLMError(
                f"Ollama request timed out after {REQUEST_TIMEOUT}s"
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"Ollama returned malformed JSON: {exc}") from exc

        text = payload.get("response")
        if text is None:
            raise LLMError(f"Ollama response had no 'response' field: {payload!r}")

        return Completion(
            text=text.strip(),
            prompt_tokens=int(payload.get("prompt_eval_count", 0)),
            output_tokens=int(payload.get("eval_count", 0)),
            eval_seconds=float(payload.get("eval_duration", 0)) / 1e9,
            total_seconds=time.time() - started,
        )

    def __repr__(self) -> str:
        return f"OllamaClient(model={self.model!r}, num_ctx={self.num_ctx})"
