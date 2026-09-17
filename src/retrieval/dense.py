"""Semantic retrieval with sentence embeddings, stored in ChromaDB.

Where BM25 matches strings, this matches meaning. It can connect "how often
does PostgreSQL rotate log files" to a passage about `log_rotation_age` that
shares almost no words with the question — which is exactly the case BM25
cannot handle, and 43 of the 55 answerable questions are shaped that way.

The cost is the mirror image: an exact identifier like `max_wal_size` becomes a
point in a 384-dimensional space alongside every other configuration parameter,
and nothing guarantees the right one is nearest.
"""

from __future__ import annotations

import contextlib
import hashlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.ingestion.chunker import Chunk
from src.retrieval.base import Retriever, ScoredChunk, rank_results

if TYPE_CHECKING:  # heavy imports stay out of module load
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# From the model card: a short query must carry this instruction, and a passage
# must not. Forgetting it does not raise — the scores just come out quietly
# worse, which is the hardest kind of bug to notice because nothing looks
# broken. Tested explicitly in tests/test_retrieval.py.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_INDEX_DIR = Path("data/indexes/chroma")
COLLECTION_PREFIX = "chunks"

# Chroma needs a distance function chosen at collection creation. Cosine is the
# right one here because the model card specifies normalized embeddings, where
# cosine similarity is the intended comparison.
_DISTANCE_SPACE = "cosine"

# Encoding 6,406 chunks on CPU is the slow step of this phase. A larger batch
# is faster but holds more in memory; 64 is a reasonable middle for a laptop.
_BATCH_SIZE = 64


def _fingerprint(chunks: list[Chunk], model_name: str) -> str:
    """Identify exactly what a persisted index was built from.

    The index is expensive to rebuild, so it is cached — but a stale cache is
    worse than no cache: it would silently evaluate the new chunker against the
    old embeddings. Hashing the chunk ids, their text, and the model name means
    any change to chunking or model produces a different collection.
    """
    digest = hashlib.sha256()
    digest.update(model_name.encode())
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode())
        digest.update(chunk.text.encode())
    return digest.hexdigest()[:16]


class DenseRetriever(Retriever):
    """Embedding retrieval over a persistent Chroma collection."""

    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        model_name: str = MODEL_NAME,
        index_dir: Path | str = DEFAULT_INDEX_DIR,
        rebuild: bool = False,
        show_progress: bool = False,
    ) -> None:
        if not chunks:
            raise ValueError("DenseRetriever needs at least one chunk")

        self.chunks = chunks
        self.model_name = model_name
        self._by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._model: SentenceTransformer | None = None

        import chromadb

        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(index_path))

        fingerprint = _fingerprint(chunks, model_name)
        self._collection_name = f"{COLLECTION_PREFIX}-{fingerprint}"

        if rebuild:
            # Deleting a collection that was never created is the normal case
            # on a first --rebuild, not an error.
            with contextlib.suppress(Exception):
                self._client.delete_collection(self._collection_name)

        existing = {c.name for c in self._client.list_collections()}
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": _DISTANCE_SPACE},
        )

        if self._collection_name not in existing or self._collection.count() == 0:
            self._build(show_progress=show_progress)

    # -- model ---------------------------------------------------------------

    @property
    def model(self) -> SentenceTransformer:
        """Loaded on first use so importing this module stays cheap."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts. No prefix — the model card is explicit that the
        instruction applies to queries only.
        """
        vectors = self.model.encode(
            texts,
            batch_size=_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query, with the instruction the model expects."""
        vector = self.model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()

    # -- index ---------------------------------------------------------------

    def _build(self, *, show_progress: bool) -> None:
        total = len(self.chunks)
        # Carriage-return progress is unreadable once redirected to a file, so
        # only use it on a terminal; otherwise print a line every ~10%.
        interactive = show_progress and sys.stdout.isatty()
        step = max(_BATCH_SIZE, (total // 10 // _BATCH_SIZE + 1) * _BATCH_SIZE)

        for start in range(0, total, _BATCH_SIZE):
            batch = self.chunks[start : start + _BATCH_SIZE]
            if show_progress:
                done = min(start + _BATCH_SIZE, total)
                if interactive:
                    print(f"  embedding {done:>6,} / {total:,}", end="\r", flush=True)
                elif start % step == 0:
                    print(f"  embedding {done:>6,} / {total:,}", flush=True)
            self._collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=self.embed_passages([chunk.text for chunk in batch]),
                metadatas=[chunk.to_metadata() for chunk in batch],
            )
        if show_progress:
            print(f"  embedded  {total:>6,} / {total:,}")

    def __len__(self) -> int:
        return len(self.chunks)

    # -- search --------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        if not query.strip():
            return []

        response = self._collection.query(
            query_embeddings=[self.embed_query(query)],
            n_results=min(k, len(self.chunks)),
            include=["distances"],
        )

        ids = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]

        scored: list[tuple[Chunk, float]] = []
        debug: dict[str, dict[str, Any]] = {}
        for chunk_id, distance in zip(ids, distances, strict=True):
            chunk = self._by_id.get(chunk_id)
            if chunk is None:
                # A stale collection can hold ids this instance does not know.
                # Skipping is safer than guessing, and the fingerprint should
                # have prevented it.
                continue
            # Chroma returns cosine *distance*; similarity is 1 - distance, so
            # the score rises with relevance like every other retriever here.
            similarity = 1.0 - float(distance)
            scored.append((chunk, similarity))
            debug[chunk_id] = {"cosine_distance": float(distance)}

        return rank_results(scored, k, debug=debug)
