"""Lexical retrieval with BM25.

BM25 scores a document by how many query terms it contains, weighted so that
rare terms count for more and long documents are not rewarded just for being
long. It matches strings, not meaning: it will find `max_wal_size` reliably and
will not connect "how often does PostgreSQL rotate log files" to
`log_rotation_age` at all.

That blind spot is the reason the dense retriever exists, and measuring where
each one wins is the point of Phase 5.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.retrieval.base import Retriever, ScoredChunk, rank_results

# Words carrying no discriminating power in a corpus that is entirely about
# PostgreSQL. "postgresql" is on the list for exactly that reason: it appears
# in almost every chunk, so matching it tells you nothing, while a question
# that mentions it would otherwise pull in arbitrary chunks.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "i",
        "you",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "so",
        "such",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "can",
        "could",
        "may",
        "might",
        "must",
        "should",
        "would",
        "will",
        "there",
        "here",
        "not",
        "no",
        "yes",
        "postgresql",
        "postgres",
    }
)

# Matches identifiers, dotted names, numbers with units, and ordinary words.
# Order matters: the identifier alternative comes first so `max_wal_size` is
# captured whole rather than as three separate words.
_TOKEN = re.compile(
    r"""
    [a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+   # snake_case identifier
  | [a-zA-Z][a-zA-Z0-9]*(?:\.[a-zA-Z0-9]+)+  # dotted name: postgresql.conf
  | \d+(?:\.\d+)?[a-zA-Z]{1,2}\b             # value with unit: 128MB, 30s
  | [a-zA-Z0-9]+                             # plain word or number
    """,
    re.VERBOSE,
)

_SPLIT_PARTS = re.compile(r"[_.]")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens for BM25, emitting identifiers both whole and in parts.

    The plan warns that splitting on underscores destroys lexical matching, and
    it is right that `max_wal_size` must survive as one token. But the opposite
    failure is the larger one here: 43 of the 55 answerable questions never
    spell an identifier out. "How often does PostgreSQL rotate log files?"
    has to reach `log_rotation_age`, and a tokenizer that only emits whole
    identifiers gives BM25 nothing to match.

    So both are emitted. `max_wal_size` indexes as
    ["max_wal_size", "max", "wal", "size"], which lets an exact mention score
    on the full identifier — still the strongest single signal, since it is
    rare — while a paraphrase can still reach it through the parts.

    Parts are only emitted for identifiers, never for ordinary words, so the
    vocabulary does not balloon.
    """
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        token = match.group(0)
        if token in STOPWORDS:
            continue
        tokens.append(token)
        if "_" in token or "." in token:
            tokens.extend(
                part
                for part in _SPLIT_PARTS.split(token)
                if len(part) > 1 and part not in STOPWORDS
            )
    return tokens


class BM25Retriever(Retriever):
    """Okapi BM25 over the chunk texts.

    Indexes `chunk.text`, which carries the heading path prefix, not
    `chunk.body`. That is the whole reason Phase 2 built the prefix: a chunk
    reading "The default is typically 128 megabytes" contains none of the words
    a question about shared_buffers would use, and BM25 cannot bridge that gap
    on its own.
    """

    name = "bm25"

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = chunks
        self._corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
        # rank_bm25 fails on an empty document, and a chunk can tokenize to
        # nothing if it is pure punctuation. Substituting a placeholder keeps
        # index positions aligned with self.chunks, which every lookup assumes.
        self._corpus_tokens = [
            tokens or ["\x00empty"] for tokens in self._corpus_tokens
        ]
        self._index = BM25Okapi(self._corpus_tokens)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)

        # A zero score means no query term appeared at all. Returning those
        # pads the result list with chunks that are not merely worse but
        # entirely irrelevant, which inflates nothing in Hit@k but makes the
        # hybrid fusion in RRF treat noise as a ranked candidate.
        scored = [
            (chunk, float(score))
            for chunk, score in zip(self.chunks, scores, strict=True)
            if score > 0
        ]
        return rank_results(scored, k)
