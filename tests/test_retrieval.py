"""Tests for the three retrievers and the interface they share.

The plan names three things this phase must prove: each retriever returns k
results, underscore-bearing identifiers survive BM25 tokenization, and RRF
fuses two known ranked lists into a hand-computed expectation.

BM25 and RRF are tested directly. The dense retriever is tested through a fake
embedding function rather than the real model — the tests must run in CI
without downloading 130MB of weights or embedding 6,406 chunks, and what needs
testing is the wiring (query prefix, distance conversion, ranking), not whether
BGE produces good vectors.
"""

from __future__ import annotations

import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.base import Retriever, ScoredChunk, rank_results
from src.retrieval.bm25 import BM25Retriever, tokenize
from src.retrieval.hybrid import (
    RRF_K,
    HybridRetriever,
    reciprocal_rank_fusion,
)


def make_chunk(
    chunk_id: str, body: str, *, heading: str = "", path: str = "d.sgml"
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_path=path,
        doc_title="Doc",
        doc_type="config",
        heading_path=tuple(heading.split(" > ")) if heading else (),
        body=body,
        chunk_index=0,
        token_count=len(body.split()),
        has_code=False,
        has_table=False,
    )


# BM25 weights a term by inverse document frequency, so a term appearing in
# every document scores zero — it carries no information. In a two-document
# corpus almost every term is in that position, which makes tiny fixtures
# behave nothing like the real 6,406-chunk index. These filler chunks give IDF
# something to work with; they share no vocabulary with the queries under test.
FILLER = [
    "Locale support refers to an application respecting cultural preferences "
    "regarding alphabets, sorting and character set conventions.",
    "A foreign data wrapper is a library that can communicate with an external "
    "data source, hiding the details of connecting to it.",
    "Triggers may be attached to tables, views and foreign tables, and fire "
    "before or after an insert, update or delete operation.",
    "Extensions bundle related SQL objects into a single unit that can be "
    "loaded or removed with one command.",
    "Tablespaces allow an administrator to define locations in the file system "
    "where the files representing objects can be stored.",
    "Text search parsers break a document into tokens and assign each token a "
    "type from a predefined set.",
]


@pytest.fixture
def corpus() -> list[Chunk]:
    chunks = [
        make_chunk(
            "c1",
            "Sets the maximum size the WAL is allowed to grow between automatic "
            "checkpoints. The default is 1 GB.",
            heading="Server Configuration > Write Ahead Log > max_wal_size",
        ),
        make_chunk(
            "c2",
            "Determines the maximum number of concurrent connections to the "
            "database server. The default is typically 100 connections.",
            heading="Server Configuration > Connections > max_connections",
        ),
        make_chunk(
            "c3",
            "VACUUM FULL rewrites the entire contents of the table into a new "
            "disk file with no extra space.",
            heading="VACUUM > Description",
            path="ref/vacuum.sgml",
        ),
        make_chunk(
            "c4",
            "The pg_stat_activity view shows one row per server process with "
            "the current query text.",
            heading="Monitoring > pg_stat_activity",
            path="monitoring.sgml",
        ),
    ]
    chunks += [
        make_chunk(f"f{i}", text, heading=f"Filler > Section {i}", path="filler.sgml")
        for i, text in enumerate(FILLER)
    ]
    return chunks


# --- Tokenization -----------------------------------------------------------


def test_underscore_identifiers_survive_whole() -> None:
    # The plan's warning: a tokenizer that splits on underscores destroys
    # lexical matching for exactly the terms this corpus is full of.
    assert "max_wal_size" in tokenize("What is the default value of max_wal_size?")
    assert "pg_stat_activity" in tokenize("Query pg_stat_activity for running queries")


def test_identifiers_are_also_emitted_in_parts() -> None:
    # The opposite failure matters more here: 43 of the 55 answerable questions
    # never spell an identifier out, so whole-identifier-only tokenization
    # leaves BM25 nothing to match on most of the set.
    tokens = tokenize("max_wal_size")
    assert tokens[0] == "max_wal_size"
    assert {"max", "wal", "size"} <= set(tokens)


def test_ordinary_words_are_not_split() -> None:
    # Parts are only emitted for identifiers; otherwise the vocabulary grows
    # for no gain.
    assert tokenize("database") == ["database"]


def test_dotted_filenames_survive() -> None:
    assert "postgresql.conf" in tokenize("Edit postgresql.conf to change it")


def test_values_with_units_stay_intact() -> None:
    # "128MB" as one token is what lets a question about a default value match
    # the passage stating it.
    assert "128mb" in tokenize("Set shared_buffers to 128MB")
    assert "30s" in tokenize("checkpoint_warning defaults to 30s")


def test_stopwords_are_dropped() -> None:
    tokens = tokenize("What is the default value of this?")
    assert "the" not in tokens
    assert "what" not in tokens
    assert "default" in tokens


def test_postgresql_is_treated_as_a_stopword() -> None:
    # It appears in nearly every chunk of a PostgreSQL corpus, so matching it
    # carries no information but would pull in arbitrary chunks.
    assert "postgresql" not in tokenize("How does PostgreSQL rotate log files?")


def test_tokenizer_is_case_insensitive() -> None:
    assert tokenize("VACUUM FULL") == tokenize("vacuum full")


# --- BM25 -------------------------------------------------------------------


def test_bm25_returns_k_results(corpus: list[Chunk]) -> None:
    retriever = BM25Retriever(corpus)
    assert len(retriever.search("default value server", k=2)) == 2


def test_bm25_never_returns_more_than_k(corpus: list[Chunk]) -> None:
    assert len(BM25Retriever(corpus).search("default", k=1)) == 1


def test_bm25_returns_fewer_than_k_when_few_chunks_match(corpus: list[Chunk]) -> None:
    # Zero-score chunks are excluded: they are not merely worse, they contain
    # no query term at all, and feeding them to RRF would treat noise as a
    # ranked candidate.
    results = BM25Retriever(corpus).search("pg_stat_activity", k=4)
    assert 0 < len(results) < 4


def test_bm25_finds_the_exact_identifier(corpus: list[Chunk]) -> None:
    results = BM25Retriever(corpus).search("max_wal_size", k=1)
    assert results[0].chunk_id == "c1"


def test_bm25_ranks_start_at_one_and_are_contiguous(corpus: list[Chunk]) -> None:
    # Hit@1 and MRR are defined over human ranks; an off-by-one here would
    # shift every metric in the Phase 5 report.
    results = BM25Retriever(corpus).search("default server database", k=3)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_bm25_scores_descend(corpus: list[Chunk]) -> None:
    results = BM25Retriever(corpus).search("default value server", k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_returns_no_duplicates(corpus: list[Chunk]) -> None:
    results = BM25Retriever(corpus).search("default server", k=4)
    assert len({r.chunk_id for r in results}) == len(results)


def test_bm25_is_deterministic(corpus: list[Chunk]) -> None:
    # Ties are common — two chunks matching the same single rare term score
    # identically — and without a deterministic tiebreak the committed results
    # file could not be reproduced.
    retriever = BM25Retriever(corpus)
    first = [r.chunk_id for r in retriever.search("default value", k=4)]
    second = [r.chunk_id for r in retriever.search("default value", k=4)]
    assert first == second


def test_bm25_searches_the_heading_prefix_not_just_the_body(
    corpus: list[Chunk],
) -> None:
    # The reason Phase 2 built heading paths: this body never says
    # "shared_buffers", and without the prefix BM25 could not reach it.
    target = make_chunk(
        "target",
        "The default is typically 128 megabytes.",
        heading="Server Configuration > Resource Consumption > shared_buffers",
    )
    assert "shared_buffers" not in target.body

    results = BM25Retriever([target, *corpus]).search("shared_buffers", k=1)
    assert results and results[0].chunk_id == "target"


def test_bm25_empty_query_returns_nothing(corpus: list[Chunk]) -> None:
    assert BM25Retriever(corpus).search("", k=3) == []


def test_bm25_stopword_only_query_returns_nothing(corpus: list[Chunk]) -> None:
    assert BM25Retriever(corpus).search("what is the", k=3) == []


def test_bm25_rejects_an_empty_corpus() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        BM25Retriever([])


def test_bm25_handles_a_chunk_that_tokenizes_to_nothing(
    corpus: list[Chunk],
) -> None:
    # rank_bm25 fails on an empty document, and index positions must stay
    # aligned with the chunk list — a misalignment would return the wrong
    # chunk for a correct score.
    punctuation_only = make_chunk("empty", "...", path="odd.sgml")
    retriever = BM25Retriever([punctuation_only, *corpus])
    assert len(retriever) == len(corpus) + 1

    results = retriever.search("max_wal_size", k=2)
    assert results[0].chunk_id == "c1"


# --- RRF --------------------------------------------------------------------


def test_rrf_matches_a_hand_computed_fusion() -> None:
    # Hand computation, as the plan asks:
    #   x: rank 1 in A (1/61) + rank 2 in B (1/62) = 0.032522
    #   y: rank 2 in A (1/62) + rank 1 in B (1/61) = 0.032522
    #   z: rank 3 in A (1/63) only                 = 0.015873
    #   w: rank 3 in B (1/63) only                 = 0.015873
    fused = dict(reciprocal_rank_fusion([["x", "y", "z"], ["y", "x", "w"]]))
    assert fused["x"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["z"] == pytest.approx(1 / 63)
    assert fused["w"] == pytest.approx(1 / 63)


def test_rrf_rewards_agreement_over_one_retrievers_confidence() -> None:
    # The whole point of the k constant. "agreed" is ranked 2nd by both lists
    # and beats "top", which is ranked 1st by one and absent from the other.
    fused = dict(reciprocal_rank_fusion([["top", "agreed"], ["other", "agreed"]]))
    assert fused["agreed"] > fused["top"]


def test_rrf_ignores_documents_missing_from_a_list() -> None:
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]]))
    assert fused["a"] == pytest.approx(1 / (RRF_K + 1))
    assert fused["b"] == pytest.approx(1 / (RRF_K + 1))


def test_rrf_output_is_sorted_best_first() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "b", "c"]])
    assert [item for item, _ in fused] == ["a", "b", "c"]


def test_rrf_ties_break_deterministically() -> None:
    first = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
    second = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
    assert first == second


def test_smaller_k_sharpens_the_advantage_of_a_high_rank() -> None:
    # What the constant actually controls is how steeply score falls with rank.
    # At k=1 the gap between rank 1 and rank 3 is 1/2 vs 1/4 — a factor of two.
    # At k=60 it is 1/61 vs 1/63, barely 3%. Flattening the curve is what lets
    # agreement between retrievers outweigh one retriever's confidence.
    sharp = dict(reciprocal_rank_fusion([["first", "second", "third"]], rrf_k=1))
    flat = dict(reciprocal_rank_fusion([["first", "second", "third"]], rrf_k=RRF_K))

    assert sharp["first"] / sharp["third"] == pytest.approx(2.0)
    assert flat["first"] / flat["third"] == pytest.approx(63 / 61, rel=1e-6)
    assert sharp["first"] / sharp["third"] > flat["first"] / flat["third"]


def test_a_single_top_hit_can_outvote_agreement_when_k_is_small() -> None:
    # The failure mode k=60 prevents. With k=1, one retriever ranking something
    # first (1/2) beats both retrievers ranking something second (1/3 + 1/3)
    # only once the lists are long enough to push the agreed item down.
    sharp = dict(
        reciprocal_rank_fusion(
            [["top", "a", "b", "agreed"], ["other", "c", "d", "agreed"]], rrf_k=1
        )
    )
    assert sharp["top"] > sharp["agreed"]

    flat = dict(
        reciprocal_rank_fusion(
            [["top", "a", "b", "agreed"], ["other", "c", "d", "agreed"]], rrf_k=RRF_K
        )
    )
    assert flat["agreed"] > flat["top"]


# --- Hybrid -----------------------------------------------------------------


class FakeRetriever(Retriever):
    """Returns a fixed ranking, so fusion can be tested without an index."""

    def __init__(self, name: str, chunks: list[Chunk]) -> None:
        self.name = name
        self._chunks = chunks

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        scored = [
            (chunk, float(len(self._chunks) - index))
            for index, chunk in enumerate(self._chunks)
        ]
        return rank_results(scored, k)


def test_hybrid_returns_k_results(corpus: list[Chunk]) -> None:
    hybrid = HybridRetriever(
        [FakeRetriever("a", corpus), FakeRetriever("b", list(reversed(corpus)))]
    )
    assert len(hybrid.search("anything", k=2)) == 2


def test_hybrid_surfaces_what_both_retrievers_agree_on(corpus: list[Chunk]) -> None:
    a = FakeRetriever("a", [corpus[0], corpus[1], corpus[2]])
    b = FakeRetriever("b", [corpus[3], corpus[1], corpus[2]])
    top = HybridRetriever([a, b]).search("q", k=1)[0]
    # c2 is ranked 2nd by both; c1 and c4 are each ranked 1st by only one.
    assert top.chunk_id == "c2"


def test_hybrid_records_each_retrievers_rank(corpus: list[Chunk]) -> None:
    # Without this, a per-category disagreement in Phase 5 is visible but not
    # explainable.
    hybrid = HybridRetriever([FakeRetriever("a", corpus), FakeRetriever("b", corpus)])
    top = hybrid.search("q", k=1)[0]
    assert top.debug["ranks"] == {"a": 1, "b": 1}
    assert top.debug["retrievers_agreeing"] == 2


def test_hybrid_deduplicates_across_retrievers(corpus: list[Chunk]) -> None:
    hybrid = HybridRetriever([FakeRetriever("a", corpus), FakeRetriever("b", corpus)])
    results = hybrid.search("q", k=4)
    assert len({r.chunk_id for r in results}) == len(results)


def test_hybrid_ranks_are_contiguous(corpus: list[Chunk]) -> None:
    hybrid = HybridRetriever([FakeRetriever("a", corpus), FakeRetriever("b", corpus)])
    results = hybrid.search("q", k=3)
    assert [r.rank for r in results] == [1, 2, 3]


def test_hybrid_needs_two_retrievers(corpus: list[Chunk]) -> None:
    with pytest.raises(ValueError, match="at least two"):
        HybridRetriever([FakeRetriever("a", corpus)])


def test_hybrid_rejects_invalid_parameters(corpus: list[Chunk]) -> None:
    pair = [FakeRetriever("a", corpus), FakeRetriever("b", corpus)]
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(pair, rrf_k=0)
    with pytest.raises(ValueError, match="fusion_depth"):
        HybridRetriever(pair, fusion_depth=0)


def test_hybrid_fuses_deeper_than_it_returns(corpus: list[Chunk]) -> None:
    # A chunk ranked 15th by one and 12th by the other is a strong consensus
    # candidate that neither would surface in its own top 5.
    hybrid = HybridRetriever(
        [FakeRetriever("a", corpus), FakeRetriever("b", corpus)], fusion_depth=4
    )
    assert len(hybrid.search("q", k=2)) == 2


# --- The shared interface ---------------------------------------------------


def test_rank_results_truncates_and_ranks(corpus: list[Chunk]) -> None:
    scored = [(chunk, float(i)) for i, chunk in enumerate(corpus)]
    results = rank_results(scored, k=2)
    assert len(results) == 2
    assert results[0].score > results[1].score
    assert [r.rank for r in results] == [1, 2]


def test_rank_results_breaks_ties_on_chunk_id(corpus: list[Chunk]) -> None:
    scored = [(chunk, 1.0) for chunk in corpus]
    ids = [r.chunk_id for r in rank_results(scored, k=4)]
    assert ids == sorted(ids)


def test_scored_chunk_exposes_both_scoring_keys(corpus: list[Chunk]) -> None:
    # Phase 5 scores Hit@k at both file and section level.
    result = rank_results([(corpus[0], 1.0)], k=1)[0]
    assert result.doc_path == "d.sgml"
    assert result.heading.endswith("max_wal_size")
