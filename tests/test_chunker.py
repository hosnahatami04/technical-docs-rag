"""Tests for both chunking strategies.

The plan names six properties this phase has to hold, and they are the spine of
this file: a code block stays intact, a table stays intact, heading paths are
right for deep nesting, and chunks respect the size limit except where an
atomic block forces an overflow.

Fixtures are hand-written DocBook rather than slices of the real corpus, so a
failure points at the chunker instead of at whatever PostgreSQL changed.
"""

from __future__ import annotations

import pytest
from lxml import etree

from src.ingestion.chunker import (
    EMBED_MODEL_LIMIT,
    Chunk,
    chunk_corpus,
    chunk_fixed,
    chunk_structural,
    count_tokens,
)
from src.ingestion.loader import Document


def build(xml: str, path: str = "test.sgml", doc_type: str = "guide") -> Document:
    """A Document from a DocBook string, with its tree attached."""
    root = etree.fromstring(xml.encode("utf-8"), etree.XMLParser(recover=True))
    return Document(
        path=path,
        title="Test Document",
        content=" ".join("".join(root.itertext()).split()),
        doc_type=doc_type,
        raw_xml=root,
    )


NESTED = """
<chapter>
 <title>Server Configuration</title>
 <para>Intro paragraph for the chapter.</para>
 <sect1>
  <title>Resource Consumption</title>
  <sect2>
   <title>Memory</title>
   <sect3>
    <title>Shared Buffers</title>
    <para>Sets the amount of memory the database server uses.</para>
   </sect3>
  </sect2>
 </sect1>
</chapter>
"""

WITH_CODE = """
<chapter>
 <title>Examples</title>
 <sect1>
  <title>Creating Tables</title>
  <para>Use CREATE TABLE to define a new table.</para>
<programlisting>
CREATE TABLE films (
    code        char(5) CONSTRAINT firstkey PRIMARY KEY,
    title       varchar(40) NOT NULL,
        did     integer NOT NULL
);
</programlisting>
  <para>The command returns when the table exists.</para>
 </sect1>
</chapter>
"""

WITH_TABLE = """
<chapter>
 <title>Parameters</title>
 <sect1>
  <title>Settings</title>
  <table>
   <title>Configuration Settings</title>
   <tgroup cols="2">
    <thead><row><entry>Name</entry><entry>Default</entry></row></thead>
    <tbody>
     <row><entry>shared_buffers</entry><entry>128MB</entry></row>
     <row><entry>work_mem</entry><entry>4MB</entry></row>
    </tbody>
   </tgroup>
  </table>
 </sect1>
</chapter>
"""

# A variablelist is a container, not a unit: each varlistentry is one parameter
# and its description, and a substantial one should become its own chunk.
# Entry lengths here mirror the real config.sgml, where the median parameter
# description is ~180 tokens — long enough to clear MIN_STANDALONE_TOKENS.
WITH_VARLIST = """
<chapter>
 <title>Resource Consumption</title>
 <sect1>
  <title>Memory</title>
  <variablelist>
   <varlistentry>
    <term>shared_buffers (integer)</term>
    <listitem><para>Sets the amount of memory the database server uses for
    shared memory buffers. The default is typically 128 megabytes, but might
    be less if your kernel settings will not support it, as determined during
    initdb. This setting must be at least 128 kilobytes. However, settings
    significantly higher than the minimum are usually needed for good
    performance. If this value is specified without units it is taken as
    blocks. This parameter can only be set at server start.</para></listitem>
   </varlistentry>
   <varlistentry>
    <term>work_mem (integer)</term>
    <listitem><para>Sets the base maximum amount of memory used by a query
    operation before writing to temporary disk files. If this value is
    specified without units it is taken as kilobytes. The default value is
    four megabytes. Note that a complex query might perform several sort and
    hash operations at the same time, with each operation generally permitted
    to use as much memory as this value specifies before it starts to write
    data into temporary files.</para></listitem>
   </varlistentry>
  </variablelist>
 </sect1>
</chapter>
"""

# Short entries — a glossary, as in acronyms.sgml — must NOT each become a
# chunk. Four tokens carries no context worth retrieving on its own.
WITH_SHORT_VARLIST = """
<chapter>
 <title>Acronyms</title>
 <sect1>
  <title>List</title>
  <variablelist>
   <varlistentry><term>AM</term>
    <listitem><para>Access Method</para></listitem></varlistentry>
   <varlistentry><term>API</term>
    <listitem><para>Application Programming Interface</para></listitem></varlistentry>
   <varlistentry><term>BKI</term>
    <listitem><para>Backend Interface</para></listitem></varlistentry>
  </variablelist>
 </sect1>
</chapter>
"""

# The other structural shape: the 220 command reference pages.
REFENTRY = """
<refentry>
 <refmeta>
  <refentrytitle>CREATE TABLE</refentrytitle>
  <manvolnum>7</manvolnum>
 </refmeta>
 <refnamediv>
  <refname>CREATE TABLE</refname>
  <refpurpose>define a new table</refpurpose>
 </refnamediv>
 <refsect1>
  <title>Description</title>
  <para>CREATE TABLE will create a new, initially empty table.</para>
  <refsect2>
   <title>Parameters</title>
   <para>The parameters are described below in detail.</para>
  </refsect2>
 </refsect1>
</refentry>
"""


# --- Heading paths ---------------------------------------------------------


def test_heading_path_tracks_deep_nesting() -> None:
    chunks = chunk_structural(build(NESTED))
    deep = next(c for c in chunks if "memory the database" in c.body)
    assert deep.heading_path == (
        "Server Configuration",
        "Resource Consumption",
        "Memory",
        "Shared Buffers",
    )


def test_heading_path_is_prefixed_onto_indexed_text() -> None:
    # The whole point of the prefix: a chunk that answers a question about
    # shared_buffers must contain those words even when the body does not.
    chunks = chunk_structural(build(NESTED))
    deep = next(c for c in chunks if "memory the database" in c.body)
    assert deep.text.startswith("Server Configuration > Resource Consumption")
    assert "Shared Buffers" in deep.text
    assert "Shared Buffers" not in deep.body  # body stays unsynthesized


def test_refentry_headings_use_the_other_structural_shape() -> None:
    # 62% of the corpus nests refsect1/refsect2 rather than sect1/sect2, and a
    # chunker that only knew sect* would emit one giant chunk for each.
    chunks = chunk_structural(build(REFENTRY, path="ref/create_table.sgml"))
    paths = {c.heading_path for c in chunks}
    assert ("CREATE TABLE", "Description") in paths
    assert ("CREATE TABLE", "Description", "Parameters") in paths


def test_body_excludes_the_heading_so_grounding_cannot_cheat() -> None:
    # Phase 6 checks generated claims against source content. If the heading we
    # synthesized were part of the body, a claim could be "supported" by text
    # PostgreSQL never wrote.
    chunks = chunk_structural(build(NESTED))
    for chunk in chunks:
        assert not chunk.body.startswith(chunk.heading)


# --- Atomic blocks ---------------------------------------------------------


def test_code_block_is_never_split() -> None:
    chunks = chunk_structural(build(WITH_CODE))
    holding = [c for c in chunks if "CREATE TABLE films" in c.body]
    assert len(holding) == 1
    body = holding[0].body
    assert "code        char(5)" in body
    assert ");" in body


def test_code_block_indentation_survives() -> None:
    # Indentation is meaning in a SQL sample; a collapsed one is unusable.
    chunks = chunk_structural(build(WITH_CODE))
    body = next(c.body for c in chunks if "CREATE TABLE films" in c.body)
    assert "        did     integer NOT NULL" in body


def test_chunk_containing_code_is_flagged() -> None:
    chunks = chunk_structural(build(WITH_CODE))
    assert any(c.has_code for c in chunks)


def test_table_is_never_split_from_its_headers() -> None:
    # A row without its header row loses the meaning of every column.
    chunks = chunk_structural(build(WITH_TABLE))
    holding = [c for c in chunks if "shared_buffers" in c.body]
    assert len(holding) == 1
    body = holding[0].body
    assert "Name" in body and "Default" in body
    assert "work_mem" in body


def test_chunk_containing_table_is_flagged() -> None:
    chunks = chunk_structural(build(WITH_TABLE))
    assert any(c.has_table for c in chunks)


def test_oversized_atomic_block_becomes_its_own_chunk() -> None:
    # The plan calls this correct behavior. The alternative — splitting — would
    # produce a fragment that helps nobody.
    big = "SELECT " + ", ".join(f"column_{i}" for i in range(900)) + ";"
    doc = build(
        f"<chapter><title>Big</title><sect1><title>S</title>"
        f"<programlisting>{big}</programlisting></sect1></chapter>"
    )
    chunks = chunk_structural(doc)
    holding = [c for c in chunks if "column_899" in c.body]
    assert len(holding) == 1
    assert holding[0].token_count > EMBED_MODEL_LIMIT
    assert holding[0].oversized is True


def test_ordinary_chunks_are_not_flagged_oversized() -> None:
    chunks = chunk_structural(build(NESTED))
    assert all(c.oversized is False for c in chunks)


# --- Lists -----------------------------------------------------------------


def test_substantial_list_entries_get_their_own_chunks() -> None:
    # A variablelist is a container. Treating it as one atomic block would
    # build a single chunk per list — in config.sgml that is 5,798 tokens
    # across 22 parameters, of which the embedder would read the first 512.
    chunks = chunk_structural(build(WITH_VARLIST))
    shared = [c for c in chunks if "shared_buffers" in c.body]
    work = [c for c in chunks if "work_mem" in c.body]
    assert len(shared) == 1
    assert len(work) == 1
    # Separate chunks: a question about one parameter should not retrieve a
    # chunk that is half about another.
    assert shared[0].chunk_id != work[0].chunk_id
    assert "work_mem" not in shared[0].body


def test_list_entry_keeps_its_term_with_its_description() -> None:
    chunks = chunk_structural(build(WITH_VARLIST))
    body = next(c.body for c in chunks if "shared_buffers" in c.body)
    assert "shared_buffers (integer)" in body
    assert "128 megabytes" in body


def test_short_list_entries_are_packed_together() -> None:
    # acronyms.sgml is a glossary of 4-to-9-token entries. Giving each its own
    # chunk drove the corpus median down to 40 tokens — roughly one sentence,
    # too little context to retrieve or generate from.
    chunks = chunk_structural(build(WITH_SHORT_VARLIST))
    holding = [c for c in chunks if "Access Method" in c.body]
    assert len(holding) == 1
    assert "Application Programming Interface" in holding[0].body
    assert "Backend Interface" in holding[0].body


# --- Inline vs block elements ----------------------------------------------

INLINE_PROSE = """
<chapter>
 <title>Views</title>
 <sect1>
  <title>Creating Views</title>
  <para>
   Refer back to the queries in <xref linkend="tutorial-join"/>.
   You can create a <firstterm>view</firstterm> over the query, which
   gives it a name you can use like an ordinary <literal>TABLE</literal>.
  </para>
 </sect1>
</chapter>
"""


def test_inline_elements_do_not_break_a_sentence() -> None:
    # <literal> appears 22,533 times inside paragraphs in this corpus. Treating
    # it as a block split sentences into fragments joined by blank lines,
    # which is unreadable in a retrieved chunk and pollutes the embedding.
    body = chunk_structural(build(INLINE_PROSE))[0].body
    assert "create a view over the query" in body
    assert "like an ordinary TABLE" in body


def test_empty_xref_does_not_strand_its_punctuation() -> None:
    # An <xref/> has no text of its own, leaving "the queries in ." behind.
    body = chunk_structural(build(INLINE_PROSE))[0].body
    assert " ." not in body


def test_block_elements_inside_a_paragraph_still_separate() -> None:
    # 10% of paragraphs wrap a code sample. Inline handling must not swallow
    # it into the sentence.
    doc = build("""
    <chapter><title>T</title><sect1><title>S</title>
     <para>Create it with <literal>CREATE VIEW</literal> as follows:
<programlisting>
CREATE VIEW v AS
    SELECT 1;
</programlisting>
     </para>
    </sect1></chapter>
    """)
    body = chunk_structural(doc)[0].body
    assert "Create it with CREATE VIEW as follows:" in body
    assert "    SELECT 1;" in body  # indentation intact


# --- Size limits -----------------------------------------------------------


def test_chunks_respect_the_target_unless_an_atomic_block_forces_overflow() -> None:
    doc = build(NESTED)
    for chunk in chunk_structural(doc, target_tokens=40):
        assert chunk.token_count <= 40 or chunk.oversized


def test_token_count_matches_the_body() -> None:
    for chunk in chunk_structural(build(WITH_CODE)):
        assert chunk.token_count == count_tokens(chunk.body)


# --- Fixed baseline --------------------------------------------------------


def test_fixed_chunking_respects_the_target() -> None:
    doc = build(NESTED)
    doc.content = " ".join(f"word{i}" for i in range(4000))
    for chunk in chunk_fixed(doc, target_tokens=100, overlap_tokens=10):
        assert chunk.token_count <= 100


def test_fixed_chunking_overlaps_consecutive_chunks() -> None:
    # Overlap is what stops a sentence landing on a boundary from being lost.
    doc = build(NESTED)
    doc.content = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_fixed(doc, target_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    first_tail = chunks[0].body.split()[-5:]
    assert any(word in chunks[1].body for word in first_tail)


def test_fixed_chunking_carries_no_structure() -> None:
    # That absence is the baseline's defining property, and what Phase 5
    # measures the structural version against.
    chunks = chunk_fixed(build(NESTED))
    assert all(c.heading_path == () for c in chunks)
    assert all(c.strategy == "fixed" for c in chunks)


def test_fixed_chunking_of_short_text_yields_one_chunk() -> None:
    doc = build(NESTED)
    doc.content = "A short document."
    assert len(chunk_fixed(doc)) == 1


def test_empty_document_yields_no_chunks() -> None:
    doc = build("<chapter><title>T</title></chapter>")
    doc.content = ""
    assert chunk_fixed(doc) == []


# --- Identity and metadata -------------------------------------------------


def test_chunk_ids_are_unique_and_stable() -> None:
    docs = [build(NESTED, path="a.sgml"), build(WITH_CODE, path="b.sgml")]
    first = chunk_corpus(docs, strategy="structural")
    second = chunk_corpus(docs, strategy="structural")
    ids = [c.chunk_id for c in first]
    assert len(ids) == len(set(ids))
    assert ids == [c.chunk_id for c in second]


def test_chunk_id_records_the_strategy() -> None:
    # Both strategies index the same corpus during the Phase 5 ablation, so
    # their IDs must not collide.
    doc = build(NESTED)
    structural = {c.chunk_id for c in chunk_structural(doc)}
    fixed = {c.chunk_id for c in chunk_fixed(doc)}
    assert not (structural & fixed)


def test_chunks_carry_source_document_metadata() -> None:
    chunks = chunk_structural(build(NESTED, path="config.sgml", doc_type="config"))
    for chunk in chunks:
        assert chunk.doc_path == "config.sgml"
        assert chunk.doc_type == "config"
        assert chunk.doc_title == "Test Document"


def test_metadata_is_flat_scalars_for_the_vector_store() -> None:
    # Chroma rejects nested values, so heading_path has to be joined.
    chunk = chunk_structural(build(NESTED))[0]
    for key, value in chunk.to_metadata().items():
        assert isinstance(value, str | int | bool), f"{key} is {type(value)}"


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        chunk_corpus([build(NESTED)], strategy="semantic")


# --- Fallbacks -------------------------------------------------------------


def test_document_without_a_tree_falls_back_to_fixed() -> None:
    doc = Document(
        path="x.sgml", title="X", content="word " * 400, doc_type="guide", raw_xml=None
    )
    chunks = chunk_structural(doc)
    assert chunks
    assert all(c.strategy == "fixed" for c in chunks)


def test_structureless_tree_still_produces_chunks() -> None:
    doc = build(
        "<chapter><title>T</title><para>Some prose with no sections at "
        "all, long enough to be worth indexing on its own.</para></chapter>"
    )
    assert chunk_structural(doc)


def test_text_property_falls_back_to_body_without_a_heading() -> None:
    chunk = Chunk(
        chunk_id="x",
        doc_path="p",
        doc_title="t",
        doc_type="guide",
        heading_path=(),
        body="content",
        chunk_index=0,
        token_count=1,
        has_code=False,
        has_table=False,
    )
    assert chunk.text == "content"
