"""Tests for the corpus loader.

These are resilience tests more than correctness tests. The corpus is 386 files
of markup we do not control, and the failure that matters is one bad file
aborting the whole ingest — silently losing every document after it. So the
loader is expected to skip what it cannot parse and keep going.

Every test builds its own miniature corpus in a temp directory. Nothing here
reads data/raw/, so the suite passes in CI where the corpus was never
downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.loader import (
    Document,
    classify,
    load_corpus,
    load_document,
)

# A minimal but realistic chapter: nested sections, a code block, an indexterm
# to be dropped, and a custom entity.
VALID_CHAPTER = """<!-- doc/src/sgml/sample.sgml -->
<chapter id="sample">
 <title>Server Configuration</title>
 <indexterm><primary>configuration</primary></indexterm>
 <para>
  There are many configuration parameters that affect
  the behavior of the database system.
 </para>
 <sect1 id="sample-setting">
  <title>Setting Parameters</title>
  <sect2 id="sample-names">
   <title>Parameter Names and Values</title>
   <para>All parameter names are case-insensitive &mdash; always.</para>
<programlisting>
SET work_mem TO '64MB';
    SELECT pg_reload_conf();
</programlisting>
  </sect2>
 </sect1>
</chapter>
"""

VALID_REFENTRY = """<refentry id="sql-createtable">
 <refmeta>
  <refentrytitle>CREATE TABLE</refentrytitle>
  <manvolnum>7</manvolnum>
  <refmiscinfo>SQL - Language Statements</refmiscinfo>
 </refmeta>
 <refnamediv>
  <refname>CREATE TABLE</refname>
  <refpurpose>define a new table</refpurpose>
 </refnamediv>
 <refsect1>
  <title>Description</title>
  <para>CREATE TABLE will create a new, initially empty table in the
  current database. It is owned by the user issuing the command.</para>
 </refsect1>
</refentry>
"""


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A two-document corpus with a ref/ subdirectory, like the real one."""
    (tmp_path / "ref").mkdir()
    (tmp_path / "sample.sgml").write_text(VALID_CHAPTER, encoding="utf-8")
    (tmp_path / "ref" / "create_table.sgml").write_text(
        VALID_REFENTRY, encoding="utf-8"
    )
    return tmp_path


# --- The happy path --------------------------------------------------------


def test_loads_documents_from_nested_directories(corpus: Path) -> None:
    # The 219 SQL command pages live under ref/, so a non-recursive walk would
    # silently miss more than half the corpus.
    docs = load_corpus(corpus)
    assert {d.path for d in docs} == {"sample.sgml", "ref/create_table.sgml"}


def test_paths_are_relative_and_posix_style(corpus: Path) -> None:
    # gold_doc_paths in the question set are matched against these strings, so
    # they must not vary with the checkout location or the host OS separator.
    docs = load_corpus(corpus)
    ref = next(d for d in docs if d.path.startswith("ref/"))
    assert ref.path == "ref/create_table.sgml"
    assert "\\" not in ref.path


def test_documents_are_returned_in_sorted_order(tmp_path: Path) -> None:
    # Stable ordering is what makes chunk IDs reproducible across runs, which
    # is what lets committed results be compared run to run.
    for name in ("zeta", "alpha", "mid"):
        (tmp_path / f"{name}.sgml").write_text(VALID_CHAPTER, encoding="utf-8")
    paths = [d.path for d in load_corpus(tmp_path)]
    assert paths == sorted(paths)


# --- Titles ----------------------------------------------------------------


def test_chapter_title_comes_from_title_element(corpus: Path) -> None:
    doc = next(d for d in load_corpus(corpus) if d.path == "sample.sgml")
    assert doc.title == "Server Configuration"


def test_refentry_title_comes_from_refentrytitle(corpus: Path) -> None:
    # A <refentry> has no direct <title> child. Reading only <title> would name
    # all 220 command pages after their filenames: "Create_Table".
    doc = next(d for d in load_corpus(corpus) if d.path.endswith("create_table.sgml"))
    assert doc.title == "CREATE TABLE"


# --- Content extraction ----------------------------------------------------


def test_code_block_whitespace_is_preserved(corpus: Path) -> None:
    # Indentation is meaning in a SQL sample. If it is collapsed, a retrieved
    # chunk shows the user an unusable example.
    doc = next(d for d in load_corpus(corpus) if d.path == "sample.sgml")
    assert "SET work_mem TO '64MB';" in doc.content
    assert "    SELECT pg_reload_conf();" in doc.content


def test_prose_line_wrapping_is_collapsed(corpus: Path) -> None:
    # DocBook hard-wraps source lines at ~70 columns. Those newlines are a file
    # format artifact, not sentence structure.
    doc = next(d for d in load_corpus(corpus) if d.path == "sample.sgml")
    assert "affect the behavior of the database system." in doc.content


def test_custom_entities_are_substituted(corpus: Path) -> None:
    doc = next(d for d in load_corpus(corpus) if d.path == "sample.sgml")
    assert "—" in doc.content  # &mdash; became an em-dash
    assert "&mdash;" not in doc.content  # and no raw entity survived


def test_indexterms_are_dropped(corpus: Path) -> None:
    # Index terms repeat words already in the prose. Leaving them in inflates
    # BM25 term frequencies for words that were never really written twice.
    doc = next(d for d in load_corpus(corpus) if d.path == "sample.sgml")
    assert doc.content.count("configuration") == 1


def test_refmeta_is_dropped(corpus: Path) -> None:
    # "CREATE TABLE 7 SQL - Language Statements" is man-page bookkeeping.
    doc = next(d for d in load_corpus(corpus) if d.path.endswith("create_table.sgml"))
    assert "manvolnum" not in doc.content
    assert "SQL - Language Statements" not in doc.content


def test_heading_text_is_kept_in_content(corpus: Path) -> None:
    # Phase 2 prepends heading paths to chunks; the headings must survive here.
    doc = next(d for d in load_corpus(corpus) if d.path == "sample.sgml")
    assert "Setting Parameters" in doc.content
    assert "Parameter Names and Values" in doc.content


def test_raw_xml_is_retained_for_phase_two(corpus: Path) -> None:
    # The chunker needs the element tree to find section and atomic-block
    # boundaries; that structure cannot be recovered from flattened text.
    doc = next(d for d in load_corpus(corpus) if d.path == "sample.sgml")
    assert doc.raw_xml is not None
    assert len(doc.raw_xml.findall(".//sect2")) == 1


# --- Bad input: the loader must degrade, not crash -------------------------


def test_missing_corpus_directory_raises_actionable_error(tmp_path: Path) -> None:
    # This one *should* raise: a missing corpus is user error, and the message
    # has to say how to fix it.
    with pytest.raises(FileNotFoundError, match="download.sh"):
        load_corpus(tmp_path / "does-not-exist")


def test_empty_file_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "empty.sgml").write_text("", encoding="utf-8")
    (tmp_path / "whitespace.sgml").write_text("\n\n   \t\n", encoding="utf-8")
    assert load_corpus(tmp_path) == []


def test_malformed_markup_does_not_abort_the_ingest(tmp_path: Path) -> None:
    # The failure that actually matters: one unparseable file must not take the
    # rest of the corpus down with it.
    (tmp_path / "a-broken.sgml").write_text(
        "<chapter><title>Broken<para>unclosed everything", encoding="utf-8"
    )
    (tmp_path / "b-garbage.sgml").write_text(
        "\x00\x01 not markup at all >>><<<", encoding="utf-8"
    )
    (tmp_path / "c-good.sgml").write_text(VALID_CHAPTER, encoding="utf-8")

    docs = load_corpus(tmp_path)
    assert "c-good.sgml" in {d.path for d in docs}


def test_unknown_entity_does_not_break_parsing(tmp_path: Path) -> None:
    # A future PostgreSQL release may add an entity we have not mapped. It
    # should degrade to its name, not kill the document.
    (tmp_path / "x.sgml").write_text(
        "<chapter><title>T</title><para>See &someNewEntity; for details. "
        "This paragraph is padded so it clears the minimum content length "
        "check that filters out stub documents.</para></chapter>",
        encoding="utf-8",
    )
    docs = load_corpus(tmp_path)
    assert len(docs) == 1
    assert "&someNewEntity;" not in docs[0].content


def test_non_document_fragments_are_skipped(tmp_path: Path) -> None:
    # version.sgml and friends are build inputs, not documentation.
    (tmp_path / "version.sgml").write_text("<date>2024-09-26</date>", encoding="utf-8")
    assert load_corpus(tmp_path) == []


def test_stub_documents_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "stub.sgml").write_text(
        "<chapter><title>TODO</title></chapter>", encoding="utf-8"
    )
    assert load_corpus(tmp_path) == []


def test_load_document_returns_none_for_unreadable_path(tmp_path: Path) -> None:
    assert load_document(tmp_path / "nope.sgml", tmp_path) is None


# --- Classification --------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "root_tag", "expected"),
    [
        # Structure beats naming: every command page is a <refentry>.
        ("ref/create_table.sgml", "refentry", "sql_command"),
        ("ref/vacuum.sgml", "refentry", "sql_command"),
        ("config.sgml", "chapter", "config"),
        ("runtime-config-resource.sgml", "sect1", "config"),
        ("func.sgml", "chapter", "guide"),
        ("functions-string.sgml", "sect1", "function_ref"),
        ("catalogs.sgml", "chapter", "catalog"),
        ("backup.sgml", "chapter", "guide"),
        ("high-availability.sgml", "chapter", "guide"),
        # A `pg`-prefixed contrib chapter is not a SQL command, which a naming
        # rule of startswith("pg") got wrong.
        ("pgbuffercache.sgml", "chapter", "extension"),
        ("pgcrypto.sgml", "chapter", "extension"),
    ],
)
def test_classify(filename: str, root_tag: str, expected: str) -> None:
    assert classify(Path(filename), root_tag) == expected


def test_doc_types_are_assigned_on_load(corpus: Path) -> None:
    by_path = {d.path: d.doc_type for d in load_corpus(corpus)}
    assert by_path["ref/create_table.sgml"] == "sql_command"


# --- Document -------------------------------------------------------------


def test_token_estimate_scales_with_content() -> None:
    doc = Document(path="p", title="t", content="x" * 4000, doc_type="guide")
    assert doc.token_estimate == 1000


def test_documents_compare_without_regard_to_parsed_tree(corpus: Path) -> None:
    # raw_xml holds lxml objects that do not compare by value; excluding it from
    # equality keeps Documents usable in assertions and sets.
    a = Document(path="p", title="t", content="c", doc_type="guide")
    b = Document(path="p", title="t", content="c", doc_type="guide")
    assert a == b
