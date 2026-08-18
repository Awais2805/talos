"""LakeClient IO: the seam every stage below labelling reads and writes through.

These run against a local lake only. The remote backend's job is to make the
same calls mean the same thing over S3, and the property that keeps that honest
is that no caller anywhere reaches past `LakeClient` into a backend's internals
-- which is asserted here as a source-level check, because it cannot be tested
by exercising the local path.
"""

import subprocess
import tempfile
from pathlib import Path

import duckdb
import pytest

from talos.common.lake.lake import LakeClient

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def lake(tmp_path):
    return LakeClient(root=str(tmp_path / "lake"))


def relation(rows):
    values = ", ".join(f"('{u}', {t})" for u, t in rows)
    return duckdb.connect().sql(f"SELECT * FROM (VALUES {values}) AS t(uid, ts)")


# --------------------------------------------------------------- parquet io

def test_write_then_read_round_trips(lake):
    uri = lake.uri("parquet", dataset="ds", feature_space="zeek_v8.2.1",
                   rel="conn.parquet")
    lake.write_parquet(relation([("a", 1), ("b", 2)]), uri)

    assert Path(uri).exists()                        # parents created for us
    assert sorted(lake.read_parquet(uri).fetchall()) == [("a", 1), ("b", 2)]


def test_read_parquet_returns_a_relation_not_rows(lake):
    """Labelling passes 2.1M rows through this; materialising them would be fatal."""
    uri = lake.uri("parquet", dataset="ds", feature_space="fs", rel="conn.parquet")
    lake.write_parquet(relation([("a", 1), ("b", 2)]), uri)

    rel = lake.read_parquet(uri)
    assert isinstance(rel, duckdb.DuckDBPyRelation)
    assert rel.limit(1).fetchall() == [("a", 1)]     # composable, not a list


def test_read_parquet_can_project_columns(lake):
    uri = lake.uri("parquet", dataset="ds", feature_space="fs", rel="conn.parquet")
    lake.write_parquet(relation([("a", 1)]), uri)

    rel = lake.read_parquet(uri, columns=["uid"])
    assert rel.columns == ["uid"]


def test_read_parquet_unions_by_name_across_a_glob(lake):
    """A capture that never saw a field has no column for it; alignment is by name."""
    con = duckdb.connect()
    base = Path(lake.uri("parquet", dataset="ds", feature_space="fs"))
    lake.write_parquet(con.sql("SELECT 'a' AS uid, 1 AS ts"), str(base / "c1/conn.parquet"))
    lake.write_parquet(con.sql("SELECT 'b' AS uid, 2 AS ts, 'dns' AS service"),
                       str(base / "c2/conn.parquet"))

    rel = lake.read_parquet(f"{base}/**/*.parquet")
    assert set(rel.columns) == {"uid", "ts", "service"}
    assert len(rel.fetchall()) == 2


def test_write_parquet_does_not_open_a_second_connection(lake):
    """The write happens on the relation's connection, not a fresh one.

    That connection is the one the caller sized and authenticated. An earlier
    version routed COPY through the client's own engine, which both failed
    across connections and would have written under settings nobody chose.
    """
    uri = lake.uri("parquet", dataset="ds", feature_space="fs", rel="conn.parquet")
    lake.write_parquet(relation([("a", 1)]), uri)

    assert Path(uri).exists()
    assert lake._engine is None            # nothing was constructed to do the write


# ------------------------------------------------------------- object io

def test_read_text(lake):
    uri = lake.uri("extracted", dataset="ds", feature_space="fs",
                   rel="_extractor_meta.json")
    Path(uri).parent.mkdir(parents=True, exist_ok=True)
    Path(uri).write_text('{"feature_space": "zeek_v8.2.1"}')
    assert "zeek_v8.2.1" in lake.read_text(uri)


def test_get_copies_to_a_local_path(lake, tmp_path):
    uri = lake.uri("raw", dataset="ds", rel="monday.pcap")
    Path(uri).parent.mkdir(parents=True, exist_ok=True)
    Path(uri).write_bytes(b"\xd4\xc3\xb2\xa1")

    out = lake.get(uri, tmp_path / "scratch" / "monday.pcap")
    assert out.read_bytes() == b"\xd4\xc3\xb2\xa1"   # parents created


def test_put_tree_merges_a_directory(lake, tmp_path):
    src = tmp_path / "logs"
    (src / "c1").mkdir(parents=True)
    (src / "c1" / "conn.log").write_text("{}\n")
    (src / "_extractor_meta.json").write_text("{}")

    uri = lake.uri("extracted", dataset="ds", feature_space="fs")
    lake.put_tree(src, uri)

    assert (Path(uri) / "c1" / "conn.log").exists()
    assert (Path(uri) / "_extractor_meta.json").exists()


# ------------------------------------------------------------ the abstraction

def test_engine_is_lazy_and_injectable(tmp_path):
    """Constructing an engine resolves credentials; a command that only lists must not."""
    client = LakeClient(root=str(tmp_path))
    assert client._engine is None

    sentinel = object()
    assert LakeClient(root=str(tmp_path), engine=sentinel).duck is sentinel


def test_no_caller_reaches_past_lakeclient_into_a_backend():
    """`LakeClient` is the only class that touches storage -- enforced, not hoped.

    Backend internals (`.fs`) leaking into callers is how a second storage
    backend stops being possible, and it is why these IO methods exist at all.
    """
    hits = subprocess.run(
        ["grep", "-rn", "backend.fs", str(REPO / "src")],
        capture_output=True, text=True).stdout.strip()
    assert hits == "", f"callers reaching into a backend:\n{hits}"
