"""Provenance: the hashes downstream staleness detection rests on.

The property that matters is that the hash changes when and only when the content does,
and that a stamped row carries exactly what the stage said it did.
"""

import json
import tempfile
from pathlib import Path

import duckdb
import pytest

from talos.common.provenance import (
    ProvenanceError, ProvMeta, ProvenanceService, content_sha, digest, stamp_sql,
)


def test_digest_is_twelve_hex_chars():
    sha = digest(b"talos")
    assert len(sha) == 12
    assert set(sha) <= set("0123456789abcdef")


def test_content_sha_matches_digest_of_the_same_bytes():
    """Streaming a file must give the same answer as hashing its bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "manifest.yaml"
        p.write_bytes(b"attacks: []\n")
        assert content_sha(p) == digest(b"attacks: []\n")


def test_content_sha_survives_a_file_larger_than_one_chunk():
    """The chunked read must not depend on the file fitting in one buffer."""
    payload = b"x" * (3 * (1 << 20) + 17)        # 3 MiB + change
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "big.pcap"
        p.write_bytes(payload)
        assert content_sha(p) == digest(payload)


def test_a_one_byte_change_changes_the_hash():
    """The whole staleness argument collapses without this."""
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        a.write_text("start: '09:20'")
        b.write_text("start: '09:21'")
        assert content_sha(a) != content_sha(b)


def test_verify_detects_an_edited_file():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "taxonomy.yaml"
        p.write_text("dos: dos")
        sha = content_sha(p)
        assert ProvenanceService.verify(p, sha)
        p.write_text("dos: ddos")                # the kind of edit that must not go unnoticed
        assert not ProvenanceService.verify(p, sha)


# ------------------------------------------------------------------ stamping

def test_stamp_sql_orders_columns_as_declared():
    """Column order is deterministic, so a re-run diffs as unchanged."""
    meta = ProvMeta.of(dataset="cic-ids-2017", manifest_sha="abc", taxonomy_sha="def")
    assert stamp_sql(meta) == (
        "'cic-ids-2017' AS dataset, 'abc' AS manifest_sha, 'def' AS taxonomy_sha")


def test_stamp_sql_escapes_quotes():
    """Quarantine reasons are free text and reach this as SQL literals."""
    meta = ProvMeta.of(reason="attacker's tail")
    assert stamp_sql(meta) == "'attacker''s tail' AS reason"


def test_stamp_sql_handles_none_and_numbers():
    meta = ProvMeta.of(rule_id=3, quarantine_reason=None, flagged=False)
    assert stamp_sql(meta) == (
        "3 AS rule_id, CAST(NULL AS VARCHAR) AS quarantine_reason, false AS flagged")


def test_a_column_name_that_is_not_an_identifier_is_refused():
    """These names are emitted as SQL, so the check belongs at construction."""
    with pytest.raises(ProvenanceError):
        ProvMeta.of(**{"drop table; --": "x"})
    with pytest.raises(ProvenanceError):
        ProvMeta.of(**{"2019_sha": "x"})


def test_stamp_appends_the_columns_to_every_row():
    con = duckdb.connect()
    rel = con.sql("SELECT * FROM (VALUES ('a', 1), ('b', 2)) AS t(uid, ts)")
    meta = ProvMeta.of(dataset="cic-ids-2017", manifest_sha="abc123def456")

    out = ProvenanceService.stamp(rel, meta)

    assert out.columns == ["uid", "ts", "dataset", "manifest_sha"]
    rows = out.fetchall()
    assert rows == [("a", 1, "cic-ids-2017", "abc123def456"),
                    ("b", 2, "cic-ids-2017", "abc123def456")]


def test_stamp_does_not_open_a_connection():
    """It must stay on the caller's connection, whose limits are the tuned ones."""
    con = duckdb.connect()
    con.execute("SET memory_limit = '256MB'")
    rel = con.sql("SELECT 1 AS uid")
    out = ProvenanceService.stamp(rel, ProvMeta.of(dataset="d"))
    assert out.fetchall() == [(1, "d")]


# ------------------------------------------------------------------- reports

def test_run_report_writes_json_and_creates_its_directory():
    with tempfile.TemporaryDirectory() as tmp:
        prov = ProvenanceService(reports=Path(tmp) / "nested" / "reports")
        path = prov.run_report("labelling_cic-ids-2017", {"rules_fired": 18, "attack": 533336})
        assert path.exists()
        assert json.loads(path.read_text()) == {"rules_fired": 18, "attack": 533336}
