"""Ingestion: the only moment provenance can be established.

Once a pcap is in the lake there is no way to tell, retrospectively, whether it
is the file that was ingested. Everything here is about recording that while it
is still knowable — and about the declared UTC offset, which is the one thing
that would have prevented the open question blocking CIC-IDS-2017's labels.
"""

import json
import os
import stat
from pathlib import Path

import pytest

from talos.common.config import Config
from talos.data.ingestion.ingest import (
    MANIFEST, IngestionError, IngestionService,
)


@pytest.fixture
def lab(tmp_path):
    cfg = Config({"lake": {"root": str(tmp_path / "lake")},
                  "datasets": {"run-01": {"source": "netem"},
                               "cic-ids-2017": {"source": "datasets"}}})
    captures = tmp_path / "captures"
    captures.mkdir()

    def pcap(name, body=b"\xd4\xc3\xb2\xa1payload"):
        path = captures / name
        path.write_bytes(body)
        return str(path)

    return IngestionService(cfg), pcap


def manifest_of(service, dataset, source):
    return json.loads(service.lake.read_text(service.manifest_uri(dataset, source)))


# ------------------------------------------------------------------- routing

def test_captures_land_in_their_source_tree(lab):
    service, pcap = lab
    report = service.ingest("run-01", [pcap("monday.pcap")])

    assert report.source == "netem"                     # from the declaration
    assert "sources/netem/raw/run-01" in report.destination
    assert Path(report.destination, "monday.pcap").exists()


def test_an_explicit_source_overrides_the_declaration(lab):
    service, pcap = lab
    report = service.ingest("run-01", [pcap("a.pcap")], source="honeypot")
    assert "sources/honeypot/raw" in report.destination


def test_an_unknown_source_is_refused(lab):
    service, pcap = lab
    with pytest.raises(IngestionError, match="sensornet"):
        service.ingest("run-01", [pcap("a.pcap")], source="sensornet")


def test_a_missing_file_is_refused_before_anything_is_written(lab):
    service, pcap = lab
    with pytest.raises(IngestionError, match="not found"):
        service.ingest("run-01", [pcap("a.pcap"), "/nonexistent/b.pcap"])
    assert not Path(service.lake.uri("raw", dataset="run-01", source="netem"),
                    "a.pcap").exists()


# ---------------------------------------------------------------- provenance

def test_every_capture_is_recorded_with_size_and_hash(lab):
    service, pcap = lab
    service.ingest("run-01", [pcap("monday.pcap")])
    entry = manifest_of(service, "run-01", "netem")["captures"][0]

    assert entry["file"] == "monday.pcap"
    assert entry["bytes"] > 0
    assert len(entry["sha256_12"]) == 12
    assert entry["modified"] and entry["ingested_at"]


def test_hashing_can_be_skipped_but_size_cannot(lab):
    """Size and mtime are free and catch the accident; the hash catches the
    deliberate and is what makes a later verify possible at all."""
    service, pcap = lab
    service.ingest("run-01", [pcap("a.pcap")], hash_contents=False)
    entry = manifest_of(service, "run-01", "netem")["captures"][0]
    assert "sha256_12" not in entry and entry["bytes"] > 0


def test_captures_accumulate_across_ingests(lab):
    """The zone is the sum of every ingest into it, not the last one."""
    service, pcap = lab
    service.ingest("run-01", [pcap("a.pcap")])
    service.ingest("run-01", [pcap("b.pcap")])

    meta = manifest_of(service, "run-01", "netem")
    assert [c["file"] for c in meta["captures"]] == ["a.pcap", "b.pcap"]
    assert meta["ingests"] == 2


def test_an_identical_recapture_is_skipped_not_rewritten(lab):
    service, pcap = lab
    service.ingest("run-01", [pcap("a.pcap")])
    report = service.ingest("run-01", [pcap("a.pcap")])
    assert report.skipped == ["a.pcap"] and report.ingested == []


def test_replacing_a_capture_is_allowed_but_recorded(lab):
    """Overwriting an immutable zone must never be silent: provenance
    downstream assumed the old bytes."""
    service, pcap = lab
    service.ingest("run-01", [pcap("a.pcap", b"first")])
    report = service.ingest("run-01", [pcap("a.pcap", b"second and longer")])
    assert report.replaced == ["a.pcap"]


# ------------------------------------------------------------- the offset

def test_a_declared_offset_is_recorded(lab):
    service, pcap = lab
    service.ingest("run-01", [pcap("a.pcap")], utc_offset="-03:00")
    assert manifest_of(service, "run-01", "netem")["utc_offset"] == "-03:00"


def test_an_undeclared_offset_is_null_not_assumed(lab):
    """None means 'not established', which is materially different from an
    offset somebody assumed. CIC-IDS-2017 is the cautionary case."""
    service, pcap = lab
    service.ingest("cic-ids-2017", [pcap("a.pcap")])
    assert manifest_of(service, "cic-ids-2017", "datasets")["utc_offset"] is None


def test_a_declared_offset_survives_a_later_ingest_that_omits_it(lab):
    service, pcap = lab
    service.ingest("run-01", [pcap("a.pcap")], utc_offset="-03:00")
    service.ingest("run-01", [pcap("b.pcap")])
    assert manifest_of(service, "run-01", "netem")["utc_offset"] == "-03:00"


def test_a_conflicting_offset_is_refused(lab):
    """One of them is wrong, and every label from this capture depends on which."""
    service, pcap = lab
    service.ingest("run-01", [pcap("a.pcap")], utc_offset="-03:00")
    with pytest.raises(IngestionError, match="-04:00"):
        service.ingest("run-01", [pcap("b.pcap")], utc_offset="-04:00")


def test_an_unparseable_offset_is_refused_at_ingest(lab):
    """Rather than at labelling time, hours into a run."""
    service, pcap = lab
    with pytest.raises(IngestionError):
        service.ingest("run-01", [pcap("a.pcap")], utc_offset="Atlantic")


# ---------------------------------------------------------------- sealing

def test_the_zone_is_sealed_after_ingest(lab):
    """Write-once is an obligation, not a property, on a filesystem."""
    service, pcap = lab
    report = service.ingest("run-01", [pcap("a.pcap")])

    assert report.sealed
    mode = Path(report.destination, "a.pcap").stat().st_mode
    assert not mode & stat.S_IWUSR


def test_a_later_ingest_unseals_writes_and_reseals(lab):
    service, pcap = lab
    service.ingest("run-01", [pcap("a.pcap")])
    report = service.ingest("run-01", [pcap("b.pcap")])

    assert report.ingested == ["b.pcap"] and report.sealed
    assert not Path(report.destination, "b.pcap").stat().st_mode & stat.S_IWUSR


def test_sealing_can_be_declined(lab):
    service, pcap = lab
    report = service.ingest("run-01", [pcap("a.pcap")], seal=False)
    assert not report.sealed
    assert Path(report.destination, "a.pcap").stat().st_mode & stat.S_IWUSR


def test_the_manifest_is_readable_after_sealing(lab):
    """It is written before the seal, and downstream has to read it."""
    service, pcap = lab
    service.ingest("run-01", [pcap("a.pcap")])
    assert manifest_of(service, "run-01", "netem")["captures"]


def test_extraction_can_still_list_a_sealed_zone(lab):
    """Sealing drops the write bit, not the read bit."""
    service, pcap = lab
    report = service.ingest("run-01", [pcap("a.pcap")])
    found = service.lake.list(report.destination)
    assert any(f.endswith("a.pcap") for f in found)
    assert any(f.endswith(MANIFEST) for f in found)
