"""Conversion: extracted logs to parquet and CSV, one read, correct zones.
"""

import json
from pathlib import Path

import pytest

from talos.common.config import Config
from talos.data.conversion.convert import ConversionError, Converter
from talos.data.extraction.base import META_FILENAME


@pytest.fixture
def lab(tmp_path):
    cfg = Config({"lake": {"root": str(tmp_path / "lake")},
                  "datasets": {"run-01": {"source": "netem"},
                               "toy": {"source": "datasets"}}})
    conv = Converter(cfg)

    def extracted(dataset, feature_space="zeek_v8.2.1", source=None,
                  complete=True, when="2026-08-11T10:00:00+00:00", captures=("Monday",)):
        source = source or cfg.source_of(dataset)
        base = Path(conv.lake.uri("extracted", dataset=dataset, source=source,
                                  feature_space=feature_space))
        for capture in captures:
            out = base / capture
            out.mkdir(parents=True, exist_ok=True)
            (out / "conn.log").write_text(
                '{"uid":"C1","ts":1.0,"proto":"tcp"}\n'
                '{"uid":"C2","ts":2.0,"proto":"udp"}\n')
            (out / "dns.log").write_text('{"uid":"C1","query":"example.com"}\n')
        base.mkdir(parents=True, exist_ok=True)
        (base / META_FILENAME).write_text(json.dumps({
            "dataset": dataset, "feature_space": feature_space,
            "extracted_at_utc": when, "complete": complete,
            "pcaps_failed": 0 if complete else 3}))
        return str(base)

    return conv, extracted


# ------------------------------------------------------------------- zones

def test_it_reads_and_writes_the_dataset_s_own_source_tree(lab):
    conv, extracted = lab
    extracted("run-01")                                  # netem
    report = conv.convert("run-01")

    assert report.source == "netem"
    assert "sources/netem/extracted" in report.src
    assert "sources/netem/parquet" in report.destinations["parquet"]


def test_a_dataset_in_another_source_is_not_found(lab):
    """The old code resolved every zone as `datasets`, so a netem dataset was
    converted from the public-corpus tree, or not found at all."""
    conv, extracted = lab
    extracted("run-01", source="netem")
    with pytest.raises(ConversionError, match="no extraction"):
        conv.convert("run-01", source="honeypot")


def test_the_tree_is_mirrored_not_flattened(lab):
    """Reshaping here would destroy the uid backbone and the capture folder."""
    conv, extracted = lab
    extracted("toy", captures=("Monday", "Tuesday"))
    report = conv.convert("toy")

    out = Path(report.destinations["parquet"])
    assert (out / "Monday" / "conn.parquet").exists()
    assert (out / "Tuesday" / "dns.parquet").exists()


def test_the_provenance_sidecar_is_not_converted_as_data(lab):
    conv, extracted = lab
    extracted("toy")
    report = conv.convert("toy")

    assert report.converted == 2                          # conn + dns, not the sidecar
    assert not (Path(report.destinations["parquet"]) / "_extractor_meta.parquet").exists()


# ------------------------------------------------------------- both formats

def test_both_formats_come_from_one_read(lab):
    conv, extracted = lab
    extracted("toy")
    report = conv.convert("toy", formats=("parquet", "csv"))

    parquet = Path(report.destinations["parquet"]) / "Monday" / "conn.parquet"
    csv = Path(report.destinations["csv"]) / "Monday" / "conn.csv"
    assert parquet.exists() and csv.exists()
    assert "uid,ts,proto" in csv.read_text()


def test_csv_lands_in_a_zone_not_a_hardcoded_path(lab):
    conv, extracted = lab
    extracted("run-01")
    report = conv.convert("run-01", formats=("csv",))
    assert "sources/netem/csv/zeek_v8.2.1/run-01" in report.destinations["csv"]


def test_an_unknown_format_is_refused(lab):
    conv, extracted = lab
    extracted("toy")
    with pytest.raises(ConversionError, match="avro"):
        conv.convert("toy", formats=("avro",))


# --------------------------------------------------- choosing an extraction

def test_the_most_recent_complete_extraction_wins(lab):
    conv, extracted = lab
    extracted("toy", feature_space="zeek_v8.2.1", when="2026-08-01T00:00:00+00:00")
    extracted("toy", feature_space="zeek_v9.0.0", when="2026-08-11T00:00:00+00:00")
    assert conv.convert("toy").feature_space == "zeek_v9.0.0"


def test_a_newer_incomplete_extraction_does_not_beat_an_older_complete_one(lab):
    """Converting a tree that is missing captures produces a parquet zone
    quietly short of flows, which then looks clean to everything downstream."""
    conv, extracted = lab
    extracted("toy", feature_space="zeek_v8.2.1", when="2026-08-01T00:00:00+00:00")
    extracted("toy", feature_space="zeek_v9.0.0", when="2026-08-11T00:00:00+00:00",
              complete=False)
    assert conv.convert("toy").feature_space == "zeek_v8.2.1"


def test_only_an_incomplete_extraction_is_refused_with_a_way_forward(lab):
    conv, extracted = lab
    extracted("toy", complete=False)
    with pytest.raises(ConversionError, match="3 capture"):
        conv.convert("toy")


def test_an_incomplete_extraction_can_be_converted_knowingly(lab):
    conv, extracted = lab
    extracted("toy", complete=False)
    assert conv.convert("toy", allow_incomplete=True).converted == 2


def test_an_explicit_feature_space_bypasses_resolution(lab):
    conv, extracted = lab
    extracted("toy", feature_space="zeek_v7", complete=False)
    assert conv.convert("toy", feature_space="zeek_v7").feature_space == "zeek_v7"


def test_no_extraction_at_all_says_what_to_run(lab):
    conv, _ = lab
    with pytest.raises(ConversionError, match="talos extract"):
        conv.convert("toy")


# ----------------------------------------------------------------- failures

def test_one_bad_log_does_not_abandon_the_rest(lab):
    conv, extracted = lab
    base = Path(extracted("toy"))
    (base / "Monday" / "broken.log").write_text("not json at all\x00\x01")

    report = conv.convert("toy", logtypes=["conn", "broken"])
    assert report.converted >= 1
    assert report.rows > 0


def test_logtypes_narrow_the_selection(lab):
    conv, extracted = lab
    extracted("toy")
    assert conv.convert("toy", logtypes=["conn"]).converted == 1


def test_rows_are_counted_from_the_write_not_a_re_read(lab):
    """The previous version re-read every output with count(*) -- a second full
    round trip per file, thousands of times."""
    conv, extracted = lab
    extracted("toy")
    assert conv.convert("toy", logtypes=["conn"]).rows == 2
