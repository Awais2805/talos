"""The whole engine, end to end, against a synthetic lake.

A purpose-built manifest and eleven flows chosen so that every branch of the
output is exercised at once: a matched attack, an attempt, a bystander inside
the window, a flow outside it, a reflection match, an overlap, and a
quarantined span. Small enough to reason about by hand, which is the point —
the real 2017 run has 2.1M flows and no way to check a specific row.
"""

import textwrap

import duckdb
import pytest

from talos.common.config import Config
from talos.common.validation import GateFailure
from talos.data.labelling.schedule.engine import LabellingEngine
from talos.data.labelling.base import CORE_COLUMNS, SchemaError, StaleOutputError
from talos.data.labelling.schedule.writer import EXTRAS

from talos.data.labelling.schedule.windows import to_utc

# Derived, not hardcoded. Hand-computed epochs put every flow an hour out on the
# first attempt; `to_utc` is independently tested in test_manifest.py, so using
# it here removes a whole class of arithmetic error from the fixture.
def at(hhmm: str, offset_seconds: float = 0.0) -> float:
    return to_utc("2017-07-04", hhmm, "-03:00") + offset_seconds

FTP_START, FTP_END = at("09:20"), at("10:20")

MANIFEST = textwrap.dedent("""\
    dataset: toy
    utc_offset: "-03:00"
    match_default: [time, attacker, victim]
    attacks:
      - {name: FTP-Patator, date: 2017-07-04, start: "09:20", end: "10:20",
         attacker_ips: [10.0.0.1, 172.16.0.1], victim_ips: [10.0.0.2]}
      - {name: PortScan, date: 2017-07-04, start: "11:00", end: "11:10",
         attacker_ips: [10.0.0.1], victim_ips: [10.0.0.2]}
""")

# A rule that fires on nothing, appended at the `attacks:` indent level.
SILENT_RULE = ("""  - {name: Heartbleed, date: 2017-07-04, start: "20:00", end: "20:10",\n"""
               """     attacker_ips: [10.0.0.1], victim_ips: [10.0.0.2]}\n""")

QUARANTINED_MANIFEST = MANIFEST + textwrap.dedent("""\
    quarantine:
      - {reason: unadjudicable-tail, justification: "attack plainly continues",
         date: 2017-07-04, start: "13:00", end: "13:10", victim_ips: [10.0.0.2]}
""")

# (uid, ts, orig, resp, orig_bytes) -- ts values are UTC epoch seconds.
FLOWS = [
    ("attack_exec", at("09:20", 10), "10.0.0.1", "10.0.0.2", 500),    # payload sent
    ("attack_reply", at("09:20", 11), "10.0.0.2", "10.0.0.1", 400),   # bidirectional
    ("attack_nat", at("09:20", 12), "172.16.0.1", "10.0.0.2", 300),   # alias
    ("attempt", at("09:20", 13), "10.0.0.1", "10.0.0.2", 0),          # empty -> attempted
    ("unmeasured", at("09:20", 14), "10.0.0.1", "10.0.0.2", None),    # NULL != zero
    ("bystander", at("09:20", 15), "10.9.9.9", "10.0.0.2", 900),      # in window, not a peer
    ("before", at("09:20", -60), "10.0.0.1", "10.0.0.2", 100),
    ("after", at("10:21", 60), "10.0.0.1", "10.0.0.2", 100),          # past the extension
    ("scan", at("11:00", 30), "10.0.0.1", "10.0.0.2", 0),             # exempt from payload
    ("quiet", at("13:00", 30), "10.5.5.5", "10.0.0.2", 50),           # quarantine span
]


@pytest.fixture
def lab(tmp_path):
    """A config, a lake with one capture of conn flows, and a manifests dir."""
    def build(manifest_text=MANIFEST, flows=FLOWS):
        manifests = tmp_path / "manifests"
        manifests.mkdir(exist_ok=True)
        (manifests / "toy.yaml").write_text(manifest_text)
        # The taxonomy is no longer copied in beside the manifests: it is its own
        # plug-in point now, resolved by name from `labelling/taxonomies/`, so a
        # manifests directory holds only manifests.

        # Source-major, as `talos init` lays it out. `toy` declares no source,
        # so it defaults to `datasets`.
        lake_root = tmp_path / "lake"
        out = (lake_root / "sources" / "datasets" / "parquet" / "zeek_v8.2.1"
               / "toy" / "Tuesday-WorkingHours")
        out.mkdir(parents=True, exist_ok=True)
        rows = ", ".join(
            f"('{u}', {ts!r}, '{o}', '{r}', {'NULL' if b is None else b})"
            for u, ts, o, r, b in flows)
        duckdb.connect().sql(
            f'SELECT * FROM (VALUES {rows}) AS t(uid, ts, "id.orig_h", "id.resp_h", orig_bytes)'
        ).write_parquet(str(out / "conn.parquet"))

        cfg = Config({"lake": {"root": str(lake_root)},
                      "reports": {"dir": str(tmp_path / "reports")}})
        return LabellingEngine(cfg, manifests=manifests)
    return build


def labelled(engine, report):
    return engine.lake.read_parquet(report.output)


def rows_by_uid(rel):
    cols = rel.columns
    return {r[cols.index("uid")]: dict(zip(cols, r)) for r in rel.fetchall()}


# ------------------------------------------------------------------ the run

def test_the_run_produces_the_expected_counts(lab):
    engine = lab()
    report = engine.label("toy", "zeek_v8.2.1")

    assert report.total_flows == 10
    # 5 in the FTP window (exec, reply, nat, attempt, unmeasured) + 1 scan
    assert report.attack_flows == 6
    assert report.benign_flows == 4
    assert report.silent_rules == ()


def test_every_original_column_survives_and_the_core_plus_extras_are_added(lab):
    """Core is the contract every method owes; extras are schedule's own."""
    engine = lab()
    rel = labelled(engine, engine.label("toy", "zeek_v8.2.1"))

    for original in ("uid", "ts", "id.orig_h", "id.resp_h", "orig_bytes"):
        assert original in rel.columns
    for added in CORE_COLUMNS + EXTRAS:
        assert added in rel.columns
    assert len(rel.columns) == 5 + len(CORE_COLUMNS) + len(EXTRAS)


def test_confidence_and_weight_go_out_as_typed_nulls_not_as_one(lab):
    """Schedule labelling has no calibrated confidence. Writing 1.0 would invent
    a number that confident learning then consumes as evidence."""
    engine = lab()
    rel = labelled(engine, engine.label("toy", "zeek_v8.2.1"))
    types = dict(zip(rel.columns, (str(t) for t in rel.types)))

    assert types["label_confidence"] == "DOUBLE"
    assert types["label_weight"] == "DOUBLE"
    assert rel.filter("label_confidence IS NOT NULL").count("*").fetchone()[0] == 0
    assert rel.filter("label_weight IS NOT NULL").count("*").fetchone()[0] == 0


def test_benign_is_written_not_reconstructed_by_anti_join(lab):
    """The output is the whole dataset, so downstream never rebuilds benign."""
    engine = lab()
    rel = labelled(engine, engine.label("toy", "zeek_v8.2.1"))
    assert rel.aggregate("count(*)").fetchone()[0] == 10


# ------------------------------------------------------------ label columns

def test_a_matched_flow_carries_its_rule_and_source(lab):
    engine = lab()
    rows = rows_by_uid(labelled(engine, engine.label("toy", "zeek_v8.2.1")))
    hit = rows["attack_exec"]
    assert hit["label_binary"] is True
    assert hit["label_class"] == "brute_force"
    assert hit["label_raw"] == "FTP-Patator"
    assert hit["label_source"] == "matched"
    assert hit["rule_id"] == 0


def test_an_unmatched_flow_is_benign_by_absence_and_says_so(lab):
    engine = lab()
    rows = rows_by_uid(labelled(engine, engine.label("toy", "zeek_v8.2.1")))
    for uid in ("bystander", "before", "after"):
        assert rows[uid]["label_binary"] is False
        assert rows[uid]["label_class"] == "benign"
        assert rows[uid]["label_source"] == "closed-world"
        assert rows[uid]["rule_id"] is None


def test_executed_attempted_and_unmeasured_are_three_distinct_answers(lab):
    """Zero is a confirmed empty connection; NULL is a missing measurement."""
    engine = lab()
    rows = rows_by_uid(labelled(engine, engine.label("toy", "zeek_v8.2.1")))
    assert rows["attack_exec"]["label_executed"] is True
    assert rows["attempt"]["label_executed"] is False
    assert rows["unmeasured"]["label_executed"] is None


def test_an_exempt_attack_never_reports_attempted(lab):
    """PortScan sends nothing by design; marking it attempted would discard it."""
    engine = lab()
    rows = rows_by_uid(labelled(engine, engine.label("toy", "zeek_v8.2.1")))
    assert rows["scan"]["label_binary"] is True
    assert rows["scan"]["label_executed"] is None


def test_provenance_is_stamped_on_every_row(lab):
    engine = lab()
    report = engine.label("toy", "zeek_v8.2.1")
    rows = rows_by_uid(labelled(engine, report))
    for row in rows.values():
        assert row["dataset"] == "toy"
        # One composite hash on the row, not one column per input. The
        # component hashes stay in the report, where a human reads them.
        assert row["label_method"] == "schedule"
        assert row["method_sha"] == report.method_sha
        assert "manifest_sha" not in row and "taxonomy_sha" not in row
        assert row["labelled_at"] == report.labelled_at
        assert row["capture"] == "Tuesday-WorkingHours"


# -------------------------------------------------------------- quarantine

def test_quarantine_flags_without_changing_the_label(lab):
    """A quarantined flow keeps its closed-world label and gains one column."""
    engine = lab(QUARANTINED_MANIFEST)
    rows = rows_by_uid(labelled(engine, engine.label("toy", "zeek_v8.2.1")))

    quiet = rows["quiet"]
    assert quiet["label_quality"] == "uncertain"
    assert quiet["quarantine_reason"] == "unadjudicable-tail"
    assert quiet["label_class"] == "benign"          # unchanged
    assert quiet["label_source"] == "closed-world"
    assert rows["attack_exec"]["label_quality"] == "certain"


def test_the_quality_columns_exist_even_with_no_regions_declared(lab):
    """The output schema must not depend on whether a manifest declares any."""
    engine = lab()
    rows = rows_by_uid(labelled(engine, engine.label("toy", "zeek_v8.2.1")))
    assert all(r["label_quality"] == "certain" for r in rows.values())
    assert all(r["quarantine_reason"] is None for r in rows.values())


# ------------------------------------------------------------- the gate

def test_a_silent_rule_aborts_the_run(lab):
    """A rule matching nothing means a wrong time or address -- the likeliest
    place for a transcription error to hide. It must not produce a table."""
    manifest = MANIFEST + SILENT_RULE
    engine = lab(manifest)
    with pytest.raises(GateFailure, match="silent rules"):
        engine.label("toy", "zeek_v8.2.1")


def test_the_run_report_is_written_even_when_the_gate_aborts(lab):
    """The report of a run that failed is the most useful one there is."""
    import json
    manifest = MANIFEST + SILENT_RULE
    engine = lab(manifest)
    with pytest.raises(GateFailure):
        engine.label("toy", "zeek_v8.2.1")

    path = engine.provenance.reports / "labelling_toy.json"
    payload = json.loads(path.read_text())
    assert payload["gate"]["passed"] is False
    assert payload["silent_rules"] == ["Heartbleed 2017-07-04 20:00-20:10"]
    assert payload["output"] is None                 # nothing was written


def test_no_table_is_written_when_the_gate_fails(lab):
    manifest = MANIFEST + SILENT_RULE
    engine = lab(manifest)
    with pytest.raises(GateFailure):
        engine.label("toy", "zeek_v8.2.1")
    target = engine.lake.uri("labelled", dataset="toy",
                             feature_space="zeek_v8.2.1", rel="conn.parquet")
    assert not engine.lake.exists(target)


def test_no_write_reports_without_producing_a_table(lab):
    engine = lab()
    report = engine.label("toy", "zeek_v8.2.1", write=False)
    assert report.output is None
    assert report.attack_flows == 6
    assert report.gate.passed


# ------------------------------------------------------------------ report

def test_the_report_round_trips_to_json_and_reads_as_a_table(lab):
    engine = lab()
    report = engine.label("toy", "zeek_v8.2.1")

    payload = report.to_dict()
    assert payload["attack_flows"] == 6
    assert payload["gate"]["passed"] is True
    assert {r["raw"] for r in payload["per_rule"]} == {"FTP-Patator", "PortScan"}

    text = report.table()
    assert "FTP-Patator" in text and "ATTACK" in text and "attack ratio" in text


# ------------------------------------------------- schema and overwrite guards

def test_the_label_columns_are_type_checked_not_just_present(lab):
    """Presence alone missed a real bug: label_executed was briefly an untyped
    NULL, so the same column had different types depending on the extractor."""
    engine = lab()
    rel = labelled(engine, engine.label("toy", "zeek_v8.2.1"))
    types = dict(zip(rel.columns, (str(t) for t in rel.types)))
    assert types["label_binary"] == "BOOLEAN"
    assert types["label_executed"] == "BOOLEAN"


def test_a_wrongly_typed_label_column_is_refused(lab):
    import duckdb
    from talos.data.labelling.schedule.writer import LabelWriter
    rel = duckdb.connect().sql(
        "SELECT 1 AS label_binary, 'x' AS label_class, 'x' AS label_source, "
        "'schedule' AS label_method, 'x' AS method_sha, "
        "CAST(NULL AS DOUBLE) AS label_confidence, "
        "CAST(NULL AS DOUBLE) AS label_weight, "
        "'x' AS dataset, 'x' AS capture, 'x' AS labelled_at, "
        "'x' AS label_raw, true AS label_executed, 1 AS rule_id, "
        "1 AS rules_matched, 'x' AS label_quality, 'x' AS quarantine_reason")
    with pytest.raises(SchemaError, match="label_binary is INTEGER"):
        LabelWriter.verify_schema(rel)


def test_relabelling_with_the_same_manifest_overwrites(lab):
    """A re-run replacing its own output is what a re-run should do."""
    engine = lab()
    first = engine.label("toy", "zeek_v8.2.1")
    again = engine.label("toy", "zeek_v8.2.1")
    assert again.output == first.output


EDITED = MANIFEST.replace('end: "10:20"', 'end: "10:25"')


def test_a_different_manifest_will_not_silently_replace_the_labels(lab):
    """Overwriting would discard labels derived from a schedule still on record,
    with nothing left to say they existed."""
    lab().label("toy", "zeek_v8.2.1")
    with pytest.raises(StaleOutputError, match="still on record"):
        lab(EDITED).label("toy", "zeek_v8.2.1")


def test_the_refusal_can_be_overridden_deliberately(lab):
    lab().label("toy", "zeek_v8.2.1")
    assert lab(EDITED).label("toy", "zeek_v8.2.1", force=True).output


# ------------------------------------------------------------- preflight

def test_nothing_to_label_says_what_to_run(lab, tmp_path):
    """Three scans should not begin with a raw DuckDB 'no files match'."""
    from talos.data.labelling.schedule.engine import LabellingError
    engine = lab()
    import shutil
    shutil.rmtree(tmp_path / "lake" / "sources" / "datasets" / "parquet")
    with pytest.raises(LabellingError, match="talos convert"):
        engine.label("toy", "zeek_v8.2.1")


def test_the_wrong_feature_space_names_the_right_one(lab):
    from talos.data.labelling.schedule.engine import LabellingError
    engine = lab()
    with pytest.raises(LabellingError, match="exists under: zeek_v8.2.1"):
        engine.label("toy", "zeek_v7.0")


def test_a_table_that_is_not_conn_is_refused_before_scanning(lab, tmp_path):
    """A capability declaration says what an extractor CAN emit; this says what
    this particular tree actually has."""
    from talos.data.labelling.schedule.engine import LabellingError
    engine = lab()
    out = (tmp_path / "lake" / "sources" / "datasets" / "parquet" / "zeek_v8.2.1"
           / "toy" / "Tuesday-WorkingHours" / "conn.parquet")
    duckdb.connect().sql("SELECT 1 AS packets, 2 AS bytes").write_parquet(str(out))
    with pytest.raises(LabellingError, match="does not look like conn"):
        engine.label("toy", "zeek_v8.2.1")
