"""The extraction plugin layer.

The property under test throughout is that two extractions cannot share a
feature space unless they genuinely produced the same kind of table. Everything
else here — resumability, failure reporting, availability — exists so that a
partial or unpinned run is visible rather than silently indistinguishable from
a clean one.
"""

from pathlib import Path

import pytest

from talos.data.extraction.base import (
    register_extractor,
    BaseExtractor, ExtractionError, ExtractionReport, get_extractor, registered,
    version_from_tag,
)
from talos.data.extraction.extractors.zeek import ZeekExtractor
from talos.data.labelling.schedule.execution import ExecutionClassifier


def zeek(**kwargs) -> ZeekExtractor:
    kwargs.setdefault("image", "zeek/zeek:8.2.1")
    return ZeekExtractor(**kwargs)


# ------------------------------------------------------------ feature space

def test_feature_space_is_tool_and_version():
    assert zeek().feature_space == "zeek_v8.2.1"


def test_settings_are_recorded_even_though_they_do_not_fork_the_path():
    """The fingerprint is in the sidecar, so a write-time check can be added
    later without re-extracting. Until then, two configs of one version share a
    directory -- which is why the fingerprint is worth recording now."""
    assert zeek(scripts=("protocols/conn/mac-logging",)).config_fingerprint() != \
        zeek(scripts=()).config_fingerprint()
    assert zeek(scripts=("a", "b")).config_sha != zeek(scripts=("b", "a")).config_sha
    assert zeek(flags="-C").config_sha == zeek(flags=["-C"]).config_sha


def test_an_extractor_with_no_settings_has_no_digest():
    class Bare(BaseExtractor):
        name = "bare"
        version = "1.0"
        def extract(self, pcap_paths, output_dir):
            return ExtractionReport()

    assert Bare().feature_space == "bare_v1.0" and Bare().config_sha == ""


def test_a_feature_space_cannot_smuggle_a_path_segment():
    class Sneaky(BaseExtractor):
        name = "x"
        version = "../../etc"
        def extract(self, pcap_paths, output_dir):
            return ExtractionReport()

    with pytest.raises(ExtractionError, match="path segment"):
        _ = Sneaky().feature_space


# ---------------------------------------------------------------- versioning

def test_a_pinned_tag_is_read_without_starting_a_container():
    assert version_from_tag("zeek/zeek:8.2.1") == "8.2.1"
    assert version_from_tag("registry.local:5000/zeek/zeek:8.2.1") == "8.2.1"


def test_a_moving_tag_is_not_accepted_as_a_version():
    """`latest` is not a version. Two runs months apart would share a feature
    space while being different builds -- the exact contamination it prevents."""
    for image in ("zeek/zeek:latest", "zeek/zeek:dev", "zeek/zeek",
                  "registry.local:5000/zeek/zeek"):
        assert version_from_tag(image) is None


def test_an_explicit_version_wins_and_needs_no_docker():
    assert zeek(image="zeek/zeek:latest", version="8.2.1").version == "8.2.1"


def test_an_unpinned_image_without_docker_refuses_rather_than_guessing(monkeypatch):
    monkeypatch.setattr("talos.data.extraction.extractors.zeek.shutil.which",
                        lambda _: None)
    with pytest.raises(ExtractionError, match="share a feature space"):
        _ = zeek(image="zeek/zeek:latest").version


# ------------------------------------------------------------- availability

def test_availability_is_reported_before_any_pcap_is_fetched():
    ok, why = zeek().available()
    assert isinstance(ok, bool)
    if not ok:
        assert "docker" in why


def test_the_unimplemented_plugin_declares_itself_unavailable():
    """Registered so the naming is settled, unavailable so a run refuses early
    rather than after downloading several hundred GB."""
    plugin = get_extractor("cicflowmeter")
    ok, why = plugin.available()
    assert not ok and "no implementation" in why


def test_both_plugins_are_registered():
    assert registered() == ("cicflowmeter", "zeek")


def test_an_unknown_extractor_lists_what_exists():
    with pytest.raises(ExtractionError, match="zeek"):
        get_extractor("nprobe")


# ------------------------------------------------------------------- the run

def test_the_invocation_mounts_the_raw_zone_read_only(tmp_path):
    cmd = zeek().command(tmp_path / "monday.pcap", tmp_path / "out")
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert any(m.endswith(":ro") and "/pcap_in" in m for m in mounts)


def test_scripts_and_flags_reach_the_invocation(tmp_path):
    cmd = zeek(flags="-C", scripts=("policy/tuning/json-logs",)).command(
        tmp_path / "a.pcap", tmp_path / "out")
    assert "-C" in cmd and "policy/tuning/json-logs" in cmd
    assert "LogAscii::use_json=T" in cmd


def test_an_already_extracted_capture_is_skipped(tmp_path):
    """The existence of the output IS the checkpoint, so a run that died 400
    pcaps into 2018 resumes instead of restarting."""
    pcap = tmp_path / "monday.pcap"
    pcap.write_bytes(b"\xd4\xc3\xb2\xa1")
    out = tmp_path / "logs"
    (out / "monday").mkdir(parents=True)
    (out / "monday" / "conn.log").write_text("{}\n")

    report = zeek().extract([str(pcap)], str(out))
    assert report.skipped == ["monday"] and report.extracted == []


def test_a_missing_pcap_is_recorded_not_raised(tmp_path):
    report = zeek().extract([str(tmp_path / "gone.pcap")], str(tmp_path / "out"))
    assert report.failed == [("gone", "pcap not found")]
    assert not report.complete


def test_the_sidecar_records_an_incomplete_run(tmp_path):
    """A run where most pcaps failed must not look like a clean one to whatever
    later picks 'the most recent feature space'."""
    import json
    report = ExtractionReport(extracted=["a"], failed=[("b", "zeek exploded")])
    path = zeek().write_metadata(str(tmp_path), "toy", report=report)
    meta = json.loads(path.read_text())

    assert meta["complete"] is False
    assert meta["pcaps_extracted"] == 1 and meta["pcaps_failed"] == 1
    assert meta["failures"][0]["capture"] == "b"
    assert meta["feature_space"] == zeek().feature_space
    assert meta["config_fingerprint"]["scripts"] == []


def test_a_clean_run_says_so(tmp_path):
    import json
    path = zeek().write_metadata(str(tmp_path), "toy",
                                 report=ExtractionReport(extracted=["a", "b"]))
    assert json.loads(path.read_text())["complete"] is True


# ------------------------------------------------- resumption, at the lake

def test_resumption_asks_the_lake_and_skips_before_fetching(tmp_path, monkeypatch):
    """The checkpoint lives in the output zone, not in local scratch.

    Each pcap gets a fresh temp dir during a run, so a local check could never
    fire -- and checking before the fetch is the whole point: resuming a run
    that died 400 pcaps into 2018 must not re-download what it already has.
    """
    from talos.common.config import Config
    from talos.cli import _extract_dataset

    lake = Config({"lake": {"root": str(tmp_path / "lake")}}).lake()
    destination = lake.uri("extracted", dataset="toy", feature_space="zeek_v1")

    # One capture already extracted in the lake, one not.
    done = Path(destination) / "monday"
    done.mkdir(parents=True)
    (done / "conn.log").write_text("{}\n")

    fetched = []
    monkeypatch.setattr(type(lake), "get",
                        lambda self, uri, local: fetched.append(uri) or Path(uri))

    class Never(BaseExtractor):
        name = "never"
        version = "1"
        def extract(self, pcap_paths, output_dir):
            return ExtractionReport(failed=[("tuesday", "not run in this test")])

    report = _extract_dataset(lake, Never(), "toy",
                              ["/raw/monday.pcap", "/raw/tuesday.pcap"], destination)

    assert report.skipped == ["monday"]
    assert fetched == ["/raw/tuesday.pcap"]      # the finished one was never downloaded


# ---------------------------------------------------------- capabilities

def test_zeek_declares_what_downstream_needs():
    from talos.data.extraction.base import BYTE_COUNTS, CONN_BACKBONE, FLOW_ID
    e = zeek()
    assert e.capabilities() == frozenset({FLOW_ID, CONN_BACKBONE, BYTE_COUNTS})
    e.require(FLOW_ID, CONN_BACKBONE, BYTE_COUNTS)      # must not raise


def test_a_stage_refuses_against_a_schema_it_cannot_read():
    """Declared, not assumed. Without a conn backbone there is nothing to join
    a schedule onto, and running anyway produces wrong-but-plausible output."""
    from talos.data.extraction.base import CONN_BACKBONE, FLOW_ID
    plugin = get_extractor("cicflowmeter")
    with pytest.raises(ExtractionError, match="conn_backbone"):
        plugin.require(CONN_BACKBONE, FLOW_ID)


def test_missing_byte_counts_nulls_the_column_rather_than_guessing():
    """Absent byte counts degrade one column, not the whole stage -- but they
    must never come out as 'attempted', which would read as a finding.

    The NULL is CAST: an untyped one would leave DuckDB to infer the column
    type, so label_executed would not be BOOLEAN for an extractor without byte
    counts and would be for one with them.
    """
    from talos.data.labelling.schedule.execution import ExecutionClassifier
    unmeasurable = ExecutionClassifier().classify_sql(measurable=False)
    assert unmeasurable == "CAST(NULL AS BOOLEAN)"
    assert "orig_bytes" in ExecutionClassifier().classify_sql(measurable=True)


def test_measurability_is_an_argument_not_mutable_state():
    """It is a property of the run, not of the expression generator. It was
    briefly a field the engine reached in and set."""
    assert not hasattr(ExecutionClassifier(), "enabled")


def test_capabilities_are_recorded_in_the_sidecar(tmp_path):
    import json
    path = zeek().write_metadata(str(tmp_path), "toy", report=ExtractionReport())
    assert json.loads(path.read_text())["capabilities"] == [
        "byte_counts", "conn_backbone", "flow_id"]


# ------------------------------------------------- the sidecar tracks the TREE

def test_an_earlier_failure_is_not_forgotten_by_a_later_partial_run(tmp_path):
    """Run 1 fails two captures. Run 2 processes a different subset and
    succeeds. The tree is still incomplete, and the sidecar must say so."""
    import json
    first = ExtractionReport(extracted=["a"], failed=[("b", "boom"), ("c", "boom")])
    zeek().write_metadata(str(tmp_path), "toy", report=first)
    previous = json.loads((tmp_path / "_extractor_meta.json").read_text())

    second = ExtractionReport(extracted=["d"])
    zeek().write_metadata(str(tmp_path), "toy", report=second, previous=previous)
    meta = json.loads((tmp_path / "_extractor_meta.json").read_text())

    assert meta["complete"] is False
    assert {f["capture"] for f in meta["failures"]} == {"b", "c"}
    assert meta["runs"] == 2


def test_a_resolved_failure_clears(tmp_path):
    import json
    zeek().write_metadata(str(tmp_path), "toy",
                          report=ExtractionReport(failed=[("b", "boom")]))
    previous = json.loads((tmp_path / "_extractor_meta.json").read_text())

    zeek().write_metadata(str(tmp_path), "toy",
                          report=ExtractionReport(extracted=["b"]), previous=previous)
    meta = json.loads((tmp_path / "_extractor_meta.json").read_text())
    assert meta["complete"] is True and meta["failures"] == []


def test_the_first_extraction_time_survives_later_runs(tmp_path):
    import json
    zeek().write_metadata(str(tmp_path), "toy", report=ExtractionReport(extracted=["a"]))
    first = json.loads((tmp_path / "_extractor_meta.json").read_text())

    zeek().write_metadata(str(tmp_path), "toy",
                          report=ExtractionReport(extracted=["b"]), previous=first)
    meta = json.loads((tmp_path / "_extractor_meta.json").read_text())
    assert meta["first_extracted_at"] == first["extracted_at_utc"]


# ------------------------------------------------------------ the registry

def test_a_plugin_without_a_class_level_name_is_refused():
    """It is read off the class at import time, without constructing anything."""
    with pytest.raises(ExtractionError, match="class-level"):
        @register_extractor
        class Nameless(BaseExtractor):
            version = "1"
            def extract(self, pcap_paths, output_dir):
                return ExtractionReport()


def test_a_plugin_needing_constructor_arguments_still_registers():
    """The old registry called cls() at import, so this broke the whole package."""
    @register_extractor
    class Fussy(BaseExtractor):
        name = "fussy"
        def __init__(self, required):
            self._v = required
        @property
        def version(self):
            return self._v
        def extract(self, pcap_paths, output_dir):
            return ExtractionReport()

    assert "fussy" in registered()
    assert get_extractor("fussy", required="2.0").version == "2.0"
