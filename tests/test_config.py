"""Config and zone resolution: the layer that makes the lake location a choice.

The point of these is that one code path serves a local directory and an S3
bucket, and that a lake whose layout Talos did not create can still be
addressed by declaring its own templates.

    python3 tests/test_config.py     # standalone
    pytest tests/ -q
"""

import sys
import tempfile
from pathlib import Path

import pytest

from talos.common import zones
from talos.common.config import Config, ConfigError
from talos.common.lake.lake import LakeClient

BUCKET = "s3://ids-datalakec48eb2cab942494ba5059fac3b3527d9"

CURRENT = {
    "lake": {"root": BUCKET,
             "zones": {"raw": "{dataset}/pcaps", "extracted": "extracted/{dataset}",
                       "parquet": "parquets/{dataset}", "labelled": "labelled/{dataset}",
                       "canonical": "mapped/{dataset}"}},
    "aws": {"region": "eu-north-1"},
}


def test_a_declared_override_wins_over_the_default_layout():
    """The escape hatch for a lake this code did not create: declare its own
    templates and every stage addresses it correctly, with no source segment."""
    c = Config(CURRENT)
    assert c.zone("labelled", "cic-ddos-2019") == f"{BUCKET}/labelled/cic-ddos-2019"
    assert c.zone("parquet", "cic-ids-2017") == f"{BUCKET}/parquets/cic-ids-2017"
    assert c.zone("raw", "cic-ids-2017") == f"{BUCKET}/cic-ids-2017/pcaps"
    assert c.zone("canonical", "cic-ids-2018") == f"{BUCKET}/mapped/cic-ids-2018"
    assert c.parquet_glob("labelled", "cic-ddos-2019") == \
        f"{BUCKET}/labelled/cic-ddos-2019/**/*.parquet"


def test_local_root_is_not_remote():
    c = Config({"lake": {"root": "./lake"}})
    assert not c.is_remote
    assert c.zone("extracted", "ds", feature_space="zeek_v8.2.1") == \
        "lake/sources/datasets/extracted/zeek_v8.2.1/ds"
    assert Config(CURRENT).is_remote


def test_an_unscoped_zone_stops_at_the_static_prefix():
    """The default extracted zone is feature-space scoped, so a dataset alone
    cannot address it. Resolution stops rather than emitting a path with a
    literal `{feature_space}` in it, which no backend could ever list."""
    c = Config({"lake": {"root": "./lake"}})
    assert c.zone("extracted") == "lake/sources"
    assert c.zone("extracted", "ds") == "lake/sources/datasets/extracted"


# ----------------------------------------------------------------- sources

def test_a_dataset_declares_which_source_it_came_from():
    """Not filing -- provenance. A honeypot capture has no attack schedule, so
    the schedule labeller must never be pointed at one."""
    c = Config({"lake": {"root": "./lake"}, "datasets": {
        "cic-ids-2017": {"role": "train"},                        # defaults
        "netem-run-01": {"source": "netem", "role": "train"},
        "hp-2026-08": {"source": "honeypot", "role": "audit"}}})

    assert c.source_of("cic-ids-2017") == "datasets"              # the default
    assert c.source_of("netem-run-01") == "netem"
    assert c.datasets_from_source("honeypot") == ["hp-2026-08"]


def test_the_source_lands_in_the_path_without_the_caller_naming_it():
    c = Config({"lake": {"root": "./lake"},
                "datasets": {"netem-run-01": {"source": "netem"}}})
    assert c.zone("raw", "netem-run-01") == "lake/sources/netem/raw/netem-run-01"
    assert c.zone("raw", "cic-ids-2017") == "lake/sources/datasets/raw/cic-ids-2017"


def test_an_unknown_source_is_refused():
    c = Config({"lake": {"root": "./lake"},
                "datasets": {"x": {"source": "sensornet"}}})
    with pytest.raises(ConfigError, match="sensornet"):
        c.source_of("x")


def test_canonical_is_shared_across_sources():
    """Zone 5 is where sources stop being separate -- it is the merged pool the
    cross-domain experiment draws from. Nesting it under a source would make
    assembly a glob and imply domains stay apart through training."""
    c = Config({"lake": {"root": "./lake"},
                "datasets": {"netem-run-01": {"source": "netem"}}})
    assert not zones.is_source_scoped(zones.DEFAULT_TEMPLATES["canonical"])
    assert c.zone("canonical", "netem-run-01", schema_version="v1") == \
        "lake/canonical/v1/netem-run-01"


def test_config_no_longer_answers_role_questions():
    """Roles moved to ExperimentSpec. Leaving a stub here that returned
    "unknown" would be worse than an AttributeError: a caller would read it as
    "not allowed to train" when it means "asked the wrong object"."""
    c = Config(CURRENT)
    assert not hasattr(c, "role")
    assert not hasattr(c, "datasets_with_role")


def test_unknown_zone_is_refused():
    try:
        Config(CURRENT).zone("nonsense")
    except ConfigError as exc:
        assert "nonsense" in str(exc)
    else:
        raise AssertionError("expected ConfigError for an unknown zone")


def test_init_creates_every_zone_for_every_source():
    """A fresh install must give somewhere to put pcaps with no documentation."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "lake"
        made = LakeClient(root=str(root)).init()

        assert {d.zone for d in made} == set(zones.ZONE_ORDER)
        for d in made:
            assert Path(d.uri).is_dir(), (d.zone, d.source)
            assert d.created

        # four source-scoped zones x three sources, plus the shared canonical
        assert len(made) == 4 * len(zones.SOURCES) + 1
        for source in zones.SOURCES:
            assert (root / "sources" / source / "raw").is_dir()

        again = LakeClient(root=str(root)).init()        # idempotent
        assert not any(d.created for d in again)


def test_init_stamps_the_lake_with_an_identity_it_never_overwrites():
    """lake.yaml is what makes a lake discoverable, and what tells you a bucket
    is a Talos lake rather than somebody's backups. Re-initialising must not
    silently reissue the id that existing artefacts refer back to."""
    with tempfile.TemporaryDirectory() as tmp:
        lake = LakeClient(root=str(Path(tmp) / "lake"))
        lake.init()
        first = lake.read_text(zones.join(lake.root, "lake.yaml"))
        assert "lake_id:" in first and "format_version: 1" in first

        lake.init()
        assert lake.read_text(zones.join(lake.root, "lake.yaml")) == first


def test_the_shared_zone_is_planned_once_not_per_source():
    plan = zones.plan("./lake")
    canonical = [d for d in plan if d.zone == "canonical"]
    assert len(canonical) == 1 and canonical[0].source is None


def test_plan_touches_nothing_and_works_for_any_backend():
    """One plan serves a directory, a bucket, or anything fsspec reaches --
    which is what makes `lake.root` a config line rather than a code change."""
    for root in ("./lake", "s3://bkt/talos", "gs://bkt/talos"):
        plan = zones.plan(root)
        assert len(plan) == 4 * len(zones.SOURCES) + 1
        assert all(d.uri.startswith(zones.normalise_root(root)) for d in plan)
    assert not Path("./lake").exists()               # nothing was created


def test_plan_reports_the_location_not_the_zone_name():
    """A lake may put a zone somewhere its name does not predict. The legacy
    bucket puts canonical at mapped/ and raw at {dataset}/pcaps with no source
    segment at all -- which is why plan returns locations, and why that layout
    is a declared override rather than something this code adapts to."""
    plan = {(d.zone, d.source): d for d in
            zones.plan("./lake", Config(CURRENT).zone_templates)}
    assert plan[("canonical", None)].name == "mapped"
    assert plan[("raw", None)].uri == "lake"         # no static prefix at all


def test_missing_config_says_what_to_do():
    try:
        Config.load(Path("/nonexistent/config.yml"))
    except ConfigError as exc:
        assert "talos init" in str(exc)
    else:
        raise AssertionError("expected ConfigError for a missing config")


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    print("config + zones")
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            print(f"  FAIL  {name}: {exc}")
            failures.append(name)
    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
