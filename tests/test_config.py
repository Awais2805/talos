"""Config and zone resolution: the layer that makes the lake location a choice.

The point of these is that one code path serves a local directory and an S3
bucket, and that the legacy config shape (aws.bucket + a flat lake: map) still
resolves to exactly the URIs the old hardcoded scripts built.

    python3 tests/test_config.py     # standalone
    pytest tests/ -q
"""

import sys
import tempfile
from pathlib import Path

from talos.common import zones
from talos.common.config import Config, ConfigError

BUCKET = "s3://ids-datalakec48eb2cab942494ba5059fac3b3527d9"

LEGACY = {
    "aws": {"bucket": "ids-datalakec48eb2cab942494ba5059fac3b3527d9",
            "region": "eu-north-1"},
    "lake": {"raw": "{dataset}/pcaps", "extracted": "extracted",
             "parquets": "parquets", "labelled": "labelled", "canonical": "mapped"},
}

CURRENT = {
    "lake": {"root": BUCKET,
             "zones": {"raw": "{dataset}/pcaps", "extracted": "extracted/{dataset}",
                       "parquets": "parquets/{dataset}", "labelled": "labelled/{dataset}",
                       "canonical": "mapped/{dataset}"}},
    "aws": {"region": "eu-north-1"},
}


def test_legacy_and_current_agree():
    """The shape change must not move a single byte of any resolved URI."""
    old, new = Config(LEGACY), Config(CURRENT)
    for zone in zones.ZONE_ORDER:
        assert old.zone(zone, "cic-ids-2017") == new.zone(zone, "cic-ids-2017"), zone


def test_resolves_to_the_previously_hardcoded_uris():
    """Exactly what profile_dataset.py and zeek_batch.sh used to build by hand."""
    c = Config(CURRENT)
    assert c.zone("labelled", "cic-ddos-2019") == f"{BUCKET}/labelled/cic-ddos-2019"
    assert c.zone("parquets", "cic-ids-2017") == f"{BUCKET}/parquets/cic-ids-2017"
    assert c.zone("raw", "cic-ids-2017") == f"{BUCKET}/cic-ids-2017/pcaps"
    assert c.zone("canonical", "cic-ids-2018") == f"{BUCKET}/mapped/cic-ids-2018"
    assert c.parquet_glob("labelled", "cic-ddos-2019") == \
        f"{BUCKET}/labelled/cic-ddos-2019/**/*.parquet"


def test_local_root_is_not_remote():
    c = Config({"lake": {"root": "./lake"}})
    assert not c.is_remote
    assert c.zone("extracted", "ds") == "lake/extracted/ds"
    assert Config(CURRENT).is_remote


def test_roles_are_readable():
    c = Config({**CURRENT, "datasets": {
        "a": {"role": "train"}, "b": {"role": "train"}, "c": {"role": "holdout"}}})
    assert c.datasets_with_role("train") == ["a", "b"]
    assert c.role("c") == "holdout"
    assert c.role("nope") == "unknown"          # unknown datasets never train


def test_unknown_zone_is_refused():
    try:
        Config(CURRENT).zone("nonsense")
    except ConfigError as exc:
        assert "nonsense" in str(exc)
    else:
        raise AssertionError("expected ConfigError for an unknown zone")


def test_init_creates_every_zone():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "lake"
        made = zones.init(str(root))
        assert set(made) == set(zones.ZONE_ORDER)
        for zone, (path, created) in made.items():
            assert path.is_dir(), zone
            assert created, zone
        # the canonical zone lives at mapped/, which is why init reports paths
        assert made["canonical"][0].name == "mapped"
        # idempotent
        again = zones.init(str(root))
        assert not any(created for _, created in again.values())


def test_init_refuses_a_remote_root():
    try:
        zones.init(BUCKET)
    except ValueError as exc:
        assert "remote" in str(exc)
    else:
        raise AssertionError("expected ValueError for a remote root")


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
