"""EDA v3's real-data plumbing: the `labelled`-zone default and validity threading.

Both landed as real, wired code (spec.py's `_adopt_feature_set`, profile_dataset's
zone default and `--include-invalid`) with zero dedicated tests before this file --
worth closing before Gate 3 runs any of it against the real lake for the first time.
"""

from __future__ import annotations

import json
import sys

import duckdb
import pytest

from talos.data.profiling.eda import profile_dataset


def parsed(*argv):
    old = sys.argv
    sys.argv = ["profile_dataset", *argv]
    try:
        return profile_dataset.parse_args()
    finally:
        sys.argv = old


# --------------------------------------------------------------- CLI defaults

def test_zone_defaults_to_labelled():
    """`parquet` has no `label_class`; profiling it would silently collapse
    every flow into one `(unlabelled)` group and hide class imbalance."""
    a = parsed("--dataset", "toy")
    assert a.zone == "labelled"


def test_include_invalid_defaults_to_false():
    """Excluding a validity-invariant violation is the default, matching the
    population the training pools actually draw from."""
    a = parsed("--dataset", "toy")
    assert a.include_invalid is False


def test_include_invalid_can_be_set():
    a = parsed("--dataset", "toy", "--include-invalid")
    assert a.include_invalid is True


def test_zone_can_be_overridden_to_parquet():
    a = parsed("--dataset", "toy", "--zone", "parquet")
    assert a.zone == "parquet"


# --------------------------------------------------- real-data validity threading

# duration = -5.0 breaks `non_negative_duration`; every other row is clean.
ROWS = [
    ("u0", 1000.0, "cap1", "benign", 1.0, 100),
    ("u1", 1001.0, "cap1", "benign", 2.0, 200),
    ("u2", 1002.0, "cap1", "attack", 3.0, 300),
    ("u3", 1003.0, "cap1", "attack", 4.0, 400),
    ("u4", 1004.0, "cap1", "benign", 5.0, 500),
    ("u5", 1005.0, "cap1", "benign", -5.0, 999),   # the one invalid row
]


@pytest.fixture
def lake(tmp_path):
    """A minimal real lake: one labelled-zone parquet, a config.yml beside it."""
    lake_root = tmp_path / "lake"
    out = (lake_root / "sources" / "datasets" / "labelled" / "zeek_v8.2.1"
           / "schedule" / "toy")
    out.mkdir(parents=True, exist_ok=True)
    rows = ", ".join(
        f"('{u}', {ts}, '{cap}', '{cls}', {dur}, {b})" for u, ts, cap, cls, dur, b in ROWS)
    duckdb.connect().sql(
        f"SELECT * FROM (VALUES {rows}) "
        f"AS t(uid, ts, capture, label_class, duration, orig_bytes)"
    ).write_parquet(str(out / "conn.parquet"))

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        f"lake:\n  root: {lake_root}\n"
        f"reports:\n  dir: {tmp_path / 'reports'}\n")
    return tmp_path, config_path


def run_profile(config_path, tmp_path, name, *extra):
    out = tmp_path / f"{name}.json"
    old = sys.argv
    sys.argv = ["profile_dataset", "--dataset", "toy", "--config", str(config_path),
               "--out", str(out), *extra]
    try:
        profile_dataset.main()
    finally:
        sys.argv = old
    return json.loads(out.read_text())


def test_the_invalid_row_is_excluded_by_default(lake):
    """The row with duration = -5.0 must not reach `by_class`'s counts."""
    tmp_path, config_path = lake
    profile = run_profile(config_path, tmp_path, "excluding")
    assert profile["meta"]["excludes_invalid"] is True
    assert profile["meta"]["rows"] == 5


def test_include_invalid_keeps_every_row(lake):
    tmp_path, config_path = lake
    profile = run_profile(config_path, tmp_path, "including", "--include-invalid")
    assert profile["meta"]["excludes_invalid"] is False
    assert profile["meta"]["rows"] == 6


def test_excluding_invalid_changes_the_measured_minimum(lake):
    """Not just a row-count check: the invalid row's duration must not reach
    the numeric feature's own statistics either."""
    tmp_path, config_path = lake
    excluding = run_profile(config_path, tmp_path, "ex2")
    including = run_profile(config_path, tmp_path, "in2", "--include-invalid")

    def overall_min(profile):
        # Merge every class's per-group minimum -- `by_class` is grouped.
        mins = [blk["numeric"]["duration"]["min"] for blk in profile["by_class"].values()]
        return min(mins)

    assert overall_min(including) == -5.0
    assert overall_min(excluding) > -5.0
