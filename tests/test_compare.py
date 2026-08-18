"""`compare.regenerate`/`compare_pair` must refuse to compare below two
profiles, and a pairwise comparison must ignore any third profile present.

Profiles here are built inline, not loaded from a committed JSON fixture --
a fixture that looks like a real captured profile is exactly the kind of
stale, silently-authoritative data these tests exist to guard against
(the `role`-key regression, DQ.1-DQ.3, came from precisely that pattern:
fixtures frozen before a schema migration, never revisited). `spec_sha:
"toy"` marks these as obviously synthetic, both to a reader and to
`load_profiles`'s ruler check -- an empty `numeric_spec` never collides with
whatever the real `Spec()` currently declares.
"""

from __future__ import annotations

import json

import pytest

from talos.data.profiling.eda import compare


def toy_profile(dataset, n=10):
    """The minimum shape `compare_one` needs, with zero real features."""
    return {
        "profile_version": 2,
        "meta": {"dataset": dataset, "source": "synthetic", "zone": "labelled",
                 "generated_utc": "2026-01-01T00:00:00+00:00", "scan_seconds": 0.0,
                 "spec_version": 1, "spec_sha": "toy", "excludes_invalid": False,
                 "sample_pct": None, "limit_rows": None, "partial": None,
                 "duckdb": "toy", "rows": n, "columns": [],
                 "numeric_features": 0, "categorical_features": 0, "skipped_features": {}},
        "benign_class": "benign",
        "numeric_spec": {}, "categorical_spec": {}, "pair_order": [], "featured_pairs": [],
        "quantiles": [], "identity_columns": [],
        "by_class": {"benign": {"n": n, "numeric": {}, "pairs_xy": [], "pairs_joint": [],
                                "categorical": {}, "distinct": {}, "nulls": {}}},
        "captures": [],
    }


def write(directory, dataset, n=10):
    (directory / f"profile_{dataset}.json").write_text(json.dumps(toy_profile(dataset, n)))


def test_one_profile_writes_no_comparison(tmp_path):
    write(tmp_path, "a")
    written = compare.regenerate(tmp_path)
    assert written == []
    assert not list(tmp_path.glob("compare_*.json"))


def test_zero_profiles_writes_no_comparison(tmp_path):
    assert compare.regenerate(tmp_path) == []


def test_a_stale_comparison_is_removed_when_a_peer_disappears(tmp_path):
    write(tmp_path, "a")
    stale = tmp_path / "compare_a.json"
    stale.write_text(json.dumps({"meta": {"dataset": "a", "compared_against": ["b"]}}))
    compare.regenerate(tmp_path)
    assert not stale.exists()


def test_two_profiles_do_compare(tmp_path):
    write(tmp_path, "a")
    write(tmp_path, "b")
    written = compare.regenerate(tmp_path)
    assert {p.name for p in written} == {"compare_a.json", "compare_b.json"}
    doc = json.loads((tmp_path / "compare_a.json").read_text())
    assert doc["meta"]["compared_against"] == ["b"]


def test_compare_pair_ignores_a_third_profile_in_the_directory(tmp_path):
    write(tmp_path, "a")
    write(tmp_path, "b")
    write(tmp_path, "c")
    out = compare.compare_pair("a", "b", tmp_path)
    assert out.name == "compare_a_vs_b.json"
    doc = json.loads(out.read_text())
    assert doc["meta"]["compared_against"] == ["b"]
    # The pool-wide file must not exist just because compare_pair ran.
    assert not (tmp_path / "compare_a.json").exists()


def test_compare_pair_refuses_a_missing_dataset(tmp_path):
    write(tmp_path, "a")
    with pytest.raises(SystemExit):
        compare.compare_pair("a", "b", tmp_path)
