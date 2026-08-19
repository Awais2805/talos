"""What `compare` will and will not put side by side.

Two rules, and every test here is one of them:

  1. Below two profiles there is no "rest" -- refuse, rather than write a
     report whose every rest-side field is null and which reads, at a glance,
     exactly like a real "no drift found" result.
  2. Only within one VIEW. A view is the population a profile describes:
     `prelabel` (raw flows), `postlabel-schedule`, `postlabel-ae`. Comparing
     across views measures the labeller and reports it as a domain difference,
     with no field saying so.

Profiles are built inline, not loaded from a committed JSON fixture -- a
fixture that looks like a real captured profile is exactly the kind of stale,
silently-authoritative data these tests exist to guard against. `spec_sha:
"toy"` marks them synthetic, and an empty `numeric_spec` never collides with
whatever the real `Spec()` currently declares.
"""

from __future__ import annotations

import json

import pytest

from talos.data.profiling.eda import compare


def toy_profile(dataset, n=10, zone="labelled", method="schedule"):
    """The minimum shape `compare_one` needs, with zero real features."""
    view = ("prelabel" if zone == "parquet"
            else f"postlabel-{method}" if method else "postlabel")
    return {
        "profile_version": 3,
        "meta": {"dataset": dataset, "source": ["synthetic"], "zone": zone,
                 "view": view, "method": method if zone == "labelled" else None,
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


def write(directory, dataset, n=10, zone="labelled", method="schedule"):
    doc = toy_profile(dataset, n, zone, method)
    slug = f"{doc['meta']['view']}_{dataset}"
    (directory / f"profile_{slug}.json").write_text(json.dumps(doc))
    return slug


# ------------------------------------------------------- rule 1: need a peer

def test_one_profile_writes_no_comparison(tmp_path):
    write(tmp_path, "a")
    assert compare.regenerate(tmp_path) == []
    assert not list(tmp_path.glob("compare_*.json"))


def test_zero_profiles_writes_no_comparison(tmp_path):
    assert compare.regenerate(tmp_path) == []


def test_a_stale_comparison_is_removed_when_a_peer_disappears(tmp_path):
    write(tmp_path, "a")
    stale = tmp_path / "compare_postlabel-schedule_a.json"
    stale.write_text(json.dumps({"meta": {"dataset": "a", "compared_against": ["b"]}}))
    compare.regenerate(tmp_path, view="postlabel-schedule")
    assert not stale.exists()


def test_two_profiles_do_compare(tmp_path):
    write(tmp_path, "a")
    write(tmp_path, "b")
    written = compare.regenerate(tmp_path)
    assert {p.name for p in written} == {"compare_postlabel-schedule_a.json",
                                         "compare_postlabel-schedule_b.json"}
    doc = json.loads((tmp_path / "compare_postlabel-schedule_a.json").read_text())
    assert doc["meta"]["compared_against"] == ["b"]
    assert doc["meta"]["view"] == "postlabel-schedule"


# ----------------------------------------------------- rule 2: one view only

def test_the_same_dataset_under_two_methods_does_not_collide(tmp_path):
    """The defect that made this whole guarantee necessary: one dataset
    profiled under two labelling methods used to overwrite one file."""
    a = write(tmp_path, "ds", method="schedule")
    b = write(tmp_path, "ds", method="ae")
    assert a != b
    assert len(list(tmp_path.glob("profile_*.json"))) == 2


def test_two_methods_are_not_compared_against_each_other(tmp_path):
    """One dataset under two methods is not two domains: each view has one
    member, so neither has a peer and nothing is written."""
    write(tmp_path, "ds", method="schedule")
    write(tmp_path, "ds", method="ae")
    assert compare.regenerate(tmp_path, view="postlabel-schedule") == []
    assert compare.regenerate(tmp_path, view="postlabel-ae") == []


def test_an_ambiguous_directory_is_refused_rather_than_guessed(tmp_path):
    write(tmp_path, "a", method="schedule")
    write(tmp_path, "b", method="ae")
    with pytest.raises(SystemExit) as exc:
        compare.load_profiles(tmp_path)
    assert "more than one view" in str(exc.value)


def test_a_named_view_selects_only_its_own_profiles(tmp_path):
    write(tmp_path, "a", method="schedule")
    write(tmp_path, "b", method="schedule")
    write(tmp_path, "c", method="ae")
    got = compare.load_profiles(tmp_path, "postlabel-schedule")
    assert sorted(got) == ["a", "b"]


def test_prelabel_and_postlabel_are_different_views(tmp_path):
    write(tmp_path, "a", zone="parquet")
    write(tmp_path, "a", zone="labelled", method="schedule")
    assert sorted(compare.load_profiles(tmp_path, "prelabel")) == ["a"]
    assert sorted(compare.load_profiles(tmp_path, "postlabel-schedule")) == ["a"]


def test_regenerate_does_not_delete_another_views_comparison(tmp_path):
    """A view falling below two peers must not clear a view that still has them."""
    write(tmp_path, "a", method="schedule")
    write(tmp_path, "b", method="schedule")
    compare.regenerate(tmp_path, view="postlabel-schedule")
    keep = tmp_path / "compare_postlabel-schedule_a.json"
    assert keep.exists()
    write(tmp_path, "solo", zone="parquet")
    compare.regenerate(tmp_path, view="prelabel")
    assert keep.exists()


# --------------------------------------------------------------- pairwise

def test_compare_pair_ignores_a_third_profile_in_the_directory(tmp_path):
    write(tmp_path, "a")
    write(tmp_path, "b")
    write(tmp_path, "c")
    out = compare.compare_pair("a", "b", tmp_path)
    assert out.name == "compare_postlabel-schedule_a_vs_b.json"
    doc = json.loads(out.read_text())
    assert doc["meta"]["compared_against"] == ["b"]
    # The pool-wide file must not exist just because compare_pair ran.
    assert not (tmp_path / "compare_postlabel-schedule_a.json").exists()


def test_compare_pair_refuses_a_missing_dataset(tmp_path):
    write(tmp_path, "a")
    with pytest.raises(SystemExit):
        compare.compare_pair("a", "b", tmp_path)


def test_compare_pair_refuses_a_peer_from_another_view(tmp_path):
    write(tmp_path, "a", method="schedule")
    write(tmp_path, "b", method="ae")
    with pytest.raises(SystemExit) as exc:
        compare.compare_pair("a", "b", tmp_path, view="postlabel-schedule")
    assert "no profile for b" in str(exc.value)
