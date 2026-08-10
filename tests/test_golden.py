"""Golden regression: the comparison stage must stay byte-identical across the refactor.

There is no other safety net. This runs the whole EDA kernel -- spec loading, the
additive class views, correlation and rank matrices, the verdict thresholds -- over
three real profiles captured from the lake on 2026-07-29, and asserts the output has
not moved. It needs no lake, no AWS and no network: profiles are sufficient
statistics, which is the whole point of the design.

Fields that legitimately change every run (timestamps) or that depend on where the
spec file lives (spec_sha, once the spec moves to its extractor) are excluded.
Everything else must match exactly.

Runs with or without pytest:

    python3 tests/test_golden.py      # standalone, no dependencies
    pytest tests/ -q
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
FIXTURES = TESTS / "fixtures" / "eda"
GOLDEN = TESTS / "golden"

DATASETS = ["cic-ids-2017", "cic-ids-2018", "cic-ddos-2019"]

# Regenerated per run, or dependent on where the spec file happens to live.
VOLATILE = {"generated_utc", "profiles_used", "spec_sha"}

# The numbers the README quotes. If these move, the README is wrong.
HEADLINE = {
    "cic-ddos-2019": (0.449336, 13, 0),
    "cic-ids-2017": (0.102348, 2, 3),
    "cic-ids-2018": (0.130306, 5, 1),
}


def _load_compare():
    """Import the comparison module from whichever layout is current."""
    try:
        from talos.data.profiling.eda import compare               # after the refactor
    except ImportError:                             # pre-refactor layout
        sys.path.insert(0, str(ROOT))
        from src.eda import compare
    return compare


def _strip(obj):
    """Drop volatile keys anywhere in the document."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def regenerate_into(work):
    """Rebuild every comparison from the fixture profiles, in a scratch directory."""
    work = Path(work)
    for src in sorted(FIXTURES.glob("profile_*.json")):
        shutil.copy(src, work / src.name)
    written = _load_compare().regenerate(work)
    assert written, "regenerate() produced nothing"
    return work


def check_fixtures():
    n_profiles = len(sorted(FIXTURES.glob("profile_*.json")))
    n_golden = len(sorted(GOLDEN.glob("compare_*.json")))
    assert n_profiles == 3, f"expected 3 fixture profiles, found {n_profiles}"
    assert n_golden == 3, f"expected 3 golden comparisons, found {n_golden}"


def check_matches_golden(work, dataset):
    name = f"compare_{dataset}.json"
    got = _strip(json.loads((Path(work) / name).read_text()))
    want = _strip(json.loads((GOLDEN / name).read_text()))
    assert set(got) == set(want), f"{dataset}: top-level keys changed"
    # Section by section, so a failure names the part that drifted rather than
    # dumping a 400 kB diff.
    for section in want:
        assert got[section] == want[section], f"{dataset}: '{section}' drifted"


def check_headline(work, dataset):
    fingerprint, biased, transfers = HEADLINE[dataset]
    doc = json.loads((Path(work) / f"compare_{dataset}.json").read_text())
    o = doc["overview"]
    assert o["domain_fingerprint"] == fingerprint, f"{dataset}: domain_fingerprint"
    assert o["n_domain_biased"] == biased, f"{dataset}: n_domain_biased"
    assert o["n_transfers"] == transfers, f"{dataset}: n_transfers"


# ------------------------------------------------------------------ pytest API

try:
    import pytest

    @pytest.fixture(scope="module")
    def regenerated(tmp_path_factory):
        return regenerate_into(tmp_path_factory.mktemp("eda"))

    def test_fixtures_present():
        check_fixtures()

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_comparison_matches_golden(regenerated, dataset):
        check_matches_golden(regenerated, dataset)

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_headline_numbers(regenerated, dataset):
        check_headline(regenerated, dataset)

except ImportError:                                 # pragma: no cover
    pass


# -------------------------------------------------------------- standalone API

def main():
    failures = []

    def run(label, fn, *args):
        try:
            fn(*args)
            print(f"  ok    {label}")
        except AssertionError as exc:
            print(f"  FAIL  {label}: {exc}")
            failures.append(label)

    print("golden regression (no lake, no AWS)")
    run("fixtures present", check_fixtures)
    with tempfile.TemporaryDirectory() as tmp:
        work = regenerate_into(tmp)
        for ds in DATASETS:
            run(f"{ds} matches golden", check_matches_golden, work, ds)
        for ds in DATASETS:
            run(f"{ds} headline numbers", check_headline, work, ds)

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
