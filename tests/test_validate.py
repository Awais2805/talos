#!/usr/bin/env python3
"""End-to-end test of the label validator against planted faults.

Builds the synthetic lake, runs the real labelling stage over it, runs the real
validator over that, and asserts that each check found the fault it was written
for -- and, just as importantly, that the one correctly-scheduled rule produced
no findings at all. A validator that flags everything is as useless as one that
flags nothing, so the clean rule is the load-bearing assertion here.

Nothing is mocked. The only difference from a production run is that the lake
is a directory instead of a bucket.

Usage:
    python tests/test_validate.py            # build fixture and test
    python tests/test_validate.py --keep     # reuse an existing fixture
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# check id -> the rule whose planted fault it must find. None means the finding
# is dataset-wide rather than scoped to one rule.
EXPECTED = [
    ("sch.window_sweep", "LateWindow", "flood already running before the window opens"),
    ("sch.window_sweep", "EarlyEnd", "flood still running after the window closes"),
    ("sch.boundary_residue", "EarlyEnd", "attack tail left in the benign class"),
    ("sch.intra_window_gaps", "Gappy", "window quiet for 20 of its 30 minutes"),
    ("sch.endpoint_coverage", "HiddenAlias", "attacker arrives from an undeclared address"),
    ("sch.rule_yield", "SilentRule", "rule scheduled against a host with no traffic"),
    ("int.uid_unique", None, "the same flow ingested twice"),
]

# These describe a healthy dataset; the fixture is healthy in these respects, so
# any finding from them is a false positive.
MUST_BE_SILENT = ["int.row_parity", "int.key_nulls", "int.label_wellformed",
                  "int.provenance_current", "int.schema_stability",
                  "int.ts_in_capture", "sch.class_day_conformance"]

# The rule with nothing wrong with it. Being flagged here is a false positive,
# and this is the assertion that keeps the checks honest.
CLEAN_RULE = "CleanAttack"


def run(cmd, env=None, label=""):
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if r.returncode not in (0, 1):        # 1 = gate tripped, expected here
        print(r.stdout[-3000:])
        print(r.stderr[-3000:], file=sys.stderr)
        sys.exit(f"{label} failed with exit {r.returncode}")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    fixture = Path(a.fixture) if a.fixture else Path(
        os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "talos_fixture"

    if not a.keep:
        print(f"building fixture at {fixture} ...")
        run([sys.executable, "tests/build_fixture.py", "--out", str(fixture)],
            label="fixture build")

    report = fixture / "validation_toy.json"
    env = dict(os.environ, TALOS_MANIFESTS=str(fixture / "manifests"))
    run([sys.executable, "src/label_validation/runner.py", "--dataset", "toy",
         "--lake-root", str(fixture), "--no-s3", "--tier", "0,1",
         "--out", str(report)], env=env, label="validator")

    doc = json.loads(report.read_text())
    findings = doc["findings"]
    got = {(f["check_id"], f["scope"].get("raw")) for f in findings}
    fired = {f["check_id"] for f in findings}

    failures = []
    print(f"\n{'planted fault':<52}{'check':<28}result")
    print("-" * 96)
    for check_id, rule, what in EXPECTED:
        ok = (check_id, rule) in got if rule else check_id in fired
        print(f"{what:<52}{check_id:<28}{'FOUND' if ok else '*** MISSED ***'}")
        if not ok:
            failures.append(f"{check_id} did not flag {rule or 'anything'} ({what})")

    print(f"\n{'must stay silent on a healthy dataset':<52}{'check':<28}result")
    print("-" * 96)
    for check_id in MUST_BE_SILENT:
        n = sum(1 for f in findings if f["check_id"] == check_id)
        print(f"{'':<52}{check_id:<28}{'silent' if n == 0 else f'*** {n} FALSE POSITIVE(S) ***'}")
        if n:
            failures.append(f"{check_id} produced {n} finding(s) on a healthy aspect")

    flagged_clean = [f["check_id"] for f in findings if f["scope"].get("raw") == CLEAN_RULE]
    print(f"\n{'correctly-scheduled rule stays unflagged':<52}"
          f"{CLEAN_RULE:<28}"
          f"{'clean' if not flagged_clean else '*** FLAGGED BY ' + ','.join(flagged_clean) + ' ***'}")
    if flagged_clean:
        failures.append(f"{CLEAN_RULE} was flagged by {flagged_clean} — false positive")

    errored = [r for r in doc["runlog"] if r["status"] == "error"]
    for r in errored:
        failures.append(f"{r['check_id']} raised {r.get('error')}")

    print(f"\n{len(findings)} finding(s): "
          + ", ".join(f"{k} {v}" for k, v in sorted(doc["summary"]["by_severity"].items())))
    if failures:
        print(f"\nFAILED — {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASSED — every planted fault found, no false positives.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
