#!/usr/bin/env python3
"""Decide whether a dataset's labels are fit to build on, and say so in one exit code.

Every other stage measures. This is the only place a measurement becomes a
refusal, and it exists because the failure mode being defended against is not a
crash: it is a labelled dataset that looks complete, trains cleanly, and reports
99% accuracy against labels that are wrong. Nothing downstream detects that --
a model fits whatever it is given -- so the refusal has to happen here, between
validation and training, or it does not happen at all.

The bar is not in this file. `gate.block_at` and `gate.always_block` live in
policy.yaml beside every other threshold, so it can be argued with, and the
argument shows up in a diff, without touching the code that enforces it.

Output is written for a CI log with nothing else on screen: what blocked, which
check said so, and why that check's word is final.

Exit codes: 0 nothing blocking, 1 blocked or nothing to judge.

Usage:
    python src/label_validation/report/gate.py --all
    python src/label_validation/report/gate.py --dataset cic-ids-2017
    python src/label_validation/report/gate.py --dataset cic-ids-2017,cic-ids-2018
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "config.yml").is_file())
sys.path.insert(0, str(ROOT))

from src.label_validation.core.finding import Severity      # noqa: E402
from src.label_validation.core import paths as _paths  # noqa: E402

REPORTS = ROOT / "reports" / "validation"
POLICY = _paths.POLICY

# Metrics are the quantitative claim, so a couple of them go in the CI line --
# enough to see the size of the problem without opening the JSON.
METRICS_SHOWN = 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dataset", default=None, help="comma-separated dataset names")
    g.add_argument("--all", action="store_true",
                   help="every report in --dir (the default)")
    p.add_argument("--dir", default=str(REPORTS))
    p.add_argument("--policy", default=str(POLICY))
    return p.parse_args()


def load_policy(path):
    doc = yaml.safe_load(Path(path).read_text()) or {}
    gate = doc.get("gate") or {}
    return (Severity.parse(gate.get("block_at", "CRITICAL")),
            tuple(gate.get("always_block") or ()),
            doc.get("version"))


def load_reports(directory, datasets=None):
    """(name, path, doc-or-None) for each dataset to judge.

    With no explicit list this is discovery by glob, like every other stage: a
    dataset enters the gate by having been validated, not by being named in a
    second place that can fall out of step. Named datasets are looked up by
    path, so a missing report is visible as a missing report rather than as a
    silent pass.
    """
    directory = Path(directory)
    if datasets:
        return [(d, directory / f"{d}.json", _read(directory / f"{d}.json"))
                for d in datasets]
    out = []
    for path in sorted(directory.glob("*.json")):
        doc = _read(path)
        if doc and doc.get("meta", {}).get("dataset") and isinstance(
                doc.get("findings"), list):
            out.append((doc["meta"]["dataset"], path, doc))
    return out


def _read(path):
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def blocking(doc, block_at, always_block):
    """Findings that block, each with the reason it does, worst first."""
    out = []
    for f in doc.get("findings", []):
        try:
            sev = Severity.parse(f.get("severity"))
        except ValueError:
            continue
        if f.get("check_id") in always_block:
            why = f"{f['check_id']} blocks on any finding, whatever its severity"
        elif sev >= block_at:
            why = f"{sev.name} is at or above the {block_at.name} bar"
        else:
            continue
        out.append((sev, why, f))
    return sorted(out, key=lambda t: (-int(t[0]), t[2].get("check_id", "")))


def _metrics_line(f):
    kv = [f"{k}={v}" for k, v in (f.get("metrics") or {}).items()
          if isinstance(v, (int, float, str)) and len(str(v)) <= 40]
    return "  ".join(kv[:METRICS_SHOWN])


def main():
    a = parse_args()
    block_at, always, version = load_policy(a.policy)
    names = [d.strip() for d in a.dataset.split(",")] if a.dataset else None
    reports = load_reports(a.dir, names)

    print(f"label gate · policy v{version} · blocks at {block_at.name} or above")
    if always:
        print(f"           · always blocks: {', '.join(always)}")
    print()

    if not reports:
        print(f"NO REPORTS in {a.dir} — nothing has been validated, so nothing can be "
              f"cleared. Run: make validate-all")
        return 1

    blocked, unmeasured = [], 0
    for name, path, doc in reports:
        if doc is None:
            print(f"{name:<16} NO REPORT  {path} does not exist")
            print(f"{'':<16}            an unvalidated dataset is not a clean one — "
                  f"run: make validate DATASET={name}\n")
            blocked.append(name)
            continue

        summary = doc.get("summary", {})
        mix = ", ".join(f"{k} {v}" for k, v in
                        sorted(summary.get("by_severity", {}).items(),
                               key=lambda kv: -int(Severity.parse(kv[0])))) or "none"
        bad = blocking(doc, block_at, always)
        n = summary.get("n_findings", len(doc.get("findings", [])))
        print(f"{name:<16} {'BLOCKED' if bad else 'pass':<8} "
              f"{len(bad)} blocking of {n} finding(s) · {mix}")

        for sev, why, f in bad:
            print(f"    {sev.name:<9} {f.get('check_id', '?'):<26} {f.get('title', '')}")
            print(f"    {'':<9} blocks because {why}")
            metrics = _metrics_line(f)
            if metrics:
                print(f"    {'':<9} {metrics}")

        skipped, errored = summary.get("checks_skipped", 0), summary.get("checks_error", 0)
        if skipped or errored:
            unmeasured += skipped + errored
            # Indented to the dataset, not to the last finding: this is a
            # statement about the whole report, and the commonest way to misread
            # a gate is to take "pass" as "everything was looked at".
            print(f"    caveat: {skipped} check(s) did not run and {errored} errored "
                  f"— unmeasured, not clean")
        if bad:
            blocked.append(name)
        print()

    if blocked:
        print(f"BLOCKED — {len(blocked)} of {len(reports)} dataset(s): "
              f"{', '.join(blocked)}")
        print("These labels are not fit to build on. Nothing here is fixed by editing a "
              "label: correct the manifest or the schedule, re-run "
              "`make validate DATASET=…`, and gate again.")
        print(f"Full evidence, and the query behind every number above, in {a.dir}/"
              "<dataset>.json — or read it as HTML with `make validate-render`.")
        return 1

    print(f"PASS — {len(reports)} dataset(s), nothing at or above {block_at.name}")
    if unmeasured:
        print(f"       {unmeasured} check(s) across the pool did not run or errored. "
              f"This clears what was measured; it says nothing about what was not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
