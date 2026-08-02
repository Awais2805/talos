#!/usr/bin/env python3
"""Run the label-validation checks over one dataset and write a findings report.

Talos labels flows from a published attack schedule -- a document, not a
measurement. This stage exists to establish, with evidence, whether that
document was transcribed correctly, whether the traffic agrees with it, and
which flows cannot be adjudicated either way. It never edits a label: findings
go to the manifests, and labelling is re-run.

Runs one dataset at a time because the heavy tiers scan the lake, and the
lake's 2019 conn zone is 72.5M flows. Tier 6 (cross-dataset) is the exception
and reads only the per-dataset JSON this writes.

Usage:
    python src/label_validation/runner.py --dataset cic-ids-2017
    python src/label_validation/runner.py --dataset cic-ddos-2019 --tier 0,1
    python src/label_validation/runner.py --dataset cic-ids-2017 --check sch.window_sweep
    # against a local fixture tree rather than S3 (see tests/fixtures):
    python src/label_validation/runner.py --dataset toy --lake-root /tmp/fixture --no-s3
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "config.yml").is_file())
sys.path.insert(0, str(ROOT))

from src.common.lake import Lake                          # noqa: E402
from src.preprocess.label.label_flows import (            # noqa: E402
    MANIFESTS, load_rules, sha, validate_rules,
)
from src.label_validation.core.finding import Severity, sort_findings  # noqa: E402
from src.label_validation.core import paths as _paths  # noqa: E402
from src.label_validation.oracle.stage import oracle_paths               # noqa: E402
from src.label_validation.core.registry import Context, load_checks, run_one, select  # noqa: E402

REPORTS = ROOT / "reports" / "validation"
POLICY = _paths.POLICY

# Zeek log types worth probing for. conn is the labelled zone's source; the rest
# are independent evidence channels. notice is listed even though the lake has
# none today -- the probe is what records that absence in the report, rather
# than the corroboration checks silently returning nothing.
LOG_TYPES = ["dns", "http", "ssl", "ssh", "ftp", "weird", "notice", "files"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--tier", default=None,
                   help="comma-separated tiers to run (default: 0,1,2,3)")
    p.add_argument("--check", default=None,
                   help="comma-separated check ids; overrides --tier")
    p.add_argument("--out", default=None)
    p.add_argument("--config", default=str(ROOT / "config.yml"))
    p.add_argument("--policy", default=str(POLICY))
    p.add_argument("--evidence-limit", type=int, default=50,
                   help="evidence rows kept per finding in the JSON")
    # Local-fixture mode. The dummy-data test drives the whole module through
    # this path, so the checks are exercised by the same code that runs on S3.
    p.add_argument("--lake-root", default=None,
                   help="read parquet from this root instead of s3://<bucket>")
    p.add_argument("--no-s3", action="store_true")
    # Same guard rails as the EDA and labelling stages.
    p.add_argument("--memory-limit", default="8GB")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--temp-dir", default=None)
    p.add_argument("--max-temp", default="20GB")
    return p.parse_args()


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    return cfg


def probe_sources(lake, root, dataset, cfg, require_s3):
    """Which log types actually exist for this dataset, as parquet globs.

    A missing log is a fact about the lake, not an error: ftp.log exists only
    for 2017, and notice.log for none of them. Checks declare what they need and
    the runner skips them with that reason recorded.
    """
    parq = cfg["lake"]["parquets"]
    lbl = cfg["lake"].get("labelled", "labelled")
    sources = {"labelled": f"{root}/{lbl}/{dataset}/**/*.parquet",
               "conn": f"{root}/{parq}/{dataset}/**/conn.parquet"}
    available = set()
    for name, glob in list(sources.items()):
        if _glob_has_files(lake, glob, require_s3):
            available.add(name)
    for lt in LOG_TYPES:
        glob = f"{root}/{parq}/{dataset}/**/{lt}.parquet"
        sources[lt] = glob
        if _glob_has_files(lake, glob, require_s3):
            available.add(lt)
    # The external oracle is a local artefact rather than a lake zone: the join
    # against DistriNet's corrected labels is built offline by
    # src/label_validation/oracle/stage.py, so its presence on disk is what makes tier 3's
    # orc.* checks runnable.
    orc_json, orc_uids = oracle_paths(dataset)
    sources["oracle"], sources["oracle_report"] = str(orc_uids), str(orc_json)
    if orc_json.exists() or orc_uids.exists():
        available.add("oracle")
    return sources, available


def _glob_has_files(lake, glob, require_s3):
    # glob() over S3 is a listing, not a scan -- cheap enough to do per log type
    # and far cheaper than discovering the whole prefix.
    try:
        row = lake.one(f"SELECT count(*) FROM glob('{glob}')")
        return bool(row and row[0])
    except Exception:
        return False


def main():
    a = parse_args()
    cfg = load_config(a.config)
    policy = yaml.safe_load(Path(a.policy).read_text())

    bucket = cfg["aws"]["bucket"]
    region = cfg["aws"]["region"]
    root = a.lake_root.rstrip("/") if a.lake_root else f"s3://{bucket}"
    require_s3 = not a.no_s3 and root.startswith("s3://")

    role = (cfg.get("datasets", {}).get(a.dataset) or {}).get("role", "unknown")
    man, rules = load_rules(a.dataset)
    validate_rules(rules)

    lake = Lake(region, a.memory_limit, a.temp_dir, a.threads, a.max_temp,
                require_s3=require_s3)
    sources, available = probe_sources(lake, root, a.dataset, cfg, require_s3)

    columns = {}
    if "labelled" in available:
        columns = lake.columns(sources["labelled"])

    spec = None
    try:
        from src.eda.spec import Spec
        spec = Spec()
    except Exception:
        pass                      # spec is optional; only T2/T6 use it

    taxonomy = yaml.safe_load((MANIFESTS / "taxonomy.yaml").read_text())
    ctx = Context(
        dataset=a.dataset, role=role, lake=lake, rules=rules, manifest=man,
        manifest_sha=sha(MANIFESTS / f"{a.dataset}.yaml"),
        taxonomy_sha=sha(MANIFESTS / "taxonomy.yaml"), taxonomy=taxonomy,
        spec=spec, policy=policy, sources=sources, columns=columns,
        bucket=bucket, region=region,
    )

    load_checks()
    only = {c.strip() for c in a.check.split(",")} if a.check else None
    tiers = ({int(t) for t in a.tier.split(",")} if a.tier
             else (None if only else {0, 1, 2, 3}))
    runnable, skipped = select(a.dataset, tiers=tiers, only=only, available=available)

    print(f"dataset    {a.dataset}  ({role})")
    print(f"root       {root}")
    print(f"logs       {', '.join(sorted(available)) or '(none found)'}")
    print(f"rules      {len(rules)} windows   manifest {ctx.manifest_sha} "
          f"taxonomy {ctx.taxonomy_sha}")
    print(f"checks     {len(runnable)} runnable, {len(skipped)} skipped\n")

    findings, runlog = [], []
    for chk in runnable:
        rec, found = run_one(chk, ctx)
        runlog.append(rec)
        findings.extend(found)
        mark = {"ok": " ", "error": "!"}[rec["status"]]
        note = rec.get("error", "")
        worst = max((f.severity for f in found), default=None)
        print(f" {mark} {chk.id:<32}{rec['seconds']:>7.1f}s  "
              f"{rec['n_findings']:>3} finding(s)"
              f"{'  ' + worst.name if worst else ''}{'  ' + note if note else ''}")

    for chk, why in skipped:
        runlog.append({"check_id": chk.id, "tier": chk.tier, "title": chk.title,
                       "status": "skipped", "reason": why, "n_findings": 0})

    findings = sort_findings(findings)
    by_sev = {}
    for f in findings:
        by_sev[f.severity.name] = by_sev.get(f.severity.name, 0) + 1

    report = {
        "validate_version": 1,
        "meta": {
            "dataset": a.dataset, "role": role, "root": root,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest_sha": ctx.manifest_sha, "taxonomy_sha": ctx.taxonomy_sha,
            "policy_version": policy.get("version"),
            "rules_total": len(rules),
            "logs_available": sorted(available),
            "tiers": sorted(tiers) if tiers else None,
        },
        "summary": {
            "n_findings": len(findings), "by_severity": by_sev,
            "checks_run": sum(1 for r in runlog if r["status"] == "ok"),
            "checks_error": sum(1 for r in runlog if r["status"] == "error"),
            "checks_skipped": sum(1 for r in runlog if r["status"] == "skipped"),
        },
        "runlog": runlog,
        "findings": [f.to_dict(a.evidence_limit) for f in findings],
    }

    out = Path(a.out) if a.out else REPORTS / f"{a.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(f"\n{len(findings)} finding(s): "
          f"{', '.join(f'{k} {v}' for k, v in sorted(by_sev.items())) or 'none'}")
    print(f"report -> {out}")

    worst = max((f.severity for f in findings), default=Severity.INFO)
    if any(r["status"] == "error" for r in runlog):
        print("!! one or more checks errored — see runlog", file=sys.stderr)
    return 0 if worst < Severity.parse(policy["gate"]["block_at"]) else 1


if __name__ == "__main__":
    sys.exit(main())
