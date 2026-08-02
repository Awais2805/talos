#!/usr/bin/env python3
"""Check registration and the context every check receives.

A check is a plain function that takes a Context and returns a list of Findings.
Registration is a decorator rather than a list in a config file so a check and
its metadata -- tier, what data it needs, which datasets it applies to -- cannot
drift apart: they are the same edit.

`needs` is the honest part. Several checks depend on Zeek logs that exist for
some datasets and not others (ftp.log is 2017-only; notice.log does not exist
anywhere until the re-extraction runs). Declaring the dependency lets the runner
skip a check with a recorded reason instead of the check either crashing or,
much worse, returning "no findings" and being read as a pass.
"""

import time
from dataclasses import dataclass, field

from src.label_validation.core.finding import Severity

# id -> Check. Populated by the decorator at import time.
CHECKS = {}


@dataclass
class Check:
    id: str
    tier: int
    title: str
    fn: callable
    # None = every dataset. Otherwise an explicit allow-list.
    datasets: tuple = None
    # Log types (or other capabilities) the check cannot run without.
    needs: tuple = ()
    # Worst severity this check is capable of emitting; used to order work and
    # to let `--fail-fast` stop before running cosmetic checks.
    max_severity: Severity = Severity.MAJOR


def check(id, tier, title, datasets=None, needs=(), max_severity=Severity.MAJOR):
    def deco(fn):
        if id in CHECKS:
            raise ValueError(f"duplicate check id {id!r}")
        CHECKS[id] = Check(id=id, tier=tier, title=title, fn=fn,
                           datasets=tuple(datasets) if datasets else None,
                           needs=tuple(needs), max_severity=max_severity)
        return fn
    return deco


def load_checks():
    """Import every check module so the decorators fire. Idempotent.

    Discovered rather than listed. A hand-maintained import list fails in the
    one way this module cannot tolerate: a check file that exists, looks
    written, and never runs -- reporting nothing, which reads exactly like
    finding nothing. Discovery makes dropping a module impossible.
    """
    import importlib
    import pkgutil

    from src.label_validation import checks as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{pkg.__name__}.{mod.name}")
    return CHECKS


def select(dataset, tiers=None, only=None, available=()):
    """Checks that apply, plus the ones that do not and why.

    Returns (runnable, skipped) where skipped carries a reason string -- a
    skipped check is reported, never silently dropped, because "did not run" and
    "ran and found nothing" are different claims about the data.
    """
    runnable, skipped = [], []
    avail = set(available)
    for c in sorted(CHECKS.values(), key=lambda c: (c.tier, c.id)):
        if only and c.id not in only:
            continue
        if tiers is not None and c.tier not in tiers:
            continue
        if c.datasets and dataset not in c.datasets:
            skipped.append((c, f"not applicable to {dataset}"))
            continue
        missing = [n for n in c.needs if n not in avail]
        if missing:
            skipped.append((c, f"missing required input(s): {', '.join(missing)}"))
            continue
        runnable.append(c)
    return runnable, skipped


@dataclass
class Context:
    """Everything a check needs, assembled once per dataset by the runner.

    Checks never open their own connection or re-read the manifest: sharing one
    Lake keeps the credential refresh in one place, and sharing one rule list
    means every check is reasoning about exactly the rules that produced the
    labels under test.
    """

    dataset: str
    role: str
    lake: object
    rules: list                 # normalised rules from label_flows.load_rules
    manifest: dict              # the raw manifest document
    manifest_sha: str
    taxonomy_sha: str
    taxonomy: dict
    spec: object = None         # src.eda.spec.Spec, for feature definitions
    policy: dict = field(default_factory=dict)
    # log type -> parquet glob. 'labelled' is the labelled conn zone; the rest
    # are raw Zeek logs in the parquets zone.
    sources: dict = field(default_factory=dict)
    columns: dict = field(default_factory=dict)   # column -> type, labelled zone
    bucket: str = ""
    region: str = ""

    def q(self, name):
        """The parquet glob for a log type, or None if the lake has none."""
        return self.sources.get(name)

    def threshold(self, key, default=None):
        return self.policy.get("thresholds", {}).get(key, default)


def run_one(chk, ctx):
    """Run a check, timing it and converting a crash into a recorded error.

    A check that raises must not abort the run: the other checks still carry
    information, and an exception is itself a reportable state. It is recorded
    as an error rather than as zero findings so it can never be misread as a
    pass.
    """
    t0 = time.time()
    try:
        findings = chk.fn(ctx) or []
        return {"check_id": chk.id, "tier": chk.tier, "title": chk.title,
                "status": "ok", "seconds": round(time.time() - t0, 2),
                "n_findings": len(findings)}, findings
    except Exception as exc:               # noqa: BLE001 -- deliberate catch-all
        return {"check_id": chk.id, "tier": chk.tier, "title": chk.title,
                "status": "error", "seconds": round(time.time() - t0, 2),
                "error": f"{type(exc).__name__}: {exc}", "n_findings": 0}, []
