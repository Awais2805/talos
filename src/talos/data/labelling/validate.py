"""The checks that run on every labelling run.

Four for v0, and the split between fatal and advisory is the important part.

**Silent rules abort.** A rule matching zero flows means a wrong time or a wrong
address, and it is the single most likely place for a transcription error to
hide. It is not a warning: a manifest with a dead rule produces a labelled table
that is confidently missing an attack class.

**Integrity aborts.** `uid` must be unique in conn, or the whole
label-by-uid-then-propagate design collapses, and no label may be null.

**Rule breadth is advisory.** Almost every rule should touch exactly one source
and one destination; a rule sweeping up many hosts is usually a manifest error,
but 2017's Infiltration-PortScan legitimately touches a /24, so this reports
rather than refuses.

**The benign audit is advisory, and it is the one that finds things.** Benign
traffic does not arrive in large synchronised bursts aimed at one host. Where it
does, labelling has missed an attack. This check is how the off-by-one-minute
window bug and two defects in CIC's own documentation were caught — none of
which any rule-level check would have noticed, because every rule fired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from talos.common.validation import Check, CheckResult, ValidationGate

# A burst is suspicious when it is both large in absolute terms and large
# relative to the dataset -- and when almost all of it involves a known victim.
BURST_MIN_FLOWS = 10_000
BURST_MIN_SHARE = 0.005
BURST_VICTIM_SHARE = 0.9


@dataclass(frozen=True)
class Burst:
    capture: str
    bucket: str
    flows: int
    victim_flows: int

    @property
    def victim_share(self) -> float:
        return self.victim_flows / self.flows if self.flows else 0.0


def silent_rules(ctx: Mapping) -> CheckResult:
    silent = ctx["silent_rules"]
    if not silent:
        return CheckResult("silent rules", True, f"{ctx['rules_total']} rules all fired")
    listed = "; ".join(r.label() for r in silent)
    return CheckResult(
        "silent rules", False,
        f"{len(silent)} of {ctx['rules_total']} matched nothing: {listed}",
        evidence={"silent": [r.label() for r in silent]})


def uid_is_unique(ctx: Mapping) -> CheckResult:
    total, distinct = ctx["total_flows"], ctx["distinct_uids"]
    ok = total == distinct
    return CheckResult(
        "uid unique in conn", ok,
        f"{total:,} rows, {distinct:,} distinct uids"
        + ("" if ok else " — uid is the join key for every enrichment log"),
        evidence={"rows": total, "distinct": distinct})


def no_null_labels(ctx: Mapping) -> CheckResult:
    nulls = ctx["null_labels"]
    return CheckResult("no null labels", nulls == 0, f"{nulls:,} null label_class",
                       evidence={"nulls": nulls})


def rule_breadth(ctx: Mapping) -> CheckResult:
    """Reports the widest rule. Wide is not wrong, but it is worth seeing."""
    widest = ctx.get("widest_rule")
    if widest is None:
        return CheckResult("rule breadth", True, "no matched rules to measure",
                           fatal=False)
    name, srcs, dsts = widest
    return CheckResult("rule breadth", True,
                       f"widest is {name}: {srcs} source(s), {dsts} destination(s)",
                       fatal=False, evidence={"rule": name, "srcs": srcs, "dsts": dsts})


def benign_audit(ctx: Mapping) -> CheckResult:
    """Large unmatched bursts aimed at a known victim look like missed attacks."""
    if not ctx.get("benign_audit_possible", True):
        # Saying "could not run" rather than "passed". This check is the one
        # that found the off-by-one-minute window bug and two defects in CIC's
        # documentation; a silent pass would be worse than no check at all.
        return CheckResult("benign burst audit", False,
                           "could not run: the manifest declares no victim IPs or "
                           "subnets, so no flow can be scored as victim-involving",
                           fatal=False)
    bursts: Sequence[Burst] = ctx["suspect_bursts"]
    if not bursts:
        return CheckResult("benign burst audit", True,
                           "no unmatched burst looks like a missed attack", fatal=False)
    worst = bursts[0]
    return CheckResult(
        "benign burst audit", False,
        f"{len(bursts)} suspect burst(s); largest {worst.capture} {worst.bucket}Z "
        f"{worst.flows:,} flows, {100 * worst.victim_share:.1f}% victim-involving",
        fatal=False,
        evidence={"bursts": [(b.capture, b.bucket, b.flows, b.victim_flows)
                             for b in bursts]})


class LabelValidator:
    """Assembles the labelling gate. The engine supplies the context."""

    @staticmethod
    def gate() -> ValidationGate:
        gate = ValidationGate("labelling")
        gate.register(Check("silent rules", silent_rules))
        gate.register(Check("uid unique in conn", uid_is_unique))
        gate.register(Check("no null labels", no_null_labels))
        gate.register(Check("rule breadth", rule_breadth, fatal=False))
        gate.register(Check("benign burst audit", benign_audit, fatal=False))
        return gate

    @staticmethod
    def suspect(bursts: Sequence[Burst], total_flows: int) -> list[Burst]:
        """Filter raw buckets down to the ones worth reporting."""
        floor = max(BURST_MIN_FLOWS, BURST_MIN_SHARE * total_flows)
        return [b for b in bursts
                if b.flows > floor and b.victim_share > BURST_VICTIM_SHARE]
