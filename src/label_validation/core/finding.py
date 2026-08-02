#!/usr/bin/env python3
"""A structured finding -- the unit of output for every validation check.

The repo already emits QA signals: compare.py returns a `verdict` string per
feature, label_flows.py prints suspect bursts and persists a count. Each invents
its own shape, so nothing downstream can rank them, filter them, or gate on
them, and a reader cannot tell a cosmetic warning from a reason to stop.

A check here returns Findings instead. Same fields every time, an ordered
severity, the evidence rows carried alongside the claim, and a `repro` string so
a reader can re-run the query that produced the number rather than trusting it.
That last field is the point: a validation module that cannot be audited is just
another unverified assertion in a chain we are trying to verify.
"""

from dataclasses import dataclass, field
from enum import IntEnum


class Severity(IntEnum):
    """Ordered so findings sort by consequence, not by discovery order.

    BLOCKER is reserved for states that make every downstream number
    meaningless -- a row-count mismatch, labels written by a manifest that no
    longer exists on disk. Nothing else earns it.
    """

    INFO = 10
    MINOR = 20
    MAJOR = 30
    CRITICAL = 40
    BLOCKER = 50

    @classmethod
    def parse(cls, name):
        try:
            return cls[str(name).strip().upper()]
        except KeyError:
            raise ValueError(f"unknown severity {name!r}; "
                             f"expected one of {[s.name for s in cls]}")


@dataclass
class Finding:
    """One defect, with the numbers that establish it.

    `metrics` is the quantitative claim and is what the scorecard and the gate
    read -- prose in `detail` is for humans only and is never parsed.

    `evidence` holds the rows that demonstrate the claim (capped by the runner
    before serialisation; a check that matches ten million flows must not try to
    carry ten million rows into a JSON report).
    """

    check_id: str
    dataset: str
    severity: Severity
    title: str
    detail: str = ""
    metrics: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    # What the finding is about: a rule id, a canonical class, a capture, a
    # feature. Free-form because checks scope differently, but always recorded
    # so findings can be grouped without re-parsing the title.
    scope: dict = field(default_factory=dict)
    repro: str = ""

    def to_dict(self, evidence_limit=50):
        return {
            "check_id": self.check_id,
            "dataset": self.dataset,
            "severity": self.severity.name,
            "severity_rank": int(self.severity),
            "title": self.title,
            "detail": self.detail,
            "metrics": self.metrics,
            "scope": self.scope,
            "evidence": self.evidence[:evidence_limit],
            "evidence_truncated": max(0, len(self.evidence) - evidence_limit),
            "repro": self.repro,
        }


def sort_findings(findings):
    """Most consequential first, then by check id so runs diff cleanly."""
    return sorted(findings, key=lambda f: (-int(f.severity), f.check_id, f.title))
