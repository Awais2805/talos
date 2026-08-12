"""Sequences a labelling run. Owns every branch, owns no algorithm.

Rules load and validate before any data is read, so
a manifest mistake costs a second rather than an hour. The attack summary and
the benign audit run before the write, so a silent rule aborts *before* a
confidently-wrong table lands in the lake. The run report is emitted either way,
because the report of a run that aborted is the most useful one there is.

Three scans of `conn` per run: the attack summary, the benign audit, and the
write. The total comes from parquet footers rather than a fourth scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from talos.common import zones
from talos.common.provenance import ProvenanceService, composite_sha, content_sha
from talos.data.extraction.base import BYTE_COUNTS, CONN_BACKBONE, FLOW_ID
from talos.common.validation import GateResult, ValidationGate
from talos.data.labelling.base import LabellingError as _LabellingError
from talos.data.labelling.schedule.closed_world import ClosedWorldResolver
from talos.data.labelling.schedule.execution import ExecutionClassifier
from talos.data.labelling.schedule.join import IntervalJoinBuilder
from talos.data.labelling.schedule.manifest import AttackRule, ManifestLoader
from talos.data.labelling.taxonomy import DEFAULT_TAXONOMY, TaxonomyMapper
from talos.data.labelling.schedule.validate import Burst, LabelValidator
from talos.data.labelling.schedule.writer import LabelWriter


# Defined with the contract, not here: `talos label` catches one type whichever
# method ran, and two classes with this name would silently not be each other.
LabellingError = _LabellingError


# What the join and the execution classifier read off `conn`. Checked from
# footers before any scan starts: a capability declaration says what an
# extractor CAN emit, this says what this particular tree actually has.
REQUIRED_COLUMNS = ("ts", "uid", "id.orig_h", "id.resp_h")


@dataclass
class LabellingReport:
    dataset: str
    feature_space: str
    source: str
    method: str = "schedule"
    total_flows: int = 0
    attack_flows: int = 0
    per_rule: tuple = ()
    silent_rules: tuple = ()
    overlaps: int = 0
    quarantined: int = 0
    gate: GateResult | None = None
    output: str | None = None
    manifest_sha: str = ""
    taxonomy_sha: str = ""
    method_sha: str = ""
    labelled_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def attack_ratio(self) -> float:
        return self.attack_flows / self.total_flows if self.total_flows else 0.0

    @property
    def benign_flows(self) -> int:
        return self.total_flows - self.attack_flows

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset, "feature_space": self.feature_space,
            "source": self.source, "output": self.output,
            "total_flows": self.total_flows, "attack_flows": self.attack_flows,
            "benign_flows": self.benign_flows, "attack_ratio": self.attack_ratio,
            "overlaps": self.overlaps, "quarantined": self.quarantined,
            "silent_rules": [r.label() for r in self.silent_rules],
            "per_rule": [dict(zip(
                ("class", "raw", "rid", "flows", "overlaps", "executed", "attempted"), r))
                for r in self.per_rule],
            "label_method": self.method, "method_sha": self.method_sha,
            "manifest_sha": self.manifest_sha, "taxonomy_sha": self.taxonomy_sha,
            "labelled_at": self.labelled_at,
            "gate": self.gate.to_dict() if self.gate else None,
            **self.extra,
        }

    def summary(self) -> str:
        """What only a schedule run has. The CLI prints it without knowing."""
        from talos.data.labelling.schedule import MANIFESTS
        lines = [f"manifest      {self.dataset} {self.manifest_sha}"
                 f"{'' if MANIFESTS.origin_of(self.dataset).built_in else '  [substituted]'}",
                 f"taxonomy      {self.taxonomy_sha}"]
        if self.gate:
            lines.append(f"\n{self.gate.report()}")
        if self.overlaps:
            lines.append(f"note: {self.overlaps:,} flow(s) matched >1 rule; "
                         f"lowest rule id wins")
        return "\n".join(lines)

    def table(self) -> str:
        """The human-facing summary, in the shape the previous runs printed."""
        head = (f"{'canonical class':<15}{'raw attack name':<25}"
                f"{'flows':>13}{'executed':>11}{'attempted':>11}")
        lines = [head]
        for cls, raw, _rid, n, _ovl, ex, att in self.per_rule:
            note = "  <-- mostly attempts" if att and att > n / 2 else ""
            lines.append(f"{cls:<15}{raw:<25}{n:>13,}"
                         f"{(f'{ex:,}' if ex else '-'):>11}"
                         f"{(f'{att:,}' if att else '-'):>11}{note}")
        lines += [
            "-" * 56,
            f"{'ATTACK':<42}{self.attack_flows:>14,}",
            f"{'BENIGN (closed-world)':<42}{self.benign_flows:>14,}",
            f"{'TOTAL':<42}{self.total_flows:>14,}",
            f"\nattack ratio {100 * self.attack_ratio:.3f}%",
        ]
        return "\n".join(lines)


class LabellingEngine:
    """`label(dataset) -> LabellingReport`."""

    #: Stamped on every row and used to scope the output path. The adapter in
    #: `labeller.py` declares the same string; a test asserts they agree.
    method = "schedule"

    def __init__(self, cfg, lake=None, manifests: str | Path | None = None,
                 provenance: ProvenanceService | None = None,
                 taxonomy: str | Path = DEFAULT_TAXONOMY, space=None,
                 method: str | None = None):
        self.cfg = cfg
        self.lake = lake if lake is not None else cfg.lake()
        self.manifests = ManifestLoader(manifests) if manifests else ManifestLoader()
        self.provenance = provenance or ProvenanceService(cfg.reports)
        self.taxonomy_name = taxonomy
        self.space = space
        # Set by the adapter, which knows about the declaration and its label
        # space. Without one the engine hashes only what it can see.
        self.sha_inputs: tuple = ()
        if method:
            self.method = method
        self.join = IntervalJoinBuilder()
        self.execution = ExecutionClassifier()
        self.writer = LabelWriter(self.execution, ClosedWorldResolver(),
                                  self.provenance, space=space)

    # ------------------------------------------------------------------ run

    def label(self, dataset: str, feature_space: str, source: str | None = None,
              write: bool = True, extractor=None,
              force: bool = False) -> LabellingReport:
        # What the extractor can supply decides what this stage may claim. Two
        # capabilities are structural -- without a conn backbone and a stable
        # flow id there is nothing to join a schedule onto, so the run refuses.
        # Byte counts are different: their absence degrades one column rather
        # than the whole stage, so it is recorded and carried on.
        can_tell_executed = True
        if extractor is not None:
            extractor.require(CONN_BACKBONE, FLOW_ID)
            can_tell_executed = extractor.supports(BYTE_COUNTS)

        taxonomy = TaxonomyMapper(self.taxonomy_name)
        manifest = self.manifests.load(dataset, taxonomy)
        regions = manifest.regions

        src, base = self._source(dataset, feature_space, source)
        report = LabellingReport(
            dataset=dataset, feature_space=feature_space, source=src,
            method=self.method,
            manifest_sha=content_sha(manifest.path),
            taxonomy_sha=content_sha(taxonomy.path),
            # One hash over BOTH inputs. It is what lands on every row; the two
            # component hashes stay in this report, where a human reads them.
            method_sha=composite_sha(*(self.sha_inputs or
                                       (manifest.path, taxonomy.path))),
            labelled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            quarantined=len(regions))
        if not can_tell_executed:
            # Never silently mark every payload-bearing attack "attempted": say
            # the measurement was unavailable, in the report and on every row.
            report.extra["label_executed"] = (
                f"NULL for every row: {extractor.name} declares no byte counts, "
                f"so an executed attack cannot be told from a connection attempt")

        duck = self.lake.duck
        self._preflight(duck, src, dataset, feature_space,
                        self.cfg.source_of(dataset), can_tell_executed)
        ctes = self.join.base_ctes(manifest.rules, src, base, regions)

        # `union_by_name` here too: every other read of this glob uses it, and a
        # schema-divergent capture would otherwise error on the count while
        # succeeding everywhere else.
        report.total_flows = duck.one(
            f"SELECT count(*) FROM read_parquet('{src}', union_by_name = true)")[0]
        report.per_rule = tuple(duck.sql(
            self._summary_query(ctes, measurable=can_tell_executed)))
        report.attack_flows = sum(r[3] for r in report.per_rule)
        report.overlaps = sum(r[4] for r in report.per_rule)

        fired = {r[2] for r in report.per_rule}
        report.silent_rules = tuple(r for r in manifest.rules if r.rid not in fired)

        report.gate = self._validate(duck, ctes, manifest.rules, report)
        self.provenance.run_report(f"labelling_{dataset}", report.to_dict())
        ValidationGate.raise_on_fail(report.gate)

        if write:
            rel = duck.relation(self.writer.build_query(
                ctes, bool(regions), measurable=can_tell_executed))
            rel = self.writer.stamp(rel, dataset, report.method, report.method_sha,
                                    report.labelled_at)
            report.output = self.writer.write(
                self.lake, rel, dataset, feature_space, method=report.method,
                method_sha=report.method_sha, source=self.cfg.source_of(dataset),
                force=force)
            self.provenance.run_report(f"labelling_{dataset}", report.to_dict())
        return report

    # -------------------------------------------------------------- helpers

    def _source(self, dataset: str, feature_space: str,
                source: str | None) -> tuple[str, str]:
        """(glob, base). `base` must describe the tree actually being read.

        `capture` is recovered by stripping `base` off each parquet's path, so a
        mismatched base does not fail — it silently yields empty captures, and
        with them the ability to split without leakage.
        """
        if source:
            base = source.split("**")[0] if "**" in source else source.rsplit("/", 1)[0] + "/"
            return source, base
        # The dataset's own declaration decides which source tree it lives in --
        # asking Config rather than defaulting means pointing this at a netem run
        # resolves to netem's parquet zone, not the public-dataset one.
        zone = self.lake.uri("parquet", dataset=dataset, feature_space=feature_space,
                             source=self.cfg.source_of(dataset))
        return f"{zone}/**/conn.parquet", f"{zone}/"

    def _preflight(self, duck, src: str, dataset: str, feature_space: str,
                   source: str, measurable: bool) -> None:
        """Fail usefully before three scans, not usefully-ish during the first.

        Without this the common mistakes -- never converted, wrong feature
        space, dataset filed under a different source -- all surface as the same
        raw DuckDB "No files found that match the pattern", which names a path
        and nothing else.
        """
        zone = self.lake.uri("parquet", dataset=dataset, source=source,
                             feature_space=feature_space)
        if not self.lake.list(zone):
            raise LabellingError(self._nothing_to_label(dataset, source, feature_space))

        columns = duck.columns(src)
        wanted = list(REQUIRED_COLUMNS) + (["orig_bytes"] if measurable else [])
        missing = [c for c in wanted if c not in columns]
        if missing:
            raise LabellingError(
                f"{src} has no {', '.join(missing)}. The schedule join reads "
                f"Zeek's conn backbone; this tree does not look like conn.")

    def _nothing_to_label(self, dataset: str, source: str, feature_space: str) -> str:
        """Name what IS there, since the answer is usually one of three things."""
        spaces = sorted({
            uri.split(f"/{dataset}/")[0].rstrip("/").rsplit("/", 1)[-1]
            for uri in self.lake.list(self.lake.uri("parquet", source=source))
            if f"/{dataset}/" in uri})
        if spaces:
            return (f"no parquet for {dataset!r} under feature space "
                    f"{feature_space!r}, but it exists under: {', '.join(spaces)}. "
                    f"Pass --feature-space, or check the extractor config.")

        elsewhere = [other for other in zones.SOURCES if other != source
                     and self.lake.list(self.lake.uri("parquet", dataset=dataset,
                                                      source=other))]
        if elsewhere:
            return (f"no parquet for {dataset!r} under source {source!r}, but it "
                    f"exists under: {', '.join(elsewhere)}. Declare the source in "
                    f"config.yml under datasets.{dataset}.source.")

        return (f"nothing to label: no parquet for {dataset!r}. "
                f"Run `talos convert --dataset {dataset}` first.")

    def _summary_query(self, ctes: str, measurable: bool = True) -> str:
        """Attack flows only. Benign is total minus attack, and total is free.

        A LEFT JOIN over every flow in the dataset would cost a full extra pass
        to learn a number parquet footers already know.
        """
        return f"""{ctes}
      SELECT m.rclass, m.rname, m.rid, count(*) AS n,
             sum(CASE WHEN m.nrules > 1 THEN 1 ELSE 0 END) AS overlaps,
             {self.execution.executed_count_sql(measurable=measurable)} AS executed,
             {self.execution.attempted_count_sql(measurable=measurable)} AS attempted
      FROM matched m JOIN conn c USING (uid)
      GROUP BY 1, 2, 3 ORDER BY 4 DESC"""

    def _breadth_query(self, ctes: str) -> str:
        return f"""{ctes}
      SELECT m.rname, count(DISTINCT c."id.orig_h") AS srcs,
             count(DISTINCT c."id.resp_h") AS dsts
      FROM matched m JOIN conn c USING (uid)
      GROUP BY 1 ORDER BY srcs + dsts DESC LIMIT 1"""

    @staticmethod
    def victim_predicate(rules) -> str | None:
        """"Does this flow involve a known victim?", or None if unanswerable.

        Built from victim IPs AND victim subnets. Omitting subnets used to make
        2017's Infiltration-PortScan target range invisible to the audit -- the
        one rule whose victims are a /24 rather than a list.

        None when a manifest declares no victims at all. The previous version
        substituted an empty IN-list, which is valid SQL that matches nothing,
        so every bucket scored zero victim involvement and the check quietly
        passed without ever being able to fail.
        """
        ips = sorted({ip for r in rules for ip in (r.victims or ())})
        prefixes = sorted({r.subnet_prefix for r in rules if r.subnet_prefix})
        if not ips and not prefixes:
            return None

        tests = []
        if ips:
            inlist = ", ".join(f"'{v}'" for v in ips)
            tests.append(f'c."id.orig_h" IN ({inlist}) OR c."id.resp_h" IN ({inlist})')
        for prefix in prefixes:
            tests.append(f"starts_with(c.\"id.orig_h\", '{prefix}') "
                         f"OR starts_with(c.\"id.resp_h\", '{prefix}')")
        return " OR ".join(f"({t})" for t in tests)

    def _benign_query(self, ctes: str, rules) -> str:
        """Ten-minute buckets of unmatched traffic, ranked by size."""
        predicate = self.victim_predicate(rules)
        return f"""{ctes}
      SELECT c.capture,
             strftime(to_timestamp(floor(c.ts/600)*600) AT TIME ZONE 'UTC',
                      '%Y-%m-%d %H:%M') AS bucket,
             count(*) AS n,
             sum(CASE WHEN {predicate} THEN 1 ELSE 0 END) AS victim_flows
      FROM conn c LEFT JOIN matched m USING (uid)
      WHERE m.rid IS NULL
      GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10"""

    def _validate(self, duck, ctes: str, rules: tuple[AttackRule, ...],
                  report: LabellingReport) -> GateResult:
        distinct = duck.one(f"{ctes} SELECT count(DISTINCT uid) FROM conn")[0]
        widest = duck.sql(self._breadth_query(ctes))
        can_audit = self.victim_predicate(rules) is not None
        buckets = ([Burst(*row) for row in duck.sql(self._benign_query(ctes, rules))]
                   if can_audit else [])

        ctx = {
            "rules_total": len(rules),
            "silent_rules": report.silent_rules,
            "total_flows": report.total_flows,
            "distinct_uids": distinct,
            # Not a null-label scan. Under closed world `label_class` is
            # `coalesce(rclass, 'benign')` and `rclass` cannot be empty -- a
            # rule without a class is refused at construction -- so nulls are
            # unrepresentable and a check for them could never fail. This is the
            # arithmetic instead: more attack flows than flows means the join
            # double-counted, which is what a non-unique uid looks like at the
            # summary level. A row-level scan belongs here once uid propagation
            # lands and the output is more than one table.
            "attack_flows": report.attack_flows,
            "widest_rule": widest[0] if widest else None,
            "suspect_bursts": LabelValidator.suspect(buckets, report.total_flows),
            "benign_audit_possible": can_audit,
        }
        report.extra["distinct_uids"] = distinct
        return LabelValidator.gate().run_all(ctx)
