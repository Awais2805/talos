"""Reconcile two branches into one label, then weigh it.

Eslami & Hamouda Eq. 17: agree, else higher confidence, else larger margin —
whose line 4 is `>=`, so the FIRST input takes an exact tie. Reads written
tables, not an in-process handoff, so either branch can be re-run alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar

import numpy as np

from talos.common.provenance import ProvMeta, ProvenanceService, composite_sha
from talos.data.labelling.base import (
    FUSED, LABELLERS, LabellingError, LabellingMethod, StaleOutputError,
    verify_schema,
)
from talos.data.labelling.behavioural.base import CERTAIN, UNCERTAIN, labelling_columns
from talos.data.labelling.fusion.confident import ConfidentLearning, NoiseReport
from talos.data.labelling.taxonomy import BENIGN


class FusionError(LabellingError):
    pass


@dataclass
class FusionReport:
    """What was reconciled, and what the noise estimate made of it."""

    dataset: str
    feature_space: str
    source: str
    method: str
    inputs: tuple[str, ...] = ()
    method_sha: str = ""
    labelled_at: str = ""
    total_flows: int = 0
    agreed: int = 0
    per_class: tuple = ()
    noise: NoiseReport | None = None
    warnings: tuple[str, ...] = ()
    output: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def agreement_rate(self) -> float:
        """How often the branches said the same thing. NOT an accuracy."""
        return self.agreed / self.total_flows if self.total_flows else 0.0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset, "feature_space": self.feature_space,
            "source": self.source, "label_method": self.method,
            "inputs": list(self.inputs), "method_sha": self.method_sha,
            "labelled_at": self.labelled_at, "output": self.output,
            "total_flows": self.total_flows, "branches_agreed": self.agreed,
            "branch_agreement_rate": self.agreement_rate,
            "per_class": [dict(zip(("class", "flows"), row)) for row in self.per_class],
            "noise": self.noise.to_dict() if self.noise else None,
            "warnings": list(self.warnings),
            "note": ("branch_agreement_rate is agreement between two models, not "
                     "accuracy: neither branch is ground truth."),
            **self.extra,
        }

    def table(self) -> str:
        lines = [f"{'class':<16}{'flows':>14}"]
        lines += [f"{cls:<16}{n:>14,}" for cls, n in self.per_class]
        lines += ["-" * 30, f"{'TOTAL':<16}{self.total_flows:>14,}"]
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [f"inputs        {' + '.join(self.inputs)}",
                 f"branches agreed on {self.agreed:,} of {self.total_flows:,} flows "
                 f"({100 * self.agreement_rate:.2f}%) — agreement, not accuracy"]
        if self.noise:
            lines += ["", self.noise.describe()]
            if self.noise.lifted:
                lines.append(f"!! balanced retention lifted: "
                             f"{', '.join(self.noise.lifted)}")
        for warning in self.warnings:
            lines.append(f"\n!! {warning}")
        return "\n".join(lines)


@LABELLERS.register
class FusedLabeller(LabellingMethod):
    """Two branches in, one weighted label out."""

    name: ClassVar[str] = "fused"
    extras: ClassVar[tuple[str, ...]] = ("branches_agreed", "label_margin")

    def __init__(self, cfg, lake=None, spec=None, allow_untrained: bool = False,
                 **kwargs):
        super().__init__(cfg, lake, spec=spec, allow_untrained=allow_untrained)
        self.settings = dict(spec.settings) if spec else {}
        self.space = spec.label_space if spec else None
        if self.space is None:
            raise FusionError(f"{self.name} needs a label space via a declaration")

        self.inputs_declared = tuple(self.settings.get("inputs") or ())
        if len(self.inputs_declared) != 2:
            raise FusionError(
                f"`inputs:` names {len(self.inputs_declared)} method(s); the "
                f"fusion rule (Eq. 17) is defined over exactly two. Order is "
                f"load-bearing: the FIRST wins any exact tie.")
        self.provenance = ProvenanceService(cfg.reports)

    # ------------------------------------------------------------ provenance

    def inputs(self, dataset: str) -> tuple:
        return self.spec.inputs() if self.spec else ()

    def method_sha(self, dataset: str) -> str:
        """Covers this declaration; the branches carry their own on every row."""
        return composite_sha(*self.inputs(dataset))

    # -------------------------------------------------------------- the rule

    def fusion_sql(self, first: str = "a", second: str = "b") -> str:
        """Eq. 17, verbatim. Line 4 is `>=`, so `first` takes an exact tie."""
        return (f"CASE\n"
                f"        WHEN {first}.label_class = {second}.label_class "
                f"THEN {first}.label_class\n"
                f"        WHEN {first}.label_confidence > {second}.label_confidence "
                f"THEN {first}.label_class\n"
                f"        WHEN {second}.label_confidence > {first}.label_confidence "
                f"THEN {second}.label_class\n"
                f"        WHEN {first}.label_margin >= {second}.label_margin "
                f"THEN {first}.label_class\n"
                f"        ELSE {second}.label_class END")

    def joined(self, duck, dataset: str, feature_space: str):
        """The two branch tables, inner-joined on uid, with the rule applied."""
        uris = [self._uri(dataset, feature_space, method)
                for method in self.inputs_declared]
        for method, uri in zip(self.inputs_declared, uris):
            if not self.lake.exists(uri):
                raise FusionError(
                    f"nothing to fuse: {uri} does not exist. Run "
                    f"`talos label --method {method} --dataset {dataset}` first.")
        self._check_spaces(duck, uris)

        chosen = self.fusion_sql()
        # Intersected with what the table actually has: EXCLUDE on an absent
        # column is a binder error, and a branch need not carry every extra.
        present = duck.relation(f"SELECT * FROM read_parquet('{uris[0]}')").columns
        drop = ", ".join(sorted(set(present) & labelling_columns()))
        return duck.relation(f"""
            SELECT a.* EXCLUDE ({drop}), a.capture AS capture,
                   {chosen} AS label_class,
                   (a.label_class = b.label_class) AS branches_agreed,
                   greatest(a.label_confidence, b.label_confidence)
                       AS label_confidence,
                   greatest(a.label_margin, b.label_margin) AS label_margin
            FROM read_parquet('{uris[0]}') a
            JOIN read_parquet('{uris[1]}') b ON a.uid = b.uid""")

    def _uri(self, dataset: str, feature_space: str, method: str) -> str:
        return self.lake.uri("labelled", dataset=dataset, feature_space=feature_space,
                             method=method, source=self.cfg.source_of(dataset),
                             rel="conn.parquet")

    def _check_spaces(self, duck, uris) -> None:
        """Two branches over different class sets cannot be reconciled at all.

        Compared by the classes actually present rather than by a stored sha,
        because the space is not a column: a branch that emitted a class this
        one has never heard of would otherwise fuse into an unpredictable label.
        """
        for uri in uris:
            found = {row[0] for row in duck.sql(
                f"SELECT DISTINCT label_class FROM read_parquet('{uri}')")}
            outside = sorted(found - set(self.space.classes))
            if outside:
                raise FusionError(
                    f"{uri} contains class(es) outside label space "
                    f"{self.space.name!r}: {', '.join(outside)}. Two branches "
                    f"labelled over different spaces cannot be fused.")

    # ------------------------------------------------------------------- run

    def label(self, dataset: str, feature_space: str, source: str | None = None,
              write: bool = True, extractor=None, force: bool = False,
              **kwargs) -> FusionReport:
        duck = self.lake.duck
        report = FusionReport(
            dataset=dataset, feature_space=feature_space,
            source=source or self.cfg.source_of(dataset), method=self.run_name,
            inputs=self.inputs_declared, method_sha=self.method_sha(dataset),
            labelled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

        fused = self.joined(duck, dataset, feature_space)
        duck.sql(f"CREATE OR REPLACE TEMP TABLE fused AS {fused.sql_query()}")
        report.total_flows = duck.one("SELECT count(*) FROM fused")[0]
        report.agreed = duck.one(
            "SELECT count(*) FROM fused WHERE branches_agreed")[0]
        report.per_class = tuple(duck.sql(
            "SELECT label_class, count(*) FROM fused GROUP BY 1 ORDER BY 2 DESC"))

        weights = self._weigh(duck, report)
        self.provenance.run_report(f"labelling_{dataset}_{self.run_name}",
                                   report.to_dict())
        if write:
            report.output = self._write(duck, dataset, feature_space, report,
                                        weights, force=force)
            self.provenance.run_report(f"labelling_{dataset}_{self.run_name}",
                                       report.to_dict())
        return report

    def _weigh(self, duck, report: FusionReport) -> bool:
        """Confident learning over the fused labels, if it can be run at all."""
        if not self.settings.get("confident_learning", True):
            report.warnings += ("confident learning disabled by declaration; "
                                "label_weight is NULL",)
            return False

        probabilities, labels = self._probabilities(duck, report)
        if probabilities is None:
            return False

        learner = ConfidentLearning(
            self.space, q=self.settings.get("q", 0.70),
            gamma=self.settings.get("gamma", 4.0),
            w_min=self.settings.get("w_min", 0.20),
            balanced=self.settings.get("balanced_retention", True))
        values, noise = learner.weights(probabilities, labels)
        report.noise = noise

        duck.sql("CREATE OR REPLACE TEMP TABLE weights (uid VARCHAR, w DOUBLE)")
        uids = [row[0] for row in duck.sql("SELECT uid FROM fused ORDER BY uid")]
        duck.con.executemany(
            "INSERT INTO weights VALUES (?, ?)",
            [(uid, float(w)) for uid, w in zip(uids, values)])
        return True

    def _probabilities(self, duck, report: FusionReport):
        """Out-of-sample probabilities for every fused row (Eq. 18).

        Returns `(None, None)` when they cannot be produced, rather than
        substituting the branches' own confidences: those are in-sample by
        construction and would make every threshold and every MAD optimistic.
        """
        from talos.data.labelling.fusion.probe import OutOfSampleProbe
        try:
            return OutOfSampleProbe(self, self.space, self.settings).run(duck, report)
        except LabellingError as exc:
            if not self.allow_untrained:
                raise
            report.warnings += (f"{exc} label_weight is NULL for every row.",)
            return None, None

    # ----------------------------------------------------------- the writing

    def select_list(self, weighted: bool) -> str:
        weight = "w.w" if weighted else "CAST(NULL AS DOUBLE)"
        quality = f"'{CERTAIN}'" if weighted else f"'{UNCERTAIN}'"
        return (f"f.* EXCLUDE (label_class, label_confidence, label_margin, "
                f"branches_agreed),\n"
                f"       f.label_class != '{BENIGN}' AS label_binary,\n"
                f"       f.label_class,\n"
                f"       '{FUSED}' AS label_source,\n"
                f"       f.label_confidence,\n"
                f"       {weight} AS label_weight,\n"
                f"       {quality} AS label_quality,\n"
                f"       f.branches_agreed, f.label_margin")

    def _write(self, duck, dataset: str, feature_space: str, report: FusionReport,
               weighted: bool, force: bool = False) -> str:
        join = " LEFT JOIN weights w ON w.uid = f.uid" if weighted else ""
        rel = duck.relation(
            f"SELECT {self.select_list(weighted)} FROM fused f{join}")
        rel = self.provenance.stamp(rel, ProvMeta.of(
            label_method=report.method, method_sha=report.method_sha,
            dataset=dataset, labelled_at=report.labelled_at))
        verify_schema(rel, self.extras)

        uri = self.output_uri(dataset, feature_space,
                              source=self.cfg.source_of(dataset))
        if self.lake.exists(uri) and not force:
            row = self.lake.read_parquet(uri, columns=["method_sha"]).limit(1).fetchone()
            if row and row[0] and row[0] != report.method_sha:
                raise StaleOutputError(
                    f"{uri}\n  was fused as {row[0]}, this run used "
                    f"{report.method_sha}. Pass --force if that is intended.")
        return self.lake.write_parquet(rel, uri)
