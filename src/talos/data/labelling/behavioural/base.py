"""What every behavioural branch has in common: pools, prediction, the table.

The branch supplies parts and an objective; this owns the sequence — read D_l
and D_s, train or load, predict over the whole dataset, write. Same division as
`LabellingEngine`: every branch here, no algorithm.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Iterator

import numpy as np

from talos.common.provenance import ProvMeta, ProvenanceService, composite_sha
from talos.data.feature.spec import DEFAULT_FEATURES, FeatureSetLoader
from talos.data.feature.vectorise import DEFAULT_BATCH, build as build_vectoriser
from talos.data.labelling.base import (
    CORE_COLUMNS, LABELLERS, LabellingError, LabellingMethod, NO_WEIGHT,
    PREDICTED, StaleOutputError, verify_schema,
)
from talos.data.labelling.behavioural.pool import PartitionLoader
from talos.data.labelling.space import OUT_OF_SPACE
from talos.data.labelling.taxonomy import BENIGN

CERTAIN, UNCERTAIN = "certain", "uncertain"

#: Pools these methods expect a partition to declare.
LARGE, SMALL = "d_l", "d_s"


def labelling_columns() -> set[str]:
    """Every column a labelling method may own: core, plus each method's extras.

    Path B reads a table another method wrote, so its input already carries a
    full set of labels. They are replaced, not merged: a `rule_id` sitting in an
    `ae-v1` table would describe a rule that had nothing to do with the row.
    """
    owned = set(CORE_COLUMNS)
    for name in LABELLERS.names():
        owned.update(getattr(LABELLERS.cls_for(name), "extras", ()))
    return owned


@dataclass
class BehaviouralReport:
    """What a branch did, and how much of it can be believed."""

    dataset: str
    feature_space: str
    source: str
    method: str
    method_sha: str = ""
    labelled_at: str = ""
    total_flows: int = 0
    per_class: tuple = ()
    mean_confidence: float | None = None
    trainable: bool = True
    features: str = ""
    partition: str = ""
    d_l_rows: int = 0
    d_s_rows: int = 0
    d_s_dropped: int = 0
    pretrain: tuple = ()
    finetune: tuple = ()
    warnings: tuple[str, ...] = ()
    output: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset, "feature_space": self.feature_space,
            "source": self.source, "label_method": self.method,
            "method_sha": self.method_sha, "labelled_at": self.labelled_at,
            "output": self.output, "total_flows": self.total_flows,
            "per_class": [dict(zip(("class", "flows"), row)) for row in self.per_class],
            "mean_confidence": self.mean_confidence,
            "trainable": self.trainable, "feature_set": self.features,
            "partition": self.partition, "d_l_rows": self.d_l_rows,
            "d_s_rows": self.d_s_rows, "d_s_dropped": self.d_s_dropped,
            "pretrain": [e.loss for e in self.pretrain],
            "finetune": [e.loss for e in self.finetune],
            "warnings": list(self.warnings),
            # Named so nobody has to infer it from `trainable` being false.
            "labels_are_meaningless": not self.trainable,
            **self.extra,
        }

    def table(self) -> str:
        lines = [f"{'class':<16}{'flows':>14}"]
        lines += [f"{cls:<16}{n:>14,}" for cls, n in self.per_class]
        lines += ["-" * 30, f"{'TOTAL':<16}{self.total_flows:>14,}"]
        if self.mean_confidence is not None:
            lines.append(f"\nmean confidence {self.mean_confidence:.4f}")
        return "\n".join(lines)

    def summary(self) -> str:
        """What only a behavioural run has. The CLI prints it without knowing."""
        # "not trained" and "the pools are empty" are different facts, and zeros
        # would say the second when the first is true.
        drawn = (f"D_l {self.d_l_rows:,}  D_s {self.d_s_rows:,}"
                 + (f"  ({self.d_s_dropped:,} out of space)" if self.d_s_dropped
                    else "")) if (self.pretrain or self.finetune) else "not drawn"
        lines = [f"feature set   {self.features}",
                 f"pools         {self.partition}  {drawn}"]
        if self.pretrain:
            lines.append(f"pretrain      {len(self.pretrain)} epochs, loss "
                         f"{self.pretrain[0].loss:.5f} -> {self.pretrain[-1].loss:.5f}")
        if self.finetune:
            lines.append(f"finetune      {len(self.finetune)} epochs, loss "
                         f"{self.finetune[0].loss:.5f} -> {self.finetune[-1].loss:.5f}")
        if "checkpoint" in self.extra:
            lines.append(f"checkpoint    {self.extra['checkpoint']}")
        for warning in self.warnings:
            lines.append(f"\n!! {warning}")
        return "\n".join(lines)


class BehaviouralMethod(LabellingMethod):
    """Path B: learn from the traffic, then say what each flow is."""

    #: Every branch reports its per-row reconstruction/uncertainty differently.
    extras: ClassVar[tuple[str, ...]] = ()

    #: The subset of `extras` that varies per row; the rest are SQL literals.
    row_extra_names: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg, lake=None, spec=None, partition=None, features=None,
                 vectoriser=None, checkpoints: Path | None = None,
                 allow_untrained: bool = False, **kwargs):
        super().__init__(cfg, lake, spec=spec, allow_untrained=allow_untrained)
        self.settings = dict(spec.settings) if spec else {}
        self.space = spec.label_space if spec else None
        if self.space is None:
            raise LabellingError(
                f"{self.name} needs a label space: it decides how many classes the "
                f"classifier can express. Run it through a method declaration.")

        self.partition = partition or PartitionLoader().load(
            self.settings.get("pools", "xdg-v3"))
        self.features = features or FeatureSetLoader().load(
            self.settings.get("features", DEFAULT_FEATURES))
        self.vectoriser = vectoriser or build_vectoriser(
            self.settings.get("vectoriser", "passthrough"), self.features)
        self.batch = int(self.settings.get("batch", DEFAULT_BATCH))
        self.provenance = ProvenanceService(cfg.reports)
        self.checkpoints = Path(checkpoints) if checkpoints else cfg.models
        self.parts: dict = {}

    # ------------------------------------------------------------ provenance

    def inputs(self, dataset: str) -> tuple:
        """Everything that decides these labels. The pool is one of them.

        Re-seeding the partition changes which rows trained the model, so two
        runs that shared a `method_sha` without it would not be the same run.
        """
        base = self.spec.inputs() if self.spec else ()
        return (*base, self.partition.path, self.features.path)

    def method_sha(self, dataset: str) -> str:
        return composite_sha(*self.inputs(dataset))

    def checkpoint_dir(self, dataset: str) -> Path:
        """Keyed by the hash, so a changed input cannot load a stale model."""
        return self.checkpoints / self.run_name / self.method_sha(dataset)

    # -------------------------------------------------------------- contract

    @abstractmethod
    def build_parts(self, width: int) -> dict:
        """The parts this branch needs, given the matrix width."""

    @abstractmethod
    def pretrain(self, batches, report: BehaviouralReport) -> tuple:
        """Self-supervised stage over D_l. Returns the epoch history."""

    @abstractmethod
    def finetune(self, X, y, report: BehaviouralReport) -> tuple:
        """Supervised stage over D_s. Returns the epoch history."""

    @abstractmethod
    def predict_proba(self, X):
        """(n, width) -> (n, n_classes)."""

    def row_extras(self, X) -> dict[str, np.ndarray]:
        """Per-row extra columns, aligned with `X`."""
        return {}

    def literal_extras(self) -> dict[str, str]:
        """Extra columns that are the same for every row, as SQL literals."""
        return {}

    # ------------------------------------------------------------------- run

    def label(self, dataset: str, feature_space: str, source: str | None = None,
              write: bool = True, extractor=None,
              force: bool = False) -> BehaviouralReport:
        self.check_extractor(extractor)
        duck = self.lake.duck
        report = BehaviouralReport(
            dataset=dataset, feature_space=feature_space,
            source=source or self.cfg.source_of(dataset), method=self.run_name,
            method_sha=self.method_sha(dataset),
            labelled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            features=f"{self.features.name} {self.features.sha}",
            partition=f"{self.partition.name} {self.partition.sha}")

        self.fit(duck, feature_space, dataset, report)
        target = self._target(duck, dataset, feature_space, report)
        self._predict_into(duck, target, report)

        self.provenance.run_report(f"labelling_{dataset}_{self.run_name}",
                                   report.to_dict())
        if write:
            report.output = self._write(duck, target, dataset, feature_space,
                                        report, force=force)
            self.provenance.run_report(f"labelling_{dataset}_{self.run_name}",
                                       report.to_dict())
        return report

    # ------------------------------------------------------------- the model

    def fit(self, duck, feature_space: str, dataset: str,
            report: BehaviouralReport) -> None:
        """Load a checkpoint built from these inputs, or train and save one."""
        sources = self.partition.sources(self.lake, feature_space, self.cfg)
        width = self.vectoriser.columns().width
        self.parts = self.build_parts(width)
        report.trainable = all(part.trainable for part in self.parts.values())

        if not report.trainable:
            self._refuse_or_warn(report)
            return

        directory = self.checkpoint_dir(dataset)
        if directory.is_dir() and any(directory.iterdir()):
            for name, part in self.parts.items():
                part.load(directory / f"{name}.pt")
            report.extra["checkpoint"] = str(directory)
            return

        report.d_l_rows = self._count(duck, sources, LARGE)
        report.pretrain = self.pretrain(
            lambda: self._batches(duck, sources, LARGE), report)
        X, y = self._supervised(duck, sources, report)
        report.finetune = self.finetune(X, y, report)

        directory.mkdir(parents=True, exist_ok=True)
        for name, part in self.parts.items():
            part.save(directory / f"{name}.pt")
        report.extra["checkpoint"] = str(directory)

    def _refuse_or_warn(self, report: BehaviouralReport) -> None:
        """An untrained model must not quietly produce a table of labels."""
        message = (
            f"{self.run_name} is running on parts that cannot learn "
            f"({', '.join(sorted(p.name for p in self.parts.values()))}). Its "
            f"output is not a prediction. Install PyTorch with "
            f"`pip install -e '.[torch]'`.")
        if not self.allow_untrained:
            raise LabellingError(message + " Pass --allow-untrained to write the "
                                           "table anyway, marked uncertain.")
        report.warnings = (*report.warnings, message)

    # -------------------------------------------------------------- the data

    def _count(self, duck, sources, pool: str) -> int:
        rel = self.partition.relation(duck, pool, sources)
        return rel.aggregate("count(*)").fetchone()[0]

    def _batches(self, duck, sources, pool: str) -> Iterator:
        """Feature batches for one pool, matrix only."""
        rel = self.partition.relation(duck, pool, sources)
        for _keys, X in self.vectoriser.batches(rel, self.batch):
            yield X

    def _supervised(self, duck, sources, report: BehaviouralReport):
        """D_s as (X, y). Rows outside the label space are dropped and counted."""
        rel = self.partition.relation(duck, SMALL, sources)
        _keys, X, raw = self.vectoriser.labelled_matrix(rel, "label_class")
        keep = np.array([label in self.space for label in raw], dtype=bool)
        report.d_s_rows = int(keep.sum())
        report.d_s_dropped = int((~keep).sum())
        if not report.d_s_rows:
            raise LabellingError(
                f"D_s is empty after dropping classes outside label space "
                f"{self.space.name!r}. Nothing can be fine-tuned on it.")
        y = np.array([self.space.index(label) for label in raw[keep]], dtype=np.int64)
        return X[keep], y

    def _target(self, duck, dataset: str, feature_space: str,
                report: BehaviouralReport):
        """The rows to label: this dataset's whole table, not just its pools."""
        uri = self.lake.uri("labelled", dataset=dataset, feature_space=feature_space,
                            method=self.partition.labels,
                            source=self.cfg.source_of(dataset), rel="conn.parquet")
        if not self.lake.exists(uri):
            raise LabellingError(
                f"nothing to label: {uri} does not exist. Path B reads the table "
                f"{self.partition.labels!r} wrote, so run that method first.")
        rel = duck.relation(f"SELECT * FROM read_parquet('{uri}', union_by_name = true)")
        report.total_flows = rel.aggregate("count(*)").fetchone()[0]
        return rel

    # -------------------------------------------------------- the prediction

    def _predict_into(self, duck, target, report: BehaviouralReport) -> None:
        """Predict in batches into a temp table, so memory does not scale with the lake."""
        extras = self.row_extra_names
        columns = ", ".join(f'"{name}" DOUBLE' for name in extras)
        duck.sql(f"CREATE OR REPLACE TEMP TABLE preds ("
                 f"uid VARCHAR, label_class VARCHAR, label_confidence DOUBLE"
                 f"{', ' + columns if columns else ''})")

        placeholders = ", ".join("?" * (3 + len(extras)))
        confidences: list[float] = []
        for keys, X in self.vectoriser.batches(target, self.batch):
            proba = np.asarray(self.predict_proba(X), dtype=np.float64)
            chosen = proba.argmax(axis=1)
            confidence = proba.max(axis=1)
            values = self.row_extras(X)
            rows = [
                (str(keys[i]), self.space.classes[chosen[i]], float(confidence[i]),
                 *(float(values[name][i]) for name in extras))
                for i in range(len(keys))]
            duck.con.executemany(f"INSERT INTO preds VALUES ({placeholders})", rows)
            confidences.extend(confidence.tolist())

        report.mean_confidence = (sum(confidences) / len(confidences)
                                  if confidences else None)
        report.per_class = tuple(duck.sql(
            "SELECT label_class, count(*) FROM preds GROUP BY 1 ORDER BY 2 DESC"))

    # ----------------------------------------------------------- the writing

    def quality_sql(self) -> str:
        """Out-of-space cannot occur — the classifier only emits in-space classes."""
        trainable = self.parts and all(p.trainable for p in self.parts.values())
        return f"'{CERTAIN}'" if trainable else f"'{UNCERTAIN}'"

    def select_list(self, target) -> str:
        drop = sorted(set(target.columns) & labelling_columns())
        exclude = f" EXCLUDE ({', '.join(drop)})" if drop else ""
        extras = "".join(f",\n             p.\"{name}\"" for name in self.row_extra_names)
        extras += "".join(f",\n             {value} AS \"{name}\""
                          for name, value in self.literal_extras().items())
        return (f"c.*{exclude},\n"
                f"             c.capture AS capture,\n"
                f"             p.label_class != '{BENIGN}' AS label_binary,\n"
                f"             p.label_class,\n"
                f"             '{PREDICTED}' AS label_source,\n"
                f"             p.label_confidence,\n"
                f"             {NO_WEIGHT} AS label_weight,\n"
                f"             {self.quality_sql()} AS label_quality{extras}")

    def _write(self, duck, target, dataset: str, feature_space: str,
               report: BehaviouralReport, force: bool = False) -> str:
        rel = duck.relation(
            f"SELECT {self.select_list(target)} FROM ({target.sql_query()}) c "
            f"JOIN preds p ON c.uid = p.uid")
        rel = self.provenance.stamp(rel, ProvMeta.of(
            label_method=report.method, method_sha=report.method_sha,
            dataset=dataset, labelled_at=report.labelled_at))
        verify_schema(rel, self.extras)

        uri = self.output_uri(dataset, feature_space,
                              source=self.cfg.source_of(dataset))
        self._refuse_stale(uri, report.method_sha, force)
        return self.lake.write_parquet(rel, uri)

    def _refuse_stale(self, uri: str, method_sha: str, force: bool) -> None:
        """Never silently replace a table built from different inputs."""
        if force or not self.lake.exists(uri):
            return
        try:
            row = self.lake.read_parquet(uri, columns=["method_sha"]).limit(1).fetchone()
        except Exception:                                # noqa: BLE001
            return
        previous = row[0] if row else None
        if previous and previous != method_sha:
            raise StaleOutputError(
                f"{uri}\n  was labelled by {self.run_name} {previous}, and this run "
                f"used {method_sha}.\n  An input changed -- the declaration, the "
                f"label space, the feature set or the pools.\n  Pass --force if "
                f"that is intended.")
