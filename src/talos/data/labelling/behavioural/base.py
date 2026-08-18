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
from talos.data.feature.featureset import DEFAULT_FEATURES, FeatureSetLoader
from talos.data.feature.vectorise import DEFAULT_BATCH, build as build_vectoriser
from talos.data.labelling.audit.adjudication import AdjudicationTable, UNKNOWN, default_path
from talos.data.labelling.base import (
    CORE_COLUMNS, LABELLERS, LabellingError, LabellingMethod, NO_WEIGHT,
    PREDICTED, StaleOutputError, verify_schema,
)
from talos.data.labelling.behavioural.pool import PartitionLoader
from talos.data.labelling.space import OUT_OF_SPACE
from talos.data.labelling.taxonomy import BENIGN
from talos.parts.mlp import require_torch

CERTAIN, UNCERTAIN = "certain", "uncertain"

#: Emitted by every behavioural branch: top-1 probability minus runner-up.
#: Fusion's tie-break (Eslami & Hamouda Eq. 17) needs it, and it cannot be
#: recovered from `label_confidence` alone once the row is written.
MARGIN = "label_margin"

#: Pools these methods expect a partition to declare.
LARGE, SMALL = "d_l", "d_s"


def labelling_columns() -> set[str]:
    """Every column a labelling method may own: core, plus each method's extras.

    Path B reads a table another method wrote, so its input already carries a
    full set of labels. They are replaced, not merged: a `rule_id` sitting in an
    `ae` table would describe a rule that had nothing to do with the row.
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
    d_s_unaudited: int = 0
    d_s_audited: bool = True
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
            "d_s_unaudited": self.d_s_unaudited,
            # Named so nobody has to infer it from a warning string -- a
            # --skip-audit-gate run's D_s is the raw schedule label, not a
            # person's verdict, same spirit as `labels_are_meaningless`.
            "d_s_audited": self.d_s_audited,
            "pretrain": [e.loss for e in self.pretrain],
            "finetune": [e.loss for e in self.finetune],
            # Rows-per-epoch alongside the loss list, kept as a second key
            # rather than reshaping `pretrain`/`finetune` -- a report reader
            # that expects a flat list of floats there must not break.
            "pretrain_rows": [e.rows for e in self.pretrain],
            "finetune_rows": [e.rows for e in self.finetune],
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
    #: `label_margin` is added by the base and need not be listed.
    row_extra_names: ClassVar[tuple[str, ...]] = ()

    @property
    def emitted_extras(self) -> tuple[str, ...]:
        return (MARGIN, *self.row_extra_names)

    def __init__(self, cfg, lake=None, spec=None, partition=None, features=None,
                 vectoriser=None, checkpoints: Path | None = None,
                 allow_untrained: bool = False, pools: str | None = None,
                 skip_audit_gate: bool = False, **kwargs):
        super().__init__(cfg, lake, spec=spec, allow_untrained=allow_untrained,
                         skip_audit_gate=skip_audit_gate)
        self.settings = dict(spec.settings) if spec else {}
        self.space = spec.label_space if spec else None
        if self.space is None:
            raise LabellingError(
                f"{self.name} needs a label space: it decides how many classes the "
                f"classifier can express. Run it through a method declaration.")

        self.partition = partition or PartitionLoader().load(
            pools or self.settings.get("pools", "xdg"))
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
        The adjudication file is included ONLY if it already exists AND
        `--skip-audit-gate` was not passed: hashing a path that `content_sha`
        cannot open would crash here, before `_audited_relation` gets a chance
        to raise the clear refusal, and a skipped-gate run does not actually
        read the file even if one happens to exist, so hashing it would name
        an input that played no part in the result.
        """
        base = self.spec.inputs() if self.spec else ()
        extra = (self.partition.path, self.features.path)
        audit = default_path(self.cfg, dataset)
        if not self.skip_audit_gate and audit.exists():
            extra = (*extra, audit)
        return (*base, *extra)

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
    def pretrain(self, batches, report: BehaviouralReport,
                 supervised: tuple | None = None, keyed=None, duck=None) -> tuple:
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
        # D_s is resolved BEFORE pretraining: Algorithm 2 line 1 bootstraps the
        # pseudo-labels its augmentation conditions on from a classifier trained
        # on D_s, and lines 4-5 refresh them from a probe over D_s embeddings.
        # A branch that does not need it (the AE) simply ignores the argument.
        X, y = self._supervised(duck, sources, report)
        report.pretrain = self.pretrain(
            lambda: self._batches(duck, sources, LARGE), report, supervised=(X, y),
            keyed=lambda: self._keyed_batches(duck, sources, LARGE), duck=duck)
        # Which loss produced these numbers, on the row beside them: weighted
        # and unweighted fine-tuning give very different per-class results and
        # the two are not otherwise distinguishable in a report.
        report.extra["class_weighting"] = bool(
            self.settings.get("class_weighting", False))
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

    def _keyed_batches(self, duck, sources, pool: str) -> Iterator:
        """The same batches, with the key each row joins back on.

        Alg. 2 line 5 reclassifies all of D_l and stops on the label-change
        FRACTION, which can only be counted by matching rows between rounds.
        A scan has no order guarantee, so the match is by uid, not by position.
        """
        rel = self.partition.relation(duck, pool, sources)
        yield from self.vectoriser.batches(rel, self.batch)

    def _audited_relation(self, duck, rel, report: BehaviouralReport):
        """`D_s`, with `label_class` replaced by a person's verdict.

        Refuses by default: a behavioural method's fine-tune set must be
        human-verified. `label_class` on the raw pool is the SCHEDULE's
        rule-based guess, the exact thing the audit process exists to check --
        fine-tuning on it directly trains the classifier to reproduce the
        schedule's own mistakes with no correction step, which is what
        actually happened before this gate. `--skip-audit-gate` overrides
        this per-run (harness/already-verified-labels use), same shape as
        `--allow-untrained`: recorded loudly in the report, never silent.
        """
        if self.skip_audit_gate:
            report.d_s_audited = False
            report.warnings = (*report.warnings,
                f"{self.run_name} ran with --skip-audit-gate: D_s is the raw "
                f"schedule label, not a human-verified one. Fine-tuning on it "
                f"can reproduce the schedule's own mistakes with no "
                f"correction step.")
            return rel
        path = default_path(self.cfg, report.dataset)
        if not path.exists():
            raise LabellingError(
                f"{self.run_name} needs an audited D_s and none exists at "
                f"{path}. Run `talos audit emit --dataset {report.dataset} "
                f"--method {self.partition.labels}`, adjudicate the CSV, then "
                f"`talos audit status` before fine-tuning.")
        table = AdjudicationTable(self.space, prior_method=self.partition.labels)
        adjudication = table.read(duck, path)
        summary = table.summarise(duck, adjudication, path)
        if summary.outstanding:
            raise LabellingError(
                f"{path}: {summary.outstanding} of {summary.rows} candidates "
                f"are still unadjudicated. D_s must be FULLY audited before "
                f"{self.run_name} can fine-tune -- finish `talos audit "
                f"status`, no partial runs.")

        # A verdict of `unknown` is a real outcome, not a label: it says a
        # person could not classify the flow, which is not ground truth for
        # anything. Distinct from `state_sql`'s confirmed/corrected split,
        # which does not separate "corrected to X" from "corrected to unknown".
        verdicts = adjudication.filter(
            f"analyst_class IS NOT NULL AND analyst_class <> '' "
            f"AND analyst_class <> '{UNKNOWN}'")
        total_pool = rel.aggregate("count(*)").fetchone()[0]
        joined = duck.relation(
            f"SELECT p.* EXCLUDE (label_class), a.analyst_class AS label_class\n"
            f"      FROM ({rel.sql_query()}) p\n"
            f"      JOIN ({verdicts.sql_query()}) a ON p.uid = a.uid")
        audited_count = joined.aggregate("count(*)").fetchone()[0]
        report.d_s_unaudited = int(total_pool - audited_count)
        return joined

    def _supervised(self, duck, sources, report: BehaviouralReport):
        """D_s as (X, y): audited rows only, then dropped outside the label space."""
        rel = self.partition.relation(duck, SMALL, sources)
        audited = self._audited_relation(duck, rel, report)
        _keys, X, raw = self.vectoriser.labelled_matrix(audited, "label_class")
        keep = np.array([label in self.space for label in raw], dtype=bool)
        report.d_s_rows = int(keep.sum())
        report.d_s_dropped = int((~keep).sum())
        if not report.d_s_rows:
            raise LabellingError(
                f"D_s is empty after dropping classes outside label space "
                f"{self.space.name!r}. Nothing can be fine-tuned on it.")
        y = np.array([self.space.index(label) for label in raw[keep]], dtype=np.int64)
        return X[keep], y

    def class_weights(self, y):
        """Inverse-frequency weight per class, or None when declared off.

        Eq. 3 is plain unweighted cross-entropy and that is now the default.
        An audited D_s is near-balanced by construction -- `CandidateSelector`
        draws per class up to a floor -- so the inverse-frequency weighting that
        once compensated for an unaudited, 80%-benign D_s is not needed when the
        audit gate is honoured. `class_weighting: true` keeps it available as
        the comparison arm.

        ADAPTATION, NOT IN THE PAPER: Eslami & Hamouda's Eq. 3 fine-tuning
        loss is plain unweighted cross-entropy -- confirmed against the paper
        directly, not inferred. Their own fine-tuning data isn't skewed the
        way a NIDS is; ours is, so unweighted cross-entropy on a set that
        is 80%+ one class makes "always predict it" the loss-minimising
        strategy, confidently, which is exactly what was found running this
        against real 2017 data (100% benign, mean confidence 0.84). Kept even
        with an audited (near-floor-balanced) D_s, since a floor is still
        imbalanced relative to true uniformity, just far less extremely. A
        class absent from `y` gets weight 0 rather than a division by zero;
        it cannot appear in any batch either way, so the weight is never used.
        """
        if not self.settings.get("class_weighting", False):
            return None
        torch = require_torch()
        counts = np.bincount(y, minlength=self.space.n_classes).astype(np.float64)
        total = counts.sum()
        safe = np.where(counts > 0, counts, 1.0)          # never actually divided by
        weight = np.where(counts > 0, total / (self.space.n_classes * safe), 0.0)
        return torch.as_tensor(weight, dtype=torch.float32, device=self.device)

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
        """Predict in batches, then write once, vectorised.

        `vectoriser.batches()` streams `target` on one long-lived DuckDB
        cursor. Running any OTHER statement on the same connection while that
        stream is still open silently ends it after the first batch -- found
        by running this against 2017's real 2.1M rows, where it wrote exactly
        one batch (65,536 rows, the configured batch size) and never touched
        the rest, with no error, and confirmed as documented DuckDB behaviour
        (a connection holds one active result at a time). So nothing touches
        `duck` until the stream has fully drained. The write itself registers
        the buffered arrays and loads them with one `CREATE TABLE AS SELECT`
        rather than `executemany`, which DuckDB's own docs say not to use for
        bulk loads -- confirmed ~1000x slower here (numpy/duckdb only, no new
        dependency: pandas/pyarrow are not installed on every box this runs
        on). Register/unregister explicitly, not by local-variable name --
        DuckDB's auto-detection inspects the CALLER's frame, and the caller
        here is `duck.sql`, not this method.
        """
        extras = self.emitted_extras
        uids: list[str] = []
        classes: list[str] = []
        confidences: list[float] = []
        extra_cols: dict[str, list[float]] = {name: [] for name in extras}

        for keys, X in self.vectoriser.batches(target, self.batch):
            proba = np.asarray(self.predict_proba(X), dtype=np.float64)
            chosen = proba.argmax(axis=1)
            confidence = proba.max(axis=1)
            # Runner-up, for Eq. 17. Sorting the row is cheap at k = 5 classes.
            ordered = np.sort(proba, axis=1)
            margin = confidence - (ordered[:, -2] if proba.shape[1] > 1 else 0.0)
            values = {MARGIN: margin, **self.row_extras(X)}
            uids.extend(str(k) for k in keys)
            classes.extend(self.space.classes[c] for c in chosen)
            confidences.extend(confidence.tolist())
            for name in extras:
                extra_cols[name].extend(np.asarray(values[name]).tolist())

        buf = {
            "uid": np.array(uids, dtype=object),
            "label_class": np.array(classes, dtype=object),
            "label_confidence": np.array(confidences, dtype=np.float64),
            **{name: np.array(extra_cols[name], dtype=np.float64) for name in extras},
        }
        duck.con.register("preds_buf", buf)
        duck.sql("CREATE OR REPLACE TEMP TABLE preds AS SELECT * FROM preds_buf")
        duck.con.unregister("preds_buf")

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
        extras = "".join(f",\n             p.\"{name}\"" for name in self.emitted_extras)
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
