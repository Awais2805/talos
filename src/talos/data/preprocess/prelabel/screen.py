"""Label-free column screening: what can be judged before any label exists.

Variance, missingness, correlation clustering and VIF. Nothing here reads a
label, which is why it runs before labelling. Drops columns and logs a reason
for every one; row-level validity is a different verb and lives elsewhere.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from talos.data.preprocess.stats import ColumnStats, ColumnProfiler, quoted

#: A single value covering this much of the column leaves nothing to learn.
DOMINANT_SHARE = 0.99

#: Above this, a column is mostly imputation -- unless it declares an indicator,
#: in which case its absence is a fact about the flow and is itself a feature.
NULL_SHARE = 0.50

#: The vault's threshold. Applied to whichever measure `method` selects.
CORRELATION = 0.90

PEARSON, SPEARMAN, MAX = "pearson", "spearman", "max"
METHODS = (PEARSON, SPEARMAN, MAX)

#: VIF is reported, never dropped on: it measures linear predictability, and
#: layer 1 is a tree model. RFECV resolves redundancy later, model-based.
VIF_REPORT_ONLY = True


#: Never features. Identity is what cross-domain generalisation exists to
#: remove; the rest are keys, provenance, or the label itself.
NOT_FEATURES = (
    "uid", "ts", "capture", "dataset", "domain_id",
    "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p",
    "label_binary", "label_class", "label_source", "label_method",
    "label_confidence", "label_weight", "label_quality", "label_raw",
    "label_margin", "label_executed", "labelled_at", "method_sha",
    "rule_id", "rules_matched", "branches_agreed",
)

_NUMERIC_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
                  "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
                  "FLOAT", "DOUBLE", "DECIMAL", "REAL")


class ScreenError(Exception):
    pass


def sources(globs: Sequence[str]) -> str:
    """One relation over any number of tables, so a screen spans domains.

    Screening per dataset and merging afterwards is not the same thing: a column
    dead in 2017 and alive in 2018 must survive, and only a verdict taken over
    the union can say so. `union_by_name` because a source need not have every
    column -- an absent one reads as NULL and the missingness screen sees it.
    """
    if not globs:
        raise ScreenError("nothing to screen: no source was resolved")
    listed = ", ".join(f"'{g}'" for g in globs)
    return f"read_parquet([{listed}], union_by_name = true)"


def resolve_globs(cfg, lake, datasets: Sequence[str], zone: str, *,
                   feature_space: str, label_method: str = "schedule",
                   source: str | None = None,
                   captures: Sequence[str] | None = None) -> list[str]:
    """Datasets -> parquet globs, one call site shared by every CLI reader.

    The `parquet` zone mirrors Zeek's per-capture folder tree (`Monday`,
    `Friday-WorkingHours`, ...) -- one `conn.parquet` per folder, not one
    directly under the dataset root -- exactly as `LabellingEngine._source`
    already reads it. A literal `rel="conn.parquet"` glob (what this used to
    be) only ever matched a single-file fixture; against a real multi-capture
    dataset it silently resolves to a path that never exists. `labelled` is
    already one consolidated file per dataset (the schedule writer joins every
    capture before writing), so it stays a literal name.

    `lake.exists()` cannot check a glob containing `**` -- it is not a literal
    path -- so existence is checked against the un-globbed dataset directory
    (`lake.list`, which recurses) before the glob is built.
    """
    if source:
        return [source]
    globs = []
    for dataset in datasets:
        scope = {"feature_space": feature_space, "source": cfg.source_of(dataset)}
        if zone == "labelled":
            scope["method"] = label_method
        base = lake.uri(zone, dataset=dataset, **scope)
        if not lake.list(base):
            raise ScreenError(f"nothing to read: {base} has no files.")
        if zone != "parquet":
            globs.append(f"{base}/conn.parquet")
        elif captures:
            globs.extend(f"{base}/{c}/conn.parquet" for c in captures)
        else:
            globs.append(f"{base}/**/conn.parquet")
    return globs


def preflight(features, columns) -> None:
    """Refuse a table missing a column the feature set reads.

    Without this the failure is unreadable: each feature is aliased to its own
    name, so an absent source column makes DuckDB resolve the reference to the
    alias being defined and report a self-reference rather than a missing one.
    """
    present = {c.lower() for c in dict(columns)}
    missing = sorted(c for c in features.source_columns if c.lower() not in present)
    if missing:
        raise ScreenError(
            f"feature set {features.name!r} reads {', '.join(missing)}, which "
            f"the source does not have. Screening would fail inside DuckDB with "
            f"a self-reference error naming the wrong problem.")


def feature_source(features, source: str,
                   carry: Sequence[str] = ()) -> tuple[str, list, list]:
    """Project a declared feature set's RAW values, one column per feature.

    Screening the table's own columns answers a different question: `orig_rate`
    and `hist_orig_rst` are not columns, and they are what a model sees. Raw,
    not transformed -- the shape evidence for D1 needs the untransformed
    distribution, and `stats` computes the log1p side itself.

    `carry` names columns to pass through unscored -- the label and the domain,
    which relevance needs beside each row. Carried here rather than joined back:
    a positional join would depend on two scans returning rows in the same
    order, which nothing guarantees.
    """
    numeric = [f.name for f in features.numeric]
    categorical = [c.name for c in features.categorical]
    projected = [f"({f.expr}) AS {quoted(f.name)}" for f in features.numeric]
    projected += [f"CAST(({c.expr}) AS VARCHAR) AS {quoted(c.name)}"
                  for c in features.categorical]
    projected += [quoted(c) for c in carry if c not in numeric + categorical]
    return f"(SELECT {', '.join(projected)} FROM {source})", numeric, categorical


def exclude_invalid(duck, source: str) -> tuple[str, int, int]:
    """Restrict a source to rows no validity invariant rejects.

    Returns `(source, excluded, total)`. Scale is measured on the population a
    model actually trains on: a near-zero duration dividing a real byte count
    gives ~1e18 B/s, a finite double that would otherwise set `max/p99` for its
    whole column.
    """
    from talos.data.preprocess.prelabel.verify import valid_sql

    total = duck.one(f"SELECT count(*) FROM {source}")[0]
    predicate = valid_sql(describe(duck, source))
    if predicate == "TRUE":
        return source, 0, int(total)
    filtered = f"(SELECT * FROM {source} WHERE {predicate})"
    kept = duck.one(f"SELECT count(*) FROM {filtered}")[0]
    return filtered, int(total) - int(kept), int(total)


def declared_indicators(features) -> set[str]:
    """Features whose absence is already carried by a missingness indicator.

    Read off the declaration rather than re-typed on the command line: the
    missingness rule would otherwise delete the very features whose NULLs are
    the signal -- `duration`, `orig_bytes` and `resp_bytes` in `conn`.
    """
    return {f.name for f in getattr(features, "numeric", ()) if f.indicator}


def describe(duck, source: str) -> dict:
    """Column name -> type for any source EXPRESSION, not just a bare glob.

    `DuckEngine.columns` wraps its argument in `read_parquet` itself, so it
    cannot describe a multi-table source. This can.
    """
    return {row[0]: row[1] for row in duck.sql(f"DESCRIBE SELECT * FROM {source}")}


def candidates(columns, exclude: Sequence[str] = NOT_FEATURES):
    """Split a table's own columns into numeric and categorical candidates.

    Takes the name -> type mapping rather than a source, so the caller decides
    how to read it: `DuckEngine.columns` wraps a bare glob in `read_parquet`
    itself, and handing it an already-wrapped expression would wrap it twice.
    """
    skip = {c.lower() for c in exclude}
    numeric: list[str] = []
    categorical: list[str] = []
    for name, kind in dict(columns).items():
        if name.lower() in skip:
            continue
        upper = str(kind).upper()
        if any(t in upper for t in _NUMERIC_TYPES):
            numeric.append(name)
        elif "VARCHAR" in upper or "BOOL" in upper:
            categorical.append(name)
    return numeric, categorical


@dataclass(frozen=True)
class Rejection:
    """A dropped column and the evidence that dropped it."""

    column: str
    check: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"column": self.column, "check": self.check,
                "reason": self.reason, "evidence": self.evidence}


@dataclass(frozen=True)
class Cluster:
    """Columns that move together, and the one kept to represent them."""

    representative: str
    members: tuple[str, ...]
    method: str
    strongest: float

    def to_dict(self) -> dict:
        return {"representative": self.representative,
                "members": list(self.members), "method": self.method,
                "strongest": self.strongest}


@dataclass
class ScreenReport:
    """What survived, what did not, and why."""

    dataset: str
    datasets: tuple[str, ...] = ()
    source: str = ""
    rows: int = 0
    method: str = MAX
    thresholds: dict[str, float] = field(default_factory=dict)
    stats: tuple[ColumnStats, ...] = ()
    kept: tuple[str, ...] = ()
    rejections: tuple[Rejection, ...] = ()
    clusters: tuple[Cluster, ...] = ()
    correlations: dict[str, dict[str, float]] = field(default_factory=dict)
    vif: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset, "datasets": list(self.datasets),
            "source": self.source, "rows": self.rows,
            "correlation_method": self.method, "thresholds": self.thresholds,
            "scope": ("one verdict over the union of every dataset listed: a "
                      "column dead in one domain and alive in another must "
                      "survive, and only a union can say so"),
            "kept": list(self.kept),
            "rejected": [r.to_dict() for r in self.rejections],
            "clusters": [c.to_dict() for c in self.clusters],
            "correlations": self.correlations,
            "vif": self.vif,
            "vif_policy": ("reported, never dropped on: VIF measures linear "
                           "predictability and layer 1 is a tree model"),
            "columns": [s.to_dict() for s in self.stats],
            "warnings": list(self.warnings),
            **self.extra,
        }

    def table(self) -> str:
        lines = [f"{'column':<22}{'kind':<13}{'null%':>8}{'domin%':>8}"
                 f"{'distinct':>10}  verdict"]
        dropped = {r.column: r.check for r in self.rejections}
        for stat in self.stats:
            verdict = f"DROP ({dropped[stat.name]})" if stat.name in dropped \
                else "keep"
            lines.append(
                f"{stat.name:<22}{stat.kind:<13}{100 * stat.null_share:>7.2f}%"
                f"{100 * stat.dominant_share:>7.2f}%{stat.distinct:>10,}  {verdict}")
        return "\n".join(lines)

    def shape_table(self) -> str:
        """The D1 evidence: which columns a mean and a sigma would be set by."""
        lines = [f"{'column':<22}{'skew':>10}{'p99/p50':>12}{'max/p99':>12}"
                 f"{'skew(log1p)':>14}{'max/p99(log1p)':>16}"]
        for stat in self.stats:
            if not (stat.raw and stat.logged):
                continue
            lines.append(
                f"{stat.name:<22}{_n(stat.raw.skewness):>10}"
                f"{_n(stat.raw.tail_ratio):>12}{_n(stat.raw.extreme_ratio):>12}"
                f"{_n(stat.logged.skewness):>14}"
                f"{_n(stat.logged.extreme_ratio):>16}")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [f"rows          {self.rows:,}",
                 f"kept          {len(self.kept)} of {len(self.stats)} columns",
                 f"correlation   {self.method} at {self.thresholds.get('correlation')}"]
        for cluster in self.clusters:
            others = ", ".join(m for m in cluster.members
                               if m != cluster.representative)
            lines.append(f"  cluster     {cluster.representative} represents "
                         f"{others}  ({cluster.method} {cluster.strongest:.4f})")
        if self.vif:
            worst = sorted(self.vif.items(), key=lambda kv: -kv[1])[:5]
            lines.append("  highest VIF " + ", ".join(
                f"{c} {v:.1f}" for c, v in worst) + "  (reported, not dropped)")
        for warning in self.warnings:
            lines.append(f"!! {warning}")
        return "\n".join(lines)


class FeatureScreener:
    """The label-free half of the vault's pruning block."""

    def __init__(self, duck, dominant: float = DOMINANT_SHARE,
                 null_share: float = NULL_SHARE,
                 correlation: float = CORRELATION, method: str = MAX,
                 indicated: Sequence[str] = ()):
        if method not in METHODS:
            raise ScreenError(
                f"correlation method {method!r} is not one of {', '.join(METHODS)}")
        self.duck = duck
        self.dominant = float(dominant)
        self.null_share = float(null_share)
        self.correlation = float(correlation)
        self.method = method
        # Columns whose absence is declared to be information, not a defect.
        self.indicated = frozenset(indicated)

    def screen(self, dataset: str, source: str, numeric: Sequence[str],
               categorical: Sequence[str]) -> ScreenReport:
        profiler = ColumnProfiler(self.duck, source)
        stats = profiler.profile(numeric, categorical)
        report = ScreenReport(
            dataset=dataset, source=source, rows=profiler.rows,
            method=self.method, stats=stats,
            thresholds={"dominant_share": self.dominant,
                        "null_share": self.null_share,
                        "correlation": self.correlation})

        # Missingness first: an all-NULL column trips both checks, and "100%
        # NULL" describes it better than "one value covers 100% of rows". A
        # column is rejected once, by the first check that catches it.
        rejections: list[Rejection] = []
        rejected: set[str] = set()
        for check in (self.drop_high_missingness, self.drop_zero_variance):
            for rejection in check(stats):
                if rejection.column in rejected:
                    continue
                rejections.append(rejection)
                rejected.add(rejection.column)

        surviving = [s.name for s in stats
                     if s.kind == "numeric" and s.name not in rejected]
        report.correlations = self.correlations(source, surviving)
        clusters = self.correlation_cluster(surviving, report.correlations)
        for cluster in clusters:
            for member in cluster.members:
                if member == cluster.representative:
                    continue
                rejections.append(Rejection(
                    column=member, check="correlated",
                    reason=(f"moves with {cluster.representative} at "
                            f"{cluster.strongest:.4f} ({cluster.method}); one "
                            f"representative is kept per cluster"),
                    evidence={"representative": cluster.representative,
                              "members": list(cluster.members),
                              "strongest": cluster.strongest}))
        rejected |= {r.column for r in rejections}

        report.vif, vif_warning = self.vif_pass(
            [c for c in surviving if c not in rejected], report.correlations)
        if vif_warning:
            report.warnings += (vif_warning,)

        report.clusters = tuple(clusters)
        report.rejections = tuple(rejections)
        report.kept = tuple(s.name for s in stats if s.name not in rejected)
        return report

    # -------------------------------------------------------------- the drops

    def drop_zero_variance(self, stats: Sequence[ColumnStats]):
        """A column one value nearly always takes cannot separate anything."""
        for stat in stats:
            if stat.dominant_share >= self.dominant:
                yield Rejection(
                    column=stat.name, check="zero-variance",
                    reason=(f"{100 * stat.dominant_share:.2f}% of rows are "
                            f"{stat.dominant_value!r}, at or above the "
                            f"{100 * self.dominant:.0f}% limit"),
                    evidence={"dominant_value": stat.dominant_value,
                              "dominant_share": stat.dominant_share,
                              "distinct": stat.distinct})

    def drop_high_missingness(self, stats: Sequence[ColumnStats]):
        """Mostly-absent columns go, unless their absence is declared to mean something.

        Zeek's NULLs are semantic: `duration IS NULL` is "no observed teardown".
        A column carrying a missingness indicator keeps that fact, so it is
        exempt -- the Engelen attempted-attack criterion depends on exactly this.
        """
        for stat in stats:
            if stat.null_share <= self.null_share:
                continue
            if stat.name in self.indicated:
                continue
            yield Rejection(
                column=stat.name, check="missingness",
                reason=(f"{100 * stat.null_share:.2f}% NULL, above the "
                        f"{100 * self.null_share:.0f}% limit, and no missingness "
                        f"indicator is declared for it"),
                evidence={"nulls": stat.nulls, "null_share": stat.null_share})

    # ------------------------------------------------------- the correlations

    def correlations(self, source: str, columns: Sequence[str]) -> dict:
        """Pearson and Spearman for every pair, from one scan each.

        Spearman is Pearson over ranks. NULLs stay NULL so `corr` drops those
        pairs; the constant rank offset that leaves behind does not move a
        correlation.
        """
        if len(columns) < 2:
            return {}
        pairs = list(itertools.combinations(columns, 2))
        out: dict[str, dict[str, float]] = {}

        selects = ", ".join(
            f"corr(CAST({quoted(a)} AS DOUBLE), CAST({quoted(b)} AS DOUBLE))"
            for a, b in pairs)
        pearson = self.duck.one(f"SELECT {selects} FROM {source}")

        ranks = ", ".join(
            f"CASE WHEN {quoted(c)} IS NULL THEN NULL ELSE "
            f"rank() OVER (ORDER BY {quoted(c)}) END AS {quoted(c)}"
            for c in columns)
        self.duck.sql(f"CREATE OR REPLACE TEMP TABLE ranked AS "
                      f"SELECT {ranks} FROM {source}")
        selects = ", ".join(
            f"corr(CAST({quoted(a)} AS DOUBLE), CAST({quoted(b)} AS DOUBLE))"
            for a, b in pairs)
        spearman = self.duck.one(f"SELECT {selects} FROM ranked")

        for i, (a, b) in enumerate(pairs):
            values = {PEARSON: _corr(pearson[i]), SPEARMAN: _corr(spearman[i])}
            values[MAX] = max(abs(values[PEARSON]), abs(values[SPEARMAN]))
            out[f"{a}|{b}"] = values
        return out

    def correlation_cluster(self, columns: Sequence[str],
                            correlations: dict) -> list[Cluster]:
        """Connected components over the pairs above the threshold."""
        parent = {c: c for c in columns}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        strongest: dict[str, float] = {}
        for key, values in correlations.items():
            a, b = key.split("|")
            value = abs(values[self.method])
            if value <= self.correlation:
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
            root = find(a)
            strongest[root] = max(strongest.get(root, 0.0), value)

        groups: dict[str, list[str]] = {}
        for column in columns:
            groups.setdefault(find(column), []).append(column)
        return [Cluster(representative=self._representative(members),
                        members=tuple(members), method=self.method,
                        strongest=strongest.get(root, 0.0))
                for root, members in groups.items() if len(members) > 1]

    def _representative(self, members: Sequence[str]) -> str:
        """Deterministic: the declared order decides, so a re-run cannot differ."""
        return members[0]

    # --------------------------------------------------------------- the VIF

    def vif_pass(self, columns: Sequence[str],
                 correlations: dict) -> tuple[dict[str, float], str]:
        """Diagonal of the inverted correlation matrix. Reported, never dropped on."""
        if len(columns) < 2:
            return {}, ""
        size = len(columns)
        index = {c: i for i, c in enumerate(columns)}
        matrix = np.eye(size)
        for key, values in correlations.items():
            a, b = key.split("|")
            if a not in index or b not in index:
                continue
            matrix[index[a], index[b]] = matrix[index[b], index[a]] = \
                values[PEARSON]

        warning = ""
        try:
            inverted = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            # Exactly collinear columns make the matrix singular; the
            # pseudo-inverse still ranks them, and the reason is worth saying.
            inverted = np.linalg.pinv(matrix)
            warning = ("the correlation matrix is singular, so VIF comes from a "
                       "pseudo-inverse: at least two columns are exactly collinear")
        return {c: float(inverted[i, i]) for c, i in index.items()}, warning


#: Eq. 2's phi relative to the reconstruction term, as Table VII selects it.
PHI_RATIO = 0.5


def suggested_phi(report: ScreenReport, base, ratio: float = PHI_RATIO) -> dict:
    """Eq. 2's phi_m for each declared constraint, at this data's scale.

    A residual `a - b/c` carries the units of its TARGET, so dividing by that
    target's typical magnitude makes it dimensionless -- after which the paper's
    ratio transfers directly. Table VII's 0.5 was selected against standardised,
    unlogged features whose residuals were already near unit scale; ours are in
    original units because the transform has to be undone for a ratio identity
    to hold at all, so the same relative emphasis needs a different number.
    """
    medians = {s.name: (s.raw.quantile(0.50) if s.raw else None)
               for s in report.stats}
    out: dict[str, dict] = {}
    for constraint in getattr(base, "constraints", ()):
        median = medians.get(constraint.target)
        if median is None or abs(median) < EPSILON_PHI:
            out[constraint.name] = {
                "phi": None, "target": constraint.target, "target_p50": median,
                "why": ("target median is zero or unmeasured, so the residual "
                        "cannot be made dimensionless from it")}
            continue
        out[constraint.name] = {
            "phi": ratio / abs(median), "target": constraint.target,
            "target_p50": median,
            "why": f"{ratio} / p50({constraint.target})"}
    return out


#: A median this close to zero cannot normalise anything.
EPSILON_PHI = 1e-9


#: Fewer digits than a double carries, so a re-run does not move `method_sha`
#: on floating-point noise. Six is far finer than any scale this measures.
CONSTANT_DIGITS = 6


def standardisation(report: ScreenReport, base) -> dict[str, dict]:
    """Each numeric feature's `centre`/`scale`, in the space it is standardised in.

    The paper says only that "numeric fields are standardized/normalized"
    (Sec. IV-A) and names no statistic, so z-score is chosen as the plain
    reading of "standardized" -- recorded as ADAPTED in docs/paper-deviations.md.
    Constants live in TRANSFORMED space because `Numeric.sql` applies them after
    the transform, so a `log1p` feature is measured on `log1p(x)`.
    """
    from talos.data.feature.featureset import LOG1P

    shapes = {s.name: s for s in report.stats}
    out: dict[str, dict] = {}
    for feature in getattr(base, "numeric", ()):
        stat = shapes.get(feature.name)
        if stat is None:
            continue
        logged = feature.transform == LOG1P
        shape = stat.logged if logged else stat.raw
        space = "log1p" if logged else "original"
        if shape is None or shape.mean is None or not shape.stddev:
            out[feature.name] = {
                "centre": None, "scale": None, "space": space,
                "why": ("standard deviation is zero or unmeasured, so the "
                        "column is constant and has no scale")}
            continue
        out[feature.name] = {
            "centre": round(shape.mean, CONSTANT_DIGITS),
            "scale": round(shape.stddev, CONSTANT_DIGITS),
            "space": space, "why": f"mean and standard deviation of {space} values"}
    return out


def annotate_declaration(base, name: str, stage: str,
                         verdicts: Mapping[str, Mapping[str, Any]],
                         constants: Mapping[str, Mapping[str, Any]] = {}) -> str:
    """The base declaration, ANNOTATED with one stage's verdict -- nothing removed.

    Every feature the base declares stays in the file. A rejected one gets a
    `{stage}: {verdict: drop, ...}` block instead of being deleted, so a later
    stage -- or the eventual training pipeline -- still has the column to apply
    its own policy to. A feature absent from `verdicts` defaults to `keep`
    (e.g. VIF, which is reported and never verdicted). `FeatureSet.active()` is
    what a consumer actually filters through; this function only records.
    """
    import yaml

    doc = {k: v for k, v in dict(base.doc).items()}
    doc["name"] = name
    for block in ("numeric", "derived", "categorical"):
        declared = doc.get(block) or {}
        if not declared:
            continue
        annotated = {}
        for key, entry in declared.items():
            entry = dict(entry or {})
            fitted = constants.get(key) if block != "categorical" else None
            if fitted and fitted.get("centre") is not None:
                entry["centre"] = fitted["centre"]
                entry["scale"] = fitted["scale"]
            entry[stage] = dict(verdicts.get(key) or {"verdict": "keep"})
            annotated[key] = entry
        doc[block] = annotated
    # Constraints are never pruned here -- nothing is removed from the
    # declaration, so no operand can go missing. `FeatureSet.active()` excludes
    # one from its own filtered view if a marked-drop feature would orphan it.

    header = [
        f"# Talos -- feature set `{name}`. GENERATED by `talos {stage}`; edit",
        "# the base declaration and re-run rather than editing this by hand.",
        "#",
        f"# base        {getattr(base, 'name', '?')}  sha {getattr(base, 'sha', '?')}",
        "#",
        "# ANNOTATED, NOT PRUNED: every feature the base declares is still here.",
        f"# a `{stage}:` block below records this stage's verdict per feature;",
        "# nothing is removed at this layer. A consumer -- a labelling method's",
        "# vectoriser today, the training pipeline later -- calls",
        "# `FeatureSet.active()` to apply its own policy over every stage's marks.",
        "#",
    ]
    marked = {k: v for k, v in verdicts.items() if v.get("verdict", "keep") != "keep"}
    for column, mark in sorted(marked.items()):
        reason = str(mark.get("reason", "")).splitlines()[0][:78] if mark.get("reason") else ""
        header.append(f"#   {column:<22} {stage}={mark.get('verdict'):<10} {reason}")
    header.append("")
    return "\n".join(header) + yaml.safe_dump(doc, sort_keys=False,
                                              default_flow_style=False)


def emit_declaration(report: ScreenReport, base, name: str) -> str:
    """The screen's verdict, layered onto the base declaration.

    A normal feature set once loaded, so it hashes and plugs in like any
    other and a method just names it. See `annotate_declaration` for the
    mark-not-drop contract this and `talos relevance`'s emit share.
    """
    verdicts = {r.column: {"verdict": "drop", "check": r.check, "reason": r.reason}
                for r in report.rejections}
    return annotate_declaration(base, name, "screen", verdicts,
                                standardisation(report, base))


def _corr(value) -> float:
    """`corr` is NULL when a pair never co-occurs; that is zero evidence, not 1.0."""
    return 0.0 if value is None else float(value)


def _n(value) -> str:
    """A constant column has undefined skew; NaN is not a measurement."""
    if value is None or not math.isfinite(value):
        return "-"
    # These ratios legitimately reach 1e7; fixed notation would collide.
    return f"{value:.3e}" if abs(value) >= 1e6 else f"{value:,.3f}"
