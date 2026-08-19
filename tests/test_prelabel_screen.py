"""Gates 0.2/0.5 — the label-free screen, over a fixture with known answers.

Every column here is built so its verdict is arithmetic, not judgement: a
constant, an all-NULL, an exact duplicate, a monotone-but-not-linear partner,
and a heavy tail whose max sits far beyond its own p99. No label is read, which
is the property that lets this run before labelling at all.
"""

from __future__ import annotations

import json
import math

import duckdb
import pytest

from talos.data.preprocess.prelabel.screen import (
    MAX, PEARSON, SPEARMAN, FeatureScreener, ScreenError, candidates, describe,
    sources,
)
from talos.data.preprocess.stats import ColumnProfiler

ROWS = 1000


class Duck:
    """`DuckEngine`'s surface, as the screener uses it.

    `execute`, not `sql`: a CREATE TABLE returns no relation, so `con.sql(...)`
    gives None and `.fetchall()` on it raises. The real engine uses `execute`
    for exactly this reason.
    """

    def __init__(self, con):
        self.con = con

    def sql(self, query):
        return self.con.execute(query).fetchall()

    def one(self, query):
        return self.con.execute(query).fetchone()

    def columns(self, source):
        return {row[0]: row[1] for row in
                self.con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}


@pytest.fixture
def duck():
    con = duckdb.connect()
    con.sql(f"""
        CREATE TABLE flows AS
        SELECT
            'u' || i                                   AS uid,
            i                                          AS base,
            i * 2                                      AS doubled,
            CAST(i AS DOUBLE) ^ 3                      AS cubed,
            CASE WHEN i % 7 = 0 THEN NULL ELSE i END   AS holey,
            CASE WHEN i < {ROWS - 1} THEN 1 ELSE 2 END AS nearly_constant,
            CAST(NULL AS BIGINT)                       AS never_set,
            CASE WHEN i = {ROWS - 1} THEN 1000000.0
                 ELSE CAST(i % 10 AS DOUBLE) END       AS spiky,
            CASE WHEN i % 3 = 0 THEN 'tcp' ELSE 'udp' END AS proto,
            CASE WHEN i % 2 = 0 THEN NULL ELSE 'x' END AS mostly_null,
            'benign'                                   AS label_class
        FROM range(0, {ROWS}) t(i)""")
    return Duck(con)


def screen(duck, **kwargs):
    numeric, categorical = candidates(duck.columns("flows"))
    return FeatureScreener(duck, **kwargs).screen(
        "toy", "flows", numeric, categorical)


# ------------------------------------------------------------- what counts

def test_identity_and_label_columns_are_not_candidates(duck):
    """Screening a label would leak it; screening a uid would score noise."""
    numeric, categorical = candidates(duck.columns("flows"))
    assert "uid" not in numeric + categorical
    assert "label_class" not in numeric + categorical
    assert "base" in numeric and "proto" in categorical


def test_candidates_are_split_by_the_table_s_own_types(duck):
    """Read from the table, so a column added upstream appears as a candidate."""
    numeric, categorical = candidates(duck.columns("flows"))
    assert set(categorical) == {"proto", "mostly_null"}
    assert "cubed" in numeric and "never_set" in numeric


# ----------------------------------------------------------------- the drops

def test_a_constant_column_is_dropped(duck):
    """`nearly_constant` is 999/1000 one value -- above the 99% limit."""
    report = screen(duck)
    dropped = {r.column: r.check for r in report.rejections}
    assert dropped["nearly_constant"] == "zero-variance"
    assert "nearly_constant" not in report.kept


def test_an_all_null_column_is_dropped_once_as_missingness(duck):
    """It trips both checks; the better reason wins and it is logged once.

    An all-NULL column is as dead as a constant, but "100% NULL" describes it
    more usefully than "one value covers 100% of rows", and a column counted
    under two checks would make every rejection total wrong.
    """
    report = screen(duck)
    entries = [r for r in report.rejections if r.column == "never_set"]
    assert [r.check for r in entries] == ["missingness"]


def test_no_column_is_rejected_twice(duck):
    """Two entries for one column double-counts it in every summary."""
    report = screen(duck)
    columns = [r.column for r in report.rejections]
    assert len(columns) == len(set(columns))


def test_a_mostly_null_column_is_dropped_on_missingness(duck):
    """`mostly_null` is 50% NULL; the default limit is strictly greater than."""
    report = screen(duck, null_share=0.40)
    dropped = {r.column: r.check for r in report.rejections}
    assert dropped["mostly_null"] == "missingness"


def test_a_declared_indicator_exempts_a_column_from_missingness(duck):
    """Zeek's NULLs are semantic; an indicator keeps that fact, so absence stays.

    This is what the Engelen attempted-attack criterion depends on: NULL and 0
    must not become the same thing.
    """
    report = screen(duck, null_share=0.40, indicated=["mostly_null"])
    dropped = {r.column for r in report.rejections}
    assert "mostly_null" not in dropped
    assert "mostly_null" in report.kept


def test_every_drop_carries_a_reason_and_evidence(duck):
    """A rejection with no evidence cannot be argued with, only believed."""
    report = screen(duck)
    assert report.rejections
    for rejection in report.rejections:
        assert rejection.reason and rejection.evidence


# ---------------------------------------------------------- the correlations

def test_an_exact_linear_duplicate_is_clustered(duck):
    """`doubled` is 2*base: Pearson and Spearman are both exactly 1."""
    report = screen(duck)
    pair = report.correlations["base|doubled"]
    assert pair[PEARSON] == pytest.approx(1.0)
    assert pair[SPEARMAN] == pytest.approx(1.0)
    kept = set(report.kept)
    assert not {"base", "doubled"} <= kept


def test_spearman_catches_a_monotone_pair_pearson_understates(duck):
    """`cubed` is monotone in `base` but not linear -- the rank measure is 1.0."""
    report = screen(duck)
    pair = report.correlations["base|cubed"]
    assert pair[SPEARMAN] == pytest.approx(1.0)
    assert pair[SPEARMAN] > pair[PEARSON]


def test_the_clustering_method_is_selectable(duck):
    """All three are computed from the same scan; only the decision differs."""
    for method in (PEARSON, SPEARMAN, MAX):
        report = screen(duck, method=method)
        assert report.method == method
        assert report.correlations["base|doubled"][method] == pytest.approx(1.0)


def test_an_unknown_correlation_method_is_refused(duck):
    """Silently defaulting would report a clustering nobody asked for."""
    with pytest.raises(ScreenError, match="not one of"):
        screen(duck, method="kendall")


def test_one_representative_survives_each_cluster(duck):
    """Keeping none loses the signal; keeping all is why clustering exists."""
    report = screen(duck)
    for cluster in report.clusters:
        survivors = [m for m in cluster.members if m in report.kept]
        assert survivors == [cluster.representative]


# ------------------------------------------------------------------ the VIF

def test_vif_is_reported_and_drops_nothing(duck):
    """VIF measures linear predictability; layer 1 is a tree model."""
    report = screen(duck)
    assert report.vif
    assert not [r for r in report.rejections if r.check == "vif"]
    assert "never dropped" in report.to_dict()["vif_policy"]


# ---------------------------------------------------------- the D1 evidence

def test_shape_is_measured_before_and_after_log1p(duck):
    """D1 is chosen from these numbers, so both sides must be on record."""
    report = screen(duck)
    spiky = next(s for s in report.stats if s.name == "spiky")
    assert spiky.raw and spiky.logged


def test_a_single_outlier_shows_up_as_an_extreme_ratio(duck):
    """`spiky` is 0-9 except one row at 1e6: max/p99 is the D1 argument."""
    report = screen(duck)
    spiky = next(s for s in report.stats if s.name == "spiky")
    assert spiky.raw.extreme_ratio > 1000
    # log1p compresses it by orders of magnitude but does not remove it.
    assert spiky.logged.extreme_ratio < spiky.raw.extreme_ratio


def test_log1p_reduces_skew_on_a_heavy_tail(duck):
    """If it did not, there would be no reason to apply it before scaling."""
    report = screen(duck)
    spiky = next(s for s in report.stats if s.name == "spiky")
    assert abs(spiky.logged.skewness) < abs(spiky.raw.skewness)


def test_null_share_and_dominant_share_are_row_fractions(duck):
    """`holey` is every 7th row NULL; the arithmetic must be exact."""
    report = screen(duck)
    holey = next(s for s in report.stats if s.name == "holey")
    expected = math.ceil(ROWS / 7) / ROWS
    assert holey.null_share == pytest.approx(expected)


def test_profiler_counts_rows_once(duck):
    """Row count is used as a denominator everywhere; one source for it."""
    profiler = ColumnProfiler(duck, "flows")
    assert profiler.rows == ROWS == profiler.rows


# --------------------------------------------------------------- the report

def test_the_report_lists_kept_and_rejected_without_overlap(duck):
    """A column both kept and rejected would make the schema ambiguous."""
    report = screen(duck)
    rejected = {r.column for r in report.rejections}
    assert not rejected & set(report.kept)
    assert rejected | set(report.kept) == {s.name for s in report.stats}


def test_no_label_column_is_read_anywhere(duck):
    """The whole point: this must be runnable before any label exists."""
    report = screen(duck)
    assert "label_class" not in {s.name for s in report.stats}


# ------------------------------------------------------------ cross-domain

@pytest.fixture
def two_domains(tmp_path):
    """Two tables where `only_2018` is constant in one domain and varies in the other."""
    con = duckdb.connect()
    paths = []
    for name, expr in (("d2017", "1"), ("d2018", "i % 97")):
        path = tmp_path / f"{name}.parquet"
        con.execute(f"""COPY (
            SELECT 'u' || i AS uid, i AS base, CAST({expr} AS BIGINT) AS only_2018
            FROM range(0, 1000) t(i)) TO '{path}' (FORMAT PARQUET)""")
        paths.append(str(path))
    return Duck(con), paths


def test_a_column_dead_in_one_domain_survives_if_alive_in_another(two_domains):
    """The reason the screen takes one verdict over the union.

    `only_2018` is constant in 2017 and would be dropped there. Screening each
    domain apart and merging afterwards would delete a column the other domain
    needs, and schema v1 is locked once for all of them.
    """
    duck, paths = two_domains
    alone = sources(paths[:1])
    numeric, categorical = candidates(describe(duck, alone))
    solo = FeatureScreener(duck).screen("d2017", alone, numeric, categorical)
    assert "only_2018" in {r.column for r in solo.rejections}

    both = sources(paths)
    numeric, categorical = candidates(describe(duck, both))
    union = FeatureScreener(duck).screen("d2017+d2018", both, numeric, categorical)
    assert "only_2018" in union.kept
    assert union.rows == 2000


def test_an_empty_source_list_is_refused(two_domains):
    """An empty union would screen nothing and report every column as clean."""
    with pytest.raises(ScreenError, match="no source was resolved"):
        sources([])


def test_describe_handles_a_multi_table_source(two_domains):
    """`DuckEngine.columns` wraps a bare glob and cannot describe a union."""
    duck, paths = two_domains
    assert set(describe(duck, sources(paths))) == {"uid", "base", "only_2018"}


# ---------------------------------------------------- the emitted schema

@pytest.fixture
def conn_like(tmp_path):
    """A table carrying every column conn reads, with planted redundancy."""
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE flows AS SELECT
            CAST(i % 5000 AS BIGINT)          AS orig_bytes,
            CAST(i % 5000 AS BIGINT) + 40     AS orig_ip_bytes,
            CAST((i * 7) % 9000 AS BIGINT)    AS resp_bytes,
            CAST((i * 7) % 9000 AS BIGINT) + 40 AS resp_ip_bytes,
            CAST(i % 40 AS BIGINT)            AS orig_pkts,
            CAST((i * 3) % 55 AS BIGINT)      AS resp_pkts,
            CAST(0 AS BIGINT)                 AS missed_bytes,
            CAST(i % 900 AS DOUBLE) / 3.0     AS duration,
            CASE WHEN i % 3 = 0 THEN 'tcp' ELSE 'udp' END AS proto,
            CASE WHEN i % 5 = 0 THEN 'S0' ELSE 'SF' END   AS conn_state,
            CAST(NULL AS BOOLEAN)             AS local_orig,
            CAST(NULL AS BOOLEAN)             AS local_resp,
            CASE WHEN i % 5 = 0 THEN 'Sr' ELSE 'ShADadFf' END AS history
        FROM range(0, 2000) t(i)""")
    return Duck(con)


def screened(conn_like):
    from talos.data.feature.featureset import FeatureSetLoader
    from talos.data.preprocess.prelabel.screen import feature_source, preflight
    base = FeatureSetLoader().load("conn")
    preflight(base, conn_like.columns("flows"))
    source, numeric, categorical = feature_source(base, "flows")
    report = FeatureScreener(conn_like).screen("toy", source, numeric, categorical)
    return base, report


def test_a_feature_set_is_screened_by_its_features_not_the_table(conn_like):
    """`orig_rate` and the history counts are expressions, not columns."""
    _base, report = screened(conn_like)
    names = {s.name for s in report.stats}
    assert "orig_rate" in names and "hist_orig_rst" in names
    assert "history" not in names       # a source column, not a feature


def test_a_rejected_column_is_marked_not_removed(conn_like, tmp_path):
    """The file never shrinks -- `.active()` is what a consumer filters through."""
    from talos.data.feature.featureset import FeatureSetLoader
    from talos.data.preprocess.prelabel.screen import emit_declaration

    base, report = screened(conn_like)
    out = tmp_path / "conn-screened.yaml"
    out.write_text(emit_declaration(report, base, "conn-screened"))

    loaded = FeatureSetLoader().load(out)
    everyone = {f.name for f in loaded.numeric} | {c.name for c in loaded.categorical}
    rejected = {r.column for r in report.rejections}
    assert rejected <= everyone, "a rejected column must still be in the file"

    marked = {f.name for f in loaded.numeric if f.verdict == "drop"}
    marked |= {c.name for c in loaded.categorical if c.verdict == "drop"}
    assert marked == rejected

    survivors = {f.name for f in loaded.active().numeric} \
        | {c.name for c in loaded.active().categorical}
    assert survivors == set(report.kept)


def test_the_emitted_schema_loads(conn_like, tmp_path):
    """A generated declaration that cannot load is worse than none."""
    from talos.data.feature.featureset import FeatureSetLoader
    from talos.data.preprocess.prelabel.screen import emit_declaration

    base, report = screened(conn_like)
    out = tmp_path / "s.yaml"
    out.write_text(emit_declaration(report, base, "s"))
    assert FeatureSetLoader().load(out).name == "s"


def test_a_constraint_orphaned_by_a_drop_stays_in_the_file_not_active(conn_like, tmp_path):
    """`missed_bytes` and `overhead_orig` are constant in this fixture, so
    `orig_ip_bytes_decomposition` loses an operand under `.active()` -- but
    nothing is pruned from the file itself; only the runtime view excludes it.
    """
    from talos.data.feature.featureset import FeatureSetLoader
    from talos.data.preprocess.prelabel.screen import emit_declaration
    import yaml

    base, report = screened(conn_like)
    dropped = {r.column for r in report.rejections}
    assert "overhead_orig" in dropped

    text = emit_declaration(report, base, "s")
    doc = yaml.safe_load(text)
    names = {c["name"] for c in doc.get("constraints") or []}
    assert "orig_ip_bytes_decomposition" in names, "constraints are never pruned from the file"

    out = tmp_path / "s.yaml"
    out.write_text(text)
    active_names = {c.name for c in FeatureSetLoader().load(out).active().constraints}
    assert "orig_ip_bytes_decomposition" not in active_names


def test_the_emitted_header_records_every_rejection(conn_like):
    """The file has to be readable on its own, not only beside its report."""
    from talos.data.preprocess.prelabel.screen import emit_declaration

    base, report = screened(conn_like)
    text = emit_declaration(report, base, "s")
    header = text.split("\n\n")[0]
    for rejection in report.rejections:
        assert rejection.column in header
    assert base.sha in header


def test_suggested_phi_scales_each_constraint_by_its_own_target(conn_like):
    """Eq. 2's phi_m is per-constraint because one scalar cannot fit them all.

    A residual carries the units of its target, so phi = ratio / p50(target)
    makes it dimensionless. On real conn data the targets span orders of
    magnitude, so the phis do too -- which is why the equation indexes phi by m.
    """
    from talos.data.preprocess.prelabel.screen import PHI_RATIO, suggested_phi

    base, report = screened(conn_like)
    phi = suggested_phi(report, base)
    assert phi, "conn declares constraints; none were scored"

    for entry in phi.values():
        if entry["phi"] is not None:
            assert entry["phi"] == pytest.approx(
                PHI_RATIO / abs(entry["target_p50"]))

    values = [e["phi"] for e in phi.values() if e["phi"] is not None]
    assert max(values) / min(values) > 10, "a single global phi would suffice"


def test_an_unnormalisable_target_yields_no_phi(conn_like):
    """Nothing can be normalised by zero; say so rather than emit an infinity."""
    from talos.data.preprocess.prelabel.screen import suggested_phi

    base, report = screened(conn_like)
    for entry in suggested_phi(report, base).values():
        if entry["phi"] is None:
            assert "cannot be made dimensionless" in entry["why"]


def test_a_source_missing_a_declared_column_is_refused(conn_like):
    """DuckDB would report a self-reference and name the wrong problem."""
    from talos.data.feature.featureset import FeatureSetLoader
    from talos.data.preprocess.prelabel.screen import preflight

    base = FeatureSetLoader().load("conn")
    with pytest.raises(ScreenError, match="which the source does not have"):
        preflight(base, {"orig_bytes": "BIGINT"})


def test_an_undefined_statistic_is_null_not_nan(duck):
    """A constant column has no skew, and `json.dumps` writes NaN unparseably.

    Left as NaN it reaches the run report as a bare `NaN` token, which is not
    valid JSON and which no strict parser will read back.
    """
    report = screen(duck)
    constant = next(s for s in report.stats if s.name == "never_set")
    assert constant.raw.skewness is None
    written = json.dumps(report.to_dict(), allow_nan=False, default=str)
    assert json.loads(written)["dataset"] == "toy"

# ------------------------------------------- declared indicators (Gate 1)


def test_declared_indicators_come_from_the_declaration(conn_like):
    from talos.data.feature.featureset import FeatureSetLoader
    from talos.data.preprocess.prelabel.screen import declared_indicators
    base = FeatureSetLoader().load("conn")
    assert declared_indicators(base) == {"duration", "orig_bytes", "resp_bytes"}


def test_a_mostly_null_declared_feature_survives_the_missingness_rule(tmp_path):
    """The defect this closes: screening `conn` deleted its own indicated features.

    `duration IS NULL` means "no observed teardown" and `conn` declares an
    indicator for it, so a mostly-NULL column is information, not an absent one.
    """
    from talos.data.feature.featureset import FeatureSetLoader
    from talos.data.preprocess.prelabel.screen import (
        declared_indicators, feature_source, preflight)
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE flows AS SELECT
            CAST(i % 5000 AS BIGINT)            AS orig_bytes,
            CAST(i % 5000 AS BIGINT) + 40       AS orig_ip_bytes,
            CAST((i * 7) % 9000 AS BIGINT)      AS resp_bytes,
            CAST((i * 7) % 9000 AS BIGINT) + 40 AS resp_ip_bytes,
            CAST(i % 40 AS BIGINT)              AS orig_pkts,
            CAST((i * 3) % 55 AS BIGINT)        AS resp_pkts,
            CAST(0 AS BIGINT)                   AS missed_bytes,
            CASE WHEN i % 10 < 9 THEN NULL
                 ELSE CAST(i % 900 AS DOUBLE) / 3.0 END AS duration,
            CASE WHEN i % 3 = 0 THEN 'tcp' ELSE 'udp' END AS proto,
            CASE WHEN i % 5 = 0 THEN 'S0' ELSE 'SF' END   AS conn_state,
            CAST(NULL AS BOOLEAN)               AS local_orig,
            CAST(NULL AS BOOLEAN)               AS local_resp,
            CASE WHEN i % 5 = 0 THEN 'Sr' ELSE 'ShADadFf' END AS history
        FROM range(0, 2000) t(i)""")
    duck = Duck(con)
    base = FeatureSetLoader().load("conn")
    preflight(base, duck.columns("flows"))
    source, numeric, categorical = feature_source(base, "flows")

    naive = FeatureScreener(duck).screen("toy", source, numeric, categorical)
    assert "duration" in {r.column for r in naive.rejections
                          if r.check == "missingness"}

    aware = FeatureScreener(
        duck, indicated=sorted(declared_indicators(base)),
    ).screen("toy", source, numeric, categorical)
    assert "duration" not in {r.column for r in aware.rejections}
    assert "duration" in aware.kept


# --------------------------------------- measured standardisation (Gate 2)


def test_a_log1p_feature_is_measured_in_log_space(conn_like):
    """Constants are applied AFTER the transform, so they must be measured there."""
    import math
    from talos.data.preprocess.prelabel.screen import standardisation
    base, report = screened(conn_like)
    fitted = standardisation(report, base)["orig_bytes"]
    assert fitted["space"] == "log1p"
    raw_mean = conn_like.one("SELECT avg(orig_bytes) FROM flows")[0]
    assert fitted["centre"] < math.log1p(raw_mean)      # log of a mean is larger
    assert fitted["centre"] == pytest.approx(
        conn_like.one("SELECT avg(ln(1 + orig_bytes)) FROM flows")[0], abs=1e-5)


def test_an_untransformed_feature_is_measured_in_original_units(conn_like):
    from talos.data.preprocess.prelabel.screen import standardisation
    base, report = screened(conn_like)
    assert standardisation(report, base)["history_len"]["space"] == "original"


def test_a_constant_column_gets_no_constants(conn_like):
    """Dividing by a zero scale is refused by the loader; do not emit one."""
    from talos.data.preprocess.prelabel.screen import standardisation
    base, report = screened(conn_like)
    fitted = standardisation(report, base)["missed_bytes"]
    assert fitted["centre"] is None and fitted["scale"] is None
    assert "no scale" in fitted["why"]


def test_the_emitted_schema_carries_measured_constants(conn_like):
    import yaml
    from talos.data.preprocess.prelabel.screen import emit_declaration
    base, report = screened(conn_like)
    doc = yaml.safe_load(emit_declaration(report, base, "conn-v1"))
    assert doc["numeric"]["orig_bytes"]["centre"] is not None
    assert doc["numeric"]["orig_bytes"]["scale"] > 0


def test_the_emitted_schema_standardises_when_loaded(conn_like, tmp_path):
    """The round trip is the point: a constant nothing reads back is not a constant."""
    from talos.data.feature.featureset import FeatureSetLoader
    from talos.data.preprocess.prelabel.screen import emit_declaration
    base, report = screened(conn_like)
    out = tmp_path / "conn-v1.yaml"
    out.write_text(emit_declaration(report, base, "conn-v1"))
    loaded = FeatureSetLoader().load(out)
    standardised = {f.name for f in loaded.numeric if f.standardised}
    assert "orig_bytes" in standardised
    assert all(f.scale > 0 for f in loaded.numeric if f.standardised)


def test_categoricals_are_never_given_constants(conn_like):
    import yaml
    from talos.data.preprocess.prelabel.screen import emit_declaration
    base, report = screened(conn_like)
    doc = yaml.safe_load(emit_declaration(report, base, "conn-v1"))
    for entry in (doc.get("categorical") or {}).values():
        assert "centre" not in (entry or {}) and "scale" not in (entry or {})


def test_the_base_declaration_is_never_mutated(conn_like):
    """`emit_declaration` copies; `conn` must stay free of invented constants."""
    from talos.data.feature.featureset import FeatureSetLoader
    from talos.data.preprocess.prelabel.screen import emit_declaration
    base, report = screened(conn_like)
    emit_declaration(report, base, "conn-v1")
    assert not [f for f in FeatureSetLoader().load("conn").numeric if f.standardised]


def test_an_impossible_row_does_not_set_the_scale(tmp_path):
    """A near-zero duration dividing a real byte count gives ~1e18 B/s."""
    from talos.data.feature.featureset import FeatureSetLoader
    from talos.data.preprocess.prelabel.screen import (
        FeatureScreener, declared_indicators, exclude_invalid, feature_source,
        preflight, standardisation)
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE flows AS SELECT
            CAST(i % 5000 AS BIGINT) AS orig_bytes,
            CAST(i % 5000 AS BIGINT) + 40 AS orig_ip_bytes,
            CAST((i * 7) % 9000 AS BIGINT) AS resp_bytes,
            CAST((i * 7) % 9000 AS BIGINT) + 40 AS resp_ip_bytes,
            CAST(i % 40 AS BIGINT) + 1 AS orig_pkts,
            CAST((i * 3) % 55 AS BIGINT) + 1 AS resp_pkts,
            CAST(0 AS BIGINT) AS missed_bytes,
            CASE WHEN i = 0 THEN 1e-12
                 ELSE CAST(i % 900 AS DOUBLE) / 3.0 + 0.5 END AS duration,
            CASE WHEN i % 3 = 0 THEN 'tcp' ELSE 'udp' END AS proto,
            CASE WHEN i % 5 = 0 THEN 'S0' ELSE 'SF' END AS conn_state,
            CAST(NULL AS BOOLEAN) AS local_orig, CAST(NULL AS BOOLEAN) AS local_resp,
            CASE WHEN i % 5 = 0 THEN 'Sr' ELSE 'ShADadFf' END AS history
        FROM range(0, 2000) t(i)""")
    con.execute("UPDATE flows SET orig_bytes = 4999 WHERE duration < 1e-9")
    # Exactly one impossible row: 4999 B over 1e-12 s is ~5e15 B/s.
    duck = Duck(con)
    base = FeatureSetLoader().load("conn")
    preflight(base, duck.columns("flows"))
    indicated = sorted(declared_indicators(base))

    def fit(src):
        source, numeric, categorical = feature_source(base, src)
        report = FeatureScreener(duck, indicated=indicated).screen(
            "toy", source, numeric, categorical)
        return standardisation(report, base)["orig_rate"]

    filtered, excluded, total = exclude_invalid(duck, "flows")
    assert excluded == 1 and total == 2000
    assert fit(filtered)["scale"] < fit("flows")["scale"]
