"""Gate 0.1 — the fusion explain report, checked against a hand-computed fixture.

One row per branch of Eq. 17 plus the NULL-confidence case the equation does not
cover. The load-bearing test is `test_every_decision_predicts_the_real_fusion`:
a breakdown that disagrees with what `fuse` actually chose is worse than no
breakdown at all, because it would be quoted as evidence.
"""

from __future__ import annotations

import json
import textwrap

import duckdb
import pytest

from talos.common.config import Config
from talos.common.plugins import ConfigPlugins
from talos.data.labelling.fusion.explain import (
    AGREED, FIRST_CONFIDENCE, FIRST_MARGIN, NULL_CONFIDENCE, SECOND_CONFIDENCE,
    SECOND_FALLBACK, TIED, WINNER,
)
from talos.data.labelling.fusion.fuse import FusedLabeller, FusionError
from talos.data.labelling.method import MethodError, MethodLoader

METHOD = textwrap.dedent("""\
    name: fused-explain-test
    implementation: fused
    label_space: core-5
    taxonomy: default
    settings:
      inputs: [ae, tabcl]
      features: conn
      confident_learning: false
""")

# uid, class, confidence, margin. Confidence None writes a typed NULL.
AE = [
    ("agree",       "ddos", 0.90, 0.50),
    ("a_confident", "ddos", 0.90, 0.10),
    ("b_confident", "ddos", 0.40, 0.10),
    ("margin_a",    "ddos", 0.70, 0.60),   # confidences tie, A has the margin
    ("margin_b",    "ddos", 0.70, 0.10),   # confidences tie, B has the margin
    ("all_tie",     "ddos", 0.70, 0.30),   # everything ties -> `>=` gives it to A
    ("null_conf",   "ddos", None, 0.90),   # A cannot measure; margin alone decides
]
TABCL = [
    ("agree",       "ddos", 0.60, 0.20),
    ("a_confident", "dos",  0.50, 0.40),
    ("b_confident", "dos",  0.80, 0.30),
    ("margin_a",    "dos",  0.70, 0.20),
    ("margin_b",    "dos",  0.70, 0.50),
    ("all_tie",     "dos",  0.70, 0.30),
    ("null_conf",   "dos",  0.30, 0.10),
]

EXPECTED = {
    "agree": AGREED,
    "a_confident": FIRST_CONFIDENCE,
    "b_confident": SECOND_CONFIDENCE,
    "margin_a": FIRST_MARGIN,
    "margin_b": SECOND_FALLBACK,
    "all_tie": FIRST_MARGIN,
    # NULL confidence is not its own decision: Eq. 17 skips both confidence
    # lines and the margin line resolves it, here in the first branch's favour.
    "null_conf": FIRST_MARGIN,
}


def write_branch(path, rows):
    # Cast explicitly: a bare 0.90 binds as DECIMAL(2,1), but a real branch
    # table is DOUBLE because `verify_schema` says so.
    def conf(value):
        return "CAST(NULL AS DOUBLE)" if value is None else f"CAST({value} AS DOUBLE)"

    values = ", ".join(
        f"('{uid}', {ts}, 'cap', 'toy', 'm', '{cls}', 'predicted', 'sha', "
        f"{conf(c)}, CAST(NULL AS DOUBLE), 'certain', '2026-01-01', "
        f"CAST({margin} AS DOUBLE))"
        for ts, (uid, cls, c, margin) in enumerate(rows))
    duckdb.connect().sql(
        f"SELECT * FROM (VALUES {values}) AS t(uid, ts, capture, dataset, "
        f"label_method, label_class, label_source, method_sha, label_confidence, "
        f"label_weight, label_quality, labelled_at, label_margin)"
    ).write_parquet(str(path))


@pytest.fixture
def lake(tmp_path):
    root = tmp_path / "lake" / "sources" / "datasets" / "labelled" / "zeek_v8.2.1"
    for method, rows in (("ae", AE), ("tabcl", TABCL)):
        out = root / method / "toy"
        out.mkdir(parents=True)
        write_branch(out / "conn.parquet", rows)
    cfg = Config({"lake": {"root": str(tmp_path / "lake")},
                  "reports": {"dir": str(tmp_path / "reports")}})
    return cfg, tmp_path


def fused(cfg, tmp_path):
    (tmp_path / "methods" / "fused-explain-test").mkdir(parents=True, exist_ok=True)
    (tmp_path / "methods" / "fused-explain-test" / "method.yaml").write_text(METHOD)
    spec = MethodLoader(ConfigPlugins(
        "method", built_in=tmp_path / "methods", filename="method.yaml",
        error=MethodError, addressable=False)).load("fused-explain-test")
    return FusedLabeller(cfg, spec=spec)


def explain(lake):
    cfg, tmp_path = lake
    return fused(cfg, tmp_path).explain("toy", "zeek_v8.2.1")


# --------------------------------------------------------------- the breakdown

def test_each_row_lands_in_its_hand_computed_category(lake):
    """One row per branch of Eq. 17; each must be counted as that branch."""
    report = explain(lake)
    expected: dict[str, int] = {}
    for decision in EXPECTED.values():
        expected[decision] = expected.get(decision, 0) + 1
    assert report.decisions == expected


def test_categories_are_exclusive_and_total(lake):
    """A row counted twice, or not at all, makes every share wrong."""
    report = explain(lake)
    assert sum(report.decisions.values()) == report.rows_joined == len(AE)


def test_every_decision_predicts_the_real_fusion(lake):
    """The breakdown must agree with the label `fuse` actually writes.

    Explain and fuse share no code path, so this is the test that stops the
    report becoming a plausible fiction quoted as evidence.
    """
    cfg, tmp_path = lake
    method = fused(cfg, tmp_path)
    duck = method.lake.duck
    actual = dict(duck.sql(
        f"SELECT uid, label_class FROM ("
        f"{method.joined(duck, 'toy', 'zeek_v8.2.1').sql_query()})"))

    ae = {uid: cls for uid, cls, _, _ in AE}
    tabcl = {uid: cls for uid, cls, _, _ in TABCL}
    for uid, decision in EXPECTED.items():
        winner = WINNER[decision]
        want = ae[uid] if winner in ("first", "both") else tabcl[uid]
        assert actual[uid] == want, f"{uid}: {decision} predicted {want}"


def test_a_null_confidence_is_counted_not_hidden(lake):
    """A NULL demotes Eq. 17 to margin-only, and says nothing about doing so.

    Both `>` tests are NULL, so neither confidence line fires and the margin
    line decides — indistinguishable, in the decision counts alone, from an
    exact tie. `margin_causes` is what keeps the two apart.
    """
    report = explain(lake)
    assert report.null_confidence == 1
    assert report.margin_causes[TIED] == 3     # margin_a, margin_b, all_tie
    assert "decides them on" in report.summary()


def test_margin_causes_account_for_every_margin_row(lake):
    """A cause missing here would understate how often a NULL decided a row."""
    report = explain(lake)
    margin_rows = (report.decisions.get(FIRST_MARGIN, 0)
                   + report.decisions.get(SECOND_FALLBACK, 0))
    assert sum(report.margin_causes.values()) == margin_rows


def test_no_decision_category_outside_eq_17(lake):
    """A sixth category would describe a rule the pipeline does not run."""
    report = explain(lake)
    assert set(report.decisions) <= {
        AGREED, FIRST_CONFIDENCE, SECOND_CONFIDENCE, FIRST_MARGIN,
        SECOND_FALLBACK}


def test_disagreement_shares_use_disagreements_as_the_base(lake):
    """Shares over all rows would dilute the number the gate exists to read."""
    report = explain(lake)
    assert report.disagreements == len(AE) - 1
    won = (report.share_of_disagreements("first")
           + report.share_of_disagreements("second"))
    assert won == pytest.approx(1.0)


# ------------------------------------------------------------------- the rest

def test_confidence_is_reported_per_branch_and_class(lake):
    """A single pooled mean cannot show one branch sitting below the other."""
    report = explain(lake)
    branches = {row[0] for row in report.confidence}
    assert branches == {"ae", "tabcl"}
    for _branch, _cls, _n, _mean, quantiles in report.confidence:
        assert len(quantiles) == 9


def test_rows_with_no_confidence_are_excluded_from_the_distribution(lake):
    """Averaging over a NULL would report a confidence the branch never gave."""
    report = explain(lake)
    counted = sum(row[2] for row in report.confidence if row[0] == "ae")
    assert counted == len(AE) - 1


def test_the_report_is_written_and_names_its_inputs(lake):
    """The order of `inputs` is load-bearing: the first wins an exact tie."""
    cfg, tmp_path = lake
    explain(lake)
    written = json.loads(
        (tmp_path / "reports" / "fusion_explain_toy.json").read_text())
    assert written["inputs"] == ["ae", "tabcl"]
    assert written["decisions"][FIRST_MARGIN] == 3   # margin_a, all_tie, null_conf
    assert written["margin_line_causes"][NULL_CONFIDENCE] == 1
    assert set(written["decision_shares"]) == set(written["decisions"])


def test_a_missing_branch_table_is_refused_by_name(lake):
    """Explaining a fusion that was never run would report zeros as a result."""
    cfg, tmp_path = lake
    table = (tmp_path / "lake" / "sources" / "datasets" / "labelled"
             / "zeek_v8.2.1" / "tabcl" / "toy" / "conn.parquet")
    table.unlink()
    with pytest.raises(FusionError, match="nothing to explain"):
        fused(cfg, tmp_path).explain("toy", "zeek_v8.2.1")


def test_branches_reading_different_features_are_refused(lake):
    """Eq. 17 compares one branch's confidence against the other's.

    That comparison says nothing when the two models saw different columns or
    trained on different splits -- a confidence is only interpretable relative
    to what produced it. `ae` reads conn over the xdg pools; `schedule` reads
    neither, so the two disagree on every input the guard compares.
    """
    cfg, tmp_path = lake
    mismatched = METHOD.replace("[ae, tabcl]", "[ae, schedule]")
    (tmp_path / "methods" / "mixed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "methods" / "mixed" / "method.yaml").write_text(
        mismatched.replace("fused-explain-test", "mixed"))
    spec = MethodLoader(ConfigPlugins(
        "method", built_in=tmp_path / "methods", filename="method.yaml",
        error=MethodError, addressable=False)).load("mixed")

    with pytest.raises(FusionError, match="disagree on features"):
        FusedLabeller(cfg, spec=spec)._check_inputs_agree()


def test_matching_branches_pass_the_check(lake):
    """fused's two inputs are declared to agree; the guard must not fire."""
    cfg, tmp_path = lake
    fused(cfg, tmp_path)._check_inputs_agree()


def test_nothing_is_written_to_the_lake(lake):
    """`--explain` is measurement; a fused table appearing would be a defect."""
    cfg, tmp_path = lake
    explain(lake)
    labelled = tmp_path / "lake" / "sources" / "datasets" / "labelled" / "zeek_v8.2.1"
    assert sorted(p.name for p in labelled.iterdir()) == ["ae", "tabcl"]
