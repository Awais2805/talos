"""Gate 13 — the fusion rule, exactly as Eq. 17 states it.

Every branch of the rule gets its own row in the fixture, so a change to the
CASE expression cannot pass by satisfying only the common path.
"""

from __future__ import annotations

import textwrap

import duckdb
import pytest

from talos.common.config import Config
from talos.common.plugins import ConfigPlugins
from talos.data.labelling.base import CORE_COLUMNS, LabellingError
from talos.data.labelling.fusion.fuse import FusedLabeller, FusionError
from talos.data.labelling.method import MethodError, MethodLoader

METHOD = textwrap.dedent("""\
    name: fused-test
    implementation: fused
    label_space: core-5
    taxonomy: default
    settings:
      inputs: {inputs}
      features: conn
      confident_learning: {cl}
""")

# uid, class, confidence, margin — one row per branch of Eq. 17.
AE = [
    ("agree",        "ddos", 0.90, 0.50),
    ("a_confident",  "ddos", 0.90, 0.10),
    ("b_confident",  "ddos", 0.40, 0.10),
    ("margin_a",     "ddos", 0.70, 0.60),   # confidences tie, A has the margin
    ("margin_b",     "ddos", 0.70, 0.10),   # confidences tie, B has the margin
    ("all_tie",      "ddos", 0.70, 0.30),   # everything ties -> `>=` gives it to A
]
TABCL = [
    ("agree",        "ddos", 0.60, 0.20),
    ("a_confident",  "dos",  0.50, 0.40),
    ("b_confident",  "dos",  0.80, 0.30),
    ("margin_a",     "dos",  0.70, 0.20),
    ("margin_b",     "dos",  0.70, 0.50),
    ("all_tie",      "dos",  0.70, 0.30),
]


def write_branch(path, rows):
    # Confidences cast explicitly: a bare 0.90 literal binds as DECIMAL(2,1),
    # and a real branch table is DOUBLE because `verify_schema` says so.
    values = ", ".join(
        f"('{uid}', {ts}, 'cap', 'toy', 'm', '{cls}', 'predicted', 'sha', "
        f"CAST({conf} AS DOUBLE), CAST(NULL AS DOUBLE), 'certain', '2026-01-01', "
        f"CAST({margin} AS DOUBLE))"
        for ts, (uid, cls, conf, margin) in enumerate(rows))
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


def fused(cfg, tmp_path, inputs="[ae, tabcl]", cl="false", **kwargs):
    (tmp_path / "methods" / "fused-test").mkdir(parents=True, exist_ok=True)
    (tmp_path / "methods" / "fused-test" / "method.yaml").write_text(
        METHOD.format(inputs=inputs, cl=cl))
    spec = MethodLoader(ConfigPlugins(
        "method", built_in=tmp_path / "methods", filename="method.yaml",
        error=MethodError, addressable=False)).load("fused-test")
    return FusedLabeller(cfg, spec=spec, **kwargs)


def classes(method, report):
    rel = method.lake.read_parquet(report.output)
    return dict(rel.project("uid, label_class").fetchall())


# ------------------------------------------------------------------ the rule

def test_every_branch_of_equation_17(lake):
    """Agreement, then confidence, then margin, then a deterministic fallback."""
    cfg, tmp_path = lake
    method = fused(cfg, tmp_path)
    chosen = classes(method, method.label("toy", "zeek_v8.2.1"))

    assert chosen["agree"] == "ddos"          # both said ddos
    assert chosen["a_confident"] == "ddos"    # 0.90 > 0.50
    assert chosen["b_confident"] == "dos"     # 0.80 > 0.40
    assert chosen["margin_a"] == "ddos"       # confidences tie, 0.60 >= 0.20
    assert chosen["margin_b"] == "dos"        # confidences tie, 0.50 > 0.10
    # Everything ties. Eq. 17 line 4 is `>=`, so the FIRST input takes it — the
    # paper calls its last line a "deterministic fallback", but that line only
    # fires when the second branch wins the margin strictly.
    assert chosen["all_tie"] == "ddos"


def test_tie_breaking_follows_the_declared_order(lake):
    """Order is load-bearing: swapping the inputs swaps who wins an exact tie."""
    cfg, tmp_path = lake
    method = fused(cfg, tmp_path, inputs="[tabcl, ae]")
    chosen = classes(method, method.label("toy", "zeek_v8.2.1"))
    assert chosen["all_tie"] == "dos"


def test_agreement_is_recorded_per_row_and_summarised(lake):
    cfg, tmp_path = lake
    method = fused(cfg, tmp_path)
    report = method.label("toy", "zeek_v8.2.1")

    assert report.total_flows == 6 and report.agreed == 1
    assert report.agreement_rate == pytest.approx(1 / 6)
    rel = method.lake.read_parquet(report.output)
    agreed = dict(rel.project("uid, branches_agreed").fetchall())
    assert agreed["agree"] is True and agreed["a_confident"] is False


def test_agreement_is_never_called_accuracy(lake):
    """The post-mortem's central failure, guarded at the reporting boundary."""
    cfg, tmp_path = lake
    report = fused(cfg, tmp_path).label("toy", "zeek_v8.2.1")
    payload = report.to_dict()

    assert "branch_agreement_rate" in payload
    for key in ("accuracy", "f1", "precision", "score"):
        assert key not in payload
    assert "not accuracy" in payload["note"]


# ------------------------------------------------------------------ refusals

def test_fusion_over_anything_but_two_inputs_is_refused(lake):
    cfg, tmp_path = lake
    with pytest.raises(FusionError, match="defined over exactly two"):
        fused(cfg, tmp_path, inputs="[ae]")


def test_a_stale_output_is_refused_unless_forced(lake):
    """Re-fusing with a changed declaration must not silently replace the table.

    `refuse_stale_output` (labelling/base.py) is shared by schedule, every
    behavioural branch and fused; only the schedule and behavioural paths had a
    test before this one.
    """
    cfg, tmp_path = lake
    fused(cfg, tmp_path).label("toy", "zeek_v8.2.1")

    # Same output URI (keyed by run_name, not by settings), different
    # declaration content -> different method_sha.
    changed = fused(cfg, tmp_path, inputs="[tabcl, ae]")
    with pytest.raises(LabellingError, match="An input changed"):
        changed.label("toy", "zeek_v8.2.1")

    # --force overwrites anyway.
    report = changed.label("toy", "zeek_v8.2.1", force=True)
    assert classes(changed, report)["all_tie"] == "dos"


def test_a_missing_branch_names_the_command_to_run(lake):
    cfg, tmp_path = lake
    method = fused(cfg, tmp_path)
    with pytest.raises(FusionError, match="talos label --method"):
        method.label("absent", "zeek_v8.2.1")


def test_a_class_outside_the_label_space_is_refused(lake, tmp_path):
    """Two branches labelled over different spaces cannot be reconciled."""
    cfg, fixture = lake
    out = (fixture / "lake" / "sources" / "datasets" / "labelled" / "zeek_v8.2.1"
           / "ae" / "toy" / "conn.parquet")
    write_branch(out, [("odd", "portscan", 0.9, 0.5)])

    with pytest.raises(FusionError, match="outside label space"):
        fused(cfg, fixture).label("toy", "zeek_v8.2.1")


# ------------------------------------------------------------------ the table

def test_output_satisfies_the_core_contract(lake):
    cfg, tmp_path = lake
    method = fused(cfg, tmp_path)
    rel = method.lake.read_parquet(method.label("toy", "zeek_v8.2.1").output)

    for column in CORE_COLUMNS + method.extras:
        assert column in rel.columns
    assert rel.filter("label_source <> 'fused'").aggregate(
        "count(*)").fetchone()[0] == 0


def test_weight_is_null_when_confident_learning_is_off(lake):
    """A weight nobody computed must be NULL, not 1.0."""
    cfg, tmp_path = lake
    method = fused(cfg, tmp_path)
    report = method.label("toy", "zeek_v8.2.1")
    rel = method.lake.read_parquet(report.output)

    assert rel.filter("label_weight IS NOT NULL").aggregate(
        "count(*)").fetchone()[0] == 0
    assert any("disabled" in w for w in report.warnings)


def test_confident_learning_without_torch_refuses_by_default(lake):
    """Substituting the branches' own confidences would be in-sample by
    construction, so the run stops rather than inventing an estimate."""
    pytest.importorskip("torch") if False else None
    from talos.parts.mlp import torch_available
    if torch_available():
        pytest.skip("torch is installed, so the probe can run")

    cfg, tmp_path = lake
    method = fused(cfg, tmp_path, cl="true")
    with pytest.raises(LabellingError, match="PyTorch is not installed"):
        method.label("toy", "zeek_v8.2.1")


def test_confident_learning_may_be_waived_explicitly(lake):
    from talos.parts.mlp import torch_available
    if torch_available():
        pytest.skip("torch is installed, so the probe runs")

    cfg, tmp_path = lake
    method = fused(cfg, tmp_path, cl="true", allow_untrained=True)
    report = method.label("toy", "zeek_v8.2.1")

    assert any("PyTorch" in w for w in report.warnings)
    rel = method.lake.read_parquet(report.output)
    assert rel.filter("label_quality <> 'uncertain'").aggregate(
        "count(*)").fetchone()[0] == 0
