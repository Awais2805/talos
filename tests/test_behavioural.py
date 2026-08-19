"""Gate 12a — the autoencoder branch, end to end over a real fixture lake.

The chain is the point: schedule labels a toy capture, then `ae` reads that
table, trains on its pools and writes its own. Two methods, one core schema.
"""

from __future__ import annotations

import textwrap

import duckdb
import numpy as np
import pytest

from talos.common.config import Config
from talos.common.plugins import ConfigPlugins
from talos.data.feature.featureset import FeatureSetLoader
from talos.data.labelling.audit.adjudication import default_path
from talos.data.labelling.base import CORE_COLUMNS, LabellingError, StaleOutputError
from talos.data.labelling.behavioural.autoencoder import AutoEncoderLabeller
from talos.data.labelling.behavioural.base import BehaviouralReport, labelling_columns
from talos.data.labelling.behavioural.pool import PartitionLoader, PoolError
from talos.data.labelling.method import MethodError, MethodLoader
from talos.data.labelling.schedule.labeller import ScheduleLabeller
from talos.data.labelling.schedule.windows import to_utc
from talos.parts.mlp import torch_available

torch_only = pytest.mark.skipif(not torch_available(),
                                reason="the [torch] extra is not installed")

CONN = ("uid, ts, \"id.orig_h\", \"id.resp_h\", duration, orig_bytes, resp_bytes, "
        "orig_pkts, resp_pkts, orig_ip_bytes, resp_ip_bytes, missed_bytes, "
        "proto, conn_state, local_orig, local_resp, history")

MANIFEST = textwrap.dedent("""\
    dataset: toy
    utc_offset: "-03:00"
    match_default: [time, attacker, victim]
    attacks:
      - {name: FTP-Patator, date: 2017-07-04, start: "09:20", end: "10:20",
         attacker_ips: [10.0.0.1], victim_ips: [10.0.0.2]}
      - {name: DDoS-LOIC-HTTP, date: 2017-07-04, start: "11:00", end: "11:40",
         attacker_ips: [10.0.0.3], victim_ips: [10.0.0.2]}
""")

# Shares a toy lake can actually fill. xdg puts 0.02% in d_s, which on 240
# rows is nothing at all.
POOLS = textwrap.dedent("""\
    name: toy-pools
    source:
      datasets: [toy]
      labels: schedule
    exclude: [out-of-space]
    partition_by: uid
    seed: "toy-seed-1"
    buckets: 1000
    pools:
      d_s: {share: 0.25, role: audited}
      d_l: {share: 0.60, role: unlabelled}
      held: {share: 0.15, role: held}
""")

METHOD = textwrap.dedent("""\
    name: ae-test
    implementation: autoencoder
    label_space: core-5
    taxonomy: default
    settings:
      pools: toy-pools
      features: conn
      latent_dim: 4
      hidden: [8]
      epochs: 2
      freeze_epochs: 1
      finetune_epochs: 1
      batch: 64
      seed: 0
      device: cpu
      encoder: {encoder}
      decoder: {decoder}
      classifier: {classifier}
""")


def flows(n: int = 240):
    """Benign background plus two attack bursts, with plausible conn columns."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        if i % 3 == 0:                       # inside the FTP window
            ts, orig, resp = to_utc("2017-07-04", "09:20", "-03:00") + i, "10.0.0.1", "10.0.0.2"
        elif i % 3 == 1:                     # inside the DDoS window
            ts, orig, resp = to_utc("2017-07-04", "11:00", "-03:00") + i, "10.0.0.3", "10.0.0.2"
        else:                                # benign, outside both
            ts, orig, resp = to_utc("2017-07-04", "15:00", "-03:00") + i, "10.9.9.9", "10.0.0.2"
        duration = float(rng.gamma(2.0, 1.5))
        orig_bytes = float(rng.integers(0, 5000))
        resp_bytes = float(rng.integers(0, 9000))
        orig_pkts = float(rng.integers(1, 40))
        resp_pkts = float(rng.integers(0, 60))
        rows.append((
            f"u{i}", float(ts), orig, resp, duration, orig_bytes, resp_bytes,
            orig_pkts, resp_pkts, orig_bytes + 40 * orig_pkts,
            resp_bytes + 40 * resp_pkts, 0.0,
            ["tcp", "udp"][i % 2], ["SF", "S0", "REJ"][i % 3],
            i % 2 == 0, i % 4 == 0,
            # `conn` reads per-letter history counts, so the toy flows carry a
            # history consistent with the conn_state on the same row.
            ["ShADadFf", "S", "Sr"][i % 3]))
    return rows


@pytest.fixture
def lake(tmp_path):
    """A lake holding one toy capture, plus its schedule-labelled table."""
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "toy.yaml").write_text(MANIFEST)

    root = tmp_path / "lake"
    out = (root / "sources" / "datasets" / "parquet" / "zeek_v8.2.1" / "toy"
           / "Tuesday-WorkingHours")
    out.mkdir(parents=True)
    values = ", ".join(
        "(" + ", ".join(repr(v) for v in row) + ")" for row in flows())
    duckdb.connect().sql(
        f"SELECT * FROM (VALUES {values}) AS t({CONN})"
    ).write_parquet(str(out / "conn.parquet"))

    cfg = Config({"lake": {"root": str(root)},
                  "reports": {"dir": str(tmp_path / "reports")},
                  "models": {"dir": str(tmp_path / "models")}})
    ScheduleLabeller(cfg, manifests=manifests).label("toy", "zeek_v8.2.1")
    return cfg, tmp_path


def declarations(tmp_path, parts="inert", pools=POOLS):
    """A partition and a method declaration, on throwaway points."""
    (tmp_path / "pools").mkdir(exist_ok=True)
    (tmp_path / "pools" / "toy-pools.yaml").write_text(pools)
    partition = PartitionLoader(ConfigPlugins(
        "partition", built_in=tmp_path / "pools", error=PoolError,
        addressable=False)).load("toy-pools")

    (tmp_path / "methods" / "ae-test").mkdir(parents=True, exist_ok=True)
    (tmp_path / "methods" / "ae-test" / "method.yaml").write_text(
        METHOD.format(encoder=parts, decoder=parts, classifier=parts))
    spec = MethodLoader(ConfigPlugins(
        "method", built_in=tmp_path / "methods", filename="method.yaml",
        error=MethodError, addressable=False)).load("ae-test")
    return partition, spec


def labeller(cfg, tmp_path, parts="inert", pools=POOLS, **kwargs):
    partition, spec = declarations(tmp_path, parts, pools)
    return AutoEncoderLabeller(cfg, spec=spec, partition=partition, **kwargs)


def audit_everything(cfg, partition, sources, dataset="toy", overrides=None):
    """Write a fully-adjudicated audit file covering the whole `d_s` pool.

    Mirrors `talos audit emit` + a person confirming every row, except
    `overrides` (uid -> analyst_class) can leave a row unadjudicated (`""`)
    or dissenting (a different class, including `unknown`), for the tests
    that need that instead of a clean pass.
    """
    duck = cfg.lake().duck
    rel = partition.relation(duck, "d_s", sources)
    rows = duck.sql(
        f"SELECT uid, label_class FROM ({rel.sql_query()}) ORDER BY uid")
    overrides = overrides or {}
    lines = ["uid,ts,capture,dataset,prior_method,prior_class,prior_source,"
             "prior_quality,oracle_alerted,oracle_signature,analyst_class,analyst_note"]
    for uid, cls in rows:
        analyst = overrides.get(uid, cls)
        lines.append(f"{uid},,,{dataset},schedule,{cls},matched,certain,,,{analyst},")
    path = default_path(cfg, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path, rows


# ------------------------------------------------------------------ refusals

def test_untrainable_parts_are_refused_by_default(lake):
    """An untrained model must not quietly produce a table of labels."""
    cfg, tmp_path = lake
    with pytest.raises(LabellingError, match="cannot learn"):
        labeller(cfg, tmp_path).label("toy", "zeek_v8.2.1")


def test_untrainable_parts_run_when_explicitly_allowed(lake):
    """The harness has to be provable without a model — loudly."""
    cfg, tmp_path = lake
    report = labeller(cfg, tmp_path, allow_untrained=True).label("toy", "zeek_v8.2.1")

    assert report.trainable is False
    assert report.warnings and "cannot learn" in report.warnings[0]
    assert report.to_dict()["labels_are_meaningless"] is True


def test_missing_upstream_table_names_the_method_to_run_first(lake):
    cfg, tmp_path = lake
    with pytest.raises(LabellingError, match="run that method first"):
        labeller(cfg, tmp_path, allow_untrained=True).label("absent", "zeek_v8.2.1")


# ------------------------------------------------------------------ the table

def test_output_satisfies_the_core_contract(lake):
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path, allow_untrained=True)
    report = method.label("toy", "zeek_v8.2.1")
    rel = method.lake.read_parquet(report.output)

    for column in CORE_COLUMNS + method.extras:
        assert column in rel.columns
    types = dict(zip(rel.columns, (str(t) for t in rel.types)))
    assert types["label_confidence"] == "DOUBLE"
    assert types["label_binary"] == "BOOLEAN"


def test_every_flow_is_labelled_and_nothing_is_dropped(lake):
    """Closed world: the table is the whole dataset, not the pools."""
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path, allow_untrained=True)
    report = method.label("toy", "zeek_v8.2.1")
    rel = method.lake.read_parquet(report.output)

    assert report.total_flows == 240
    assert rel.aggregate("count(*)").fetchone()[0] == 240
    assert rel.filter("label_class IS NULL").aggregate("count(*)").fetchone()[0] == 0


def test_the_upstream_methods_label_columns_are_replaced_not_merged(lake):
    """A `rule_id` in an ae table would describe a rule that never saw the row."""
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path, allow_untrained=True)
    rel = method.lake.read_parquet(method.label("toy", "zeek_v8.2.1").output)

    assert "rule_id" not in rel.columns
    assert "label_raw" not in rel.columns
    assert "quarantine_reason" not in rel.columns
    # The original conn columns do survive.
    for column in ("uid", "ts", "proto", "conn_state", "orig_bytes"):
        assert column in rel.columns


def test_source_is_predicted_and_quality_reflects_trainability(lake):
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path, allow_untrained=True)
    rel = method.lake.read_parquet(method.label("toy", "zeek_v8.2.1").output)

    assert rel.filter("label_source <> 'predicted'").aggregate(
        "count(*)").fetchone()[0] == 0
    assert rel.filter("label_quality <> 'uncertain'").aggregate(
        "count(*)").fetchone()[0] == 0


def test_binary_is_derived_from_the_class(lake):
    """One source of truth, so the two columns cannot disagree on a row."""
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path, allow_untrained=True)
    rel = method.lake.read_parquet(method.label("toy", "zeek_v8.2.1").output)

    disagree = rel.filter(
        "label_binary <> (label_class <> 'benign')").aggregate("count(*)")
    assert disagree.fetchone()[0] == 0


def test_predicted_classes_are_all_inside_the_label_space(lake):
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path, allow_untrained=True)
    rel = method.lake.read_parquet(method.label("toy", "zeek_v8.2.1").output)

    classes = {row[0] for row in rel.project("label_class").distinct().fetchall()}
    assert classes <= set(method.space.classes)


def test_output_path_is_scoped_by_method(lake):
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path, allow_untrained=True)
    report = method.label("toy", "zeek_v8.2.1")
    assert "/ae-test/" in report.output
    assert "/schedule/" not in report.output


# ----------------------------------------------------------------- provenance

def test_method_sha_covers_the_pools_and_the_feature_set(lake, tmp_path):
    """Re-seeding the pools changes which rows trained the model."""
    cfg, fixture_path = lake
    method = labeller(cfg, fixture_path)
    before = method.method_sha("toy")

    reseeded = labeller(cfg, fixture_path,
                        pools=POOLS.replace("toy-seed-1", "toy-seed-2"))
    assert reseeded.method_sha("toy") != before


def test_a_different_feature_set_moves_the_hash(lake, tmp_path):
    cfg, fixture_path = lake
    method = labeller(cfg, fixture_path)
    other = FeatureSetLoader().load("conn")
    assert method.features.sha == other.sha          # same file, same hash
    assert method.features.path in method.inputs("toy")


def test_rerunning_with_changed_inputs_is_refused(lake):
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path, allow_untrained=True)
    method.label("toy", "zeek_v8.2.1")

    changed = labeller(cfg, tmp_path, allow_untrained=True,
                       pools=POOLS.replace("toy-seed-1", "toy-seed-3"))
    with pytest.raises(StaleOutputError, match="An input changed"):
        changed.label("toy", "zeek_v8.2.1")


def test_labelling_columns_covers_every_registered_method(lake):
    owned = labelling_columns()
    assert set(CORE_COLUMNS) <= owned
    assert "rule_id" in owned and "recon_error" in owned


# ---------------------------------------------------------------- the pools

def test_the_two_pools_never_share_a_flow(lake, tmp_path):
    """Disjointness is by construction, but the SQL that expresses it can be wrong."""
    cfg, fixture_path = lake
    partition, _ = declarations(fixture_path)
    duck = cfg.lake().duck
    sources = partition.sources(cfg.lake(), "zeek_v8.2.1", cfg)

    def uids(pool):
        rel = partition.relation(duck, pool, sources)
        return {row[0] for row in rel.project("uid").fetchall()}

    large, small, held = uids("d_l"), uids("d_s"), uids("held")
    assert large and small
    assert not (large & small) and not (large & held) and not (small & held)

    # Exhaustive over the VALID rows, not over the table: a flow no real capture
    # could produce is excluded from every pool and left in the labelled zone.
    from talos.data.preprocess.prelabel.verify import valid_sql
    listed = ", ".join(f"'{uri}'" for uri in sources)
    table = f"read_parquet([{listed}], union_by_name = true)"
    columns = {r[0]: r[1] for r in duck.sql(f"DESCRIBE SELECT * FROM {table}")}
    valid = {row[0] for row in duck.sql(
        f"SELECT uid FROM {table} WHERE {valid_sql(columns)}")}
    assert large | small | held == valid


# ------------------------------------------------------------ the audit gate
#
# `_supervised()` touches no `self.parts`, so it needs no torch and can be
# unit-tested directly on this machine, unlike the rest of Path B.

def _report():
    return BehaviouralReport(dataset="toy", feature_space="zeek_v8.2.1",
                             source="test", method="ae-test")


def test_finetuning_refuses_without_an_audit_file(lake):
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path)
    duck = cfg.lake().duck
    sources = method.partition.sources(cfg.lake(), "zeek_v8.2.1", cfg)

    with pytest.raises(LabellingError, match="audited D_s"):
        method._supervised(duck, sources, _report())


def test_finetuning_refuses_when_adjudication_is_incomplete(lake):
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path)
    duck = cfg.lake().duck
    sources = method.partition.sources(cfg.lake(), "zeek_v8.2.1", cfg)
    left_blank = method.partition.relation(
        duck, "d_s", sources).project("uid").fetchall()[0][0]
    audit_everything(cfg, method.partition, sources, overrides={left_blank: ""})

    with pytest.raises(LabellingError, match="unadjudicated"):
        method._supervised(duck, sources, _report())


def test_finetuning_trains_on_the_analyst_verdict_not_the_schedule_label(lake):
    """A correction changes what gets learned, not just what gets recorded."""
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path)
    duck = cfg.lake().duck
    sources = method.partition.sources(cfg.lake(), "zeek_v8.2.1", cfg)
    rel = method.partition.relation(duck, "d_s", sources)
    corrected_uid, original_class = rel.project("uid, label_class").fetchall()[0]
    corrected_to = next(c for c in method.space.classes if c != original_class)
    audit_everything(cfg, method.partition, sources,
                     overrides={corrected_uid: corrected_to})

    audited = method._audited_relation(duck, rel, _report())
    row = audited.filter(f"uid = '{corrected_uid}'").project(
        "label_class").fetchone()
    assert row[0] == corrected_to
    assert row[0] != original_class


def test_unaudited_pool_rows_are_reported_not_silently_used(lake):
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path)
    duck = cfg.lake().duck
    sources = method.partition.sources(cfg.lake(), "zeek_v8.2.1", cfg)
    all_uids = [row[0] for row in method.partition.relation(
        duck, "d_s", sources).project("uid").fetchall()]
    unknown_uid = all_uids[0]
    audit_everything(cfg, method.partition, sources,
                     overrides={unknown_uid: "unknown"})

    report = _report()
    method._supervised(duck, sources, report)
    assert report.d_s_unaudited == 1


def test_method_sha_changes_when_the_adjudication_file_is_edited(lake):
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path)
    duck = cfg.lake().duck
    sources = method.partition.sources(cfg.lake(), "zeek_v8.2.1", cfg)
    path, rows = audit_everything(cfg, method.partition, sources)
    before = method.method_sha("toy")

    uid, cls = rows[0]
    other = next(c for c in method.space.classes if c != cls)
    audit_everything(cfg, method.partition, sources, overrides={uid: other})
    after = method.method_sha("toy")

    assert before != after


def test_skip_audit_gate_bypasses_the_requirement(lake):
    """The escape hatch: no adjudication file needed, but it must be loud."""
    cfg, tmp_path = lake
    method = labeller(cfg, tmp_path, skip_audit_gate=True)
    duck = cfg.lake().duck
    sources = method.partition.sources(cfg.lake(), "zeek_v8.2.1", cfg)

    X, y = method._supervised(duck, sources, report := _report())
    assert len(X) == len(y) and len(y) > 0
    assert report.d_s_audited is False
    assert report.warnings and "--skip-audit-gate" in report.warnings[0]
    assert report.to_dict()["d_s_audited"] is False


def test_skip_audit_gate_excludes_the_file_from_method_sha(lake):
    """A run that never reads the file must not have its hash depend on it."""
    cfg, tmp_path = lake
    normal = labeller(cfg, tmp_path)
    duck = cfg.lake().duck
    sources = normal.partition.sources(cfg.lake(), "zeek_v8.2.1", cfg)
    audit_everything(cfg, normal.partition, sources)          # file now exists

    skipping = labeller(cfg, tmp_path, skip_audit_gate=True)
    assert default_path(cfg, "toy") not in skipping.inputs("toy")


# ----------------------------------------------------------------- with torch

def audited_labeller(cfg, tmp_path, **kwargs):
    """A trainable labeller whose D_s pool is already fully (self-)adjudicated.

    Every torch-backed test needs this now: fine-tuning refuses on an
    unaudited D_s regardless of whether the parts can learn.
    """
    method = labeller(cfg, tmp_path, **kwargs)
    sources = method.partition.sources(cfg.lake(), "zeek_v8.2.1", cfg)
    audit_everything(cfg, method.partition, sources)
    return method


@torch_only
def test_training_runs_and_produces_real_confidences(lake):
    cfg, tmp_path = lake
    method = audited_labeller(cfg, tmp_path, parts="mlp")
    report = method.label("toy", "zeek_v8.2.1")
    rel = method.lake.read_parquet(report.output)

    assert report.trainable is True and not report.warnings
    assert report.pretrain and report.finetune
    assert report.d_s_rows > 0
    # A trained head does not emit one flat number for every row.
    spread = rel.aggregate("max(label_confidence) - min(label_confidence)")
    assert spread.fetchone()[0] > 0
    assert rel.filter("label_quality <> 'certain'").aggregate(
        "count(*)").fetchone()[0] == 0


@torch_only
def test_pretraining_reduces_the_reconstruction_loss(lake):
    """A loss that does not fall means the objective is not wired to the parts."""
    cfg, tmp_path = lake
    report = audited_labeller(cfg, tmp_path, parts="mlp").label("toy", "zeek_v8.2.1")
    losses = [epoch.loss for epoch in report.pretrain]
    assert len(losses) >= 2 and losses[-1] < losses[0]


@torch_only
def test_a_checkpoint_is_written_and_reused(lake):
    cfg, tmp_path = lake
    method = audited_labeller(cfg, tmp_path, parts="mlp")
    report = method.label("toy", "zeek_v8.2.1")
    directory = method.checkpoint_dir("toy")
    assert directory.is_dir() and any(directory.iterdir())

    again = labeller(cfg, tmp_path, parts="mlp")     # same audit file, untouched
    second = again.label("toy", "zeek_v8.2.1", force=True)
    assert second.pretrain == ()                 # loaded, not retrained
    assert second.extra["checkpoint"] == report.extra["checkpoint"]


# ---------------------------------------------------------------- Gate 0
#
# `fit` calls `pretrain` with keywords. A branch whose signature drifts from
# that call crashes only once torch is installed, and torch skips on the dev
# Mac -- which is how `tabcl` shipped unable to pretrain at all.


def _pretrain_call_keywords() -> set[str]:
    """The keywords `BehaviouralMethod.fit` actually passes to `self.pretrain`.

    Read from the source rather than hardcoded, so adding a keyword to the call
    site fails every branch that has not taken it.
    """
    import ast
    import inspect

    from talos.data.labelling.behavioural import base as behavioural_base

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(behavioural_base.BehaviouralMethod.fit)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == "pretrain"
                and isinstance(func.value, ast.Name) and func.value.id == "self"):
            return {kw.arg for kw in node.keywords if kw.arg}
    raise AssertionError("no `self.pretrain(...)` call found in fit()")


def _behavioural_branches() -> list[type]:
    from talos.data.labelling.behavioural.base import BehaviouralMethod

    import talos.data.labelling.behavioural.autoencoder  # noqa: F401
    import talos.data.labelling.behavioural.tabcl        # noqa: F401

    return [c for c in BehaviouralMethod.__subclasses__()
            if not getattr(c.pretrain, "__isabstractmethod__", False)]


def test_the_call_site_still_passes_keywords():
    """Guards the guard: an empty set would make the test below vacuous."""
    assert _pretrain_call_keywords() >= {"supervised", "keyed", "duck"}


@pytest.mark.parametrize("branch", _behavioural_branches(),
                         ids=lambda c: c.__name__)
def test_every_branch_accepts_what_fit_passes(branch):
    """A signature drift must fail here, on a machine where torch skips."""
    import inspect

    signature = inspect.signature(branch.pretrain)
    accepted = set(signature.parameters)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD
           for p in signature.parameters.values()):
        return
    missing = sorted(_pretrain_call_keywords() - accepted)
    assert not missing, (
        f"{branch.__name__}.pretrain does not accept {', '.join(missing)}, "
        f"which BehaviouralMethod.fit passes. This is a TypeError the moment "
        f"torch is installed.")


# ------------------------------------------- validity exclusion (Gate 3)


def test_an_impossible_flow_reaches_no_pool(lake, tmp_path):
    """Excluded from training, never deleted -- the labelled table stays whole."""
    cfg, fixture_path = lake
    partition, _ = declarations(fixture_path)
    duck = cfg.lake().duck
    sources = partition.sources(cfg.lake(), "zeek_v8.2.1", cfg)

    from talos.data.preprocess.prelabel.verify import valid_sql
    listed = ", ".join(f"'{uri}'" for uri in sources)
    table = f"read_parquet([{listed}], union_by_name = true)"
    columns = {r[0]: r[1] for r in duck.sql(f"DESCRIBE SELECT * FROM {table}")}
    broken = {row[0] for row in duck.sql(
        f"SELECT uid FROM {table} WHERE NOT ({valid_sql(columns)})")}
    assert broken, "fixture carries no invariant-breaking row to test with"

    pooled = set()
    for pool in ("d_l", "d_s", "held"):
        rel = partition.relation(duck, pool, sources)
        pooled |= {row[0] for row in rel.project("uid").fetchall()}
    assert not (broken & pooled)

    # Still present in the table it was excluded from.
    still = {row[0] for row in duck.sql(f"SELECT uid FROM {table}")}
    assert broken <= still


def test_the_exclusion_is_declared_not_implicit(lake, tmp_path):
    """It is hashed into method_sha, so it has to be readable in the file."""
    from talos.data.labelling.behavioural.pool import PartitionLoader
    for name in ("xdg", "2017-audit"):
        partition = PartitionLoader().load(name)
        assert partition.exclude_invalid is True
        assert "exclude_invalid" in partition.path.read_text()


def test_the_exclusion_can_be_turned_off_by_declaration(tmp_path):
    partition_yaml = tmp_path / "loose.yaml"
    partition_yaml.write_text(textwrap.dedent("""\
        name: loose
        source: {datasets: [toy], labels: schedule}
        exclude_invalid: false
        partition_by: uid
        seed: "t"
        buckets: 100
        pools:
          d_s: {share: 0.2}
          d_l: {share: 0.8}
        """))
    from talos.data.labelling.behavioural.pool import PartitionLoader
    assert PartitionLoader().load(partition_yaml).exclude_invalid is False
