"""Gate 14 — the audited slice, and scoring methods against a person's verdict.

The audited rows are the only ground truth in the project, so this is where the
word "accuracy" becomes legitimate. Everything else the pipeline reports about
labels is agreement, and the tests below keep the two apart.
"""

from __future__ import annotations

import textwrap

import duckdb
import pytest

from talos.common.duck import DuckEngine
from talos.common.plugins import ConfigPlugins
from talos.data.labelling.audit import (
    AdjudicationTable, AuditError, CandidateSelector, LabelBenchmark, UNKNOWN,
)
from talos.data.labelling.behavioural.pool import PartitionLoader, PoolError
from talos.data.labelling.space import LabelSpaceLoader

POOLS = textwrap.dedent("""\
    name: toy-pools
    source:
      datasets: [toy]
      labels: schedule
    exclude: [out-of-space]
    partition_by: uid
    seed: "audit-seed-1"
    buckets: 1000
    pools:
      d_s: {share: 0.40, role: audited}
      d_l: {share: 0.45, role: unlabelled}
      held: {share: 0.15, role: held}
""")

CLASSES = ["benign", "ddos", "dos", "brute_force", "botnet"]


def labelled_rows(n: int = 400):
    """A labelled table: every class present, `botnet` deliberately rare."""
    rows = []
    for i in range(n):
        label = "botnet" if i % 97 == 0 else CLASSES[i % 4]
        rows.append((f"u{i}", float(1_500_000_000 + i), "cap-a", "toy", "schedule",
                     label, label != "benign", "matched", "certain",
                     "10.0.0.1", "10.0.0.2", 1.0))
    return rows


@pytest.fixture
def lake(tmp_path):
    """A schedule-labelled table on disk, plus a partition over it."""
    out = (tmp_path / "lake" / "sources" / "datasets" / "labelled" / "zeek_v8.2.1"
           / "schedule" / "toy")
    out.mkdir(parents=True)
    values = ", ".join("(" + ", ".join(repr(v) for v in row) + ")"
                       for row in labelled_rows())
    duckdb.connect().sql(
        f"SELECT * FROM (VALUES {values}) AS t(uid, ts, capture, dataset, "
        f"label_method, label_class, label_binary, label_source, label_quality, "
        f'"id.orig_h", "id.resp_h", duration)'
    ).write_parquet(str(out / "conn.parquet"))
    # The engine the pipeline actually uses, not a bare connection.
    duck = DuckEngine()

    (tmp_path / "pools").mkdir()
    (tmp_path / "pools" / "toy-pools.yaml").write_text(POOLS)
    partition = PartitionLoader(ConfigPlugins(
        "partition", built_in=tmp_path / "pools", error=PoolError,
        addressable=False)).load("toy-pools")

    sources = [str(out / "conn.parquet")]
    return duck, partition, sources, tmp_path


@pytest.fixture
def space():
    return LabelSpaceLoader().load("core-5")


# ------------------------------------------------------------ the candidates

def test_candidates_come_only_from_the_audit_pool(lake):
    """Auditing a flow that also pretrains the model destroys the whole point."""
    duck, partition, sources, _ = lake
    chosen, _ = CandidateSelector(partition, floor=5).select(duck, sources)

    picked = {row[0] for row in duck.sql(
        f"SELECT uid FROM ({chosen.sql_query()}) c")}
    for other in ("d_l", "held"):
        elsewhere = {row[0] for row in duck.sql(
            f"SELECT uid FROM ({partition.relation(duck, other, sources).sql_query()}) p")}
        assert not (picked & elsewhere)


def test_per_class_floor_is_met_where_the_pool_can_supply_it(lake):
    duck, partition, sources, _ = lake
    _, selection = CandidateSelector(partition, floor=5).select(duck, sources)
    counts = dict(selection.per_class)

    for label in ("benign", "ddos", "dos", "brute_force"):
        assert counts[label] == 5, f"{label}: {counts}"


def test_a_class_the_pool_cannot_fill_is_reported_not_topped_up(lake):
    """The invariant: never reach outside the pool to satisfy a floor.

    `botnet` is rare by construction. Borrowing one more from `d_l` would put an
    audited flow into the pretraining set — exactly what the partition exists to
    make impossible — so the shortfall is surfaced instead.
    """
    duck, partition, sources, _ = lake
    chosen, selection = CandidateSelector(partition, floor=50).select(duck, sources)

    assert selection.shortfalls, "botnet cannot supply 50 rows; that must be said"
    short = {s.label for s in selection.shortfalls}
    assert "botnet" in short

    picked = {row[0] for row in duck.sql(f"SELECT uid FROM ({chosen.sql_query()}) c")}
    outside = {row[0] for row in duck.sql(
        f"SELECT uid FROM ({partition.relation(duck, 'd_l', sources).sql_query()}) p")}
    assert not (picked & outside)


def test_selection_is_deterministic(lake):
    """The same declaration must pick the same rows, or a verdict cannot be matched."""
    duck, partition, sources, _ = lake
    first, _ = CandidateSelector(partition, floor=5).select(duck, sources)
    second, _ = CandidateSelector(partition, floor=5).select(duck, sources)
    assert ({r[0] for r in duck.sql(f"SELECT uid FROM ({first.sql_query()}) c")}
            == {r[0] for r in duck.sql(f"SELECT uid FROM ({second.sql_query()}) c")})


def test_the_cap_refuses_an_unreadable_pile(lake):
    """Every candidate is read by a person, so the total is bounded deliberately."""
    duck, partition, sources, _ = lake
    with pytest.raises(AuditError, match="exceeds the cap"):
        CandidateSelector(partition, floor=50, cap=10).select(duck, sources)


def test_an_unknown_pool_name_is_refused(lake):
    _, partition, _, _ = lake
    with pytest.raises(AuditError, match="declares no pool"):
        CandidateSelector(partition, pool="d_x")


# ---------------------------------------------------------- emit and ingest

def emitted(duck, partition, sources, space, tmp_path, floor=5):
    chosen, _ = CandidateSelector(partition, floor=floor).select(duck, sources)
    table = AdjudicationTable(space)
    return table, table.emit(duck, chosen, tmp_path / "audit.csv")


def test_emitted_file_has_the_verdict_columns_empty(lake, space):
    duck, partition, sources, tmp_path = lake
    _, path = emitted(duck, partition, sources, space, tmp_path)

    rel = duck.relation(f"SELECT * FROM read_csv_auto('{path}', header = true)")
    for column in ("uid", "prior_class", "oracle_alerted", "analyst_class"):
        assert column in rel.columns
    # Nothing pre-fills a verdict. The whole point is that a person types it.
    assert duck.one(f"SELECT count(*) FROM ({rel.sql_query()}) a "
                    f"WHERE analyst_class IS NOT NULL")[0] == 0


def test_ingest_refuses_two_verdicts_for_one_flow(lake, space, tmp_path):
    duck, partition, sources, fixture = lake
    table, path = emitted(duck, partition, sources, space, fixture)
    text = path.read_text().splitlines()
    path.write_text("\n".join(text + [text[1]]))       # duplicate a data row

    with pytest.raises(AuditError, match="more than once"):
        table.read(duck, path)


def test_ingest_refuses_a_class_outside_the_label_space(lake, space):
    duck, partition, sources, fixture = lake
    table, path = emitted(duck, partition, sources, space, fixture)
    rows = path.read_text().splitlines()
    header = rows[0].split(",")
    verdict = header.index("analyst_class")
    fields = rows[1].split(",")
    fields[verdict] = "portscan"                        # excluded from core-5
    rows[1] = ",".join(fields)
    path.write_text("\n".join(rows))

    with pytest.raises(AuditError, match="outside label space"):
        table.read(duck, path)


def adjudicate(path, verdicts: dict[str, str]):
    """Fill in analyst_class for the named uids, as a person would."""
    rows = path.read_text().splitlines()
    header = rows[0].split(",")
    uid, verdict = header.index("uid"), header.index("analyst_class")
    out = [rows[0]]
    for line in rows[1:]:
        fields = line.split(",")
        if fields[uid] in verdicts:
            fields[verdict] = verdicts[fields[uid]]
        out.append(",".join(fields))
    path.write_text("\n".join(out))


def test_state_distinguishes_confirmed_corrected_and_unadjudicated(lake, space):
    duck, partition, sources, fixture = lake
    table, path = emitted(duck, partition, sources, space, fixture)
    rel = duck.relation(f"SELECT * FROM read_csv_auto('{path}', header = true, "
                        f"all_varchar = true)")
    uids = [row[0] for row in duck.sql(f"SELECT uid, prior_class FROM "
                                       f"({rel.sql_query()}) a ORDER BY uid")]
    priors = dict(duck.sql(f"SELECT uid, prior_class FROM ({rel.sql_query()}) a"))

    adjudicate(path, {uids[0]: priors[uids[0]],           # agrees -> confirmed
                      uids[1]: "benign" if priors[uids[1]] != "benign" else "ddos"})
    summary = table.summarise(duck, table.read(duck, path), path)

    assert summary.confirmed == 1 and summary.corrected == 1
    assert summary.outstanding == summary.rows - 2
    assert len(summary.sha) == 12


# ------------------------------------------------------------- the benchmark

def two_methods(duck, tmp_path, truth: dict[str, str], wrong: set[str]):
    """Two labelled tables: one matching `truth`, one wrong on `wrong`."""
    paths = {}
    for name, flip in (("good", set()), ("bad", wrong)):
        rows = ", ".join(
            f"('{uid}', '{'benign' if uid in flip and label != 'benign' else label}')"
            for uid, label in truth.items())
        path = tmp_path / f"{name}.parquet"
        duck.relation(f"SELECT * FROM (VALUES {rows}) AS t(uid, label_class)"
                      ).write_parquet(str(path))
        paths[name] = str(path)
    return paths


def test_benchmark_scores_per_class_against_the_verdict(lake, space):
    duck, partition, sources, fixture = lake
    table, path = emitted(duck, partition, sources, space, fixture)
    rel = duck.relation(f"SELECT * FROM read_csv_auto('{path}', header = true, "
                        f"all_varchar = true)")
    priors = dict(duck.sql(f"SELECT uid, prior_class FROM ({rel.sql_query()}) a"))
    adjudicate(path, priors)                            # every verdict confirms

    adjudicated = table.read(duck, path)
    wrong = {uid for uid, label in priors.items() if label == "ddos"}
    report = LabelBenchmark(space).run(
        duck, adjudicated, two_methods(duck, fixture, priors, wrong), name="audit.csv")

    scores = {m.method: m for m in report.methods}
    good = {c.label: c for c in scores["good"].per_class}
    bad = {c.label: c for c in scores["bad"].per_class}

    assert good["ddos"].recall == 1.0
    assert bad["ddos"].recall == 0.0                     # every ddos called benign
    assert bad["ddos"].confused_with[0][0] == "benign"
    assert good["benign"].recall == 1.0                  # untouched by the flip


def test_unknown_verdicts_are_excluded_from_scoring(lake, space):
    """A flow a person could not classify is not evidence a method got it wrong."""
    duck, partition, sources, fixture = lake
    table, path = emitted(duck, partition, sources, space, fixture)
    rel = duck.relation(f"SELECT * FROM read_csv_auto('{path}', header = true, "
                        f"all_varchar = true)")
    priors = dict(duck.sql(f"SELECT uid, prior_class FROM ({rel.sql_query()}) a"))
    verdicts = dict(priors)
    undecided = sorted(verdicts)[0]
    verdicts[undecided] = UNKNOWN
    adjudicate(path, verdicts)

    report = LabelBenchmark(space).run(
        duck, table.read(duck, path), two_methods(duck, fixture, priors, set()))
    assert report.adjudicated == len(priors) - 1


def test_no_pooled_accuracy_is_published_anywhere(lake, space):
    """Per class and never pooled — the slice has a floor, so the mix is not the
    population's and one figure would weight the classes by the wrong priors."""
    duck, partition, sources, fixture = lake
    table, path = emitted(duck, partition, sources, space, fixture)
    rel = duck.relation(f"SELECT * FROM read_csv_auto('{path}', header = true, "
                        f"all_varchar = true)")
    priors = dict(duck.sql(f"SELECT uid, prior_class FROM ({rel.sql_query()}) a"))
    adjudicate(path, priors)

    report = LabelBenchmark(space).run(
        duck, table.read(duck, path), two_methods(duck, fixture, priors, set()))
    payload = report.to_dict()

    for method in payload["methods"]:
        assert "accuracy" not in method and "f1" not in method
        assert "recall" not in method                    # only per class
    assert "per-class floor" in payload["note"]


def test_agreement_is_reported_separately_and_named_as_agreement(lake, space):
    """The post-mortem's central failure was reporting one of these as the other."""
    duck, partition, sources, fixture = lake
    table, path = emitted(duck, partition, sources, space, fixture)
    rel = duck.relation(f"SELECT * FROM read_csv_auto('{path}', header = true, "
                        f"all_varchar = true)")
    priors = dict(duck.sql(f"SELECT uid, prior_class FROM ({rel.sql_query()}) a"))
    adjudicate(path, priors)

    wrong = {uid for uid, label in priors.items() if label == "dos"}
    report = LabelBenchmark(space).run(
        duck, table.read(duck, path), two_methods(duck, fixture, priors, wrong))

    assert report.agreement
    a, b, agree, total = report.agreement[0]
    assert (a, b) == ("bad", "good") and agree < total
    assert "NOT accuracy" in report.describe()


def test_a_file_with_no_verdicts_cannot_score_anything(lake, space):
    duck, partition, sources, fixture = lake
    table, path = emitted(duck, partition, sources, space, fixture)
    with pytest.raises(ValueError, match="nothing here that could score"):
        LabelBenchmark(space).run(duck, table.read(duck, path), {})


# ------------------------------------------------------------------ the oracle

def alerts_on(uids, rows):
    """A Suricata-shaped alert table hitting exactly those flows."""
    times = {uid: ts for uid, ts, *_ in rows}
    values = ", ".join(
        f"(CAST({times[uid]} AS DOUBLE), '10.0.0.1', '10.0.0.2', 'ET SCAN {uid}')"
        for uid in uids)
    return f"SELECT * FROM (VALUES {values}) AS a(ts, src_ip, dest_ip, signature)"


def test_the_adjudicator_gets_independent_evidence(lake, space):
    """Gate 14's premise: a person sees what a detector that never read the
    manifest thought, next to what the machine guessed."""
    from talos.data.labelling.oracle import Corroborator

    duck, partition, sources, fixture = lake
    chosen, _ = CandidateSelector(partition, floor=5).select(duck, sources)
    # ORDER BY: a bare scan is unordered and DuckDB may return rows in a
    # different order under parallelism, which would make this test flap.
    picked = [row[0] for row in duck.sql(
        f"SELECT uid FROM ({chosen.sql_query()}) c ORDER BY uid")]
    flagged = set(picked[:3])

    joined = Corroborator().join_sql(sources[0],
                                     alerts_on(flagged, labelled_rows()))
    duck.sql(f"CREATE OR REPLACE TEMP TABLE oracle_hits AS {joined}")
    path = AdjudicationTable(space).emit(
        duck, chosen, fixture / "with_oracle.csv",
        alerts="SELECT uid, alerted, signature FROM oracle_hits")

    rel = duck.relation(f"SELECT * FROM read_csv_auto('{path}', header = true)")
    verdicts = dict(duck.sql(
        f"SELECT uid, oracle_alerted FROM ({rel.sql_query()}) a"))
    signatures = dict(duck.sql(
        f"SELECT uid, oracle_signature FROM ({rel.sql_query()}) a"))

    assert all(verdicts[uid] for uid in flagged)
    assert all(not verdicts[uid] for uid in picked[3:])
    assert all(signatures[uid] for uid in flagged)


def test_without_an_oracle_the_columns_are_null_not_false(lake, space):
    """"Suricata said nothing" and "nobody asked Suricata" are different facts."""
    duck, partition, sources, fixture = lake
    _, path = emitted(duck, partition, sources, space, fixture)
    rel = duck.relation(f"SELECT * FROM read_csv_auto('{path}', header = true)")

    assert duck.one(f"SELECT count(*) FROM ({rel.sql_query()}) a "
                    f"WHERE oracle_alerted IS NOT NULL")[0] == 0


# --------------------------------------------------------------------- the page

def payload_of(path):
    """The JSON the page embeds, which is what the adjudicator actually sees."""
    import html as _html, json, re
    raw = re.search(r'type="application/json">(.*?)</script>',
                    path.read_text(), re.S).group(1)
    return json.loads(_html.unescape(raw))


def test_the_page_shows_the_flow_not_just_the_verdict(lake, space):
    """Nobody can adjudicate a uid. The features have to be on the page."""
    from talos.data.labelling.audit import AuditPage

    duck, partition, sources, fixture = lake
    _, csv = emitted(duck, partition, sources, space, fixture)
    page = AuditPage(space, "toy").render(duck, csv, fixture / "toy.html")
    data = payload_of(page)

    # Against the file, not literals: both the count and the columns follow
    # from whatever the extractor put in the labelled table.
    assert len(data["rows"]) == len(csv.read_text().splitlines()) - 1
    for column in ("duration", "id.orig_h", "id.resp_h"):
        assert column in data["flow"]
    assert data["rows"][0]["duration"] is not None

    # And what it shows is the FLOW, not the labels: no label column leaks in.
    from talos.data.labelling.behavioural.base import labelling_columns
    assert not (set(data["flow"]) & labelling_columns())


def test_the_page_offers_the_label_space_plus_unknown(lake, space):
    from talos.data.labelling.audit import AuditPage

    duck, partition, sources, fixture = lake
    _, csv = emitted(duck, partition, sources, space, fixture)
    data = payload_of(AuditPage(space, "toy").render(duck, csv, fixture / "p.html"))

    assert data["classes"] == list(space.classes) + [UNKNOWN]
    assert "portscan" not in data["classes"]        # excluded from core-5


def test_the_page_is_self_contained_and_sends_nothing(lake, space):
    """It opens from file:// on a laptop or after scp off a headless box."""
    import re
    from talos.data.labelling.audit import AuditPage

    duck, partition, sources, fixture = lake
    _, csv = emitted(duck, partition, sources, space, fixture)
    text = AuditPage(space, "toy").render(duck, csv, fixture / "p.html").read_text()

    assert not re.search(r'(src|href)="https?:|@import|fetch\(|XMLHttpRequest', text)


def test_the_page_starts_with_no_verdicts_recorded(lake, space):
    from talos.data.labelling.audit import AuditPage

    duck, partition, sources, fixture = lake
    _, csv = emitted(duck, partition, sources, space, fixture)
    data = payload_of(AuditPage(space, "toy").render(duck, csv, fixture / "p.html"))

    assert all(not row["analyst_class"] for row in data["rows"])
