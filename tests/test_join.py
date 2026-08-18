"""The matching semantics, exercised as real SQL against real parquet.

No mocks. Every case below builds a tiny conn parquet, runs the generated query
through DuckDB and asserts on which flows were claimed — because the thing that
can be wrong here is the SQL, and a test that stubs the SQL tests nothing.

Most of these are scars. Bidirectional matching, alias sets, victim-only
matching and the absent-vs-empty victim distinction were each a real
mislabelling before they were a rule, and the whole point of testing them
against synthetic flows is that it happens *before* a 2.1M-row run rather than
being inferred from a suspicious attack ratio afterwards.
"""

import duckdb
import pytest

from talos.data.labelling.schedule.join import IntervalJoinBuilder
from talos.data.labelling.schedule.windows import Window

BUILDER = IntervalJoinBuilder()


def rule(rid=0, name="Attack", t0=100.0, t1=200.0, attackers=None,
         victims=("10.0.0.2",), subnet=None) -> Window:
    return Window(rid=rid, name=name, t0=t0, t1=t1, t1_raw=t1,
                  attackers=attackers, victims=victims, subnet_prefix=subnet,
                  date="2017-07-04", start="09:20", end="10:20")


@pytest.fixture
def lake(tmp_path):
    """Writes conn flows to a capture-shaped tree and runs the join over them."""
    base = tmp_path / "parquet" / "ds"

    def run(flows, rules, capture="Tuesday-WorkingHours"):
        con = duckdb.connect()
        rows = ", ".join(
            f"('{uid}', {ts!r}, '{orig}', '{resp}', {ob})"
            for uid, ts, orig, resp, ob in flows)
        out = base / capture
        out.mkdir(parents=True, exist_ok=True)
        con.sql(f"""SELECT * FROM (VALUES {rows})
                    AS t(uid, ts, "id.orig_h", "id.resp_h", orig_bytes)"""
                ).write_parquet(str(out / "conn.parquet"))

        sql = BUILDER.base_ctes(rules, f"{base}/**/conn.parquet", f"{base}/")
        return con.sql(sql + """
            SELECT m.uid, m.rid, m.rname, m.nrules, c.capture
            FROM matched m JOIN conn c USING (uid) ORDER BY m.uid""").fetchall()

    return run


def claimed(rows):
    return {r[0] for r in rows}


# ------------------------------------------------------------- bidirectional

def test_a_flow_matches_in_either_direction(lake):
    """Response traffic is part of the attack. The datasets' own labelling is
    direction-dependent, which published audits blame for mislabelled flows."""
    rows = lake(
        [("fwd", 150.0, "10.0.0.1", "10.0.0.2", 100),
         ("rev", 150.0, "10.0.0.2", "10.0.0.1", 100)],
        [rule(attackers=("10.0.0.1",), victims=("10.0.0.2",))])
    assert claimed(rows) == {"fwd", "rev"}


def test_a_flow_between_two_bystanders_inside_the_window_is_not_claimed(lake):
    """The whole reason this is not a timestamp filter: victims keep serving
    legitimate traffic throughout an attack."""
    rows = lake(
        [("attack", 150.0, "10.0.0.1", "10.0.0.2", 100),
         ("bystander", 150.0, "10.9.9.9", "10.0.0.2", 100),
         ("unrelated", 150.0, "10.8.8.8", "10.7.7.7", 100)],
        [rule(attackers=("10.0.0.1",), victims=("10.0.0.2",))])
    assert claimed(rows) == {"attack"}


# -------------------------------------------------------------- alias sets

def test_any_listed_attacker_address_matches(lake):
    """2017's attacker sits behind NAT and appears internally as 172.16.0.1.
    Without the alias one rule matched zero flows while 94,614 sat in it."""
    rows = lake(
        [("public", 150.0, "205.174.165.73", "192.168.10.50", 100),
         ("post_nat", 150.0, "172.16.0.1", "192.168.10.50", 100)],
        [rule(attackers=("205.174.165.73", "172.16.0.1"),
              victims=("192.168.10.50", "205.174.165.68"))])
    assert claimed(rows) == {"public", "post_nat"}


# ------------------------------------------------------------- victim only

def test_victim_only_rules_match_on_the_victim_alone(lake):
    """Reflection attacks spoof their sources, so the attacker carries nothing."""
    rows = lake(
        [("spoof_a", 150.0, "1.2.3.4", "192.168.50.1", 0),
         ("spoof_b", 150.0, "5.6.7.8", "192.168.50.1", 0),
         ("reply", 150.0, "192.168.50.1", "9.9.9.9", 0),
         ("elsewhere", 150.0, "1.2.3.4", "10.0.0.9", 0)],
        [rule(attackers=None, victims=("192.168.50.1",))])
    assert claimed(rows) == {"spoof_a", "spoof_b", "reply"}


def test_a_rules_table_where_every_attacker_is_null_still_binds(lake):
    """2019 is reflection-only. An untyped all-NULL column binds as INTEGER and
    fails inside list_contains rather than anywhere informative."""
    rows = lake(
        [("f", 150.0, "1.2.3.4", "192.168.50.1", 0)],
        [rule(rid=0, attackers=None, victims=("192.168.50.1",)),
         rule(rid=1, t0=300.0, t1=400.0, attackers=None, victims=("192.168.50.4",))])
    assert claimed(rows) == {"f"}


# ---------------------------------------------------------- subnet victims

def test_a_subnet_victim_matches_by_prefix(lake):
    rows = lake(
        [("in_lan", 150.0, "192.168.10.8", "192.168.10.25", 0),
         ("out_of_lan", 150.0, "192.168.10.8", "10.0.0.5", 0)],
        [rule(attackers=("192.168.10.8",), subnet="192.168.10.")])
    assert claimed(rows) == {"in_lan"}


def test_a_subnet_rule_does_not_match_every_peer(lake):
    """THE bug. A subnet rule carries no victim_ips; reading that absence alone
    as 'no victim constraint' made one rule match every flow in its window."""
    rows = lake(
        [("in_lan", 150.0, "192.168.10.8", "192.168.10.25", 0),
         ("to_internet", 150.0, "192.168.10.8", "8.8.8.8", 0)],
        [rule(attackers=("192.168.10.8",), victims=None, subnet="192.168.10.")])
    assert claimed(rows) == {"in_lan"}
    assert "to_internet" not in claimed(rows)


def test_a_rule_with_no_victim_constraint_at_all_matches_any_peer(lake):
    """Genuinely absent means any peer -- distinct from the case above."""
    rows = lake(
        [("a", 150.0, "10.0.0.1", "8.8.8.8", 0),
         ("b", 150.0, "10.0.0.1", "10.0.0.2", 0)],
        [rule(attackers=("10.0.0.1",), victims=None, subnet=None)])
    assert claimed(rows) == {"a", "b"}


# ------------------------------------------------------------ time bounds

def test_the_window_start_is_inclusive_and_the_end_exclusive(lake):
    """Windows are minute-extended and clamped at the next start, so a closed
    end would let two back-to-back attacks both claim the boundary instant."""
    rows = lake(
        [("before", 99.99, "10.0.0.1", "10.0.0.2", 0),
         ("at_start", 100.0, "10.0.0.1", "10.0.0.2", 0),
         ("inside", 150.0, "10.0.0.1", "10.0.0.2", 0),
         ("at_end", 200.0, "10.0.0.1", "10.0.0.2", 0),
         ("after", 200.01, "10.0.0.1", "10.0.0.2", 0)],
        [rule(t0=100.0, t1=200.0, attackers=("10.0.0.1",), victims=("10.0.0.2",))])
    assert claimed(rows) == {"at_start", "inside"}


def test_back_to_back_windows_tile_without_double_claiming(lake):
    """2019's WebDDoS/SYN pair. The shared instant belongs to the later window."""
    rows = lake(
        [("boundary", 200.0, "10.0.0.1", "10.0.0.2", 0)],
        [rule(rid=0, name="WebDDoS", t0=100.0, t1=200.0,
              attackers=("10.0.0.1",), victims=("10.0.0.2",)),
         rule(rid=1, name="SYN", t0=200.0, t1=300.0,
              attackers=("10.0.0.1",), victims=("10.0.0.2",))])
    assert rows == [("boundary", 1, "SYN", 1, "Tuesday-WorkingHours")]


# --------------------------------------------------------------- overlaps

def test_the_lowest_rule_id_wins_and_the_overlap_is_counted(lake):
    """Deterministic, and independent of scan order -- so a re-run cannot
    silently reassign a class."""
    rows = lake(
        [("both", 150.0, "10.0.0.1", "10.0.0.2", 0)],
        [rule(rid=0, name="First", attackers=("10.0.0.1",), victims=("10.0.0.2",)),
         rule(rid=1, name="Second", attackers=("10.0.0.1",), victims=("10.0.0.2",))])
    assert rows == [("both", 0, "First", 2, "Tuesday-WorkingHours")]


def test_each_flow_appears_once_however_many_rules_claim_it(lake):
    """The aggregate is per uid; a flow must never be duplicated into the output."""
    rows = lake(
        [("f", 150.0, "10.0.0.1", "10.0.0.2", 0)],
        [rule(rid=i, attackers=("10.0.0.1",), victims=("10.0.0.2",)) for i in range(4)])
    assert len(rows) == 1 and rows[0][3] == 4


# ---------------------------------------------------------------- capture

def test_the_capture_folder_is_recovered_from_the_parquet_path(lake):
    """`capture` is what makes leakage-safe splitting possible: attacks are
    contiguous bursts, so a random flow-level split reports a fictional score."""
    rows = lake([("f", 150.0, "10.0.0.1", "10.0.0.2", 0)],
                [rule(attackers=("10.0.0.1",), victims=("10.0.0.2",))],
                capture="Friday-WorkingHours-Afternoon")
    assert rows[0][4] == "Friday-WorkingHours-Afternoon"


# ------------------------------------------------------------ SQL hygiene

def test_rule_names_with_quotes_are_escaped():
    sql = BUILDER.rules_table([rule(name="O'Brien's scan")])
    assert "'O''Brien''s scan'" in sql


def test_nulls_in_the_rules_table_are_typed():
    sql = BUILDER.rules_table([rule(attackers=None, victims=("v",), subnet=None)])
    assert "CAST(NULL AS VARCHAR[])" in sql          # attacker list
    assert "CAST(NULL AS VARCHAR)" in sql            # subnet prefix


def test_an_empty_rules_table_is_refused():
    """It would match nothing and silently label the whole capture benign."""
    with pytest.raises(ValueError, match="benign"):
        BUILDER.rules_table([])


def test_quarantine_regions_use_the_identical_predicate():
    """A second matcher would drift from the first, invisibly."""
    rules_only = BUILDER.base_ctes([rule()], "s", "b")
    with_regions = BUILDER.base_ctes([rule()], "s", "b", regions=[rule(name="why")])

    assert "quarantined AS" not in rules_only        # byte-identical to a pre-quarantine build
    assert with_regions.count(BUILDER.match_predicate()) == 2
