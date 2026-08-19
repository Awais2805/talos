"""Gates 0.3/0.4 — row verification and the history alphabet, on planted violations.

Each invariant gets a row that breaks it and rows that do not, so a check that
silently stopped running would show as zero rather than as the planted count.
The subtle one is `orig_rate_below_line_rate`: its violation is a finite number,
so no NaN or Inf test catches it.
"""

from __future__ import annotations

import duckdb
import pytest

from talos.data.preprocess.prelabel.verify import (
    INVARIANTS, LINE_RATE, RowVerifier, history_alphabet,
)


class Duck:
    def __init__(self, con):
        self.con = con

    def sql(self, query):
        return self.con.execute(query).fetchall()

    def one(self, query):
        return self.con.execute(query).fetchone()

    def columns(self, source):
        return {row[0]: row[1] for row in
                self.con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}


# One planted violation each, plus three rows that are entirely fine.
FLOWS = """
    CREATE TABLE flows AS SELECT * FROM (VALUES
      ('ok1',  100.0,  200.0, 140.0, 240.0, 5.0, 5.0, 1.0,  'ShADadFf'),
      ('ok2',    0.0,    0.0,   0.0,   0.0, 0.0, 0.0, 0.5,  'Sr'),
      ('ok3',   50.0,   60.0,  90.0, 100.0, 2.0, 3.0, 2.0,  'ShAFf'),
      ('negb', -10.0,  200.0, 140.0, 240.0, 5.0, 5.0, 1.0,  'ShADadFf'),
      ('negd', 100.0,  200.0, 140.0, 240.0, 5.0, 5.0, -1.0, 'ShADadFf'),
      ('hdr',  100.0,  200.0,  50.0, 240.0, 5.0, 5.0, 1.0,  'ShADadFf'),
      ('nopk', 100.0,  200.0, 140.0, 240.0, 0.0, 5.0, 1.0,  'ShADadFf'),
      ('fast', 1.0e9,  200.0, 1.0e9, 240.0, 5.0, 5.0, 1.0e-9, 'ShADadFf'),
      ('null', NULL,   NULL,  NULL,  NULL,  NULL, NULL, NULL, NULL)
    ) AS t(uid, orig_bytes, resp_bytes, orig_ip_bytes, resp_ip_bytes,
           orig_pkts, resp_pkts, duration, history)
"""


@pytest.fixture
def duck():
    con = duckdb.connect()
    con.execute(FLOWS)
    return Duck(con)


def audit(duck):
    return RowVerifier(duck).audit("toy", "flows", duck.columns("flows"))


def rows_for(report, name):
    return next(v for v in report.violations if v.name == name).rows


# ---------------------------------------------------------- the invariants

def test_a_negative_byte_count_is_flagged(duck):
    """A byte count cannot be negative."""
    assert rows_for(audit(duck), "non_negative_orig_bytes") == 1


def test_a_negative_duration_is_flagged(duck):
    """A connection cannot end before it starts."""
    assert rows_for(audit(duck), "non_negative_duration") == 1


def test_ip_bytes_below_payload_is_flagged(duck):
    """Headers cannot be negative, and log1p would clamp that away silently."""
    assert rows_for(audit(duck), "orig_ip_bytes_cover_payload") == 1


def test_payload_with_zero_packets_is_flagged(duck):
    """Payload cannot arrive in zero packets."""
    assert rows_for(audit(duck), "orig_bytes_need_packets") == 1


def test_an_impossible_rate_is_flagged_though_it_is_finite(duck):
    """The reason this check exists: no NaN or Inf test would catch it.

    1e9 bytes over 1e-9 seconds is 1e18 B/s -- a perfectly finite double that
    would sail through every non-finite check and then set the scale for the
    whole column.
    """
    report = audit(duck)
    assert rows_for(report, "orig_rate_below_line_rate") == 1
    assert not any(report.non_finite.values())
    assert 1.0e9 / 1.0e-9 > LINE_RATE


def test_nulls_are_never_violations(duck):
    """Absence is the missingness screen's business, not this one's.

    Conflating them is how `orig_bytes IS NULL` and `orig_bytes = 0` stopped
    being distinguishable once before.
    """
    report = audit(duck)
    # The all-NULL row breaks nothing, so no count may reach the full 9 rows.
    for violation in report.violations:
        assert violation.rows <= 1


def test_a_clean_row_breaks_nothing(duck):
    """Three rows are valid by construction; the counts prove nothing else fires."""
    report = audit(duck)
    assert report.total_violating == 5


# ------------------------------------------------------- absent columns

def test_a_check_with_no_column_is_reported_as_skipped_not_passed(duck):
    """A silently un-run check reads exactly like a passing one."""
    duck.con.execute("CREATE TABLE thin AS SELECT 'u' AS uid, 1.0 AS duration")
    report = RowVerifier(duck).audit("toy", "thin", duck.columns("thin"))
    skipped = [v for v in report.violations if not v.checked]
    assert skipped
    for violation in skipped:
        assert violation.skipped_because.startswith("no column")
    checked = [v for v in report.violations if v.checked]
    assert [v.name for v in checked] == ["non_negative_duration"]


def test_every_invariant_states_a_reason(duck):
    """An invariant with no reason is an assertion nobody can argue with."""
    for invariant in INVARIANTS:
        assert invariant.reason and invariant.columns


# ----------------------------------------------------- non-finite values

def test_nan_and_infinity_are_counted(duck):
    """`coalesce` catches NULL but not Inf, which propagates through log1p."""
    duck.con.execute("""CREATE TABLE odd AS SELECT * FROM (VALUES
        (CAST('NaN' AS DOUBLE)), (CAST('Infinity' AS DOUBLE)), (1.0)
    ) AS t(orig_bytes)""")
    report = RowVerifier(duck).audit("toy", "odd", duck.columns("odd"))
    assert report.non_finite["orig_bytes"] == 2


# -------------------------------------------------------- the alphabet

def test_the_history_alphabet_is_measured_from_the_data(duck):
    """Declared from the manual, an unemitted letter becomes a dead column."""
    alphabet = history_alphabet(duck, "flows")
    assert set(alphabet) == set("ShADadFfr")
    assert alphabet["S"] == 8       # every non-null history starts with S


def test_the_alphabet_counts_letters_not_rows(duck):
    """`d` appears once per row that has it; `F` and `f` are separate letters."""
    alphabet = history_alphabet(duck, "flows")
    assert alphabet["F"] == 7 and alphabet["f"] == 7
    assert alphabet["r"] == 1


def test_a_checksum_letter_would_be_surfaced(duck):
    """Zeek is documented as running -C, so a 'C' contradicts the pipeline."""
    duck.con.execute("""CREATE TABLE bad AS SELECT * FROM (VALUES
        ('ShACadFf'), ('Sr')) AS t(history)""")
    report = RowVerifier(duck).audit("toy", "bad", duck.columns("bad"))
    report.alphabet = history_alphabet(duck, "bad")
    assert "C" in report.alphabet
    assert "-C" in report.summary()


# ----------------------------------------------------------- the report

def test_nothing_is_dropped(duck):
    """Row selection is experiment-scoped; this stage only ever flags."""
    report = audit(duck)
    assert report.rows == 9
    assert "never dropped" in report.to_dict()["policy"]


# ------------------------------------------ the reusable predicate (Gate 2/3)

COLS = {"orig_bytes": "BIGINT", "resp_bytes": "BIGINT", "duration": "DOUBLE",
        "orig_ip_bytes": "BIGINT", "resp_ip_bytes": "BIGINT",
        "orig_pkts": "BIGINT", "resp_pkts": "BIGINT"}


def _kept(rows_sql: str) -> list:
    from talos.data.preprocess.prelabel.verify import valid_sql
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t AS {rows_sql}")
    columns = {r[0]: r[1] for r in con.execute("DESCRIBE t").fetchall()}
    return con.execute(
        f"SELECT tag FROM t WHERE {valid_sql(columns)} ORDER BY tag").fetchall()


def test_a_null_is_never_a_violation():
    """Absence is the missingness screen's business, not validity's."""
    assert _kept("""SELECT 'null' AS tag, CAST(NULL AS BIGINT) AS orig_bytes,
                    CAST(1 AS BIGINT) AS resp_bytes, CAST(1.0 AS DOUBLE) AS duration,
                    CAST(9 AS BIGINT) AS orig_ip_bytes, CAST(9 AS BIGINT) AS resp_ip_bytes,
                    CAST(1 AS BIGINT) AS orig_pkts, CAST(1 AS BIGINT) AS resp_pkts
                 """) == [("null",)]


def test_a_negative_byte_count_is_rejected():
    assert _kept("""SELECT 'bad' AS tag, CAST(-1 AS BIGINT) AS orig_bytes,
                    CAST(1 AS BIGINT) AS resp_bytes, CAST(1.0 AS DOUBLE) AS duration,
                    CAST(9 AS BIGINT) AS orig_ip_bytes, CAST(9 AS BIGINT) AS resp_ip_bytes,
                    CAST(1 AS BIGINT) AS orig_pkts, CAST(1 AS BIGINT) AS resp_pkts
                 """) == []


def test_a_faster_than_line_rate_flow_is_rejected():
    assert _kept("""SELECT 'fast' AS tag, CAST(1000000 AS BIGINT) AS orig_bytes,
                    CAST(1 AS BIGINT) AS resp_bytes, CAST(1e-9 AS DOUBLE) AS duration,
                    CAST(1000040 AS BIGINT) AS orig_ip_bytes, CAST(9 AS BIGINT) AS resp_ip_bytes,
                    CAST(1 AS BIGINT) AS orig_pkts, CAST(1 AS BIGINT) AS resp_pkts
                 """) == []


def test_an_uncheckable_table_keeps_every_row():
    """A skipped check must not reject everything; that is the same silence inverted."""
    from talos.data.preprocess.prelabel.verify import valid_sql
    assert valid_sql({"uid": "VARCHAR"}) == "TRUE"


def test_the_predicate_can_be_aliased_for_a_joined_relation():
    from talos.data.preprocess.prelabel.verify import valid_sql
    sql = valid_sql(COLS, alias="p")
    assert 'p."orig_bytes"' in sql and '"orig_bytes" >= 0' not in sql.replace('p."orig_bytes"', "")


def test_every_declared_invariant_reaches_the_predicate():
    """A new invariant that the filter silently ignores is the defect this catches."""
    from talos.data.preprocess.prelabel.verify import INVARIANTS, valid_sql
    sql = valid_sql(COLS)
    assert sql.count("coalesce(") == len(INVARIANTS)
