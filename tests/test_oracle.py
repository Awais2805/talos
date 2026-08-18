"""The Suricata oracle: independent evidence, and the epistemics around it.

Two things are being tested. That alerts parse and join correctly — mechanical.
And that the module refuses to overclaim — which is the part that matters, since
the deleted pipeline's central failure was reporting agreement as if it were
accuracy.

Suricata itself is never invoked here. Running a binary is I/O; the logic is
parsing, joining and scoring, and those are exercised against a fixture eve.json.
"""

import json
import textwrap

import duckdb
import pytest

from talos.data.labelling.oracle import (
    Corroborator, OffsetProbe, SuricataError, SuricataOracle,
)
from talos.data.labelling.schedule.windows import Window, to_utc

ORACLE = SuricataOracle()


def eve_line(ts, src, dst, sig="ET SCAN Potential SSH Scan", event="alert"):
    return json.dumps({
        "timestamp": ts, "event_type": event,
        "src_ip": src, "src_port": 4444, "dest_ip": dst, "dest_port": 22,
        "proto": "TCP",
        "alert": {"signature": sig, "category": "Attempted Recon",
                  "signature_id": 2001219, "severity": 2},
    })


def window(rid=0, t0=0.0, t1=600.0, attackers=("10.0.0.1",), victims=("10.0.0.2",)):
    return Window(rid=rid, name="FTP-Patator", t0=t0, t1=t1, t1_raw=t1,
                  attackers=attackers, victims=victims, subnet_prefix=None,
                  date="2017-07-04", start="09:20", end="10:20")


# ------------------------------------------------------------------ parsing

def test_utc_timestamps_parse_to_epoch():
    from datetime import datetime, timezone
    expected = datetime(2017, 7, 4, 12, 20, 15, tzinfo=timezone.utc).timestamp()
    assert ORACLE.parse_timestamp("2017-07-04T12:20:15+0000") == expected
    assert ORACLE.parse_timestamp("2017-07-04T12:20:15.000000+00:00") == expected


def test_a_non_utc_offset_is_converted_not_refused():
    """`09:20:15-0300` and `12:20:15+0000` are the SAME INSTANT.

    Refusing the first would throw away valid data: the offset is in the string,
    so Suricata running on a non-UTC host is not a problem.
    """
    from datetime import datetime, timezone
    assert ORACLE.parse_timestamp("2017-07-04T09:20:15.000000-0300") == datetime(
        2017, 7, 4, 12, 20, 15, tzinfo=timezone.utc).timestamp()


def test_a_timestamp_with_no_offset_is_refused():
    """It names no instant, and would silently adopt the reader's timezone."""
    with pytest.raises(SuricataError, match="no UTC offset"):
        ORACLE.parse_timestamp("2017-07-04T09:20:15")


def test_alerts_stream_through_duckdb_rather_than_python(tmp_path):
    """2019 produces ~5M alerts. Holding those as objects, and inlining them as
    literal SQL, was a 1.6 GB read and a ~300 MB query string."""
    from talos.common.duck import DuckEngine
    eve = tmp_path / "eve.json"
    eve.write_text("\n".join(
        eve_line(f"2017-07-04T12:20:{i:02d}+0000", "10.0.0.1", "10.0.0.2")
        for i in range(30)))

    rel = ORACLE.alerts_relation(DuckEngine(), eve)
    assert rel.columns == ["ts", "src_ip", "dest_ip", "signature"]
    assert rel.aggregate("count(*)").fetchone()[0] == 30


def test_the_relation_preserves_the_offset_rather_than_dropping_it(tmp_path):
    """The bug this pins: DuckDB's JSON sniffer types eve `timestamp` as naive
    TIMESTAMP and discards the offset, so `12:20:15+0000` and `09:20:15-0300`
    -- the same instant -- both arrive as 12:20:15 and the second is three hours
    wrong. Nothing downstream can see that, and `OffsetProbe` (whose entire job
    is settling the unevidenced 2017 offset) would have reported it confidently.
    """
    from talos.common.duck import DuckEngine
    eve = tmp_path / "eve.json"
    eve.write_text("\n".join([
        eve_line("2017-07-04T12:20:15.000000+0000", "10.0.0.1", "10.0.0.2"),
        eve_line("2017-07-04T09:20:15.000000-0300", "10.0.0.1", "10.0.0.2"),
    ]))

    rel = ORACLE.alerts_relation(DuckEngine(), eve)
    epochs = [row[0] for row in rel.project("ts").fetchall()]
    assert epochs[0] == epochs[1] == ORACLE.parse_timestamp(
        "2017-07-04T12:20:15+0000")


def test_a_relation_over_offsetless_timestamps_is_refused(tmp_path):
    """Refused before the relation is handed over, not after."""
    from talos.common.duck import DuckEngine
    eve = tmp_path / "eve.json"
    eve.write_text(eve_line("2017-07-04T12:20:15.000000", "10.0.0.1", "10.0.0.2"))

    with pytest.raises(SuricataError, match="no UTC offset"):
        ORACLE.alerts_relation(DuckEngine(), eve)


def test_only_alert_records_are_taken(tmp_path):
    eve = tmp_path / "eve.json"
    eve.write_text("\n".join([
        eve_line("2017-07-04T12:20:15+0000", "10.0.0.1", "10.0.0.2"),
        eve_line("2017-07-04T12:20:16+0000", "10.0.0.1", "10.0.0.2", event="flow"),
        json.dumps({"event_type": "stats", "timestamp": "2017-07-04T12:20:17+0000"}),
    ]))
    assert len(ORACLE.parse_alerts(eve)) == 1


def test_a_torn_final_line_is_skipped_not_fatal(tmp_path):
    """eve.json is append-only; a run cut short leaves a partial record."""
    eve = tmp_path / "eve.json"
    eve.write_text(eve_line("2017-07-04T12:20:15+0000", "10.0.0.1", "10.0.0.2")
                   + '\n{"event_type": "alert", "timestam')
    assert len(ORACLE.parse_alerts(eve)) == 1


def test_the_docker_invocation_forces_utc():
    from pathlib import Path
    cmd = ORACLE.command(Path("/pcaps/monday.pcap"), Path("/out"), "docker")
    assert "TZ=UTC" in cmd


def test_availability_is_reported_not_raised():
    ok, how = ORACLE.available()
    assert isinstance(ok, bool) and how          # a caller can skip rather than crash


# --------------------------------------------------------------- the report

@pytest.fixture
def corroborated(tmp_path):
    """A labelled table plus alerts, joined."""
    def build(flows, alerts):
        uri = str(tmp_path / "conn.parquet")
        rows = ", ".join(
            f"('{u}', {ts!r}, {dur!r}, '{o}', '{r}', {str(lb).lower()}, '{lc}')"
            for u, ts, dur, o, r, lb, lc in flows)
        duckdb.connect().sql(
            f"""SELECT * FROM (VALUES {rows}) AS t(uid, ts, duration, "id.orig_h",
                "id.resp_h", label_binary, label_class)""").write_parquet(uri)

        eve = tmp_path / "eve.json"
        eve.write_text("\n".join(eve_line(ts, s, d) for ts, s, d in alerts))
        parsed = ORACLE.parse_alerts(eve)

        con = duckdb.connect()
        return Corroborator().build(_Duck(con), "toy", uri,
                                    ORACLE.alerts_table(parsed), len(parsed))
    return build


class _Duck:
    """Minimal DuckEngine surface — the Corroborator only needs sql/one."""
    def __init__(self, con):
        self.con = con

    def sql(self, q):
        return self.con.execute(q).fetchall()

    def one(self, q):
        return self.con.execute(q).fetchone()


def test_corroboration_and_dissent_are_counted_separately(corroborated):
    report = corroborated(
        flows=[("a", 100.0, 5.0, "10.0.0.1", "10.0.0.2", True, "brute_force"),
               ("b", 200.0, 5.0, "10.0.0.1", "10.0.0.2", True, "brute_force"),
               ("c", 300.0, 5.0, "10.9.9.9", "10.0.0.2", False, "benign"),
               ("d", 400.0, 5.0, "10.8.8.8", "10.0.0.2", False, "benign")],
        alerts=[("1970-01-01T00:01:41+0000", "10.0.0.1", "10.0.0.2"),   # ts 101 -> a
                ("1970-01-01T00:05:01+0000", "10.9.9.9", "10.0.0.2")])  # ts 301 -> c

    assert report.attack_flows == 2 and report.attack_corroborated == 1
    assert report.benign_flows == 2 and report.benign_alerted == 1
    assert report.corroboration_rate == 0.5
    assert report.dissent_rate == 0.5


def test_the_report_exposes_no_combined_accuracy(corroborated):
    """The deleted pipeline measured agreement and called it accuracy.

    Absence of an alert is not evidence of benign, so no single number is
    meaningful -- and none is offered, deliberately.
    """
    report = corroborated(
        flows=[("a", 100.0, 5.0, "10.0.0.1", "10.0.0.2", True, "brute_force")],
        alerts=[("1970-01-01T00:01:41+0000", "10.0.0.1", "10.0.0.2")])

    for banned in ("accuracy", "agreement", "score", "f1", "precision"):
        assert not any(banned in name.lower() for name in dir(report))
    assert "not evidence of benign" in report.to_dict()["note"]


def test_alerts_that_join_to_nothing_are_surfaced(corroborated):
    """A large unjoined count means the clock or the 5-tuple does not line up,
    and neither rate can be trusted until that is understood."""
    report = corroborated(
        flows=[("a", 100.0, 5.0, "10.0.0.1", "10.0.0.2", True, "brute_force")],
        alerts=[("1970-01-01T00:01:41+0000", "10.0.0.1", "10.0.0.2"),
                ("1970-01-01T10:00:00+0000", "1.2.3.4", "5.6.7.8")])
    assert report.alerts_total == 2
    assert report.unjoined_alerts >= 1


def test_dissent_examples_are_listed_for_follow_up(corroborated):
    report = corroborated(
        flows=[("c", 300.0, 5.0, "10.9.9.9", "10.0.0.2", False, "benign")],
        alerts=[("1970-01-01T00:05:01+0000", "10.9.9.9", "10.0.0.2")])
    assert report.dissent_examples[0]["uid"] == "c"


# ------------------------------------------------------------- offset probe

class FakeAlert:
    def __init__(self, ts, src="10.0.0.1", dst="10.0.0.2"):
        self.ts, self.src_ip, self.dest_ip = ts, src, dst


def test_the_probe_recovers_a_deliberately_wrong_offset():
    """The schedule is declared -03:00; the capture is really -04:00.

    Alerts are placed an hour after the declared windows, which is exactly what
    a one-hour offset error looks like -- and is invisible in the labelled
    output, because every window moves together.
    """
    w = window(t0=to_utc("2017-07-04", "09:20", "-03:00"),
               t1=to_utc("2017-07-04", "10:20", "-03:00"))
    true_windows = [FakeAlert(w.t0 + 3600 + i * 10) for i in range(40)]

    verdict = OffsetProbe().probe(true_windows, [w], declared="-03:00")

    assert verdict.conclusive
    assert verdict.agrees is False
    assert verdict.best.offset == "-04:00"
    assert "every window belongs 1h later" in verdict.summary()


def test_the_probe_confirms_a_correct_offset():
    w = window(t0=to_utc("2017-07-04", "09:20", "-03:00"),
               t1=to_utc("2017-07-04", "10:20", "-03:00"))
    alerts = [FakeAlert(w.t0 + i * 10) for i in range(40)]

    verdict = OffsetProbe().probe(alerts, [w], declared="-03:00")
    assert verdict.conclusive and verdict.agrees is True
    assert "empirically supported" in verdict.summary()


def test_too_few_alerts_is_inconclusive_not_a_guess():
    """An unevidenced answer is worth more than a confident wrong one."""
    w = window(t0=1000.0, t1=1600.0)
    verdict = OffsetProbe().probe([FakeAlert(1100.0)], [w], declared="-03:00")

    assert not verdict.conclusive
    assert verdict.agrees is None
    assert "INCONCLUSIVE" in verdict.summary()


def test_alerts_between_unrelated_hosts_are_ignored():
    """Otherwise background noise votes on the timezone."""
    w = window(t0=1000.0, t1=1600.0)
    noise = [FakeAlert(1100.0 + i, src="8.8.8.8", dst="9.9.9.9") for i in range(40)]
    assert OffsetProbe().score(noise, [w], shift=0.0) == 0


def test_shift_direction_matches_the_epoch_arithmetic():
    """window = naive_local - offset, so a later offset moves windows later."""
    assert OffsetProbe.shift_for("-03:00", "-04:00") == 3600.0
    assert OffsetProbe.shift_for("-03:00", "-02:00") == -3600.0
    assert OffsetProbe.shift_for("-03:00", "-03:00") == 0.0
