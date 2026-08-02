#!/usr/bin/env python3
"""Build a synthetic lake with known labelling faults, for testing the validator.

A validation module that reports nothing is indistinguishable from a broken one.
The only way to know these checks work is to hand them data whose faults we
planted ourselves and confirm they find exactly those and nothing else.

The fixture is a real, complete pipeline run in miniature: synthetic Zeek conn
flows go through the actual labelling stage, against a real manifest, and the
validator reads the result through exactly the code path it uses on S3. Nothing
here is stubbed, so a test passing means the shipped code works, not that a mock
agrees with itself.

Each rule below is designed to trip exactly one check, and CleanAttack is
designed to trip none -- a false positive on that rule is as much a failure as a
missed fault on the others.

Usage:
    python tests/build_fixture.py --out /tmp/talos_fixture
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DAY = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()   # fixture time origin


def at(hh, mm, ss=0):
    """Epoch seconds for a wall-clock time on the fixture day (UTC == local)."""
    return DAY + hh * 3600 + mm * 60 + ss


ATTACKER = "10.0.0.9"
VICTIM = "10.0.0.1"
GHOST = "10.0.0.99"        # attacker address the manifest does not declare
ABSENT = "10.0.0.55"       # victim that never appears in the traffic

# name, start, end, canonical class -- transcribed into the fixture manifest
RULES = [
    ("CleanAttack", "10:00", "10:10", "dos"),
    ("LateWindow", "12:00", "12:10", "dos"),
    ("EarlyEnd", "13:00", "13:10", "ddos"),
    ("Gappy", "14:00", "14:30", "ddos"),
    ("HiddenAlias", "15:00", "15:10", "portscan"),
    ("SilentRule", "16:00", "16:10", "botnet"),
]

MANIFEST = """\
# Synthetic manifest for validator testing. Times are UTC so the fixture's
# expected epochs can be read straight off the clock.
dataset: toy
utc_offset: "+00:00"
capture_window: 2020-01-01 (single day)
match_default: [time, attacker, victim]

attacks:
  # Correct in every respect: traffic fills the window and stops with it.
  - {name: CleanAttack, date: 2020-01-01, start: "10:00", end: "10:10",
     attacker_ips: [10.0.0.9], victim_ips: [10.0.0.1]}
  # Real traffic starts ten minutes before this window opens.
  - {name: LateWindow, date: 2020-01-01, start: "12:00", end: "12:10",
     attacker_ips: [10.0.0.9], victim_ips: [10.0.0.1]}
  # Real traffic runs fifteen minutes past this window's close.
  - {name: EarlyEnd, date: 2020-01-01, start: "13:00", end: "13:10",
     attacker_ips: [10.0.0.9], victim_ips: [10.0.0.1]}
  # Thirty-minute window, but the attack only runs in the first and last five.
  - {name: Gappy, date: 2020-01-01, start: "14:00", end: "14:30",
     attacker_ips: [10.0.0.9], victim_ips: [10.0.0.1]}
  # Most of the attack comes from an address this manifest never lists.
  - {name: HiddenAlias, date: 2020-01-01, start: "15:00", end: "15:10",
     attacker_ips: [10.0.0.9], victim_ips: [10.0.0.1]}
  # Scheduled against a host that produced no traffic at all.
  - {name: SilentRule, date: 2020-01-01, start: "16:00", end: "16:10",
     attacker_ips: [10.0.0.9], victim_ips: [10.0.0.55]}
"""

TAXONOMY = """\
# Synthetic taxonomy for validator testing.
canonical_classes:
  benign:   traffic not matched by any attack rule
  dos:      single-source denial of service
  ddos:     distributed denial of service
  portscan: port / service reconnaissance
  botnet:   C2 traffic

requires_payload:
  - CleanAttack
  - LateWindow

raw_to_canonical:
  CleanAttack: dos
  LateWindow: dos
  EarlyEnd: ddos
  Gappy: ddos
  HiddenAlias: portscan
  SilentRule: botnet
"""


def block(tag, t0, seconds, rate, orig, resp, port=80, proto="tcp",
          conn_state="SF", duration=0.5, obytes=500, rbytes=1500):
    """One SELECT generating `rate` flows/second of one traffic pattern.

    Generated in SQL rather than Python: building a few hundred thousand rows
    through executemany takes minutes, through range() it takes no time at all.
    """
    n = int(seconds * rate)
    return f"""
    SELECT {t0}::DOUBLE + (i::DOUBLE / {rate}) AS ts,
           '{tag}_' || i::VARCHAR AS uid,
           '{orig}' AS "id.orig_h", (1024 + (i % 40000))::BIGINT AS "id.orig_p",
           '{resp}' AS "id.resp_h", {port}::BIGINT AS "id.resp_p",
           '{proto}' AS proto,
           {'NULL' if port not in (80, 443) else "'http'"}::VARCHAR AS service,
           {duration}::DOUBLE AS duration,
           {obytes}::BIGINT AS orig_bytes, {rbytes}::BIGINT AS resp_bytes,
           '{conn_state}' AS conn_state,
           false AS local_orig, true AS local_resp, 0::BIGINT AS missed_bytes,
           'ShADadFf' AS history,
           5::BIGINT AS orig_pkts, {obytes + 200}::BIGINT AS orig_ip_bytes,
           5::BIGINT AS resp_pkts, {rbytes + 200}::BIGINT AS resp_ip_bytes,
           6::BIGINT AS ip_proto
    FROM range(0, {n}) t(i)"""


def build_conn(con):
    """The synthetic capture: benign background plus one block per planted fault."""
    parts = [
        # Benign background across the whole day, to both the victim and elsewhere.
        block("bgA", at(9, 0), 7 * 3600, 2, "10.0.1.5", "10.0.2.7", port=443),
        block("bgB", at(9, 0), 7 * 3600, 1, "10.0.1.6", VICTIM, port=443),

        # CleanAttack -- fills its window exactly (the schedule is minute-granular,
        # so labelling extends the end through 10:10:59; the traffic does too).
        block("clean", at(10, 0), 660, 100, ATTACKER, VICTIM, conn_state="S0"),

        # LateWindow -- the flood is already running before the window opens.
        block("late", at(11, 50), 1260, 100, ATTACKER, VICTIM, conn_state="S0"),

        # EarlyEnd -- the flood keeps running well past the window's close.
        block("early", at(13, 0), 1500, 100, ATTACKER, VICTIM, conn_state="S0"),

        # Gappy -- attack only in the first and last five minutes of a 30-min window.
        block("gap1", at(14, 0), 300, 100, ATTACKER, VICTIM, conn_state="S0"),
        block("gap2", at(14, 25), 360, 100, ATTACKER, VICTIM, conn_state="S0"),

        # HiddenAlias -- most of the attack arrives from an undeclared address.
        block("ghost", at(15, 0), 660, 10, GHOST, VICTIM, port=22, conn_state="REJ"),
        block("declared", at(15, 0), 660, 0.3, ATTACKER, VICTIM, port=22,
              conn_state="REJ"),
    ]
    con.execute("CREATE TABLE conn AS " + "\nUNION ALL\n".join(parts))
    # A duplicated uid: the same flow ingested twice, as happens when a pcap
    # split is converted to parquet more than once.
    con.execute("INSERT INTO conn SELECT * FROM conn WHERE uid LIKE 'clean_1%' "
                "AND CAST(substr(uid, 7) AS BIGINT) < 5")
    return con.execute("SELECT count(*) FROM conn").fetchone()[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/tmp/talos_fixture")
    ap.add_argument("--keep", action="store_true", help="do not wipe an existing fixture")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.keep:
        shutil.rmtree(out)
    mans = out / "manifests"
    conn_dir = out / "parquets" / "toy" / "day1"
    lab_dir = out / "labelled" / "toy"
    for d in (mans, conn_dir, lab_dir):
        d.mkdir(parents=True, exist_ok=True)
    (mans / "toy.yaml").write_text(MANIFEST)
    (mans / "taxonomy.yaml").write_text(TAXONOMY)

    con = duckdb.connect()
    n = build_conn(con)
    conn_path = conn_dir / "conn.parquet"
    con.execute(f"COPY (SELECT * FROM conn ORDER BY ts) TO '{conn_path}' (FORMAT PARQUET)")
    print(f"synthetic conn: {n:,} flows -> {conn_path}")

    env_note = f"TALOS_MANIFESTS={mans}"
    cmd = [sys.executable, str(ROOT / "src/preprocess/label/label_flows.py"),
           "--dataset", "toy",
           "--source", f"{out}/parquets/toy/**/conn.parquet",
           "--out", str(lab_dir / "conn.parquet"),
           "--report", str(out / "labelling_toy.json")]
    print(f"\n$ {env_note} {' '.join(cmd)}\n")
    import os
    env = dict(os.environ, TALOS_MANIFESTS=str(mans))
    r = subprocess.run(cmd, env=env, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit("fixture labelling failed")

    print(f"\nfixture ready at {out}")
    print(f"  manifests  {mans}")
    print(f"  conn       {conn_path}")
    print(f"  labelled   {lab_dir / 'conn.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
