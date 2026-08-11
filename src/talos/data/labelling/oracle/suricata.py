"""Suricata over the same pcaps, as a second opinion the schedule never saw.

Signature detection is the one source of evidence about these captures that is
genuinely independent of the manifests: it reads packets, not documentation, and
it has no idea what CIC published. That independence is the whole value —
"prefer independent evidence over volume" — and it is why one oracle is worth
more than another forty self-referential checks.

What it can and cannot say, which the API is shaped to keep straight:

  an alert on a schedule-attack flow  CORROBORATES the label
  an alert on a schedule-benign flow  is a CANDIDATE MISSED ATTACK
  no alert on a flow                  says NOTHING

"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_IMAGE = "jasonish/suricata:latest"
EVE = "eve.json"

# Read `timestamp` as text, not as a timestamp.
#
# Left to itself DuckDB's JSON sniffer types an eve timestamp as TIMESTAMP --
# naive -- and DROPS the offset while doing it, so `12:20:15+0000` and
# `09:20:15-0300` both arrive as naive 12:20:15. They are the same instant, but
# the second one is now three hours wrong, and nothing downstream can tell.
# That is precisely the ambiguity the oracle exists to remove, and it would have
# been removed wrongly, silently, on the one measurement (`OffsetProbe`) that
# has no other check on it.
#
# Taking the column as VARCHAR and casting to TIMESTAMPTZ ourselves keeps the
# offset: both spellings above then give epoch 1499170815.123456.
#
# Naming columns also pins the read: fields we do not list are ignored rather
# than sniffed, so an eve schema that gains a field cannot change our types.
EVE_COLUMNS = ("{'timestamp': 'VARCHAR', 'event_type': 'VARCHAR', "
               "'src_ip': 'VARCHAR', 'dest_ip': 'VARCHAR', "
               "'alert': 'STRUCT(signature VARCHAR)'}")

# Fields we take off an eve alert record. Everything else is Suricata's business.
ALERT_FIELDS = ("timestamp", "src_ip", "src_port", "dest_ip", "dest_port", "proto",
                "signature", "category", "signature_id", "severity")


class SuricataError(Exception):
    pass


@dataclass(frozen=True)
class Alert:
    ts: float                   # epoch seconds, UTC
    src_ip: str
    src_port: int | None
    dest_ip: str
    dest_port: int | None
    proto: str
    signature: str
    category: str
    signature_id: int | None
    severity: int | None


class SuricataOracle:
    """Runs Suricata over a pcap and turns its alerts into rows.

    Runs the local binary when there is one, otherwise a container — the same
    arrangement `ZeekExtractor` uses, and for the same reason: the analysis tool
    is a deployment choice, not a property of the platform.
    """

    def __init__(self, image: str = DEFAULT_IMAGE, binary: str = "suricata",
                 config: str | None = None):
        self.image = image
        self.binary = binary
        self.config = config

    # ------------------------------------------------------------ invocation

    def available(self) -> tuple[bool, str]:
        """(can run, how). Reported rather than raised, so a caller can skip."""
        if shutil.which(self.binary):
            return True, "local"
        if shutil.which("docker"):
            return True, "docker"
        return False, ("neither a suricata binary nor docker is available; "
                       "corroboration cannot run")

    def command(self, pcap: Path, out_dir: Path, how: str) -> list[str]:
        """The invocation, as a list. Separated out so tests can read it."""
        if how == "local":
            cmd = [self.binary, "-r", str(pcap), "-l", str(out_dir)]
            if self.config:
                cmd += ["-c", self.config]
            return cmd
        return [
            "docker", "run", "--rm",
            "-e", "TZ=UTC",                    # see the module docstring
            "-v", f"{pcap.parent}:/pcap:ro",
            "-v", f"{out_dir}:/out",
            self.image,
            "suricata", "-r", f"/pcap/{pcap.name}", "-l", "/out",
        ]

    def run(self, pcap: str | Path, out_dir: str | Path) -> Path:
        """Analyse one pcap; return the path to its eve.json."""
        pcap, out_dir = Path(pcap).resolve(), Path(out_dir).resolve()
        if not pcap.exists():
            raise SuricataError(f"no such pcap: {pcap}")
        out_dir.mkdir(parents=True, exist_ok=True)

        ok, how = self.available()
        if not ok:
            raise SuricataError(how)

        env = {**os.environ, "TZ": "UTC"}
        result = subprocess.run(self.command(pcap, out_dir, how), env=env,
                                capture_output=True, text=True)
        eve = out_dir / EVE
        # Suricata exits non-zero on some rule-load warnings while still having
        # produced a perfectly good eve.json, so the artefact is the test, not
        # the exit code -- but a missing artefact reports the exit code.
        if not eve.exists():
            raise SuricataError(
                f"suricata produced no {EVE} (exit {result.returncode})\n"
                f"{result.stderr.strip()[:2000]}")
        return eve

    # --------------------------------------------------------------- parsing

    @staticmethod
    def parse_timestamp(value: str) -> float:
        """eve timestamp -> epoch seconds. The offset must be present.

        Any offset is fine -- `09:20:15-0300` and `12:20:15+0000` are the same
        instant and both convert correctly. A timestamp with none cannot be
        resolved at all, and would silently adopt the reader's timezone, which
        is the one ambiguity this oracle exists to remove.
        """
        from datetime import datetime

        text = value.strip()
        # "…+0000" -> "…+00:00", which fromisoformat needs
        if len(text) > 5 and text[-5] in "+-" and text[-3] != ":":
            text = f"{text[:-2]}:{text[-2:]}"
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError as exc:
            raise SuricataError(f"unreadable eve timestamp {value!r}") from exc
        if stamp.utcoffset() is None:
            raise SuricataError(
                f"eve timestamp {value!r} carries no UTC offset, so it names no "
                f"instant. Run suricata with TZ set, or with an eve-log "
                f"timestamp-format that includes the offset.")
        return stamp.timestamp()

    @staticmethod
    def read_sql(eve: str | Path) -> str:
        """The eve read, as SQL. One definition, used by both readers below.

        `ignore_errors` because eve is append-only and a run cut short leaves a
        torn final record.
        """
        return (f"read_json_auto('{eve}', format='newline_delimited', "
                f"ignore_errors=true, columns={EVE_COLUMNS})")

    def alerts_relation(self, duck, eve: str | Path):
        """Alerts as a DuckDB relation, streamed off disk.

        The preferred path for anything real. eve.json is newline-delimited
        JSON, which DuckDB reads natively without materialising the file, and
        the result joins directly against conn instead of being inlined as
        literal SQL.

        The offset check runs FIRST: a file whose timestamps name no instant
        must be refused before a relation over it can be handed to anyone.
        """
        self._check_offsets(duck, eve)
        return duck.relation(f"""
            SELECT epoch(CAST(timestamp AS TIMESTAMPTZ)) AS ts,
                   src_ip, dest_ip, alert.signature AS signature
            FROM {self.read_sql(eve)}
            WHERE event_type = 'alert'""")

    @classmethod
    def _check_offsets(cls, duck, eve: str | Path) -> None:
        """Refuse a file whose timestamps name no instant. Checked on one row.

        The rule is `parse_timestamp`'s, applied to the raw string, so there is
        one definition of "carries an offset" rather than a second one here that
        could drift. One row is enough: Suricata formats every record in a run
        the same way.
        """
        row = duck.one(f"""
            SELECT timestamp FROM {cls.read_sql(eve)}
            WHERE event_type = 'alert' LIMIT 1""")
        if row and row[0]:
            cls.parse_timestamp(str(row[0]))

    def parse_alerts(self, eve: str | Path) -> list[Alert]:
        """Alert records as objects. Streamed, but still one list in memory.

        For small captures and for tests. Anything at dataset scale should use
        `alerts_relation`: 2019 produces ~5M alerts, and holding those as Python
        objects costs more than the analysis does.
        """
        alerts = []
        with open(eve, encoding="utf-8") as fh:
          for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue                    # eve is append-only; a torn last line is normal
            if record.get("event_type") != "alert":
                continue
            body = record.get("alert") or {}
            alerts.append(Alert(
                  ts=self.parse_timestamp(record["timestamp"]),
                  src_ip=record.get("src_ip", ""),
                  src_port=record.get("src_port"),
                  dest_ip=record.get("dest_ip", ""),
                  dest_port=record.get("dest_port"),
                  proto=record.get("proto", ""),
                  signature=body.get("signature", ""),
                  category=body.get("category", ""),
                  signature_id=body.get("signature_id"),
                  severity=body.get("severity"),
              ))
        return alerts

    @staticmethod
    def alerts_table(alerts: list[Alert]) -> str:
        """Alerts as an inline VALUES table, for joining against conn."""
        if not alerts:
            return ("SELECT * FROM (VALUES (CAST(NULL AS DOUBLE), CAST(NULL AS VARCHAR), "
                    "CAST(NULL AS VARCHAR), CAST(NULL AS VARCHAR))) "
                    "AS a(ts, src_ip, dest_ip, signature) WHERE false")
        rows = ",\n    ".join(
            "({}, '{}', '{}', '{}')".format(
                a.ts, a.src_ip, a.dest_ip, a.signature.replace("'", "''"))
            for a in alerts)
        return (f"SELECT * FROM (VALUES\n    {rows}\n  ) "
                f"AS a(ts, src_ip, dest_ip, signature)")
