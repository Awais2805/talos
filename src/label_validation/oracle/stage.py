#!/usr/bin/env python3
"""An externally produced ground truth, and the join that makes it comparable.

Every check in tiers 0-3 reasons from inside this project. The integrity checks
ask whether our labelled zone is arithmetically consistent with itself; the
schedule checks ask whether our transcription of a published web page is
internally coherent; even the corroboration tier, which reads logs the labels
were never attached to, is reading Zeek output about the same packets our labels
describe. None of them can say the label is WRONG, because none of them has a
second opinion to compare against.

This module supplies one. Engelen, Lanvin, Rimmer and Joosen (CNS 2022) re-did
the labelling of CIC-IDS-2017 and CSE-CIC-IDS-2018 from the original PCAPs by
hand -- reading attack tooling output, following TCP streams, and separating
attacks that executed from attacks that merely connected. Their corrected label
sets were produced without reference to our manifests, our Zeek extraction or
our taxonomy, so where they disagree with us the disagreement is evidence rather
than a restatement.

There is no such corrected set for CIC-DDoS-2019. That dataset is skipped, with
the reason recorded, and never reported as a failure.

WHAT THE JOIN CAN AND CANNOT SAY
--------------------------------
The two sides do not construct flows the same way and never will. CNS2022 ran
CICFlowMeter with a hard 120-second ACTIVE timeout, which chops a long
connection into a chain of flow records; Zeek uses protocol-specific INACTIVITY
timeouts and reports one conn record for the whole connection. So one Zeek flow
maps to N >= 1 oracle rows, and the two flow COUNTS are expected to differ by a
large factor. Any comparison that treats a count difference as a labelling error
is measuring flow-construction semantics and calling it ground truth.

CICFlowMeter also fixes direction from the first packet it happens to see, while
Zeek fixes it from the SYN, so the two disagree about which end is the
originator often enough that an ordered 5-tuple join silently loses those flows.

The join therefore matches on the UNORDERED 5-tuple -- the two (address, port)
endpoints sorted, plus the transport -- AND requires the two flow intervals to
overlap in time. Direction agreement is carried through as a boolean rather than
being required, so the reversal rate is a number we report instead of a
population we lose. ICMP keys on the address pair alone, because CICFlowMeter
writes zero ports for ICMP while Zeek writes type and code there; joining those
on ports would manufacture a total mismatch out of an encoding difference.

The unit of comparison is the Zeek uid, never the row. For each uid we record
the set of oracle labels whose intervals overlap it and a PURITY -- the share of
those overlapping rows that agree on one canonical class. A uid matched by
fifteen oracle rows that all say `DoS Hulk` is one agreement, not fifteen.

Provenance, including the fact that the source states NO LICENCE anywhere, is
written into the report alongside the numbers.

Usage:
    # default: describe what exists and what would be fetched, touching nothing
    python src/label_validation/oracle/stage.py --dataset cic-ids-2017

    # the archive is 343 MB (2017) / 9.7 GB (2018); --fetch is opt-in
    python src/label_validation/oracle/stage.py --dataset cic-ids-2017 --fetch
    python src/label_validation/oracle/stage.py --dataset cic-ids-2017 --stage
    python src/label_validation/oracle/stage.py --dataset cic-ids-2017 --join

    # stage from CSVs already unpacked elsewhere, and join a local fixture
    python src/label_validation/oracle/stage.py --dataset toy --stage-from /tmp/csv \
        --join --lake-root /tmp/fixture --no-s3
"""

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "config.yml").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.lake import Lake                          # noqa: E402
from src.label_validation.core.paths import portable      # noqa: E402

REPORTS = ROOT / "reports" / "validation"
CACHE = ROOT / "data" / "oracle"

# Engelen, Lanvin, Rimmer, Joosen -- "Troubleshooting an Intrusion Detection
# Dataset: the CICIDS2017 Case Study" and its 2022 successor. The archives sit
# behind an open Apache directory index with no authentication and no licence
# statement of any kind: not a LICENSE file, not a header on the index, not a
# clause in the paper. That absence is recorded in every report this writes,
# because "we could download it" is not the same fact as "we may redistribute
# it", and only the first one has been established.
CITATION = ("L. D'hooge, mirror of: G. Engelen, V. Rimmer, W. Joosen, "
            "\"Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 "
            "Case Study\" (IEEE S&P Workshops 2021) and M. Lanvin et al., "
            "\"Errors in the CICIDS2017 dataset and the resulting improved "
            "version\" (CNS 2022). DistriNet, KU Leuven.")

LICENCE = ("not stated -- the source publishes no licence, no terms of use and "
           "no redistribution clause; treat as all rights reserved")

BASE_URL = "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets"

ORACLE_SOURCES = {
    "cic-ids-2017": {
        "archive": "CICIDS2017_improved.zip",
        "url": f"{BASE_URL}/CICIDS2017_improved.zip",
        "bytes": 343_549_013,
        # Stored uncompressed inside the zip, so extraction is a straight copy.
        "entries": ("monday.csv", "tuesday.csv", "wednesday.csv", "thursday.csv",
                    "friday.csv"),
    },
    "cic-ids-2018": {
        "archive": "CSECICIDS2018_improved.zip",
        "url": f"{BASE_URL}/CSECICIDS2018_improved.zip",
        "bytes": 10_426_851_729,
        # Ten capture days at 3-4.2 GB each. Nothing here ever holds one in
        # memory; see stage().
        "entries": ("Wednesday-14-02-2018.csv", "Thursday-15-02-2018.csv",
                    "Friday-16-02-2018.csv", "Tuesday-20-02-2018.csv",
                    "Wednesday-21-02-2018.csv", "Thursday-22-02-2018.csv",
                    "Friday-23-02-2018.csv", "Wednesday-28-02-2018.csv",
                    "Thursday-01-03-2018.csv", "Friday-02-03-2018.csv"),
    },
}

# Datasets for which no corrected label set exists, and why. Recorded rather
# than omitted: a dataset absent from both tables would be indistinguishable
# from one whose oracle simply has not been staged yet.
NO_ORACLE = {
    "cic-ddos-2019": "DistriNet's CNS2022 re-labelling covers CIC-IDS-2017 and "
                     "CSE-CIC-IDS-2018 only. No manually forensicated label set "
                     "has been published for CIC-DDoS-2019 by anyone, so its "
                     "labels cannot be checked against an external ground truth "
                     "at all -- absence of an oracle, not absence of agreement.",
}

# Columns the join reads. The improved CSVs carry 91; the other 82 are
# CICFlowMeter statistics computed from a flow construction we do not share, so
# reading them would invite comparisons that cannot mean anything.
CSV_COLUMNS = ("id", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol",
               "Timestamp", "Flow Duration", "Label", "Attempted Category")

# CICFlowMeter reads an ARP frame as though it were IPv4 and lands on these two
# addresses. Neither is routable; their presence marks a row the extractor
# invented. Same constant as checks/artefact.py, restated here because this
# module must not import a check.
ARTEFACT_IPS = ("8.0.6.4", "8.6.0.1")

PROTO_BY_NUMBER = {1: "icmp", 6: "tcp", 17: "udp"}

# The oracle's second label axis. An attack row carrying one of these is an
# attack that was attempted but did not do what its name claims -- a distinction
# our own labels have no column for, which is why disagreements involving it
# indict neither side.
ATTEMPTED_CATEGORIES = {
    0: "no payload sent",
    1: "port or system closed",
    2: "attack startup or teardown artefact",
    3: "no malicious payload",
    4: "attack artefact",
    5: "attack implemented incorrectly",
    6: "target system unresponsive",
    -1: "not applicable",
}

# Oracle label -> our canonical class, keyed on the normalised label (see
# _norm). Both CIC's original spellings and DistriNet's corrected ones are
# listed, including the CP1252 en-dash CIC's 2017 CSVs carry in the Web Attack
# names, which normalisation folds to the same key as a plain hyphen.
#
# A label absent from this table is NOT silently binned: it is carried through
# the join as class `unmapped` and reported, because a taxonomy that quietly
# discards labels it does not recognise would report perfect agreement on a
# dataset it had mostly failed to read.
ORACLE_TO_CANONICAL = {
    "benign": "benign",
    # 2017
    "ftp-patator": "brute_force",
    "ssh-patator": "brute_force",
    "dos-slowloris": "dos",
    "dos-slowhttptest": "dos",
    "dos-hulk": "dos",
    "dos-goldeneye": "dos",
    "heartbleed": "heartbleed",
    "web-attack-brute-force": "web_attack",
    "web-attack-xss": "web_attack",
    "web-attack-sql-injection": "web_attack",
    "infiltration": "infiltration",
    # DistriNet split the 2017 infiltration scenario's internal sweep out under
    # its own name; our taxonomy classes that sweep as portscan, so this is the
    # one place the two vocabularies are deliberately reconciled rather than
    # matched literally.
    "infiltration-portscan": "portscan",
    "infiltration-nmap": "portscan",
    "infiltration-cooldisk": "infiltration",
    "infiltration-dropbox-download": "infiltration",
    "botnet": "botnet",
    "botnet-ares": "botnet",
    "bot": "botnet",
    "portscan": "portscan",
    "ddos": "ddos",
    # 2018
    "ftp-bruteforce": "brute_force",
    "ssh-bruteforce": "brute_force",
    "dos-attacks-goldeneye": "dos",
    "dos-attacks-slowloris": "dos",
    "dos-attacks-slowhttptest": "dos",
    "dos-attacks-hulk": "dos",
    "ddos-attack-loic-udp": "ddos",
    "ddos-attack-hoic": "ddos",
    "ddos-attacks-loic-http": "ddos",
    "ddos-loic-http": "ddos",
    "ddos-loic-udp": "ddos",
    "ddos-hoic": "ddos",
    "brute-force-web": "web_attack",
    "brute-force-xss": "web_attack",
    "sql-injection": "web_attack",
    "web-attack-brute-force-web": "web_attack",
}

UNMAPPED = "unmapped"


# ------------------------------------------------------------------ plumbing

def _lit(v):
    return "'" + str(v).replace("'", "''") + "'"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(label):
    """Oracle label -> a key that survives CIC's punctuation.

    The same attack is spelled `DoS Hulk`, `DoS attacks-Hulk` and
    `Web Attack \\x96 Brute Force` across the published files. Everything that is
    not alphanumeric collapses to a single hyphen, so the key depends on the
    words and not on which dash the CSV happened to be written with.
    """
    out, prev_sep = [], True
    for ch in str(label).lower():
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
        elif not prev_sep:
            out.append("-")
            prev_sep = True
    return "".join(out).strip("-")


def _norm_sql(expr):
    """The SQL twin of _norm. Both sides of the join must key identically."""
    return (f"trim(regexp_replace(lower({expr}), '[^a-z0-9]+', '-', 'g'), '-')")


def _endpoint_key(ip_a, port_a, ip_b, port_b, proto):
    """The unordered 5-tuple, as (lower, upper) endpoint strings.

    Sorting the two endpoints removes direction from the key, which is the whole
    point: CICFlowMeter and Zeek disagree about which end originated a
    connection often enough that an ordered key drops real matches. ICMP keys on
    addresses alone because the two tools put different things in the port
    fields, and a key built from those would never match.
    """
    def ep(ip, port):
        return (f"CASE WHEN {proto} = 'icmp' THEN {ip} "
                f"ELSE {ip} || '|' || coalesce(CAST({port} AS VARCHAR), '') END")
    a, b = ep(ip_a, port_a), ep(ip_b, port_b)
    return f"least({a}, {b})", f"greatest({a}, {b})"


def oracle_paths(dataset, reports=REPORTS):
    """Where --join writes, and where the runner looks for its capability probe."""
    return (Path(reports) / f"oracle_{dataset}.json",
            Path(reports) / f"oracle_{dataset}_uid.parquet")


def staged_dir(cache, dataset):
    return Path(cache) / "staged" / dataset


def staged_files(cache, dataset):
    return sorted(staged_dir(cache, dataset).glob("*.parquet"))


def curl_command(src, dest):
    """The exact command to fetch the archive, for a user who owns the machine.

    Printed rather than run by default. The 2018 archive is 9.7 GB and lands on
    infrastructure this module does not administer; a validation tool that helps
    itself to ten gigabytes of someone else's bandwidth on import is not a tool
    anyone should trust with their lake.
    """
    return (f"mkdir -p {dest.parent}\n"
            f"curl -L --fail --retry 3 --continue-at - \\\n"
            f"  -o {dest} \\\n"
            f"  {src['url']}\n"
            f"# expect exactly {src['bytes']:,} bytes; licence: {LICENCE}")


# ------------------------------------------------------------------- fetch

def fetch(dataset, cache):
    """Download the archive. Only ever reached through an explicit --fetch."""
    src = ORACLE_SOURCES[dataset]
    dest = Path(cache) / "archives" / src["archive"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == src["bytes"]:
        print(f"archive already complete: {dest} ({src['bytes']:,} bytes)")
        return dest
    import urllib.request                      # imported here, not at module load
    print(f"fetching {src['url']}\n     -> {dest}  ({src['bytes']:,} bytes)")
    with urllib.request.urlopen(src["url"]) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    got = dest.stat().st_size
    if got != src["bytes"]:
        sys.exit(f"size mismatch: got {got:,}, expected {src['bytes']:,} — "
                 f"the download is incomplete or the source has changed")
    return dest


# ------------------------------------------------------------------- stage

def _stage_select(csv_path, day, ignore_errors):
    """CSV -> the ten columns the join needs, typed and keyed.

    Read as all-VARCHAR and cast explicitly. Type inference over a 4 GB file
    samples a prefix and then dies partway through on the first `Infinity` or
    empty cell in a column it decided was numeric -- and the columns it would
    guess wrong on are CICFlowMeter's rate statistics, which we do not read at
    all. try_cast turns an unparseable cell into a NULL we can count instead of
    a crash three gigabytes into a conversion.
    """
    proto = ("CASE try_cast(\"Protocol\" AS INTEGER) "
             + " ".join(f"WHEN {n} THEN {_lit(p)}" for n, p in PROTO_BY_NUMBER.items())
             + " ELSE 'proto-' || coalesce(trim(\"Protocol\"), '?') END")
    ts = "epoch_us(try_cast(trim(\"Timestamp\") AS TIMESTAMP)) / 1e6"
    label = "trim(coalesce(\"Label\", ''))"
    # ` - Attempted` is a suffix on the label, not a separate column, so the base
    # attack name has to be recovered before anything can be mapped.
    base = f"trim(regexp_replace({label}, '(?i)\\s*-\\s*attempted\\s*$', ''))"
    lo, hi = _endpoint_key('src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto')
    return f"""
    WITH raw AS (
      SELECT try_cast("id" AS BIGINT) AS oracle_id,
             trim("Src IP") AS src_ip, try_cast("Src Port" AS INTEGER) AS src_port,
             trim("Dst IP") AS dst_ip, try_cast("Dst Port" AS INTEGER) AS dst_port,
             {proto} AS proto,
             {ts} AS ts,
             try_cast("Flow Duration" AS DOUBLE) / 1e6 AS duration,
             {label} AS label, {base} AS base_label,
             regexp_matches({label}, '(?i)-\\s*attempted\\s*$') AS attempted,
             coalesce(try_cast("Attempted Category" AS INTEGER), -1) AS attempted_category
      FROM read_csv_auto({_lit(str(csv_path))}, header = true, all_varchar = true,
                         ignore_errors = {'true' if ignore_errors else 'false'}))
    SELECT oracle_id, ts, ts + coalesce(duration, 0.0) AS ts_end, duration,
           src_ip, src_port, dst_ip, dst_port, proto,
           {lo} AS ep_lo, {hi} AS ep_hi,
           label, base_label, {_norm_sql('base_label')} AS label_key,
           attempted, attempted_category,
           CASE WHEN proto = 'proto-0' THEN 'protocol_zero'
                WHEN src_ip IN {ARTEFACT_IPS} OR dst_ip IN {ARTEFACT_IPS} THEN 'arp_as_ipv4'
                WHEN proto <> 'icmp' AND coalesce(src_port, 0) = 0
                                    AND coalesce(dst_port, 0) = 0 THEN 'zero_ports'
                END AS artefact,
           {_lit(day)} AS oracle_day
    FROM raw"""


def _convert(con, csv_path, out_path, day, ignore_errors):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({_stage_select(csv_path, day, ignore_errors)}) "
                f"TO {_lit(str(out_path))} (FORMAT PARQUET, COMPRESSION ZSTD)")
    return con.execute(f"SELECT count(*), count(*) FILTER (WHERE ts IS NULL), "
                       f"count(*) FILTER (WHERE artefact IS NOT NULL) "
                       f"FROM read_parquet({_lit(str(out_path))})").fetchone()


def stage(dataset, cache, con, source=None, force=False, ignore_errors=False):
    """Unpack the archive and convert each capture day to a narrow parquet.

    Streaming throughout, and deliberately so: the 2018 entries are 3-4.2 GB of
    uncompressed CSV each, on a box that has twice wedged itself on this lake.
    Each entry is copied out of the zip in 4 MB blocks to a scratch file,
    converted by DuckDB row group by row group, and deleted before the next one
    starts, so peak memory is a buffer and peak disk is one capture day.

    The staged parquet keeps ten columns of ninety-one and is roughly two orders
    of magnitude smaller than its source, which is what makes the join
    re-runnable without re-reading the archive.
    """
    out_dir = staged_dir(cache, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(cache) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    staged = []

    if source:
        src = Path(source)
        csvs = sorted(src.glob("*.csv")) if src.is_dir() else [src]
        if not csvs:
            sys.exit(f"no .csv found at {src}")
        for path in csvs:
            day = path.stem
            out = out_dir / f"{day}.parquet"
            if out.exists() and not force:
                print(f"  = {day:<28} already staged")
                staged.append(out)
                continue
            n, bad_ts, art = _convert(con, path, out, day, ignore_errors)
            print(f"  + {day:<28} {n:>12,} rows  {art:>10,} artefact  "
                  f"{bad_ts:>8,} unparsed ts")
            staged.append(out)
        return staged

    src = ORACLE_SOURCES[dataset]
    archive = Path(cache) / "archives" / src["archive"]
    if not archive.exists():
        sys.exit(f"archive not present: {archive}\n\n"
                 f"fetch it yourself (nothing here downloads by default):\n\n"
                 f"{curl_command(src, archive)}\n")
    with zipfile.ZipFile(archive) as zf:
        entries = [i for i in zf.infolist()
                   if i.filename.lower().endswith(".csv") and not i.is_dir()]
        for info in sorted(entries, key=lambda i: i.filename):
            day = Path(info.filename).stem
            out = out_dir / f"{day}.parquet"
            if out.exists() and not force:
                print(f"  = {day:<28} already staged")
                staged.append(out)
                continue
            scratch = tmp_dir / Path(info.filename).name
            with zf.open(info) as fh, open(scratch, "wb") as w:
                shutil.copyfileobj(fh, w, 1 << 22)
            try:
                n, bad_ts, art = _convert(con, scratch, out, day, ignore_errors)
            finally:
                scratch.unlink(missing_ok=True)
            print(f"  + {day:<28} {n:>12,} rows  {art:>10,} artefact  "
                  f"{bad_ts:>8,} unparsed ts")
            staged.append(out)
    return staged


def provenance(dataset, cache, staged):
    """What the numbers rest on, including what is not known about it."""
    src = ORACLE_SOURCES.get(dataset)
    if not src:
        return {"registered": False, "dataset": dataset,
                "note": "staged from a path supplied on the command line; no "
                        "published archive is registered for this dataset",
                "licence": "unknown", "staged_files": [p.name for p in staged]}
    archive = Path(cache) / "archives" / src["archive"]
    on_disk = archive.stat().st_size if archive.exists() else None
    return {
        "registered": True,
        "source": src["archive"],
        "url": src["url"],
        "bytes_expected": src["bytes"],
        "bytes_on_disk": on_disk,
        # A size that does not match is not a warning to be filed away: the
        # archive is either truncated or has been republished, and either way
        # the rows staged from it are not the rows this table describes.
        "bytes_match": (on_disk == src["bytes"]) if on_disk is not None else None,
        "retrieved_utc": (datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc)
                          .isoformat(timespec="seconds") if archive.exists() else None),
        "licence": LICENCE,
        "citation": CITATION,
        "entries_expected": list(src["entries"]),
        "entries_staged": [p.stem for p in staged],
        "entries_missing": sorted({Path(e).stem for e in src["entries"]}
                                  - {p.stem for p in staged}),
    }


# -------------------------------------------------------------------- join

MATCH_TABLE = """
CREATE OR REPLACE TEMP TABLE match_rollup (
  uid VARCHAR, base_label VARCHAR, oracle_class VARCHAR,
  n BIGINT, n_dir BIGINT, n_attempted BIGINT,
  c0 BIGINT, c1 BIGINT, c2 BIGINT, c3 BIGINT, c4 BIGINT, c5 BIGINT, c6 BIGINT,
  c_other BIGINT, ts_min DOUBLE, ts_max DOUBLE)"""

ATT_CATS = (0, 1, 2, 3, 4, 5, 6)


def _map_cte():
    rows = ",\n        ".join(f"({_lit(k)}, {_lit(v)})"
                              for k, v in sorted(ORACLE_TO_CANONICAL.items()))
    return f"lmap AS (SELECT * FROM (VALUES\n        {rows}\n      ) AS t(k, canonical))"


def _zeek_cte(glob, lo, hi):
    """The labelled zone, narrowed to one oracle day and to the columns joined on.

    The lower bound is the oracle day's first flow minus a lookback rather than
    the day itself: a Zeek connection that began earlier and was still open can
    legitimately overlap the day's first oracle rows. The bound exists so DuckDB
    can drop row groups on the parquet ts statistics, which is what keeps this
    affordable against 2018's overlapping captures.
    """
    z_lo, z_hi = _endpoint_key('"id.orig_h"', '"id.orig_p"',
                               '"id.resp_h"', '"id.resp_p"', "proto")
    return f"""z AS (
      SELECT uid, ts, coalesce(duration, 0.0) AS dur, proto,
             "id.orig_h" AS orig_h, "id.orig_p" AS orig_p,
             "id.resp_h" AS resp_h, "id.resp_p" AS resp_p,
             {z_lo} AS ep_lo, {z_hi} AS ep_hi
      FROM read_parquet({_lit(glob)}, union_by_name = true)
      WHERE ts IS NOT NULL AND ts >= {lo} AND ts <= {hi})"""


def _overlap_on(tol):
    """Closed-interval overlap, padded by a tolerance on each side.

    Both sides are derived from the same PCAP, so their clocks agree; the
    tolerance is for the boundary case, where a flow's last packet and the next
    flow's first are the same instant recorded twice, and for the fact that a
    Zeek conn with a NULL duration is a point interval that would otherwise
    match only an exactly equal timestamp.
    """
    return (f"o.ts <= z.ts + z.dur + {tol} AND z.ts <= o.ts_end + {tol}")


def _day_join(glob, day_parquet, lo, hi, tol):
    cats = ",\n             ".join(
        f"count(*) FILTER (WHERE o.attempted AND o.attempted_category = {c}) AS c{c}"
        for c in ATT_CATS)
    same_dir = ("o.src_ip = z.orig_h AND o.dst_ip = z.resp_h AND "
                "(z.proto = 'icmp' OR (o.src_port = z.orig_p AND o.dst_port = z.resp_p))")
    return f"""
    WITH {_map_cte()},
    {_zeek_cte(glob, lo, hi)},
    o AS (SELECT * FROM read_parquet({_lit(str(day_parquet))})
          WHERE artefact IS NULL AND ts IS NOT NULL)
    SELECT z.uid, o.base_label,
           coalesce(m.canonical, {_lit(UNMAPPED)}) AS oracle_class,
           count(*) AS n,
           count(*) FILTER (WHERE {same_dir}) AS n_dir,
           count(*) FILTER (WHERE o.attempted) AS n_attempted,
           {cats},
           count(*) FILTER (WHERE o.attempted
                              AND o.attempted_category NOT BETWEEN 0 AND 6) AS c_other,
           min(o.ts) AS ts_min, max(o.ts_end) AS ts_max
    FROM z JOIN o
      ON o.ep_lo = z.ep_lo AND o.ep_hi = z.ep_hi AND o.proto = z.proto
     AND {_overlap_on(tol)}
    LEFT JOIN lmap m ON m.k = o.label_key
    GROUP BY 1, 2, 3"""


# sum() over BIGINT widens to HUGEINT, which parquet stores as DOUBLE -- and a
# count that reads back as 3.0 is a count a check has to remember to round. Every
# aggregate here is cast back down where it is written.
ROLLUP = """
CREATE OR REPLACE TEMP TABLE per_uid AS
WITH pl AS (
  SELECT uid, base_label, oracle_class, sum(n) AS n, sum(n_dir) AS n_dir,
         sum(n_attempted) AS n_att, {csum}, sum(c_other) AS c_other,
         min(ts_min) AS lo, max(ts_max) AS hi
  FROM match_rollup GROUP BY 1, 2, 3),
pc AS (SELECT uid, oracle_class, sum(n) AS n FROM pl GROUP BY 1, 2),
byc AS (
  SELECT uid, sum(n) AS rows_total,
         arg_max(oracle_class, n) AS top_class, max(n) AS top_class_rows,
         list_sort(list_distinct(list(oracle_class))) AS oracle_classes
  FROM pc GROUP BY uid),
byl AS (
  SELECT uid, sum(n_dir) AS rows_dir, sum(n_att) AS rows_attempted,
         arg_max(base_label, n) AS top_label, max(n) AS top_label_rows,
         list_sort(list_distinct(list(base_label))) AS oracle_labels,
         {cout}, sum(c_other) AS att_cat_other,
         min(lo) AS ts_min, max(hi) AS ts_max
  FROM pl GROUP BY uid)
SELECT c.uid,
       CAST(c.rows_total AS BIGINT) AS oracle_rows,
       CAST(l.rows_dir AS BIGINT) AS oracle_rows_dir_agree,
       c.top_class AS oracle_top_class,
       CAST(c.top_class_rows AS BIGINT) AS oracle_top_class_rows,
       l.top_label AS oracle_top_label,
       CAST(l.top_label_rows AS BIGINT) AS oracle_top_label_rows,
       c.oracle_classes, l.oracle_labels,
       c.top_class_rows::DOUBLE / nullif(c.rows_total, 0) AS oracle_purity,
       CAST(l.rows_attempted AS BIGINT) AS oracle_attempted_rows,
       {ccols}, CAST(l.att_cat_other AS BIGINT) AS att_cat_other,
       l.ts_min AS oracle_ts_min, l.ts_max AS oracle_ts_max
FROM byc c JOIN byl l USING (uid)"""


def _rollup_sql():
    return ROLLUP.format(
        csum=", ".join(f"sum(c{c}) AS c{c}" for c in ATT_CATS),
        cout=", ".join(f"sum(c{c}) AS att_cat_{c}" for c in ATT_CATS),
        ccols=", ".join(f"CAST(l.att_cat_{c} AS BIGINT) AS att_cat_{c}"
                        for c in ATT_CATS))


def _flows_sql(glob, spans, lookback, tol, matched_only):
    """One row per Zeek flow inside the oracle's coverage, matched or not.

    Unmatched flows are kept because they ARE the coverage finding: a check that
    could only see the matches would have to take the denominator on trust, and
    the denominator is the number that decides whether any of the other numbers
    mean anything. --matched-only exists for when disk is the binding
    constraint, and is recorded in the report when used.
    """
    where = " OR ".join(f"(ts >= {a - lookback} AND ts <= {b + tol})" for a, b in spans)
    keep = ["uid", "ts", "duration", "capture", "proto", "label_class", "label_raw",
            "rule_id", '"id.orig_h"', '"id.resp_h"']
    return f"""
    CREATE OR REPLACE TEMP TABLE flows AS
    SELECT {', '.join('l.' + k for k in keep)},
           coalesce(p.oracle_rows, 0) AS oracle_rows,
           coalesce(p.oracle_rows_dir_agree, 0) AS oracle_rows_dir_agree,
           p.oracle_top_class, p.oracle_top_class_rows,
           p.oracle_top_label, p.oracle_top_label_rows,
           p.oracle_classes, p.oracle_labels, p.oracle_purity,
           coalesce(p.oracle_attempted_rows, 0) AS oracle_attempted_rows,
           {', '.join(f'coalesce(p.att_cat_{c}, 0) AS att_cat_{c}' for c in ATT_CATS)},
           coalesce(p.att_cat_other, 0) AS att_cat_other,
           p.oracle_ts_min, p.oracle_ts_max
    FROM (SELECT {', '.join(keep)} FROM read_parquet({_lit(glob)}, union_by_name = true)
          WHERE {where}) l
    {'JOIN' if matched_only else 'LEFT JOIN'} per_uid p ON p.uid = l.uid"""


def join(dataset, cache, lake, glob, out_json, out_parquet, tol=1.0,
         lookback=3600.0, matched_only=False):
    """Build the overlap join and write the per-uid artefact and its report."""
    staged = staged_files(cache, dataset)
    if not staged:
        src = ORACLE_SOURCES.get(dataset)
        hint = ("\n\n" + curl_command(src, Path(cache) / "archives" / src["archive"])
                + "\n\nthen: --stage") if src else ""
        sys.exit(f"nothing staged for {dataset} in {staged_dir(cache, dataset)}{hint}")

    con = lake.con
    con.execute("SET preserve_insertion_order = false")
    lake.sql(MATCH_TABLE)

    days, spans, oracle_rows, artefact_rows, unparsed = [], [], 0, 0, 0
    for path in staged:
        row = lake.one(f"""
            SELECT count(*), count(*) FILTER (WHERE artefact IS NOT NULL),
                   count(*) FILTER (WHERE ts IS NULL),
                   min(ts) FILTER (WHERE artefact IS NULL),
                   max(ts_end) FILTER (WHERE artefact IS NULL),
                   count(*) FILTER (WHERE artefact IS NULL AND proto = 'icmp')
            FROM read_parquet({_lit(str(path))})""")
        n, art, bad, lo, hi, icmp = row
        oracle_rows += int(n)
        artefact_rows += int(art)
        unparsed += int(bad)
        rec = {"day": path.stem, "rows": int(n), "artefact_rows": int(art),
               "unparsed_timestamp": int(bad), "icmp_rows": int(icmp),
               "ts_min": lo, "ts_max": hi, "joinable_rows": int(n) - int(art)}
        if lo is None or hi is None:
            rec["matched_oracle_rows"] = 0
            days.append(rec)
            continue
        spans.append((float(lo), float(hi)))
        z_lo, z_hi = float(lo) - lookback, float(hi) + tol
        lake.sql(f"INSERT INTO match_rollup {_day_join(glob, path, z_lo, z_hi, tol)}")
        # The mirror of coverage: oracle rows for which we have no flow at all.
        # A semi-join rather than a DISTINCT over the match, so it costs one
        # scan of an already-narrow parquet instead of a second full join.
        rec["matched_oracle_rows"] = int(lake.one(f"""
            WITH {_zeek_cte(glob, z_lo, z_hi)},
            o AS (SELECT * FROM read_parquet({_lit(str(path))})
                  WHERE artefact IS NULL AND ts IS NOT NULL)
            SELECT count(*) FROM o WHERE EXISTS (
              SELECT 1 FROM z WHERE o.ep_lo = z.ep_lo AND o.ep_hi = z.ep_hi
                AND o.proto = z.proto AND {_overlap_on(tol)})""")[0])
        days.append(rec)
        print(f"  · {path.stem:<28} {rec['joinable_rows']:>12,} joinable  "
              f"{rec['matched_oracle_rows']:>12,} matched")

    if not spans:
        sys.exit(f"every staged file for {dataset} has an unusable timestamp range")

    lake.sql(_rollup_sql())
    lake.sql(_flows_sql(glob, spans, lookback, tol, matched_only))

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    lake.sql(f"COPY flows TO {_lit(str(out_parquet))} "
             f"(FORMAT PARQUET, COMPRESSION ZSTD)")

    report = _report(lake, dataset, cache, staged, glob, days, spans, tol,
                     lookback, matched_only, oracle_rows, artefact_rows,
                     unparsed, out_parquet)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def _report(lake, dataset, cache, staged, glob, days, spans, tol, lookback,
            matched_only, oracle_rows, artefact_rows, unparsed, out_parquet):
    kinds = [{"artefact": r[0], "rows": int(r[1])} for r in lake.sql(
        f"SELECT artefact, count(*) FROM read_parquet("
        f"{_lit(str(staged_dir(cache, dataset) / '*.parquet'))}) "
        f"WHERE artefact IS NOT NULL GROUP BY 1 ORDER BY 2 DESC")]

    labels = [{"label": r[0], "canonical": r[1], "rows": int(r[2]),
               "attempted_rows": int(r[3])} for r in lake.sql(f"""
        WITH {_map_cte()}
        SELECT o.base_label, coalesce(m.canonical, {_lit(UNMAPPED)}), count(*),
               count(*) FILTER (WHERE o.attempted)
        FROM read_parquet({_lit(str(staged_dir(cache, dataset) / '*.parquet'))}) o
        LEFT JOIN lmap m ON m.k = o.label_key
        WHERE o.artefact IS NULL
        GROUP BY 1, 2 ORDER BY 3 DESC""")]

    total, matched, dir_agree, agree = lake.one("""
        SELECT count(*), count(*) FILTER (WHERE oracle_rows > 0),
               count(*) FILTER (WHERE oracle_rows_dir_agree > 0),
               count(*) FILTER (WHERE oracle_rows_dir_agree > 0
                                  AND oracle_top_class = label_class)
        FROM flows""")

    by_class = [{"label_class": r[0], "flows": int(r[1]), "matched": int(r[2]),
                 "matched_dir_agree": int(r[3]), "agree": int(r[4]),
                 "mean_purity": r[5]} for r in lake.sql("""
        SELECT label_class, count(*), count(*) FILTER (WHERE oracle_rows > 0),
               count(*) FILTER (WHERE oracle_rows_dir_agree > 0),
               count(*) FILTER (WHERE oracle_rows_dir_agree > 0
                                  AND oracle_top_class = label_class),
               avg(oracle_purity)
        FROM flows GROUP BY 1 ORDER BY 2 DESC""")]

    z_lo, z_hi = lake.one("SELECT min(ts), max(ts) FROM flows")
    o_lo, o_hi = min(a for a, _ in spans), max(b for _, b in spans)

    return {
        "oracle_version": 1,
        "meta": {
            "dataset": dataset, "generated_utc": _now(), "labelled_glob": glob,
            "tolerance_seconds": tol, "lookback_seconds": lookback,
            "matched_only": matched_only,
            "uid_parquet": portable(out_parquet),
            "uid_parquet_bytes": out_parquet.stat().st_size
            if out_parquet.exists() else None,
        },
        "provenance": provenance(dataset, cache, staged),
        "method": {
            "join_key": "unordered 5-tuple (endpoints sorted) + transport, "
                        "ICMP keyed on addresses only",
            "time_predicate": "closed-interval overlap of [ts, ts+duration] on "
                              f"both sides, padded by {tol}s",
            "direction": "carried as a boolean, never required",
            "unit": "one Zeek uid; oracle rows overlapping it are aggregated, "
                    "never counted as separate agreements",
            "not_comparable": "flow COUNTS. CICFlowMeter ran with a 120s active "
                              "timeout and chops long connections; Zeek does not. "
                              "A count difference is flow-construction semantics, "
                              "not a labelling error.",
        },
        "oracle": {
            "rows_total": oracle_rows, "artefact_rows": artefact_rows,
            "artefact_kinds": kinds, "unparsed_timestamp_rows": unparsed,
            "joinable_rows": oracle_rows - artefact_rows,
            "matched_rows": sum(d.get("matched_oracle_rows", 0) for d in days),
            "unmatched_rows": (oracle_rows - artefact_rows
                               - sum(d.get("matched_oracle_rows", 0) for d in days)),
            "ts_min": o_lo, "ts_max": o_hi,
            "by_day": days, "by_label": labels,
            "unmapped_labels": [x["label"] for x in labels
                                if x["canonical"] == UNMAPPED],
        },
        "join": {
            "zeek_flows_in_span": int(total), "zeek_flows_matched": int(matched),
            "zeek_flows_dir_agree": int(dir_agree),
            "zeek_flows_class_agree": int(agree),
            "coverage": round(matched / total, 9) if total else None,
            "direction_reversal_rate": round((matched - dir_agree) / matched, 9)
            if matched else None,
            "covered_spans": [[a, b] for a, b in spans],
            "zeek_ts_min": z_lo, "zeek_ts_max": z_hi,
            # Two clocks that disagree by hours produce zero coverage and no
            # error, which is the failure mode most likely to be misread as
            # "the oracle says we are wrong about everything".
            "clock_delta_seconds": round(float(o_lo) - float(z_lo), 3)
            if z_lo is not None else None,
            "by_class": by_class,
        },
    }


# --------------------------------------------------------------------- CLI

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--fetch", action="store_true",
                   help="download the archive (OFF by default; prints the curl "
                        "command instead)")
    p.add_argument("--stage", action="store_true",
                   help="unpack the archive and convert each capture day to parquet")
    p.add_argument("--stage-from", default=None,
                   help="stage from a .csv or a directory of .csv already unpacked")
    p.add_argument("--join", action="store_true",
                   help="build the overlap join against the labelled zone")
    p.add_argument("--cache-dir", default=str(CACHE))
    p.add_argument("--force", action="store_true", help="restage files already staged")
    p.add_argument("--ignore-errors", action="store_true",
                   help="skip unparseable CSV rows instead of failing the conversion")
    p.add_argument("--tolerance", type=float, default=1.0,
                   help="seconds of slack on the interval-overlap test")
    p.add_argument("--lookback", type=float, default=3600.0,
                   help="seconds before an oracle day to consider Zeek flows from")
    p.add_argument("--matched-only", action="store_true",
                   help="omit unmatched Zeek flows from the per-uid parquet")
    p.add_argument("--out", default=None)
    p.add_argument("--out-parquet", default=None)
    p.add_argument("--config", default=str(ROOT / "config.yml"))
    p.add_argument("--lake-root", default=None)
    p.add_argument("--no-s3", action="store_true")
    p.add_argument("--memory-limit", default="8GB")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--temp-dir", default=None)
    p.add_argument("--max-temp", default="20GB")
    return p.parse_args()


def _describe(dataset, cache):
    """The default action: say what is on disk and what would be fetched."""
    print(f"dataset    {dataset}")
    if dataset in NO_ORACLE:
        print(f"oracle     none published\n           {NO_ORACLE[dataset]}")
        return 0
    src = ORACLE_SOURCES.get(dataset)
    staged = staged_files(cache, dataset)
    if src:
        archive = Path(cache) / "archives" / src["archive"]
        size = archive.stat().st_size if archive.exists() else 0
        print(f"source     {src['url']}")
        print(f"licence    {LICENCE}")
        print(f"archive    {archive}")
        print(f"           {'present, ' + format(size, ',') + ' bytes' if size else 'absent'}"
              f"  (expected {src['bytes']:,})")
    else:
        print("source     no published archive registered for this dataset")
    print(f"staged     {len(staged)} file(s) in {staged_dir(cache, dataset)}")
    for p in staged:
        print(f"           {p.name}")
    j, u = oracle_paths(dataset)
    print(f"join       {'present' if j.exists() else 'absent'}  {j}")
    print(f"           {'present' if u.exists() else 'absent'}  {u}")
    if src and not (Path(cache) / 'archives' / src['archive']).exists():
        print(f"\nnothing here downloads by default. To fetch it yourself:\n\n"
              f"{curl_command(src, Path(cache) / 'archives' / src['archive'])}\n")
    return 0


def main():
    a = parse_args()
    cache = Path(a.cache_dir)

    if a.dataset in NO_ORACLE:
        print(f"{a.dataset}: no external oracle exists.\n{NO_ORACLE[a.dataset]}")
        return 0
    if not (a.fetch or a.stage or a.stage_from or a.join):
        return _describe(a.dataset, cache)

    cfg = yaml.safe_load(Path(a.config).read_text())
    root = (a.lake_root.rstrip("/") if a.lake_root
            else f"s3://{cfg['aws']['bucket']}")
    # Only --join reads the lake. Fetching a public archive over HTTP and
    # converting it to parquet on local disk need a DuckDB connection but no AWS
    # session, and demanding one turns an expired login into a spurious failure
    # of the one stage that could have run offline.
    require_s3 = a.join and not a.no_s3 and root.startswith("s3://")
    lake = Lake(cfg["aws"]["region"], a.memory_limit, a.temp_dir, a.threads,
                a.max_temp, require_s3=require_s3)

    if a.fetch:
        if a.dataset not in ORACLE_SOURCES:
            sys.exit(f"no archive registered for {a.dataset}")
        fetch(a.dataset, cache)

    if a.stage or a.stage_from:
        print(f"staging {a.dataset} -> {staged_dir(cache, a.dataset)}")
        stage(a.dataset, cache, lake.con, source=a.stage_from, force=a.force,
              ignore_errors=a.ignore_errors)

    if a.join:
        lbl = cfg["lake"].get("labelled", "labelled")
        glob = f"{root}/{lbl}/{a.dataset}/**/*.parquet"
        j, u = oracle_paths(a.dataset)
        out_json = Path(a.out) if a.out else j
        out_parquet = Path(a.out_parquet) if a.out_parquet else u
        print(f"joining {a.dataset} against {glob}")
        rep = join(a.dataset, cache, lake, glob, out_json, out_parquet,
                   tol=a.tolerance, lookback=a.lookback,
                   matched_only=a.matched_only)
        jn = rep["join"]
        print(f"\nzeek flows in span   {jn['zeek_flows_in_span']:,}")
        print(f"matched              {jn['zeek_flows_matched']:,} "
              f"({(jn['coverage'] or 0):.2%})")
        print(f"direction agrees     {jn['zeek_flows_dir_agree']:,}")
        print(f"class agrees         {jn['zeek_flows_class_agree']:,}")
        print(f"oracle rows unmatched {rep['oracle']['unmatched_rows']:,} of "
              f"{rep['oracle']['joinable_rows']:,} joinable")
        if rep["oracle"]["unmapped_labels"]:
            print(f"!! unmapped oracle labels: "
                  f"{', '.join(rep['oracle']['unmapped_labels'])}")
        print(f"\nreport -> {out_json}\nflows  -> {out_parquet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
