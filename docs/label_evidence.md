# Talos — label validation evidence log

Every claim below is a measurement or a primary-source quote, recorded so that
manifest corrections can be justified rather than asserted. Two independent
lines of evidence: the published CIC sources our manifests were transcribed
from, and the EDA profiles measured off our own labelled lake.

Nothing here has been acted on yet. Corrections belong in the manifests, and
labelling is re-run afterwards — the validator never edits a label.

Measured against EDA spec `a6832d31d1c2` (v2), identical across all three
profiles, so every cross-dataset number is on the same ruler.

---

## A. Source verification — what CIC actually publishes

Fetched from the three UNB dataset pages; the ICISSP 2018 paper was read as PDF
directly. Cross-checked against the DistriNet CNS2022 audits
(`intrusion-detection.distrinet-research.be/CNS2022/`).

### A1 — No CIC source states a timezone for any of the three datasets 🔴

The strings "UTC", "ADT", "AST", "Atlantic" and any numeric offset appear
**nowhere** on the CIC pages and nowhere in the ICISSP paper. All three of our
`utc_offset` values are inferences, not transcriptions.

This is the highest-consequence finding in the log: every label derives from a
wall-clock→epoch conversion, so a wrong offset moves every window in a dataset
together while leaving the labelled output looking complete. `sch.offset_calibration`
exists specifically to pin these empirically.

Related: an independent source converting CIC-DDoS-2019's local times to UTC
implies **UTC−3 on both days**, but 2018-12-01 is after DST ended (2018-11-04)
and should be AST (UTC−4). Our manifest uses −03:00 for day 1 and −04:00 for
day 2. One of the two is wrong; only the traffic can settle it.

### A2 — 2017 Infiltration has no consolidated `14:19–14:35` window 🔴

CIC publishes only:
```
Meta exploit Win Vista (14:19 and 14:20-14:21 p.m.) and (14:33 -14:35)
```
There is **no** summary line giving a single 14:19–14:35 block. Our manifest
comment claims there is, and that claim is false. As labelled, 14:22–14:32
(11 minutes) is attack on no documentation at all.

The manifest's *other* justification — that the real Vista↔Kali callback traffic
sits at ~14:28–14:30, inside the gap — may still hold. `sch.window_sweep` on the
three documented sub-windows is what should decide it.

Also: Thursday's Infiltration sub-times (`14:20–14:21`, `14:33`, `14:35`) appear
byte-identical in Friday's PortScan "firewall rules on" list, including the
degenerate `14:33 – 14:33` and `14:35 - 14:35`. That looks like copy-paste
contamination in CIC's own documentation. Treat both lists as low-confidence
around those minutes.

### A3 — 2018 second-stage infiltration IS labellable 🟠

Our manifest records "CIC never names the infected host". That is wrong. CIC
publishes the infected hosts in the **Victim** column — `172.31.69.24` (28-02)
and `172.31.69.13` (01-03) — and those same hosts perform the second-stage
internal scan. What CIC does not publish is the scan targets or a separate time
window for the phase.

DistriNet derived the phases as running to **18:39** and **19:38**, far past
CIC's published finish times of 14:40 and 15:37. Our windows therefore
under-cover the attack substantially.

This may rescue 2018's `infiltration` class, which is currently 8 flows.

### A4 — Confirmed transcription decisions

| Claim | Verdict |
|---|---|
| 2017 NAT chain `205.174.165.73 → 205.174.165.80 → 172.16.0.1` | **Supported** by the page and independently by ICISSP Figure 1 |
| 2017 Heartbleed `172.16.0.11` is a typo | **Confirmed** — the reply path *in the same paragraph* uses `172.16.0.1` |
| 2017 DDoS-LOIT attackers are `205.174.165.69/70/71` | **Supported**, verbatim |
| 2018 Table 2 duplicates the 01-03 14:00–15:37 Infiltration row | **Confirmed** — rows 4 and 5 are byte-identical |
| 2019 server IPs swapped vs the infrastructure table | Our correction stands; the table is internally contradicted by the captures |
| 2019 `PortMap` (table) vs `PortScan` (prose) | **`Portmap` is correct** — the dataset's own file naming uses it |
| 2019 capture dates are 2018-11-03 / 2018-12-01 | **Corroborated** independently; CIC never acknowledges it |

### A5 — 2019 prose swaps training and testing days 🔴

The prose calls the 12-attack set the *training* day and the 7-attack set the
*testing* day; the timing table and the actual capture dates both make 03-11
(7 attacks) the **first** day. Our manifest's parenthetical labels follow the
prose and are inverted. No effect on labels — we key on date and folder — but
the documentation is wrong.

### A6 — Other published 2018 defects (DistriNet)

- **DoS-SlowHTTPTest**: no successful attack exists in the dataset; the tool
  targeted **port 21**. See B1 — we reproduced this exactly.
- **DDoS-HOIC** started **6 minutes later** than documented.
- **2018-02-20**: some flows labelled `DDoS-LOIC-HTTP` are actually LOIC-UDP.
- **Heartleech** is listed in CIC's tooling table but never appears in the data.

---

## B. Measured against our own labelled lake

### B1 — 2018 `DoS-SlowHTTPTest` is 105,550 refused FTP connections 🔴

Four independently-computed populations agree **to the unit**:

| measure on 2018 `dos` | count |
|---|---|
| `resp_port` in bin [20,53) — i.e. port 21 | 105,550 |
| `conn_state = REJ` | 105,550 |
| `history = "Sr"` (SYN out, RST back) | 105,550 |
| flows matched by the `DoS-SlowHTTPTest` rule | 105,550 |

100% port 21, 100% REJ, 100% `service = NULL`, 100% `orig_bytes = 0`. REJ means
the victim answered the SYN with a RST. SlowHTTPTest works by holding half-open
HTTP sockets; a refused connection holds nothing. There is no HTTP, no port 80,
no payload, no socket.

**5.43% of 2018's `dos` class is labelled as an executed HTTP slow-DoS while
being refused FTP connections.** Independently confirms the published finding.

### B2 — `requires_payload` exempts SlowHTTPTest on a false premise 🔴

`taxonomy.yaml` excludes `DoS-SlowHTTPTest` from `requires_payload` reasoning
that slow-DoS "holds sockets open by sending almost nothing". The data says the
sockets were never opened.

Meanwhile `FTP-BruteForce` *is* in `requires_payload`. The two populations are
**indistinguishable across all 17 numeric and 10 categorical features** — same
REJ, same port 21, same `Sr` history, same zero payload. The only discriminator
is the calendar day. One is recorded as a failed attempt (`label_executed=false`),
the other as an executed denial of service (`n/a`).

The exemption launders 105,550 non-events into the executed-attack pool.

### B3 — 2019's benign class is ~90% attack 🔴

Four independent measurements:

**Distributional identity** — JS divergence, fixed bins:

| feature | 2019ben vs 2019ddos | vs 2017ben | vs 2018ben | 2017ben vs 2018ben |
|---|---|---|---|---|
| conn_state | **0.064** | 0.743 | 0.604 | 0.155 |
| service | **0.038** | 0.661 | 0.526 | 0.074 |
| resp_pkts | **0.020** | 0.715 | 0.642 | 0.165 |
| byte_ratio | **0.076** | 0.728 | 0.692 | 0.029 |

2019 benign is 11.6× closer to 2019 *attack* than to 2017 benign on
`conn_state`, 35× on `resp_pkts`. 2017 and 2018 benign agree closely — "benign"
is a coherent concept and 2019's copy is not in it.

**No feature separates the classes.** Largest `|rho|` with `label_binary` across
all 16 features is 0.197; 14 of 16 below 0.11. Verdicts: 13 of 16 features
"domain-biased", 3 dead, **0 transfers**.

**One-rule test.** Majority baseline (always "ddos") = 99.0597%. Best
single-feature rule = 99.14%. Maximum lift over a constant predictor:
**+0.08 percentage points.**

**The benign class is one burst inside an attack window.** A single 10-minute
bucket, `2018-12-01 15:40Z`, holds **556,201 flows = 81.6% of every benign flow
in the dataset**, 99.8% of them touching the attack victim. Top-10 buckets =
619,586 flows = **90.8% of all benign, 98.0% victim-involving**. That bucket sits
in the 4-minute gap between MSSQL closing (15:46Z) and NetBIOS opening (15:50Z).

**Genuine residue: ~65,600 flows**, not the ~110k `config.yml` asserts.
`resp_bytes_per_pkt` is the one feature where 2019 benign does not resemble 2019
attack (JS 0.606) and does resemble 2017/2018 benign (0.057/0.047). It is
defined for only 65,647 benign flows; the other 616,411 have `resp_pkts = 0` —
nobody answered.

### B4 — The clock recovers the label 🔴

`hour_utc` alone, as a plain categorical:

| dataset | scope | NMI(hour; class) | one-rule accuracy | baseline | lift |
|---|---|---|---|---|---|
| 2017 | attack classes | **0.861** | **95.31%** | 44.63% | **+50.7 pp** |
| 2018 | attack classes | 0.686 | 88.23% | 52.54% | +35.7 pp |
| 2019 | all classes | 0.187 | 99.08% | 99.06% | +0.02 pp |

In 2017 every attack class occupies a near-disjoint hour set. The labels are a
function of the schedule, because they *are* the schedule. Any evaluation must
exclude time-derived features before it means anything.

### B5 — Same class, different phenomena 🔴

Mean JS across 17 features:

| comparison | mean JS |
|---|---|
| `dos` 2017 vs `dos` 2018 — **same label** | **0.331** |
| `dos` vs `ddos` within 2018 — different labels | **0.138** |
| `ddos` 2017 vs `ddos` 2019 | **0.902** |
| `portscan` 2017 vs `ddos` 2019 — different labels, different datasets | 0.436 |

`dos` carried across datasets diverges **2.4× more** than `dos` vs `ddos` within
one dataset. 2017 `ddos` is `conn_state = RSTO` for 100.000% of its flows; 2019
`ddos` has RSTO at 0.000% and is `S0` 96.2%. Their `history` value sets share
**zero** strings.

A SYN flood and a SYN scan are the same flow record — which is why `portscan`
2017 sits *closer* to `ddos` 2019 than `ddos` 2017 does.

### B6 — 2017 `ddos` has one source IP 🟠

`distinct id.orig_h = 1`, `distinct id.resp_h = 1` across 95,683 flows. The
taxonomy defines `ddos` as "distributed / reflected-amplified". The manifest
explains the collapse (three attackers behind one NAT) but that does not make
the label true at flow level. The class is also a 4-way constant:
`orig_bytes` min = max = 20, `resp_bytes` p01…p99 all 11,595, `resp_port`
min = max = 80.

Separately: `spec.yaml` justifies dropping identity columns because "2019's
spoofed sources alone run to millions of distinct addresses". The measured value
is **3**. And `local_orig = true` for 100.000% of all 72.5M 2019 flows.

### B7 — `orig_bytes` is corrupt in the classes that make it look useful 🔴

| dataset | class | `orig_bytes.max` | `orig_ip_bytes.max` | `orig_pkts.max` | implied B/pkt |
|---|---|---|---|---|---|
| 2017 | portscan | 1,985,545,102 | 21,258 | 119 | **16,685,253** |
| 2017 | dos | 1,567,578,927 | 8,026 | 27 | **58,058,479** |
| 2018 | botnet | 658,893 | 3,037 | 41 | 16,071 |

Zeek's `orig_bytes`/`resp_bytes` are sequence-number-derived and wrap on RST and
scan traffic; `*_ip_bytes` are counted on the wire. A 1.99 GB payload in 119
packets is an overflow, not a measurement.

`orig_bytes` carries the **largest** label correlation in 2017 (−0.442) and the
largest attack-class NMI (0.835). The strongest single label signal in the
dataset rides on a corrupt column. Use `*_ip_bytes`, or gate on
`orig_bytes <= orig_ip_bytes`.

### B8 — 2018 capture spans overlap 🔴

The labelling MATCH predicate is time + IP only — there is **no capture
predicate** — so capture spans must be disjoint.

| capture | span (UTC) | duration |
|---|---|---|
| Friday-02-03-2018 | 2018-02-27 12:18 → 2018-03-03 00:39 | **3.51 d** |
| Friday-23-02-2018 | 2018-02-21 12:33 → 2018-02-23 23:46 | 2.47 d |
| Thursday-01-03-2018 | 2018-02-27 12:18 → 2018-03-01 23:40 | 2.47 d |

Every other capture is ~0.5 d, as its name implies. Five pairs overlap, worst by
**59.4 hours**. **Three captures share `ts_min` to the millisecond**
(`1519733906.348`).

Either the same wall-clock traffic is duplicated across captures — which inflates
benign counts and makes `capture` unusable as a split key — or `ts` carries
outliers a schedule-keyed labeller will silently mis-window.

### B9 — 2017 `portscan` contains completed sessions 🟠

The class is mostly right (REJ 83.1% + S0 13.3% = 96.4%). The residue is not:
3.48% (8,273 flows) in completed states, and Zeek identified a real application
protocol on 3,790 of them — **3,126 completed DNS lookups**, plus http, smb,
ssl, ftp, ssh, ntp, kerberos.

Mechanism: `Infiltration-PortScan` uses `victim_subnet: 192.168.10.0/24` with no
port or behaviour constraint, so every flow the infected host sent to the LAN in
that 41-minute window is labelled `portscan`. Since `portscan` is
single-dataset, this is 3.5% contamination on 100% of the class's evidence base.

### B10 — Class support

| class | 2017 | 2018 | 2019 | verdict |
|---|---|---|---|---|
| ddos | 95,683 | 1,373,323 | 71,851,781 | 98.0% from 2019 alone |
| dos | 190,343 | 1,945,058 | 0 | usable, but see B5 |
| brute_force | 6,462 | 286,006 | 0 | 97.8% from 2018 |
| botnet | 736 | 97,386 | 0 | effectively single-dataset |
| portscan | 238,039 | 0 | 0 | **single-dataset** |
| web_attack | 2,047 | 427 | 0 | **599 executed flows total** |
| infiltration | **25** | **8** | 0 | **not a class** |
| heartbleed | **1** | 0 | 0 | **not a class** |

Executed-evidence budget pooled across all datasets: `heartbleed` 1 flow,
`infiltration` 14, `web_attack` 599, `portscan` **0**.

Cross-domain claims are defensible for `dos`, `brute_force` and — with the B5
caveat — `ddos`. `heartbleed` and `infiltration` should be dropped or folded,
not evaluated.

### B11 — 2018 `infiltration` flows outlast their own window 🟠

Max duration 13,254.7 s (3.68 h) against a longest scheduled window of 5,820 s
(97 min) — a 2.28× overrun. All 8 flows: `conn_state` OTH 50% / S1 37.5% /
RSTR 12.5%, i.e. **not one completed connection**, on ports 31337 and 54751 —
and all 8 are marked `label_executed = true`.

### B12 — 2019 nulls are being read as zeros 🟠

9,454,727 `ddos` flows (13.2%) and 34,453 `benign` flows have NULL `duration`;
the spec coalesces them to 0. **100% of the "duration == 0" mass in 2019 is
actually unmeasurable**, not instantaneous. Same for `orig_bytes`/`resp_bytes`.

### B13 — 2019 `ddos` merges orthogonal mechanisms 🟠

All 19 rules collapse to one class: UDP reflection (9 vectors), a TCP SYN flood,
and an L7 `WebDDoS` of 348 flows. `orig_bytes = 0` for 17,766,391 flows (24.7%),
impossible for UDP amplification where every reflected packet carries payload —
that is the SYN-flood share bleeding in. Stray services inside the class:
`syslog` 41,015, `dns` 2,441, `ssh` 1,794, `http` 1,538, plus `geneve`/`vxlan`/
`ayiya` tunnels.

---

## B14 — First independent cross-check against the DistriNet oracle (2017) ✅

Staged 2026-07-31 from `CICIDS2017_improved.zip` (343,549,013 bytes, byte count as
published). 2,099,976 rows across five capture days, 1,510 CICFlowMeter artefact rows
excluded, **zero unparsed timestamps**. All 27 distinct label spellings in the real data
normalise into 16 keys, **100% mapped** to canonical classes, zero unmapped.

Aggregate comparison, ours against theirs (their `- Attempted` rows split out, since we do
not model that axis):

| class | ours | oracle all | oracle executed | vs executed |
|---|---|---|---|---|
| benign | 1,585,846 | 1,581,058 | 1,581,058 | +4,788 (0.3%) |
| **botnet** | **736** | 4,803 | **736** | **±0 — exact** |
| brute_force | 6,462 | 6,972 | 6,933 | −471 |
| ddos | 95,683 | 95,144 | 95,144 | +539 (0.6%) |
| dos | 190,343 | 177,510 | 171,634 | +18,709 |
| heartbleed | 1 | 11 | 11 | −10 — **see below** |
| infiltration | 25 | 81 | 36 | −11 |
| portscan | 238,039 | 230,831 | 230,831 | +7,208 (3%) |
| web_attack | 2,047 | 2,056 | 104 | totals agree to 0.4% |
| **TOTAL** | 2,119,182 | 2,098,466 | 2,086,487 | 0.98% apart |

Two independent extractions of the same PCAPs, by different tools and different teams,
landing within 1% overall. **Botnet matches exactly at 736 executed flows.**

### The heartbleed "discrepancy" is a CICFlowMeter artefact, and we are right

Ours: 1 flow. Theirs: 11. Inspecting them settles it — all eleven share one 5-tuple and one
source port (45022), and run as consecutive 119.3-second slices with a 25.1s remainder:

```
18:12:15  119.3s   18:22:16  119.3s
18:14:15  119.3s   18:24:17  119.3s
18:16:16  119.3s   18:26:17  119.3s
18:18:16  119.3s   18:28:17  119.3s
18:20:16  119.3s   18:30:17  119.3s
                   18:32:18   25.1s
total 1,217.8s = 20.3 min = the Heartbleed window
```

They are one connection cut eleven ways by CICFlowMeter's 120-second **active** timeout.
Zeek uses inactivity timeouts and keeps the session whole. Our count is the correct one.

This is the adjudication table earning its place: *flow count differs → indicts neither*.
Read naively, "we have 1, the corrected dataset has 11" looks like we missed 10 attack
flows. It is the opposite.

The same mechanism explains most of the remaining gaps — `web_attack` totals agree to 0.4%
once their `- Attempted` axis is separated, and `portscan` running 3% higher for us is the
direction Rosay et al. predict, because Zeek correctly splits RST-then-SYN sessions that
CICFlowMeter merges.

**Caveat:** this is an aggregate comparison. The per-flow interval-overlap join
(`orc.*` checks) needs the labelled zone from S3 and has not run yet.

## C. Actions

Manifest corrections (P2), each traceable to evidence above:

1. Demote all three `utc_offset` values to empirically-derived; pin with
   `sch.offset_calibration` before anything else. **[A1]**
2. Replace 2017's contiguous Infiltration span with the three documented
   sub-windows; let the sweep decide whether the gap is justified. Correct the
   false justification comment. **[A2]**
3. Add 2018 second-stage infiltration rules using the published victim hosts as
   scanners; pin windows by sweep rather than copying DistriNet. **[A3]**
4. Fix 2019's inverted training/testing day labels. **[A5]**
5. Constrain `Infiltration-PortScan` beyond a bare subnet match, or accept and
   document 3.5% contamination. **[B9]**
6. Revisit `requires_payload`: SlowHTTPTest's exemption rests on a false
   premise. **[B2]**
7. Declare quarantine regions (P3) for 2019's inter-window residue and 2018's
   infiltration windows. **[B3, A3]**

Downstream constraints, not manifest edits:

8. Never use `orig_bytes`/`resp_bytes` as features; use `*_ip_bytes`. **[B7]**
9. Exclude `hour_utc` and time-derived features from any evaluation. **[B4]**
10. Do not evaluate `heartbleed` or `infiltration`. **[B10]**
11. Resolve 2018's capture overlaps before using `capture` as a split key. **[B8]**
