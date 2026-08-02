# Talos — Ground-Truth Label Validation

**A complete account of the programme: why it was undertaken, what was built, what was found,
what can and cannot be claimed, and what follows.**

Report date 2026-07-31 · validation module v1 · policy v1 · taxonomy `c1730fc59e18`
Manifests: 2017 `d65451d1dd2e` · 2018 `dbb041f7101b` · 2019 `b95b939d549a`

---

## Executive summary

Talos labels network flows by matching each flow's start time and endpoints against attack
schedules transcribed from CIC's published documentation. This is **weak supervision from a
document, not measurement**. Every claim the project makes — cross-domain transfer, per-class
recall, the decision to hold 2019 out — rests entirely on those labels being right, and until now
nothing had tested whether they were.

We built a seven-tier validation module (57 checks, ~10,800 lines), verified the instrument against
deliberately planted faults, and then verified our labels against an independently produced
corrected label set.

**The headline: for CIC-IDS-2017 our labels agree with an independent hand-labelling at 99.34%
across 2,107,787 flows, at 99.76% coverage, with a measured clock delta of 0.0 seconds.**

Three findings matter more than that number:

1. **The dominant error mode is inherited, not introduced.** Of the ~8,570 attack flows misfiled as
   benign, 6,276 sit in windows CIC's own documentation is independently known to have got wrong. Our
   transcription was faithful; the source was not.
2. **CIC-DDoS-2019's benign class is roughly 90% attack traffic.** It cannot serve as a
   false-positive baseline, which is what `config.yml` currently assumes it can do.
3. **No canonical class survives the cross-dataset coherence test.** `dos` in 2017 and `dos` in 2018
   are further apart than `dos` and `ddos` are *within* 2018. This is the finding with the largest
   consequence for the project's design, and it is about the taxonomy rather than about accuracy.

---

# Part I — Why this exists

## 1.1 The failure mode we were exposed to

Schedule-derived labelling fails **silently**. There is no exception, no error, no obviously wrong
output. A window transcribed from the wrong row, a timezone assumed rather than read, a NAT alias
omitted, an attack that was scheduled but never actually ran — each produces a complete-looking
labelled dataset. The first symptom is a model that scores 0.99 and transfers nothing.

That risk is not hypothetical for this project. Before any validation code existed, four defects
were already visible in the repo's own outputs:

| Dataset | Defect | Evidence |
|---|---|---|
| 2019 | ~613k attack-tail flows sitting in the benign class; the largest single burst is 556,201 flows in the four minutes between MSSQL closing and NetBIOS opening, 99.8% victim-directed | `reports/labelling_cic-ddos-2019.json` |
| 2018 | FTP-BruteForce is 192,294 of 192,300 empty connections — corroborated by the *absence* of `ftp.log` in 2018 entirely, because Zeek never instantiated an FTP analyser | labelling report + lake inventory |
| 2018 | Infiltration is 8 flows across ~4.5 scheduled hours | labelling report |
| 2017 | Web attacks are mostly connection attempts: BruteForce 1,213/1,356, XSS 652/679 | labelling report |

## 1.2 Two structural gaps found while planning

**`label_flows.py` could not run at all.** After a move to `src/preprocess/label/`, `parents[2]`
resolved to `src/` rather than the repo root, so `MANIFESTS` pointed at a directory that has never
existed, and the Makefile still called the pre-move path. **The stage that produces every label in
the project was broken.** This had to be fixed before anything else could proceed.

**No independent evidence existed.** Zeek extraction ran as bare `zeek -C -r <pcap>` with no site
policy, so `notice.log` was never produced anywhere in the lake — no scan detection, no brute-force
detection, no heartbleed detector. The strongest channel for validating schedule-derived labels had
never been generated.

## 1.3 What the literature told us to expect

- **Published CIC schedules contain errors.** Engelen, Rimmer & Joosen (WTMC 2021) and the DistriNet
  CNS2022 audits document mislabelled windows, attacks that never executed, and attacks entirely
  absent from CIC's labels.
- **The extractor injects artefacts.** Rosay et al. (ICISSP 2022) show CICFlowMeter — used to build
  the published CSVs — produces artefacts in *every* dataset it generated, including ARP frames
  misparsed as IPv4 and a TCP-termination bug that shreds one session into several flows.
- **Correct labels can still be worthless.** Arp et al. (USENIX Security 2022) name spurious
  correlations as a pervasive security-ML pitfall, with a network-intrusion example — a model that
  learns an attacker's IP range. Jacobs et al. (CCS 2022) demonstrate it on CIC-IDS-2017 itself:
  models separate DDoS traffic using **two bits of the TTL field**, because the attacker ran Kali
  (TTL 64) and the victim Windows (TTL 128).
- **Noise estimates need calibrating.** Confident Learning (Northcutt et al., JAIR 2021) assumes
  class-conditional noise. Schedule-derived noise is not — whether a flow is mislabelled depends on
  its own features, such as whether it straddled a boundary.

## 1.4 Source verification: what CIC actually publishes

Before validating labels against traffic, we verified the manifests against their sources — the CIC
dataset pages and the ICISSP 2018 paper, read directly.

| # | Finding | Severity |
|---|---|---|
| A1 | **No CIC source states a timezone for any of the three datasets.** "UTC", "ADT", "AST" and any numeric offset appear nowhere on the pages or in the paper. All three `utc_offset` values were inferences presented as transcriptions. | 🔴 |
| A2 | **2017's Infiltration `14:19–14:35` span does not exist in any source.** CIC publishes only `14:19`, `14:20–14:21`, `14:33–14:35`. Our manifest claimed the page "summarises this as a single block" — it does not. | 🔴 |
| A3 | **2018's second-stage infiltration IS labellable.** CIC publishes the infected hosts in the *Victim* column. What it withholds is the scan targets and a separate window. Our manifest said the opposite. | 🟠 |
| A5 | 2019's prose swaps training/testing days; the timing table and capture dates disagree with it. | 🔴 |
| A6 | DistriNet: 2018's `DoS-Slowhttptest` does not exist in the data (the tool hit port 21); `DDoS-HOIC` starts 6 minutes late; 2018-02-20 LOIC-UDP flows are labelled LOIC-HTTP. | 🟠 |

**Why A1 matters most.** Every label derives from a wall-clock→epoch conversion. A wrong offset
moves *every window in a dataset together*, while leaving the output looking complete. This drove
the design of `sch.offset_calibration`.

---

# Part II — What we built and why

## 2.1 Design principles

Five commitments, each chosen because the obvious alternative fails in a specific way.

**1. Validation never mutates a label.** Corrections go into the *manifest*, then labelling is
re-run. Labels stay schedule-derived, and every correction is a reviewable YAML diff carrying its
evidence. The alternative — patching labels directly — would make the labels un-derivable from any
document, destroying the property that makes them auditable at all.

**2. "Did not run" ≠ "ran and found nothing".** A check whose input is missing is recorded as
skipped *with a reason*. A dependency that fails to import produces an INFO finding, not silence.
The alternative reads a missing `scikit-learn` as a clean bill of health.

**3. Every finding carries its own reproduction query.** A validation module that cannot be audited
is just another unverified assertion in the chain being verified.

**4. Thresholds live in policy, not in code.** `policy.yaml` holds all 71, so a reader can disagree
with a judgement without recompiling the evidence. Checks always report the raw number beside the
verdict.

**5. Unadjudicable flows are quarantined, not guessed.** This is *how* 99–100% is reached — by
declining to pretend about the minority that cannot be verified, and reporting coverage alongside
accuracy.

## 2.2 Architecture

```
src/validate/
  finding.py       Finding dataclass + ordered Severity — the structured-QA abstraction the repo lacked
  registry.py      @check decorator, capability declaration, crash isolation, pkgutil auto-discovery
  runner.py        CLI orchestration; probes which logs exist; builds one shared Context
  policy.yaml      71 thresholds + gate policy
  expectations.yaml 892 lines of declarative per-class behavioural bands
  checks/          integrity · schedule · behaviour · corroborate · noise · probes · cross · artefact · quarantine · oracle
  oracle.py        external ground-truth staging + interval-overlap join engine
  sample.py        stratified certification sample with evidence bundles
  score.py         precision, Wilson intervals, Cohen's kappa, coverage
  render.py        offline self-contained HTML
  gate.py          exit-code policy
tests/
  build_fixture.py synthetic lake with planted faults
  test_validate.py the negative control
```

**Reuse over reimplementation.** `src.common.lake.Lake` (the only credential helper with memory and
spill guard rails), `src.eda.spec.Spec` (feature definitions on the same ruler as the EDA reports),
`src.eda.render` (identical visual system), `src.eda.compare.js_divergence` (so a divergence quoted
here and one quoted in an EDA report are the same number computed the same way).

## 2.3 The seven tiers

| Tier | n | Question | Cost |
|---|---|---|---|
| T0 integrity | 12 | Does the labelled zone hold together? | one scan |
| T1 schedule | 15 | Is the schedule we labelled from the one that ran? | one scan (see 2.4) |
| T2 behaviour | 4 | Does traffic behave as the label claims? | two scans |
| T3 corroboration | 13 | What do sources that never saw the schedule say? | joins |
| T4 noise | 5 | How much label noise remains? | sampled |
| T5 probes | 4 | Is the label about the traffic or the testbed? | sampled |
| T6 cross-dataset | 4 | Does a class mean the same thing everywhere? | offline |

## 2.4 The central measurement

`sch.window_sweep` is the strongest single check, and its design is what makes the tier affordable.

A correct attack window separates two traffic regimes — loud inside, quiet outside. So rather than
asking "how many flows did this rule match", it asks **"does the traffic change at the boundary the
schedule asserts?"** That is measurable without knowing anything about the attack.

The naive implementation re-scans the lake once per candidate window shift — sixty scans of 72.5M
flows. Instead, one scan buckets every endpoint-matching flow by its offset from the rule's start
and end at 10-second granularity; **every candidate shift is then a cumulative sum over that small
table, computed offline.** Sixty offsets cost one pass, not sixty.

This was prototyped against synthetic data before being written against the lake: 249k flows, a
window deliberately misaligned by 300 seconds, and the sweep located both true edges exactly and
quantified 121,800 flows falling outside the claimed window.

## 2.5 Engineering defects found and fixed along the way

| Defect | Why it mattered |
|---|---|
| `label_flows.py` unrunnable (`parents[2]`, wrong manifests path, stale Makefile) | The labelling stage itself was broken |
| `load_checks()` used a hand-maintained import list | `artefact.py` existed and **never ran** — a check that reports nothing is indistinguishable from one that finds nothing. Replaced with `pkgutil` discovery |
| `oracle.py` demanded an AWS session for `--fetch`/`--stage` | An expired login broke the one stage that needed only HTTP and local disk |
| 23 thresholds lived only as code defaults | Violated the module's own stated principle; all now in `policy.yaml` |
| Two unbounded SQL aggregates in the oracle checks | Would have exhausted memory on a 2018-sized run |
| Two false positives in `sch.offset_calibration` | Caught by the positive control — see §3.2 |

---

# Part III — How we verified the instrument

A validation module that reports nothing is indistinguishable from a broken one. Three controls.

## 3.1 Negative control — planted faults

`tests/build_fixture.py` builds a synthetic lake and runs it through the **real** labelling stage
and the **real** validator. Nothing is mocked; only the S3 bucket is swapped for a directory. Seven
faults are planted, each designed to trip exactly one check:

```
flood already running before the window opens    sch.window_sweep         FOUND
flood still running after the window closes      sch.window_sweep         FOUND
attack tail left in the benign class             sch.boundary_residue     FOUND
window quiet for 20 of its 30 minutes            sch.intra_window_gaps    FOUND
attacker arrives from an undeclared address      sch.endpoint_coverage    FOUND
rule scheduled against a host with no traffic    sch.rule_yield           FOUND
the same flow ingested twice                     int.uid_unique           FOUND
```

Detection was numerically exact — the 20-minute gap, the 60,000-flow residue, the 97.7% unexplained
peers.

## 3.2 Positive control — the rule that must stay silent

One rule in the fixture is correctly scheduled and must produce **no findings**. This is the
load-bearing assertion, and it earned its place immediately: it caught **two genuine false
positives** in `sch.offset_calibration` — one proposing a window "correction" worth a *single flow*,
another sliding a window onto a neighbouring attack's traffic. Both were fixed by gating on material
gain and excluding overlap with other rules.

## 3.3 Benign control — inside the corroboration metric

`cor.corroboration_rate` measures benign flows alongside attack classes. This turned out to matter
enormously (§4.4): the control revealed the metric is far weaker than its name suggests.

---

# Part IV — What we found

## 4.1 Method for the external oracle

The DistriNet CNS2022 corrected label set is the same PCAPs, relabelled by hand after forensic
examination. Retrieved 2026-07-31: `CICIDS2017_improved.zip`, 343,549,013 bytes — the exact
published size. 2,099,976 rows across five capture days, 1,510 CICFlowMeter artefact rows excluded,
**zero unparsed timestamps**. All 27 distinct label spellings normalise into 16 keys, **100% mapped**.

**The join cannot be an equality join.** CICFlowMeter uses a hard 120-second *active* timeout; Zeek
uses protocol-specific *inactivity* timeouts. One Zeek flow maps to N oracle rows. Direction can also
be reversed — CICFlowMeter fixes direction from the first packet it sees, Zeek from the SYN.

So the join is on the **unordered 5-tuple plus interval overlap**, carrying a `direction_agrees`
flag. Adjudication rules were fixed **before** looking at results:

| Disagreement | Indicts |
|---|---|
| Flow count differs | Neither — flow-construction semantics |
| One Zeek flow ↔ many oracle rows | Neither — timeout semantics |
| Oracle row is an `8.0.6.4` / protocol-0 artefact | Theirs — CICFlowMeter parsing bug |
| Class clash on an overlapping, direction-agreeing match | **Investigate — the only diagnostic case** |
| We say benign, oracle says attack, in a known-mislabelled window | **Ours** — inherited CIC error |
| We say attack, oracle says `- Attempted` | Neither — we lack their second label axis |

## 4.2 Result: 2017 agreement

| Metric | Value |
|---|---|
| Oracle coverage | **99.76%** (2,114,093 / 2,119,182) |
| **Class agreement** | **99.34%** (2,093,830 / 2,107,787) |
| Clock delta | **0.0 s** |
| Direction reversal | 0.30% |
| Oracle rows unmatched | **4** of 2,098,466 |

Per class: `ddos`, `brute_force`, `web_attack`, `botnet`, `infiltration`, `heartbleed` all **100.0%**;
`dos` 99.999%; `benign` 99.46%; `portscan` 97.71%.

**The clock delta of 0.0s independently confirms 2017's UTC offset** — the inference that no CIC
source could support (A1). Two routes agree: `sch.offset_calibration` returns zero findings, and the
oracle join agrees to the second.

## 4.3 Result: the errors are inherited

`orc.missed_attacks` (CRITICAL) — 3,475 benign flows are genuine attacks per the oracle, plus 5,095
attempt-only. **6,276 fall in windows CIC is documented to have got wrong:**

| Oracle label | Flows | Known defect |
|---|---|---|
| Botnet | 4,228 | Ares failed-C2 tail running to 20:01:21Z |
| Infiltration – Portscan | 2,026 | NMAP scan starting 17:33:30Z, before the published time |
| Infiltration | 22 | "Cool Disk – MAC" — absent from CIC's original labels entirely |

The module rediscovered these **independently**. It was given window definitions to test against but
not told which flows to expect.

## 4.4 Result: the corroboration metric is weak, and its own control proved it

Benign corroborates at **92.5%** — *higher* than `dos` (89.2%), and vastly higher than `portscan`
(1.6%). The metric is largely measuring **whether a flow appears in any application-layer log**,
which benign traffic does by default.

This is reported as a construct limitation rather than quoted as a result. It is currently a coverage
statistic wearing the name of a correctness one.

## 4.5 Result: a worked example of why adjudication rules matter

Heartbleed: ours **1** flow, oracle **11**. Read naively, we missed ten attacks.

All eleven oracle rows share one 5-tuple and one source port (45022), running as consecutive
119.3-second slices with a 25.1s remainder — totalling 1,217.8s, exactly the 20-minute Heartbleed
window.

```
18:12:15  119.3s      18:22:16  119.3s
18:14:15  119.3s      18:24:17  119.3s
18:16:16  119.3s      18:26:17  119.3s
18:18:16  119.3s      18:28:17  119.3s
18:20:16  119.3s      18:30:17  119.3s
                      18:32:18   25.1s
```

They are **one connection cut eleven ways by CICFlowMeter's 120-second timeout.** Our count is
correct. Without the pre-registered adjudication table this would have been logged as ten missed
attacks.

## 4.6 Result: structural integrity holds

All row-parity, uid-uniqueness, null, timestamp, capture-disjointness, label-wellformedness and
provenance checks pass on 2017. **44 checks ran, zero errors.**

Two data properties surfaced:

- **`int.bytes_physically_possible`** — 813 flows (0.038%) report more payload bytes than wire bytes,
  concentrated in `dos` (0.38%). Zeek derives `orig_bytes` from TCP sequence numbers, which wrap on
  RST-heavy traffic. **Consequence: never use `orig_bytes`/`resp_bytes` as features; use `*_ip_bytes`.**
  This matters because `orig_bytes` carries the *largest* label correlation in 2017 (−0.442) — the
  strongest apparent signal rides on a corrupt column.
- **`int.null_not_zero`** — `duration`, `orig_bytes`, `resp_bytes` are null for 8.1% of `dos` flows
  and coalesced to zero by the EDA spec, merging "instantaneous" with "unmeasurable".

## 4.7 Result: window fidelity is partially refuted

| Finding | Scale |
|---|---|
| `DoS-Slowhttptest` continues past its scheduled end | **26,038 flows** |
| `DoS-Slowloris` continues past its scheduled end | 3,834 flows |
| `SSH-Patator` continues past its scheduled end | 448 flows |
| `Infiltration` 15:04–15:45 idle | 42 of 41 scheduled minutes |
| `PortScan` 13:55–14:36 idle | 31 minutes |
| `Heartbleed` matched | **1 flow (0.05/min)** |

`sch.rule_ambiguity` finds one rule pair overlapping in time *and* endpoints **across different
classes** — a flow there is assigned by manifest ordering, not evidence.

## 4.8 Result: behavioural conformance is partially refuted

`beh.executed_ratio` reproduces the Engelen finding independently, from our own extraction:

| Attack | Flows | Never sent a byte |
|---|---|---|
| `WebAttack-XSS` | 679 | **652 (96.0%)** |
| `WebAttack-BruteForce` | 1,356 | **1,213 (89.5%)** |
| `Infiltration` | 25 | 19 (76.0%) |

These carry an attack label while carrying no attack.

## 4.9 Result: the cross-dataset taxonomy is refuted

A canonical class should resemble itself across datasets more than two *different* classes resemble
each other within one. **Every shared class fails.**

| Class | Same-class JS across datasets | Nearest different-class JS within a dataset | Ratio |
|---|---|---|---|
| `ddos` (2019↔2018) | 0.789 | 0.153 | **5.15×** |
| `infiltration` (2017↔2018) | 0.671 | 0.153 | 4.38× |
| `brute_force` (2017↔2018) | 0.539 | 0.153 | 3.52× |
| `web_attack` (2017↔2018) | 0.519 | 0.153 | 3.39× |
| `ddos` (2019↔2017) | 0.850 | 0.244 | 3.49× |
| `dos` (2017↔2018) | 0.323 | 0.153 | 2.11× |

**`xds.benign_baseline` (CRITICAL)** — CIC-DDoS-2019's benign class sits closer to its own `ddos`
(JS 0.106) than to 2018's benign (JS 0.450). Roughly 613k of its 682k "benign" flows are attack tails
in the gaps between scheduled floods.

---

# Part V — Accuracy: what we can and cannot claim

## 5.1 What is established

> **For CIC-IDS-2017, our labels agree with an independently produced hand-labelling at 99.34% over
> 2,107,787 direction-agreeing matched flows, at 99.76% coverage, with the UTC offset independently
> confirmed.**

That is a measurement, not an assertion. It is reproducible from `reports/validation/` and every
number carries the query that produced it.

## 5.2 What that number is *not*

**It is not a per-class precision with an interval.** Aggregate agreement conceals class-level
fragility: `heartbleed` agrees at 100% — on **one flow**. `infiltration` agrees at 100% on **25**.
Agreement must always be read beside its support.

**It is not proof.** The oracle is itself a human judgement over the same PCAPs. Agreement is strong
evidence; it is not ground truth handed down.

**It is not coverage-adjusted.** 2017 currently has zero quarantine regions, so its coverage is 100%.
2018 and 2019 have 19 declared regions between them, and their accuracy figures **must** be quoted
with coverage: *"99.4% precise on 97.1% of flows"* is honest; the first half alone is not.

**It says nothing about 2018 or 2019.** Those runs were in flight when this report was written.
Critically, **no corrected label set exists for CIC-DDoS-2019** — the dataset where our own analysis
finds the most severe problems is the one with no independent check available.

## 5.3 The route to a certified figure

Built and verified on synthetic data, not yet drawn on real data:

1. `make validate-sample DATASET=cic-ids-2017` — stratified sample, ~400/class for ±2–3% at 95%,
   ~2000 for ~±1%. Each flow carries an evidence bundle (matched rule and window, conn state,
   history, plus any http URI / ssl SNI / dns query / ssh auth / ftp reply code) so most adjudicate
   in seconds without returning to the lake.
2. Adjudicate the `verdict` column.
3. `make validate-score` — per-class precision with **Wilson** intervals (not the normal
   approximation, which misbehaves at proportions near 1, exactly where we expect to be), Cohen's
   kappa with observed and expected agreement reported separately, and coverage.

The scorer was exercised end-to-end on the fixture and produces a per-class **"certified at 99%
floor: yes/no"** verdict.

---

# Part VI — Caveats and threats to validity

## 6.1 Internal validity

**`sch.endpoint_coverage` is miscalibrated.** It fires on 17 of 18 rules at 94–100% "unexplained"
traffic. On inspection this is largely a threshold artefact: the victim web server carries ordinary
load during attack windows, so most in-window traffic legitimately is *not* the attack. The check
needs a per-rule out-of-window baseline. **Reported rather than quietly suppressed**, because a
validation module that hides its own noisy checks is not trustworthy about anything else.

**The oracle join reuses our own endpoint predicate.** `sch.*` checks import `_victim_ok` and
`rules_sql` from `label_flows.py` so the sweep measures exactly the population that was labelled. If
that predicate is wrong, the sweep inherits the error. `int.relabel_determinism` has the same
property and its docstring says so explicitly — it is a *reproducibility* check, not a correctness one.

## 6.2 Construct validity

**Corroboration rate measures log presence, not attack evidence** (§4.4), established by its own
benign control.

**Confident learning's assumption is violated here.** It assumes class-conditional noise;
schedule-derived noise depends on a flow's own features. `noi.cl_calibration` exists to measure
whether the estimator still recovers planted errors, and its pass verdict explicitly limits what it
licenses.

## 6.3 External validity

**No oracle for 2019.** Nobody has published a corrected label set for CIC-DDoS-2019. This is a gap
in the field, not just in this project — and it is why `noi.cl_calibration` matters: calibrate the
estimator where the oracle can check it, then use it where nothing can.

**Licence.** The DistriNet archive publishes no licence, terms of use, or redistribution clause. It
is treated as all-rights-reserved: cited and used, never redistributed. The provenance record in
`oracle_cic-ids-2017.json` states this explicitly.

## 6.4 Irreducible limits

**Behaviourally identical benign traffic.** DistriNet document that 2017's DDoS LOIC-HTTP produces
flows *outside* the attack window that are byte-identical to attack flows. No schedule-based method
can separate them. **This is a floor on achievable label accuracy, not a defect to fix.**

**Encrypted payloads.** 2017's Infiltration Dropbox downloads cannot be adjudicated from the wire.
DistriNet made a judgement call and labelled them benign; ours may differ.

## 6.5 Not yet measured

**Tiers 4 and 5 have never run**, because `scikit-learn` is declared in `requirements.txt` but not
installed. Both emit `INFO: check did not run` rather than passing silently. Given Jacobs et al.'s
TTL result on this exact dataset, **the shortcut hypothesis (H6) should be regarded as likely
refuted until measured**, not as an open question.

**2018 and 2019 validation are incomplete.** Runs were in flight at report time. Both also require a
re-label first, because the P2 manifest corrections changed their content hashes.

---

# Part VII — What this means for Talos

## 7.1 The good news

The labelling method **works**. 99.34% agreement with an independent forensic labelling, on a
dataset with documented schedule errors, using a completely different extractor, is a strong result.
The manifests are careful, the NAT aliases are right, the offset is right, and the closed-world
assumption holds for the great majority of flows.

The infrastructure now exists to *prove* that rather than hope it. Every future manifest edit is
regression-tested against the oracle, and every re-label is provenance-checked.

## 7.2 The problem that changes the project's design

**The taxonomy does not survive contact with the data.** Talos exists to test whether a detector
trained on one testbed transfers to another. That question is only meaningful if `dos` means the same
thing in both. It does not — and by a wide margin.

This has a concrete consequence: **a cross-domain result about `dos` or `ddos`, as currently defined,
would measure a rename rather than a detector.** Three options, in increasing order of ambition:

1. **Narrow the classes** until they cohere — e.g. split `ddos` into reflection/amplification vs
   volumetric TCP flood, which are different phenomena sharing a label today.
2. **Restrict the claim** to classes that pass the coherence test, and report the rest as
   single-domain observations.
3. **Make the incoherence the finding.** "Canonical attack taxonomies used across CIC datasets do not
   denote consistent phenomena, and cross-domain results built on them are not interpretable" is a
   defensible, publishable claim, and this module is the evidence for it.

Option 3 is the strongest scientific position, and it costs nothing that the other two do not.

## 7.3 The 2019 holdout decision needs revisiting

`config.yml` justifies holding 2019 out on the grounds that its benign class provides a
false-positive baseline while 2017/2018 supply the rest. **Its benign class is ~90% attack traffic.**
It cannot provide a false-positive baseline at all. The holdout may still be right for recall under
domain shift, but the stated reasoning needs correcting.

## 7.4 Feature constraints now established by measurement

- `orig_bytes`/`resp_bytes` are corrupt in the classes that make them look useful — use `*_ip_bytes`.
- `hour_utc` and time-derived features recover 95% of 2017's attack taxonomy; they must be excluded
  from any evaluation or the result measures the schedule.
- `heartbleed` (1 flow) and `infiltration` (25 / 8) cannot carry a cross-domain claim and should be
  dropped or folded.

---

# Part VIII — Future work

**Immediate, cheap, high value**

1. **Draw the certification sample for 2017.** Converts 99.34% aggregate agreement into per-class
   precision with intervals. The machinery is built and tested.
2. **Install `scikit-learn`, run tiers 4–5.** Directly tests the shortcut hypothesis against Jacobs
   et al.'s TTL finding.
3. **Complete 2018/2019**: re-label against corrected manifests, re-validate.

**Manifest corrections the oracle has now justified**

4. Extend the Botnet-ARES window to its failed-C2 tail (4,228 flows); correct the Infiltration NMAP
   start to 17:33:30Z (2,026 flows); add the Cool Disk MAC window (22 flows). Each is a YAML edit
   plus a re-label, **with the oracle as the regression test** — a closed loop that did not exist
   before this work.

**Infrastructure**

5. **Stage the 2018 oracle on EC2.** It cannot be done locally: the archive is 9.7 GB and unpacks to
   ~37 GB against 21 GB free. Without it there is no independent verification for 2018.
6. **Re-extract with Zeek detection policies** to generate `notice.log`. Research established the
   naive approach fails four ways (`scan.zeek` no longer exists in Zeek 8.x; heartbleed and FTP
   brute-force are not in `local`; the SQLi threshold is 50 requests/5min against a 2-minute attack;
   1-hour notice suppression would destroy boundary measurement). The highest-value channel is an
   **Intel-framework feed synthesised from CIC's own published attacker addresses** — zero
   thresholds, high recall, and it answers precisely what the schedule cannot: *was the documented
   attacker on the wire, and exactly when?*

**Methodological**

7. Re-calibrate `sch.endpoint_coverage` against a per-rule out-of-window baseline.
8. Resolve the taxonomy per §7.2.
9. **Publish the CIC-DDoS-2019 corrected labels.** Nobody has done this work. Our re-extraction plus
   this module is the closest anyone has come, and it would be a genuine contribution.

---

# Appendix A — Check inventory (57)

**T0 integrity (12)** — `art.cicflowmeter_artefacts` · `int.bytes_physically_possible` ·
`int.capture_spans_disjoint` · `int.key_nulls` · `int.label_wellformed` · `int.null_not_zero` ·
`int.provenance_current` · `int.relabel_determinism` · `int.row_parity` · `int.schema_stability` ·
`int.ts_in_capture` · `int.uid_unique`

**T1 schedule (15)** — `art.date_anchor` · `qtn.coverage` · `qtn.declared_regions_fire` ·
`qtn.overlap_with_attack` · `sch.boundary_residue` · `sch.class_day_conformance` ·
`sch.duration_fits_window` · `sch.endpoint_coverage` · `sch.extension_audit` ·
`sch.interval_overlap_delta` · `sch.intra_window_gaps` · `sch.offset_calibration` ·
`sch.rule_ambiguity` · `sch.rule_yield` · `sch.window_sweep`

**T2 behaviour (4)** — `beh.benign_contamination` · `beh.class_conformance` · `beh.executed_ratio` ·
`beh.port_anomaly`

**T3 corroboration (13)** — `cor.analyzer_presence` · `cor.auth_evidence` ·
`cor.corroboration_rate` · `cor.http_payload` · `cor.notice_agreement` · `cor.reflection_service` ·
`cor.weird_concentration` · `orc.attempted_axis` · `orc.class_agreement` · `orc.coverage` ·
`orc.direction_reversal` · `orc.false_attacks` · `orc.missed_attacks`

**T4 noise (5)** — `noi.cl_calibration` · `noi.confident_learning` · `noi.duplicate_conflict` ·
`noi.knn_disagreement` · `noi.margin_outliers`

**T5 probes (4)** — `prb.class_support` · `prb.domain_discriminator` · `prb.identifier_shortcut` ·
`prb.temporal_shortcut`

**T6 cross-dataset (4)** — `xds.benign_baseline` · `xds.class_coheres` · `xds.class_support_matrix` ·
`xds.taxonomy_audit`

# Appendix B — Related documents

- `docs/label_evidence.md` — the evidence log. Every source-verification finding and every
  measurement, with numbers and primary-source quotes.
- `docs/label_certification_runbook.md` — the ordered procedure from "labels exist" to a defensible
  precision statement.
- `reports/validation/index.html` — offline HTML reports, one per dataset.

# Appendix C — Reproduction

```bash
aws login
pip install -r requirements.txt
make label    DATASET=cic-ids-2017
make validate DATASET=cic-ids-2017
make oracle-fetch DATASET=cic-ids-2017 && make oracle-stage DATASET=cic-ids-2017
make oracle-join  DATASET=cic-ids-2017
make validate DATASET=cic-ids-2017      # orc.* now runnable
make validate-gate
```

The negative control is `python3 tests/test_validate.py`. Every finding in
`reports/validation/<dataset>.json` carries a `repro` field containing the query behind its numbers.

# Appendix D — Status at report time

| Item | Status |
|---|---|
| Module (57 checks, 7 tiers) | complete, 0 errors on full sweep |
| Negative + positive controls | passing |
| 2017 label validation | complete — 82 findings, 44 checks, 0 errors |
| 2017 external oracle | complete — 99.34% agreement |
| 2018 / 2019 validation | in flight; both need a re-label first (manifest hashes changed) |
| 2018 external oracle | blocked — needs ~37 GB, EC2 required |
| Tiers 4–5 | never run — `scikit-learn` not installed |
| Certification sample | built and tested on synthetic data; not drawn on real data |
| Nothing committed | 32+ files changed/new, all pending review |
