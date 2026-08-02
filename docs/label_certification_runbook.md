# Talos — certifying the labels

The goal is a defensible statement of the form:

> `cic-ids-2017` labels are **99.4% precise (95% CI 98.7–99.8%)** on **97.1% of flows**,
> with **83% of attack flows independently corroborated**, against manifest `5c210d17e9b1`.

Every clause in that sentence comes from a different stage below, and none of them is
substitutable for another. A pile of green checks is not a precision figure; a precision
figure without coverage is a half-truth; and both are worthless if the labels were produced
by a manifest nobody can reproduce.

Run the stages in order. Each one is cheap to repeat and expensive to skip.

---

## 0. Preconditions

```bash
aws login                                # the lake is behind an expiring session
pip install -r requirements.txt          # sklearn/xgboost unlock T4 and T5
```

Without sklearn, T4/T5 emit `INFO: check did not run` findings rather than silently
passing. That is deliberate — a missing dependency must never read as a clean result —
but it also means no noise estimate and no shortcut probe.

## 1. Make provenance current

```bash
make label DATASET=cic-ids-2017
```

`int.provenance_current` compares the manifest and taxonomy hashes recorded *in the data*
against the files on disk. Any manifest edit — including a comment — invalidates it, and
correctly so: the labels must be reproducible from the schedule that is readable today, not
from one that existed at some point in the past.

This is a BLOCKER by policy. Nothing downstream is interpretable until it passes.

## 2. Structural validation

```bash
make validate DATASET=cic-ids-2017       # tiers 0-3
make validate-deep DATASET=cic-ids-2017  # tiers 4-5, needs sklearn
make validate-cross                      # tier 6, offline, instant
open reports/validation/index.html
```

Read them in tier order, because the tiers are not independent:

- **T0 failing makes every later number meaningless.** A row-count mismatch or a duplicate
  uid changes what population the other checks are describing.
- **T1 is where labelling errors actually live.** `sch.window_sweep` is the strongest single
  check: it asks whether traffic changes at the boundary the schedule asserts, and reports
  the offset at which it *does* change — which is the correction to apply.
- **`sch.offset_calibration` deserves particular attention.** No CIC source states a
  timezone for any of the three datasets, so all three `utc_offset` values are inferences.
  A shift shared by several unrelated rules is the capture clock, not the schedule, and it
  moves every window in the dataset at once.
- **T2/T3 catch the failure T0/T1 cannot**: a window transcribed perfectly and aimed at the
  wrong traffic. 2018's `DoS-SlowHTTPTest` is 105,550 flows on port 21, all refused.
- **T5 is about what the labels are worth, not whether they are right.** A label can be
  perfectly correct and still useless: if `prb.identifier_shortcut` scores near-perfect, a
  downstream model learns attacker IPs rather than attack behaviour.

## 3. Independent corroboration

Everything above reasons from the schedule or from the traffic we labelled with it. This
stage is the only one that brings in a source which could not have copied our answer.

```bash
make oracle-fetch  DATASET=cic-ids-2017   # 328 MB (2018 is 9.7 GB)
make oracle-stage  DATASET=cic-ids-2017
make oracle-join   DATASET=cic-ids-2017
make validate DATASET=cic-ids-2017        # orc.* checks now runnable
```

The oracle is the DistriNet CNS2022 corrected label set — the same PCAPs, relabelled by
hand after forensic examination. **Read `orc.coverage` first**: it gates the interpretation
of every other oracle number, because a low coverage figure means the join failed, not that
the labels agree.

**Flow counts will not match, and that is not a finding.** CICFlowMeter chops flows at a
120-second active timeout; Zeek uses inactivity timeouts. One Zeek flow maps to N oracle
rows. The adjudication table is built into the checks:

| Disagreement | Indicts |
|---|---|
| Flow count differs | Neither — flow-construction semantics |
| One Zeek flow ↔ many oracle rows | Neither — timeout semantics |
| Oracle row is an `8.0.6.4` / protocol-0 artefact | Theirs — a CICFlowMeter parsing bug |
| Class clash on an overlapping, direction-agreeing match | **Investigate — the only diagnostic case** |
| We say benign, oracle says attack, in a known-mislabelled window | **Ours** — we inherited a CIC error |
| We say attack, oracle says `- Attempted` | Neither — we lack their second label axis |

There is **no corrected label set for CIC-DDoS-2019**. Nobody has done this work on it.
That is why stage 4 exists.

## 4. Calibrate the estimator, then use it where there is no oracle

```bash
make validate-deep DATASET=cic-ids-2017   # includes noi.cl_calibration
```

`noi.cl_calibration` injects label errors at known rates and measures how many confident
learning recovers — the protocol from Arp et al. (USENIX Security 2022), who reached 84%
detection at 0.2% false flags.

The logic of the chain matters: calibrate on 2017 and 2018, where the oracle can check the
answer, and only then trust `noi.confident_learning` on 2019, where nothing can. An
uncalibrated noise estimate is not evidence, and the check says so in its own findings.

## 5. Certify

```bash
make validate-sample DATASET=cic-ids-2017   # writes reports/validation/samples/<ds>.csv
# fill in the `verdict` column: correct / wrong / uncertain / benign / a class name
make validate-score  DATASET=cic-ids-2017
```

This is the only stage that produces a *measured* precision. Everything before it ranks
suspects; none of it certifies anything.

Sizing: ~400 flows per class gives roughly ±2–3% at 95%; ~2000 gives about ±1%. Each
sampled flow carries its evidence bundle — the matched rule and its window, conn state and
history, plus any http URI, ssl SNI, dns query, ssh auth or ftp reply code — so most
adjudicate in seconds without going back to the lake.

`score.py` reports **Wilson** intervals, not the normal approximation, because the normal
approximation misbehaves badly at proportions near 1 — which is exactly where we expect to
be. Flows marked `uncertain` are excluded from precision and reported separately.

## 6. Quote coverage with accuracy, always

```bash
make validate DATASET=cic-ddos-2019       # qtn.coverage reports the share quarantined
```

Some flows cannot be adjudicated from the schedule at any level of effort: 2019's
inter-window residue, where the flood plainly has not stopped but no window says which
vector it is; 2018's infiltration windows, where CIC names the infected host but never the
second stage's targets. Nineteen such regions are declared in the manifests, each with a
written justification.

Those flows are marked `label_quality = 'uncertain'` and excluded from training and
evaluation pools. **This is how 99–100% is reached — by declining to guess about the
minority that cannot be verified, not by pretending about it.** A precision figure quoted
without its coverage is the one number in this document that would be dishonest.

## 7. Gate

```bash
make validate-gate
```

Exits non-zero if any dataset carries a finding at or above the policy's blocking severity,
or from a check in `always_block`. Nothing here is fixed by editing a label: corrections go
into the manifest, labelling is re-run, and the cycle repeats from stage 1.

---

## What the module cannot tell you

Stated plainly, because a validation report that implies completeness it does not have is
worse than no report.

- **Behaviourally identical benign traffic.** DistriNet found that 2017's DDoS LOIC-HTTP
  has identical flows occurring outside the attack window. A schedule-based labeller calls
  those benign and they are indistinguishable from the attack. This is an irreducible floor
  on achievable label accuracy, not a defect to fix.
- **Encrypted payloads.** 2017's Infiltration Dropbox downloads cannot be adjudicated from
  the wire; DistriNet made a judgement call and labelled them benign. Ours may differ.
- **Confident learning assumes class-conditional noise.** Schedule-derived noise is not:
  whether a flow is mislabelled depends on its own features — whether it straddled a
  boundary, whether it carried payload. CL is empirically useful here but carries no
  consistency guarantee, and `noi.cl_calibration`'s pass verdict says so explicitly.
- **The oracle is not available for 2019**, and its own labels are a human judgement, not
  ground truth handed down. Agreement with it is strong evidence, not proof.
