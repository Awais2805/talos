# Running labelling and validation on EC2

One command, ~3 hours, unattended. Run it in `tmux` — an SSH drop otherwise kills a two-hour scan.

```bash
ssh <your-host>
tmux new -s talos
cd ~/talos && git pull
bash scripts/run_label_validation.sh
```

That is the whole procedure. The rest of this document explains what it does and how to read the
result.

---

## Why a script rather than a sequence of make targets

`runner.py` writes `reports/validation/<dataset>.json` with `write_text` — it **overwrites, it does
not merge**. Running the tiers in separate invocations therefore does not accumulate a report; each
run discards the previous one. A `validate` pass followed by a tier-4 pass leaves a report
containing a single check, and the 44 that came before it are gone.

Everything a dataset needs goes in one invocation: `--tier 0,1,2,3,4,6`. Tier 5 does not exist.

Order matters for the same kind of reason. The oracle join has to happen **before** validation,
because tier 3's six `orc.*` checks read the join artefact off disk. Run validation first and they
are recorded as *skipped: missing required input (oracle)* — never as a silent pass, but never as
evidence either.

## What the script does

| Stage | What | Notes |
|---|---|---|
| 0 | preflight | aborts if AWS credentials are dead or the synthetic self-test fails |
| 1 | clean | moves stale artefacts to `reports/_archive_<stamp>/`; **deletes nothing** |
| 2 | label ×3 | required — provenance is content-hashed, so stale labels block every report |
| 3 | oracle ×2 | fetch, stage, join for 2017 and 2018 |
| 4 | validate ×3 | one pass per dataset, all tiers |
| 5 | render, gate | HTML report and the exit code |

Guard rails are sized from the box (`nproc`, `free -g`), not from the 16 GB laptop the Makefile
defaults assume. DuckDB's spill is capped so a long scan cannot fill the root volume and wedge
`sshd`.

Kept deliberately: `data/oracle/` (~10 GB of downloads) and `reports/eda/` (what tier 6 compares
against).

## Resuming

Each stage is independent, so a failure part-way costs only that stage:

```bash
SKIP_LABEL=1  bash scripts/run_label_validation.sh    # labels already good
SKIP_ORACLE=1 bash scripts/run_label_validation.sh    # skip the 9.7 GB download
DATASETS="cic-ids-2017" bash scripts/run_label_validation.sh
```

Per-stage logs land in `logs/label_run_<stamp>/`.

## Disk

2018's oracle is the constraint: 9.7 GB archive → ~37 GB unpacked. The script warns below 20 GB
free. Staging converts one capture day at a time and deletes each CSV as it goes, so peak usage is
archive + one CSV ≈ 14 GB rather than 47.

## Pulling the results back

```bash
# laptop
rsync -av <your-host>:~/talos/reports/validation/ ~/talos/reports/validation/
open ~/talos/reports/validation/index.html
```

Self-contained HTML, no external dependencies.

---

## What to expect, so a surprise is a real surprise

| Dataset | Expect |
|---|---|
| **2017** | Cleanest. Oracle coverage ~99.76%, class agreement ~99.04% of matched flows (~99.34% setting direction reversals aside), clock delta 0.0s, `sch.offset_calibration` silent. Materially different means the *run* is wrong, not the labels. |
| **2018** | Three CRITICALs already confirmed: FTP-BruteForce 100% empty connections (192,294/192,294); DoS-SlowHTTPTest modal port 21, not 80; 5 overlapping capture pairs, 3 sharing a `ts_min`. |
| **2019** | Will fail the gate on `xds.benign_baseline` — its benign class is ~90% attack traffic. Correct behaviour, and a deliberate decision is needed about whether it can serve as a holdout at all. |

`int.provenance_current` firing as a **BLOCKER** on any dataset means labelling did not run or did
not finish. It is the one finding that invalidates the rest of that dataset's report rather than
describing it.

## Two checks not to act on

- **`sch.endpoint_coverage`** fires on nearly every rule at 94–100% "unexplained" traffic. Threshold
  artefact: the victim server carries ordinary load during attack windows, so most in-window traffic
  legitimately is not the attack. Needs a per-rule baseline.
- **Transient S3 errors.** Two 2018 checks failed on the laptop from network blips. The underlying
  cause — a swallowed exception cached as "no columns" — is fixed. In-region they should not recur;
  if they do it is network, and re-running the one check is enough.

## Certification (separate, and required for a stated precision figure)

Everything above establishes *agreement*. Only this establishes *precision with an interval*.

```bash
make validate-sample DATASET=cic-ids-2017     # ~400/class → ±2-3% at 95%
# adjudicate the `verdict` column: correct / wrong / uncertain / benign / a class name
make validate-score  DATASET=cic-ids-2017
```

Each sampled row carries its evidence — matched rule and window, conn state, history, plus any http
URI, ssl SNI, dns query, ssh auth or ftp reply code — so most decide in seconds without going back
to the lake.

---

## The loop

When validation finds a real error, **do not patch the labels**. Fix the *manifest* and re-run.
Labels stay derived from the schedule, the fix is a reviewable YAML diff, and the oracle becomes the
regression test.
