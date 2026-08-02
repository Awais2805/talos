# `src/validate` — ground-truth validation

This module answers one question: **are the labels right?**

Talos labels flows by matching each flow's start time and endpoints against attack schedules
transcribed from CIC's published documentation. That is weak supervision from a document, not
measurement, and it fails silently — a mistranscribed window or an assumed timezone produces a
complete-looking labelled dataset with no error raised. Everything here exists to make that
failure visible.

**Scope discipline.** This module validates *ground truth*. It does not evaluate models, measure
feature quality, or estimate what a detector would score. Checks that drifted into those questions
were removed. If a proposed check does not change your belief about whether a label is correct, it
belongs somewhere else.

---

## Adding a check

Write a function, decorate it, done. Discovery is automatic — `load_checks()` walks this package
with `pkgutil`, so there is no import list to update and no way to add a file that silently never
runs.

```python
from src.validate.finding import Finding, Severity
from src.validate.registry import check

@check(id="int.my_check", tier=0, title="one line, present tense",
       needs=("labelled",), max_severity=Severity.MAJOR)
def my_check(ctx):
    """Why this check exists, and what a failure MEANS.

    Not what the code does — that is readable. What a reader should conclude
    about the labels when this fires.
    """
    n = ctx.lake.one(f"SELECT count(*) FROM read_parquet('{ctx.q('labelled')}') WHERE ...")[0]
    if not n:
        return []                      # [] means clean
    return [Finding(
        check_id="int.my_check", dataset=ctx.dataset, severity=Severity.MAJOR,
        title=f"{n:,} flow(s) ...",
        detail="What this means for the labels.",
        metrics={"flows": n},          # the quantitative claim; the gate reads this
        evidence=[{...}],              # rows demonstrating it
        repro="SELECT ...",            # the query behind the number
    )]
```

### The five rules

1. **`[]` means clean.** Never return a finding to say "I ran".
2. **Declare `needs`.** The runner skips a check whose input is missing and records the reason.
   Without it, a missing log makes your check return `[]`, which reads as a pass. *"Did not run"
   and "ran and found nothing" are different claims about the data.*
3. **Every finding carries `repro`.** A validation module that cannot be audited is another
   unverified assertion in the chain being verified.
4. **Thresholds go in `policy.yaml`**, read via `ctx.threshold("key", default)`. Report the raw
   number in `metrics` beside the verdict so a reader can disagree without editing code.
5. **Be economical with scans.** 2019's labelled zone is 72.5M flows. Aggregate everything you need
   in one query rather than one per question — see `schedule.py::_sweep`, which serves three checks
   from a single pass.

### What `ctx` gives you

| | |
|---|---|
| `ctx.q(name)` | parquet glob for `labelled`, `conn`, `http`, `ssl`, `dns`, `ssh`, `ftp`, `oracle` |
| `ctx.lake.sql(q)` / `.one(q)` / `.con` | DuckDB, S3 credentials kept fresh across long scans |
| `ctx.rules` | normalised attack windows: `rid, name, canonical, t0, t1, attackers, victims, prefix, needs_payload` |
| `ctx.columns` | column → type for the labelled zone. **Guard every reference** — `service` is absent from some 2019 files |
| `ctx.threshold(k, d)` | policy lookup |
| `ctx.dataset`, `ctx.taxonomy`, `ctx.spec`, `ctx.manifest` | |

Dotted Zeek columns must be double-quoted in SQL: `c."id.orig_h"`.

---

## Tiers

Ordered by what they appeal to, not by cost. Later tiers are only meaningful if earlier ones pass.

| Tier | Appeals to | Checks |
|---|---|---|
| **0 integrity** | arithmetic | Is the labelled zone a faithful, reproducible function of its inputs? |
| **1 schedule** | the traffic's own shape | Does a window bound the traffic it claims to? `sch.window_sweep` is the core measurement. |
| **2 behaviour** | protocol semantics | Does a class behave the way its name implies? Driven by `expectations.yaml` — adding a class is a YAML edit. |
| **3 corroboration** | sources that never saw the schedule | Zeek's application logs, and the external DistriNet oracle (`orc.*`) — the strongest evidence here. |
| **4 consistency** | the labels themselves | The one contradiction provable without outside evidence: identical flows, different classes. |
| **6 cross-dataset** | the other datasets | Does a canonical class mean the same thing everywhere? Reads EDA profiles offline. |

Tier 5 is deliberately absent. It held model-based shortcut probes, which measure what a detector
would learn rather than whether a label is true — a real question, but a modelling one.

---

## Layout

```
finding.py       Finding + ordered Severity
registry.py      @check, capability declaration, crash isolation, auto-discovery, Context
runner.py        CLI; probes available logs; builds one shared Context per dataset
policy.yaml      every threshold that turns a measurement into a verdict
expectations.yaml  declarative per-class behavioural bands (tier 2)
checks/          one module per tier
oracle.py        external ground-truth staging + interval-overlap join
sample.py        stratified certification sample with evidence bundles
score.py         precision, Wilson intervals, Cohen's kappa, coverage
render.py        offline self-contained HTML
gate.py          exit-code policy
```

## Running it

```bash
make validate DATASET=cic-ids-2017     # tiers 0-3
make validate-cross DATASET=…          # tier 6, offline
make validate-render && make validate-gate
python3 tests/test_validate.py         # the negative control — run this after any change
```

## The negative control

`tests/build_fixture.py` builds a synthetic lake with seven planted faults and runs it through the
**real** labelling stage and the **real** validator. Nothing is mocked; only the bucket is a
directory. `tests/test_validate.py` asserts all seven are found — **and that the one correctly
scheduled rule produces no findings.**

That last assertion is the load-bearing one. It has already caught two false positives that would
otherwise have shipped. Run it after any change.
