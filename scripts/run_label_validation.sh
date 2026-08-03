#!/usr/bin/env bash
#
# Talos — one clean, uninterrupted labelling + label-validation run.
#
# Why this exists rather than a sequence of Makefile targets:
#
#   runner.py OVERWRITES reports/validation/<dataset>.json. It does not merge.
#   So `make validate` followed by `make validate-deep` does not add tier 4 to
#   the report — it replaces a 44-check report with a 1-check one. Every tier a
#   dataset needs has to go in a single invocation, and that is what this does.
#
# It is also ordered so the lake is scanned once per dataset: labelling first,
# then the oracle join (which produces the artefact tier 3's orc.* checks need),
# then one validation pass. Running validation before the join silently records
# six checks as "skipped: missing required input (oracle)".
#
# Nothing is deleted. Stale artefacts are moved into reports/_archive_<stamp>/.
# The oracle download cache (data/oracle/) and the EDA profiles (reports/eda/)
# are deliberately kept: the first is ~10 GB of downloads, the second is what
# tier 6 compares against.
#
# Usage
#   tmux new -s talos
#   bash scripts/run_label_validation.sh 2>&1 | tee run.log
#
# Resume after a failure — each stage is independent:
#   SKIP_LABEL=1  bash scripts/run_label_validation.sh
#   SKIP_ORACLE=1 bash scripts/run_label_validation.sh
#   DATASETS="cic-ids-2017" bash scripts/run_label_validation.sh

# Deliberately not `set -e`: runner.py exits 1 when it finds a blocking
# severity. That is a result, not a failure, and the run must continue.
set -uo pipefail

# ---------------------------------------------------------------- locate repo
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
[ -f config.yml ] || { echo "!! no config.yml in $ROOT — wrong directory"; exit 1; }

PY="$(test -x .venv/bin/python && echo "$ROOT/.venv/bin/python" || echo python3)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGDIR="$ROOT/logs/label_run_$STAMP"
mkdir -p "$LOGDIR"

DATASETS="${DATASETS:-cic-ids-2017 cic-ids-2018 cic-ddos-2019}"
ORACLE_DATASETS="${ORACLE_DATASETS:-cic-ids-2017 cic-ids-2018}"
# Tier 5 does not exist. Tier 4 is one SQL check; tier 6 is offline and instant.
TIERS="${TIERS:-0,1,2,3,4,6}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
rule() { printf '%s\n' "----------------------------------------------------------------"; }

declare -a RESULTS=()

run() {                                   # run <label> <cmd...>
    local name="$1"; shift
    local log="$LOGDIR/${name//[^A-Za-z0-9_.-]/_}.log"
    local t0=$SECONDS
    printf '\n'; rule; bold "▶ $name"
    printf '  log  %s\n' "${log#$ROOT/}"; rule
    "$@" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    local dt=$(( SECONDS - t0 ))
    RESULTS+=("$(printf '%-46s rc=%-3s %dm%02ds' "$name" "$rc" $((dt/60)) $((dt%60)))")
    printf '\n  %s finished rc=%s in %dm%02ds\n' "$name" "$rc" $((dt/60)) $((dt%60))
    return $rc
}

# ------------------------------------------------------------------ guard rails
# Sized from the box actually underneath this, not from the 16 GB laptop the
# Makefile defaults assume. DuckDB spills to disk; capping it is what stops a
# long scan filling the root volume and wedging sshd.
CORES="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
MEM_GB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
[ -n "${MEM_GB:-}" ] && [ "$MEM_GB" -gt 0 ] 2>/dev/null || MEM_GB=8
DUCK_TMP="${DUCK_TMP:-/tmp/talos_duckdb}"
mkdir -p "$DUCK_TMP"
FREE_GB="$(df -Pk "$DUCK_TMP" | awk 'NR==2{printf "%d", $4/1048576}')"

MEMLIMIT="${MEMLIMIT:-$(( MEM_GB * 60 / 100 ))GB}"
MAXTEMP="${MAXTEMP:-$(( FREE_GB > 12 ? FREE_GB - 10 : 4 ))GB}"
VAL_FLAGS=(--temp-dir "$DUCK_TMP" --threads "$CORES" --memory-limit "$MEMLIMIT" --max-temp "$MAXTEMP")

# ================================================================== 1. preflight
bold "Talos — clean labelling + label-validation run"
printf '  repo      %s\n  python    %s\n  run id    %s\n' "$ROOT" "$PY" "$STAMP"
printf '  cores %s   mem %sGB   free(%s) %sGB\n' "$CORES" "$MEM_GB" "$DUCK_TMP" "$FREE_GB"
printf '  duckdb    memory-limit=%s  max-temp=%s\n' "$MEMLIMIT" "$MAXTEMP"
printf '  datasets  %s\n  tiers     %s\n' "$DATASETS" "$TIERS"

git -C "$ROOT" log --oneline -1 2>/dev/null | sed 's/^/  commit    /'

if [ "$FREE_GB" -lt 20 ]; then
    echo "!! only ${FREE_GB}GB free on $DUCK_TMP — the 2018 oracle needs ~45GB total."
    echo "   Continuing, but expect oracle staging for 2018 to fail."
fi

# Credentials must be live before a two-hour run, not discovered missing inside it.
if ! aws sts get-caller-identity >"$LOGDIR/preflight_aws.log" 2>&1; then
    echo "!! aws sts get-caller-identity failed — no usable credentials. Aborting."
    cat "$LOGDIR/preflight_aws.log"; exit 1
fi
echo "  aws       ok"

# The synthetic fixture proves the checks still catch known-planted faults. If
# this fails the run is not worth starting, because a quiet report would be
# indistinguishable from a broken one.
if ! run "00_selftest" "$PY" tests/test_validate.py; then
    echo "!! self-test failed — the checks are not trustworthy. Aborting."; exit 1
fi

# ================================================================= 2. clean out
ARCHIVE="$ROOT/reports/_archive_$STAMP"
bold "▶ 01_clean — moving stale artefacts to ${ARCHIVE#$ROOT/}"
mkdir -p "$ARCHIVE"
moved=0
# Only archive the labelling reports if this run is going to regenerate them.
# Otherwise a SKIP_LABEL=1 resume moves aside evidence it will never rebuild,
# and the final summary reports labels it cannot see as missing.
CLEAN_TARGETS="reports/validation/*.json reports/validation/*.html reports/validation/*.parquet"
[ "${SKIP_LABEL:-0}" != "1" ] && CLEAN_TARGETS="reports/labelling_*.json $CLEAN_TARGETS"
for f in $CLEAN_TARGETS; do
    [ -e "$f" ] || continue
    mkdir -p "$ARCHIVE/$(dirname "${f#reports/}")"
    mv "$f" "$ARCHIVE/${f#reports/}" && moved=$((moved+1))
done
rm -rf "${DUCK_TMP:?}"/* 2>/dev/null
printf '  %d artefact(s) archived; duckdb spill cleared\n' "$moved"
printf '  kept: data/oracle (downloads), reports/eda (tier 6 inputs)\n'
RESULTS+=("$(printf '%-46s rc=0   archived %d file(s)' 01_clean "$moved")")

# ==================================================================== 3. label
# Required, not cosmetic: provenance is content-hashed, so every validation
# report is unquotable until the labels carry the current manifest sha.
if [ "${SKIP_LABEL:-0}" != "1" ]; then
    for d in $DATASETS; do
        run "10_label_$d" "$PY" src/preprocess/label/label_flows.py \
            --dataset "$d" "${VAL_FLAGS[@]}" \
            || echo "!! labelling $d returned non-zero — later stages will be wrong"
    done
else
    echo "  (skipping labelling: SKIP_LABEL=1)"
fi

# =================================================================== 4. oracle
# DistriNet's hand-forensicated corrected labels — the only ground truth here
# that did not come from the CIC schedule, so the only thing that can catch an
# error we inherited rather than introduced. None exists for 2019.
if [ "${SKIP_ORACLE:-0}" != "1" ]; then
    for d in $ORACLE_DATASETS; do
        case " $DATASETS " in *" $d "*) ;; *) continue ;; esac
        run "20_oracle_fetch_$d" "$PY" src/label_validation/oracle/stage.py \
            --dataset "$d" --fetch
        run "21_oracle_stage_$d" "$PY" src/label_validation/oracle/stage.py \
            --dataset "$d" --stage
        run "22_oracle_join_$d"  "$PY" src/label_validation/oracle/stage.py \
            --dataset "$d" --join "${VAL_FLAGS[@]}"
    done
else
    echo "  (skipping oracle: SKIP_ORACLE=1)"
fi

# ================================================================= 5. validate
# One invocation per dataset covering every tier, because the report is
# rewritten wholesale on each run.
for d in $DATASETS; do
    run "30_validate_$d" "$PY" src/label_validation/runner.py \
        --dataset "$d" --tier "$TIERS" "${VAL_FLAGS[@]}"
done

# =========================================================== 6. render and gate
run "40_render" "$PY" src/label_validation/report/render.py
run "50_gate"   "$PY" src/label_validation/report/gate.py --all
GATE_RC=$?

# =================================================================== 7. summary
printf '\n'; rule; bold "RUN SUMMARY  ($STAMP)"; rule
printf '%s\n' "${RESULTS[@]}"
rule
"$PY" - <<'PYEOF'
import hashlib, json, pathlib
R = pathlib.Path("reports"); V = R / "validation"
for d in ("cic-ids-2017", "cic-ids-2018", "cic-ddos-2019"):
    man = R.parent / "src/preprocess/manifests" / f"{d}.yaml"
    cur = hashlib.sha256(man.read_bytes()).hexdigest()[:12] if man.is_file() else "?"
    v = V / f"{d}.json"
    if not v.is_file():
        print(f"  {d:14} NO REPORT"); continue
    r = json.loads(v.read_text()); m, s = r["meta"], r["summary"]
    fresh = "fresh" if m.get("manifest_sha") == cur else "STALE"
    sev = " ".join(f"{k}={v}" for k, v in sorted(s["by_severity"].items()))
    print(f"  {d:14} {fresh:6} tiers={m.get('tiers')} "
          f"run={s['checks_run']} skip={s['checks_skipped']} err={s['checks_error']}")
    print(f"  {'':14} {s['n_findings']} findings: {sev or 'none'}")
    o = V / f"oracle_{d}.json"
    if o.is_file():
        j = json.loads(o.read_text())["join"]
        print(f"  {'':14} oracle coverage={j['coverage']:.4%} "
              f"agree={j['zeek_flows_class_agree']/j['zeek_flows_matched']:.4%} "
              f"clock_delta={j['clock_delta_seconds']}s")
PYEOF
rule
printf '  logs    %s\n  reports %s\n' "${LOGDIR#$ROOT/}" "reports/validation/index.html"
printf '  gate exit code: %s  (%s)\n' "$GATE_RC" \
    "$([ "$GATE_RC" -eq 0 ] && echo 'no blocking findings' || echo 'blocking findings present — read the report')"
exit "$GATE_RC"
