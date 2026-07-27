#!/usr/bin/env bash
# Talos — Zeek feature extraction (batch)
# Reads raw pcap archives from S3, runs Zeek per split (in parallel),
# uploads JSON logs to processed/zeek/. Resumable via _DONE markers; disk-flat.
#
# Usage:  PARALLEL=6 ./zeek_batch.sh 2>&1 | tee ~/zeek_batch.log
set -uo pipefail

BUCKET=ids-datalakec48eb2cab942494ba5059fac3b3527d9
RAW_PREFIXES=(${RAW_ONLY:-cic-ids-2017/pcaps cic-ddos-2019/pcaps cic-ids-2018/pcaps})
# zone-2 prefix: must match where the lake's Zeek logs (and _DONE markers) live,
# otherwise the resume check misses and everything re-extracts
PROC=${PROC:-extracted}
ZEEK_IMG=zeek/zeek:8.2.1
WORK=$HOME/zeekwork
JOBS=${PARALLEL:-$(nproc)}

preflight() {
  command -v docker >/dev/null 2>&1 || {
    echo "FATAL: docker not installed — sudo apt-get install -y docker.io"; exit 1; }
  docker info >/dev/null 2>&1 || {
    echo "FATAL: docker daemon down or user lacks permission —"
    echo "       sudo systemctl enable --now docker && sudo usermod -aG docker \$USER (then re-login)"; exit 1; }
  docker image inspect "$ZEEK_IMG" >/dev/null 2>&1 || {
    echo "pulling $ZEEK_IMG ..."; docker pull "$ZEEK_IMG" >/dev/null || {
      echo "FATAL: cannot pull $ZEEK_IMG"; exit 1; }; }
  echo "preflight OK: docker up, $ZEEK_IMG present"
}

process_one() {
  local key="$1" dataset rest outpref
  dataset=${key%%/*}
  rest="${key#${dataset}/pcaps/}"; rest="${rest%.zip}"; rest="${rest%.rar}"; rest="${rest%.pcap}"
  outpref="$PROC/$dataset/$rest"

  [ -n "$(aws s3 ls "s3://$BUCKET/$outpref/_DONE" 2>/dev/null)" ] && { echo "SKIP $key"; return 0; }

  local d="$WORK/tmp"; rm -rf "$d"; mkdir -p "$d/in" "$d/out"
  echo "$(date '+%F %T') PROCESS $key"

  aws s3 cp "s3://$BUCKET/$key" "$d/in/" >/dev/null || { echo "DL FAIL $key"; rm -rf "$d"; return 1; }

  local f="$d/in/$(basename "$key")"
  if [[ "$key" == *.zip ]]; then
    echo "  unzipping $(basename "$key") ..."
    unzip -o -q "$f" -d "$d/in/" || { echo "UNZIP FAIL $key"; rm -rf "$d"; return 1; }
    rm -f "$f"
  elif [[ "$key" == *.rar ]]; then
    # some CIC days ship as .rar (e.g. cic-ids-2018 Tuesday-20-02-2018).
    # prefer RARLAB unrar: Ubuntu's unar mis-handles RAR5 / multi-GB members.
    echo "  unrar $(basename "$key") ..."
    if command -v unrar >/dev/null 2>&1; then
      unrar x -y "$f" "$d/in/" || { echo "UNRAR FAIL $key"; rm -rf "$d"; return 1; }
    elif command -v unar >/dev/null 2>&1; then
      unar -f -o "$d/in/" "$f" || { echo "UNRAR FAIL $key"; rm -rf "$d"; return 1; }
    else
      echo "NO RAR TOOL $key — install: sudo apt-get install -y unrar"; rm -rf "$d"; return 1
    fi
    rm -f "$f"
  fi

  # detect pcaps by MAGIC (files are often extensionless splits)
  mapfile -t pcaps < <(
    find "$d/in" -type f ! -iname '*.zip' ! -iname '*.rar' | while IFS= read -r ff; do
      case "$(file -b "$ff" 2>/dev/null)" in *pcap*|*tcpdump*|*capture*) printf '%s\n' "$ff";; esac
    done)
  [ ${#pcaps[@]} -eq 0 ] && { echo "NO PCAP $key"; rm -rf "$d"; return 1; }

  local total=${#pcaps[@]}
  echo "  extracting $total pcap(s) with $JOBS parallel workers ..."

  # live loader
  ( while :; do
      c=$(find "$d/out" -name .complete 2>/dev/null | wc -l)
      l=$(find "$d/out" -name '*.log' 2>/dev/null | wc -l)
      printf '\r    progress: %d/%d splits done | %d log files written   ' "$c" "$total" "$l"
      [ "$c" -ge "$total" ] && break
      sleep 1
    done; printf '\n' ) &
  local prog=$!

  # parallel worker pool
  local running=0 p rel pstem
  for p in "${pcaps[@]}"; do
    rel="${p#$d/in/}"; pstem="${rel//\//_}"
    ( mkdir -p "$d/out/$pstem" "$d/err"
      docker run --rm -v "$d":/data -w "/data/out/$pstem" "$ZEEK_IMG" \
        zeek -C -r "/data/in/$rel" LogAscii::use_json=T >"$d/err/$pstem.txt" 2>&1 \
        || echo "ZEEK FAIL $rel :: $(head -c 200 "$d/err/$pstem.txt" | tr '\n' ' ')" >>"$d/err/failures.txt"
      touch "$d/out/$pstem/.complete" ) &
    running=$((running+1))
    if (( running >= JOBS )); then wait -n 2>/dev/null; running=$((running-1)); fi
  done
  wait
  kill "$prog" 2>/dev/null; wait "$prog" 2>/dev/null

  [ -z "$(find "$d/out" -name '*.log' -type f -print -quit)" ] && {
    echo "NO LOGS $key"
    [ -s "$d/err/failures.txt" ] && head -3 "$d/err/failures.txt" | sed 's/^/    /'
    rm -rf "$d"; return 1; }

  local nlogs; nlogs=$(find "$d/out" -name '*.log' -type f | wc -l)
  echo "  uploading $nlogs log files -> $outpref"
  aws s3 cp "$d/out/" "s3://$BUCKET/$outpref/" --recursive --exclude "*" --include "*.log" >/dev/null \
    || { echo "UPLOAD FAIL $key"; rm -rf "$d"; return 1; }
  aws s3api put-object --bucket "$BUCKET" --key "$outpref/_DONE" >/dev/null

  rm -rf "$d"
  echo "$(date '+%F %T') DONE  $key -> $outpref"
}

preflight
for pref in "${RAW_PREFIXES[@]}"; do
  echo "==== $pref ===="
  aws s3 ls "s3://$BUCKET/$pref/" --recursive \
    | awk '{ $1=$2=$3=""; sub(/^ +/,""); print }' \
    | grep -vE '(/_DONE$|/$|^$)' \
    | while IFS= read -r key; do process_one "$key"; done
done
echo "ALL DONE"
