#!/usr/bin/env bash
set -euo pipefail

APP="./target/release/momentum"
PROMPT_FILE="${1:-prompts/example1.txt}"
PROFILE="${2:-0}"
ITER=${3:-3}
OUTDIR="${4:-bench_results}"
mkdir -p "$OUTDIR"

for i in $(seq 1 "$ITER"); do
  TIMESTAMP=$(date +%Y%m%d%H%M%S)
  START=$(date +%s%N)
  /usr/bin/time -f "%M" -o /tmp/peak_rss.tmp "$APP" --bench --profile "$PROFILE" --prompt-file "$PROMPT_FILE" > /tmp/bench_stdout.txt 2>/tmp/bench_stderr.txt || true
  END=$(date +%s%N)

  ELAPSED_NS=$((END-START))
  ELAPSED_S=$(awk "BEGIN {print $ELAPSED_NS/1e9}")
  PEAK_RSS_KB=$(cat /tmp/peak_rss.tmp | tr -d '[:space:]')
  TOKENS=0
  if [ -f /tmp/bench_out.json ]; then
    TOKENS=$(jq -r '.tokens // 0' /tmp/bench_out.json || echo 0)
  fi

  echo "$PROFILE,$TIMESTAMP,$ELAPSED_S,$PEAK_RSS_KB,$TOKENS" >> "$OUTDIR/result_${PROFILE}.csv"
  echo "iter $i profile $PROFILE elapsed ${ELAPSED_S}s peak_rss ${PEAK_RSS_KB}KB tokens ${TOKENS}"
  sleep 1
done
