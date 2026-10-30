#!/bin/bash
# Keep a train_lazer.py run going until it reaches TOTAL steps: if the process dies (macOS has
# killed it a few times), restart from the last checkpoint and its step (<out>.step).
#   scripts/train_supervised.sh <out.safetensors> <init.safetensors> <start step> <total steps> [extra args...]
set -u
OUT=$1; INIT=$2; START=$3; TOTAL=$4; shift 4
STEPFILE="${OUT%.safetensors}.step"
[ -f "$STEPFILE" ] || echo "$START" > "$STEPFILE"
for attempt in $(seq 1 20); do
  DONE=$(cat "$STEPFILE")
  [ "$DONE" -ge "$TOTAL" ] && { echo "done at step $DONE"; exit 0; }
  SRC=$INIT; [ "$DONE" -gt "$START" ] && SRC=$OUT
  cp "$SRC" "${OUT%.safetensors}-resume.safetensors"
  echo "=== attempt $attempt: from step $DONE ($SRC) $(date +%H:%M:%S)"
  uv run --extra train scripts/train_lazer.py --init "${OUT%.safetensors}-resume.safetensors" \
    --start-step "$DONE" --steps $((TOTAL - DONE)) --out "$OUT" "$@"
  echo "=== exit $? at $(date +%H:%M:%S)"
  sleep 5
done
