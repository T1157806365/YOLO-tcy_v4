#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-full}" # screen|full
case "$MODE" in
  full) CFG=configs/experiments/p3_product_residual/S2_product_residual.yaml ;;
  screen) CFG=configs/experiments/p3_product_residual_screen/S2_product_residual.yaml ;;
  *) echo "Usage: bash scripts/train_p3_product.sh [screen|full]" >&2; exit 2 ;;
esac
[ -f "$CFG" ] || { echo "Missing YAML $CFG" >&2; exit 1; }
mkdir -p logs pid
LOG="logs/p3_product_${MODE}.log"
PID="pid/p3_product_${MODE}.pid"
nohup python -u train_rlsfa_gtga_bhlr_v2.py --cfg "$CFG" > "$LOG" 2>&1 &
echo $! > "$PID"
echo "PID=$(cat "$PID") LOG=$LOG YAML=$CFG"
echo "Watch: tail -f $LOG"
