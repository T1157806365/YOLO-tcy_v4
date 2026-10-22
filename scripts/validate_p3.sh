#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:-S2_product_residual}"  # S0_baseline | S1_conv_residual | S2_product_residual
SCOPE="${2:-full}"               # full | screen
SPLIT="${3:-test}"               # val | test
DEVICE="${4:-0}"                 # GPU number
case "$NAME" in
  S0_baseline|S1_conv_residual|S2_product_residual) ;;
  *) echo "Invalid model name $NAME" >&2; exit 2 ;;
esac
case "$SCOPE" in
  full) PROJECT=runs/p3_product_residual_full ;;
  screen) PROJECT=runs/p3_product_residual_screen ;;
  *) echo "Invalid scope $SCOPE" >&2; exit 2 ;;
esac
WEIGHTS="$PROJECT/yolo26m_UAVCB_best20261021_${NAME}/weights/best.pt"
[ -f "$WEIGHTS" ] || { echo "Missing checkpoint: $WEIGHTS" >&2; exit 1; }
python val_rlsfa_gtga_bhlr_v2.py \
  --weights "$WEIGHTS" \
  --split "$SPLIT" \
  --device "$DEVICE" \
  --batch 4 \
  --workers 4 \
  --conf 0.001 \
  --iou 0.70 \
  --max-det 300 \
  --plots
