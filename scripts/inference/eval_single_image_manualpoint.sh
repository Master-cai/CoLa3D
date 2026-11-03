#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1

COLA3D_CKPT="${COLA3D_CKPT:?set COLA3D_CKPT to a trained decomposable decoder .pt}"
PRETRAINED_GEO="${PRETRAINED_GEO:-}"
GEO_CKPT="${GEO_CKPT:-}"
SS_FLOW_CKPT="${SS_FLOW_CKPT:-}"
SLAT_FLOW_CKPT="${SLAT_FLOW_CKPT:-}"
DECODER_CONFIG="${DECODER_CONFIG:-}"
STAGE="${STAGE:-all}"
SEED="${SEED:-42}"
PORT="${PORT:-18080}"
RESOLUTION="${RESOLUTION:-1024}"

INPUT_RGBA="${INPUT_RGBA:-}"
INPUT_IMAGE="${INPUT_IMAGE:-}"
INPUT_MASK="${INPUT_MASK:-}"
INPUT_ARGS=()
if [ -n "$INPUT_RGBA" ]; then
  INPUT_ARGS+=( --input-rgba "$INPUT_RGBA" )
  DEF_ID="$(basename "${INPUT_RGBA%.*}")"
elif [ -n "$INPUT_IMAGE" ] && [ -n "$INPUT_MASK" ]; then
  INPUT_ARGS+=( --input-rgb "$INPUT_IMAGE" --input-mask "$INPUT_MASK" --input-mask-mode auto )
  DEF_ID="$(basename "$(dirname "$INPUT_IMAGE")")_$(basename "${INPUT_IMAGE%.*}")"
else
  echo "ERROR: set INPUT_RGBA, or both INPUT_IMAGE and INPUT_MASK" >&2
  exit 1
fi
INPUT_ID="${INPUT_ID:-$DEF_ID}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/single_image_manualpoint/${INPUT_ID}_seed${SEED}}"
SHAPE_SLAT_PATH="${OUTPUT_DIR}/${INPUT_ID}/latents/shape_slat.pt"
MANUAL_POINTS_PATH="${OUTPUT_DIR}/${INPUT_ID}/manual_prompt_points.pt"

GEO_ARGS=()
if [[ -n "$GEO_CKPT" ]]; then
  GEO_ARGS+=(--geo-ckpt "$GEO_CKPT")
elif [[ -n "$PRETRAINED_GEO" ]]; then
  GEO_ARGS+=(--pretrained-geo "$PRETRAINED_GEO")
else
  echo "Set GEO_CKPT, or explicitly set PRETRAINED_GEO for the base decoder." >&2
  exit 2
fi
DEC_ARGS=()
[ -n "$DECODER_CONFIG" ] && DEC_ARGS=( --decoder-config "$DECODER_CONFIG" )

[[ -n "${PIPELINE_CONFIG:-}" ]] && DEC_ARGS+=(--pipeline-config "$PIPELINE_CONFIG")
[[ -n "${SS_FLOW_CONFIG:-}" ]] && DEC_ARGS+=(--ss-flow-config "$SS_FLOW_CONFIG")

COMMON=( --input-id "$INPUT_ID" --output-dir "$OUTPUT_DIR"
         --cola3d-ckpt "$COLA3D_CKPT" "${GEO_ARGS[@]}" "${DEC_ARGS[@]}"
         --resolution "$RESOLUTION" --ss-coord-resolution 64 --slat-flow-resolution "$RESOLUTION"
         --seed "$SEED" "${INPUT_ARGS[@]}" )

run_prepare() {
  if [ -f "$SHAPE_SLAT_PATH" ]; then
    echo "[prepare] cached shape_slat exists: $SHAPE_SLAT_PATH"; return 0
  fi
  SS_FLOW_CKPT="${SS_FLOW_CKPT:?set SS_FLOW_CKPT (finetuned SS flow .pt) for the prepare stage}"
  SLAT_FLOW_CKPT="${SLAT_FLOW_CKPT:?set SLAT_FLOW_CKPT (finetuned SLat flow .pt) for the prepare stage}"
  echo "[prepare] $INPUT_ID -> $OUTPUT_DIR"
  python scripts/inference/single_image_manualpoint.py --stage prepare \
    "${COMMON[@]}" --ss-flow-ckpt "$SS_FLOW_CKPT" --slat-flow-ckpt "$SLAT_FLOW_CKPT"
}

run_ui() {
  RUN_DIR="$OUTPUT_DIR" PORT="$PORT" DECODER_CONFIG="$DECODER_CONFIG" COLA3D_CKPT="$COLA3D_CKPT" ./scripts/inference/launch_manual_prompt_ui.sh
}

run_decode() {
  if [ ! -f "$MANUAL_POINTS_PATH" ]; then
    echo "ERROR: manual prompt points missing: $MANUAL_POINTS_PATH (run STAGE=ui first)" >&2; exit 1
  fi
  echo "[decode] $INPUT_ID -> $OUTPUT_DIR"
  python scripts/inference/single_image_manualpoint.py --stage decode \
    "${COMMON[@]}" --load-shape-slats-from "$OUTPUT_DIR" --manual-prompt-points-from "$OUTPUT_DIR"
}

case "$STAGE" in
  prepare) run_prepare ;;
  ui)      run_prepare; run_ui ;;
  decode)  run_decode ;;
  all)
    run_prepare
    if [ -f "$MANUAL_POINTS_PATH" ]; then
      echo "[ui] manual points already exist; skipping UI launch"
    else
      mkdir -p "$OUTPUT_DIR"
      echo "[ui] launching manual prompt picker in background (PORT=$PORT)"
      RUN_DIR="$OUTPUT_DIR" PORT="$PORT" DECODER_CONFIG="$DECODER_CONFIG" COLA3D_CKPT="$COLA3D_CKPT" ./scripts/inference/launch_manual_prompt_ui.sh > "${OUTPUT_DIR}/_manual_ui.log" 2>&1 &
      UI_PID=$!
      trap '[ -n "${UI_PID:-}" ] && kill "$UI_PID" 2>/dev/null || true' EXIT INT TERM
      echo "[ui] open the browser URL printed in ${OUTPUT_DIR}/_manual_ui.log, pick points, click Save."
      echo "[ui] waiting for $MANUAL_POINTS_PATH ..."
      while [ ! -f "$MANUAL_POINTS_PATH" ]; do
        kill -0 "$UI_PID" 2>/dev/null || { echo "ERROR: UI exited before save; see ${OUTPUT_DIR}/_manual_ui.log" >&2; exit 1; }
        sleep 5
      done
      kill "$UI_PID" 2>/dev/null || true
    fi
    run_decode
    ;;
  *) echo "ERROR: STAGE must be prepare|ui|decode|all, got '$STAGE'" >&2; exit 2 ;;
esac
