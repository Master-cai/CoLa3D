#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:?set RUN_DIR to the cached run}"
PORT="${PORT:-18080}"
MAX_PARTS="${MAX_PARTS:-}"
DECODER_CONFIG="${DECODER_CONFIG:-}"
SERVER_NAME="${SERVER_NAME:-127.0.0.1}"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "RUN_DIR is not a directory: $RUN_DIR" >&2
  exit 1
fi
if [[ -z "$DECODER_CONFIG" && -n "${COLA3D_CKPT:-}" ]]; then
  candidate="$(dirname "$(dirname "$COLA3D_CKPT")")/config.json"
  [[ -f "$candidate" ]] && DECODER_CONFIG="$candidate"
fi
ARGS=(--run-dir "$RUN_DIR" --port "$PORT" --server-name "$SERVER_NAME"
      --display-stride "${DISPLAY_STRIDE:-1}")
[[ -n "$MAX_PARTS" ]] && ARGS+=(--max-parts "$MAX_PARTS")
[[ -n "$DECODER_CONFIG" ]] && ARGS+=(--decoder-config "$DECODER_CONFIG")
echo "Manual prompt picker: http://${SERVER_NAME}:${PORT}"
echo "Default capacity: ${MAX_PARTS:-from decoder config or prepare metadata}"
python scripts/inference/ui_manual_prompt_points.py "${ARGS[@]}" "$@"
