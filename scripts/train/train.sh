#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 RECIPE_OR_CONFIG OUTPUT_DIR [train.py arguments...]" >&2
    echo "Set CUDA_VISIBLE_DEVICES and NGPU explicitly for multi-GPU training." >&2
    exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
CONFIG="$1"
case "$CONFIG" in
    free|point|mask) CONFIG="configs/train/cola3d/${CONFIG}_q8.json" ;;
    ss_flow|slat_flow|geometry_decoder) CONFIG="configs/train/finetune/${CONFIG}.json" ;;
esac
OUTPUT_DIR="$2"
shift 2
NGPU="${NGPU:-1}"
if [[ ! "$NGPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "NGPU must be a positive integer." >&2
    exit 2
fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" \
    train.py --config "$CONFIG" --output-dir "$OUTPUT_DIR" "$@"
