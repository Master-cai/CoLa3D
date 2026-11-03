#!/usr/bin/env bash


set -euo pipefail

DATASET="${1:?dataset: InstPIFu_3D_FRONT or MIDI_3D_FRONT}"
DATA_ROOT="${2:?data root, e.g. datasets/InstPIFu-3D-FRONT/Train}"
RESOLUTION="${3:-1024}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
SHAPE_NAME="shape_enc_next_dc_f16c32_fp16_${RESOLUTION}"

metadata() { "$PYTHON" data_toolkit/build_metadata.py "$DATASET" --root "$DATA_ROOT"; }

if [[ "$DATASET" == InstPIFu_3D_FRONT ]]; then
    "$PYTHON" data_toolkit/prepare_instpifu_images.py --root "$DATA_ROOT" --max-workers "${MAX_WORKERS:-4}"
fi

if [[ "$RESOLUTION" == 1024 ]]; then
    : "${BLENDER_PATH:?set BLENDER_PATH to a Blender 4.5 executable}"
    metadata
    "$PYTHON" data_toolkit/dump_mesh.py "$DATASET" --root "$DATA_ROOT" --max_workers "${MAX_WORKERS:-4}"
    metadata
elif [[ "$RESOLUTION" != 512 ]]; then
    echo "Resolution must be 1024 or 512" >&2
    exit 2
fi

"$PYTHON" data_toolkit/dual_grid_multi_obj.py "$DATASET" --root "$DATA_ROOT" --resolution "$RESOLUTION" --max_workers 1
metadata
"$PYTHON" data_toolkit/encode_shape_latent_multi_obj.py --root "$DATA_ROOT" --resolution "$RESOLUTION"
metadata

if [[ "$RESOLUTION" == 1024 ]]; then
    "$PYTHON" data_toolkit/encode_utonia_features_multi_obj.py \
        --root "$DATA_ROOT" --shape_latent_root "$DATA_ROOT/shape_latents/$SHAPE_NAME"
    "$PYTHON" data_toolkit/encode_ss_latent.py --root "$DATA_ROOT" \
        --shape_latent_name "$SHAPE_NAME" --resolution 64
    metadata
fi
