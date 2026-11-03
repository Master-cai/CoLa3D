<a id="training"></a>

# 🏋️ Training

## 🛠️ 1. Prepare the environment

```bash
./setup.sh --new-env --all --data
conda activate "$PWD/.env"
hf download microsoft/TRELLIS.2-4B --local-dir pretrained_ckpts
export BLENDER_PATH="/path/to/blender"  # Blender 4.5
```

## 📦 2. Download and prepare data

We use the 3D-FRONT data prepared by [MIDI-3D](https://github.com/VAST-AI-Research/MIDI-3D)
and [InstPIFu](https://github.com/GAP-LAB-CUHK-SZ/InstPIFu).
Download the prepared training assets:

```bash
hf download mastercai/CoLa3D-Training --repo-type dataset --include 'archives/*.tar.zst' --local-dir .
python scripts/data/extract_training.py
```

### ⚙️ Preprocess

Generate training inputs:

```bash
./scripts/data/prepare.sh InstPIFu_3D_FRONT datasets/InstPIFu-3D-FRONT/Train 1024
./scripts/data/prepare.sh MIDI_3D_FRONT datasets/MIDI-3D-FRONT/Train 1024
```

For geometry-decoder fine-tuning, also generate the 512-resolution inputs:

```bash
./scripts/data/prepare.sh InstPIFu_3D_FRONT datasets/InstPIFu-3D-FRONT/Train 512
./scripts/data/prepare.sh MIDI_3D_FRONT datasets/MIDI-3D-FRONT/Train 512
```

## 🔧 3. Configs

The training launcher accepts these recipe names, or an explicit config path.

| Recipe | Trains |
|---|---|
| `free` | Free decomposition head |
| `point` | Point decomposition head |
| `mask` | Mask decomposition head |
| `geometry_decoder` | Geometry decoder at resolution 512 |
| `ss_flow` | Sparse-structure flow |
| `slat_flow` | Shape latent flow |

## 🏋️ 4. Train

```sh
scripts/train/train.sh "<recipe-or-config>" "<output_dir>" [train.py arguments]
```

Set `CUDA_VISIBLE_DEVICES` and `NGPU` for multi-GPU training. Examples:

```sh
NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train/train.sh free runs/free
scripts/train/train.sh geometry_decoder runs/geometry_decoder
scripts/train/train.sh ss_flow runs/ss_flow
scripts/train/train.sh slat_flow runs/slat_flow
```

Each run writes `config.json` and a `ckpts/` directory. Use the EMA checkpoint
(`*_ema0.9999_step*.pt`) for inference.

Set your trained checkpoint paths in `configs/inference/` for inference.
