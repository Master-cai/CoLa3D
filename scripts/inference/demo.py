#!/usr/bin/env python3
"""Generate a decomposed scene from one image with Free, Mask, or Point prompts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_models import ensure_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "assets/demo/image.png", help="RGBA image, or RGB image with --mask")
    parser.add_argument("--mode", choices=["free", "mask", "point"], default="free")
    parser.add_argument("--prompt-masks", type=Path, help="Directory of object_*_mask.png files")
    parser.add_argument("--points", type=Path, help="JSON with coordinate_system=latent_grid and points=[[x,y,z], ...]")
    parser.add_argument("--mask", type=Path, help="Foreground mask for an RGB image")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--family", choices=["q8", "q20"], default="q8")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    for path in (args.image, args.mask):
        if path is not None and not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    if args.output is None:
        args.output = ROOT / "outputs/demo" / args.mode
    bundled = args.image.resolve() == (ROOT / "assets/demo/image.png").resolve()
    if args.mode == "mask":
        args.prompt_masks = args.prompt_masks or (ROOT / "assets/demo/masks" if bundled else None)
        if args.prompt_masks is None or not list(args.prompt_masks.glob("object_*_mask.png")):
            parser.error("Mask mode requires --prompt-masks containing object_*_mask.png files")
    if args.mode == "point":
        args.points = args.points or (ROOT / "assets/demo/points.json" if bundled else None)
        if args.points is None or not args.points.is_file():
            parser.error("Point mode requires a --points JSON file")
    config = json.loads((ROOT / f"configs/inference/{args.family}.json").read_text())
    config["settings"]["seed"] = args.seed
    checkpoints = args.checkpoint_root.resolve()
    ensure_models(config, [args.mode, "geometry_decoder", "ss_flow", "slat_flow"] + (["free"] if args.mode == "point" else []), checkpoints)
    command = [sys.executable, str(ROOT / "scripts/inference/run_inference.py"),
               "--dataset-class", "single_image", "--output-dir", str(args.output.resolve()),
               "--decoder-config", str(ROOT / config["model_configs"][args.mode]),
               "--ss-flow-config", str(ROOT / config["model_configs"]["ss_flow"]),
               "--pipeline-config", str(ROOT / config["pipeline_config"])]
    for flag, key in (("cola3d-ckpt", args.mode), ("geo-ckpt", "geometry_decoder"),
                      ("ss-flow-ckpt", "ss_flow"), ("ss-flow-ema-ckpt", "ss_flow"),
                      ("slat-flow-ckpt", "slat_flow")):
        command += ["--" + flag, str(checkpoints / config["checkpoints"][key])]
    if args.mask:
        command += ["--input-rgb", str(args.image.resolve()), "--input-mask", str(args.mask.resolve())]
    else:
        command += ["--input-rgba", str(args.image.resolve())]
    for key, value in config["settings"].items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command += [flag, str(value)]
    if args.mode == "mask":
        command += ["--input-prompt-mask-dir", str(args.prompt_masks.resolve())]
    if args.mode == "point":
        points = json.loads(args.points.read_text())
        if points.get("coordinate_system") != "latent_grid":
            parser.error("Point coordinates must use the latent_grid coordinate system")
        import numpy as np
        requested = np.asarray(points.get("points"), dtype=np.float64)
        if requested.ndim != 2 or requested.shape[1] != 3 or not len(requested) or not np.isfinite(requested).all():
            parser.error("Expected a non-empty finite array of points with shape (N, 3)")
        cache = args.output.resolve() / "scene"
        prepare = [sys.executable, str(Path(__file__).resolve()), "--mode", "free",
                   "--image", str(args.image.resolve()), "--family", args.family,
                   "--output", str(cache), "--checkpoint-root", str(checkpoints), "--seed", str(args.seed)]
        if args.mask:
            prepare += ["--mask", str(args.mask.resolve())]
        result = subprocess.run(prepare, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
        import torch
        from trellis2.datasets.single_image import SingleImageDataset
        scene_id = SingleImageDataset._sanitize_instance_id(args.image.stem)
        scene = cache / scene_id
        latent = torch.load(scene / "latents/shape_slat.pt", map_location="cpu", weights_only=True)
        coords = latent["coords"][:, -3:].numpy()
        selected = [int(np.argmin(((coords - point) ** 2).sum(1))) for point in requested]
        if len(set(selected)) != len(selected):
            raise ValueError("Multiple point prompts map to the same voxel; choose distinct object points")
        torch.save({"prompt_points": torch.from_numpy(coords[selected]).long().unsqueeze(0),
                    "valid_mask": torch.ones((1, len(selected)), dtype=torch.bool)},
                   scene / "manual_prompt_points.pt")
        command += ["--load-shape-slats-from", str(cache), "--manual-prompt-points-from", str(cache),
                    "--prompt-point-source", "manual"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
