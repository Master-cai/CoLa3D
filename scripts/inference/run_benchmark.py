#!/usr/bin/env python3
"""Run CoLa3D inference on one benchmark dataset and query mode.

Free generates each scene from its image. Point and Mask reuse Free latents
unless the dataset config requests independent generation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from mesh_eval.benchmark import discover_benchmark
DATASET_NAMES = {"instpifu": "InstPIFu", "metascenes": "MetaScenes", "metascenes_hard": "MetaScenes_Hard",
                 "replica": "Replica", "scannetpp": "ScanNetpp"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def verify_glb(path: Path) -> None:
    import struct
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"Invalid GLB: {path}")
    length = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20:20 + length])
    primitives = [p for mesh in doc.get("meshes", []) for p in mesh.get("primitives", [])]
    if not any(doc["accessors"][p["attributes"]["POSITION"]]["count"] >= 3 for p in primitives):
        raise ValueError(f"Mesh has no geometry: {path}")


def expected_part_files(info: dict[str, Any], min_tokens: int) -> set[str]:

    parts = info.get("parts")
    if not isinstance(parts, list) or len(parts) != info.get("num_parts"):
        raise ValueError("Incomplete part metadata in scene_info.json")
    if sorted(part["part_id"] for part in parts) != list(range(len(parts))):
        raise ValueError("Duplicate or missing part IDs in scene_info.json")
    return {f"part_{part['part_id']}.glb" for part in parts
            if part["voxels"] >= min_tokens and part.get("empty_decode") is not True}


def verify_scene(root: Path, sample_id: str, min_tokens: int) -> None:
    scene = root / sample_id
    info = read_json(scene / "scene_info.json")
    if info.get("sha256") != sample_id or int(info.get("num_parts", 0)) <= 0:
        raise ValueError(f"Not a completed inference result: {scene}")
    expected = expected_part_files(info, min_tokens)
    if not expected:
        raise ValueError(f"Scene has no decodable parts: {scene}")
    directory = scene / "geometry" / "post_standard"
    actual = {path.name for path in directory.glob("part_*.glb")}
    if actual != expected:
        raise ValueError(f"{directory}: missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}")
    for name in sorted(expected):
        verify_glb(directory / name)
    for path in (scene / "latents/shape_slat.pt", scene / "segmentation.npz"):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing inference artifact: {path}")


def resolve_settings(config: dict[str, Any], dataset: str, mode: str) -> dict[str, Any]:
    settings = dict(config["settings"])
    dataset_settings = dict(config["datasets"][dataset])
    point_settings = dataset_settings.pop("point", {})
    dataset_settings.pop("reuse_free_latent", None)
    settings.update(dataset_settings)
    if mode == "point":
        settings.update(config.get("point_settings", {}))
        settings.update(point_settings)
    return settings


def build_command(args: argparse.Namespace, config: dict[str, Any], samples: list[dict[str, Any]],
                  output: Path, selection: Path) -> list[str]:
    checkpoints = {key: args.checkpoint_root / name for key, name in config["checkpoints"].items()}
    generate_latent = args.mode == "free" or not config["datasets"][args.dataset].get("reuse_free_latent", True)
    needed = [args.mode, "geometry_decoder"] + (["ss_flow", "slat_flow"] if generate_latent else [])
    from download_models import ensure_models
    ensure_models(config, needed, args.checkpoint_root)
    command = [sys.executable, str(REPOSITORY / "scripts/inference/run_inference.py"),
               "--cola3d-ckpt", str(checkpoints[args.mode]),
               "--geo-ckpt", str(checkpoints["geometry_decoder"]),
               "--decoder-config", str(REPOSITORY / config["model_configs"][args.mode]),
               "--ss-flow-config", str(REPOSITORY / config["model_configs"]["ss_flow"]),
               "--pipeline-config", str(REPOSITORY / config["pipeline_config"]),
               "--output-dir", str(output), "--dataset-instances", str(selection),
               "--summary-name", f"summary_shard{args.shard_index}.json"]
    if generate_latent:
        command += ["--ss-flow-ckpt", str(checkpoints["ss_flow"]), "--ss-flow-ema-ckpt", str(checkpoints["ss_flow"]),
                    "--slat-flow-ckpt", str(checkpoints["slat_flow"])]
    else:
        free_root = args.output_root / getattr(args, "output_dataset", args.dataset) / "free"
        for sample in samples:
            if not (free_root / sample["id"] / "latents/shape_slat.pt").is_file():
                raise FileNotFoundError(f"Run --mode free first: no scene latent for {sample['id']} in {free_root}")
        command += ["--load-shape-slats-from", str(free_root)]
    name = DATASET_NAMES[args.dataset]
    command += ["--dataset-class", "benchmark_views",
                "--benchmark-root", str(args.benchmark_root), "--benchmark-dataset", name]
    for key, value in resolve_settings(config, args.dataset, args.mode).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command += [flag, str(value)]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Override the configuration selected for the dataset")
    parser.add_argument("--dataset", choices=sorted(set(DATASET_NAMES) | {"3dfront"}), required=True)
    parser.add_argument("--mode", choices=["free", "point", "mask"], required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=REPOSITORY / "checkpoints")
    parser.add_argument("--benchmark-root", type=Path, default=REPOSITORY / "benchmarks")
    parser.add_argument("--output-root", type=Path, default=REPOSITORY / "outputs")
    parser.add_argument("--sample-id", action="append", help="Run only these sample IDs.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    args.output_dataset = args.dataset
    if args.dataset == "3dfront":
        args.dataset = "instpifu"
    for key in ("checkpoint_root", "benchmark_root", "output_root"):
        setattr(args, key, getattr(args, key).resolve())

    if args.config is None:
        args.config = REPOSITORY / "configs/inference" / ("q20.json" if args.dataset == "metascenes_hard" else "q8.json")
    config = read_json(args.config)
    if args.dataset not in config["datasets"]:
        raise ValueError(f"{args.config.name} does not cover {args.dataset}; choices: {sorted(config['datasets'])}")
    manifest = discover_benchmark(args.benchmark_root, DATASET_NAMES[args.dataset])
    samples = [s for s in manifest["samples"] if s["dataset"] == DATASET_NAMES[args.dataset]]
    if args.sample_id:
        unknown = set(args.sample_id) - {s["id"] for s in samples}
        if unknown:
            raise ValueError(f"Unknown sample IDs: {sorted(unknown)}")
        samples = [s for s in samples if s["id"] in set(args.sample_id)]
    samples = samples[args.shard_index::args.num_shards]
    if not samples:
        raise ValueError("No samples selected")

    output = args.output_root / args.output_dataset / args.mode
    output.mkdir(parents=True, exist_ok=True)
    selection = output / f".selection_shard{args.shard_index}.txt"
    selection.write_text("\n".join(s["id"] for s in samples) + "\n")
    command = build_command(args, config, samples, output, selection)
    env = dict(os.environ, PYTHONPATH=f"{REPOSITORY}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    result = subprocess.run(command, cwd=REPOSITORY, env=env, check=False)
    min_tokens = int(resolve_settings(config, args.dataset, args.mode)["min_tokens"])
    failed = []
    for sample in samples:
        try:
            verify_scene(output, sample["id"], min_tokens)
        except Exception as exc:
            failed.append(f"{sample['id']}: {exc}")
    print(f"{args.dataset}/{args.mode}: {len(samples) - len(failed)}/{len(samples)} scenes complete -> {output}")
    for line in failed:
        print(f"  FAILED {line}")
    return 1 if failed or result.returncode else 0


if __name__ == "__main__":
    sys.exit(main())
