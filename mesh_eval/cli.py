"""Evaluate CoLa3D predictions against a benchmark's ground truth."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["instpifu", "metascenes", "metascenes_hard", "replica", "scannetpp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS + ["3dfront"], required=True)
    parser.add_argument("--mode", choices=["free", "point", "mask"], required=True)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "benchmarks")
    parser.add_argument("--pred-root", type=Path, help="Default: outputs/<dataset>/<mode>")
    parser.add_argument("--output", type=Path, help="Default: results/<dataset>/<mode>")
    parser.add_argument("--config", type=Path, help="Default: configs/eval/<dataset>_<mode>.json")
    parser.add_argument("--sample-id", action="append", help="Evaluate only these canonical sample IDs.")
    parser.add_argument("--evaluation-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Parallel CPU processes for sampling and alignment (results are identical to 1).")
    parser.add_argument("--adapter-module", action="append", default=[],
                        help="Import a module that registers extra adapters.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and pairing without computing metrics.")
    args = parser.parse_args()
    args.pred_root = args.pred_root or ROOT / "outputs" / args.dataset / args.mode
    args.output = args.output or ROOT / "results" / args.dataset / args.mode
    if args.dataset == "3dfront":
        args.dataset = "instpifu"
    args.config = args.config or ROOT / "configs" / "eval" / f"{args.dataset}_{args.mode}.json"
    return args


def main() -> None:
    args = parse_args()
    recipe = json.loads(args.config.read_text())
    for extension in args.adapter_module:
        importlib.import_module(extension)
    if recipe["dataset"] != args.dataset or recipe["mode"] != args.mode:
        raise ValueError("--dataset/--mode disagree with the config.")
    from mesh_eval.pipelines.inputs import load_benchmark
    from mesh_eval.pipelines.runner import evaluate_pairs, prepare_pairs

    paths, args.sample_ids, args.manifest_samples = load_benchmark(
        args.benchmark_root, recipe["dataset"], args.sample_id)
    args.gt_root = paths["gt_root"]
    args.metadata = paths.get("metadata")
    args.gt_mask_root = paths.get("gt_mask_root")
    pairs = prepare_pairs(args, recipe)
    if args.dry_run:
        print(json.dumps({"paired_samples": len(pairs)}, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = {"recipe": recipe, "evaluation_seed": args.evaluation_seed, "pred_root": str(args.pred_root),
                "sample_ids": [p.canonical_sample_id for p in pairs],
                "geometry": json.loads((ROOT / recipe["geometry_config"]).read_text())}
    (args.output / "run_config.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps(evaluate_pairs(args, recipe, pairs), indent=2))
