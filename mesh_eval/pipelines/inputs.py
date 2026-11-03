
from __future__ import annotations

from pathlib import Path
from typing import Any

from mesh_eval.benchmark import discover_benchmark

DATASET_NAMES = {"instpifu": "InstPIFu", "metascenes": "MetaScenes", "metascenes_hard": "MetaScenes_Hard",
                 "replica": "Replica", "scannetpp": "ScanNetpp"}


def asset_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Asset paths must be relative to the benchmark root: {relative}")
    return root / path


def load_benchmark(benchmark_root: Path, dataset: str, sample_ids: list[str] | None = None
                   ) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    name = DATASET_NAMES[dataset]
    manifest = discover_benchmark(benchmark_root, name)
    config = manifest["datasets"][name]
    samples = [sample for sample in manifest["samples"] if sample["dataset"] == name]
    canonical = [sample["canonical_sample_id"] for sample in samples]
    if len(samples) != config["sample_count"] or len(canonical) != len(set(canonical)):
        raise ValueError(f"{name}: sample count or uniqueness mismatch in benchmark assets.")
    if sample_ids:
        if len(sample_ids) != len(set(sample_ids)) or not set(sample_ids) <= set(canonical):
            raise ValueError("Requested sample IDs must be unique members of the benchmark.")
        mapping = {sample["canonical_sample_id"]: sample for sample in samples}
        samples = [mapping[sample_id] for sample_id in sample_ids]
        canonical = list(sample_ids)
    for sample in samples:
        objects = sample["gt"]["objects"]
        if not objects or len({str(obj["id"]) for obj in objects}) != len(objects):
            raise ValueError(f"Invalid GT object list: {sample['canonical_sample_id']}")
        files = [obj.get(key) for obj in objects for key in ("mesh", "mask")]
        files += [sample.get(key) for key in ("image", "instance_mask", "metadata")]
        for relative in filter(None, files):
            if not asset_path(benchmark_root, relative).is_file():
                raise FileNotFoundError(asset_path(benchmark_root, relative))
    paths = {"gt_root": asset_path(benchmark_root, config.get("gt_scene_dir", config.get("gt_data_root")))}
    if config.get("prompt_gt_mask_root"):
        paths["gt_mask_root"] = asset_path(benchmark_root, config["prompt_gt_mask_root"])
    if config.get("processed_root"):
        paths["metadata"] = asset_path(benchmark_root, config["processed_root"]) / "metadata.csv"
    return paths, canonical, samples
