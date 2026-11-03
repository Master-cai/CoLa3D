from __future__ import annotations

from pathlib import Path
from typing import Any

DATASET_NAMES = {"instpifu": "InstPIFu", "metascenes": "MetaScenes", "metascenes_hard": "MetaScenes_Hard",
                 "replica": "Replica", "scannetpp": "ScanNetpp"}


def discover_instpifu(root: Path) -> list[dict[str, Any]]:
    samples = []
    for image in sorted((root / 'InstPIFu/raw/renders').glob('*/*/render_0000.png')):
        scene, frame = image.parent.parent.name, image.parent.name
        canonical = f'{scene}/{frame}'
        objects = [{'id': str(int(mesh.parent.name.split('_', 1)[0])),
                    'mesh': mesh.relative_to(root).as_posix()}
                   for mesh in sorted((root / 'InstPIFu/gt' / scene / 'renders' / frame / 'objects').glob('[0-9][0-9][0-9]_*/object_world.glb'))]
        masks = sorted((root / 'InstPIFu/raw/masks' / canonical).glob('*.png'))
        if not objects or not masks:
            raise ValueError(f'{canonical}: missing InstPIFu GT objects or visible masks')
        mask_ids = [str(int(path.name.split('_', 1)[0])) for path in masks]
        if len(mask_ids) != len(set(mask_ids)) or set(mask_ids) != {obj['id'] for obj in objects}:
            raise ValueError(f'{canonical}: visible mask IDs must match GT object IDs exactly')
        samples.append({'id': f'instpifu_{scene}_{frame}', 'dataset': 'InstPIFu',
                        'canonical_sample_id': canonical, 'image': image.relative_to(root).as_posix(),
                        'mask_format': 'instpifu_visible_masks',
                        'prompt_masks': [path.relative_to(root).as_posix() for path in masks],
                        'gt': {'objects': objects}})
    return samples


def discover_benchmark(root: Path, dataset_name: str | None = None) -> dict[str, Any]:
    root = Path(root)
    configs = {
        "InstPIFu": {"gt_scene_dir": "InstPIFu/gt", "prompt_gt_mask_root": "InstPIFu/raw/masks"},
        "MetaScenes": {"gt_data_root": "MetaScenes"},
        "MetaScenes_Hard": {"gt_data_root": "MetaScenes_Hard"},
        "Replica": {"gt_data_root": "Replica", "prompt_gt_mask_root": "Replica/instance_masks"},
        "ScanNetpp": {"gt_data_root": "ScanNetpp", "prompt_gt_mask_root": "ScanNetpp/instance_masks"},
    }
    if dataset_name is not None and dataset_name not in configs:
        raise ValueError(f"Unknown benchmark dataset: {dataset_name}")
    samples = []
    datasets = {}
    for name, config in configs.items():
        if dataset_name is not None and name != dataset_name:
            continue
        selected = []
        if name == "InstPIFu":
            selected = discover_instpifu(root)
        elif name.startswith("MetaScenes"):
            for image in sorted((root / name / "anno").glob("*/rgb.png")):
                key = image.parent.name
                if len(key) != 4 or not key.isdecimal():
                    raise ValueError(f"Invalid benchmark sample directory: {image.parent}")
                base = f"{name}/anno/{key}"
                objects = [{"id": mesh.stem.split("_", 1)[0], "mesh": mesh.relative_to(root).as_posix()}
                           for mesh in sorted((image.parent / "objects").glob("*.glb"),
                                              key=lambda path: int(path.stem.split("_", 1)[0]))]
                selected.append({"id": f"{name.lower()}_{key}", "dataset": name,
                                 "canonical_sample_id": key, "image": f"{base}/rgb.png",
                                 "instance_mask": f"{base}/instance_mask.png", "metadata": f"{base}/metadata.json",
                                 "mask_format": "metascenes_palette", "gt": {"objects": objects}})
        else:
            directory = config["gt_data_root"]
            images = sorted(path for path in (root / directory / "rgb").glob("*/*")
                            if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
            for image in images:
                scene, frame = image.parent.name, image.stem
                base = f"{directory}/GTmesh/{scene}"
                masks = sorted((root / directory / "instance_masks" / scene / frame).glob("obj_*.png"))
                objects = [{"id": mask.stem.removeprefix("obj_"), "mesh": f"{base}/{mask.stem}.ply",
                            "mask": mask.relative_to(root).as_posix()} for mask in masks]
                selected.append({"id": f"{name.lower()}_{scene}_{frame}", "dataset": name,
                                 "canonical_sample_id": f"{scene}/{frame}", "image": image.relative_to(root).as_posix(),
                                 "instance_mask": f"{directory}/masks/{scene}/{frame}.png",
                                 "metadata": f"{base}/manifest.json", "mask_format": "positive_id",
                                 "gt": {"objects": objects}})
        canonical = [sample["canonical_sample_id"] for sample in selected]
        if not selected or len(canonical) != len(set(canonical)):
            raise ValueError(f"{name}: benchmark is empty or has duplicate sample IDs")
        datasets[name] = dict(config, sample_count=len(selected))
        samples.extend(selected)
    return {"schema_version": 1, "datasets": datasets, "samples": samples}
