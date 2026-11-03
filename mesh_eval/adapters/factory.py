
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from mesh_eval.adapters.registry import GT_ADAPTERS, PRED_ADAPTERS


@dataclass
class GTConfig:
    gt_scene_dir: Path
    gt_data_root: Path
    sample_ids: list[str]
    dataset_name: str = "scannetpp"
    scannetpp_root: Path | None = None
    normal_render_max_dim: int = 0
    options: dict[str, Any] = field(default_factory=dict)

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.gt_data_root / path


@dataclass
class PredictionConfig:
    pred_dir: Path
    sample_ids: list[str]
    gt_data_root: Path
    mesh_variant: str = 'post_standard'
    prompt_match_min_iou: float = 0.5
    metadata_csv: Path | None = None
    prompt_gt_mask_root: Path | None = None
    checkpoint_name: str = 'cola3d'
    options: dict[str, Any] = field(default_factory=dict)

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.pred_dir / path


def build_adapters(recipe: dict[str, Any], paths: dict[str, Any], sample_ids: list[str]) -> tuple[Any, Any]:
    gt_config = GTConfig(gt_scene_dir=paths['gt_root'],gt_data_root=paths['gt_root'],sample_ids=sample_ids,dataset_name=recipe['dataset'],options=dict(recipe.get('gt_options',{})))
    pred_config = PredictionConfig(pred_dir=paths['pred_root'],sample_ids=sample_ids,mesh_variant=recipe['mesh_variant'],metadata_csv=paths.get('metadata'),prompt_gt_mask_root=paths.get('gt_mask_root'),gt_data_root=paths['gt_root'],prompt_match_min_iou=recipe['prompt_match_min_iou'],options=dict(recipe.get('pred_options',{})))
    return GT_ADAPTERS[recipe['gt_adapter']].build(gt_config), PRED_ADAPTERS[recipe['pred_adapter']].build(pred_config)
