from __future__ import annotations
import logging
from pathlib import Path
from typing import Mapping
from mesh_eval.adapters.pred.cola3d_metascenes import CoLa3DMetaScenesPredAdapter
from mesh_eval.adapters.utils import path_or_none
from mesh_eval.adapters.utils.prompt_masks import PromptMaskMatch, match_metascenes_prompt_masks
logger = logging.getLogger(__name__)

class CoLa3DMetaScenesMaskPredAdapter(CoLa3DMetaScenesPredAdapter):
    adapter_name = 'cola3d-metascenes-mask'

    def __init__(self, config: object) -> None:
        super().__init__(config)
        gt_root = path_or_none(getattr(self.config, 'gt_data_root', None))
        if gt_root is None:
            raise ValueError('cola3d-metascenes-mask adapter requires gt_data_root in config (used to locate per-view instance_mask.png + metadata.json)')
        self.gt_data_root = gt_root
        self.prompt_match_min_iou = float(getattr(self.config, 'prompt_match_min_iou', 0.5))

    def _requires_object_match_keys(self) -> bool:
        return True

    def _build_prompt_mask_matches(self, *, variant_dir: Path, sample_dir: Path, scene_id: str, frame_id: str) -> Mapping[int, PromptMaskMatch]:
        view_dir = self.gt_data_root / 'anno' / scene_id
        if not view_dir.is_dir():
            logger.warning('MetaScenes view dir not found: %s', view_dir)
            return {}
        matches = match_metascenes_prompt_masks(variant_dir=variant_dir, view_dir=view_dir, min_iou=self.prompt_match_min_iou)
        if not matches:
            logger.warning('%s/%s: no prompt_i.png matched MetaScenes GT instances under %s', scene_id, frame_id, view_dir)
        return matches
