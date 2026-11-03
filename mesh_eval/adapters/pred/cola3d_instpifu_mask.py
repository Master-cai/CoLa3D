from __future__ import annotations
import logging
from pathlib import Path
from typing import Mapping
from mesh_eval.adapters.pred.cola3d_instpifu import CoLa3DInstPIFuPredAdapter
from mesh_eval.adapters.utils import path_or_none
from mesh_eval.adapters.utils.prompt_masks import PromptMaskMatch, match_instpifu_prompt_masks
logger = logging.getLogger(__name__)

class CoLa3DInstPIFuMaskPredAdapter(CoLa3DInstPIFuPredAdapter):
    adapter_name = 'cola3d-instpifu-mask'

    def __init__(self, config: object) -> None:
        super().__init__(config)
        prompt_root = path_or_none(getattr(self.config, 'prompt_gt_mask_root', None))
        if prompt_root is None:
            if self.metadata_csv is None:
                raise ValueError('prompt_gt_mask_root is required when metadata_csv is not set')
            prompt_root = self.metadata_csv.parent / 'raw' / 'masks'
        self.prompt_gt_mask_root = prompt_root
        self.prompt_match_min_iou = float(getattr(self.config, 'prompt_match_min_iou', 0.5))

    def _requires_object_match_keys(self) -> bool:
        return True

    def _build_prompt_mask_matches(self, *, variant_dir: Path, sample_dir: Path, scene_id: str, rendertask_id: str) -> Mapping[int, PromptMaskMatch]:
        gt_mask_dir = self.prompt_gt_mask_root / scene_id / rendertask_id
        if not gt_mask_dir.is_dir():
            logger.warning('GT prompt-mask directory not found: %s', gt_mask_dir)
            return {}
        matches = match_instpifu_prompt_masks(variant_dir=variant_dir, gt_mask_dir=gt_mask_dir, min_iou=self.prompt_match_min_iou)
        if not matches:
            logger.warning('%s/%s: no prompt_i.png matched GT masks under %s', scene_id, rendertask_id, gt_mask_dir)
        return matches
