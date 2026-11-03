from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import List, Mapping, Optional, Set
from mesh_eval.adapters.pred.base import BasePredAdapter
from mesh_eval.adapters.utils.prompt_masks import PromptMaskMatch
from mesh_eval.domain import InputImageRecord, StandardObjectRecord, StandardSampleRecord
logger = logging.getLogger(__name__)
_SAMPLE_RE = re.compile('^replica_(replica-scan\\d+)_(\\d+)$')
_VALID_MESH_VARIANTS: Set[str] = {'raw', 'post_standard', 'post_aggressive', 'post_mask_nms'}

class CoLa3DReplicaPredAdapter(BasePredAdapter):
    adapter_name = 'cola3d-replica'
    dataset_name = 'replica'

    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.pred_dir = Path(self.config.pred_dir)
        self.checkpoint_name = getattr(self.config, 'checkpoint_name', None)
        if not self.checkpoint_name:
            raise ValueError('cola3d-replica adapter requires checkpoint_name in config')
        self.mesh_variant = getattr(self.config, 'mesh_variant', 'post_standard')
        if self.mesh_variant not in _VALID_MESH_VARIANTS:
            raise ValueError(f"Invalid mesh_variant '{self.mesh_variant}'; expected one of {sorted(_VALID_MESH_VARIANTS)}")
        self.sample_ids = set(getattr(self.config, 'sample_ids', None) or [])

    def get_samples(self) -> List[StandardSampleRecord]:
        samples: List[StandardSampleRecord] = []
        dataset_dir = self.pred_dir
        if not dataset_dir.is_dir():
            raise FileNotFoundError(dataset_dir)
        for sample_dir in sorted((path for path in dataset_dir.iterdir() if path.is_dir())):
            match = _SAMPLE_RE.match(sample_dir.name)
            if match is None:
                logger.debug('Skipping non-sample subdir: %s', sample_dir.name)
                continue
            scene_id, frame_id = (match.group(1), match.group(2))
            canonical_id = f'{scene_id}/{frame_id}'
            if self.sample_ids and canonical_id not in self.sample_ids and (sample_dir.name not in self.sample_ids) and (scene_id not in self.sample_ids):
                continue
            sample = self._build_sample(scene_id=scene_id, frame_id=frame_id, sample_dir=sample_dir)
            if sample is not None:
                samples.append(sample)
        return samples

    def _build_sample(self, *, scene_id: str, frame_id: str, sample_dir: Path) -> Optional[StandardSampleRecord]:
        canonical_id = f'{scene_id}/{frame_id}'
        variant_dir = sample_dir / 'geometry' / self.mesh_variant
        if not variant_dir.is_dir():
            logger.warning('%s: geometry/%s/ not found under %s', canonical_id, self.mesh_variant, sample_dir)
            return None
        prompt_matches = self._build_prompt_mask_matches(variant_dir=variant_dir, sample_dir=sample_dir, scene_id=scene_id, frame_id=frame_id)
        objects = self._build_objects(variant_dir, prompt_matches=prompt_matches)
        if not objects:
            logger.warning('%s: no part_*.glb under %s, skipping sample', canonical_id, variant_dir)
            return None
        if self._requires_object_match_keys() and (not any((obj.match_key for obj in objects))):
            logger.warning('%s: no prompt-mask object matches found, skipping sample', canonical_id)
            return None
        diagnostics: List[str] = []
        scene_mesh = variant_dir / 'full_scene_colored.glb'
        if not scene_mesh.is_file():
            diagnostics.append(f'Missing scene mesh under geometry/{self.mesh_variant}/full_scene_colored.glb')
        input_images: List[InputImageRecord] = []
        cond_image = sample_dir / 'cond_image.png'
        if cond_image.is_file():
            input_images.append(InputImageRecord(cond_image, 'cond_image.png'))
        return StandardSampleRecord(source_sample_id=sample_dir.name, canonical_sample_id=canonical_id, dataset_name=self.dataset_name, adapter_name=self.adapter_name, objects=objects, scene_mesh_path=scene_mesh if scene_mesh.is_file() else None, sample_dir=sample_dir, asset_root=sample_dir, input_images=input_images, metadata={'scene_id': scene_id, 'frame_id': frame_id, 'mesh_variant': self.mesh_variant, 'checkpoint_name': self.checkpoint_name}, diagnostics=diagnostics)

    def _build_prompt_mask_matches(self, *, variant_dir: Path, sample_dir: Path, scene_id: str, frame_id: str) -> Mapping[int, PromptMaskMatch]:
        return {}

    def _requires_object_match_keys(self) -> bool:
        return False

    @staticmethod
    def _build_objects(variant_dir: Path, *, prompt_matches: Mapping[int, PromptMaskMatch] | None=None) -> List[StandardObjectRecord]:
        objects: List[StandardObjectRecord] = []
        part_paths = sorted(variant_dir.glob('part_*.glb'), key=lambda path: int(path.stem.split('_', 1)[1]) if path.stem.split('_', 1)[1].isdigit() else 10 ** 9)
        for part_path in part_paths:
            part_id = part_path.stem
            part_match = re.match('^part_(\\d+)$', part_id)
            part_index = int(part_match.group(1)) if part_match else None
            prompt_match = prompt_matches.get(part_index) if prompt_matches is not None and part_index is not None else None
            metadata = {}
            if prompt_match is not None:
                metadata = {'prompt_mask_path': str(prompt_match.prompt_path), 'gt_mask_path': str(prompt_match.gt_mask_path), 'prompt_mask_iou': prompt_match.iou}
            objects.append(StandardObjectRecord(object_id=part_id, display_id=part_id, canonical_name=None, match_key=prompt_match.match_key if prompt_match else None, mesh_path=part_path, metadata=metadata))
        return objects
