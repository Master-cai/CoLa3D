from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional
from mesh_eval.adapters.pred.base import BasePredAdapter
from mesh_eval.adapters.utils import load_scene_hash_mapping, path_or_none
from mesh_eval.adapters.utils.prompt_masks import PromptMaskMatch
from mesh_eval.domain import InputImageRecord, StandardObjectRecord, StandardSampleRecord
logger = logging.getLogger(__name__)
_SHA256_RE = re.compile('^[0-9a-f]{64}$')
_VALID_MESH_VARIANTS = {'raw', 'post_standard', 'post_aggressive', 'post_mask_nms'}

class CoLa3DInstPIFuPredAdapter(BasePredAdapter):
    adapter_name = 'cola3d-instpifu'
    dataset_name = '3dfront-instpifu'

    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.pred_dir = Path(self.config.pred_dir)
        self.sample_ids = set(getattr(self.config, 'sample_ids', None) or [])
        self.metadata_csv = path_or_none(getattr(self.config, 'metadata_csv', None))
        self.scene_hash_to_file_identifier: Dict[str, str] = load_scene_hash_mapping(self.metadata_csv)
        self.mesh_variant: Optional[str] = getattr(self.config, 'mesh_variant', None)
        if self.mesh_variant is not None and self.mesh_variant not in _VALID_MESH_VARIANTS:
            raise ValueError(f"Invalid mesh_variant '{self.mesh_variant}'; expected None or one of {sorted(_VALID_MESH_VARIANTS)}")

    def get_samples(self) -> List[StandardSampleRecord]:
        samples: List[StandardSampleRecord] = []
        for sample_dir in sorted((path for path in self.pred_dir.iterdir() if path.is_dir())):
            scene_hash = sample_dir.name
            canonical_match = re.fullmatch(r'instpifu_([^/]+)_(rendertask\d+)', scene_hash)
            if canonical_match:
                file_identifier = '/'.join(canonical_match.groups())
            elif _SHA256_RE.match(scene_hash):
                file_identifier = self.scene_hash_to_file_identifier.get(scene_hash)
            else:
                logger.debug('Skipping unrecognized InstPIFu directory: %s', sample_dir.name)
                continue
            if file_identifier is None:
                logger.debug('SHA256 %s not found in metadata_csv, skipping', scene_hash)
                continue
            if self.sample_ids and scene_hash not in self.sample_ids and (file_identifier not in self.sample_ids):
                continue
            if '/' in file_identifier:
                scene_id, rendertask_id = file_identifier.split('/', 1)
            else:
                scene_id, rendertask_id = (file_identifier, '')
            if self.mesh_variant is not None:
                variant_dir = sample_dir / 'geometry' / self.mesh_variant
                if not variant_dir.is_dir():
                    logger.warning('%s: geometry/%s/ not found under %s, skipping', scene_hash, self.mesh_variant, sample_dir)
                    continue
            else:
                variant_dir = sample_dir
            num_parts = self._read_num_parts(sample_dir)
            prompt_matches = self._build_prompt_mask_matches(variant_dir=variant_dir, sample_dir=sample_dir, scene_id=scene_id, rendertask_id=rendertask_id)
            objects = self._build_objects(variant_dir, num_parts, prompt_matches=prompt_matches)
            if not objects:
                continue
            if self._requires_object_match_keys() and (not any((obj.match_key for obj in objects))):
                logger.warning('%s: no prompt-mask object matches found, skipping sample', scene_hash)
                continue
            scene_mesh = variant_dir / 'full_scene_colored.glb'
            input_images: List[InputImageRecord] = []
            cond_image = sample_dir / 'cond_image.png'
            if cond_image.is_file():
                input_images.append(InputImageRecord(cond_image, 'cond_image.png'))
            samples.append(StandardSampleRecord(source_sample_id=scene_hash, canonical_sample_id=file_identifier, dataset_name=self.dataset_name, adapter_name=self.adapter_name, objects=objects, scene_mesh_path=scene_mesh if scene_mesh.is_file() else None, sample_dir=sample_dir, asset_root=sample_dir, input_images=input_images, metadata={'sha256': scene_hash, 'scene_id': scene_id, 'rendertask_id': rendertask_id, 'mesh_variant': self.mesh_variant}))
        return samples

    def _build_prompt_mask_matches(self, *, variant_dir: Path, sample_dir: Path, scene_id: str, rendertask_id: str) -> Mapping[int, PromptMaskMatch]:
        return {}

    def _requires_object_match_keys(self) -> bool:
        return False

    @staticmethod
    def _read_num_parts(sample_dir: Path) -> Optional[int]:
        info_path = sample_dir / 'scene_info.json'
        if info_path.is_file():
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                return int(info['num_parts'])
            except (json.JSONDecodeError, KeyError, ValueError):
                logger.warning('Bad scene_info.json: %s', info_path)
        return None

    @staticmethod
    def _build_objects(sample_dir: Path, num_parts: Optional[int], *, prompt_matches: Mapping[int, PromptMaskMatch] | None=None) -> List[StandardObjectRecord]:
        if num_parts is not None:
            part_paths = [sample_dir / f'part_{i}.glb' for i in range(num_parts)]
        else:
            part_paths = sorted(sample_dir.glob('part_*.glb'))
        objects: List[StandardObjectRecord] = []
        for part_path in part_paths:
            if not part_path.is_file():
                continue
            part_id = part_path.stem
            part_match = re.match('^part_(\\d+)$', part_id)
            part_index = int(part_match.group(1)) if part_match else None
            prompt_match = prompt_matches.get(part_index) if prompt_matches is not None and part_index is not None else None
            metadata = {}
            if prompt_match is not None:
                metadata = {'prompt_mask_path': str(prompt_match.prompt_path), 'gt_mask_path': str(prompt_match.gt_mask_path), 'prompt_mask_iou': prompt_match.iou}
            objects.append(StandardObjectRecord(object_id=part_id, display_id=part_id, canonical_name=None, match_key=prompt_match.match_key if prompt_match else None, mesh_path=part_path, metadata=metadata))
        return objects
