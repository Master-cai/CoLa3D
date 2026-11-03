from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from mesh_eval.adapters.gt.base import BaseGTAdapter
from mesh_eval.domain import InputImageRecord, StandardObjectRecord, StandardSampleRecord
logger = logging.getLogger(__name__)
_OBJ_FILE_RE = re.compile('^(\\d+)_(.+)\\.glb$')

class MetaScenesGTAdapter(BaseGTAdapter):
    adapter_name = 'metascenes-gt'
    dataset_name = 'metascenes'

    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.gt_data_root = Path(self.config.gt_data_root)
        self.sample_ids = set(getattr(self.config, 'sample_ids', None) or [])

    def get_samples(self) -> List[StandardSampleRecord]:
        anno_root = self.gt_data_root / 'anno'
        if not anno_root.is_dir():
            logger.warning('MetaScenes anno directory not found: %s', anno_root)
            return []
        samples: List[StandardSampleRecord] = []
        for view_dir in sorted(anno_root.iterdir()):
            if not view_dir.is_dir():
                continue
            sample_id = view_dir.name
            if len(sample_id) != 4 or not sample_id.isdecimal():
                continue
            if self.sample_ids and sample_id not in self.sample_ids:
                continue
            sample = self._build_sample(sample_id, view_dir)
            if sample is not None:
                samples.append(sample)
        return samples

    def _build_sample(self, canonical_id: str, view_dir: Path) -> Optional[StandardSampleRecord]:
        objects_dir = view_dir / 'objects'
        if not objects_dir.is_dir():
            logger.warning('%s: objects/ not found in %s', canonical_id, view_dir)
            return None
        diagnostics: List[str] = []
        objects: List[StandardObjectRecord] = []
        for mesh_path in sorted(objects_dir.glob('*.glb'), key=lambda path: int(_OBJ_FILE_RE.match(path.name).group(1)) if _OBJ_FILE_RE.match(path.name) else 10 ** 9):
            match = _OBJ_FILE_RE.match(mesh_path.name)
            if match is None:
                diagnostics.append(f'Unparsable GLB filename: {mesh_path.name}')
                continue
            inst_id = match.group(1)
            category = match.group(2)
            objects.append(StandardObjectRecord(object_id=inst_id, display_id=f'{inst_id}_{category}', canonical_name=category, match_key=inst_id, mesh_path=mesh_path, metadata={'inst_id': inst_id, 'category': category}))
        if not objects:
            logger.debug('No GT objects in %s', view_dir)
            return None
        input_images: List[InputImageRecord] = []
        for filename in ('rgb.png', 'instance_mask.png'):
            img_path = view_dir / filename
            if img_path.is_file():
                input_images.append(InputImageRecord(img_path, filename))
        sample_metadata: Dict[str, Any] = {}
        instance_mask_path = view_dir / 'instance_mask.png'
        if instance_mask_path.is_file():
            sample_metadata['instance_mask_path'] = str(instance_mask_path)
        inst_ids_full = self._load_inst_ids_full(view_dir)
        if inst_ids_full:
            sample_metadata['inst_ids_full'] = inst_ids_full
        sample_metadata['inst_ids_per_object'] = [obj.metadata.get('inst_id') for obj in objects]
        return StandardSampleRecord(source_sample_id=canonical_id, canonical_sample_id=canonical_id, dataset_name=self.dataset_name, adapter_name=self.adapter_name, objects=objects, scene_mesh_path=None, sample_dir=view_dir, asset_root=view_dir, input_images=input_images, render_config=None, metadata=sample_metadata, diagnostics=diagnostics)

    def _load_inst_ids_full(self, view_dir: Path) -> Optional[List[str]]:
        metadata_path = view_dir / 'metadata.json'
        if not metadata_path.is_file():
            return None
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        inst_ids = meta.get('inst_ids')
        if not inst_ids:
            return None
        return [str(x) for x in inst_ids]
