from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import List, Optional
from mesh_eval.adapters.gt.base import BaseGTAdapter
from mesh_eval.adapters.utils import path_or_none
from mesh_eval.domain import InputImageRecord, StandardObjectRecord, StandardSampleRecord
logger = logging.getLogger(__name__)
OBJ_DIR_RE = re.compile('^(\\d{3})_(.+)$')

class InstPIFuGTAdapter(BaseGTAdapter):
    adapter_name = 'instpifu'
    dataset_name = '3dfront-instpifu'

    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.gt_scene_dir = Path(self.config.gt_scene_dir)
        self.sample_list_path = path_or_none(getattr(self.config, 'sample_list_path', None))
        self.sample_ids = set(getattr(self.config, 'sample_ids', None) or [])

    def _load_sample_list(self) -> List[str]:
        entries: List[str] = []
        for line in self.sample_list_path.read_text(encoding='utf-8').strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split('/')
            if len(parts) == 3 and parts[1] == 'renders':
                entries.append(f'{parts[0]}/{parts[2]}')
            else:
                entries.append(line)
        return entries

    def get_samples(self) -> List[StandardSampleRecord]:
        if self.sample_list_path and self.sample_list_path.is_file():
            return self._get_samples_from_list()
        return self._get_samples_by_scan()

    def _get_samples_from_list(self) -> List[StandardSampleRecord]:
        sample_keys = self._load_sample_list()
        samples: List[StandardSampleRecord] = []
        for canonical_id in sample_keys:
            if self.sample_ids and canonical_id not in self.sample_ids:
                continue
            scene_id, rendertask_id = canonical_id.split('/', 1)
            rt_dir = self.gt_scene_dir / scene_id / 'renders' / rendertask_id
            if not rt_dir.is_dir():
                logger.debug('GT rendertask dir not found: %s', rt_dir)
                continue
            sample = self._build_sample(scene_id, rendertask_id, rt_dir)
            if sample is not None:
                samples.append(sample)
        return samples

    def _get_samples_by_scan(self) -> List[StandardSampleRecord]:
        samples: List[StandardSampleRecord] = []
        for scene_dir in sorted(self.gt_scene_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            renders_dir = scene_dir / 'renders'
            if not renders_dir.is_dir():
                continue
            scene_id = scene_dir.name
            for rt_dir in sorted(renders_dir.iterdir()):
                if not rt_dir.is_dir():
                    continue
                rendertask_id = rt_dir.name
                canonical_id = f'{scene_id}/{rendertask_id}'
                if self.sample_ids and canonical_id not in self.sample_ids:
                    continue
                sample = self._build_sample(scene_id, rendertask_id, rt_dir)
                if sample is not None:
                    samples.append(sample)
        return samples

    def _build_sample(self, scene_id: str, rendertask_id: str, rt_dir: Path) -> Optional[StandardSampleRecord]:
        canonical_id = f'{scene_id}/{rendertask_id}'
        objects_dir = rt_dir / 'objects'
        if not objects_dir.is_dir():
            logger.warning('objects not found: %s', objects_dir)
            return None
        objects: List[StandardObjectRecord] = []
        for obj_dir in sorted(objects_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            m = OBJ_DIR_RE.match(obj_dir.name)
            if not m:
                continue
            idx = int(m.group(1))
            class_name = m.group(2)
            mesh_path = obj_dir / 'object_world.glb'
            if not mesh_path.is_file():
                logger.debug('GT mesh not found: %s', mesh_path)
                continue
            objects.append(StandardObjectRecord(object_id=str(idx), display_id=f'{idx}_{class_name}', canonical_name=class_name, match_key=str(idx), mesh_path=mesh_path))
        if not objects:
            logger.debug('No GT objects for %s', canonical_id)
            return None
        input_images: List[InputImageRecord] = []
        for filename in ('rgb.png', 'rgb_masked_visible_raw_desc.png'):
            img_path = rt_dir / filename
            if img_path.is_file():
                input_images.append(InputImageRecord(img_path, filename))
        scene_mesh = rt_dir / 'visible_scene_raw_desc.glb'
        return StandardSampleRecord(source_sample_id=canonical_id, canonical_sample_id=canonical_id, dataset_name=self.dataset_name, adapter_name=self.adapter_name, objects=objects, scene_mesh_path=scene_mesh if scene_mesh.is_file() else None, sample_dir=rt_dir, asset_root=rt_dir, input_images=input_images, metadata={'scene_id': scene_id, 'rendertask_id': rendertask_id})
