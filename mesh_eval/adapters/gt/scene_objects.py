from __future__ import annotations
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from mesh_eval.adapters.gt.base import BaseGTAdapter
from mesh_eval.domain import InputImageRecord, StandardObjectRecord, StandardSampleRecord
logger = logging.getLogger(__name__)
_OBJ_MASK_RE = re.compile('^obj_(\\d+)\\.png$')

class SceneObjectsGTAdapter(BaseGTAdapter):
    adapter_name = 'scene-objects-gt'

    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.dataset_name = self.config.dataset_name
        self.gt_data_root = Path(self.config.gt_data_root)
        self.sample_ids = set(getattr(self.config, 'sample_ids', None) or [])
        self._manifest_cache: Dict[str, Dict[int, dict]] = {}
        scannetpp_root = getattr(self.config, 'scannetpp_root', None)
        self.scannetpp_root = Path(scannetpp_root) if scannetpp_root else None
        self.normal_render_max_dim = int(getattr(self.config, 'normal_render_max_dim', 0) or 0)
        self._transforms_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def get_samples(self) -> List[StandardSampleRecord]:
        instance_root = self.gt_data_root / 'instance_masks'
        if not instance_root.is_dir():
            logger.warning('instance_masks directory not found: %s', instance_root)
            return []
        samples: List[StandardSampleRecord] = []
        for scene_dir in sorted(instance_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_id = scene_dir.name
            manifest = self._load_manifest(scene_id)
            if manifest is None:
                continue
            for frame_dir in sorted(scene_dir.iterdir()):
                if not frame_dir.is_dir():
                    continue
                frame_id = frame_dir.name
                if not self._sample_requested(scene_id, frame_id):
                    continue
                sample = self._build_sample(scene_id, frame_id, manifest)
                if sample is not None:
                    samples.append(sample)
        return samples

    def _sample_requested(self, scene_id: str, frame_id: str) -> bool:
        if not self.sample_ids:
            return True
        source_id = f'{scene_id}/{frame_id}'
        return source_id in self.sample_ids or scene_id in self.sample_ids

    def _load_transforms(self, scene_id: str) -> Optional[Dict[str, Any]]:
        if scene_id in self._transforms_cache:
            return self._transforms_cache[scene_id]
        if self.scannetpp_root is None:
            self._transforms_cache[scene_id] = None
            return None
        path = self.scannetpp_root / scene_id / 'dslr' / 'nerfstudio' / 'transforms_undistorted.json'
        if not path.is_file():
            logger.warning('transforms_undistorted.json not found for scene %s: %s', scene_id, path)
            self._transforms_cache[scene_id] = None
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
        intr = {'w': int(data['w']), 'h': int(data['h']), 'fl_x': float(data['fl_x']), 'fl_y': float(data['fl_y']), 'cx': float(data['cx']), 'cy': float(data['cy'])}
        frame_index: Dict[str, dict] = {'__intr': intr}
        for entry in list(data.get('frames', [])) + list(data.get('test_frames', [])):
            file_path = entry.get('file_path')
            if not file_path:
                continue
            stem = Path(file_path).stem
            frame_index[stem] = entry
        self._transforms_cache[scene_id] = frame_index
        return frame_index
    _NS_TO_MESH_R = ((0.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    @classmethod
    def _ns_to_mesh(cls, c2w_ns: List[List[float]]) -> List[List[float]]:
        R = cls._NS_TO_MESH_R
        out = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                out[i][j] = sum((R[i][k] * c2w_ns[k][j] for k in range(4)))
        return out

    def _load_camera(self, scene_id: str, frame_id: str) -> Optional[Dict[str, Any]]:
        transforms = self._load_transforms(scene_id)
        if transforms is None:
            return None
        frame = transforms.get(frame_id)
        if frame is None:
            return None
        intr = transforms['__intr']
        width, height = (intr['w'], intr['h'])
        if self.normal_render_max_dim > 0 and max(width, height) > self.normal_render_max_dim:
            scale = self.normal_render_max_dim / max(width, height)
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))
        return {'width': width, 'height': height, 'fl_x': intr['fl_x'], 'fl_y': intr['fl_y'], 'cx': intr['cx'], 'cy': intr['cy'], 'fov': 2.0 * math.atan(intr['w'] / (2.0 * intr['fl_x'])), 'cam_to_world': self._ns_to_mesh(frame['transform_matrix'])}

    def _load_manifest(self, scene_id: str) -> Optional[Dict[int, dict]]:
        cached = self._manifest_cache.get(scene_id)
        if cached is not None:
            return cached
        manifest_path = self.gt_data_root / 'GTmesh' / scene_id / 'manifest.json'
        if not manifest_path.is_file():
            logger.warning('manifest.json not found for scene %s: %s', scene_id, manifest_path)
            return None
        data = json.loads(manifest_path.read_text(encoding='utf-8'))
        instances = {int(item['id']): item for item in data.get('instances', []) if 'id' in item}
        self._manifest_cache[scene_id] = instances
        return instances

    def _resolve_rgb_path(self, scene_id: str, frame_id: str) -> Optional[Path]:
        rgb_dir = self.gt_data_root / 'rgb' / scene_id
        for ext in ('.JPG', '.jpg', '.png', '.PNG', '.jpeg', '.JPEG'):
            candidate = rgb_dir / f'{frame_id}{ext}'
            if candidate.is_file():
                return candidate
        return None

    def _discover_visible_ids(self, scene_id: str, frame_id: str) -> List[int]:
        frame_mask_dir = self.gt_data_root / 'instance_masks' / scene_id / frame_id
        visible_ids: List[int] = []
        for mask_path in sorted(frame_mask_dir.glob('obj_*.png')):
            match = _OBJ_MASK_RE.match(mask_path.name)
            if match is None:
                continue
            visible_ids.append(int(match.group(1)))
        if visible_ids:
            return visible_ids
        dense_mask_path = self.gt_data_root / 'masks' / scene_id / f'{frame_id}.png'
        if not dense_mask_path.is_file():
            return []
        visible = {int(pixel) for pixel in Image.open(dense_mask_path).getdata()}
        visible.discard(0)
        return sorted(visible)

    def _build_sample(self, scene_id: str, frame_id: str, manifest: Dict[int, dict]) -> Optional[StandardSampleRecord]:
        source_id = f'{scene_id}/{frame_id}'
        gt_scene_dir = self.gt_data_root / 'GTmesh' / scene_id
        visible_ids = self._discover_visible_ids(scene_id, frame_id)
        diagnostics: List[str] = []
        objects: List[StandardObjectRecord] = []
        for canonical_id in visible_ids:
            mesh_path = gt_scene_dir / f'obj_{canonical_id}.ply'
            if not mesh_path.is_file():
                diagnostics.append(f'Missing GT mesh for canonical id {canonical_id}: {mesh_path}')
                continue
            instance_meta = manifest.get(canonical_id, {})
            label = instance_meta.get('label')
            objects.append(StandardObjectRecord(object_id=str(canonical_id), display_id=str(canonical_id), canonical_name=label, match_key=str(canonical_id), mesh_path=mesh_path, metadata={'canonical_id': canonical_id, 'label': label, 'members': list(instance_meta.get('members', []))}))
        if not objects:
            logger.debug('No GT objects found for %s', source_id)
            return None
        input_images: List[InputImageRecord] = []
        rgb_path = self._resolve_rgb_path(scene_id, frame_id)
        if rgb_path is not None:
            input_images.append(InputImageRecord(rgb_path, rgb_path.name))
        dense_mask_path = self.gt_data_root / 'masks' / scene_id / f'{frame_id}.png'
        if dense_mask_path.is_file():
            input_images.append(InputImageRecord(dense_mask_path, dense_mask_path.name))
        scene_mesh_path = gt_scene_dir / 'scene_mesh.ply'
        manifest_path = gt_scene_dir / 'manifest.json'
        frame_mask_dir = self.gt_data_root / 'instance_masks' / scene_id / frame_id
        render_config = self._load_camera(scene_id, frame_id)
        if render_config is None and self.scannetpp_root is not None:
            diagnostics.append(f'DSLR camera not found for {scene_id}/{frame_id}; normal eval will fall back to camera_sampler')
        return StandardSampleRecord(source_sample_id=source_id, canonical_sample_id=source_id, dataset_name=self.dataset_name, adapter_name=self.adapter_name, objects=objects, scene_mesh_path=scene_mesh_path if scene_mesh_path.is_file() else None, sample_dir=gt_scene_dir, asset_root=gt_scene_dir, input_images=input_images, render_config=render_config, metadata={'scene_id': scene_id, 'frame_id': frame_id, 'manifest_path': str(manifest_path), 'mask_dir': str(frame_mask_dir), 'mask_path': str(dense_mask_path) if dense_mask_path.is_file() else None}, diagnostics=diagnostics)
