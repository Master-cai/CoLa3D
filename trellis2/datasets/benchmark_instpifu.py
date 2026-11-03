from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mesh_eval.benchmark import discover_instpifu
from .benchmark_views import BenchmarkViewsDataset


class InstPIFuBenchmarkDataset(BenchmarkViewsDataset):
    """Read InstPIFu images and masks; bake GT occupancy only for Point."""

    def __init__(self, data_root: str, dataset_name: str = 'InstPIFu', image_size: int = 512,
                 emit_prompt_masks: bool = False, load_scene_voxels: bool = False,
                 instances: str | None = None) -> None:
        if dataset_name != 'InstPIFu':
            raise ValueError(f'Unsupported InstPIFu dataset: {dataset_name}')
        self.data_root = Path(data_root).resolve()
        self.image_size = image_size
        self.emit_prompt_masks = emit_prompt_masks
        self._load_scene_voxels = load_scene_voxels
        self._scene_voxel_cache_root = None
        selected = None
        if instances:
            path = Path(instances)
            values = path.read_text().splitlines() if path.is_file() else instances.split(',')
            selected = {value.strip() for value in values if value.strip() and not value.lstrip().startswith('#')}
        self._selection = {}
        self.instances = []
        for sample in discover_instpifu(self.data_root):
            key = sample['id']
            if selected is not None and key not in selected:
                continue
            self._selection[key] = sample
            self.instances.append(({'dataset': 'InstPIFu'}, key))
        if not self.instances:
            raise ValueError('No selected InstPIFu views')
        if selected is not None and selected != set(self._selection):
            raise ValueError(f'Unknown InstPIFu IDs: {sorted(selected - set(self._selection))}')

    def _image_pack(self, instance: str) -> dict[str, Any]:
        sample = self._selection[instance]
        image = Image.open(self._asset(sample['image'])).convert('RGBA')
        object_map = {int(obj['id']): index for index, obj in enumerate(sample['gt']['objects'])}
        masks, object_ids, meshes = [], [], []
        for relative in sample['prompt_masks']:
            path = self._asset(relative)
            mask = Image.open(path).convert('L')
            if not np.any(np.array(mask) > 127):
                continue
            original_id = int(Path(relative).name.split('_', 1)[0])
            masks.append(mask)
            object_ids.append(object_map[original_id])
            meshes.append(self._asset(sample['gt']['objects'][object_map[original_id]]['mesh']))
        if not masks:
            raise ValueError(f'{instance}: no nonempty visible masks')
        union = np.stack([np.array(mask) > 127 for mask in masks]).any(0)
        bbox = self._foreground_bbox(union)
        if bbox is None:
            raise ValueError(f'{instance}: empty foreground')
        cropped = image.crop(bbox).resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = torch.from_numpy(np.array(cropped.getchannel('A'))).float() / 255.
        if float(alpha.min()) > .99:
            union_image = Image.fromarray(union.astype(np.uint8) * 255, mode='L')
            alpha = torch.from_numpy(np.array(union_image.crop(bbox).resize((self.image_size, self.image_size), Image.Resampling.NEAREST))).float() / 255.
        cond = torch.from_numpy(np.array(cropped.convert('RGB'))).permute(2, 0, 1).float() / 255.
        pack = {'cond_image': cond * alpha.unsqueeze(0)}
        if self.emit_prompt_masks:
            prompts = torch.stack([(torch.from_numpy(np.array(mask.crop(bbox).resize((self.image_size, self.image_size), Image.Resampling.NEAREST))).float() / 255. > .5).float() for mask in masks])
            keep = prompts.flatten(1).sum(1) >= 1
            if not bool(keep.any()):
                raise ValueError(f'{instance}: all masks empty after crop')
            pack.update(prompt_masks=prompts[keep], prompt_mask_num_queries=int(keep.sum()),
                        prompt_mask_object_ids=torch.tensor(object_ids, dtype=torch.long)[keep],
                        prompt_object_ids=torch.tensor(object_ids, dtype=torch.long)[keep],
                        prompt_permutation=torch.arange(len(masks), dtype=torch.long)[keep],
                        prompt_view_id='render_0000.png',
                        prompt_gt_mesh_paths=[path for path, retained in zip(meshes, keep.tolist()) if retained])
        return pack

    def read_cond_image(self, instance: str, root: dict[str, str] | None = None) -> torch.Tensor:
        return self._image_pack(instance)['cond_image']

    def get_instance(self, root: dict[str, str], instance: str) -> dict[str, Any]:
        pack = self._image_pack(instance)
        if self._load_scene_voxels:
            sample = self._selection[instance]
            path = self._ensure_scene_voxels(self.data_root, sample['canonical_sample_id'],
                                         sample['gt']['objects'], self._scene_voxel_cache_root)
            with np.load(path, allow_pickle=False) as data:
                coords = torch.from_numpy(data['coords'].copy()).to(torch.int32)
                ids = torch.from_numpy(data['instance_id'].astype(np.int64))
                count = int(data['num_objects'])
            from trellis2.modules.sparse import SparseTensor
            coords = torch.cat([torch.zeros((len(coords), 1), dtype=torch.int32), coords], dim=1)
            pack.update(vertices=SparseTensor(torch.zeros((len(coords), 3), dtype=torch.float32), coords),
                        instance_id=ids, num_objects=count)
        return pack

    @staticmethod
    def _bake_parts(assets: list[tuple[int, Path]], output: Path, fingerprint: str) -> list[tuple[int, Any, Any]]:
        import numpy as np
        target = output / f'{fingerprint}.parts.npz'
        try:
            with np.load(target, allow_pickle=False) as data:
                ids = data['original_inst_ids'].tolist()
                if ids != [item[0] for item in assets]:
                    raise ValueError('Stale InstPIFu part cache')
                return [(iid, data[f'vertices_{index}'], data[f'faces_{index}']) for index, iid in enumerate(ids)]
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass
        blender = os.environ.get('BLENDER_PATH') or shutil.which('blender')
        if not blender or not Path(blender).is_file():
            raise FileNotFoundError('InstPIFu Point first-run preparation needs Blender 4.5; set BLENDER_PATH to its executable.')
        objects_path = output / f'{fingerprint}.objects.json'
        objects_path.write_text(json.dumps([{'id': iid, 'mesh': str(path)} for iid, path in assets]))
        script = Path(__file__).resolve().parents[2] / 'data_toolkit/blender_script/dump_benchmark_parts.py'
        with (output / f'{fingerprint}.blender.log').open('w') as log:
            subprocess.run([blender, '-b', '--threads', '1', '-P', str(script), '--',
                            '--objects', str(objects_path), '--output', str(target)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        with np.load(target, allow_pickle=False) as data:
            return [(iid, data[f'vertices_{index}'], data[f'faces_{index}']) for index, (iid, _) in enumerate(assets)]


    def _ensure_scene_voxels(self, root: Path, sample_id: str, objects: list[dict[str, Any]], cache_root: Path | None = None) -> Path:
        import numpy as np
        parts = sample_id.split('/')
        if len(parts) != 2 or any(not part or part in {'.', '..'} or '\\' in part for part in parts):
            raise ValueError(f'Invalid canonical sample ID: {sample_id}')
        root = root.resolve()
        objects = sorted(objects, key=lambda item: int(item['id']))
        ids = [int(item['id']) for item in objects]
        if not ids or len(ids) != len(set(ids)) or len(ids) > 127:
            raise ValueError(f'{sample_id}: invalid InstPIFu object IDs')
        assets = []
        digest = hashlib.sha256(b'instpifu-blender45-dual-grid-1024-v1')
        for item in objects:
            path = (root / item['mesh']).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(path)
            iid = int(item['id'])
            digest.update(json.dumps([iid, path.relative_to(root).as_posix()]).encode())
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            assets.append((iid, path))
        fingerprint = digest.hexdigest()
        directory = (Path(cache_root) if cache_root is not None else Path(__file__).resolve().parents[2] / 'outputs/.benchmark_cache') / 'InstPIFu'
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / ('__'.join(parts) + '.npz')

        def valid() -> bool:
            try:
                with np.load(target, allow_pickle=False) as data:
                    return (str(data['_asset_fingerprint']) == fingerprint and int(data['resolution']) == 1024
                            and int(data['num_objects']) == len(ids) and np.array_equal(data['original_inst_ids'], ids)
                            and data['coords'].ndim == 2 and data['coords'].shape[1] == 3
                            and data['instance_id'].shape == (len(data['coords']),))
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                return False

        with target.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if valid():
                return target
            import torch
            import o_voxel
            meshes = self._bake_parts(assets, directory, fingerprint)
            joined = torch.from_numpy(np.concatenate([mesh[1] for mesh in meshes])).float()
            lo, hi = joined.min(0).values, joined.max(0).values
            center = (lo + hi) / 2
            scale = 0.99999 / (hi - lo).max()
            coords_parts, instances, counts = [], [], []
            for index, (iid, vertices, faces) in enumerate(meshes):
                coords, _, _ = o_voxel.convert.mesh_to_flexible_dual_grid(
                    (torch.from_numpy(vertices).float() - center) * scale, torch.from_numpy(faces).long(),
                    grid_size=1024, aabb=[[-.5, -.5, -.5], [.5, .5, .5]],
                    face_weight=1., boundary_weight=.2, regularization_weight=1e-2, timing=False)
                coords = coords.numpy().astype(np.int32)
                if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords) or coords.min() < 0 or coords.max() >= 1024:
                    raise ValueError(f'{sample_id}: invalid voxels for GT object {iid}')
                coords_parts.append(coords)
                instances.append(np.full(len(coords), index, dtype=np.int8))
                counts.append(len(coords))
            with tempfile.NamedTemporaryFile(dir=directory, prefix=target.stem + '.', suffix='.npz', delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, coords=np.concatenate(coords_parts), instance_id=np.concatenate(instances),
                                    num_objects=np.int32(len(ids)), original_inst_ids=np.array(ids, dtype=np.int32),
                                    per_obj_voxels=np.array(counts, dtype=np.int32), center=center.numpy(),
                                    scale=scale.numpy(), resolution=np.int32(1024), _asset_fingerprint=np.array(fingerprint))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        return target
