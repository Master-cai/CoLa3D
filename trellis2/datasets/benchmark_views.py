from __future__ import annotations

import colorsys
import fcntl
import hashlib
import os
import tempfile
import zipfile
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from mesh_eval.benchmark import discover_benchmark

logger = logging.getLogger(__name__)


class BenchmarkViewsDataset:
    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        image_size: int = 512,
        emit_prompt_masks: bool = False,
        load_scene_voxels: bool = False,
        instances: Optional[str] = None,
    ) -> None:
        manifest = discover_benchmark(Path(data_root), dataset_name)
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported benchmark manifest schema")
        self.data_root = Path(data_root).resolve()
        self.image_size = image_size
        self.emit_prompt_masks = emit_prompt_masks
        self._load_scene_voxels = load_scene_voxels
        self._scene_voxel_cache_root = None
        self._selection: Dict[str, Dict[str, Any]] = {}
        self.instances: List[Tuple[Dict[str, str], str]] = []
        selected = None
        if instances:
            path = Path(instances)
            values = path.read_text().splitlines() if path.is_file() else instances.split(",")
            selected = {value.strip() for value in values if value.strip() and not value.lstrip().startswith("#")}
        for sample in manifest["samples"]:
            if sample["dataset"] != dataset_name:
                continue
            key = sample["id"]
            if selected is not None and key not in selected:
                continue
            if key in self._selection:
                raise ValueError(f"Duplicate benchmark sample: {key}")
            mask_format = sample["mask_format"]
            if mask_format not in {"metascenes_palette", "positive_id"}:
                raise ValueError(f"Unsupported view mask format: {mask_format}")
            meta = {
                "key": key,
                "rgb": self._asset(sample["image"]),
                "mask": self._asset(sample["instance_mask"]),
                "metadata": self._asset(sample.get("metadata")),
                "dataset": dataset_name,
                "alpha_source": "metascenes_rgb_palette" if mask_format == "metascenes_palette" else "positive_id",
                "gt_objects": sample.get("gt", {}).get("objects", []),
                "canonical_sample_id": sample["canonical_sample_id"],
            }
            self._selection[key] = meta
            self.instances.append(({"dataset": dataset_name}, key))
        if not self.instances:
            raise ValueError(f"No selected views for {dataset_name}")
        if selected is not None and selected != set(self._selection):
            raise ValueError(f"Unknown selected benchmark IDs: {sorted(selected - set(self._selection))}")

    def _asset(self, relative: Optional[str]) -> Optional[str]:
        if relative is None:
            return None
        if Path(relative).is_absolute():
            raise ValueError("Benchmark asset paths must be relative to the data root")
        path = (self.data_root / relative).resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError(f"Benchmark asset escapes data root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)

    def _resolve_gt_mesh_paths(self, meta: Dict[str, Any], kept_object_ids: List[int]) -> List[Optional[str]]:
        mapping = {int(obj["id"]): obj.get("mesh") for obj in meta["gt_objects"]}
        return [self._asset(mapping.get(object_id)) for object_id in kept_object_ids]

    @staticmethod
    def _build_metascenes_palette(num_instances: int) -> np.ndarray:
        palette = np.zeros((num_instances + 1, 3), dtype=np.uint8)
        for idx in range(1, num_instances + 1):
            hue = (idx - 1) / max(num_instances, 1) % 1.0
            sat = 0.85 if idx % 2 == 0 else 1.0
            val = 1.0 if idx % 3 != 0 else 0.75
            red, green, blue = colorsys.hsv_to_rgb(hue, sat, val)
            palette[idx] = [round(red * 255), round(green * 255), round(blue * 255)]
        return palette

    @staticmethod
    def _safe_int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _decode_metascenes_mask(self, mask_path: str, metadata_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
        alpha_bool = mask_rgb.sum(axis=-1) > 0
        if not metadata_path or not Path(metadata_path).is_file():
            unique_colors, inverse = np.unique(mask_rgb.reshape(-1, 3), axis=0, return_inverse=True)
            label_lookup = np.zeros(len(unique_colors), dtype=np.int32)
            next_label = 1
            for color_idx, color in enumerate(unique_colors):
                if np.any(color != 0):
                    label_lookup[color_idx] = next_label
                    next_label += 1
            labels = label_lookup[inverse].reshape(mask_rgb.shape[:2])
            return (labels, alpha_bool, {})
        metadata = json.loads(Path(metadata_path).read_text())
        inst_ids = metadata.get('inst_ids', [])
        palette = self._build_metascenes_palette(len(inst_ids))
        unique_colors, inverse = np.unique(mask_rgb.reshape(-1, 3), axis=0, return_inverse=True)
        diff = unique_colors[:, None, :].astype(np.int32) - palette[None, :, :].astype(np.int32)
        nearest_palette_idx = (diff * diff).sum(axis=-1).argmin(axis=1).astype(np.int32)
        labels = nearest_palette_idx[inverse].reshape(mask_rgb.shape[:2])
        labels[~alpha_bool] = 0
        object_id_map = {palette_idx: self._safe_int(inst_ids[palette_idx - 1], palette_idx) for palette_idx in np.unique(labels) if palette_idx > 0 and palette_idx <= len(inst_ids)}
        return (labels, alpha_bool, object_id_map)

    def _load_mask_data(self, meta: Dict[str, str], need_prompt_labels: bool) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        alpha_source = meta.get('alpha_source', 'dp_recon')
        if alpha_source == 'metascenes_rgb_palette':
            if need_prompt_labels:
                return self._decode_metascenes_mask(meta['mask'], meta.get('metadata'))
            mask_rgb = np.array(Image.open(meta['mask']).convert('RGB'))
            alpha_bool = mask_rgb.sum(axis=-1) > 0
            labels = alpha_bool.astype(np.int32)
            return (labels, alpha_bool, {})
        mask_arr = np.array(Image.open(meta['mask']))
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        labels = mask_arr.astype(np.int32, copy=False)
        if alpha_source == 'positive_id':
            alpha_bool = labels != 0
        else:
            alpha_bool = (labels != 255) & (labels != 0)
        return (labels, alpha_bool, {})

    @staticmethod
    def _foreground_bbox(alpha_bool: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        nz = np.argwhere(alpha_bool)
        if nz.size == 0:
            return None
        ymin, xmin = nz.min(axis=0)
        ymax, xmax = nz.max(axis=0)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        hsize = max(xmax - xmin, ymax - ymin) / 2.0
        return (int(cx - hsize), int(cy - hsize), int(cx + hsize), int(cy + hsize))

    def _crop_resize_cond_image(self, rgb_path: str, alpha_bool: np.ndarray, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
        rgb_img = Image.open(rgb_path).convert('RGB')
        alpha_arr = alpha_bool.astype(np.uint8) * 255
        rgba_img = Image.merge('RGBA', (*rgb_img.split(), Image.fromarray(alpha_arr, mode='L')))
        rgba_img = rgba_img.crop(bbox).resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha_t = torch.from_numpy(np.array(rgba_img.getchannel(3))).float() / 255.0
        rgb_t = torch.from_numpy(np.array(rgba_img.convert('RGB'))).permute(2, 0, 1).float() / 255.0
        return rgb_t * alpha_t.unsqueeze(0)

    def _build_prompt_masks(self, labels: np.ndarray, bbox: Tuple[int, int, int, int], object_id_map: Dict[int, int], meta: Dict[str, str]) -> Tuple[torch.Tensor, List[int], List[Optional[str]]]:
        visible_labels = sorted((int(value) for value in np.unique(labels) if int(value) not in (0, 255)))
        if not visible_labels:
            raise RuntimeError('instance mask has no labelled foreground objects')
        per_label_masks: List[torch.Tensor] = []
        kept_object_ids: List[int] = []
        dropped_labels: List[int] = []
        for label in visible_labels:
            bin_arr = (labels == label).astype(np.uint8) * 255
            mask_img = Image.fromarray(bin_arr, mode='L').crop(bbox).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            mask_t = (torch.from_numpy(np.array(mask_img)).float() / 255.0 > 0.5).float()
            if mask_t.sum() == 0:
                dropped_labels.append(label)
                continue
            per_label_masks.append(mask_t)
            kept_object_ids.append(object_id_map.get(label, label))
        if dropped_labels:
            logger.warning('BenchmarkViews: dropped %d empty prompt mask(s) after bbox crop (labels=%s, kept=%s)', len(dropped_labels), dropped_labels, kept_object_ids)
        if not per_label_masks:
            raise RuntimeError('All instance_mask labels were eliminated by the cond_image bbox crop')
        gt_mesh_paths = self._resolve_gt_mesh_paths(meta, kept_object_ids)
        return (torch.stack(per_label_masks, dim=0), kept_object_ids, gt_mesh_paths)

    def __len__(self) -> int:
        return len(self.instances)

    def read_cond_image(self, instance: str, root: Optional[Dict[str, str]]=None) -> Optional[torch.Tensor]:
        meta = self._selection[instance]
        _, alpha_bool, _ = self._load_mask_data(meta, need_prompt_labels=False)
        bbox = self._foreground_bbox(alpha_bool)
        if bbox is None:
            return None
        return self._crop_resize_cond_image(meta['rgb'], alpha_bool, bbox)

    def read_raw_rgb_and_mask(self, instance: str) -> Tuple[torch.Tensor, torch.Tensor]:
        meta = self._selection[instance]
        rgb_img = Image.open(meta['rgb']).convert('RGB')
        if meta.get('alpha_source') == 'metascenes_rgb_palette':
            mask_rgb = np.array(Image.open(meta['mask']).convert('RGB'))
            mask_arr = self._decode_rgb_palette_mask(mask_rgb)
        else:
            mask_arr = np.array(Image.open(meta['mask']))
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        rgb_t = torch.from_numpy(np.array(rgb_img)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_arr.astype(np.uint8))
        return (rgb_t, mask_t)

    @staticmethod
    def _decode_rgb_palette_mask(mask_rgb: np.ndarray) -> np.ndarray:
        flat = mask_rgb.reshape(-1, 3)
        non_bg = np.any(flat != 0, axis=1)
        out = np.zeros((flat.shape[0],), dtype=np.uint8)
        if not np.any(non_bg):
            return out.reshape(mask_rgb.shape[:2])
        colors, inverse = np.unique(flat[non_bg], axis=0, return_inverse=True)
        if colors.shape[0] > 254:
            raise ValueError(f'MetaScenes mask has {colors.shape[0]} colors; uint8 id mask supports 254')
        out[non_bg] = inverse.astype(np.uint8) + 1
        return out.reshape(mask_rgb.shape[:2])

    def get_instance(self, root: Dict[str, str], instance: str) -> Dict[str, Any]:
        meta = self._selection[instance]
        labels, alpha_bool, object_id_map = self._load_mask_data(meta, need_prompt_labels=self.emit_prompt_masks)
        bbox = self._foreground_bbox(alpha_bool)
        if bbox is None:
            raise RuntimeError(f'Empty instance_mask (no foreground) for {instance}')
        cond_image = self._crop_resize_cond_image(meta['rgb'], alpha_bool, bbox)
        pack: Dict[str, Any] = {'cond_image': cond_image}
        raw_rgb, instance_mask_full = self.read_raw_rgb_and_mask(instance)
        pack['raw_rgb'] = raw_rgb
        pack['instance_mask_full'] = instance_mask_full
        if self.emit_prompt_masks:
            prompt_masks, object_ids, gt_mesh_paths = self._build_prompt_masks(labels, bbox, object_id_map, meta)
            pack['prompt_masks'] = prompt_masks
            pack['prompt_num_queries'] = int(prompt_masks.shape[0])
            pack['prompt_object_ids'] = torch.tensor(object_ids, dtype=torch.long)
            pack['prompt_gt_mesh_paths'] = gt_mesh_paths
        scene = self.read_metascenes_scene_voxels(instance, cache_root=self._scene_voxel_cache_root)
        if scene is not None:
            pack.update(scene)
        return pack

    def read_metascenes_scene_voxels(self, instance: str, cache_root: Optional[Path]=None) -> Optional[Dict[str, Any]]:
        meta = self._selection[instance]
        if not self._load_scene_voxels or meta["dataset"] not in {"MetaScenes", "MetaScenes_Hard"}:
            return None
        cache_path = self._ensure_scene_voxels(self.data_root, meta["dataset"], meta["canonical_sample_id"],
                                         meta["gt_objects"], cache_root=cache_root)
        with np.load(cache_path) as z:
            coords = torch.from_numpy(np.ascontiguousarray(z['coords'])).to(torch.int32)
            instance_id = torch.from_numpy(z['instance_id'].astype(np.int64))
            num_objects = int(z['num_objects'])
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f'{cache_path}: coords must have shape (N, 3), got {coords.shape}')
        if instance_id.ndim != 1 or instance_id.shape[0] != coords.shape[0]:
            raise ValueError(f'{cache_path}: instance_id must have shape ({coords.shape[0]},), got {instance_id.shape}')
        if coords.numel() > 0:
            if int(coords.min()) < 0 or int(coords.max()) > 1023:
                raise ValueError(f'{cache_path}: coords out of expected [0, 1023] range')
        if num_objects < 0:
            raise ValueError(f'{cache_path}: num_objects must be non-negative, got {num_objects}')
        if instance_id.numel() > 0 and (int(instance_id.min()) < 0 or int(instance_id.max()) >= num_objects):
            raise ValueError(f'{cache_path}: instance_id values must be in [0, {num_objects - 1}]')
        coords_with_batch = torch.cat([torch.zeros((coords.shape[0], 1), dtype=torch.int32), coords], dim=-1)
        dummy_feats = torch.zeros((coords.shape[0], 3), dtype=torch.float32)
        import trellis2.modules.sparse as sp_local
        vertices = sp_local.SparseTensor(dummy_feats, coords_with_batch)
        return {'vertices': vertices, 'instance_id': instance_id, 'num_objects': num_objects}

    @staticmethod
    def _ensure_scene_voxels(root: Path, dataset: str, sample_id: str, objects: list[dict[str, Any]], cache_root: Path | None = None) -> Path:
        if dataset not in {"MetaScenes", "MetaScenes_Hard"}:
            raise ValueError(f"Unsupported scene voxel dataset: {dataset}")
        if len(sample_id) != 4 or not sample_id.isdecimal():
            raise ValueError(f"Invalid sample ID: {sample_id}")
        root = Path(root).resolve()
        metadata_path = root / dataset / "anno" / sample_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        object_map = {int(obj["id"]): obj for obj in objects}
        if len(object_map) != len(objects):
            raise ValueError(f"{sample_id}: duplicate GT object IDs")
        order = metadata.get("object_ids", sorted(object_map))
        if not isinstance(order, list):
            raise ValueError(f"{metadata_path}: object_ids must be a list")
        ids = [int(value) for value in order]
        if len(ids) != len(set(ids)) or set(ids) != set(object_map):
            raise ValueError(f"{metadata_path}: object_ids must match GT objects exactly")
        objects = [object_map[iid] for iid in ids]
        assets = []
        digest = hashlib.sha256(b"metascenes-dual-grid-1024-v1")
        if not objects:
            raise ValueError(f"{sample_id}: no GT objects")
        ids = [int(obj["id"]) for obj in objects]
        if len(ids) != len(set(ids)) or len(ids) > 127:
            raise ValueError(f"{sample_id}: invalid GT object IDs")
        for obj in objects:
            path = (root / obj["mesh"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(path)
            digest.update(json.dumps([int(obj["id"]), path.relative_to(root).as_posix()]).encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            assets.append((int(obj["id"]), path))
        fingerprint = digest.hexdigest()
        directory = (Path(cache_root) if cache_root is not None else Path(__file__).resolve().parents[2] / "outputs" / ".benchmark_cache") / dataset
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (sample_id + ".npz")
        import numpy as np

        def valid() -> bool:
            if not target.is_file():
                return False
            try:
                with np.load(target, allow_pickle=False) as data:
                    return (str(data["_asset_fingerprint"]) == fingerprint
                            and int(data["resolution"]) == 1024
                            and int(data["num_objects"]) == len(ids)
                            and np.array_equal(data["original_inst_ids"], ids)
                            and data["coords"].ndim == 2 and data["coords"].shape[1] == 3
                            and data["instance_id"].shape == (len(data["coords"]),))
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                return False

        with target.with_suffix(".lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if valid():
                return target
            import torch
            import trimesh
            import o_voxel

            meshes = []
            for iid, path in assets:
                loaded = trimesh.load(str(path), force="scene")
                mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
                if isinstance(mesh, list):
                    mesh = trimesh.util.concatenate(mesh)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(f"{path}: no Trimesh geometry")
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                if not vertices.size or not faces.size or not np.isfinite(vertices).all():
                    raise ValueError(f"{path}: empty or nonfinite GT geometry")
                meshes.append((iid, vertices, faces))
            vertices = np.concatenate([mesh[1] for mesh in meshes], axis=0)
            vmin, vmax = vertices.min(axis=0), vertices.max(axis=0)
            max_extent = float((vmax - vmin).max())
            if max_extent <= 0.0:
                raise ValueError(f"{sample_id}: degenerate scene AABB")
            center = (vmin + vmax) / 2.0
            scale = float(0.99999 / max_extent)
            all_coords, all_instance, counts = [], [], []
            for index, (iid, vertices, faces) in enumerate(meshes):
                normalized = (vertices - center) * scale
                coords, _, _ = o_voxel.convert.mesh_to_flexible_dual_grid(
                    torch.from_numpy(normalized.astype(np.float32)),
                    torch.from_numpy(faces.astype(np.int64)), grid_size=1024,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False,
                )
                coords = coords.detach().cpu().numpy().astype(np.int32)
                if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords) or coords.min() < 0 or coords.max() >= 1024:
                    raise ValueError(f"{sample_id}: invalid voxels for GT object {iid}")
                all_coords.append(coords)
                all_instance.append(np.full(len(coords), index, dtype=np.int8))
                counts.append(len(coords))
            with tempfile.NamedTemporaryFile(dir=directory, prefix=target.stem + ".", suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, coords=np.concatenate(all_coords), instance_id=np.concatenate(all_instance),
                                    num_objects=np.int32(len(meshes)), original_inst_ids=np.array(ids, dtype=np.int32),
                                    per_obj_voxels=np.array(counts, dtype=np.int32), center=center.astype(np.float32),
                                    scale=np.float32(scale), resolution=np.int32(1024), _asset_fingerprint=np.array(fingerprint))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        return target
