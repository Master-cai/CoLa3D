import logging
import os
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
import re
import numpy as np
import pandas as pd
import torch
from PIL import Image
from .components import StandardDatasetBase
from .flexi_dual_grid import FlexiDualGridVisMixin
from ..modules import sparse as sp
from ..representations import Mesh
from ..utils.data_utils import load_balanced_group_indices
logger = logging.getLogger(__name__)
try:
    import o_voxel
except ImportError:
    o_voxel = None
PROMPT_POINT_DOWNSAMPLE_RATIO = 16

class FlexiDualGridMultiObjDataset(FlexiDualGridVisMixin, StandardDatasetBase):

    def __init__(self, roots: str, resolution: int=512, max_active_voxels: int=1000000, max_num_faces: Optional[int]=None, min_aesthetic_score: Optional[float]=None, max_objects: int=8, min_objects: Optional[int]=None, instances: Optional[str]=None, overfit_id: Optional[str]=None, overfit_repeats: int=8, render_root: Optional[Union[str, Dict[str, str]]]=None, image_size: int=512, strict_cond_image: bool=True, skip_mesh: bool=False, skip_per_obj: bool=False, preload_mode: Optional[str]=None, shape_latent_root: Optional[Union[str, Dict[str, str]]]=None, utonia_feats_root: Optional[Union[str, Dict[str, str]]]=None, skip_first_frame: Optional[Dict[str, bool]]=None, dataset_repeat: Optional[Dict[str, int]]=None, prompt_mask_mode: Optional[str]=None, prompt_mask_root: Optional[Union[str, Dict[str, str]]]=None, strict_prompt_masks: bool=True, random_prompt_order: bool=False, prompt_query_mode: str='both'):
        if o_voxel is None:
            raise ImportError('o_voxel library is required for this dataset')
        if prompt_query_mode not in {'mask', 'point', 'both', 'none'}:
            raise ValueError(f"prompt_query_mode must be one of {{'mask','point','both','none'}}, got {prompt_query_mode!r}")
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_active_voxels = max_active_voxels
        self.max_num_faces = max_num_faces
        self.max_objects = max_objects
        self.min_objects = min_objects
        self._instance_whitelist = None
        self.value_range = (0, 1)
        self.overfit_id = overfit_id
        self.overfit_repeats = overfit_repeats
        self.render_root = render_root
        self.image_size = image_size
        self.strict_cond_image = strict_cond_image
        self.skip_mesh = skip_mesh
        self.skip_per_obj = skip_per_obj
        self.preload_mode = preload_mode
        self.shape_latent_root = shape_latent_root
        self.utonia_feats_root = utonia_feats_root
        self.skip_first_frame = skip_first_frame
        self.dataset_repeat = dataset_repeat
        self.prompt_mask_root = prompt_mask_root
        self.strict_prompt_masks = strict_prompt_masks
        self.random_prompt_order = random_prompt_order
        self.prompt_query_mode = prompt_query_mode
        self._voxel_cache: Dict[str, Dict] = {}
        if instances is not None:
            if os.path.exists(instances):
                with open(instances, 'r') as f:
                    rows = [line.strip() for line in f.readlines()]
                self._instance_whitelist = set([x for x in rows if len(x) > 0])
            else:
                rows = [x.strip() for x in instances.split(',')]
                self._instance_whitelist = set([x for x in rows if len(x) > 0])
        super().__init__(roots)
        if self.utonia_feats_root is not None:
            before = len(self.instances)
            kept: List[Tuple] = []
            for root, sha256 in self.instances:
                ufr = self._resolve_utonia_feats_root(root)
                if ufr is None or os.path.exists(os.path.join(ufr, f'{sha256}.npz')):
                    kept.append((root, sha256))
            dropped = before - len(kept)
            if dropped > 0:
                print(f'[FlexiDualGridMultiObj] Dropped {dropped}/{before} instances missing utonia_feats_root npz files')
            self.instances = kept
        if self.dataset_repeat and isinstance(self.roots, dict):
            original = self.instances
            repeated = []
            for root, sha256 in original:
                n = 1
                for key, root_dict in self.roots.items():
                    if root is root_dict:
                        n = self.dataset_repeat.get(key, 1)
                        break
                repeated.extend([(root, sha256)] * n)
            self.instances = repeated
        self._render_file_cache: Dict[str, List[str]] = {}
        self._sha256_to_file_id: Dict[str, str] = {}
        if 'file_identifier' in self.metadata.columns:
            for sha256 in self.metadata.index:
                fid = self.metadata.loc[sha256, 'file_identifier']
                if pd.notna(fid):
                    self._sha256_to_file_id[sha256] = fid
        if 'dual_grid_size' in self.metadata.columns:
            self.loads = [self.metadata.loc[sha256, 'dual_grid_size'] for _, sha256 in self.instances]
        else:
            self.loads = [1] * len(self.instances)
        if self.overfit_id is not None:
            self._setup_overfit_mode()
        if self.render_root is not None and self.overfit_id is None:
            self._prewarm_render_file_cache()
            if self._should_emit_prompt_masks():
                before = len(self.instances)
                kept = [(root, sha256) for root, sha256 in self.instances if len(self._render_file_cache.get(sha256, [])) > 0]
                dropped = before - len(kept)
                if dropped > 0:
                    print(f'[FlexiDualGridMultiObj] Dropped {dropped}/{before} instances with empty render dir (prompt-mask mode needs rendered masks)')
                self.instances = kept
                if 'dual_grid_size' in self.metadata.columns:
                    self.loads = [self.metadata.loc[sha256, 'dual_grid_size'] for _, sha256 in self.instances]
                else:
                    self.loads = [1] * len(self.instances)
        if self.preload_mode is not None and self.overfit_id is None:
            self._init_voxel_cache()

    def _setup_overfit_mode(self):
        found = False
        overfit_load = None
        for i, (root, sha256) in enumerate(self.instances):
            if sha256 == self.overfit_id:
                self._overfit_root = root
                self._overfit_instance = sha256
                overfit_load = self.loads[i]
                found = True
                break
        if not found:
            raise ValueError(f"Overfit ID '{self.overfit_id}' not found in dataset. Available instances: {len(self.instances)}")
        self.loads = [overfit_load] * self.overfit_repeats
        print(f"[FlexiDualGridMultiObjDataset] OVERFIT MODE: Using instance '{self.overfit_id}' repeated {self.overfit_repeats} times")

    def _resolve_render_root(self, root) -> Optional[str]:
        if isinstance(self.render_root, dict):
            dataset_key = self._resolve_dataset_key(root)
            return None if dataset_key is None else self.render_root.get(dataset_key)
        return self.render_root

    def _resolve_root_mapping(self, mapping, root) -> Optional[str]:
        if mapping is None:
            return None
        if isinstance(mapping, dict):
            if isinstance(self.roots, dict):
                for key, root_dict in self.roots.items():
                    if root is root_dict:
                        return mapping.get(key)
            return None
        return mapping

    def _resolve_shape_latent_root(self, root) -> Optional[str]:
        return self._resolve_root_mapping(self.shape_latent_root, root)

    def _resolve_utonia_feats_root(self, root) -> Optional[str]:
        return self._resolve_root_mapping(self.utonia_feats_root, root)

    def _should_skip_first_frame(self, root) -> bool:
        if not self.skip_first_frame or not isinstance(self.roots, dict):
            return False
        dataset_key = self._resolve_dataset_key(root)
        if dataset_key is None:
            return False
        return self.skip_first_frame.get(dataset_key, False)

    def _resolve_prompt_mask_root(self, root) -> Optional[str]:
        if isinstance(self.prompt_mask_root, dict):
            dataset_key = self._resolve_dataset_key(root)
            return None if dataset_key is None else self.prompt_mask_root.get(dataset_key)
        return self.prompt_mask_root

    def _resolve_dataset_key(self, root) -> Optional[str]:
        if not isinstance(self.roots, dict) or not isinstance(root, dict):
            return None
        for probe_key in ('base', 'mesh_dump', 'dual_grid'):
            probe_value = root.get(probe_key)
            if not isinstance(probe_value, str):
                continue
            if 'InstPIFu-3D-FRONT' in probe_value:
                return 'InstPIFu_3D_FRONT'
            if 'MIDI-3D-FRONT' in probe_value:
                return 'MIDI_3D_FRONT'
        for key, root_dict in self.roots.items():
            if root is root_dict or root == root_dict:
                return key
            stable_keys = ('base', 'mesh_dump', 'dual_grid')
            if all((root.get(k) == root_dict.get(k) for k in stable_keys)):
                return key
        return None

    def _resolve_prompt_mask_mode(self, root) -> Optional[str]:
        dataset_key = self._resolve_dataset_key(root)
        if dataset_key == 'InstPIFu_3D_FRONT':
            return 'instpifu_visible_masks'
        if dataset_key == 'MIDI_3D_FRONT':
            return 'midi_semantic_png'
        return None

    @staticmethod
    def _crop_bbox_from_binary_mask(binary_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        nonzero = binary_mask.nonzero()
        if nonzero[0].size == 0:
            return None
        bbox = [nonzero[1].min(), nonzero[0].min(), nonzero[1].max(), nonzero[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        return (int(center[0] - hsize), int(center[1] - hsize), int(center[0] + hsize), int(center[1] + hsize))

    def _list_render_candidates(self, instance: str, root) -> List[str]:
        render_root = self._resolve_render_root(root)
        if render_root is None:
            return []
        file_id = self._sha256_to_file_id.get(instance)
        if file_id is None:
            return []
        render_dir = Path(render_root) / file_id
        if instance not in self._render_file_cache:
            self._render_file_cache[instance] = self._scan_render_dir(render_dir)
        view_candidates = self._render_file_cache[instance]
        if self._should_skip_first_frame(root) and len(view_candidates) > 1:
            view_candidates = [p for p in view_candidates if 'render_0000' not in os.path.basename(p)]
        return view_candidates

    @staticmethod
    def _scan_render_dir(render_dir: Path) -> List[str]:
        if not render_dir.is_dir():
            return []
        render_files: List[str] = []
        with os.scandir(render_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                if not entry.name.startswith('render_'):
                    continue
                suffix = Path(entry.name).suffix.lower()
                if suffix not in {'.webp', '.png', '.jpg', '.jpeg'}:
                    continue
                render_files.append(entry.path)
        render_files.sort()
        return render_files

    def _scan_render_candidates_for_instance(self, root, sha256: str) -> Tuple[str, List[str]]:
        file_id = self._sha256_to_file_id.get(sha256)
        if file_id is None:
            return (sha256, [])
        render_root = self._resolve_render_root(root)
        if render_root is None:
            return (sha256, [])
        return (sha256, self._scan_render_dir(Path(render_root) / file_id))

    @staticmethod
    def _load_dual_grid_original_indices(dual_grid_root: str, instance: str) -> List[int]:
        meta_path = Path(dual_grid_root) / instance / 'meta.json'
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        original_indices: List[int] = []
        for obj_idx, obj_info in enumerate(meta['objects']):
            original_indices.append(int(obj_info.get('original_object_id', obj_info.get('index', obj_idx))))
        return original_indices

    @staticmethod
    def _load_instpifu_raw_object_indices(render_root: str, file_id: str) -> Optional[List[int]]:
        objects_dir = Path(render_root).parent / 'objects' / file_id
        if objects_dir.is_dir():
            names = sorted((p.name for p in objects_dir.iterdir() if p.is_dir()))
        else:
            mask_dir = Path(render_root).parent / 'masks' / file_id
            if not mask_dir.is_dir():
                return None
            names = sorted({p.name for p in mask_dir.glob('*.png')})
        object_indices: List[int] = []
        for name in names:
            match = re.match('^(\\d+)_', name)
            if match is None:
                return None
            object_indices.append(int(match.group(1)))
        if len(object_indices) == 0:
            return None
        return sorted(set(object_indices)) if not objects_dir.is_dir() else object_indices

    def _load_original_indices(self, root, instance: str, render_root: Optional[str], file_id: Optional[str], prompt_mode: Optional[str]) -> List[int]:
        dual_grid_indices = self._load_dual_grid_original_indices(root['dual_grid'], instance)
        meta_path = Path(root['dual_grid']) / instance / 'meta.json'
        with meta_path.open() as handle:
            object_infos = json.load(handle)['objects']
        explicit_ids = ['original_object_id' in item for item in object_infos]
        if any(explicit_ids) and not all(explicit_ids):
            raise ValueError(f'Partial original_object_id metadata: {meta_path}')
        if object_infos and all(explicit_ids):
            if len(set(dual_grid_indices)) != len(dual_grid_indices):
                raise ValueError(f'Duplicate original_object_id metadata: {meta_path}')
            return dual_grid_indices

        if prompt_mode != 'instpifu_visible_masks' or render_root is None or file_id is None:
            return dual_grid_indices
        instpifu_indices = self._load_instpifu_raw_object_indices(render_root, file_id)
        if instpifu_indices is None:
            return dual_grid_indices
        if len(instpifu_indices) != len(dual_grid_indices):
            return dual_grid_indices
        return instpifu_indices

    @staticmethod
    def _build_original_to_contiguous_index(original_indices: List[int]) -> Dict[int, int]:
        return {int(original_idx): contiguous_idx for contiguous_idx, original_idx in enumerate(original_indices)}

    @staticmethod
    def _validate_prompt_object_ids(prompt_object_ids: Optional[torch.Tensor], num_objects: int, instance: str, render_root: Optional[str], file_id: Optional[str]) -> None:
        if prompt_object_ids is None:
            return
        if prompt_object_ids.numel() == 0:
            raise RuntimeError(f'Empty prompt_object_ids for instance={instance} file_id={file_id} under render_root={render_root}')
        object_ids = prompt_object_ids.detach().cpu().tolist()
        invalid = [idx for idx in object_ids if idx < 0 or idx >= num_objects]
        if invalid:
            raise RuntimeError(f'Invalid prompt_object_ids={object_ids} for instance={instance} file_id={file_id} num_objects={num_objects} render_root={render_root}')
        if len(set(object_ids)) != len(object_ids):
            raise RuntimeError(f'Duplicate prompt_object_ids={object_ids} for instance={instance} file_id={file_id} num_objects={num_objects} render_root={render_root}')

    @staticmethod
    def _encode_grid_keys(coords: torch.Tensor, grid_size: int) -> torch.Tensor:
        coords64 = coords.to(torch.int64)
        grid_size64 = int(grid_size)
        return coords64[:, 0] * grid_size64 * grid_size64 + coords64[:, 1] * grid_size64 + coords64[:, 2]

    def _build_prompt_points(self, latent_coords: torch.Tensor, scene_coords: torch.Tensor, instance_id: torch.Tensor, num_objects: int, instance: str, shape_latent_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        if latent_coords.numel() == 0:
            raise RuntimeError(f'Empty latent coords for instance={instance} at {shape_latent_path}')
        downsample_ratio = PROMPT_POINT_DOWNSAMPLE_RATIO
        if self.resolution % downsample_ratio != 0:
            raise RuntimeError(f'Resolution {self.resolution} not divisible by PROMPT_POINT_DOWNSAMPLE_RATIO={downsample_ratio}')
        latent_resolution = self.resolution // downsample_ratio
        ds_scene_coords = torch.div(scene_coords.to(torch.int64), downsample_ratio, rounding_mode='floor').clamp_(0, latent_resolution - 1)
        latent_keys = self._encode_grid_keys(latent_coords, latent_resolution)
        scene_keys = self._encode_grid_keys(ds_scene_coords, latent_resolution)
        sorted_latent_keys, sorted_latent_order = torch.sort(latent_keys)
        match_pos = torch.searchsorted(sorted_latent_keys, scene_keys)
        valid = match_pos < sorted_latent_keys.numel()
        valid_scene_keys = scene_keys[valid]
        valid_instance_ids = instance_id.to(torch.int64)[valid]
        valid_match_pos = match_pos[valid]
        valid = sorted_latent_keys[valid_match_pos] == valid_scene_keys
        matched_latent_keys = valid_scene_keys[valid]
        matched_object_ids = valid_instance_ids[valid]
        if matched_latent_keys.numel() == 0:
            raise RuntimeError(f'No latent voxels matched GT voxels for instance={instance} at {shape_latent_path}')
        total_latent_cells = latent_resolution ** 3
        pair_keys = matched_object_ids * total_latent_cells + matched_latent_keys
        unique_pair_keys = torch.unique(pair_keys, sorted=True)
        unique_object_ids = torch.div(unique_pair_keys, total_latent_cells, rounding_mode='floor')
        unique_latent_keys = unique_pair_keys % total_latent_cells
        unique_latent_pos = torch.searchsorted(sorted_latent_keys, unique_latent_keys)
        unique_latent_indices = sorted_latent_order[unique_latent_pos]
        prompt_points: List[torch.Tensor] = []
        prompt_object_ids: List[int] = []
        chosen_keys: set = set()
        for object_id in range(num_objects):
            object_mask = unique_object_ids == object_id
            if not torch.any(object_mask):
                raise RuntimeError(f'Object {object_id} has no latent voxel for instance={instance} (latent_resolution={latent_resolution}, shape_latent={shape_latent_path})')
            object_latent_indices = unique_latent_indices[object_mask]
            object_latent_keys = unique_latent_keys[object_mask]
            num_candidates = int(object_latent_indices.numel())
            if self.random_prompt_order:
                perm = torch.randperm(num_candidates).tolist()
            else:
                perm = torch.argsort(object_latent_keys).tolist()
            chosen_latent_idx = -1
            for cand_pos in perm:
                key = int(object_latent_keys[cand_pos].item())
                if key in chosen_keys:
                    continue
                chosen_latent_idx = int(object_latent_indices[cand_pos].item())
                chosen_keys.add(key)
                break
            if chosen_latent_idx < 0:
                fallback_pos = perm[0]
                chosen_latent_idx = int(object_latent_indices[fallback_pos].item())
            prompt_points.append(latent_coords[chosen_latent_idx])
            prompt_object_ids.append(object_id)
        points_tensor = torch.stack(prompt_points, dim=0).to(torch.int32)
        ids_tensor = torch.tensor(prompt_object_ids, dtype=torch.long)
        if self.random_prompt_order and num_objects > 1:
            slot_perm = torch.randperm(num_objects)
            points_tensor = points_tensor[slot_perm]
            ids_tensor = ids_tensor[slot_perm]
        return (points_tensor, ids_tensor)

    def _should_emit_prompt_masks(self) -> bool:
        return self.prompt_query_mode in {'mask', 'both'}

    def _should_emit_prompt_points(self) -> bool:
        return self.prompt_query_mode in {'point', 'both'}

    def _load_instpifu_prompt_masks(self, render_root: str, file_id: str, root, original_to_contiguous: Optional[Dict[int, int]]=None) -> Tuple[List[Image.Image], List[int], List[Optional[str]]]:
        prompt_root = self._resolve_prompt_mask_root(root)
        if prompt_root is None:
            prompt_root = str(Path(render_root).parent / 'masks')
        mask_dir = Path(prompt_root) / file_id
        if not mask_dir.is_dir():
            return ([], [], [])
        objects_dir = Path(render_root).parent / 'objects' / file_id
        mask_images: List[Image.Image] = []
        object_ids: List[int] = []
        gt_mesh_paths: List[Optional[str]] = []
        for mask_path in sorted(mask_dir.glob('*.png')):
            match = re.match('^(\\d+)_', mask_path.name)
            if match is None:
                continue
            original_object_id = int(match.group(1))
            if original_to_contiguous is not None:
                object_id = original_to_contiguous.get(original_object_id, -1)
            else:
                object_id = original_object_id
            mask_img = Image.open(mask_path)
            if mask_img.mode != 'L':
                mask_img = mask_img.convert('L')
            mask_arr = np.array(mask_img)
            if (mask_arr > 127).sum() == 0:
                continue
            mask_images.append(mask_img)
            object_ids.append(object_id)
            gt_path: Optional[str] = None
            if objects_dir.is_dir():
                obj_pattern = f'{original_object_id:03d}_*'
                obj_matches = sorted(objects_dir.glob(obj_pattern))
                for obj_dir in obj_matches:
                    glb = obj_dir / 'object_world.glb'
                    if glb.is_file():
                        gt_path = str(glb)
                        break
            gt_mesh_paths.append(gt_path)
        return (mask_images, object_ids, gt_mesh_paths)

    @staticmethod
    def _frame_suffix_from_render_name(render_name: str) -> Optional[str]:
        match = re.search('render_(\\d+)\\.', render_name)
        if match is None:
            return None
        return match.group(1)

    def _load_midi_prompt_masks(self, render_dir: Path, render_name: str, original_to_contiguous: Optional[Dict[int, int]]=None) -> Tuple[List[Image.Image], List[int], List[Optional[str]]]:
        frame_suffix = self._frame_suffix_from_render_name(render_name)
        if frame_suffix is None:
            return ([], [], [])
        semantic_path = render_dir / f'semantic_{frame_suffix}.png'
        if not semantic_path.is_file():
            return ([], [], [])
        semantic = np.array(Image.open(semantic_path))
        object_ids = sorted((int(v) for v in np.unique(semantic) if int(v) > 0))
        mask_images: List[Image.Image] = []
        visible_ids: List[int] = []
        for object_id in object_ids:
            original_object_id = object_id - 1
            if original_to_contiguous is not None:
                if original_object_id not in original_to_contiguous:
                    continue
                gt_object_id = original_to_contiguous[original_object_id]
            else:
                gt_object_id = original_object_id
            binary_mask = (semantic == object_id).astype(np.uint8) * 255
            if binary_mask.sum() == 0:
                continue
            mask_images.append(Image.fromarray(binary_mask, mode='L'))
            visible_ids.append(gt_object_id)
        return (mask_images, visible_ids, [None] * len(mask_images))

    @staticmethod
    def _apply_crop_to_mask(mask_image: Image.Image, crop_bbox: Tuple[int, int, int, int], image_size: int) -> torch.Tensor:
        cropped_mask = mask_image.crop(crop_bbox)
        resized_mask = cropped_mask.resize((image_size, image_size), Image.Resampling.NEAREST)
        mask_tensor = torch.tensor(np.array(resized_mask)).float() / 255.0
        return (mask_tensor > 0.5).float()

    def _load_cond_and_prompt(self, instance: str, root, original_to_contiguous: Optional[Dict[int, int]]=None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[str], Optional[List[Optional[str]]]]:
        render_root = self._resolve_render_root(root)
        if render_root is None:
            return (None, None, None, None, None, None)
        file_id = self._sha256_to_file_id.get(instance)
        if file_id is None:
            return (None, None, None, None, None, None)
        view_candidates = self._list_render_candidates(instance, root)
        if len(view_candidates) == 0:
            return (None, None, None, None, None, None)
        image_path = view_candidates[np.random.randint(0, len(view_candidates))]
        image = Image.open(image_path)
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        render_dir = Path(render_root) / file_id
        prompt_mode = self._resolve_prompt_mask_mode(root)
        prompt_mask_images: List[Image.Image] = []
        prompt_object_ids: List[int] = []
        prompt_gt_mesh_paths: List[Optional[str]] = []
        if prompt_mode == 'instpifu_visible_masks':
            prompt_mask_images, prompt_object_ids, prompt_gt_mesh_paths = self._load_instpifu_prompt_masks(render_root, file_id, root, original_to_contiguous=original_to_contiguous)
        elif prompt_mode == 'midi_semantic_png':
            prompt_mask_images, prompt_object_ids, prompt_gt_mesh_paths = self._load_midi_prompt_masks(render_dir, Path(image_path).name, original_to_contiguous=original_to_contiguous)
        if len(prompt_mask_images) > 0:
            crop_source = np.stack([np.array(mask.convert('L')) > 127 for mask in prompt_mask_images], axis=0).any(axis=0)
        else:
            crop_source = np.array(image.getchannel(3)) > 0
        crop_bbox = self._crop_bbox_from_binary_mask(crop_source.astype(np.uint8))
        if crop_bbox is None:
            return (None, None, None, None, None, None)
        image = image.crop(crop_bbox)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
        if len(prompt_mask_images) > 0 and (prompt_mode == 'midi_semantic_png' or float(alpha.min()) > 0.99):
            union_full = Image.fromarray(crop_source.astype(np.uint8) * 255, mode='L')
            union_cropped = union_full.crop(crop_bbox).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            alpha = torch.tensor(np.array(union_cropped)).float() / 255.0
        cond_image = torch.tensor(np.array(image.convert('RGB'))).permute(2, 0, 1).float() / 255.0
        cond_image = cond_image * alpha.unsqueeze(0)
        prompt_masks = None
        prompt_object_ids_tensor = None
        prompt_permutation = None
        prompt_gt_mesh_paths_ordered: Optional[List[Optional[str]]] = None
        if len(prompt_mask_images) > 0:
            order = np.arange(len(prompt_mask_images), dtype=np.int64)
            if self.random_prompt_order and len(order) > 1:
                order = np.random.permutation(order)
            prompt_masks = torch.stack([self._apply_crop_to_mask(prompt_mask_images[idx], crop_bbox, self.image_size) for idx in order], dim=0)
            prompt_object_ids_tensor = torch.tensor([prompt_object_ids[idx] for idx in order], dtype=torch.long)
            prompt_permutation = torch.tensor(order.tolist(), dtype=torch.long)
            prompt_gt_mesh_paths_ordered = [prompt_gt_mesh_paths[idx] for idx in order]
        return (cond_image, prompt_masks, prompt_object_ids_tensor, prompt_permutation, Path(image_path).name, prompt_gt_mesh_paths_ordered)

    def _load_cond_image(self, instance: str, root) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        cond_image, _, _, _, prompt_view_id, _ = self._load_cond_and_prompt(instance, root)
        return (cond_image, prompt_view_id)

    def _prewarm_render_file_cache(self):
        unique_instances: List[Tuple[Any, str]] = []
        seen_sha256: set = set()
        for root, sha256 in self.instances:
            if sha256 in seen_sha256 or sha256 in self._render_file_cache:
                continue
            seen_sha256.add(sha256)
            unique_instances.append((root, sha256))
        workers_env = os.environ.get('TRELLIS2_RENDER_PREWARM_WORKERS', '1')
        try:
            workers = int(workers_env)
        except ValueError:
            workers = 1
        if workers <= 1:
            for root, sha256 in unique_instances:
                key, files = self._scan_render_candidates_for_instance(root, sha256)
                self._render_file_cache[key] = files
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self._scan_render_candidates_for_instance, root, sha256) for root, sha256 in unique_instances]
                for future in as_completed(futures):
                    key, files = future.result()
                    self._render_file_cache[key] = files
        print(f'[FlexiDualGridMultiObjDataset] Pre-warmed render file cache for {len(self._render_file_cache)} instances (unique scan targets: {len(unique_instances)}, workers: {max(workers, 1)})')

    def _init_voxel_cache(self):
        if self.preload_mode == 'disk':
            return
        if self.preload_mode != 'memory':
            return
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            rank = dist.get_rank()
            self.preload_mode = 'disk'
            if rank == 0:
                print(f'[FlexiDualGridMultiObjDataset] Multi-rank detected (world_size={world_size}), downgrading preload_mode from "memory" to "disk" (lazy caching during training)')
            return
        unique_instances: Dict[str, str] = {}
        for root, sha256 in self.instances:
            if sha256 not in unique_instances:
                unique_instances[sha256] = root['dual_grid']
        self._preload_to_memory(unique_instances)

    def _preload_to_memory(self, unique_instances: Dict[str, str]) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        def _load_one(sha256: str, dual_grid_root: str) -> Tuple[str, Dict]:
            obj_dir = Path(dual_grid_root) / sha256
            with open(obj_dir / 'meta.json', 'r') as f:
                meta = json.load(f)
            objects = []
            for obj_info in meta['objects']:
                obj_path = obj_dir / obj_info['file']
                coords, attr = o_voxel.io.read_vxz(str(obj_path), num_threads=2)
                objects.append({'coords': coords.numpy().astype(np.int32), 'vertices': attr['vertices'].numpy(), 'intersected': attr['intersected'].numpy()})
            return (sha256, {'meta': meta, 'objects': objects})
        n = len(unique_instances)
        print(f'[FlexiDualGridMultiObjDataset] Pre-loading {n} instances into memory...')
        max_workers = min(8, n)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_one, sha256, root): sha256 for sha256, root in unique_instances.items()}
            iterator = as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc='Preloading voxels')
            for future in iterator:
                sha256, data = future.result()
                self._voxel_cache[sha256] = data
        self._print_cache_stats()

    def _print_cache_stats(self) -> None:
        total_voxels = sum((sum((obj['coords'].shape[0] for obj in entry['objects'])) for entry in self._voxel_cache.values()))
        total_bytes = sum((sum((obj['coords'].nbytes + obj['vertices'].nbytes + obj['intersected'].nbytes for obj in entry['objects'])) for entry in self._voxel_cache.values()))
        print(f'[FlexiDualGridMultiObjDataset] Voxel cache: {len(self._voxel_cache)} instances, {total_voxels / 1000000.0:.1f}M voxels, {total_bytes / 1000000000.0:.2f} GB')

    def __len__(self) -> int:
        if self.overfit_id is not None:
            return self.overfit_repeats
        return len(self.instances)

    def __getitem__(self, index):
        if self.overfit_id is not None:
            try:
                return self.get_instance(self._overfit_root, self._overfit_instance)
            except Exception as e:
                print(f'Error loading overfit instance {self._overfit_instance}: {e}')
                raise
        return super().__getitem__(index)

    def __str__(self) -> str:
        lines = [super().__str__(), f'  - Resolution: {self.resolution}', f'  - Max Objects: {self.max_objects}', f'  - Min Objects: {self.min_objects}']
        if self.overfit_id is not None:
            lines.append(f'  - OVERFIT MODE: {self.overfit_id} x {self.overfit_repeats}')
        return '\n'.join(lines)

    def filter_metadata(self, metadata):
        stats = {}
        if self._instance_whitelist is not None:
            metadata = metadata[metadata.index.isin(self._instance_whitelist)]
            stats['Instance whitelist'] = len(metadata)
        if 'multi_obj_converted' in metadata.columns:
            metadata = metadata[metadata['multi_obj_converted'] == True]
            stats['Multi-Object Converted'] = len(metadata)
        elif 'dual_grid_converted' in metadata.columns:
            metadata = metadata[metadata['dual_grid_converted'] == True]
            stats['Dual Grid Converted'] = len(metadata)
        if self.min_aesthetic_score is not None and 'aesthetic_score' in metadata.columns:
            metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        if 'dual_grid_size' in metadata.columns:
            metadata = metadata[metadata['dual_grid_size'] <= self.max_active_voxels]
            stats[f'Active Voxels <= {self.max_active_voxels}'] = len(metadata)
        if self.max_num_faces is not None and 'num_faces' in metadata.columns:
            metadata = metadata[metadata['num_faces'] <= self.max_num_faces]
            stats[f'Faces <= {self.max_num_faces}'] = len(metadata)
        obj_col = 'num_objects' if 'num_objects' in metadata.columns else 'dual_grid_num_objects' if 'dual_grid_num_objects' in metadata.columns else None
        if obj_col is not None:
            metadata = metadata[metadata[obj_col] <= self.max_objects]
            stats[f'Objects <= {self.max_objects}'] = len(metadata)
            if self.min_objects is not None:
                metadata = metadata[metadata[obj_col] >= self.min_objects]
                stats[f'Objects >= {self.min_objects}'] = len(metadata)
        return (metadata, stats)

    @torch.no_grad()
    def visualize_sample(self, sample: Dict[str, Any]) -> torch.Tensor:
        if 'mesh' in sample:
            mesh = sample['mesh']
            if isinstance(mesh, list) and len(mesh) > 0 and isinstance(mesh[0], list):
                sample = dict(sample)
                sample['mesh'] = [m[0] if isinstance(m, list) and len(m) == 1 else m for m in mesh]
            return super().visualize_sample(sample)
        cond_image = sample.get('cond_image')
        if isinstance(cond_image, torch.Tensor):
            if cond_image.dim() == 3:
                return cond_image.unsqueeze(0)
            return cond_image
        batch_size = 1
        num_objects = sample.get('num_objects')
        if isinstance(num_objects, torch.Tensor):
            batch_size = int(num_objects.shape[0]) if num_objects.dim() > 0 else 1
        elif isinstance(num_objects, list):
            batch_size = len(num_objects)
        elif isinstance(num_objects, int):
            batch_size = 1
        elif 'vertices' in sample and hasattr(sample['vertices'], 'coords'):
            coords = sample['vertices'].coords
            if coords.numel() > 0:
                batch_size = int(coords[:, 0].max().item()) + 1
        device = torch.device('cpu')
        if 'vertices' in sample and hasattr(sample['vertices'], 'feats'):
            device = sample['vertices'].feats.device
        return torch.zeros(batch_size, 3, self.image_size, self.image_size, dtype=torch.float32, device=device)

    def _read_from_disk_cache(self, cache_path: Path) -> Optional[Dict]:
        if not cache_path.exists():
            return None
        try:
            data = np.load(str(cache_path), allow_pickle=False)
            num_objects = int(data['meta_num_objects'])
            objects = []
            for i in range(num_objects):
                objects.append({'coords': data[f'obj_{i}_coords'], 'vertices': data[f'obj_{i}_vertices'], 'intersected': data[f'obj_{i}_intersected']})
            return {'meta': {'num_objects': num_objects}, 'objects': objects}
        except Exception:
            return None

    def _write_to_disk_cache(self, cache_path: Path, meta: Dict, objects: List[Dict[str, np.ndarray]]) -> None:
        import tempfile
        if cache_path.exists():
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            arrays: Dict[str, np.ndarray] = {'meta_num_objects': np.array(meta['num_objects'], dtype=np.int32)}
            for i, obj in enumerate(objects):
                arrays[f'obj_{i}_coords'] = obj['coords']
                arrays[f'obj_{i}_vertices'] = obj['vertices']
                arrays[f'obj_{i}_intersected'] = obj['intersected']
            fd, tmp_path = tempfile.mkstemp(suffix='.npz', dir=str(cache_path.parent))
            os.close(fd)
            np.savez(tmp_path, **arrays)
            os.replace(tmp_path, str(cache_path))
        except Exception as e:
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            print(f'[FlexiDualGridMultiObjDataset] Warning: disk cache write failed for {cache_path}: {e}')

    def _build_from_cache(self, cached: Dict) -> Dict[str, Any]:
        meta = cached['meta']
        all_coords = []
        all_vertices = []
        all_intersected = []
        all_instance_id = []
        per_obj_coords = []
        per_obj_vertices = []
        per_obj_intersected = []
        for obj_idx, obj_data in enumerate(cached['objects']):
            coords = torch.from_numpy(np.ascontiguousarray(obj_data['coords']))
            vertices_feats = torch.from_numpy(obj_data['vertices']).float() / 255.0
            intersected_raw = torch.from_numpy(obj_data['intersected'])
            intersected_feats = torch.cat([intersected_raw % 2, intersected_raw // 2 % 2, intersected_raw // 4 % 2], dim=-1).bool()
            instance_id = torch.full((coords.shape[0],), obj_idx, dtype=torch.long)
            all_coords.append(coords)
            all_vertices.append(vertices_feats)
            all_intersected.append(intersected_feats)
            all_instance_id.append(instance_id)
            if not self.skip_per_obj:
                per_obj_coords.append(coords.clone())
                per_obj_vertices.append(vertices_feats.clone())
                per_obj_intersected.append(intersected_feats.clone())
        coords = torch.cat(all_coords, dim=0)
        vertices_feats = torch.cat(all_vertices, dim=0)
        intersected_feats = torch.cat(all_intersected, dim=0)
        instance_id = torch.cat(all_instance_id, dim=0)
        coords_with_batch = torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1)
        vertices = sp.SparseTensor(vertices_feats, coords_with_batch)
        intersected = vertices.replace(intersected_feats)
        result = {'vertices': vertices, 'intersected': intersected, 'instance_id': instance_id, 'num_objects': meta['num_objects']}
        if not self.skip_per_obj:
            result['per_obj_coords'] = per_obj_coords
            result['per_obj_vertices'] = per_obj_vertices
            result['per_obj_intersected'] = per_obj_intersected
        return result

    def read_mesh(self, root: str, instance: str) -> Dict[str, Any]:
        mesh_path = Path(root) / f'{instance}.pickle'
        with open(mesh_path, 'rb') as f:
            dump = pickle.load(f)
        start = 0
        vertices = []
        faces = []
        per_obj_meshes = []
        for obj in dump['objects']:
            if obj['vertices'].size == 0 or obj['faces'].size == 0:
                continue
            obj_verts = torch.from_numpy(obj['vertices']).float()
            obj_faces = torch.from_numpy(obj['faces']).long()
            vertices.append(obj_verts)
            faces.append(obj_faces + start)
            start += len(obj_verts)
            per_obj_meshes.append(Mesh(vertices=obj_verts.clone(), faces=obj_faces.clone()))
        all_vertices = torch.cat(vertices, dim=0)
        all_faces = torch.cat(faces, dim=0)
        vertices_min = all_vertices.min(dim=0)[0]
        vertices_max = all_vertices.max(dim=0)[0]
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        all_vertices = (all_vertices - center) * scale
        for mesh in per_obj_meshes:
            mesh.vertices = (mesh.vertices - center) * scale
        assert torch.all(all_vertices >= -0.5) and torch.all(all_vertices <= 0.5), 'vertices out of range'
        return {'mesh': Mesh(vertices=all_vertices, faces=all_faces), 'per_obj_meshes': per_obj_meshes}

    def read_dual_grid_multi_obj(self, root: str, instance: str) -> Dict[str, Any]:
        if self.preload_mode == 'memory' and instance in self._voxel_cache:
            return self._build_from_cache(self._voxel_cache[instance])
        if self.preload_mode == 'disk':
            cache_path = Path(root) / '_npz_cache' / f'{instance}.npz'
            cached = self._read_from_disk_cache(cache_path)
            if cached is not None:
                return self._build_from_cache(cached)
        obj_dir = Path(root) / instance
        meta_path = obj_dir / 'meta.json'
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        all_coords = []
        all_vertices = []
        all_intersected = []
        all_instance_id = []
        per_obj_coords = []
        per_obj_vertices = []
        per_obj_intersected = []
        _raw_objects = [] if self.preload_mode == 'disk' else None
        for obj_idx, obj_info in enumerate(meta['objects']):
            obj_path = obj_dir / obj_info['file']
            coords, attr = o_voxel.io.read_vxz(str(obj_path), num_threads=4)
            if _raw_objects is not None:
                _raw_objects.append({'coords': coords.numpy().astype(np.int32), 'vertices': attr['vertices'].numpy(), 'intersected': attr['intersected'].numpy()})
            vertices_feats = (attr['vertices'] / 255.0).float()
            intersected_feats = torch.cat([attr['intersected'] % 2, attr['intersected'] // 2 % 2, attr['intersected'] // 4 % 2], dim=-1).bool()
            instance_id = torch.full_like(attr['instance_id'].squeeze(-1), obj_idx, dtype=torch.long)
            all_coords.append(coords)
            all_vertices.append(vertices_feats)
            all_intersected.append(intersected_feats)
            all_instance_id.append(instance_id)
            if not self.skip_per_obj:
                per_obj_coords.append(coords.clone())
                per_obj_vertices.append(vertices_feats.clone())
                per_obj_intersected.append(intersected_feats.clone())
        if _raw_objects is not None:
            cache_path = Path(root) / '_npz_cache' / f'{instance}.npz'
            self._write_to_disk_cache(cache_path, meta, _raw_objects)
        coords = torch.cat(all_coords, dim=0)
        vertices_feats = torch.cat(all_vertices, dim=0)
        intersected_feats = torch.cat(all_intersected, dim=0)
        instance_id = torch.cat(all_instance_id, dim=0)
        coords_with_batch = torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1)
        vertices = sp.SparseTensor(vertices_feats, coords_with_batch)
        intersected = vertices.replace(intersected_feats)
        result = {'vertices': vertices, 'intersected': intersected, 'instance_id': instance_id, 'num_objects': meta['num_objects']}
        if not self.skip_per_obj:
            result['per_obj_coords'] = per_obj_coords
            result['per_obj_vertices'] = per_obj_vertices
            result['per_obj_intersected'] = per_obj_intersected
        return result

    def read_cond_image(self, instance: str, root=None) -> Optional[torch.Tensor]:
        cond_image, _ = self._load_cond_image(instance, root)
        return cond_image

    def get_instance(self, root: Dict[str, str], instance: str) -> Dict[str, Any]:
        if not self.skip_mesh:
            mesh_data = self.read_mesh(root['mesh_dump'], instance)
        else:
            mesh_data = {}
        dual_grid_data = self.read_dual_grid_multi_obj(root['dual_grid'], instance)
        pack = {**mesh_data, **dual_grid_data}
        pack['sha256'] = instance
        pack['dataset_key'] = self._resolve_dataset_key(root) or ''
        render_root = self._resolve_render_root(root)
        file_id = self._sha256_to_file_id.get(instance)
        prompt_mode = None
        prompt_masks = None
        prompt_mask_object_ids = None
        prompt_permutation = None
        prompt_view_id = None
        if self._should_emit_prompt_masks():
            if render_root is not None and file_id is not None:
                prompt_mode = self._resolve_prompt_mask_mode(root)
            original_to_contiguous = self._build_original_to_contiguous_index(self._load_original_indices(root, instance, render_root=render_root, file_id=file_id, prompt_mode=prompt_mode))
            cond_image, prompt_masks, prompt_mask_object_ids, prompt_permutation, prompt_view_id, prompt_gt_mesh_paths = self._load_cond_and_prompt(instance, root=root, original_to_contiguous=original_to_contiguous)
        else:
            prompt_gt_mesh_paths = None
            cond_image, _ = self._load_cond_image(instance, root=root)
        if render_root is not None and self.strict_cond_image and (cond_image is None):
            raise RuntimeError(f'Missing cond_image for instance={instance} under render_root={render_root}')
        if cond_image is not None:
            pack['cond_image'] = cond_image
        if self._should_emit_prompt_masks() and prompt_mode is not None and self.strict_prompt_masks and (prompt_masks is None):
            raise RuntimeError(f'Missing prompt masks for instance={instance} under render_root={render_root}')
        if self._should_emit_prompt_masks() and prompt_masks is not None:
            keep = prompt_masks.flatten(1).sum(dim=1) >= 1
            if not bool(keep.all()):
                dropped = (~keep).nonzero(as_tuple=False).flatten().tolist()
                dropped_obj_ids = [int(prompt_mask_object_ids[i].item()) for i in dropped] if prompt_mask_object_ids is not None else dropped
                logger.debug('instance=%s file_id=%s render_root=%s dropped %d empty prompts (object_ids=%s)', instance, file_id, render_root, len(dropped), dropped_obj_ids)
                prompt_masks = prompt_masks[keep]
                if prompt_mask_object_ids is not None:
                    prompt_mask_object_ids = prompt_mask_object_ids[keep]
                if prompt_permutation is not None:
                    prompt_permutation = prompt_permutation[keep]
                if prompt_gt_mesh_paths is not None:
                    keep_idx = keep.nonzero(as_tuple=False).flatten().tolist()
                    prompt_gt_mesh_paths = [prompt_gt_mesh_paths[i] for i in keep_idx]
                if prompt_masks.shape[0] == 0:
                    raise RuntimeError(f'All prompt masks empty after filter for instance={instance} file_id={file_id} render_root={render_root}')
            self._validate_prompt_object_ids(prompt_mask_object_ids, num_objects=int(pack['num_objects']), instance=instance, render_root=render_root, file_id=file_id)
            pack['prompt_masks'] = prompt_masks
            pack['prompt_mask_num_queries'] = int(prompt_masks.shape[0])
            pack['prompt_mask_object_ids'] = prompt_mask_object_ids
            pack['prompt_permutation'] = prompt_permutation
            pack['prompt_view_id'] = prompt_view_id
            if prompt_gt_mesh_paths is not None:
                pack['prompt_gt_mesh_paths'] = prompt_gt_mesh_paths
        slr = self._resolve_shape_latent_root(root)
        latent_coords = None
        if slr is not None:
            npz_path = Path(slr) / f'{instance}.npz'
            with np.load(npz_path) as npz:
                latent_coords = torch.from_numpy(npz['coords'].astype(np.int32))
                latent_feats = torch.from_numpy(npz['feats'].copy())
            coords_with_batch = torch.cat([torch.zeros(latent_coords.shape[0], 1, dtype=torch.int32), latent_coords], dim=-1)
            from fractions import Fraction
            downsample_ratio = self.resolution // (latent_coords.max().item() + 1)
            latent_scale = tuple((Fraction(downsample_ratio, 1) for _ in range(3)))
            precomputed_z = sp.SparseTensor(latent_feats, coords_with_batch)
            precomputed_z._scale = latent_scale
            pack['precomputed_z'] = precomputed_z
        elif self._should_emit_prompt_points():
            raise RuntimeError(f"prompt_query_mode='{self.prompt_query_mode}' requires shape_latent_root, but none was resolved for instance={instance}")
        if self._should_emit_prompt_points():
            assert latent_coords is not None
            prompt_points, prompt_object_ids = self._build_prompt_points(latent_coords=latent_coords, scene_coords=pack['vertices'].coords[:, 1:4], instance_id=pack['instance_id'], num_objects=int(pack['num_objects']), instance=instance, shape_latent_path=npz_path)
            self._validate_prompt_object_ids(prompt_object_ids, num_objects=int(pack['num_objects']), instance=instance, render_root=render_root, file_id=file_id)
            pack['prompt_points'] = prompt_points
            pack['prompt_point_num_queries'] = int(prompt_points.shape[0])
            pack['prompt_point_object_ids'] = prompt_object_ids
        if latent_coords is not None:
            ufr = self._resolve_utonia_feats_root(root)
            if ufr is not None:
                utonia_npz_path = os.path.join(ufr, f'{instance}.npz')
                if not os.path.exists(utonia_npz_path):
                    raise FileNotFoundError(f'utonia_feats_root is set but {utonia_npz_path} is missing. Run encode_utonia_features_multi_obj.py first or remove utonia_feats_root from config.')
                with np.load(utonia_npz_path) as u_npz:
                    utonia_coords = torch.from_numpy(u_npz['coords'].astype(np.int32))
                    utonia_feats = torch.from_numpy(u_npz['feats'].copy())
                assert torch.equal(utonia_coords, latent_coords), f'utonia_coords != latent_coords for {instance} (shapes {utonia_coords.shape} vs {latent_coords.shape}); re-run encode_utonia_features_multi_obj.py.'
                precomputed_utonia = sp.SparseTensor(utonia_feats, coords_with_batch)
                precomputed_utonia._scale = latent_scale
                pack['precomputed_utonia'] = precomputed_utonia
        return pack

    @staticmethod
    def collate_fn(batch: List[Dict], split_size: Optional[int]=None) -> Any:
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['vertices'].feats.shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            expected_keys = set(sub_batch[0].keys())
            for sample_idx, sample in enumerate(sub_batch[1:], start=1):
                sample_keys = set(sample.keys())
                if sample_keys != expected_keys:
                    missing = sorted(expected_keys - sample_keys)
                    extra = sorted(sample_keys - expected_keys)
                    raise KeyError(f'Inconsistent keys in batch at sample_idx={sample_idx}: missing={missing}, extra={extra}')
            keys = list(sub_batch[0].keys())
            for k in keys:
                first_val = sub_batch[0][k]
                if isinstance(first_val, torch.Tensor):
                    if any((b[k].shape != first_val.shape for b in sub_batch)):
                        pack[k] = [b[k] for b in sub_batch]
                    else:
                        pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(first_val, sp.SparseTensor):
                    pack[k] = sp.sparse_cat([b[k] for b in sub_batch], dim=0)
                elif isinstance(first_val, list):
                    pack[k] = [b[k] for b in sub_batch]
                elif isinstance(first_val, (int, float)):
                    pack[k] = torch.tensor([b[k] for b in sub_batch])
                else:
                    pack[k] = [b[k] for b in sub_batch]
            packs.append(pack)
        if split_size is None:
            return packs[0]
        return packs
