from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional
import numpy as np
from mesh_eval.core.data_structures import MeshPair
from mesh_eval.core.evaluator import MeshEvaluator
from mesh_eval.domain import InputImageRecord, PairedSampleRecord

def _paired_sample_to_mesh_pair(pair: PairedSampleRecord, *, surface_point_sample_count: Optional[int], object_matching: str='filename') -> MeshPair:
    pred_objects = [obj for obj in pair.pred_sample.objects if obj.mesh_path is not None]
    gt_objects = [obj for obj in pair.gt_sample.objects if obj.mesh_path is not None]
    pred_paths = [str(obj.mesh_path) for obj in pred_objects]
    gt_paths = [str(obj.mesh_path) for obj in gt_objects]
    if not pred_paths and pair.pred_sample.scene_mesh_path is not None:
        pred_paths = [str(pair.pred_sample.scene_mesh_path)]
    input_images = _merge_input_images(pair.pred_sample.input_images, pair.gt_sample.input_images)
    pred_names = [_object_match_name(obj) for obj in pred_objects] or None
    gt_names = [_object_match_name(obj) for obj in gt_objects] or None
    pred_old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate((idx for idx, obj in enumerate(pair.pred_sample.objects) if obj.mesh_path is not None))}
    gt_old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate((idx for idx, obj in enumerate(pair.gt_sample.objects) if obj.mesh_path is not None))}
    remapped_matches = [(pred_old_to_new[pred_idx], gt_old_to_new[gt_idx]) for pred_idx, gt_idx in pair.object_matches if pred_idx in pred_old_to_new and gt_idx in gt_old_to_new]
    gt_meta = pair.gt_sample.metadata or {}
    inst_ids_per_object_all = gt_meta.get('inst_ids_per_object')
    inst_ids_per_mesh: Optional[List[str]] = None
    if inst_ids_per_object_all is not None:
        kept_indices = [idx for idx, obj in enumerate(pair.gt_sample.objects) if obj.mesh_path is not None]
        inst_ids_per_mesh = [str(inst_ids_per_object_all[idx]) for idx in kept_indices if idx < len(inst_ids_per_object_all) and inst_ids_per_object_all[idx] is not None]
        if len(inst_ids_per_mesh) != len(kept_indices):
            inst_ids_per_mesh = None
    pre_alignment_matching = object_matching not in ('centroid', 'mask_iou')
    mesh_pair = MeshPair(pred_meshes=[], gt_meshes=[], sample_id=pair.actual_sample_id, canonical_sample_id=pair.canonical_sample_id, pred_object_names=pred_names, gt_object_names=gt_names, input_images=input_images, render_config=pair.gt_sample.render_config, eval_diagnostics={'canonical_sample_id': pair.canonical_sample_id, 'pair_diagnostics': list(pair.diagnostics), 'unmatched_pred_indices': list(pair.unmatched_pred_indices), 'unmatched_gt_indices': list(pair.unmatched_gt_indices)}, precomputed_object_matches=remapped_matches if pre_alignment_matching else None, gt_instance_mask_path=gt_meta.get('instance_mask_path'), gt_inst_ids_full=gt_meta.get('inst_ids_full'), gt_inst_ids_per_mesh=inst_ids_per_mesh, _loader_type='standard_paths', _lazy_paths={'pred_paths': pred_paths, 'gt_paths': gt_paths, 'pred_scene_mesh_path': str(pair.pred_sample.scene_mesh_path) if pair.pred_sample.scene_mesh_path else None, 'gt_scene_mesh_path': str(pair.gt_sample.scene_mesh_path) if pair.gt_sample.scene_mesh_path else None}, _loaded=False)
    gt_surface_points = _load_gt_surface_points(gt_objects, surface_point_sample_count=surface_point_sample_count)
    if gt_surface_points is not None:
        mesh_pair.gt_surface_points = gt_surface_points
    return mesh_pair

def _object_match_name(obj: object) -> str:
    return getattr(obj, 'match_key', None) or getattr(obj, 'canonical_name', None) or getattr(obj, 'display_id', None) or getattr(obj, 'object_id', '')

def _merge_input_images(pred_images: Iterable[InputImageRecord], gt_images: Iterable[InputImageRecord]) -> List[tuple[str, str]]:
    merged: Dict[str, str] = {}
    for item in list(pred_images) + list(gt_images):
        merged[item.dest_name] = str(item.path)
    return [(path, dest_name) for dest_name, path in merged.items()]

def _load_gt_surface_points(gt_objects: List[object], *, surface_point_sample_count: Optional[int]) -> Optional[List[np.ndarray]]:
    surface_paths = [getattr(obj, 'surface_points_path', None) for obj in gt_objects]
    if not any(surface_paths):
        return None
    points_list: List[np.ndarray] = []
    for surface_path in surface_paths:
        if surface_path is None:
            return None
        data = np.load(surface_path, allow_pickle=True).tolist()
        points = data['surface_points'].astype(np.float32)
        if surface_point_sample_count is not None:
            if points.shape[0] < surface_point_sample_count:
                raise ValueError(f'Surface point count {points.shape[0]} < required {surface_point_sample_count}: {surface_path}')
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(points.shape[0], surface_point_sample_count, replace=False)
            points = points[indices]
        points_list.append(points)
    return points_list

def _surface_point_sample_count(gt_adapter: object, pred_adapter: object) -> Optional[int]:
    for config_obj in (getattr(gt_adapter, 'config', None), getattr(pred_adapter, 'config', None)):
        if config_obj is None:
            continue
        value = getattr(config_obj, 'num_surface_samples_per_object', None)
        if value is not None:
            return int(value)
    return None
