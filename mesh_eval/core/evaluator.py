from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from mesh_eval.config import GeometryConfig
import os
import csv
import json
import logging
import shutil
import threading
from contextlib import nullcontext
from copy import deepcopy
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import trimesh
from mesh_eval.core.data_structures import MeshPair
from mesh_eval.core.sampling import surface_points
from mesh_eval.metrics.alignment import align_point_clouds, _coarse_align_bbox
from mesh_eval.metrics.geom_metrics import compute_chamfer_distance, compute_fscore, compute_volume_iou, normalize_points
from mesh_eval.utils.mesh_utils import match_objects_by_centroid, match_objects_by_mask_iou, match_objects_by_name, transform_centroids
logger = logging.getLogger(__name__)

def _serialize_transform(transform: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, list]]:
    if not transform:
        return None
    out: Dict[str, list] = {}
    for key, value in transform.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out

class MeshEvaluator:

    def __init__(self, config: GeometryConfig, *, output_dir: Path, object_matching: str, device: str='cuda:0') -> None:
        if object_matching not in {'centroid', 'filename'}:
            raise ValueError(f'Unsupported CoLa3D object matching mode: {object_matching}')
        self.config = config
        self.output_dir = str(output_dir)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and (not torch.cuda.is_available()):
            raise RuntimeError('CUDA is unavailable; select --device cpu explicitly to evaluate on CPU.')
        self.adapter = SimpleNamespace(config=SimpleNamespace(object_matching=object_matching))
        self._gpu_semaphore = threading.Semaphore(1)
        self._cameras_lock = threading.Lock()

    def _save_sample_results(self, sample_id: str, metrics: Dict):
        sample_dir = os.path.join(self.output_dir, 'intermediates', sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        results_path = os.path.join(sample_dir, 'results.json')
        public_metrics = {k: v for k, v in metrics.items() if not str(k).startswith('__')}
        with open(results_path, 'w') as f:
            json.dump(public_metrics, f, indent=2)
        logger.debug(f'Sample results saved to {results_path}')
        scene_alignment_info = metrics.get('__scene_alignment_info')
        if scene_alignment_info is not None:
            alignment_path = os.path.join(sample_dir, 'alignment_decision.json')
            with open(alignment_path, 'w') as f:
                json.dump(scene_alignment_info, f, indent=2)
            logger.debug(f'Alignment decision saved to {alignment_path}')

    def _metrics_autocast_context(self):
        return nullcontext()

    def _get_chamfer_distance_backend(self):
        return None

    def _copy_input_images(self, pair: MeshPair) -> None:
        if not self.config.save_intermediates:
            return
        if not pair.input_images:
            return
        sample_dir = os.path.join(self.output_dir, 'intermediates', pair.sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        for src_path, dest_name in pair.input_images:
            if not os.path.isfile(src_path):
                logger.warning(f'{pair.sample_id}: Input image not found, skipping: {src_path}')
                continue
            dest_path = os.path.join(sample_dir, dest_name)
            try:
                shutil.copy2(src_path, dest_path)
                logger.debug(f'Copied input image: {src_path} -> {dest_path}')
            except Exception as e:
                logger.warning(f'{pair.sample_id}: Failed to copy {src_path}: {e}')

    def _save_summary_results(self, summary: Dict):
        results_path = os.path.join(self.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f'Summary results saved to {results_path}')

    def _eval_scene_3d(self, pair: MeshPair) -> Dict[str, float]:
        pred_np, gt_np = self._cpu_phase_scene_3d(pair)
        return self._gpu_phase_scene_3d(pred_np, gt_np)

    def _cpu_phase_scene_3d(self, pair: MeshPair) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug(f'{pair.sample_id}: Sampling pointclouds...')
        scene_sampling_strategy = getattr(self.config, 'scene_sampling_strategy', 'per_object_concat')
        if scene_sampling_strategy == 'per_object_concat':
            pred_points = self._sample_scene_points_from_object_meshes(pair.pred_meshes)
            gt_points = self._sample_scene_points_from_object_meshes(pair.gt_meshes)
        elif scene_sampling_strategy == 'gt_matched_total':
            gt_points = self._sample_scene_points_from_object_meshes(pair.gt_meshes)
            total_points = len(gt_points)
            merged_pred = self._merge_meshes_for_scene(pair.pred_meshes)
            pred_points = self._sample_n_points_from_mesh(merged_pred, total_points)
        elif scene_sampling_strategy == 'merged_mesh':
            merged_pred = self._merge_meshes_for_scene(pair.pred_meshes)
            merged_gt = self._merge_meshes_for_scene(pair.gt_meshes)
            pred_points = self._sample_points_from_mesh(merged_pred)
            gt_points = self._sample_points_from_mesh(merged_gt)
        else:
            raise ValueError(f'Unsupported scene_sampling_strategy: {scene_sampling_strategy}')
        logger.debug(f'{pair.sample_id}: Sampled {len(pred_points)} pred, {len(gt_points)} gt points')
        if self.config.alignment_scene:
            pred_batch = pred_points.unsqueeze(0)
            gt_batch = gt_points.unsqueeze(0)
            pred_np = pred_points.numpy()
            gt_np = gt_points.numpy()
            rotation_search = False
            coarse_points_np = None
            pre_fine_points_np = None
            pca_state = 'pca' if self.config.alignment_coarse_pca else 'no-pca'
            rotation_search = getattr(self.config, 'alignment_coarse_rotation_search', False)
            global_method = getattr(self.config, 'alignment_global_method', 'none')
            global_voxel_ratio = getattr(self.config, 'alignment_global_voxel_size_ratio', 0.02)
            global_ransac_n = getattr(self.config, 'alignment_global_ransac_n', 3)
            global_ransac_max_iter = getattr(self.config, 'alignment_global_ransac_max_iter', 100000)
            if rotation_search and global_method != 'none':
                logger.info(f'{pair.sample_id}: rotation_search skipped (global_method={global_method} handles arbitrary rotations)')
                rotation_search = False
            if rotation_search:
                logger.info(f'{pair.sample_id}: Starting alignment (coarse-{pca_state}+rotation_search+{self.config.alignment_method})...')
                pred_coarse, coarse_info = align_point_clouds(pred_batch, gt_batch, coarse_method='bbox', fine_method='none', coarse_pca=self.config.alignment_coarse_pca)
                best_R, best_score = self._find_best_90deg_rotation(pred_coarse.squeeze(0).numpy(), gt_batch.squeeze(0).numpy())
                pred_rotated_np = pred_coarse.squeeze(0).numpy() @ best_R.T
                logger.info(f'{pair.sample_id}: Best coarse rotation score (inlier_ratio)={best_score:.4f}')
                coarse_points_np = pred_rotated_np
                pre_fine_points_np = pred_rotated_np
                pred_rotated_batch = torch.from_numpy(pred_rotated_np).float().unsqueeze(0)
                pred_batch_aligned, fine_info = align_point_clouds(pred_rotated_batch, gt_batch, coarse_method='none', fine_method=self.config.alignment_method, max_iterations=self.config.alignment_max_iterations, tolerance=self.config.alignment_tolerance, fine_max_points=getattr(self.config, 'alignment_max_points', None))
                transform_info = {'coarse_scale': coarse_info['coarse_scale'], 'coarse_R': coarse_info['coarse_R'], 'rotation_search_R': [best_R], 'fine_R': fine_info['fine_R'], 'fine_t': fine_info['fine_t']}
            else:
                global_label = f'+{global_method}' if global_method != 'none' else ''
                logger.info(f'{pair.sample_id}: Starting alignment (coarse-{pca_state}{global_label}+{self.config.alignment_method})...')
                pred_batch_aligned, transform_info = align_point_clouds(pred_batch, gt_batch, coarse_method='bbox', global_method=global_method, fine_method=self.config.alignment_method, coarse_pca=self.config.alignment_coarse_pca, max_iterations=self.config.alignment_max_iterations, tolerance=self.config.alignment_tolerance, global_voxel_size_ratio=global_voxel_ratio, global_ransac_n=global_ransac_n, global_ransac_max_iter=global_ransac_max_iter, fine_max_points=getattr(self.config, 'alignment_max_points', None))
                coarse_scale = float(transform_info['coarse_scale'][0])
                coarse_R = np.asarray(transform_info['coarse_R'][0], dtype=np.float64)
                pred_min, pred_max = (pred_np.min(axis=0), pred_np.max(axis=0))
                gt_min, gt_max = (gt_np.min(axis=0), gt_np.max(axis=0))
                pred_center = (pred_max + pred_min) / 2.0
                gt_center = (gt_max + gt_min) / 2.0
                coarse_points_np = (pred_np - pred_center) * coarse_scale @ coarse_R.T + gt_center
                if 'global_R' in transform_info:
                    g_R = np.asarray(transform_info['global_R'][0], dtype=np.float64)
                    g_t = np.asarray(transform_info['global_t'][0], dtype=np.float64)
                    pre_fine_points_np = (g_R @ coarse_points_np.T).T + g_t
                else:
                    pre_fine_points_np = coarse_points_np
            pred_points_aligned = pred_batch_aligned.squeeze(0)
            if pre_fine_points_np is not None and self.config.alignment_method != 'none':
                pre_fine_stats = self._compute_alignment_quality_stats(pre_fine_points_np, gt_np)
                final_stats = self._compute_alignment_quality_stats(pred_points_aligned.numpy(), gt_np)
                good_thr = float(getattr(self.config, 'alignment_fallback_good_mean_dist', 0.1))
                worse_delta = float(getattr(self.config, 'alignment_fallback_min_worse_delta', 0.0))
                pre_fine_good = pre_fine_stats['mean_dist'] <= good_thr
                final_worse = final_stats['mean_dist'] > pre_fine_stats['mean_dist'] + worse_delta
                fallback_enabled = bool(getattr(self.config, 'alignment_fallback_to_coarse', True))
                fallback_used = bool(fallback_enabled and pre_fine_good and final_worse)
                selected_stage = 'pre_fine' if fallback_used else 'final'
                if fallback_used:
                    pred_points_aligned = torch.from_numpy(pre_fine_points_np).float()
                    logger.info(f"{pair.sample_id}: Pre-fine fallback triggered (pre_fine_mean={pre_fine_stats['mean_dist']:.4f}, final_mean={final_stats['mean_dist']:.4f})")
                if pair.eval_diagnostics is None:
                    pair.eval_diagnostics = {}
                pair.eval_diagnostics['scene_alignment'] = {'selected_stage': selected_stage, 'fallback_used': fallback_used, 'fallback_enabled': fallback_enabled, 'pre_fine_good': bool(pre_fine_good), 'pre_fine_mean_dist': pre_fine_stats['mean_dist'], 'pre_fine_median_dist': pre_fine_stats['median_dist'], 'final_mean_dist': final_stats['mean_dist'], 'final_median_dist': final_stats['median_dist'], 'good_mean_dist_threshold': good_thr, 'worse_delta_threshold': worse_delta, 'alignment_method': self.config.alignment_method, 'global_method': global_method}
            _src_ctr = (pred_np.min(0) + pred_np.max(0)) / 2.0
            _dst_ctr = (gt_np.min(0) + gt_np.max(0)) / 2.0
            _cR = np.asarray(transform_info['coarse_R'][0], dtype=np.float64)
            if rotation_search and 'rotation_search_R' in transform_info:
                _cR = np.asarray(transform_info['rotation_search_R'][0], dtype=np.float64) @ _cR
            _fallback = pair.eval_diagnostics.get('scene_alignment', {}).get('fallback_used', False)
            if _fallback:
                _fR = np.eye(3, dtype=np.float64)
                _ft = np.zeros(3, dtype=np.float64)
            else:
                _fR = np.asarray(transform_info['fine_R'][0], dtype=np.float64)
                _ft = np.asarray(transform_info['fine_t'][0], dtype=np.float64)
            scene_xform: Dict[str, Any] = {'src_center': _src_ctr, 'dst_center': _dst_ctr, 'coarse_scale': float(transform_info['coarse_scale'][0]), 'coarse_R': _cR, 'fine_R': _fR, 'fine_t': _ft}
            if 'global_R' in transform_info:
                scene_xform['global_R'] = np.asarray(transform_info['global_R'][0], dtype=np.float64)
                scene_xform['global_t'] = np.asarray(transform_info['global_t'][0], dtype=np.float64)
            pair.scene_alignment_transform = scene_xform
            coarse_scale = transform_info['coarse_scale'][0]
            global_method_used = getattr(self.config, 'alignment_global_method', 'none')
            logger.info(f'{pair.sample_id}: coarse_scale={coarse_scale:.4f}, global_method={global_method_used}, fine_method={self.config.alignment_method}')
            if self.config.save_intermediates and coarse_points_np is not None:
                self._save_pointcloud(coarse_points_np, pair.sample_id, 'pred_coarse_aligned')
                self._save_alignment_visualization(coarse_points_np, gt_np, pair.sample_id, suffix='_coarse')
        else:
            pred_points_aligned = pred_points
        if getattr(self.config, 'scene_joint_normalize', True):
            combined = torch.cat([pred_points_aligned, gt_points], dim=0)
            combined_min = combined.min(dim=0, keepdim=True)[0]
            combined_max = combined.max(dim=0, keepdim=True)[0]
            combined_range = (combined_max - combined_min).max()
            combined_center = (combined_max + combined_min) / 2
            if combined_range > 1e-08:
                pred_for_metrics = (pred_points_aligned - combined_center) / combined_range * 1.9
                gt_for_metrics = (gt_points - combined_center) / combined_range * 1.9
            else:
                pred_for_metrics = pred_points_aligned
                gt_for_metrics = gt_points
        else:
            pred_for_metrics = pred_points_aligned
            gt_for_metrics = gt_points
        if self.config.save_intermediates:
            pred_np_final = pred_for_metrics.numpy() if isinstance(pred_for_metrics, torch.Tensor) else pred_for_metrics
            gt_np_final = gt_for_metrics.numpy() if isinstance(gt_for_metrics, torch.Tensor) else gt_for_metrics
            self._save_pointcloud(pred_np_final, pair.sample_id, 'pred_final_aligned')
            self._save_pointcloud(gt_np_final, pair.sample_id, 'gt_sampled')
            self._save_alignment_visualization(pred_np_final, gt_np_final, pair.sample_id, suffix='_final')
        return (pred_for_metrics.numpy().astype(np.float32), gt_for_metrics.numpy().astype(np.float32))

    def _gpu_phase_scene_3d(self, pred_np: np.ndarray, gt_np: np.ndarray) -> Dict[str, float]:
        pred_t = torch.from_numpy(pred_np).float()
        gt_t = torch.from_numpy(gt_np).float()
        chamfer_backend = self._get_chamfer_distance_backend()
        with self._gpu_semaphore, self._metrics_autocast_context():
            cd_total, cd_pred2gt, cd_gt2pred = compute_chamfer_distance(pred_t.unsqueeze(0).to(self.device), gt_t.unsqueeze(0).to(self.device), chamfer_distance_cls=chamfer_backend, chunk_size=self.config.chunk_size)
            fscore, precision, recall = compute_fscore(pred_t.unsqueeze(0).to(self.device), gt_t.unsqueeze(0).to(self.device), tau=self.config.fscore_threshold)
            iou = compute_volume_iou(pred_t.unsqueeze(0).to(self.device), gt_t.unsqueeze(0).to(self.device), mode='bbox')
        return {'scene_cd': cd_total.item(), 'scene_cd_pred2gt': cd_pred2gt.item(), 'scene_cd_gt2pred': cd_gt2pred.item(), 'scene_fscore': fscore.mean().item(), 'scene_precision': precision.mean().item(), 'scene_recall': recall.mean().item(), 'scene_iou_bbox': iou.mean().item()}

    def _eval_object_3d(self, pair: MeshPair) -> Dict[str, float]:
        obj_pairs, n, mode = self._cpu_phase_object_3d(pair)
        return self._gpu_phase_object_3d(obj_pairs, n, mode)

    def _cpu_phase_object_3d(self, pair: MeshPair) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], int, str]:
        pred_meshes = pair.pred_meshes
        gt_meshes = pair.gt_meshes
        matches, match_mode = self._select_object_matches(pair)
        self._save_object_matches(pair, matches, match_mode)
        num_objects = len(matches)
        if num_objects == 0:
            logger.warning(f'{pair.sample_id}: no matched object pairs (mode={match_mode})')
        object_pairs = []
        init_from_scene = getattr(self.config, 'alignment_object_init_from_scene', False) and pair.scene_alignment_transform is not None
        for pred_idx, gt_idx in matches:
            pred_points_raw = self._sample_points_from_mesh(pred_meshes[pred_idx])
            gt_points_raw = self._sample_points_from_mesh(gt_meshes[gt_idx])
            if init_from_scene:
                pred_np = pred_points_raw.numpy().astype(np.float64)
                pred_np = transform_centroids(pred_np, pair.scene_alignment_transform)
                pred_points_raw = torch.from_numpy(pred_np.astype(np.float32))
            pred_points = normalize_points(pred_points_raw.unsqueeze(0), uniform=getattr(self.config, 'object_normalize_uniform', True)).squeeze(0)
            gt_points = normalize_points(gt_points_raw.unsqueeze(0), uniform=getattr(self.config, 'object_normalize_uniform', True)).squeeze(0)
            if self.config.alignment_object:
                pred_batch = pred_points.unsqueeze(0)
                gt_batch = gt_points.unsqueeze(0)
                pred_batch_aligned, _ = align_point_clouds(pred_batch, gt_batch, coarse_method='none', fine_method=self.config.alignment_method, max_iterations=self.config.alignment_max_iterations, tolerance=self.config.alignment_tolerance, fine_max_points=getattr(self.config, 'alignment_max_points', None))
                pred_points = pred_batch_aligned.squeeze(0)
            if self.config.save_intermediates and getattr(self.config, 'save_object_alignment', False):
                gt_name = pair.gt_object_names[gt_idx] if pair.gt_object_names and gt_idx < len(pair.gt_object_names) else None
                pred_name = pair.pred_object_names[pred_idx] if pair.pred_object_names and pred_idx < len(pair.pred_object_names) else None
                if gt_name:
                    obj_name = f"pred_{pred_idx}_{pred_name or 'x'}__gt_{gt_idx}_{gt_name}"
                else:
                    obj_name = f'pred_{pred_idx}__gt_{gt_idx}'
                self._save_object_alignment_visualization(pred_points.numpy().astype(np.float32), gt_points.numpy().astype(np.float32), pair.sample_id, obj_name)
            if getattr(self.config, 'object_iou_before_normalization', False):
                iou_pred_pts = pred_points_raw
                iou_gt_pts = gt_points_raw
            else:
                iou_pred_pts = pred_points
                iou_gt_pts = gt_points
            object_pairs.append((pred_points.numpy().astype(np.float32), gt_points.numpy().astype(np.float32), iou_pred_pts.numpy().astype(np.float32), iou_gt_pts.numpy().astype(np.float32)))
        return (object_pairs, num_objects, match_mode)

    def _gpu_phase_object_3d(self, object_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], num_objects: int, match_mode: str) -> Dict[str, float]:
        cd_total_list = []
        cd_p2g_list = []
        cd_g2p_list = []
        fscore_list = []
        precision_list = []
        recall_list = []
        iou_list = []
        chamfer_backend = self._get_chamfer_distance_backend()
        for pred_np, gt_np, iou_pred_np, iou_gt_np in object_pairs:
            pred_t = torch.from_numpy(pred_np).float()
            gt_t = torch.from_numpy(gt_np).float()
            iou_pred_t = torch.from_numpy(iou_pred_np).float()
            iou_gt_t = torch.from_numpy(iou_gt_np).float()
            with self._gpu_semaphore, self._metrics_autocast_context():
                cd_total, cd_pred2gt, cd_gt2pred = compute_chamfer_distance(pred_t.unsqueeze(0).to(self.device), gt_t.unsqueeze(0).to(self.device), chamfer_distance_cls=chamfer_backend, chunk_size=self.config.chunk_size)
                fscore, precision, recall = compute_fscore(pred_t.unsqueeze(0).to(self.device), gt_t.unsqueeze(0).to(self.device), tau=self.config.fscore_threshold)
                iou = compute_volume_iou(iou_pred_t.unsqueeze(0).to(self.device), iou_gt_t.unsqueeze(0).to(self.device), mode='bbox')
            cd_total_list.append(cd_total.item())
            cd_p2g_list.append(cd_pred2gt.item())
            cd_g2p_list.append(cd_gt2pred.item())
            fscore_list.append(fscore.item())
            precision_list.append(precision.item())
            recall_list.append(recall.item())
            iou_list.append(iou.item())
        return {'object_cd': np.mean(cd_total_list) if cd_total_list else 0.0, 'object_cd_pred2gt': np.mean(cd_p2g_list) if cd_p2g_list else 0.0, 'object_cd_gt2pred': np.mean(cd_g2p_list) if cd_g2p_list else 0.0, 'object_fscore': np.mean(fscore_list) if fscore_list else 0.0, 'object_precision': np.mean(precision_list) if precision_list else 0.0, 'object_recall': np.mean(recall_list) if recall_list else 0.0, 'object_iou_bbox': np.mean(iou_list) if iou_list else 0.0, 'object_count': num_objects, '__object_match_mode': match_mode, '__object_cd_sum': float(np.sum(cd_total_list)) if cd_total_list else 0.0, '__object_cd_pred2gt_sum': float(np.sum(cd_p2g_list)) if cd_p2g_list else 0.0, '__object_cd_gt2pred_sum': float(np.sum(cd_g2p_list)) if cd_g2p_list else 0.0, '__object_fscore_sum': float(np.sum(fscore_list)) if fscore_list else 0.0, '__object_precision_sum': float(np.sum(precision_list)) if precision_list else 0.0, '__object_recall_sum': float(np.sum(recall_list)) if recall_list else 0.0, '__object_iou_bbox_sum': float(np.sum(iou_list)) if iou_list else 0.0, '__object_metric_count': int(num_objects)}

    def _select_object_matches(self, pair: MeshPair) -> Tuple[List[Tuple[int, int]], str]:
        pred_meshes = pair.pred_meshes
        gt_meshes = pair.gt_meshes
        scene_transform = pair.scene_alignment_transform
        if pair.precomputed_object_matches is not None:
            return (list(pair.precomputed_object_matches), 'precomputed')
        matching = getattr(self.adapter.config, 'object_matching', 'none')
        if matching == 'filename':
            if pair.pred_object_names and pair.gt_object_names:
                return (match_objects_by_name(pair.pred_object_names, pair.gt_object_names), 'filename')
            logger.warning(f'{pair.sample_id}: filename matching requested but object names not provided, falling back to centroid matching')
            return (match_objects_by_centroid(pred_meshes, gt_meshes, scene_transform), 'centroid')
        if matching == 'centroid':
            return (match_objects_by_centroid(pred_meshes, gt_meshes, scene_transform), 'centroid')
        if matching == 'mask_iou':
            mask_matches = self._match_objects_by_mask_iou(pair, scene_transform)
            if mask_matches is None:
                logger.warning(f'{pair.sample_id}: mask_iou matching unavailable (missing instance mask / camera / inst_ids), falling back to centroid')
                return (match_objects_by_centroid(pred_meshes, gt_meshes, scene_transform), 'centroid')
            return (mask_matches, 'mask_iou')
        n = min(len(pred_meshes), len(gt_meshes))
        return ([(i, i) for i in range(n)], 'index')

    def _match_objects_by_mask_iou(self, pair: MeshPair, scene_transform: Optional[Dict]) -> Optional[List[Tuple[int, int]]]:
        render_config = pair.render_config
        mask_path = pair.gt_instance_mask_path
        inst_ids_full = pair.gt_inst_ids_full
        inst_ids_per_mesh = pair.gt_inst_ids_per_mesh
        if not (render_config and mask_path and inst_ids_full and inst_ids_per_mesh):
            return None
        if not pair.pred_meshes or not pair.gt_meshes:
            return None
        if len(inst_ids_per_mesh) != len(pair.gt_meshes):
            logger.warning(f'{pair.sample_id}: inst_ids_per_mesh length ({len(inst_ids_per_mesh)}) != gt_meshes ({len(pair.gt_meshes)})')
            return None
        intrinsic = np.array([[float(render_config['fl_x']), 0.0, float(render_config['cx'])], [0.0, float(render_config['fl_y']), float(render_config['cy'])], [0.0, 0.0, 1.0]])
        c2w_gl = np.asarray(render_config['cam_to_world'], dtype=np.float64)
        c2w_cv = c2w_gl @ np.diag([1.0, -1.0, -1.0, 1.0])
        width = int(render_config['width'])
        height = int(render_config['height'])
        gt_inst_masks = self._decode_palette_mask_per_inst(mask_path=mask_path, inst_ids_full=inst_ids_full, inst_ids_per_mesh=inst_ids_per_mesh, image_size=(width, height))
        if gt_inst_masks is None:
            return None
        return match_objects_by_mask_iou(pair.pred_meshes, gt_inst_masks, intrinsic=intrinsic, cam_to_world_opencv=c2w_cv, image_size=(width, height), scene_transform=scene_transform)

    @staticmethod
    def _decode_palette_mask_per_inst(*, mask_path: str, inst_ids_full: List[str], inst_ids_per_mesh: List[str], image_size: Tuple[int, int]) -> Optional[List[np.ndarray]]:
        from PIL import Image
        import colorsys
        try:
            mask_img = np.asarray(Image.open(mask_path).convert('RGB'), dtype=np.int32)
        except Exception as exc:
            logger.warning(f'Failed to load instance mask {mask_path}: {exc}')
            return None
        n = len(inst_ids_full)
        palette = np.zeros((n + 1, 3), dtype=np.int32)
        for i in range(1, n + 1):
            hue = (i - 1) / max(n, 1) % 1.0
            sat = 0.85 if i % 2 == 0 else 1.0
            val = 1.0 if i % 3 != 0 else 0.75
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            palette[i] = [round(r * 255), round(g * 255), round(b * 255)]
        flat = mask_img.reshape(-1, 3)
        diff = flat[:, None, :] - palette[None, :, :]
        dist2 = (diff ** 2).sum(axis=-1)
        decoded = dist2.argmin(axis=1).reshape(mask_img.shape[:2])
        inst_to_palette_idx = {iid: idx + 1 for idx, iid in enumerate(inst_ids_full)}
        per_inst_masks: List[np.ndarray] = []
        for iid in inst_ids_per_mesh:
            palette_idx = inst_to_palette_idx.get(str(iid))
            if palette_idx is None:
                logger.warning(f'inst_id {iid} not found in inst_ids_full while decoding mask {mask_path}')
                per_inst_masks.append(np.zeros(decoded.shape, dtype=bool))
                continue
            per_inst_masks.append(decoded == palette_idx)
        return per_inst_masks

    def _get_intermediate_path(self, sample_id: str, name: str, ext: str) -> str:
        sample_dir = os.path.join(self.output_dir, 'intermediates', sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        return os.path.join(sample_dir, f'{name}.{ext}')

    def _save_pointcloud(self, points: np.ndarray, sample_id: str, name: str):
        fmt = getattr(self.config, 'intermediate_pointcloud_format', 'ply')
        path = self._get_intermediate_path(sample_id, name, fmt)
        if fmt == 'ply':
            cloud = trimesh.PointCloud(points)
            cloud.export(path)
        elif fmt == 'npy':
            np.save(path, points)
        logger.debug(f'Saved pointcloud to {path}')

    def _save_mesh(self, meshes: List[trimesh.Trimesh], sample_id: str, name: str):
        path = self._get_intermediate_path(sample_id, name, 'glb')
        scene = trimesh.Scene()
        for mesh in meshes:
            scene.add_geometry(mesh)
        scene.export(path)
        logger.debug(f'Saved mesh to {path}')

    def _save_object_matches(self, pair: MeshPair, matches: List[Tuple[int, int]], match_mode: str) -> None:
        if not self.config.save_intermediates:
            return
        pred_objs = pair.pred_meshes
        gt_objs = pair.gt_meshes

        def _key_list(names_attr, count, fallback_prefix: str) -> List[str]:
            names = getattr(pair, names_attr, None) or []
            return [str(names[i]) if i < len(names) and names[i] is not None else f'{fallback_prefix}{i}' for i in range(count)]
        payload = {'match_mode': match_mode, 'matches': [[int(p), int(g)] for p, g in matches], 'pred_object_names': _key_list('pred_object_names', len(pred_objs), 'pred_'), 'gt_object_names': _key_list('gt_object_names', len(gt_objs), 'gt_'), 'pred_object_count': len(pred_objs), 'gt_object_count': len(gt_objs), 'scene_alignment_transform': _serialize_transform(pair.scene_alignment_transform)}
        path = self._get_intermediate_path(pair.sample_id, 'object_matches', 'json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        logger.debug(f'Saved object_matches.json to {path}')

    def _save_scene_meshes(self, pair: MeshPair) -> None:
        if not self.config.save_intermediates:
            return
        lazy = getattr(pair, '_lazy_paths', None) or {}
        for side, meshes, scene_key in [('gt', pair.gt_meshes, 'gt_scene_mesh_path'), ('pred', pair.pred_meshes, 'pred_scene_mesh_path')]:
            original_path = lazy.get(scene_key)
            if original_path and os.path.isfile(original_path):
                dst = self._get_intermediate_path(pair.sample_id, f'{side}_scene', 'glb')
                shutil.copy2(original_path, dst)
                logger.debug(f'Copied original scene mesh to {dst}')
            elif meshes:
                self._save_mesh(meshes, pair.sample_id, f'{side}_scene')

    @staticmethod
    def _compute_alignment_quality_stats(pred_points: np.ndarray, gt_points: np.ndarray) -> Dict[str, float]:
        from scipy.spatial import cKDTree
        tree = cKDTree(gt_points)
        distances, _ = tree.query(pred_points, k=1)
        return {'mean_dist': float(distances.mean()), 'median_dist': float(np.median(distances)), 'rmse_dist': float(np.sqrt(np.mean(np.square(distances))))}

    def _save_alignment_visualization(self, pred_points: np.ndarray, gt_points: np.ndarray, sample_id: str, suffix: str=''):
        combined_pts = np.vstack([pred_points, gt_points])
        center = combined_pts.mean(axis=0)
        pred_centered = pred_points - center
        gt_centered = gt_points - center
        colors_pred = np.tile([255, 0, 0, 255], (len(pred_centered), 1))
        colors_gt = np.tile([0, 0, 255, 255], (len(gt_centered), 1))
        combined_pts = np.vstack([pred_centered, gt_centered])
        combined_colors = np.vstack([colors_pred, colors_gt])
        combined_cloud = trimesh.PointCloud(combined_pts, colors=combined_colors)
        filename = f'alignment_visualization{suffix}'
        path = self._get_intermediate_path(sample_id, filename, 'ply')
        combined_cloud.export(path)
        stats = self._compute_alignment_quality_stats(pred_points, gt_points)
        logger.info(f"Alignment quality for {sample_id}{suffix}: mean_dist={stats['mean_dist']:.4f}, median={stats['median_dist']:.4f}")
        logger.debug(f'Saved alignment visualization to {path}')

    def _save_object_alignment_visualization(self, pred_points: np.ndarray, gt_points: np.ndarray, sample_id: str, object_name: str) -> None:
        import re
        obj_dir = os.path.join(self.output_dir, 'intermediates', sample_id, 'obj_alignment')
        os.makedirs(obj_dir, exist_ok=True)
        colors_pred = np.tile([255, 0, 0, 255], (len(pred_points), 1))
        colors_gt = np.tile([0, 0, 255, 255], (len(gt_points), 1))
        combined_cloud = trimesh.PointCloud(np.vstack([pred_points, gt_points]), colors=np.vstack([colors_pred, colors_gt]))
        safe_name = re.sub('[^\\w\\-.]', '_', object_name)
        path = os.path.join(obj_dir, f'{safe_name}.ply')
        combined_cloud.export(path)
        logger.debug(f'Saved object alignment visualization to {path}')

    @staticmethod
    def _generate_24_rotations() -> List[np.ndarray]:
        from itertools import permutations, product
        rotations = []
        for perm in permutations(range(3)):
            for signs in product([1, -1], repeat=3):
                R = np.zeros((3, 3))
                for i, (p, s) in enumerate(zip(perm, signs)):
                    R[i, p] = s
                if np.linalg.det(R) > 0:
                    rotations.append(R)
        return rotations

    def _find_best_90deg_rotation(self, pred_coarse: np.ndarray, gt: np.ndarray, subsample: int=3000) -> Tuple[np.ndarray, float]:
        from scipy.spatial import cKDTree
        gt_center = (gt.max(axis=0) + gt.min(axis=0)) / 2
        pred_centered = pred_coarse - gt_center
        rng = np.random.RandomState(42)
        if len(gt) > subsample:
            gt_sub = gt[rng.choice(len(gt), subsample, replace=False)]
        else:
            gt_sub = gt
        tree = cKDTree(gt_sub)
        threshold = np.linalg.norm(gt.max(axis=0) - gt.min(axis=0)) * 0.1
        best_R = np.eye(3)
        best_score = -1.0
        for R in self._generate_24_rotations():
            rotated = pred_centered @ R.T + gt_center
            if len(rotated) > subsample:
                rotated_sub = rotated[rng.choice(len(rotated), subsample, replace=False)]
            else:
                rotated_sub = rotated
            dists, _ = tree.query(rotated_sub)
            score = float((dists < threshold).sum()) / len(rotated_sub)
            if score > best_score:
                best_score = score
                best_R = R
        logger.debug(f'Rotation search: best_score={best_score:.4f}, R=\n{best_R}')
        return (best_R, best_score)

    def _merge_meshes_for_scene(self, meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
        if len(meshes) == 1:
            return meshes[0]
        return trimesh.util.concatenate(meshes)

    def _sample_points_from_mesh(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        points = surface_points(mesh, int(self.config.num_points))
        return torch.from_numpy(points).float()

    def _sample_n_points_from_mesh(self, mesh: trimesh.Trimesh, n: int) -> torch.Tensor:
        points = surface_points(mesh, n)
        return torch.from_numpy(points).float()

    def _sample_scene_points_from_object_meshes(self, meshes: List[trimesh.Trimesh]) -> torch.Tensor:
        all_points = []
        points_per_object = int(getattr(self.config, 'scene_points_per_object', 20480))
        for mesh in meshes:
            points = surface_points(mesh, points_per_object)
            all_points.append(points)
        combined = np.concatenate(all_points, axis=0)
        return torch.from_numpy(combined).float()

    def _aggregate_results(self, all_metrics: List[Dict]) -> Dict[str, float]:
        if not all_metrics:
            return {}
        numeric_keys = [k for k in all_metrics[0].keys() if k != 'sample_id' and (not str(k).startswith('__'))]
        result = {}
        for key in numeric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                result[f'{key}_mean'] = np.mean(values)
                result[f'{key}_std'] = np.std(values)
        return result
