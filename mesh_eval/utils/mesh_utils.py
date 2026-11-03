import numpy as np
import trimesh
from typing import Dict, List, Optional, Tuple


def transform_centroids(centroids: np.ndarray, transform: Dict[str, np.ndarray]) -> np.ndarray:
    v = (centroids - transform['src_center']) * transform['coarse_scale']
    v = v @ transform['coarse_R'].T + transform['dst_center']
    if 'global_R' in transform:
        v = (transform['global_R'] @ v.T).T + transform['global_t']
    v = (transform['fine_R'] @ v.T).T + transform['fine_t']
    return v

def match_objects_by_centroid(pred_meshes: List[trimesh.Trimesh], gt_meshes: List[trimesh.Trimesh], scene_transform: Optional[Dict]=None) -> List[Tuple[int, int]]:
    from scipy.optimize import linear_sum_assignment
    pred_centroids = np.array([m.centroid for m in pred_meshes])
    gt_centroids = np.array([m.centroid for m in gt_meshes])
    if scene_transform is not None:
        pred_centroids = transform_centroids(pred_centroids, scene_transform)
    cost = np.linalg.norm(pred_centroids[:, None, :] - gt_centroids[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(zip(row_ind.tolist(), col_ind.tolist()))

def transform_points(points: np.ndarray, transform: Dict[str, np.ndarray]) -> np.ndarray:
    return transform_centroids(points, transform)

def match_objects_by_mask_iou(pred_meshes: List[trimesh.Trimesh], gt_inst_masks: List[np.ndarray], *, intrinsic: np.ndarray, cam_to_world_opencv: np.ndarray, image_size: Tuple[int, int], scene_transform: Optional[Dict]=None, surface_samples: int=100000, rng_seed: int=42) -> List[Tuple[int, int]]:
    from scipy.optimize import linear_sum_assignment
    width, height = (int(image_size[0]), int(image_size[1]))
    K = np.asarray(intrinsic, dtype=np.float64)
    w2c = np.linalg.inv(np.asarray(cam_to_world_opencv, dtype=np.float64))
    pred_silhouettes: List[np.ndarray] = []
    rng = np.random.default_rng(rng_seed)
    for mesh in pred_meshes:
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, surface_samples, seed=int(rng.integers(0, 2 ** 31 - 1)))
        except Exception:
            pts = np.asarray(mesh.vertices, dtype=np.float64)
        pts = np.asarray(pts, dtype=np.float64)
        if scene_transform is not None and len(pts) > 0:
            pts = transform_points(pts, scene_transform)
        pred_silhouettes.append(_project_points_to_silhouette(pts, K, w2c, width, height))
    cost = np.zeros((len(pred_meshes), len(gt_inst_masks)), dtype=np.float64)
    for i, sil in enumerate(pred_silhouettes):
        for j, gt_mask in enumerate(gt_inst_masks):
            cost[i, j] = -_mask_iou(sil, gt_mask)
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(zip(row_ind.tolist(), col_ind.tolist()))

def _project_points_to_silhouette(points_world: np.ndarray, K: np.ndarray, w2c: np.ndarray, width: int, height: int) -> np.ndarray:
    silhouette = np.zeros((height, width), dtype=bool)
    if len(points_world) == 0:
        return silhouette
    pts_h = np.hstack([points_world, np.ones((len(points_world), 1))])
    cam = (w2c @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    in_front = z > 1e-06
    if not np.any(in_front):
        return silhouette
    cam = cam[in_front]
    pix = (K @ cam.T).T
    pix = pix[:, :2] / pix[:, 2:3]
    u = np.round(pix[:, 0]).astype(np.int64)
    v = np.round(pix[:, 1]).astype(np.int64)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    silhouette[v[valid], u[valid]] = True
    if silhouette.any():
        padded = np.pad(silhouette, 1, mode='constant', constant_values=False)
        dilated = padded[0:-2, 0:-2] | padded[0:-2, 1:-1] | padded[0:-2, 2:] | padded[1:-1, 0:-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:] | padded[2:, 0:-2] | padded[2:, 1:-1] | padded[2:, 2:]
        silhouette = dilated
    return silhouette

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0

def match_objects_by_name(pred_names: List[str], gt_names: List[str]) -> List[Tuple[int, int]]:
    gt_name_to_idx = {name: idx for idx, name in enumerate(gt_names)}
    matches = []
    for pred_idx, name in enumerate(pred_names):
        if name in gt_name_to_idx:
            matches.append((pred_idx, gt_name_to_idx[name]))
    return matches
