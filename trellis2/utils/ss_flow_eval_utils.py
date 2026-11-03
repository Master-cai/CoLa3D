

from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Dict
import numpy as np
from scipy.spatial import cKDTree

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegistrationResult:
    aligned_points: np.ndarray
    transform: np.ndarray
    coarse_scale: float
    coarse_src_center: np.ndarray
    coarse_dst_center: np.ndarray
    global_method_used: str
    global_fallback: bool
    icp_used: bool
    icp_fallback: bool
    icp_iterations: int
    mean_nn_before: float
    mean_nn_after_global: float
    mean_nn_after_icp: float


def compute_voxel_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:

    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    pred_sum = pred_bool.sum()
    gt_sum = gt_bool.sum()
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    recall = intersection / gt_sum if gt_sum > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def mean_nearest_neighbor_distance(src: np.ndarray, dst: np.ndarray) -> float:
    if len(src) == 0 or len(dst) == 0:
        return float("inf")
    tree = cKDTree(dst)
    distances, _ = tree.query(src, k=1)
    return float(np.mean(distances))


def make_homogeneous_transform(
    linear: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    transform[:3, 3] = translation
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.astype(np.float32)
    points_h = np.concatenate(
        [points, np.ones((len(points), 1), dtype=points.dtype)], axis=1
    )
    aligned = (transform @ points_h.T).T[:, :3]
    return aligned.astype(np.float32)


def coarse_align_bbox(
    src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:

    src_min = src.min(axis=0)
    src_max = src.max(axis=0)
    dst_min = dst.min(axis=0)
    dst_max = dst.max(axis=0)
    src_diag = np.linalg.norm(src_max - src_min)
    dst_diag = np.linalg.norm(dst_max - dst_min)
    scale = float(dst_diag / src_diag) if src_diag > 1e-08 else 1.0
    src_center = (src_max + src_min) / 2.0
    dst_center = (dst_max + dst_min) / 2.0
    linear = np.eye(3, dtype=np.float64) * scale
    translation = dst_center - linear @ src_center
    transform = make_homogeneous_transform(linear, translation)
    aligned = apply_transform(src, transform)
    return (
        aligned,
        transform,
        scale,
        src_center.astype(np.float64),
        dst_center.astype(np.float64),
    )


def _compute_fpfh_features(pcd, voxel_size: float):
    import open3d as o3d

    pcd_down = pcd.voxel_down_sample(voxel_size)
    if len(pcd_down.points) == 0:
        return (pcd_down, None)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return (pcd_down, fpfh)


def global_register_fpfh_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    voxel_size_ratio: float = 0.02,
    ransac_n: int = 3,
    max_iteration: int = 100000,
) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    dst_range = dst.max(axis=0) - dst.min(axis=0)
    voxel_size = float(np.linalg.norm(dst_range) * voxel_size_ratio)
    if voxel_size < 1e-06:
        raise RuntimeError("voxel_size too small for global registration")
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src.astype(np.float64))
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(dst.astype(np.float64))
    src_down, src_fpfh = _compute_fpfh_features(pcd_src, voxel_size)
    dst_down, dst_fpfh = _compute_fpfh_features(pcd_dst, voxel_size)
    if src_fpfh is None or dst_fpfh is None:
        raise RuntimeError("empty downsampled cloud for global registration")
    max_corr_dist = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        dst_down,
        src_fpfh,
        dst_fpfh,
        mutual_filter=True,
        max_correspondence_distance=max_corr_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                max_corr_dist
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration, 0.999
        ),
    )
    transform = np.asarray(result.transformation, dtype=np.float64)
    aligned = apply_transform(src, transform)
    return (aligned, transform)


def fine_align_icp(
    src: np.ndarray,
    dst: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray, int]:
    import open3d as o3d

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src.astype(np.float64))
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(dst.astype(np.float64))
    src_range = src.max(axis=0) - src.min(axis=0)
    max_corr = float(np.linalg.norm(src_range) * 0.3)
    if max_corr < 1e-06:
        max_corr = 1.0
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations,
        relative_fitness=tolerance,
        relative_rmse=tolerance,
    )
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=False
    )
    result = o3d.pipelines.registration.registration_icp(
        pcd_src,
        pcd_dst,
        max_correspondence_distance=max_corr,
        init=np.eye(4),
        estimation_method=estimation,
        criteria=criteria,
    )
    transform = np.asarray(result.transformation, dtype=np.float64)
    aligned = apply_transform(src, transform)
    return (aligned, transform, int(max_iterations))


def register_point_clouds(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    use_global_registration: bool = True,
    global_method: str = "fpfh_ransac",
    voxel_size_ratio: float = 0.02,
    icp_max_iterations: int = 50,
    icp_tolerance: float = 0.0001,
) -> RegistrationResult:

    if len(src) == 0 or len(dst) == 0:
        identity = np.eye(4, dtype=np.float64)
        return RegistrationResult(
            aligned_points=src.astype(np.float32),
            transform=identity,
            coarse_scale=1.0,
            coarse_src_center=np.zeros(3, dtype=np.float64),
            coarse_dst_center=np.zeros(3, dtype=np.float64),
            global_method_used="none_empty",
            global_fallback=True,
            icp_used=False,
            icp_fallback=True,
            icp_iterations=0,
            mean_nn_before=float("inf"),
            mean_nn_after_global=float("inf"),
            mean_nn_after_icp=float("inf"),
        )
    coarse_points, coarse_transform, scale, src_center, dst_center = coarse_align_bbox(
        src, dst
    )
    mean_before = mean_nearest_neighbor_distance(coarse_points, dst)
    current_points = coarse_points
    current_transform = coarse_transform
    global_method_used = "none"
    global_fallback = False
    if use_global_registration:
        global_fn = global_register_fpfh_ransac
        try:
            global_points, global_transform = global_fn(
                current_points, dst, voxel_size_ratio=voxel_size_ratio
            )
            mean_after_global = mean_nearest_neighbor_distance(global_points, dst)
            if mean_after_global <= mean_before:
                current_points = global_points
                current_transform = global_transform @ current_transform
                global_method_used = global_method
            else:
                mean_after_global = mean_before
                global_method_used = f"{global_method}_reverted"
                global_fallback = True
        except Exception as exc:
            LOGGER.warning("Global registration fallback (%s): %s", global_method, exc)
            mean_after_global = mean_before
            global_method_used = f"{global_method}_failed"
            global_fallback = True
    else:
        mean_after_global = mean_before
    icp_used = False
    icp_fallback = False
    icp_iterations = 0
    try:
        icp_points, icp_transform, icp_iterations = fine_align_icp(
            current_points,
            dst,
            max_iterations=icp_max_iterations,
            tolerance=icp_tolerance,
        )
        mean_after_icp = mean_nearest_neighbor_distance(icp_points, dst)
        if mean_after_icp <= mean_after_global:
            current_points = icp_points
            current_transform = icp_transform @ current_transform
            icp_used = True
        else:
            mean_after_icp = mean_after_global
            icp_fallback = True
    except Exception as exc:
        LOGGER.warning("ICP fallback: %s", exc)
        mean_after_icp = mean_after_global
        icp_fallback = True
    return RegistrationResult(
        aligned_points=current_points.astype(np.float32),
        transform=current_transform.astype(np.float64),
        coarse_scale=scale,
        coarse_src_center=src_center,
        coarse_dst_center=dst_center,
        global_method_used=global_method_used,
        global_fallback=global_fallback,
        icp_used=icp_used,
        icp_fallback=icp_fallback,
        icp_iterations=icp_iterations,
        mean_nn_before=mean_before,
        mean_nn_after_global=mean_after_global,
        mean_nn_after_icp=mean_after_icp,
    )
