import torch
import numpy as np
import logging
from typing import Optional
logger = logging.getLogger(__name__)

def _compute_kabsch_rotation_from_cov(H: torch.Tensor) -> torch.Tensor:
    U, _, V = torch.svd(H)
    R = torch.matmul(V, U.transpose(1, 2))
    det_R = torch.det(R)
    if torch.any(det_R < 0):
        V_fixed = V.clone()
        V_fixed[det_R < 0, :, -1] *= -1.0
        R = torch.matmul(V_fixed, U.transpose(1, 2))
    return R

def align_point_clouds(src: torch.Tensor, dst: torch.Tensor, coarse_method: str='bbox', global_method: str='none', fine_method: str='icp', coarse_pca: bool=True, max_iterations: int=50, tolerance: float=0.0001, global_voxel_size_ratio: float=0.02, global_ransac_n: int=3, global_ransac_max_iter: int=100000, fine_max_points: Optional[int]=None) -> tuple[torch.Tensor, dict]:
    B, N, _ = src.shape
    device = src.device
    transformed_src = src.clone()
    transform_info = {'coarse_scale': [], 'coarse_R': [], 'fine_R': [], 'fine_t': []}
    for b in range(B):
        src_np = src[b].detach().cpu().numpy().astype(np.float64)
        dst_np = dst[b].detach().cpu().numpy().astype(np.float64)
        if coarse_method == 'bbox':
            src_coarse, coarse_scale, coarse_R = _coarse_align_bbox(src_np, dst_np, use_pca=coarse_pca)
        else:
            src_coarse = src_np
            coarse_scale = 1.0
            coarse_R = np.eye(3)
        transform_info['coarse_scale'].append(coarse_scale)
        transform_info['coarse_R'].append(coarse_R)
        if global_method == 'fpfh_ransac':
            src_global, global_R, global_t = _global_align_fpfh_ransac(src_coarse, dst_np, voxel_size_ratio=global_voxel_size_ratio, ransac_n=global_ransac_n, max_iteration=global_ransac_max_iter)
        elif global_method == 'teaser':
            src_global, global_R, global_t = _global_align_fpfh_teaser(src_coarse, dst_np, voxel_size_ratio=global_voxel_size_ratio)
        else:
            src_global = src_coarse
            global_R = np.eye(3)
            global_t = np.zeros(3)
        transform_info.setdefault('global_R', []).append(global_R)
        transform_info.setdefault('global_t', []).append(global_t)
        if fine_method == 'icp':
            src_fine, R, t = _fine_align_icp(src_global, dst_np, max_iterations, tolerance, max_points=fine_max_points)
        elif fine_method == 'filterreg':
            src_fine, R, t = _fine_align_filterreg(src_global, dst_np, max_iterations, tolerance, max_points=fine_max_points)
        elif fine_method == 'cpd':
            src_fine, R, t = _fine_align_cpd(src_global, dst_np, max_iterations, tolerance, max_points=fine_max_points)
        else:
            src_fine = src_global
            R = np.eye(3)
            t = np.zeros(3)
        transform_info['fine_R'].append(R)
        transform_info['fine_t'].append(t)
        transformed_src[b] = torch.tensor(src_fine, dtype=torch.float32, device=device)
    return (transformed_src, transform_info)

def _coarse_align_bbox(src: np.ndarray, dst: np.ndarray, use_pca: bool=True):
    src_min, src_max = (src.min(axis=0), src.max(axis=0))
    dst_min, dst_max = (dst.min(axis=0), dst.max(axis=0))
    src_diag = np.linalg.norm(src_max - src_min)
    dst_diag = np.linalg.norm(dst_max - dst_min)
    if src_diag > 1e-08:
        scale = dst_diag / src_diag
    else:
        scale = 1.0
    src_center = (src_max + src_min) / 2
    dst_center = (dst_max + dst_min) / 2
    src_centered = (src - src_center) * scale
    dst_centered = dst - dst_center
    if not use_pca:
        src_aligned = src_centered + dst_center
        logger.debug(f'Coarse align: scale={scale:.4f}, pca=off')
        return (src_aligned, scale, np.eye(3))
    src_axes = _compute_pca_axes(src_centered)
    dst_axes = _compute_pca_axes(dst_centered)
    best_rotation = np.eye(3)
    best_score = -1
    for R_candidate in _generate_rotation_candidates(src_axes, dst_axes):
        src_rotated = src_centered @ R_candidate.T
        score = _compute_inlier_ratio(src_rotated, dst_centered, threshold_ratio=0.1)
        if score > best_score:
            best_score = score
            best_rotation = R_candidate
    src_aligned = src_centered @ best_rotation.T + dst_center
    logger.debug(f'Coarse align: scale={scale:.4f}, best_inlier_ratio={best_score:.4f}')
    return (src_aligned, scale, best_rotation)

def _compute_pca_axes(points: np.ndarray) -> np.ndarray:
    cov = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, idx].T
    return axes

def _generate_rotation_candidates(src_axes: np.ndarray, dst_axes: np.ndarray):
    from itertools import permutations, product
    candidates = []
    for perm in permutations([0, 1, 2]):
        for signs in product([1, -1], repeat=3):
            dst_permuted = np.array([dst_axes[p] * s for p, s in zip(perm, signs)])
            R = src_axes.T @ dst_permuted
            if np.abs(np.linalg.det(R) - 1.0) < 0.1:
                candidates.append(R)
    return candidates

def _compute_inlier_ratio(src: np.ndarray, dst: np.ndarray, threshold_ratio: float=0.1) -> float:
    from scipy.spatial import cKDTree
    dst_range = dst.max(axis=0) - dst.min(axis=0)
    threshold = np.linalg.norm(dst_range) * threshold_ratio
    tree = cKDTree(dst)
    distances, _ = tree.query(src, k=1)
    inliers = distances < threshold
    inlier_ratio = inliers.sum() / len(src)
    return inlier_ratio

def _uniform_subsample(points: np.ndarray, max_points: Optional[int]) -> np.ndarray:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points
    step = int(np.ceil(len(points) / max_points))
    return points[::step]

def _fine_align_icp(src: np.ndarray, dst: np.ndarray, max_iterations: int, tolerance: float, max_points: Optional[int]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import open3d as o3d
    src_est = _uniform_subsample(src, max_points)
    dst_est = _uniform_subsample(dst, max_points)
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_est)
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(dst_est)
    init_transform = np.eye(4)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations, relative_fitness=tolerance, relative_rmse=tolerance)
    src_range = src_est.max(axis=0) - src_est.min(axis=0)
    max_dist = np.linalg.norm(src_range) * 0.3
    result = o3d.pipelines.registration.registration_icp(pcd_src, pcd_dst, max_correspondence_distance=max_dist, init=init_transform, estimation_method=estimation, criteria=criteria)
    T = result.transformation
    src_homo = np.hstack([src, np.ones((len(src), 1))])
    src_aligned = (T @ src_homo.T).T[:, :3]
    R = T[:3, :3]
    t = T[:3, 3]
    logger.debug(f'ICP: fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.4f}')
    return (src_aligned, R, t)

def _fine_align_filterreg(src: np.ndarray, dst: np.ndarray, max_iterations: int, tolerance: float, max_points: Optional[int]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from probreg import filterreg
    src_est = _uniform_subsample(src, max_points)
    dst_est = _uniform_subsample(dst, max_points)
    tf_param, _, _ = filterreg.registration_filterreg(src_est, dst_est, objective_type='pt2pt', maxiter=max_iterations, tol=tolerance, sigma2=None, update_sigma2=True)
    R = tf_param.rot
    t = tf_param.t
    src_aligned = src @ R.T + t
    return (src_aligned, R, t)

def _fine_align_cpd(src: np.ndarray, dst: np.ndarray, max_iterations: int, tolerance: float, max_points: Optional[int]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from probreg import cpd as probreg_cpd
    cpd_max_points = 5000
    if max_points is not None and 0 < max_points < cpd_max_points:
        cpd_max_points = max_points
    if len(src) > cpd_max_points:
        idx = np.random.choice(len(src), size=cpd_max_points, replace=False)
        src_sample = src[idx]
    else:
        src_sample = src
    if len(dst) > cpd_max_points:
        idx = np.random.choice(len(dst), size=cpd_max_points, replace=False)
        dst_sample = dst[idx]
    else:
        dst_sample = dst
    cpd_solver = probreg_cpd.RigidCPD(source=src_sample.astype(np.float32), use_cuda=False)
    tf_param, _, _ = cpd_solver.registration(target=dst_sample.astype(np.float32), maxiter=max_iterations, tol=tolerance)
    R = tf_param.rot
    t = tf_param.t
    s = tf_param.scale
    src_aligned = src @ R.T * s + t
    return (src_aligned, R, t)

def _compute_fpfh_features(pcd, voxel_size: float):
    import open3d as o3d
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return (pcd_down, fpfh)

def _global_align_fpfh_ransac(src: np.ndarray, dst: np.ndarray, voxel_size_ratio: float=0.02, ransac_n: int=3, max_iteration: int=100000):
    import open3d as o3d
    dst_range = dst.max(axis=0) - dst.min(axis=0)
    voxel_size = np.linalg.norm(dst_range) * voxel_size_ratio
    if voxel_size < 1e-06:
        logger.warning('Global align: voxel_size too small (%.6f), skipping', voxel_size)
        return (src.copy(), np.eye(3), np.zeros(3))
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src)
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(dst)
    src_down, src_fpfh = _compute_fpfh_features(pcd_src, voxel_size)
    dst_down, dst_fpfh = _compute_fpfh_features(pcd_dst, voxel_size)
    max_corr_dist = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(src_down, dst_down, src_fpfh, dst_fpfh, mutual_filter=True, max_correspondence_distance=max_corr_dist, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n=ransac_n, checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr_dist)], criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, 0.999))
    T = result.transformation
    R = T[:3, :3]
    t = T[:3, 3]
    src_homo = np.hstack([src, np.ones((len(src), 1))])
    src_aligned = (T @ src_homo.T).T[:, :3]
    logger.debug('FPFH+RANSAC global: fitness=%.4f, rmse=%.4f, correspondences=%d', result.fitness, result.inlier_rmse, len(result.correspondence_set))
    return (src_aligned, R, t)

def _global_align_fpfh_teaser(src: np.ndarray, dst: np.ndarray, voxel_size_ratio: float=0.02, noise_bound: float=None):
    import open3d as o3d
    try:
        import teaserpp_python
    except ImportError as e:
        raise ImportError('TEASER++ registration requires teaserpp-python. Install with: pip install teaserpp-python') from e
    dst_range = dst.max(axis=0) - dst.min(axis=0)
    voxel_size = np.linalg.norm(dst_range) * voxel_size_ratio
    if voxel_size < 1e-06:
        logger.warning('Global align: voxel_size too small (%.6f), skipping', voxel_size)
        return (src.copy(), np.eye(3), np.zeros(3))
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src)
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(dst)
    src_down, src_fpfh = _compute_fpfh_features(pcd_src, voxel_size)
    dst_down, dst_fpfh = _compute_fpfh_features(pcd_dst, voxel_size)
    src_feat = np.asarray(src_fpfh.data).T
    dst_feat = np.asarray(dst_fpfh.data).T
    from scipy.spatial import cKDTree
    tree = cKDTree(dst_feat)
    _, indices = tree.query(src_feat, k=1)
    src_pts = np.asarray(src_down.points)
    dst_pts = np.asarray(dst_down.points)
    corr_src = src_pts
    corr_dst = dst_pts[indices.flatten()]
    if noise_bound is None:
        noise_bound = voxel_size * 1.5
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(corr_src.T, corr_dst.T)
    solution = solver.getSolution()
    R = solution.rotation
    t = solution.translation
    src_aligned = src @ R.T + t
    logger.debug('FPFH+TEASER++ global: %d correspondences, noise_bound=%.4f', len(corr_src), noise_bound)
    return (src_aligned, R, t)

def point_alignment(src, dst, max_iterations=50, tolerance=0.0001, mode='icp', **kwargs):
    return align_point_clouds(src, dst, coarse_method='bbox', global_method=kwargs.get('global_method', 'none'), fine_method=mode if mode != 'icp_scale' else 'icp', max_iterations=max_iterations, tolerance=tolerance, global_voxel_size_ratio=kwargs.get('global_voxel_size_ratio', 0.02), global_ransac_n=kwargs.get('global_ransac_n', 3), global_ransac_max_iter=kwargs.get('global_ransac_max_iter', 100000))
