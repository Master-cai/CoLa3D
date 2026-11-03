

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


GRID_RES = 64
SOR_K = 30
SOR_N_STD = 1.0
DEFAULT_MOGE2_NAME = "Ruicheng/moge-2-vitl-normal"


DEFAULT_COND_FOV_DEG = 40.0
DEFAULT_COND_IMAGE_SIZE = 1024


def load_moge2_model(
    model_name: str = DEFAULT_MOGE2_NAME, device: torch.device | str = "cuda"
):

    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(model_name)
    return model.to(device).eval()


@torch.no_grad()
def run_moge2_pointmap(
    model,
    rgb: torch.Tensor,
    device: torch.device | str,
) -> Tuple[np.ndarray, np.ndarray]:


    if rgb.dim() != 3 or rgb.shape[0] != 3:
        raise ValueError(f"run_moge2_pointmap expects (3,H,W), got {tuple(rgb.shape)}")
    output = model.infer(rgb.to(device))
    pointmap = output["points"].detach().cpu().float().numpy()
    valid_mask = output["mask"].detach().cpu().bool().numpy()
    return pointmap, valid_mask


def _statistical_outlier_inliers(
    points: np.ndarray, k: int = SOR_K, n_std: float = SOR_N_STD
) -> np.ndarray:


    n = len(points)
    if n <= k + 1:
        return np.ones(n, dtype=bool)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + n_std * mean_dists.std()
    return mean_dists < threshold


def axis_swap_camera_to_blender(points: np.ndarray) -> np.ndarray:

    return np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])


def fit_self_normalize(points: np.ndarray) -> Tuple[np.ndarray, float]:

    vmin = points.min(axis=0)
    vmax = points.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = float((vmax - vmin).max())
    if extent < 1e-6:
        extent = 1.0
    scale = 0.99999 / extent
    return center.astype(np.float64), scale


def normalize_to_voxels(
    points: np.ndarray, center: np.ndarray, scale: float
) -> np.ndarray:

    norm = (points - center) * scale
    coords = np.floor((norm + 0.5) * GRID_RES).astype(np.int64)
    return np.clip(coords, 0, GRID_RES - 1)


def points_to_voxel_coords(points: np.ndarray) -> np.ndarray:


    inliers = _statistical_outlier_inliers(points)
    pts = points[inliers]
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    swapped = axis_swap_camera_to_blender(pts)
    center, scale = fit_self_normalize(swapped)
    coords = normalize_to_voxels(swapped, center, scale)
    return np.unique(coords, axis=0)


def sample_interior_pixel(valid_mask_2d: np.ndarray) -> Optional[Tuple[int, int]]:


    if not valid_mask_2d.any():
        return None


    padded = np.pad(valid_mask_2d.astype(np.uint8), 1, constant_values=0)
    dt = distance_transform_edt(padded)[1:-1, 1:-1]
    flat = int(np.argmax(dt))
    y, x = np.unravel_index(flat, valid_mask_2d.shape)
    return int(y), int(x)


def _import_register_point_clouds():

    from trellis2.utils.ss_flow_eval_utils import register_point_clouds

    return register_point_clouds


def align_voxels_to_pred(
    moge_voxels: np.ndarray,
    pred_coords: np.ndarray,
    *,
    method: str = "fpfh_ransac",
) -> Tuple[np.ndarray, Dict[str, Any]]:

    info: Dict[str, Any] = {"used": False, "method": method, "error": None}
    if len(moge_voxels) == 0 or len(pred_coords) == 0:
        info["error"] = "empty input"
        return np.eye(4, dtype=np.float64), info
    register_point_clouds = _import_register_point_clouds()
    try:
        reg = register_point_clouds(
            moge_voxels.astype(np.float64),
            pred_coords.astype(np.float64),
            use_global_registration=True,
            global_method=method,
        )
    except Exception as e:
        info["error"] = f"register: {e}"
        return np.eye(4, dtype=np.float64), info
    info.update(
        used=True,
        mean_nn_before=float(reg.mean_nn_before),
        mean_nn_after_global=float(reg.mean_nn_after_global),
        mean_nn_after_icp=float(reg.mean_nn_after_icp),
        global_method_used=reg.global_method_used,
        global_fallback=bool(reg.global_fallback),
        icp_used=bool(reg.icp_used),
        icp_fallback=bool(reg.icp_fallback),
        icp_iterations=int(reg.icp_iterations),
        transform_norm=float(np.linalg.norm(reg.transform - np.eye(4))),
    )
    return reg.transform.astype(np.float64), info


def apply_T_to_voxel(voxel_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:

    if voxel_xyz.size == 0:
        return voxel_xyz.astype(np.int64)
    pts = voxel_xyz.astype(np.float64) + 0.5
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    transformed = (T @ pts_h.T).T[:, :3]
    coords = np.floor(transformed).astype(np.int64)
    return np.clip(coords, 0, GRID_RES - 1)


def apply_T_to_point_float(point_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:

    p_h = np.concatenate([point_xyz.astype(np.float64), [1.0]])
    return (T @ p_h)[:3]


def apply_T_rotation_to_dir(direction: np.ndarray, T: np.ndarray) -> np.ndarray:

    out = T[:3, :3] @ direction.astype(np.float64)
    n = np.linalg.norm(out)
    return out / n if n > 1e-12 else out


def compute_cond_bbox(mask_raw: np.ndarray) -> Tuple[int, int, int, int]:


    alpha = (mask_raw != 0) & (mask_raw != 255)
    nz = np.argwhere(alpha)
    if nz.size == 0:
        return (0, 0, mask_raw.shape[1], mask_raw.shape[0])
    ymin, xmin = nz.min(axis=0)
    ymax, xmax = nz.max(axis=0)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    hsize = max(xmax - xmin, ymax - ymin) / 2.0
    return (int(cx - hsize), int(cy - hsize), int(cx + hsize), int(cy + hsize))


def map_pixel_to_cond(
    pixel_yx: Tuple[int, int],
    resized_hw: Tuple[int, int],
    raw_hw: Tuple[int, int],
    cond_bbox_xyxy: Tuple[int, int, int, int],
    cond_image_size: int = DEFAULT_COND_IMAGE_SIZE,
) -> Tuple[float, float]:


    py_resized, px_resized = pixel_yx
    h_resized, w_resized = resized_hw
    h_raw, w_raw = raw_hw


    py_raw = (py_resized + 0.5) * (h_raw / h_resized) - 0.5
    px_raw = (px_resized + 0.5) * (w_raw / w_resized) - 0.5


    x_min, y_min, x_max, y_max = cond_bbox_xyxy
    bbox_size = x_max - x_min
    if bbox_size <= 0:
        return (cond_image_size / 2.0, cond_image_size / 2.0)
    py_bbox = py_raw - y_min
    px_bbox = px_raw - x_min


    py_cond = py_bbox * (cond_image_size / bbox_size)
    px_cond = px_bbox * (cond_image_size / bbox_size)
    return (py_cond, px_cond)


def pixel_to_camera_ray(
    pixel_yx_cond: Tuple[float, float],
    cond_image_size: int = DEFAULT_COND_IMAGE_SIZE,
    fov_deg: float = DEFAULT_COND_FOV_DEG,
) -> np.ndarray:


    py, px = pixel_yx_cond
    half = (cond_image_size - 1) / 2.0
    half_tan = math.tan(math.radians(fov_deg) / 2.0)
    dx = (px - half) / half * half_tan
    dy = (py - half) / half * half_tan
    cam_dir = np.array([dx, dy, 1.0], dtype=np.float64)
    return cam_dir / np.linalg.norm(cam_dir)


def axis_swap_dir(direction: np.ndarray) -> np.ndarray:

    return np.array([direction[0], direction[2], -direction[1]], dtype=np.float64)


def raycast_into_pred(
    *,
    cam_origin_canon: np.ndarray,
    ray_dir_canon: np.ndarray,
    pred_keys_sorted: np.ndarray,
    max_steps: int = 400,
    step_size: float = 0.5,
    tolerance: int = 0,
) -> Optional[np.ndarray]:


    offsets = _l_inf_offsets(int(max(0, tolerance)))
    last_center_key = -1
    pos = cam_origin_canon.astype(np.float64).copy()
    for _ in range(max_steps):
        center = np.floor(pos).astype(np.int64)


        if (center >= 0).all() and (center < GRID_RES).all():
            ckey = int(_encode_keys(center))
            if ckey != last_center_key:
                last_center_key = ckey
                if offsets.shape[0] == 1:
                    i = int(np.searchsorted(pred_keys_sorted, ckey))
                    if i < pred_keys_sorted.size and int(pred_keys_sorted[i]) == ckey:
                        return center
                else:
                    for off in offsets:
                        cand = center + off
                        if (cand < 0).any() or (cand >= GRID_RES).any():
                            continue
                        key = int(_encode_keys(cand))
                        i = int(np.searchsorted(pred_keys_sorted, key))
                        if (
                            i < pred_keys_sorted.size
                            and int(pred_keys_sorted[i]) == key
                        ):
                            return cand
        pos += ray_dir_canon * step_size
    return None


def _encode_keys(coords: np.ndarray) -> np.ndarray:

    return (
        coords[..., 0] * (GRID_RES * GRID_RES)
        + coords[..., 1] * GRID_RES
        + coords[..., 2]
    ).astype(np.int64)


def _l_inf_offsets(radius: int) -> np.ndarray:

    if radius <= 0:
        return np.zeros((1, 3), dtype=np.int64)
    side = np.arange(-radius, radius + 1, dtype=np.int64)
    grid = np.stack(np.meshgrid(side, side, side, indexing="ij"), axis=-1).reshape(
        -1, 3
    )

    linf = np.abs(grid).max(axis=1)
    l1 = np.abs(grid).sum(axis=1)
    order = np.lexsort((l1, linf))
    return grid[order]


def snap_to_pred(
    voxel: np.ndarray,
    pred_keys_sorted: np.ndarray,
    *,
    snap_radius: int = 2,
) -> Tuple[Optional[np.ndarray], int]:


    offsets = _l_inf_offsets(snap_radius)
    for off in offsets:
        cand = voxel + off
        if (cand < 0).any() or (cand >= GRID_RES).any():
            continue
        key = int(_encode_keys(cand))
        pos = int(np.searchsorted(pred_keys_sorted, key))
        if pos < pred_keys_sorted.size and pred_keys_sorted[pos] == key:
            return cand.astype(np.int64), int(np.abs(off).max())
    return None, snap_radius + 1


@dataclass
class PromptBuildResult:
    prompt_points: torch.Tensor
    valid_mask: torch.Tensor
    info: Dict[str, Any] = field(default_factory=dict)


def _to_pixel_grid(
    rgb: torch.Tensor,
    instance_mask: torch.Tensor,
    target_max_side: int,
) -> Tuple[torch.Tensor, np.ndarray]:


    _, h, w = rgb.shape
    if max(h, w) <= target_max_side:
        return rgb, instance_mask.numpy().astype(np.uint8)
    scale = target_max_side / float(max(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    rgb_resized = (
        torch.nn.functional.interpolate(
            rgb.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        .squeeze(0)
        .clamp_(0.0, 1.0)
    )
    mask_np = instance_mask.numpy().astype(np.uint8)

    mask_resized = (
        torch.nn.functional.interpolate(
            torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float(),
            size=(new_h, new_w),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .numpy()
        .astype(np.uint8)
    )
    return rgb_resized, mask_resized


def build_prompt_points_from_mask(
    *,
    raw_rgb: torch.Tensor,
    instance_mask: torch.Tensor,
    pred_coords: torch.Tensor,
    moge_model,
    device: torch.device | str,
    align: bool = True,
    align_method: str = "fpfh_ransac",
    snap_radius: int = 2,
    moge_input_max_side: int = 1024,
    max_parts: Optional[int] = None,
    pixel_mode: str = "hybrid",
    cond_image_size: int = DEFAULT_COND_IMAGE_SIZE,
    cond_fov_deg: float = DEFAULT_COND_FOV_DEG,
    raycast_max_steps: int = 400,
    raycast_step_size: float = 0.5,
    raycast_tolerance: int = 1,
) -> PromptBuildResult:


    if pred_coords.dim() != 2 or pred_coords.shape[1] != 3:
        raise ValueError(
            f"pred_coords expected (Pn, 3), got {tuple(pred_coords.shape)}"
        )
    if instance_mask.dim() != 2:
        raise ValueError(
            f"instance_mask expected (H, W), got {tuple(instance_mask.shape)}"
        )
    if pixel_mode not in ("moge_depth", "raycast", "hybrid"):
        raise ValueError(
            f"pixel_mode must be 'moge_depth', 'raycast', or 'hybrid', got {pixel_mode!r}"
        )

    pred_dev = pred_coords.device
    pred_np = pred_coords.detach().cpu().to(torch.int64).numpy()
    pred_keys_sorted = np.sort(_encode_keys(pred_np))


    rgb_in, mask_np = _to_pixel_grid(raw_rgb, instance_mask, moge_input_max_side)
    raw_mask_np = instance_mask.numpy().astype(np.uint8)
    cond_bbox = compute_cond_bbox(raw_mask_np)


    pointmap, moge_valid = run_moge2_pointmap(moge_model, rgb_in, device)
    H, W = mask_np.shape
    if pointmap.shape[:2] != (H, W):
        raise RuntimeError(
            f"MoGe2 pointmap {pointmap.shape[:2]} ≠ mask {(H, W)}; "
            "mask resize did not match MoGe2 input grid."
        )


    fg_2d = moge_valid & (mask_np != 0) & (mask_np != 255)
    finite_xyz = np.isfinite(pointmap).all(axis=-1)
    fg_2d &= finite_xyz

    info: Dict[str, Any] = {
        "strategy": "mask_moge2",
        "pixel_mode": pixel_mode,
        "num_objects": 0,
        "num_valid": 0,
        "dropped": [],
        "valid_obj_ids": [],
        "dilated_radius": [],
        "matched_per_obj": [],
        "collisions": 0,
        "fallback": "snap" if pixel_mode == "moge_depth" else "raycast_first_hit",
        "snap_radius": int(snap_radius),
        "cond_image_size": int(cond_image_size),
        "cond_fov_deg": float(cond_fov_deg),
        "align": {"used": False},
        "per_obj": [],
    }


    unique_ids = sorted(int(v) for v in np.unique(mask_np) if int(v) not in (0, 255))
    info["num_objects"] = len(unique_ids)

    if max_parts is not None and len(unique_ids) > max_parts:

        sizes = {oid: int((mask_np == oid).sum()) for oid in unique_ids}
        unique_ids = sorted(unique_ids, key=lambda o: -sizes[o])[:max_parts]
        info["max_parts_clipped"] = True

    if not fg_2d.any() or len(unique_ids) == 0:
        prompt_points = torch.zeros(1, 1, 3, dtype=torch.long, device=pred_dev)
        valid_mask = torch.zeros(1, 1, dtype=torch.bool, device=pred_dev)
        info["dropped"] = list(range(len(unique_ids)))
        info["dilated_radius"] = [0] * len(unique_ids)
        info["matched_per_obj"] = [0] * len(unique_ids)
        return PromptBuildResult(prompt_points, valid_mask, info)


    fg_pts = pointmap[fg_2d]
    inlier_flat = _statistical_outlier_inliers(fg_pts)
    alive_2d = np.zeros_like(fg_2d)
    alive_2d[fg_2d] = inlier_flat
    survivors = fg_pts[inlier_flat]

    if len(survivors) == 0:
        prompt_points = torch.zeros(1, 1, 3, dtype=torch.long, device=pred_dev)
        valid_mask = torch.zeros(1, 1, dtype=torch.bool, device=pred_dev)
        info["dropped"] = list(range(len(unique_ids)))
        info["dilated_radius"] = [0] * len(unique_ids)
        info["matched_per_obj"] = [0] * len(unique_ids)
        info["error"] = "all_sor_filtered"
        return PromptBuildResult(prompt_points, valid_mask, info)

    swapped_survivors = axis_swap_camera_to_blender(survivors)
    center, scale = fit_self_normalize(swapped_survivors)
    surv_voxels_pre = normalize_to_voxels(swapped_survivors, center, scale)
    moge_voxels = np.unique(surv_voxels_pre, axis=0)


    fg_yx = np.argwhere(fg_2d)
    survivor_yx = fg_yx[inlier_flat]
    survivor_oids = mask_np[survivor_yx[:, 0], survivor_yx[:, 1]]


    if align:
        T, align_info = align_voxels_to_pred(moge_voxels, pred_np, method=align_method)
        info["align"] = align_info
    else:
        T = np.eye(4, dtype=np.float64)
        info["align"] = {"used": False, "method": "identity"}


    surv_voxels_aligned = apply_T_to_voxel(surv_voxels_pre, T)
    pair_keys = survivor_oids.astype(np.int64) * (GRID_RES**3 + 1) + _encode_keys(
        surv_voxels_aligned
    )
    _, first_idx = np.unique(pair_keys, return_index=True)
    moge_aligned_per_obj: Dict[str, List[List[int]]] = {}
    for idx in first_idx:
        oid = int(survivor_oids[idx])
        v = surv_voxels_aligned[idx].tolist()
        moge_aligned_per_obj.setdefault(str(oid), []).append(
            [int(v[0]), int(v[1]), int(v[2])]
        )
    info["moge_aligned_per_obj"] = moge_aligned_per_obj


    raw_h, raw_w = raw_mask_np.shape
    res_h, res_w = mask_np.shape
    cam_origin_norm = (-center) * scale
    cam_origin_vox = (cam_origin_norm + 0.5) * GRID_RES
    cam_origin_canon = apply_T_to_point_float(cam_origin_vox, T)
    info["cam_origin_canon"] = [round(float(v), 3) for v in cam_origin_canon]


    chosen_voxels: List[np.ndarray] = []
    valid_obj_ids: List[int] = []
    chosen_keys: set = set()
    matched_per_obj: List[int] = []
    dropped: List[int] = []
    dilated_radius: List[int] = []
    collisions = 0
    per_obj: List[Dict[str, Any]] = []

    for oid in unique_ids:
        obj_alive_2d = (mask_np == oid) & alive_2d
        if not obj_alive_2d.any():
            dropped.append(oid)
            matched_per_obj.append(0)
            dilated_radius.append(0)
            per_obj.append({"oid": oid, "reason": "no_alive_pixel"})
            continue
        sample = sample_interior_pixel(obj_alive_2d)
        if sample is None:
            dropped.append(oid)
            matched_per_obj.append(0)
            dilated_radius.append(0)
            per_obj.append({"oid": oid, "reason": "empty_after_filter"})
            continue
        y, x = sample


        cam_xyz = pointmap[y, x]
        swapped = axis_swap_camera_to_blender(cam_xyz[None])[0]
        voxel_pre = normalize_to_voxels(swapped[None], center, scale)[0]
        voxel_aligned = apply_T_to_voxel(voxel_pre[None], T)[0]
        moge_snapped, moge_snap_dist = snap_to_pred(
            voxel_aligned,
            pred_keys_sorted,
            snap_radius=snap_radius,
        )

        py_cond, px_cond = map_pixel_to_cond(
            pixel_yx=(y, x),
            resized_hw=(res_h, res_w),
            raw_hw=(raw_h, raw_w),
            cond_bbox_xyxy=cond_bbox,
            cond_image_size=cond_image_size,
        )
        cam_dir = pixel_to_camera_ray(
            pixel_yx_cond=(py_cond, px_cond),
            cond_image_size=cond_image_size,
            fov_deg=cond_fov_deg,
        )
        blender_dir = axis_swap_dir(cam_dir)
        ray_dir_canon = apply_T_rotation_to_dir(blender_dir, T)
        ray_hit = raycast_into_pred(
            cam_origin_canon=cam_origin_canon,
            ray_dir_canon=ray_dir_canon,
            pred_keys_sorted=pred_keys_sorted,
            max_steps=raycast_max_steps,
            step_size=raycast_step_size,
            tolerance=raycast_tolerance,
        )

        if pixel_mode == "moge_depth":
            primary_snapped = moge_snapped
            primary_snap_dist = moge_snap_dist if moge_snapped is not None else 0
            primary_source = "moge_snap"
            fail_reason = "no_pred_within_snap_radius"
        elif pixel_mode == "raycast":
            primary_snapped = ray_hit
            primary_snap_dist = 0
            primary_source = "raycast"
            fail_reason = "raycast_no_hit"
        else:

            if moge_snapped is not None:
                primary_snapped = moge_snapped
                primary_snap_dist = moge_snap_dist
                primary_source = "moge_snap"
            elif ray_hit is not None:
                primary_snapped = ray_hit
                primary_snap_dist = 0
                primary_source = "raycast_fallback"
            else:
                primary_snapped = None
                primary_snap_dist = 0
                primary_source = "neither"
            fail_reason = "moge_snap_and_raycast_both_failed"

        if primary_snapped is None:
            dropped.append(oid)
            matched_per_obj.append(0)
            dilated_radius.append(0)
            per_obj.append(
                {
                    "oid": oid,
                    "pixel": (y, x),
                    "voxel_pre_align": voxel_pre.tolist(),
                    "voxel_aligned": voxel_aligned.tolist(),
                    "pixel_cond": (round(py_cond, 2), round(px_cond, 2)),
                    "ray_dir_canon": ray_dir_canon.round(4).tolist(),
                    "moge_snapped": (
                        moge_snapped.tolist() if moge_snapped is not None else None
                    ),
                    "moge_snap_dist": (
                        int(moge_snap_dist) if moge_snapped is not None else None
                    ),
                    "ray_hit": ray_hit.tolist() if ray_hit is not None else None,
                    "reason": fail_reason,
                }
            )
            continue
        snapped = primary_snapped
        snap_dist = primary_snap_dist

        key = int(_encode_keys(snapped))
        if key in chosen_keys:
            collisions += 1

            dropped.append(oid)
            matched_per_obj.append(0)
            dilated_radius.append(snap_dist)
            per_obj.append(
                {
                    "oid": oid,
                    "pixel": (y, x),
                    "snapped_voxel": snapped.tolist(),
                    "reason": "collision_with_earlier_object",
                }
            )
            continue
        chosen_keys.add(key)
        chosen_voxels.append(snapped)
        valid_obj_ids.append(oid)
        matched_per_obj.append(1)
        dilated_radius.append(snap_dist)
        per_obj.append(
            {
                "oid": oid,
                "pixel": (y, x),
                "snapped_voxel": snapped.tolist(),
                "snap_dist": snap_dist,
                "source": primary_source,


                "voxel_pre_align": voxel_pre.tolist(),
                "voxel_aligned": voxel_aligned.tolist(),
                "pixel_cond": (round(py_cond, 2), round(px_cond, 2)),
                "ray_dir_canon": ray_dir_canon.round(4).tolist(),
                "moge_snapped": (
                    moge_snapped.tolist() if moge_snapped is not None else None
                ),
                "moge_snap_dist": (
                    int(moge_snap_dist) if moge_snapped is not None else None
                ),
                "ray_hit": ray_hit.tolist() if ray_hit is not None else None,
            }
        )

    info["valid_obj_ids"] = valid_obj_ids
    info["dropped"] = dropped
    info["dilated_radius"] = dilated_radius
    info["matched_per_obj"] = matched_per_obj
    info["collisions"] = collisions
    info["per_obj"] = per_obj
    info["num_valid"] = len(valid_obj_ids)

    if len(chosen_voxels) == 0:
        prompt_points = torch.zeros(1, 1, 3, dtype=torch.long, device=pred_dev)
        valid_mask = torch.zeros(1, 1, dtype=torch.bool, device=pred_dev)
        return PromptBuildResult(prompt_points, valid_mask, info)

    pp = torch.from_numpy(np.stack(chosen_voxels, axis=0)).long().to(pred_dev)
    prompt_points = pp.unsqueeze(0)
    valid_mask = torch.ones(
        1, prompt_points.shape[1], dtype=torch.bool, device=pred_dev
    )


    chosen_lin = _encode_keys(prompt_points[0].cpu().numpy())
    hit_pos = np.searchsorted(pred_keys_sorted, chosen_lin)
    in_range = hit_pos < pred_keys_sorted.size
    clamped = np.clip(hit_pos, 0, pred_keys_sorted.size - 1)
    ok = in_range & (pred_keys_sorted[clamped] == chosen_lin)
    if not bool(ok.all()):
        miss = int((~ok).sum())
        raise AssertionError(
            f"build_prompt_points_from_mask self-check failed: "
            f"{miss} chosen voxel(s) not in pred coords (snap logic bug)"
        )

    return PromptBuildResult(prompt_points, valid_mask, info)
