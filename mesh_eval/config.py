
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class GeometryConfig:
    alignment_coarse_pca: bool = False
    alignment_coarse_rotation_search: bool = False
    alignment_fallback_good_mean_dist: float = 0.1
    alignment_fallback_min_worse_delta: float = 0.0
    alignment_fallback_to_coarse: bool = True
    alignment_global_method: str = 'fpfh_ransac'
    alignment_global_ransac_max_iter: int = 100000
    alignment_global_ransac_n: int = 3
    alignment_global_voxel_size_ratio: float = 0.02
    alignment_max_iterations: int = 200
    alignment_max_points: int | None = None
    alignment_method: str = 'filterreg'
    alignment_object: bool = True
    alignment_object_init_from_scene: bool = True
    alignment_scene: bool = True
    alignment_tolerance: float = 0.001
    chunk_size: int = 10240
    fscore_threshold: float = 0.1
    granularity_object: bool = True
    granularity_scene: bool = True
    intermediate_pointcloud_format: str = 'ply'
    metrics_2d_normal: bool = False
    metrics_2d_render: bool = False
    metrics_3d_geom: bool = True
    num_points: int = 20480
    object_iou_before_normalization: bool = False
    object_normalize_uniform: bool = True
    save_intermediates: bool = True
    save_object_alignment: bool = False
    scene_joint_normalize: bool = True
    scene_points_per_object: int = 20480
    scene_sampling_strategy: str = 'per_object_concat'

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> GeometryConfig:
        config = cls(**values)
        if not config.metrics_3d_geom or config.metrics_2d_render or config.metrics_2d_normal:
            raise ValueError('This evaluator supports the CoLa3D geometry recipe only.')
        return config
