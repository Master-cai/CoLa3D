from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import trimesh

@dataclass
class MeshPair:
    pred_meshes: List[trimesh.Trimesh]
    gt_meshes: List[trimesh.Trimesh]
    sample_id: str
    canonical_sample_id: Optional[str] = None
    render_config: Optional[Dict] = None
    gt_render_image: Optional[str] = None
    scene_image: Optional[str] = None
    input_images: List[Tuple[str, str]] = field(default_factory=list)
    pred_object_names: Optional[List[str]] = None
    gt_object_names: Optional[List[str]] = None
    gt_surface_points: Optional[List[Any]] = None
    eval_diagnostics: Dict[str, Any] = field(default_factory=dict)
    precomputed_object_matches: Optional[List[Tuple[int, int]]] = None
    scene_alignment_transform: Optional[Dict[str, Any]] = field(default=None, repr=False)
    gt_instance_mask_path: Optional[str] = field(default=None, repr=False)
    gt_inst_ids_full: Optional[List[str]] = field(default=None, repr=False)
    gt_inst_ids_per_mesh: Optional[List[str]] = field(default=None, repr=False)
    _loader_type: Optional[str] = field(default=None, repr=False)
    _lazy_paths: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _loaded: bool = field(default=True, repr=False)

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._loader_type != 'standard_paths':
            raise ValueError(f'Unsupported mesh loader: {self._loader_type}')
        self._load_standard_paths()
        self._loaded = True

    def _load_standard_paths(self) -> None:
        pred_paths = self._lazy_paths.get('pred_paths', [])
        gt_paths = self._lazy_paths.get('gt_paths', [])
        self.pred_meshes = self._load_mesh_list(pred_paths)
        self.gt_meshes = self._load_mesh_list(gt_paths)

    @staticmethod
    def _load_mesh_list(paths: List[str]) -> List[trimesh.Trimesh]:
        meshes: List[trimesh.Trimesh] = []
        for mesh_path in paths:
            loaded = trimesh.load(mesh_path)
            if isinstance(loaded, trimesh.Scene):
                scene_meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if scene_meshes:
                    meshes.append(scene_meshes[0] if len(scene_meshes) == 1 else trimesh.util.concatenate(scene_meshes))
                else:
                    meshes.append(loaded.dump(concatenate=True))
            elif isinstance(loaded, trimesh.Trimesh):
                meshes.append(loaded)
            else:
                raise ValueError(f'Unsupported mesh type from {mesh_path}: {type(loaded)}')
        return meshes
