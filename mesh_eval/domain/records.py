from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class InputImageRecord:
    path: Path
    dest_name: str

@dataclass
class StandardObjectRecord:
    object_id: str
    display_id: str
    canonical_name: Optional[str]
    match_key: Optional[str]
    mesh_path: Optional[Path]
    surface_points_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    @property
    def effective_name(self) -> str:
        return self.canonical_name or self.match_key or self.display_id or self.object_id

@dataclass
class StandardSampleRecord:
    source_sample_id: str
    canonical_sample_id: Optional[str]
    dataset_name: str
    adapter_name: str
    objects: List[StandardObjectRecord] = field(default_factory=list)
    scene_mesh_path: Optional[Path] = None
    sample_dir: Optional[Path] = None
    asset_root: Optional[Path] = None
    input_images: List[InputImageRecord] = field(default_factory=list)
    render_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    @property
    def resolved_sample_key(self) -> str:
        return self.canonical_sample_id or self.source_sample_id

    def get_object_dir(self) -> Optional[Path]:
        return self.sample_dir

@dataclass
class PairedSampleRecord:
    pred_sample: StandardSampleRecord
    gt_sample: StandardSampleRecord
    canonical_sample_id: str
    object_matches: List[Tuple[int, int]] = field(default_factory=list)
    unmatched_pred_indices: List[int] = field(default_factory=list)
    unmatched_gt_indices: List[int] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)

    @property
    def actual_sample_id(self) -> str:
        return self.pred_sample.source_sample_id
