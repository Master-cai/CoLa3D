
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class AdapterSpec:
    module: str
    class_name: str

    def build(self, config: object) -> Any:
        return getattr(import_module(self.module), self.class_name)(config)


GT_ADAPTERS: dict[str, AdapterSpec] = {}
PRED_ADAPTERS: dict[str, AdapterSpec] = {}


def register_gt(name: str, spec: AdapterSpec) -> None:
    if name in GT_ADAPTERS:
        raise ValueError(f'GT adapter already registered: {name}')
    GT_ADAPTERS[name] = spec


def register_pred(name: str, spec: AdapterSpec) -> None:
    if name in PRED_ADAPTERS:
        raise ValueError(f'Prediction adapter already registered: {name}')
    PRED_ADAPTERS[name] = spec


for _name, _class in [('instpifu', 'InstPIFu'), ('metascenes', 'MetaScenes')]:
    register_gt(_name, AdapterSpec('mesh_eval.adapters.gt.' + _name, _class + 'GTAdapter'))
for _name in ('replica', 'scannetpp'):
    register_gt(_name, AdapterSpec('mesh_eval.adapters.gt.scene_objects', 'SceneObjectsGTAdapter'))
for _name, _class in [('instpifu', 'InstPIFu'), ('metascenes', 'MetaScenes'), ('replica', 'Replica'), ('scannetpp', 'ScanNetpp')]:
    for _suffix, _class_suffix in [('', ''), ('_mask', 'Mask')]:
        register_pred('cola3d_' + _name + _suffix, AdapterSpec('mesh_eval.adapters.pred.cola3d_' + _name + _suffix, 'CoLa3D' + _class + _class_suffix + 'PredAdapter'))
