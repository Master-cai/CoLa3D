from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import threading
from types import SimpleNamespace
from typing import Any
ROOT = Path(__file__).resolve().parents[2]

def configure_seed(seed: int, sample_id: str) -> int:
    import numpy as np
    import open3d as o3d
    import torch
    base = int(seed)
    os.environ['MESH_EVAL_SAMPLE_SEED'] = str(base)
    derived = (base ^ int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:4], 'little')) % 2 ** 31
    random.seed(derived)
    np.random.seed(derived)
    torch.manual_seed(derived)
    o3d.utility.random.seed(derived)
    return derived

def prepare_pairs(args: argparse.Namespace, recipe: dict[str, Any]) -> list[Any]:
    from mesh_eval.adapters.factory import build_adapters
    from mesh_eval.pairing.strategies import StandardPairer
    from mesh_eval.pairing.validation import validate_samples
    from mesh_eval.pipelines.evaluation import _paired_sample_to_mesh_pair
    ids = args.sample_ids
    paths = {'gt_root': args.gt_root, 'pred_root': args.pred_root, 'metadata': args.metadata, 'gt_mask_root': args.gt_mask_root}
    gt, pred = build_adapters(recipe, paths, ids)
    ground_truth, predictions = (gt.get_samples(), pred.get_samples())
    validate_samples(ground_truth, ids, 'GT', getattr(args, 'manifest_samples', None))
    validate_samples(predictions, ids, 'prediction')
    paired = StandardPairer().pair(ground_truth, predictions)
    by_id = {pair.canonical_sample_id: pair for pair in paired}
    if set(by_id) != set(ids):
        raise ValueError('Adapter pairing does not cover the full selection.')
    return [_paired_sample_to_mesh_pair(by_id[s], surface_point_sample_count=None, object_matching=recipe['object_matching']) for s in ids]

def _cpu_phases(worker_args: tuple, pair: Any) -> tuple:

    from mesh_eval.config import GeometryConfig
    from mesh_eval.core.evaluator import MeshEvaluator
    config, output, object_matching, seed = worker_args
    evaluator = MeshEvaluator(GeometryConfig.from_dict(config), output_dir=Path(output), object_matching=object_matching, device='cpu')
    configure_seed(seed, pair.canonical_sample_id)
    pair.ensure_loaded()
    scene_arrays = evaluator._cpu_phase_scene_3d(pair)
    object_phase = evaluator._cpu_phase_object_3d(pair)
    return scene_arrays, object_phase

def evaluate_pairs(args: argparse.Namespace, recipe: dict[str, Any], pairs: list[Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from mesh_eval.core.evaluator import MeshEvaluator
    config = json.loads((ROOT / recipe['geometry_config']).read_text())
    from mesh_eval.config import GeometryConfig
    evaluator = MeshEvaluator(GeometryConfig.from_dict(config), output_dir=args.output, object_matching=recipe['object_matching'], device=args.device)
    all_metrics = []
    worker_args = (config, str(args.output), recipe['object_matching'], args.evaluation_seed)
    num_workers = max(1, int(getattr(args, 'num_workers', 1)))
    if num_workers == 1:
        cpu_results = (_cpu_phases(worker_args, pair) for pair in pairs)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=num_workers, mp_context=multiprocessing.get_context('spawn'))
        cpu_results = pool.map(_cpu_phases, [worker_args] * len(pairs), pairs)
    for pair, (scene_arrays, object_phase) in zip(pairs, cpu_results):
        metrics = evaluator._gpu_phase_scene_3d(*scene_arrays)
        metrics.update(evaluator._gpu_phase_object_3d(*object_phase))
        metrics['sample_id'] = pair.sample_id
        evaluator._save_sample_results(pair.sample_id, metrics)
        if not all((np.isfinite(value) for key, value in metrics.items() if isinstance(value, (int, float)))):
            raise ValueError(f'Nonfinite metrics: {pair.canonical_sample_id}')
        if metrics.get('object_count', 0) <= 0:
            raise ValueError(f'No object matches: {pair.canonical_sample_id}')
        all_metrics.append(metrics)
    if num_workers > 1:
        pool.shutdown()
    summary = evaluator._aggregate_results(all_metrics)
    evaluator._save_summary_results(summary)
    direction = recipe['metric_direction']
    cd = 'cd_gt2pred' if direction == 'gt_to_pred' else 'cd'
    coverage = 'recall' if direction == 'gt_to_pred' else 'fscore'
    table = {level: {'cd': summary[f'{level}_{cd}_mean'], 'coverage': summary[f'{level}_{coverage}_mean'], 'iou': summary[f'{level}_iou_bbox_mean']} for level in ['scene', 'object']}
    table['evaluation_seed'] = args.evaluation_seed
    (args.output / 'table.json').write_text(json.dumps(table, indent=2) + '\n')
    return table
