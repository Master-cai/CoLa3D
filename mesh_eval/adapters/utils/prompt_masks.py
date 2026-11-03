from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
_PROMPT_MASK_RE = re.compile('^prompt_(\\d+)\\.png$')
_INSTPIFU_GT_MASK_RE = re.compile('^(\\d+)_.*\\.png$')
_INSTANCE_GT_MASK_RE = re.compile('^obj_(\\d+)\\.png$')
_METASCENES_OBJ_FILE_RE = re.compile('^(\\d+)_(.+)\\.glb$')

@dataclass(frozen=True)
class PromptMaskMatch:
    part_index: int
    match_key: str
    prompt_path: Path
    gt_mask_path: Path
    iou: float

@dataclass(frozen=True)
class _PromptMaskItem:
    part_index: int
    path: Path
    mask: np.ndarray

@dataclass(frozen=True)
class _GTMaskItem:
    match_key: str
    path: Path
    raw_mask: np.ndarray
    resized_mask: np.ndarray

def match_instpifu_prompt_masks(*, variant_dir: Path, gt_mask_dir: Path, min_iou: float=0.5) -> Dict[int, PromptMaskMatch]:
    prompt_items = _load_prompt_masks(variant_dir)
    raw_gt_items = _load_instpifu_gt_masks(gt_mask_dir)
    if not prompt_items or not raw_gt_items:
        return {}
    crop_source = np.stack([item.raw_mask for item in raw_gt_items], axis=0).any(axis=0)
    crop_bbox = _crop_bbox_from_binary_mask(crop_source.astype(np.uint8))
    if crop_bbox is None:
        return {}
    target_size = (int(prompt_items[0].mask.shape[1]), int(prompt_items[0].mask.shape[0]))
    gt_items = [_GTMaskItem(match_key=item.match_key, path=item.path, raw_mask=item.raw_mask, resized_mask=_load_cropped_resized_binary_mask(item.path, crop_bbox=crop_bbox, target_size=target_size)) for item in raw_gt_items]
    return _match_by_iou(prompt_items, gt_items, min_iou=min_iou)

def _load_prompt_masks(variant_dir: Path) -> List[_PromptMaskItem]:
    items: List[_PromptMaskItem] = []
    for mask_path in sorted(variant_dir.glob('prompt_*.png')):
        match = _PROMPT_MASK_RE.match(mask_path.name)
        if match is None:
            continue
        mask = _load_binary_mask(mask_path)
        if mask.sum() == 0:
            continue
        items.append(_PromptMaskItem(part_index=int(match.group(1)), path=mask_path, mask=mask))
    return items

def _load_instpifu_gt_masks(gt_mask_dir: Path) -> List[_GTMaskItem]:
    items: List[_GTMaskItem] = []
    for mask_path in sorted(gt_mask_dir.glob('*.png')):
        match = _INSTPIFU_GT_MASK_RE.match(mask_path.name)
        if match is None:
            continue
        mask = _load_binary_mask(mask_path)
        if mask.sum() == 0:
            continue
        items.append(_GTMaskItem(match_key=str(int(match.group(1))), path=mask_path, raw_mask=mask, resized_mask=mask))
    return items

def match_instance_prompt_masks(*, variant_dir: Path, gt_mask_dir: Path, min_iou: float=0.5) -> Dict[int, PromptMaskMatch]:
    prompt_items = _load_prompt_masks(variant_dir)
    raw_gt_items = _load_instance_gt_masks(gt_mask_dir)
    if not prompt_items or not raw_gt_items:
        return {}
    crop_source = np.stack([item.raw_mask for item in raw_gt_items], axis=0).any(axis=0)
    crop_bbox = _crop_bbox_from_binary_mask(crop_source.astype(np.uint8))
    if crop_bbox is None:
        return {}
    target_size = (int(prompt_items[0].mask.shape[1]), int(prompt_items[0].mask.shape[0]))
    gt_items = [_GTMaskItem(match_key=item.match_key, path=item.path, raw_mask=item.raw_mask, resized_mask=_load_cropped_resized_binary_mask(item.path, crop_bbox=crop_bbox, target_size=target_size)) for item in raw_gt_items]
    return _match_by_iou(prompt_items, gt_items, min_iou=min_iou)

def _load_instance_gt_masks(gt_mask_dir: Path) -> List[_GTMaskItem]:
    items: List[_GTMaskItem] = []
    for mask_path in sorted(gt_mask_dir.glob('obj_*.png')):
        match = _INSTANCE_GT_MASK_RE.match(mask_path.name)
        if match is None:
            continue
        mask = _load_binary_mask(mask_path)
        if mask.sum() == 0:
            continue
        items.append(_GTMaskItem(match_key=str(int(match.group(1))), path=mask_path, raw_mask=mask, resized_mask=mask))
    return items

def _match_by_iou(prompt_items: List[_PromptMaskItem], gt_items: List[_GTMaskItem], *, min_iou: float) -> Dict[int, PromptMaskMatch]:
    candidates: List[Tuple[float, int, int]] = []
    for prompt_idx, prompt_item in enumerate(prompt_items):
        for gt_idx, gt_item in enumerate(gt_items):
            iou = _mask_iou(prompt_item.mask, gt_item.resized_mask)
            if iou >= min_iou:
                candidates.append((iou, prompt_idx, gt_idx))
    candidates.sort(key=lambda item: (-item[0], prompt_items[item[1]].part_index, gt_items[item[2]].match_key))
    matches: Dict[int, PromptMaskMatch] = {}
    used_prompts: set[int] = set()
    used_gt: set[int] = set()
    for iou, prompt_idx, gt_idx in candidates:
        if prompt_idx in used_prompts or gt_idx in used_gt:
            continue
        prompt_item = prompt_items[prompt_idx]
        gt_item = gt_items[gt_idx]
        matches[prompt_item.part_index] = PromptMaskMatch(part_index=prompt_item.part_index, match_key=gt_item.match_key, prompt_path=prompt_item.path, gt_mask_path=gt_item.path, iou=float(iou))
        used_prompts.add(prompt_idx)
        used_gt.add(gt_idx)
    return matches

def _load_resized_binary_mask(path: Path, *, target_size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert('L').resize(target_size, Image.Resampling.NEAREST)
        return np.asarray(resized) > 127

def _load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('L')) > 127

def _load_cropped_resized_binary_mask(path: Path, *, crop_bbox: Tuple[int, int, int, int], target_size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert('L').crop(crop_bbox)
        resized = mask.resize(target_size, Image.Resampling.NEAREST)
        return np.asarray(resized) > 127

def _crop_bbox_from_binary_mask(binary_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    nonzero = binary_mask.nonzero()
    if nonzero[0].size == 0:
        return None
    bbox = [nonzero[1].min(), nonzero[0].min(), nonzero[1].max(), nonzero[0].max()]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    half_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    return (int(center[0] - half_size), int(center[1] - half_size), int(center[0] + half_size), int(center[1] + half_size))

def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)

def match_metascenes_prompt_masks(*, variant_dir: Path, view_dir: Path, min_iou: float=0.5) -> Dict[int, 'PromptMaskMatch']:
    import json
    prompt_items = _load_prompt_masks(variant_dir)
    if not prompt_items:
        return {}
    instance_mask_path = view_dir / 'instance_mask.png'
    metadata_path = view_dir / 'metadata.json'
    objects_dir = view_dir / 'objects'
    if not (instance_mask_path.is_file() and metadata_path.is_file() and objects_dir.is_dir()):
        return {}
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    inst_ids_full = [str(x) for x in meta.get('inst_ids', [])]
    if not inst_ids_full:
        return {}
    kept_inst_ids: List[str] = []
    for mesh_path in sorted(objects_dir.glob('*.glb'), key=lambda p: int(_METASCENES_OBJ_FILE_RE.match(p.name).group(1)) if _METASCENES_OBJ_FILE_RE.match(p.name) else 10 ** 9):
        m = _METASCENES_OBJ_FILE_RE.match(mesh_path.name)
        if m is not None:
            kept_inst_ids.append(m.group(1))
    if not kept_inst_ids:
        return {}
    per_inst_full = _decode_metascenes_palette_mask(instance_mask_path, inst_ids_full)
    if per_inst_full is None:
        return {}
    raw_gt_items: List[_GTMaskItem] = []
    for iid in kept_inst_ids:
        mask = per_inst_full.get(iid)
        if mask is None or mask.sum() == 0:
            continue
        raw_gt_items.append(_GTMaskItem(match_key=iid, path=instance_mask_path, raw_mask=mask, resized_mask=mask))
    if not raw_gt_items:
        return {}
    crop_source = np.stack([item.raw_mask for item in raw_gt_items], axis=0).any(axis=0)
    crop_bbox = _crop_bbox_from_binary_mask(crop_source.astype(np.uint8))
    if crop_bbox is None:
        return {}
    target_size = (int(prompt_items[0].mask.shape[1]), int(prompt_items[0].mask.shape[0]))
    gt_items = [_GTMaskItem(match_key=item.match_key, path=item.path, raw_mask=item.raw_mask, resized_mask=_crop_resize_decoded_mask(item.raw_mask, crop_bbox=crop_bbox, target_size=target_size)) for item in raw_gt_items]
    return _match_by_iou(prompt_items, gt_items, min_iou=min_iou)

def _decode_metascenes_palette_mask(instance_mask_path: Path, inst_ids_full: List[str]) -> Optional[Dict[str, np.ndarray]]:
    import colorsys
    try:
        mask_img = np.asarray(Image.open(instance_mask_path).convert('RGB'), dtype=np.int32)
    except Exception:
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
    decoded = (diff ** 2).sum(axis=-1).argmin(axis=1).reshape(mask_img.shape[:2])
    return {iid: decoded == idx + 1 for idx, iid in enumerate(inst_ids_full)}

def _crop_resize_decoded_mask(binary_mask: np.ndarray, *, crop_bbox: Tuple[int, int, int, int], target_size: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(binary_mask.astype(np.uint8) * 255, mode='L')
    cropped = pil.crop(crop_bbox)
    resized = cropped.resize(target_size, Image.Resampling.NEAREST)
    return np.asarray(resized) > 127
