from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple
from mesh_eval.domain import PairedSampleRecord, StandardObjectRecord, StandardSampleRecord

class StandardPairer:

    def __init__(self, *, sample_match_field: str='canonical_sample_id', object_match_field: str='match_key', infer_sample_from_object_keys: bool=False) -> None:
        self.sample_match_field = sample_match_field
        self.object_match_field = object_match_field
        self.infer_sample_from_object_keys = infer_sample_from_object_keys

    def pair(self, gt_samples: Sequence[StandardSampleRecord], pred_samples: Sequence[StandardSampleRecord]) -> List[PairedSampleRecord]:
        gt_by_key: Dict[str, StandardSampleRecord] = {}
        gt_ids_by_object_key: Dict[str, List[str]] = defaultdict(list)
        gt_by_room_id: Dict[str, StandardSampleRecord] = {}
        for sample in gt_samples:
            key = self._sample_key(sample)
            gt_by_key[key] = sample
            gt_by_room_id[sample.resolved_sample_key] = sample
            for obj in sample.objects:
                object_key = self._object_key(obj)
                if object_key:
                    gt_ids_by_object_key[object_key].append(sample.resolved_sample_key)
        paired: List[PairedSampleRecord] = []
        for pred_sample in pred_samples:
            diagnostics: List[str] = list(pred_sample.diagnostics)
            key = self._sample_key(pred_sample)
            gt_sample = gt_by_key.get(key)
            canonical_id = pred_sample.canonical_sample_id or pred_sample.source_sample_id
            if gt_sample is None and self.infer_sample_from_object_keys:
                inferred = self._infer_gt_sample(pred_sample, gt_ids_by_object_key, gt_by_room_id)
                if inferred is not None:
                    gt_sample = inferred
                    canonical_id = inferred.resolved_sample_key
                    diagnostics.append('Canonical sample ID inferred from object match keys.')
            if gt_sample is None:
                diagnostics.append('GT sample could not be resolved for prediction sample.')
                continue
            matches, unmatched_pred, unmatched_gt = self._match_objects(pred_sample.objects, gt_sample.objects)
            paired.append(PairedSampleRecord(pred_sample=pred_sample, gt_sample=gt_sample, canonical_sample_id=canonical_id, object_matches=matches, unmatched_pred_indices=unmatched_pred, unmatched_gt_indices=unmatched_gt, diagnostics=diagnostics))
        return paired

    def _sample_key(self, sample: StandardSampleRecord) -> str:
        if self.sample_match_field == 'source_sample_id':
            return sample.source_sample_id
        return sample.resolved_sample_key

    def _object_key(self, obj: StandardObjectRecord) -> Optional[str]:
        if self.object_match_field == 'canonical_name':
            return obj.canonical_name
        if self.object_match_field == 'object_id':
            return obj.object_id
        return obj.match_key or obj.canonical_name

    def _match_objects(self, pred_objects: Sequence[StandardObjectRecord], gt_objects: Sequence[StandardObjectRecord]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        pred_keys = [self._object_key(obj) for obj in pred_objects]
        gt_keys = [self._object_key(obj) for obj in gt_objects]
        matches: List[Tuple[int, int]] = []
        if any(pred_keys) and any(gt_keys):
            gt_index = {str(key): idx for idx, key in enumerate(gt_keys) if key is not None}
            matches = [(pred_idx, gt_index[str(key)]) for pred_idx, key in enumerate(pred_keys) if key is not None and str(key) in gt_index]
        if not matches:
            matches = [(idx, idx) for idx in range(min(len(pred_objects), len(gt_objects)))]
        matched_pred = {pred_idx for pred_idx, _ in matches}
        matched_gt = {gt_idx for _, gt_idx in matches}
        unmatched_pred = [idx for idx in range(len(pred_objects)) if idx not in matched_pred]
        unmatched_gt = [idx for idx in range(len(gt_objects)) if idx not in matched_gt]
        return (matches, unmatched_pred, unmatched_gt)

    def _infer_gt_sample(self, pred_sample: StandardSampleRecord, gt_ids_by_object_key: Dict[str, List[str]], gt_by_room_id: Dict[str, StandardSampleRecord]) -> Optional[StandardSampleRecord]:
        candidates: Optional[set[str]] = None
        scores: Dict[str, int] = defaultdict(int)
        for obj in pred_sample.objects:
            object_key = self._object_key(obj)
            if not object_key:
                continue
            room_ids = set(gt_ids_by_object_key.get(object_key, []))
            for room_id in room_ids:
                scores[room_id] += 1
            if not room_ids:
                continue
            candidates = room_ids if candidates is None else candidates & room_ids
            if candidates and len(candidates) == 1:
                return gt_by_room_id[next(iter(candidates))]
        if candidates and len(candidates) == 1:
            return gt_by_room_id[next(iter(candidates))]
        if scores:
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                return gt_by_room_id[ranked[0][0]]
        return None
