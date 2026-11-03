
from __future__ import annotations

from collections import Counter
from typing import Any


def validate_samples(samples: list[Any], expected: list[str], label: str, manifest_samples: list[dict[str,Any]] | None = None) -> None:
    found=[sample.canonical_sample_id for sample in samples]
    missing=sorted(set(expected)-set(found))
    duplicates=[key for key,count in Counter(found).items() if count>1]
    if missing or duplicates or set(found)-set(expected):
        raise ValueError(f'{label}: missing={missing}, duplicate={duplicates}, unexpected={sorted(set(found)-set(expected))}')
    expected_objects={sample['canonical_sample_id']:{str(obj['id']) for obj in sample['gt']['objects']} for sample in (manifest_samples or [])}
    for sample in samples:
        if not sample.objects:
            raise ValueError(f'{label}: empty object list for {sample.canonical_sample_id}')
        for obj in sample.objects:
            if obj.mesh_path is None or not obj.mesh_path.is_file():
                raise FileNotFoundError(f'{label} mesh missing: {sample.canonical_sample_id}/{obj.object_id}')
        if label=='GT' and expected_objects:
            actual={str(obj.match_key) for obj in sample.objects}
            if actual!=expected_objects[sample.canonical_sample_id]:
                raise ValueError(f'GT object IDs do not match the manifest: {sample.canonical_sample_id}')
