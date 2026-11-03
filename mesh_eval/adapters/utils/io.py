from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, Optional

def load_json(path: Path) -> object:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def load_json_list(path: Path) -> list:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f'Expected list JSON at {path}, got {type(data)}')
    return data

def load_scene_hash_mapping(metadata_csv: Optional[Path]) -> Dict[str, str]:
    if metadata_csv is None or not metadata_csv.is_file():
        return {}
    with metadata_csv.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        mapping: Dict[str, str] = {}
        for row in reader:
            scene_hash = row.get('sha256')
            room_id = row.get('file_identifier')
            if scene_hash and room_id:
                mapping[scene_hash] = room_id
    return mapping

def path_or_none(value: object) -> Optional[Path]:
    if not value:
        return None
    return Path(str(value))
