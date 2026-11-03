
import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Optional, Set
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--room_ids_json",
        type=str,
        default=None,
        help="Path to JSON file containing valid room IDs for filtering",
    )
    parser.add_argument(
        "--exclude_room_ids_json",
        type=str,
        default=None,
        help="Path to JSON file containing room IDs to exclude",
    )


def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _load_room_ids(json_path: str) -> Set[str]:
    with open(json_path, "r") as f:
        room_ids = json.load(f)
    if not isinstance(room_ids, list):
        raise ValueError(
            f"Expected a JSON list in {json_path}, got {type(room_ids).__name__}"
        )
    return set(room_ids)


def get_metadata(root: Optional[str] = None, **kwargs) -> pd.DataFrame:
    if root is None:
        raise ValueError("root directory must be specified")
    room_ids_json: Optional[str] = kwargs.get("room_ids_json")
    exclude_room_ids_json: Optional[str] = kwargs.get("exclude_room_ids_json")
    include_ids: Optional[Set[str]] = None
    exclude_ids: Optional[Set[str]] = None
    if room_ids_json is not None:
        include_ids = _load_room_ids(room_ids_json)
        print(f"Loaded {len(include_ids)} room IDs from {room_ids_json}")
    if exclude_room_ids_json is not None:
        exclude_ids = _load_room_ids(exclude_room_ids_json)
        print(
            f"Loaded {len(exclude_ids)} room IDs to exclude from {exclude_room_ids_json}"
        )
    raw_dir = Path(root) / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    records = []
    total_scanned = 0
    for scene_dir in sorted(raw_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_name = scene_dir.name
        for house_path in sorted(scene_dir.iterdir()):
            if not house_path.is_dir():
                continue
            house = house_path.name
            for glb_file in sorted(house_path.iterdir()):
                if not glb_file.name.endswith(".glb") or glb_file.name.endswith(
                    "_full.glb"
                ):
                    continue
                total_scanned += 1
                room = glb_file.stem
                room_id = f"{house}/{room}"
                if include_ids is not None and room_id not in include_ids:
                    continue
                if exclude_ids is not None and room_id in exclude_ids:
                    continue
                sha256 = get_file_hash(str(glb_file))
                local_path = str(Path("raw") / scene_name / house / glb_file.name)
                records.append(
                    {
                        "sha256": sha256,
                        "local_path": local_path,
                        "file_identifier": room_id,
                        "house": house,
                        "room": room,
                    }
                )
    if len(records) == 0:
        raise ValueError(f"No GLB files found in {raw_dir}")
    print(f"Scanned {total_scanned} GLB files, kept {len(records)} after filtering")
    return pd.DataFrame.from_records(records)


def download(metadata, output_dir, **kwargs) -> pd.DataFrame:
    downloaded = {}
    for _, row in metadata.iterrows():
        sha256 = row["sha256"]
        local_path = row["local_path"]
        downloaded[sha256] = local_path
    return pd.DataFrame(downloaded.items(), columns=["sha256", "local_path"])


def foreach_instance(
    metadata,
    output_dir,
    func,
    max_workers=None,
    desc="Processing objects",
    no_file=False,
) -> pd.DataFrame:
    metadata = metadata.to_dict("records")
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with (
            ThreadPoolExecutor(max_workers=max_workers) as executor,
            tqdm(total=len(metadata), desc=desc) as pbar,
        ):

            def worker(metadatum):
                try:
                    sha256 = metadatum["sha256"]
                    if no_file:
                        record = func(None, metadatum)
                    else:
                        local_path = metadatum["local_path"]
                        file = os.path.join(output_dir, local_path)
                        record = func(file, metadatum)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(
                        f"Error processing object {metadatum.get('sha256', 'unknown')}: {e}"
                    )
                    pbar.update()

            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing: {e}")
    return pd.DataFrame.from_records(records)
