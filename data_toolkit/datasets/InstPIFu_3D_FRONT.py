
import os
import argparse
import hashlib
from pathlib import Path
from typing import Optional, Set
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd


def add_args(parser: argparse.ArgumentParser) -> None:
    pass


def get_file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_metadata(root: Optional[str] = None, **kwargs) -> pd.DataFrame:
    if root is None:
        raise ValueError("root directory must be specified")
    scenes_dir = Path(root) / "raw" / "scenes"
    if not scenes_dir.exists():
        raise FileNotFoundError(f"Scenes directory not found: {scenes_dir}")
    records = []
    glb_files = sorted(scenes_dir.rglob("*.glb"))
    print(f"Found {len(glb_files)} GLB files in {scenes_dir}")

    def process_glb(glb_path):
        scene_id = glb_path.parent.name
        rendertask_id = glb_path.stem
        local_path = str(Path("raw") / "scenes" / scene_id / glb_path.name)
        file_identifier = f"{scene_id}/{rendertask_id}"
        sha256 = get_file_hash(str(glb_path))
        return {
            "sha256": sha256,
            "local_path": local_path,
            "file_identifier": file_identifier,
            "scene_id": scene_id,
            "rendertask_id": rendertask_id,
        }

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.map(process_glb, glb_files),
                total=len(glb_files),
                desc="Computing SHA256",
            )
        )
    records = [r for r in results if r is not None]
    if len(records) == 0:
        raise ValueError(f"No GLB files found in {scenes_dir}")
    df = pd.DataFrame.from_records(records)
    duplicates = df[df.duplicated(subset="sha256", keep=False)]
    if len(duplicates) > 0:
        print(
            f"WARNING: {len(duplicates)} SHA256 collisions detected! Re-hashing with file_identifier to resolve."
        )
        collision_sha = set(duplicates["sha256"])
        for idx, row in df.iterrows():
            if row["sha256"] in collision_sha:
                combined = f"{row['sha256']}:{row['file_identifier']}"
                df.at[idx, "sha256"] = hashlib.sha256(combined.encode()).hexdigest()
        still_dup = df[df.duplicated(subset="sha256", keep=False)]
        if len(still_dup) > 0:
            raise RuntimeError(
                f"Still have {len(still_dup)} SHA256 collisions after re-hashing!"
            )
    print(f"Generated metadata for {len(df)} samples")
    return df


def download(metadata, output_dir, **kwargs) -> pd.DataFrame:
    downloaded = {}
    for _, row in metadata.iterrows():
        downloaded[row["sha256"]] = row["local_path"]
    return pd.DataFrame(downloaded.items(), columns=["sha256", "local_path"])


def foreach_instance(
    metadata, output_dir, func, max_workers=None, desc="Processing", no_file=False
) -> pd.DataFrame:
    metadata_records = metadata.to_dict("records")
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with (
            ThreadPoolExecutor(max_workers=max_workers) as executor,
            tqdm(total=len(metadata_records), desc=desc) as pbar,
        ):

            def worker(metadatum):
                try:
                    sha256 = metadatum["sha256"]
                    if no_file:
                        record = func(None, metadatum)
                    else:
                        local_path = metadatum["local_path"]
                        file_path = os.path.join(output_dir, local_path)
                        record = func(file_path, metadatum)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing {metadatum.get('sha256', 'unknown')}: {e}")
                    pbar.update()

            executor.map(worker, metadata_records)
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error during processing: {e}")
    return pd.DataFrame.from_records(records)
