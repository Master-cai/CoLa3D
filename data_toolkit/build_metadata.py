"""Build scene metadata and merge completed preparation records."""

from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def merge_records(directory: Path) -> None:
    import pandas as pd

    records = sorted((directory / "new_records").glob("part_*.csv"))
    if not records:
        return
    metadata_path = directory / "metadata.csv"
    metadata = (
        pd.read_csv(metadata_path).set_index("sha256")
        if metadata_path.exists()
        else pd.DataFrame()
    )
    for path in records:
        update = pd.read_csv(path).set_index("sha256")
        if not update.index.is_unique:
            raise ValueError(f"Duplicate scene identifiers in {path}")
        metadata = update.combine_first(metadata)
    metadata.index.name = "sha256"
    metadata.to_csv(metadata_path)
    archive = directory / "merged_records"
    archive.mkdir(exist_ok=True)
    stamp = time.time_ns()
    for path in records:
        path.rename(archive / f"{stamp}_{path.name}")
    print(f"{directory}: {len(metadata)} records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["InstPIFu_3D_FRONT", "MIDI_3D_FRONT"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--room_ids_json", type=str)
    parser.add_argument("--exclude_room_ids_json", type=str)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "metadata.csv").exists():
        adapter = importlib.import_module(f"data_toolkit.datasets.{args.dataset}")
        metadata = adapter.get_metadata(
            root=str(root),
            room_ids_json=args.room_ids_json,
            exclude_room_ids_json=args.exclude_room_ids_json,
        )
        metadata.to_csv(root / "metadata.csv", index=False)
        print(f"{root}: {len(metadata)} scenes")
    directories = [root / "mesh_dumps"]
    directories.extend(sorted(root.glob("dual_grid_[0-9]*")))
    directories.extend(sorted((root / "shape_latents").glob("*")))
    directories.extend(sorted((root / "ss_latents").glob("*")))
    directories.extend(sorted(root.glob("utonia_feats*")))
    for directory in directories:
        if directory.is_dir():
            merge_records(directory)


if __name__ == "__main__":
    main()
