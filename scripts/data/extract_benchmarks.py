"""Safely extract the five benchmark archives into a repository checkout."""

import argparse
from pathlib import Path, PurePosixPath
import tarfile

import zstandard

from extract_training import ensure_directory, extract_file


DATASETS = ("InstPIFu", "MetaScenes", "MetaScenes_Hard", "Replica", "ScanNetpp")


def extract_archives(archive_dir: Path, output_dir: Path) -> tuple[int, int]:
    archives = [(archive_dir / f"{dataset}.tar.zst", dataset) for dataset in DATASETS]
    for archive, _ in archives:
        if archive.is_symlink() or not archive.is_file():
            raise ValueError(f"Missing or unsafe benchmark archive: {archive}")
    output_dir = output_dir.absolute()
    for parent in reversed((output_dir, *output_dir.parents)):
        ensure_directory(parent)
    seen: set[str] = set()
    written = matched = 0
    for archive, dataset in archives:
        archive_members = 0
        with archive.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed, read_across_frames=True) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as contents:
                    for member in contents:
                        path = PurePosixPath(member.name)
                        parts = path.parts
                        if (
                            not member.isreg()
                            or path.is_absolute()
                            or ".." in parts
                            or "\\" in member.name
                            or member.name != path.as_posix()
                            or len(parts) < 3
                            or parts[:2] != ("benchmarks", dataset)
                        ):
                            raise ValueError(f"Unsafe archive member in {archive}: {member.name}")
                        if member.name in seen:
                            raise ValueError(f"Duplicate archive member: {member.name}")
                        seen.add(member.name)
                        archive_members += 1
                        stream = contents.extractfile(member)
                        if stream is None:
                            raise ValueError(f"Cannot read member: {member.name}")
                        with stream:
                            if extract_file(stream, output_dir.joinpath(*parts), member.size):
                                written += 1
                            else:
                                matched += 1
                while reader.read(1024 * 1024):
                    pass
        if archive_members == 0:
            raise ValueError(f"Empty benchmark archive: {archive}")
        print(f"Verified {archive.name}: {written} extracted, {matched} already identical", flush=True)
    return written, matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("archives"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    written, matched = extract_archives(args.archive_dir, args.output_dir)
    print(f"Done: {written} extracted, {matched} already identical")


if __name__ == "__main__":
    main()
