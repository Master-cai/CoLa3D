import argparse
import hashlib
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import zstandard


DATASETS = ("InstPIFu-3D-FRONT", "MIDI-3D-FRONT")
SHARD_NAME = re.compile(r"(InstPIFu-3D-FRONT|MIDI-3D-FRONT)-(\d{5})-of-(\d{5})\.tar\.zst")


def find_shards(archive_dir: Path) -> list[tuple[Path, str]]:
    groups: dict[str, dict[int, Path]] = {name: {} for name in DATASETS}
    totals: dict[str, int] = {}
    for path in sorted(archive_dir.glob("*.tar.zst")):
        if not path.name.startswith(tuple(f"{name}-" for name in DATASETS)):
            continue
        match = SHARD_NAME.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise ValueError(f"Invalid archive: {path}")
        dataset, index_text, total_text = match.groups()
        index, total = int(index_text), int(total_text)
        if total < 1 or index >= total or totals.get(dataset, total) != total:
            raise ValueError(f"Invalid or mixed shard counts: {path}")
        totals[dataset] = total
        if index in groups[dataset]:
            raise ValueError(f"Duplicate shard index: {path}")
        groups[dataset][index] = path
    result: list[tuple[Path, str]] = []
    for dataset, shards in groups.items():
        if not shards:
            raise ValueError(f"No archives found for {dataset} in {archive_dir}")
        missing = sorted(set(range(totals[dataset])) - shards.keys())
        if missing:
            raise ValueError(f"Missing {dataset} shard indices: {missing}")
        result.extend((shards[index], dataset) for index in sorted(shards))
    return result


def ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Symlink directory is not allowed: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")
        return
    ensure_directory(path.parent)
    path.mkdir(exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Unsafe directory: {path}")


def digest_stream(stream: BinaryIO, output: BinaryIO | None = None) -> tuple[int, bytes]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        size += len(chunk)
        digest.update(chunk)
        if output is not None:
            output.write(chunk)
    return size, digest.digest()


def extract_file(stream: BinaryIO, target: Path, expected_size: int) -> bool:
    ensure_directory(target.parent)
    if target.is_symlink():
        raise ValueError(f"Symlink target is not allowed: {target}")
    if target.exists():
        if not target.is_file():
            raise ValueError(f"Not a regular file: {target}")
        incoming = digest_stream(stream)
        with target.open("rb") as existing:
            current = digest_stream(existing)
        if incoming[0] != expected_size or incoming != current:
            raise ValueError(f"Existing file differs from archive: {target}")
        return False
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".extract-", delete=False) as output:
            temporary = Path(output.name)
            size, _ = digest_stream(stream, output)
            output.flush()
            os.fsync(output.fileno())
        if size != expected_size:
            raise ValueError(f"Truncated member: {target}")
        temporary.chmod(0o644)
        os.link(temporary, target)
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_archives(archive_dir: Path, output_dir: Path) -> tuple[int, int]:
    shards = find_shards(archive_dir)
    output_dir = output_dir.absolute()
    for parent in reversed((output_dir, *output_dir.parents)):
        ensure_directory(parent)
    seen: set[str] = set()
    written = matched = 0
    for archive, dataset in shards:
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
                            or len(parts) < 5
                            or parts[:4] != ("datasets", dataset, "Train", "raw")
                        ):
                            raise ValueError(f"Unsafe archive member in {archive}: {member.name}")
                        if member.name in seen:
                            raise ValueError(f"Duplicate archive member: {member.name}")
                        seen.add(member.name)
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
        print(f"Verified {archive.name}: {written} extracted, {matched} already identical", flush=True)
    return written, matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, default=Path("archives"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    written, matched = extract_archives(args.archive_dir, args.output_dir)
    print(f"Done: {written} extracted, {matched} already identical")


if __name__ == "__main__":
    main()
