import argparse
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def build_rgba(rgb_path: Path, mask_dir: Path) -> Image.Image:
    with Image.open(rgb_path) as source:
        rgb = np.array(source.convert("RGB"))
    mask_paths = sorted(
        path for path in mask_dir.glob("*.png")
        if re.match(r"^\d+_", path.name)
    )
    if not mask_paths:
        raise ValueError(f"No numeric-prefixed PNG masks: {mask_dir}")
    foreground = np.zeros(rgb.shape[:2], dtype=bool)
    for mask_path in mask_paths:
        with Image.open(mask_path) as source:
            if source.size != (rgb.shape[1], rgb.shape[0]):
                raise ValueError(f"RGB/mask size mismatch: {rgb_path}, {mask_path}")
            mask = np.array(source)
        if mask.ndim != 2:
            raise ValueError(f"Mask must be single-channel: {mask_path}")
        mask = mask > 0
        foreground |= mask
    if not foreground.any():
        raise ValueError(f"Empty foreground: {mask_dir}")
    rgb[~foreground] = 0
    alpha = foreground.astype(np.uint8) * 255
    return Image.fromarray(np.dstack((rgb, alpha)), "RGBA")


def prepare_image(paths: tuple[Path, Path]) -> str:
    render_dir, mask_dir = paths
    rgb_path = render_dir / "rgb.png"
    output_path = render_dir / "render_0000.png"
    if not rgb_path.is_file():
        if not output_path.is_file():
            raise FileNotFoundError(f"Missing rgb.png and render_0000.png: {render_dir}")
        with Image.open(output_path) as legacy:
            legacy.load()
            if legacy.mode != "RGBA":
                raise ValueError(f"Legacy conditioning image must be RGBA: {output_path}")
            if legacy.getchannel("A").getextrema()[1] == 0:
                raise ValueError(f"Empty legacy foreground: {output_path}")
        return "legacy"
    image = build_rgba(rgb_path, mask_dir)
    if output_path.is_file():
        try:
            with Image.open(output_path) as existing:
                if existing.mode == "RGBA" and np.array_equal(np.array(existing), np.array(image)):
                    return "unchanged"
        except (OSError, ValueError):
            pass
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=render_dir, prefix=".render_0000.", suffix=".png", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        image.save(temporary_path, format="PNG")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return "prepared"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build InstPIFu conditioning RGBA images from RGB and instance masks")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    scenes_dir = args.root / "raw" / "scenes"
    scenes = sorted(scenes_dir.glob("*/*.glb"))
    if not scenes:
        parser.error(f"No scene/render-task GLBs: {scenes_dir}")
    work = [
        (args.root / "raw" / "renders" / path.parent.name / path.stem,
         args.root / "raw" / "masks" / path.parent.name / path.stem)
        for path in scenes
    ]
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=min(args.max_workers, available_cpus)) as pool:
        for status in pool.map(prepare_image, work):
            counts[status] = counts.get(status, 0) + 1
    print(f"InstPIFu images: {len(work)} samples, {counts}")


if __name__ == "__main__":
    main()
