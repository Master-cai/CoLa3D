
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class SingleImageDataset:

    def __init__(
        self,
        *,
        input_rgb: Optional[str] = None,
        input_mask: Optional[str] = None,
        input_rgba: Optional[str] = None,
        prompt_mask_dir: Optional[str] = None,
        image_size: int = 512,
        instance_id: Optional[str] = None,
        mask_mode: str = "auto",
        mask_threshold: int = 0,
    ) -> None:
        if input_rgba:
            if input_rgb or input_mask:
                raise ValueError(
                    "--input-rgba is mutually exclusive with --input-rgb/--input-mask"
                )
            self.input_rgba = Path(input_rgba).expanduser().resolve()
            self.input_rgb = None
            self.input_mask = None
            if not self.input_rgba.is_file():
                raise FileNotFoundError(f"input RGBA not found: {self.input_rgba}")
            default_id = self.input_rgba.stem
        else:
            if not input_rgb or not input_mask:
                raise ValueError(
                    "single_image requires either --input-rgba or both --input-rgb and --input-mask"
                )
            self.input_rgba = None
            self.input_rgb = Path(input_rgb).expanduser().resolve()
            self.input_mask = Path(input_mask).expanduser().resolve()
            if not self.input_rgb.is_file():
                raise FileNotFoundError(f"input RGB not found: {self.input_rgb}")
            if not self.input_mask.is_file():
                raise FileNotFoundError(f"input mask not found: {self.input_mask}")
            default_id = self.input_rgb.stem

        self.prompt_mask_dir = (
            Path(prompt_mask_dir).expanduser().resolve() if prompt_mask_dir else None
        )
        if self.prompt_mask_dir is not None and not self.prompt_mask_dir.is_dir():
            raise FileNotFoundError(
                f"single-image prompt mask directory not found: {self.prompt_mask_dir}"
            )

        if mask_mode not in {"auto", "binary", "instance", "palette"}:
            raise ValueError(
                f"mask_mode must be auto/binary/instance/palette, got {mask_mode!r}"
            )
        self.image_size = int(image_size)
        self.mask_mode = mask_mode
        self.mask_threshold = int(np.clip(mask_threshold, 0, 255))
        self.instance_id = self._sanitize_instance_id(instance_id or default_id)

        self.instances: List[Tuple[Dict[str, str], str]] = [
            ({"dataset": "single_image"}, self.instance_id)
        ]
        self._render_file_cache: Dict[str, List[str]] = {}

    def __len__(self) -> int:
        return 1

    @staticmethod
    def _sanitize_instance_id(value: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
        clean = clean.strip("._-")
        return clean or "single_image"

    @staticmethod
    def _remap_labels(labels: np.ndarray) -> np.ndarray:
        out = np.zeros(labels.shape, dtype=np.uint8)
        unique_ids = [int(v) for v in np.unique(labels) if int(v) != 0]
        if len(unique_ids) > 254:
            raise ValueError(
                f"single-image mask has {len(unique_ids)} non-zero ids; uint8 prompt ids support at most 254"
            )
        for new_id, old_id in enumerate(unique_ids, start=1):
            out[labels == old_id] = new_id
        return out

    def _decode_grayscale_mask(self, mask_arr: np.ndarray) -> np.ndarray:
        if self.mask_mode == "palette":
            raise ValueError(
                "--input-mask-mode=palette requires an RGB/RGBA mask image"
            )

        labels = mask_arr.astype(np.int64)
        if self.mask_mode == "binary":
            return (labels > self.mask_threshold).astype(np.uint8)

        unique_nonzero = [int(v) for v in np.unique(labels) if int(v) != 0]
        if self.mask_mode == "instance":
            return self._remap_labels(labels)

        if len(unique_nonzero) <= 1:
            return (labels > self.mask_threshold).astype(np.uint8)
        if len(unique_nonzero) <= 32:
            return self._remap_labels(labels)
        return (labels > self.mask_threshold).astype(np.uint8)

    def _decode_color_mask(self, mask_arr: np.ndarray) -> np.ndarray:
        rgb = mask_arr[..., :3].astype(np.uint8)
        alpha = mask_arr[..., 3] if mask_arr.shape[-1] >= 4 else None

        if self.mask_mode == "instance":
            return self._decode_grayscale_mask(rgb[..., 0])

        if self.mask_mode == "binary":
            foreground = (rgb > self.mask_threshold).any(axis=-1)
            if alpha is not None:
                foreground |= alpha > self.mask_threshold
            return foreground.astype(np.uint8)

        flat = rgb.reshape(-1, 3)
        non_bg = (flat > self.mask_threshold).any(axis=1)
        colors, inverse = np.unique(flat[non_bg], axis=0, return_inverse=True)
        if self.mask_mode == "palette" or (1 < colors.shape[0] <= 254):
            if colors.shape[0] > 254:
                raise ValueError(
                    f"single-image palette mask has {colors.shape[0]} colors; uint8 prompt ids support at most 254"
                )
            out = np.zeros((flat.shape[0],), dtype=np.uint8)
            if colors.shape[0] > 0:
                out[non_bg] = inverse.astype(np.uint8) + 1
            return out.reshape(rgb.shape[:2])

        foreground = non_bg.reshape(rgb.shape[:2])
        if alpha is not None:
            foreground |= alpha > self.mask_threshold
        return foreground.astype(np.uint8)

    def _load_rgb_alpha_mask(self) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        if self.input_rgba is not None:
            rgba_img = Image.open(self.input_rgba).convert("RGBA")
            alpha_arr = np.array(rgba_img.getchannel("A"), dtype=np.uint8)
            instance_mask = (alpha_arr > self.mask_threshold).astype(np.uint8)
            return rgba_img.convert("RGB"), alpha_arr, instance_mask

        assert self.input_rgb is not None and self.input_mask is not None
        rgb_img = Image.open(self.input_rgb).convert("RGB")
        mask_img = Image.open(self.input_mask)
        if mask_img.size != rgb_img.size:
            raise ValueError(
                f"input RGB/mask size mismatch: rgb={rgb_img.size}, mask={mask_img.size}"
            )

        mask_arr = np.array(mask_img)
        if mask_arr.ndim == 2:
            instance_mask = self._decode_grayscale_mask(mask_arr)
        elif mask_arr.ndim == 3 and mask_arr.shape[-1] in (3, 4):
            instance_mask = self._decode_color_mask(mask_arr)
        else:
            raise ValueError(f"unsupported mask shape: {mask_arr.shape}")

        alpha_arr = (instance_mask > 0).astype(np.uint8) * 255
        return rgb_img, alpha_arr, instance_mask

    @staticmethod
    def _crop_bbox_from_alpha(
        alpha_arr: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        nz = np.argwhere(alpha_arr > 0)
        if nz.size == 0:
            return None

        ymin, xmin = nz.min(axis=0)
        ymax, xmax = nz.max(axis=0)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        hsize = max(xmax - xmin, ymax - ymin) / 2.0
        return (
            int(cx - hsize),
            int(cy - hsize),
            int(cx + hsize),
            int(cy + hsize),
        )

    def read_cond_image(
        self,
        instance: str,
        root: Optional[Dict[str, str]] = None,
    ) -> Optional[torch.Tensor]:
        rgb_img, alpha_arr, _ = self._load_rgb_alpha_mask()
        bbox = self._crop_bbox_from_alpha(alpha_arr)
        if bbox is None:
            return None

        rgba_img = Image.merge(
            "RGBA",
            (*rgb_img.split(), Image.fromarray(alpha_arr, mode="L")),
        )
        rgba_img = rgba_img.crop(bbox).resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )

        alpha_t = torch.from_numpy(np.array(rgba_img.getchannel(3))).float() / 255.0
        rgb_t = (
            torch.from_numpy(np.array(rgba_img.convert("RGB"))).permute(2, 0, 1).float()
            / 255.0
        )
        return rgb_t * alpha_t.unsqueeze(0)

    def read_raw_rgb_and_mask(self, instance: str) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_img, _, instance_mask = self._load_rgb_alpha_mask()
        rgb_t = torch.from_numpy(np.array(rgb_img)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(instance_mask.astype(np.uint8))
        return rgb_t, mask_t

    def read_prompt_masks(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.prompt_mask_dir is None:
            return None, None

        mask_paths = sorted(self.prompt_mask_dir.glob("object_*_mask.png"))
        if not mask_paths:
            raise FileNotFoundError(
                f"no object_*_mask.png files under {self.prompt_mask_dir}"
            )

        rgb_img, alpha_arr, _ = self._load_rgb_alpha_mask()
        bbox = self._crop_bbox_from_alpha(alpha_arr)
        if bbox is None:
            raise RuntimeError(
                "Cannot build prompt masks from an empty foreground mask"
            )

        prompt_masks: List[torch.Tensor] = []
        object_ids: List[int] = []
        expected_size = rgb_img.size
        for fallback_id, mask_path in enumerate(mask_paths, start=1):
            mask_img = Image.open(mask_path).convert("L")
            if mask_img.size != expected_size:
                raise ValueError(
                    f"prompt mask size mismatch: {mask_path}={mask_img.size}, "
                    f"condition={expected_size}"
                )

            mask_img = mask_img.crop(bbox).resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST
            )
            mask = torch.from_numpy(np.array(mask_img, dtype=np.uint8) > 0).float()
            if not bool(mask.any()):
                continue

            match = re.search(r"object_(\d+)_mask$", mask_path.stem)
            object_id = int(match.group(1)) if match else fallback_id
            prompt_masks.append(mask)
            object_ids.append(object_id)

        if not prompt_masks:
            raise RuntimeError(
                f"all prompt masks are empty after crop/resize under {self.prompt_mask_dir}"
            )
        return torch.stack(prompt_masks, dim=0), torch.tensor(
            object_ids, dtype=torch.long
        )

    def get_instance(
        self,
        root: Dict[str, str],
        instance: str,
    ) -> Dict[str, Any]:
        if instance != self.instance_id:
            raise KeyError(f"unknown single-image instance: {instance}")
        cond_image = self.read_cond_image(instance, root=root)
        if cond_image is None:
            raise RuntimeError(f"Empty foreground mask for {instance}")
        raw_rgb, instance_mask = self.read_raw_rgb_and_mask(instance)
        prompt_masks, prompt_object_ids = self.read_prompt_masks()
        sample = {
            "cond_image": cond_image,
            "raw_rgb": raw_rgb,
            "instance_mask_full": instance_mask,
            "single_image_input": {
                "input_rgb": (
                    str(self.input_rgb) if self.input_rgb is not None else None
                ),
                "input_mask": (
                    str(self.input_mask) if self.input_mask is not None else None
                ),
                "input_rgba": (
                    str(self.input_rgba) if self.input_rgba is not None else None
                ),
                "mask_mode": self.mask_mode,
                "mask_threshold": self.mask_threshold,
                "num_mask_ids": int(
                    len([v for v in np.unique(instance_mask.numpy()) if int(v) != 0])
                ),
            },
        }
        if prompt_masks is not None and prompt_object_ids is not None:
            sample["prompt_masks"] = prompt_masks
            sample["prompt_mask_num_queries"] = int(prompt_masks.shape[0])
            sample["prompt_mask_object_ids"] = prompt_object_ids
        return sample
