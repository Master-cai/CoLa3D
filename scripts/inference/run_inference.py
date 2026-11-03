

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decomposable Scene Inference")
    parser.add_argument(
        "--cola3d-ckpt", required=True, help="CoLa3D + MaskDecoder checkpoint (.pt)"
    )
    parser.add_argument(
        "--geo-ckpt", help="Geometry decoder checkpoint (.pt)", default=None
    )
    parser.add_argument(
        "--pretrained-geo",
        default=None,
        help="Load geo backbone from pretrained .safetensors (fp16) instead of --geo-ckpt. Overrides decoder dtype to fp16 for correct precision.",
    )
    parser.add_argument(
        "--ss-flow-ckpt",
        default=None,
        help="Fine-tuned SS flow denoiser checkpoint (.pt).",
    )
    parser.add_argument(
        "--ss-flow-ema-ckpt",
        default=None,
        help="SS flow EMA checkpoint (auto-detect if not set)",
    )
    parser.add_argument(
        "--slat-flow-ckpt",
        default=None,
        help="Fine-tuned SLat flow denoiser checkpoint (.pt).",
    )
    parser.add_argument(
        "--pretrained-slat-flow",
        action="store_true",
        help="Explicitly use the base SLat flow from the pipeline config.",
    )
    parser.add_argument("--pipeline-config", default="pretrained_ckpts/pipeline.json")
    parser.add_argument("--summary-name", default="summary.json",
                        help="Run summary filename inside --output-dir (one per concurrent shard).")
    parser.add_argument(
        "--decoder-config",
        default=None,
        help="Decoder training config JSON. If not set, auto-resolved from --cola3d-ckpt run dir config.json",
    )
    parser.add_argument("--ss-flow-config", default="configs/train/finetune/ss_flow.json")
    parser.add_argument(
        "--dataset-class",
        choices=["flexi_dual_grid_multi_obj", "single_image", "benchmark_views"],
        default="flexi_dual_grid_multi_obj",
        help="Processed scenes, one image, or benchmark views.",
    )
    parser.add_argument("--benchmark-root", help="Root of the benchmark asset package.")
    parser.add_argument("--benchmark-dataset", choices=["InstPIFu", "MetaScenes", "MetaScenes_Hard", "Replica", "ScanNetpp"])
    parser.add_argument(
        "--dataset-instances",
        default=None,
        help="Path to instance list .txt (required for --dataset-class flexi_dual_grid_multi_obj)",
    )
    parser.add_argument(
        "--input-rgb",
        default=None,
        help="RGB image path for --dataset-class single_image. Requires --input-mask.",
    )
    parser.add_argument(
        "--input-mask",
        default=None,
        help="Mask path for --dataset-class single_image. Use with --input-rgb.",
    )
    parser.add_argument(
        "--input-rgba",
        default=None,
        help="RGBA image path for --dataset-class single_image. Mutually exclusive with --input-rgb/--input-mask; alpha becomes a single foreground mask id.",
    )
    parser.add_argument(
        "--input-prompt-mask-dir",
        default=None,
        help="Optional directory of object_*_mask.png files for single-image prompt-mask inference. Masks stay independent, so overlaps are preserved.",
    )
    parser.add_argument(
        "--input-id",
        default=None,
        help="Optional scene id / output directory name for --dataset-class single_image. Defaults to the input image stem.",
    )
    parser.add_argument(
        "--input-mask-mode",
        choices=["auto", "binary", "instance", "palette"],
        default="auto",
        help="How to decode --input-mask. auto: low-cardinality grayscale or RGB colors become instance ids, high-cardinality soft masks become one binary foreground; binary: all foreground is id 1; instance: remap non-zero grayscale labels; palette: remap non-black RGB colors.",
    )
    parser.add_argument(
        "--input-mask-threshold",
        type=int,
        default=0,
        help="Foreground threshold in [0,255] for binary/soft masks and RGBA alpha. Default 0 mirrors the existing non-zero foreground convention.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset roots JSON string (auto-detected if not set)",
    )
    parser.add_argument(
        "--render-idx",
        type=int,
        default=1,
        help="Fixed render view index for cond image (default: 1, i.e. render_0001). Set -1 for random.",
    )
    parser.add_argument(
        "--prompt-point-source",
        choices=["rebuild", "gt", "gt_native", "mask_moge2", "manual"],
        default="rebuild",
        help="Where prompt points come from for query_source='prompt_point' decoders. Ignored by fixed and prompt_mask decoders.",
    )
    parser.add_argument(
        "--prompt-point-rebuild-fallback",
        choices=["dilate", "nearest"],
        default="dilate",
        help="Fallback strategy when rebuilding prompt points on predicted coords.",
    )
    parser.add_argument(
        "--prompt-point-dilate-max",
        type=int,
        default=2,
        help="Maximum dilation radius for prompt-point rebuild fallback.",
    )
    parser.add_argument(
        "--prompt-point-align-pred",
        dest="prompt_point_align_pred",
        action="store_true",
        default=True,
        help="Align predicted coords to GT coords before rebuilding prompt points.",
    )
    parser.add_argument(
        "--no-prompt-point-align-pred",
        "--no-prompt-point-align",
        dest="prompt_point_align_pred",
        action="store_false",
        help="Disable prompt-point predicted-to-GT alignment.",
    )
    parser.add_argument(
        "--prompt-point-align-method",
        choices=["axis_search", "none"],
        default="axis_search",
        help="Alignment for dataset prompt-point reconstruction.",
    )
    parser.add_argument(
        "--prompt-point-selection",
        choices=["center", "lex"],
        default="center",
        help="How to select one prompt voxel from each rebuilt object set.",
    )
    parser.add_argument(
        "--prompt-point-snap-radius",
        type=int,
        default=2,
        help="Maximum nearest-neighbor snap radius for mask_moge2 prompt points.",
    )
    parser.add_argument(
        "--prepare-manual",
        action="store_true",
        help="Prepare cached SLats for later manual prompt-point annotation instead of running the final prompted decode.",
    )
    parser.add_argument(
        "--manual-prompt-points-from",
        default=None,
        help="Root containing per-scene manual_prompt_points.pt files. Defaults to --load-shape-slats-from.",
    )
    parser.add_argument(
        "--moge2-model-name",
        default="Ruicheng/moge-2-vitl-normal",
        help="MoGe2 model name used when --prompt-point-source=mask_moge2.",
    )
    parser.add_argument(
        "--moge2-input-max-side",
        type=int,
        default=1024,
        help="Maximum RGB side length passed to MoGe2 for mask_moge2 prompt points.",
    )
    parser.add_argument(
        "--shape-latent-source",
        choices=["flow", "gt"],
        default="flow",
        help="Source of the shape latent fed to CoLa3D and per-part decode. 'flow' (default): SS-flow + SLat-flow sampling chain. 'gt': dataset-emitted precomputed_z (GT shape encoder output) plus the matching precomputed_utonia npz — equivalent to the trainer snapshot input, so quality reflects CoLa3D/decoder only (no SLat-flow noise). Requires --data-dir with shape_latents and utonia_feats fields.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed for SS / SLat / Tex flow noise. Reseeded at the start of every scene (mirrors Trellis2ImageTo3DPipeline.run, which calls torch.manual_seed(seed) once per image). Default 42. Ignored entirely under --load-shape-slats-from since the cached SLats already encode the seed they were sampled with — pass the matching cached run instead of re-running diffusion.",
    )
    parser.add_argument(
        "--load-shape-slats-from",
        default=None,
        help="Run dir containing cached <sha256>/latents/shape_slat.pt (and optionally scene_tex_slat.pt). When set, skip SS-Flow + SLat-Flow and load the cached shape_slat instead — useful for re-running CoLa3D decode + mesh export (and texture decode) on top of a frozen geometry. With --with-texture, scene_tex_slat.pt is preferred but a missing cache falls back to sampling tex_slat fresh from the cached shape (use --skip-tex-flow-when-cached to disable that fallback).",
    )
    parser.add_argument(
        "--skip-tex-flow-when-cached",
        action="store_true",
        help="Under --load-shape-slats-from + --with-texture, refuse to fall back to sampling tex_slat fresh when scene_tex_slat.pt is missing from the cache (default: fall back). Use to enforce strict reuse of cached tex_slat (and save VRAM by skipping tex_flow_model load).",
    )
    parser.add_argument("--output-dir", default="outputs/decomposable_inference")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Mesh extraction resolution. Default: auto from decoder config (512 or 1024)",
    )
    parser.add_argument(
        "--ss-coord-resolution",
        type=int,
        default=None,
        help="SS coord target resolution. Default: auto from decoder config (32 or 64)",
    )
    parser.add_argument(
        "--slat-flow-resolution",
        type=str,
        default="auto",
        help="SLat flow model resolution: '512', '1024', or 'auto' (detect from decoder config)",
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index for dataset sharding (inclusive). For multi-GPU parallel runs.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index for dataset sharding (exclusive). If None, process to the end.",
    )
    parser.add_argument("--min-tokens", type=int, default=3)
    parser.add_argument("--assign-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--fragment-min-voxels", type=int, default=5)
    parser.add_argument("--fragment-ratio", type=float, default=0.05)
    parser.add_argument(
        "--min-object-voxels",
        type=int,
        default=20,
        help="Minimum voxel count for a query/CC to survive (smaller queries are removed as noise). Default 20 based on InstPIFu test-set object size distribution at 64^3 coord grid.",
    )
    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=0.8,
        help="Asymmetric containment threshold for mask_nms: a proposal is absorbed when intersection / min(|A|, |B|) exceeds this value. Default 0.8.",
    )
    parser.add_argument(
        "--postprocess-mode",
        type=str,
        default="standard",
        choices=["standard", "aggressive", "mask_nms", "both", "all"],
        help="standard: NMS + fragment filter. aggressive: single-CC + multi-label protection. mask_nms: instance-level NMS on per-query CCs. both: standard + aggressive side-by-side. all: standard + aggressive + mask_nms side-by-side.",
    )
    parser.add_argument(
        "--with-texture",
        action="store_true",
        help="Enable texture path: sample tex_slat once on the scene, then decode PBR per-part GLBs under texture/<variant>/.",
    )
    parser.add_argument(
        "--tex-flow-ckpt",
        default=None,
        help="Custom tex SLat flow denoiser checkpoint (.pt). If not set, uses pretrained from --pipeline-config.",
    )
    parser.add_argument(
        "--tex-decoder-ckpt",
        default=None,
        help="Custom tex decoder safetensors. If not set, uses pretrained.",
    )
    parser.add_argument(
        "--tex-texture-size",
        type=int,
        default=2048,
        help="UV bake resolution per textured GLB (default 2048).",
    )
    parser.add_argument(
        "--tex-decimation-target",
        type=int,
        default=200000,
        help="Decimation target passed to o_voxel.postprocess.to_glb (default 200k).",
    )
    parser.add_argument(
        "--skip-part-texture",
        action="store_true",
        help="With --with-texture still sample scene tex-SLat (needed for --online-utonia-rgb), but skip per-part tex decode and CuMesh GLB baking. Useful for fast runs where only geometry metrics are needed. Post-processing segmentation is still produced.",
    )
    parser.add_argument(
        "--skip-unseg-decode", action="store_true", help="Export only segmented meshes."
    )
    parser.add_argument(
        "--skip-raw-decode",
        action="store_true",
        help="Export only postprocessed segmentation meshes.",
    )
    parser.add_argument(
        "--online-utonia",
        action="store_true",
        help="For input_mode=latent_utonia_1024: extract Utonia level-4 features online (decode→1024³ voxelize→forward_level4) instead of loading precomputed dumps.",
    )
    parser.add_argument(
        "--online-utonia-rgb",
        action="store_true",
        help="When --online-utonia is set, also feed per-voxel base_color (from tex_slat_decoder) into Utonia. Requires --with-texture so a scene tex-SLat is sampled. Default off to keep the cheap geometry-only path.",
    )
    parser.add_argument(
        "--online-utonia-normal",
        action="store_true",
        help="Also feed per-voxel mesh normals into online Utonia. Off by default: the released heads were trained on XYZ-only Utonia features.",
    )
    return parser


def _parse_arguments(
    argv: Optional[List[str]] = None,
) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.geo_ckpt and (not args.pretrained_geo):
        parser.error("provide --geo-ckpt or --pretrained-geo")
    if (
        not args.load_shape_slats_from
        and args.shape_latent_source == "flow"
        and (not args.ss_flow_ckpt)
    ):
        parser.error("--ss-flow-ckpt is required for flow inference")
    if (
        not args.load_shape_slats_from
        and args.shape_latent_source == "flow"
        and (not (args.slat_flow_ckpt or args.pretrained_slat_flow))
    ):
        parser.error(
            "provide --slat-flow-ckpt, or explicitly choose --pretrained-slat-flow"
        )
    if args.slat_flow_ckpt and args.pretrained_slat_flow:
        parser.error(
            "--slat-flow-ckpt and --pretrained-slat-flow are mutually exclusive"
        )
    if args.prompt_point_align_method == "none":
        args.prompt_point_align_pred = False
    args.use_ss_flow = (
        not args.load_shape_slats_from and args.shape_latent_source == "flow"
    )
    if args.dataset_class == "flexi_dual_grid_multi_obj":
        if not args.dataset_instances:
            parser.error(
                "--dataset-instances is required when --dataset-class=flexi_dual_grid_multi_obj"
            )
    elif args.dataset_class == "benchmark_views":
        if not all((args.benchmark_root, args.benchmark_dataset)):
            parser.error("benchmark_views requires --benchmark-root and --benchmark-dataset")
    elif args.dataset_class == "single_image":
        if args.input_rgba:
            if args.input_rgb or args.input_mask:
                parser.error(
                    "--input-rgba is mutually exclusive with --input-rgb/--input-mask"
                )
        elif not (args.input_rgb and args.input_mask):
            parser.error(
                "--dataset-class=single_image requires either --input-rgba or both --input-rgb and --input-mask"
            )
    if args.prompt_point_source == "manual" and (not args.load_shape_slats_from):
        if args.prepare_manual:
            args.prompt_point_source = "gt"
        else:
            parser.error(
                "--prompt-point-source manual requires --load-shape-slats-from (manual annotations are bound to a specific cached run's SLat coords). For the first pass, use --prepare-manual to generate cached SLats."
            )
    if args.manual_prompt_points_from is None:
        args.manual_prompt_points_from = args.load_shape_slats_from
    return (parser, args)


_CLI_ARGUMENTS = _parse_arguments() if __name__ == "__main__" else None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from trellis2.modules import sparse as sp
from trellis2.representations import Mesh
from trellis2.utils.segment_postprocess import (
    postprocess_segmentation,
    postprocess_segmentation_aggressive,
    postprocess_segmentation_mask_nms,
)
from trellis2.utils.timing import stage_timer as _stage_timer
from trellis2.datasets.flexi_dual_grid_multi_obj import (
    FlexiDualGridMultiObjDataset,
    PROMPT_POINT_DOWNSAMPLE_RATIO,
)


class StageTimer:


    def __init__(self) -> None:
        self._t: Dict[str, float] = {}
        self._stack: List[Tuple[str, float]] = []

    def tic(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._stack.append((name, time.time()))

    def toc(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if not self._stack:
            return
        name, start = self._stack.pop()
        self._t[name] = round(self._t.get(name, 0.0) + (time.time() - start), 3)

    def as_dict(self) -> Dict[str, float]:
        return dict(self._t)


@dataclass
class SceneInferenceResult:
    sha256: str
    coords: torch.Tensor
    shape_slat: sp.SparseTensor
    img_feats: torch.Tensor
    multihot: torch.Tensor
    primary_labels: torch.Tensor
    num_parts: int
    part_slats: List[sp.SparseTensor] = field(default_factory=list)
    part_meshes: List[Mesh] = field(default_factory=list)
    raw_multihot: Optional[torch.Tensor] = None
    raw_primary_labels: Optional[torch.Tensor] = None
    raw_num_parts: int = 0
    raw_part_meshes: List[Mesh] = field(default_factory=list)
    agg_multihot: Optional[torch.Tensor] = None
    agg_primary_labels: Optional[torch.Tensor] = None
    agg_num_parts: int = 0
    agg_part_meshes: List[Mesh] = field(default_factory=list)
    mask_nms_multihot: Optional[torch.Tensor] = None
    mask_nms_primary_labels: Optional[torch.Tensor] = None
    mask_nms_num_parts: int = 0
    mask_nms_part_meshes: List[Mesh] = field(default_factory=list)
    unseg_full_mesh: Optional[Mesh] = None
    scene_tex_slat: Optional[sp.SparseTensor] = None
    textured_post_meshes: List[Any] = field(default_factory=list)
    textured_agg_meshes: List[Any] = field(default_factory=list)
    textured_mask_nms_meshes: List[Any] = field(default_factory=list)
    textured_unseg_mesh: Optional[Any] = None
    prompt_masks: Optional[torch.Tensor] = None
    raw_multihot_full: Optional[torch.Tensor] = None
    prompt_rebuild_info: Optional[dict] = None
    timings: Dict[str, float] = field(default_factory=dict)
    prepare_manual: bool = False


def load_decomposable_decoder(
    decoder_config: dict,
    cola3d_ckpt_path: str,
    geo_ckpt_path: str,
    pretrained_geo_safetensors: str = None,
    device: str = "cuda",
):

    from trellis2 import models

    config_args = dict(decoder_config["args"])
    config_args = {
        f"cola3d_{k[4:]}" if k.startswith("pat_") else k: v
        for k, v in config_args.items()
    }
    if pretrained_geo_safetensors:
        config_args.pop("use_dtype", None)
        config_args["use_fp16"] = True
        logger.debug(f"  Using pretrained geo safetensors -> forced fp16 dtype")
    model_cls = getattr(models, decoder_config["name"])
    decoder = model_cls(**config_args)
    cola3d_ckpt = torch.load(cola3d_ckpt_path, map_location="cpu")
    cola3d_state = {
        k: v
        for k, v in cola3d_ckpt.items()
        if k.startswith("pat.")
        or k.startswith("cola3d.")
        or k.startswith("mask_decoder.")
        or k.startswith("mask_prompt_encoder.")
        or k.startswith("point_prompt_encoder.")
    }
    if any((k.startswith("pat.") for k in cola3d_state)):
        cola3d_state = {
            f"cola3d.{k[4:]}" if k.startswith("pat.") else k: v
            for k, v in cola3d_state.items()
        }
        logger.debug("  Remapped pat.* -> cola3d.* (legacy checkpoint compat)")
    cnn_prefix = "mask_prompt_encoder.cnn."
    encoder_prefix = "mask_prompt_encoder.encoder."
    if any((k.startswith(cnn_prefix) for k in cola3d_state)):
        remapped = {}
        n_renamed = 0
        for k, v in cola3d_state.items():
            if k.startswith(cnn_prefix):
                remapped[encoder_prefix + k[len(cnn_prefix) :]] = v
                n_renamed += 1
            else:
                remapped[k] = v
        cola3d_state = remapped
        logger.debug(
            f"  Remapped {n_renamed} mask_prompt_encoder.cnn.* keys -> .encoder.* (divCenterPE ckpt compat)"
        )
    if pretrained_geo_safetensors:
        from safetensors.torch import load_file as load_safetensors

        geo_ckpt = load_safetensors(pretrained_geo_safetensors, device="cpu")
        geo_label = f"pretrained safetensors ({Path(pretrained_geo_safetensors).name})"
    else:
        geo_ckpt = torch.load(geo_ckpt_path, map_location="cpu")
        geo_label = f"checkpoint ({Path(geo_ckpt_path).name})"
    merged = {}
    merged.update(geo_ckpt)
    merged.update(cola3d_state)
    from trellis2.utils.decoder_checkpoint import load_decoder_state_dict

    retained = load_decoder_state_dict(decoder, merged)
    if retained:
        logger.debug(
            f"  Retained {len(retained)} independently pretrained Utonia state entries"
        )
    decoder.to(device)
    decoder.eval()
    logger.debug(
        f"  Loaded DecomposableVaeDecoder: {len(cola3d_state)} CoLa3D keys + {len(geo_ckpt)} geo keys ({geo_label})"
    )
    return decoder


def _prepare_single_prompt_batch(
    prompt_tensor: Optional[torch.Tensor],
    prompt_num_queries: Optional[Any],
    max_queries: int,
    device: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    if prompt_tensor is None or prompt_num_queries is None:
        return (None, None)
    if not isinstance(prompt_tensor, torch.Tensor):
        raise TypeError(f"Expected prompt tensor, got {type(prompt_tensor)!r}")
    if prompt_tensor.ndim == 2:
        prompt_tensor = prompt_tensor.unsqueeze(0)
    elif prompt_tensor.ndim == 3 and prompt_tensor.shape[-1] != 3:
        prompt_tensor = prompt_tensor.unsqueeze(0)
    available_queries = prompt_tensor.shape[1]
    if isinstance(prompt_num_queries, torch.Tensor):
        requested_queries = int(prompt_num_queries.reshape(-1)[0].item())
    else:
        requested_queries = int(prompt_num_queries)
    num_valid = max(0, min(requested_queries, available_queries, max_queries))
    padded = torch.zeros(
        (prompt_tensor.shape[0], max_queries, *prompt_tensor.shape[2:]),
        dtype=prompt_tensor.dtype,
        device=device,
    )
    valid_mask = torch.zeros(
        (prompt_tensor.shape[0], max_queries), dtype=torch.bool, device=device
    )
    if num_valid > 0:
        padded[:, :num_valid] = prompt_tensor[:, :num_valid].to(device)
        valid_mask[:, :num_valid] = True
    return (padded, valid_mask)


def prepare_decoder_prompt_kwargs(
    decoder, data: Optional[dict], device: str
) -> Dict[str, Optional[torch.Tensor]]:

    query_source = getattr(decoder, "query_source", "fixed")
    if query_source == "fixed":
        return {"prompt_masks": None, "prompt_points": None, "prompt_valid_mask": None}
    if data is None:
        raise RuntimeError(
            f"{query_source} decoder requires dataset prompt inputs, but data is None"
        )
    if query_source == "prompt_mask":
        prompt_num_queries = data.get(
            "prompt_mask_num_queries", data.get("prompt_num_queries")
        )
        prompt_masks, valid_mask = _prepare_single_prompt_batch(
            data.get("prompt_masks"), prompt_num_queries, decoder.max_parts, device
        )
        if prompt_masks is None or valid_mask is None:
            raise RuntimeError(
                "prompt_mask decoder requires prompt_masks and prompt_num_queries in dataset sample"
            )
        return {
            "prompt_masks": prompt_masks,
            "prompt_points": None,
            "prompt_valid_mask": valid_mask,
        }
    if query_source == "prompt_point":
        prompt_num_queries = data.get(
            "prompt_point_num_queries", data.get("prompt_num_queries")
        )
        prompt_points, valid_mask = _prepare_single_prompt_batch(
            data.get("prompt_points"), prompt_num_queries, decoder.max_parts, device
        )
        if prompt_points is None or valid_mask is None:
            raise RuntimeError(
                "prompt_point decoder requires prompt_points and prompt_num_queries in dataset sample"
            )
        return {
            "prompt_masks": None,
            "prompt_points": prompt_points.long(),
            "prompt_valid_mask": valid_mask,
        }
    raise ValueError(f"Unsupported query_source: {query_source}")


def _snap_prompt_points_to_latent(
    decoder,
    latent: sp.SparseTensor,
    prompt_points: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:

    latent_res = int(decoder.latent_resolution)
    snapped = prompt_points.clone()
    total_snapped = 0
    max_shift = 0
    for b in range(prompt_points.shape[0]):
        sample_valid = valid_mask[b].bool()
        if not sample_valid.any():
            continue
        start, stop = (latent.layout[b].start, latent.layout[b].stop)
        coords_b = latent.coords[start:stop, 1:4].to(
            dtype=prompt_points.dtype, device=prompt_points.device
        )
        pts = prompt_points[b, sample_valid]
        base = latent_res * latent_res
        keys_latent = (
            coords_b[:, 0] * base + coords_b[:, 1] * latent_res + coords_b[:, 2]
        )
        keys_pts = pts[:, 0] * base + pts[:, 1] * latent_res + pts[:, 2]
        sorted_keys, _ = torch.sort(keys_latent)
        hit_pos = torch.searchsorted(sorted_keys, keys_pts)
        in_range = hit_pos < sorted_keys.numel()
        hit_pos_clamped = hit_pos.clamp_max(sorted_keys.numel() - 1)
        matched = in_range & (sorted_keys[hit_pos_clamped] == keys_pts)
        missing_mask = ~matched
        if not missing_mask.any():
            continue
        miss_pts = pts[missing_mask]
        diffs = coords_b.unsqueeze(0) - miss_pts.unsqueeze(1)
        nearest_idx = diffs.abs().sum(dim=-1).argmin(dim=-1)
        nearest_coords = coords_b[nearest_idx]
        shift = int((miss_pts - nearest_coords).abs().max().item())
        max_shift = max(max_shift, shift)
        total_snapped += int(missing_mask.sum().item())
        valid_indices = torch.where(sample_valid)[0]
        snapped[b, valid_indices[missing_mask]] = nearest_coords
    if total_snapped > 0:
        logger.debug(
            f"  [prompt-point snap] {total_snapped} points → nearest occupied latent coord (max_shift={max_shift} voxels)",
        )
    return snapped


def _dilate_keys(
    seed_coords: torch.Tensor, radius: int, latent_res: int
) -> torch.Tensor:

    if radius <= 0 or seed_coords.numel() == 0:
        return FlexiDualGridMultiObjDataset._encode_grid_keys(seed_coords, latent_res)
    side = 2 * radius + 1
    offsets = (
        torch.stack(
            torch.meshgrid(
                torch.arange(-radius, radius + 1, device=seed_coords.device),
                torch.arange(-radius, radius + 1, device=seed_coords.device),
                torch.arange(-radius, radius + 1, device=seed_coords.device),
                indexing="ij",
            ),
            dim=-1,
        )
        .reshape(-1, 3)
        .long()
    )
    expanded = (seed_coords.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)
    in_range = (expanded >= 0).all(dim=-1) & (expanded < latent_res).all(dim=-1)
    expanded = expanded[in_range]
    keys = FlexiDualGridMultiObjDataset._encode_grid_keys(expanded, latent_res)
    return torch.unique(keys, sorted=True)


def _align_gt_to_pred(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    resolution: int,
    method: str = "axis_search",
) -> Tuple[Optional[np.ndarray], dict]:

    info: dict = {
        "used": False,
        "method": method,
        "iou_before": None,
        "iou_after": None,
        "transform_norm": None,
        "error": None,
    }
    if method == "none":
        return (None, info)
    if method != "axis_search":
        raise ValueError("Dataset prompt alignment must be axis_search or none")
    try:
        from trellis2.utils.ss_flow_eval_utils import compute_voxel_metrics
    except Exception as e:
        info["error"] = f"import: {e}"
        return (None, info)
    pred_np = pred_xyz.detach().cpu().numpy().astype(np.int64)
    gt_np = gt_xyz.detach().cpu().numpy().astype(np.int64)
    if pred_np.size == 0 or gt_np.size == 0:
        info["error"] = "empty input"
        return (None, info)
    pred_occ = np.zeros((resolution,) * 3, dtype=bool)
    pred_occ[pred_np[:, 0], pred_np[:, 1], pred_np[:, 2]] = True
    gt_occ = np.zeros((resolution,) * 3, dtype=bool)
    gt_occ[gt_np[:, 0], gt_np[:, 1], gt_np[:, 2]] = True
    iou_before = float(compute_voxel_metrics(pred_occ, gt_occ)["iou"])
    info["iou_before"] = iou_before
    from itertools import permutations

    rots = []
    seen = set()
    for perm in permutations(range(3)):
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    M = np.zeros((3, 3))
                    M[0, perm[0]] = sx
                    M[1, perm[1]] = sy
                    M[2, perm[2]] = sz
                    if np.isclose(np.linalg.det(M), 1.0):
                        key = tuple(M.flatten().tolist())
                        if key not in seen:
                            seen.add(key)
                            rots.append(M)
    pc = pred_np.astype(np.float64).mean(axis=0)
    gc = gt_np.astype(np.float64).mean(axis=0)
    best_iou = -1.0
    best_T_gt2pred = None
    best_rot_idx = -1
    for idx, R in enumerate(rots):
        T_test = np.eye(4)
        T_test[:3, :3] = R
        T_test[:3, 3] = pc - R @ gc
        pts_h = np.concatenate(
            [gt_np.astype(np.float64) + 0.5, np.ones((len(gt_np), 1))], axis=1
        )
        transformed = (T_test @ pts_h.T).T[:, :3]
        idx_t = np.clip(np.round(transformed - 0.5).astype(np.int64), 0, resolution - 1)
        tmp_occ = np.zeros_like(pred_occ)
        tmp_occ[idx_t[:, 0], idx_t[:, 1], idx_t[:, 2]] = True
        iou_t = float(compute_voxel_metrics(tmp_occ, pred_occ)["iou"])
        if iou_t > best_iou:
            best_iou = iou_t
            best_T_gt2pred = T_test
            best_rot_idx = idx
    T_inv = best_T_gt2pred
    info["best_rot_idx"] = best_rot_idx
    info["iou_after"] = best_iou
    info["transform_norm"] = float(np.linalg.norm(T_inv - np.eye(4)))
    info["used"] = True
    return (T_inv, info)


def _apply_T_to_int_coords(
    coords: torch.Tensor, T: np.ndarray, resolution: int
) -> torch.Tensor:

    if coords.numel() == 0:
        return coords
    pts = coords.detach().cpu().numpy().astype(np.float64) + 0.5
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    transformed = (T @ pts_h.T).T[:, :3]
    idx = np.round(transformed - 0.5).astype(np.int64)
    idx = np.clip(idx, 0, resolution - 1)
    return torch.from_numpy(idx).to(coords.device).to(coords.dtype)


def _rebuild_prompt_points_on_pred(
    decoder,
    latent: sp.SparseTensor,
    data: dict,
    *,
    fallback: str = "dilate",
    dilate_max: int = 2,
    align_pred: bool = False,
    align_method: str = "axis_search",
    selection: str = "center",
) -> Tuple[torch.Tensor, torch.Tensor, dict]:

    latent_res = int(decoder.latent_resolution)
    max_parts = int(decoder.max_parts)
    device = latent.coords.device
    vertices = data["vertices"]
    if not isinstance(vertices, sp.SparseTensor):
        raise TypeError(f"data['vertices'] expected SparseTensor, got {type(vertices)}")
    assert (
        vertices.coords[:, 0] == 0
    ).all(), "_rebuild_prompt_points expects B=1 in eval"
    scene_coords = vertices.coords[:, 1:4].to(torch.int64).to(device)
    instance_id = data["instance_id"].to(torch.int64).to(device)
    num_objects = int(data["num_objects"])
    if num_objects > max_parts:
        raise RuntimeError(
            f"num_objects={num_objects} exceeds decoder.max_parts={max_parts}"
        )
    assert len(latent.layout) == 1, "_rebuild_prompt_points expects single-batch latent"
    start, stop = (latent.layout[0].start, latent.layout[0].stop)
    pred_xyz = latent.coords[start:stop, 1:4].to(torch.int64)
    if pred_xyz.numel() == 0:
        empty_pp = torch.zeros(1, max_parts, 3, dtype=torch.long, device=device)
        empty_v = torch.zeros(1, max_parts, dtype=torch.bool, device=device)
        info = dict(
            num_objects=num_objects,
            num_valid=0,
            dropped=list(range(num_objects)),
            dilated_radius=[0] * num_objects,
            matched_per_obj=[0] * num_objects,
            collisions=0,
            fallback=fallback,
        )
        print(
            "  [prompt-point rebuild] WARNING: shape_slat empty, all prompts dropped",
            flush=True,
        )
        return (empty_pp, empty_v, info)
    assert (
        pred_xyz.max().item() < latent_res
    ), f"pred coords OOB: max={pred_xyz.max().item()} >= latent_res={latent_res}"
    pred_keys = FlexiDualGridMultiObjDataset._encode_grid_keys(pred_xyz, latent_res)
    sorted_pred, sorted_pred_order = torch.sort(pred_keys)
    ds_scene_coords = (scene_coords // PROMPT_POINT_DOWNSAMPLE_RATIO).clamp_(
        0, latent_res - 1
    )
    align_info: dict = {"used": False}
    ds_scene_pre_align = ds_scene_coords
    if align_pred:
        if "precomputed_z" in data and data["precomputed_z"] is not None:
            gt_lat_xyz = data["precomputed_z"].coords[:, 1:4].to(torch.int64).to(device)
        else:
            gt_lat_xyz = torch.unique(ds_scene_coords, dim=0)
        T_inv, align_info = _align_gt_to_pred(
            pred_xyz=pred_xyz,
            gt_xyz=gt_lat_xyz,
            resolution=latent_res,
            method=align_method,
        )
        if T_inv is not None:
            ds_scene_coords = _apply_T_to_int_coords(ds_scene_coords, T_inv, latent_res)
            align_info["T_inv"] = T_inv.tolist()
    scene_keys = FlexiDualGridMultiObjDataset._encode_grid_keys(
        ds_scene_coords, latent_res
    )
    total_latent_cells = latent_res**3
    pair_keys = instance_id * total_latent_cells + scene_keys
    unique_pair_keys = torch.unique(pair_keys, sorted=True)
    unique_object_ids = torch.div(
        unique_pair_keys, total_latent_cells, rounding_mode="floor"
    )
    unique_scene_keys = unique_pair_keys % total_latent_cells
    chosen_xyz: List[torch.Tensor] = []
    valid_obj_ids: List[int] = []
    chosen_keys: set = set()
    matched_per_obj: List[int] = []
    dropped: List[int] = []
    dilated_radius: List[int] = [0] * num_objects
    collisions = 0
    for obj_id in range(num_objects):
        obj_mask = unique_object_ids == obj_id
        obj_scene_keys = unique_scene_keys[obj_mask]
        cand_keys = _intersect_with_pred(obj_scene_keys, sorted_pred)
        matched_per_obj.append(int(cand_keys.numel()))
        if cand_keys.numel() == 0 and fallback == "dilate":
            obj_seed_coords = ds_scene_coords[instance_id == obj_id]
            obj_seed_coords = torch.unique(obj_seed_coords, dim=0)
            for r in range(1, dilate_max + 1):
                dilated_keys = _dilate_keys(obj_seed_coords, r, latent_res)
                cand_keys = _intersect_with_pred(dilated_keys, sorted_pred)
                if cand_keys.numel() > 0:
                    dilated_radius[obj_id] = r
                    break
        if cand_keys.numel() == 0:
            dropped.append(obj_id)
            continue
        if selection == "center" and cand_keys.numel() > 1:
            cand_x = (cand_keys // (latent_res * latent_res)).to(torch.float32)
            cand_y = (cand_keys // latent_res % latent_res).to(torch.float32)
            cand_z = (cand_keys % latent_res).to(torch.float32)
            cx, cy, cz = (cand_x.mean(), cand_y.mean(), cand_z.mean())
            d2 = (cand_x - cx) ** 2 + (cand_y - cy) ** 2 + (cand_z - cz) ** 2
            order = torch.argsort(d2).tolist()
        else:
            order = torch.argsort(cand_keys).tolist()
        chosen_pred_idx = -1
        for pos in order:
            k = int(cand_keys[pos].item())
            if k in chosen_keys:
                continue
            sp_pos = int(torch.searchsorted(sorted_pred, cand_keys[pos]).item())
            chosen_pred_idx = int(sorted_pred_order[sp_pos].item())
            chosen_keys.add(k)
            break
        if chosen_pred_idx < 0:
            collisions += 1
            sp_pos = int(torch.searchsorted(sorted_pred, cand_keys[order[0]]).item())
            chosen_pred_idx = int(sorted_pred_order[sp_pos].item())
        chosen_xyz.append(pred_xyz[chosen_pred_idx])
        valid_obj_ids.append(obj_id)
    num_valid = len(valid_obj_ids)
    if num_valid > 0:
        prompt_points = torch.stack(chosen_xyz, dim=0).unsqueeze(0).long().to(device)
        valid_mask = torch.ones(1, num_valid, dtype=torch.bool, device=device)
    else:
        prompt_points = torch.zeros(1, 1, 3, dtype=torch.long, device=device)
        valid_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    if num_valid > 0:
        chosen = prompt_points[0]
        chosen_lin = FlexiDualGridMultiObjDataset._encode_grid_keys(chosen, latent_res)
        hit_pos = torch.searchsorted(sorted_pred, chosen_lin)
        in_range = hit_pos < sorted_pred.numel()
        clamped = hit_pos.clamp_max(sorted_pred.numel() - 1)
        ok = in_range & (sorted_pred[clamped] == chosen_lin)
        assert (
            ok.all().item()
        ), f"_rebuild_prompt_points self-check failed: {(~ok).sum().item()} valid prompt(s) not in pred coords (rebuild logic bug)"
    info = dict(
        num_objects=num_objects,
        num_valid=num_valid,
        dropped=dropped,
        valid_obj_ids=valid_obj_ids,
        dilated_radius=dilated_radius,
        matched_per_obj=matched_per_obj,
        collisions=collisions,
        fallback=fallback,
        align=align_info,
    )
    parts = [f"valid={num_valid}/{num_objects}"]
    if dropped:
        parts.append(f"dropped={dropped}")
    dilated_objs = [(i, r) for i, r in enumerate(dilated_radius) if r > 0]
    if dilated_objs:
        parts.append(
            "dilated=[" + ",".join((f"{i}:r={r}" for i, r in dilated_objs)) + "]"
        )
    parts.append(f"matched_per_obj={matched_per_obj}")
    if collisions:
        parts.append(f"collisions={collisions}")
    parts.append(f"fallback={fallback}")
    if align_info.get("used"):
        parts.append(
            f"align(IoU {align_info.get('iou_before', 0):.2f}→{align_info.get('iou_after', 0):.2f},|T-I|={align_info.get('transform_norm', 0):.1f})"
        )
    elif align_pred:
        parts.append(f"align=skipped({align_info.get('error', '?')})")
    logger.debug(f"  [prompt-point rebuild] {' '.join(parts)}")
    return (prompt_points, valid_mask, info)


def _intersect_with_pred(
    query_keys: torch.Tensor, sorted_pred: torch.Tensor
) -> torch.Tensor:

    if query_keys.numel() == 0:
        return query_keys
    hit_pos = torch.searchsorted(sorted_pred, query_keys)
    in_range = hit_pos < sorted_pred.numel()
    clamped = hit_pos.clamp_max(sorted_pred.numel() - 1)
    matched = in_range & (sorted_pred[clamped] == query_keys)
    return query_keys[matched]


_PART_COLOR_HEX = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
]


def _build_prompt_points_from_mask_moge2(
    decoder,
    latent: sp.SparseTensor,
    data: dict,
    *,
    moge_model,
    align: bool,
    align_method: str,
    snap_radius: int,
    moge_input_max_side: int,
    pixel_mode: str = "hybrid",
    raycast_tolerance: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:

    from trellis2.utils.prompt_point_moge2 import build_prompt_points_from_mask

    if moge_model is None:
        raise RuntimeError(
            "prompt_point_source=mask_moge2 requires --prompt-point-source mask_moge2 to load MoGe2; main() should have populated moge_model before calling here."
        )
    if "raw_rgb" not in data or "instance_mask_full" not in data:
        raise RuntimeError(
            "mask_moge2 strategy needs data['raw_rgb'] and data['instance_mask_full']; use a dataset that provides raw RGB and full-resolution instance masks."
        )
    assert len(latent.layout) == 1, "mask_moge2 expects single-batch latent"
    start, stop = (latent.layout[0].start, latent.layout[0].stop)
    pred_xyz = latent.coords[start:stop, 1:4].to(torch.int64)
    result = build_prompt_points_from_mask(
        raw_rgb=data["raw_rgb"],
        instance_mask=data["instance_mask_full"],
        pred_coords=pred_xyz,
        moge_model=moge_model,
        device=latent.coords.device,
        align=align,
        align_method=align_method,
        snap_radius=snap_radius,
        moge_input_max_side=moge_input_max_side,
        max_parts=int(decoder.max_parts),
        pixel_mode=pixel_mode,
        raycast_tolerance=raycast_tolerance,
    )
    info = result.info
    parts = [f"valid={info['num_valid']}/{info['num_objects']}"]
    if info["dropped"]:
        parts.append(f"dropped={info['dropped']}")
    nonzero_dilations = [(i, r) for i, r in enumerate(info["dilated_radius"]) if r > 0]
    if nonzero_dilations:
        parts.append(
            "snap_dist=[" + ",".join((f"{i}:{r}" for i, r in nonzero_dilations)) + "]"
        )
    if info["collisions"]:
        parts.append(f"collisions={info['collisions']}")
    align_blk = info.get("align", {})
    if align_blk.get("used"):
        parts.append(
            f"align(NN {align_blk.get('mean_nn_before', 0):.2f}→{align_blk.get('mean_nn_after_icp', 0):.2f},|T-I|={align_blk.get('transform_norm', 0):.1f})"
        )
    elif align and align_blk.get("error"):
        parts.append(f"align=skipped({align_blk['error']})")
    logger.debug(f"  [prompt-mask-moge2] {' '.join(parts)}")
    return (result.prompt_points, result.valid_mask, info)


def run_prompted_cola3d_head(
    decoder,
    latent: sp.SparseTensor,
    img_features: Optional[torch.Tensor],
    data: Optional[dict],
    ext_features: Optional[sp.SparseTensor] = None,
    *,
    prompt_strategy: str = "rebuild",
    rebuild_fallback: str = "dilate",
    rebuild_dilate_max: int = 2,
    rebuild_align_pred: bool = False,
    rebuild_align_method: str = "axis_search",
    rebuild_selection: str = "center",
    moge_model=None,
    moge2_align: bool = True,
    moge2_align_method: str = "fpfh_ransac",
    moge2_snap_radius: int = 2,
    moge2_input_max_side: int = 1024,
    moge2_pixel_mode: str = "hybrid",
    moge2_raycast_tolerance: int = 1,
    manual_points_path: Optional[Path] = None,
    scene_dir: Optional[Path] = None,
) -> Tuple[
    sp.SparseTensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[dict]
]:

    latent_cola3d = latent.replace(latent.feats.to(decoder.cola3d_dtype))
    rebuild_info: Optional[dict] = None
    query_source = getattr(decoder, "query_source", "fixed")
    use_constructed_prompt_points = (
        query_source == "prompt_point"
        and prompt_strategy in {"rebuild", "mask_moge2", "manual"}
    )
    if use_constructed_prompt_points:
        prompt_kwargs: Dict[str, Optional[torch.Tensor]] = {
            "prompt_masks": None,
            "prompt_points": None,
            "prompt_valid_mask": None,
        }
    else:
        prompt_kwargs = prepare_decoder_prompt_kwargs(
            decoder, data=data, device=latent_cola3d.device
        )
    if query_source == "prompt_point":
        if prompt_strategy == "rebuild":
            new_pp, new_vm, rebuild_info = _rebuild_prompt_points_on_pred(
                decoder,
                latent=latent_cola3d,
                data=data,
                fallback=rebuild_fallback,
                dilate_max=rebuild_dilate_max,
                align_pred=rebuild_align_pred,
                align_method=rebuild_align_method,
                selection=rebuild_selection,
            )
            prompt_kwargs["prompt_points"] = new_pp
            prompt_kwargs["prompt_valid_mask"] = new_vm
        elif prompt_strategy == "mask_moge2":
            new_pp, new_vm, rebuild_info = _build_prompt_points_from_mask_moge2(
                decoder,
                latent=latent_cola3d,
                data=data,
                moge_model=moge_model,
                align=moge2_align,
                align_method=moge2_align_method,
                snap_radius=moge2_snap_radius,
                moge_input_max_side=moge2_input_max_side,
                pixel_mode=moge2_pixel_mode,
                raycast_tolerance=moge2_raycast_tolerance,
            )
            prompt_kwargs["prompt_points"] = new_pp
            prompt_kwargs["prompt_valid_mask"] = new_vm
        elif prompt_strategy == "manual":
            if manual_points_path is None:
                raise ValueError("prompt_strategy='manual' requires manual_points_path")
            mp = Path(manual_points_path)
            if not mp.is_file():
                raise FileNotFoundError(
                    f"manual prompt points file missing: {mp}. Run scripts/inference/ui_manual_prompt_points.py to create it."
                )
            blob = torch.load(mp, map_location=latent_cola3d.device)
            new_pp = blob["prompt_points"].to(latent_cola3d.device).long()
            new_vm = blob["valid_mask"].to(latent_cola3d.device).bool()
            if new_pp.dim() != 3 or new_pp.shape[-1] != 3:
                raise ValueError(
                    f"manual prompt_points expected (B, Q, 3), got {tuple(new_pp.shape)}"
                )
            if new_vm.shape != new_pp.shape[:2]:
                raise ValueError(
                    f"manual valid_mask {tuple(new_vm.shape)} mismatch with prompt_points {tuple(new_pp.shape)}"
                )
            num_valid = int(new_vm.sum().item())
            metadata = blob.get("metadata", {}) or {}
            rebuild_info = {
                "strategy": "manual",
                "num_objects": num_valid,
                "num_valid": num_valid,
                "dropped": [],
                "dilated_radius": [],
                "matched_per_obj": [],
                "collisions": 0,
                "valid_obj_ids": list(blob.get("object_ids", []) or []),
                "fallback": "n/a",
                "manual_metadata": metadata,
                "manual_points_path": str(mp),
            }
            logger.debug(f"  [manual] loaded {num_valid} prompt points from {mp}")
            prompt_kwargs["prompt_points"] = new_pp
            prompt_kwargs["prompt_valid_mask"] = new_vm
        elif prompt_strategy == "gt":
            if prompt_kwargs["prompt_points"] is not None:
                prompt_kwargs["prompt_points"] = _snap_prompt_points_to_latent(
                    decoder,
                    latent=latent_cola3d,
                    prompt_points=prompt_kwargs["prompt_points"],
                    valid_mask=prompt_kwargs["prompt_valid_mask"],
                )
        elif prompt_strategy == "gt_native":
            pass
        else:
            raise ValueError(f"unknown prompt_strategy={prompt_strategy!r}")
    prefused_input = None
    prompt_latent_source = latent_cola3d
    if (
        query_source == "prompt_point"
        and getattr(decoder, "point_feature_source", "latent") == "latent_utonia"
    ):
        prefused_input = decoder.cola3d.compute_input_features(
            latent_cola3d, ext_features
        )
        prompt_latent_source = prefused_input
    prompt_queries, prompt_valid, prompt_q_coords = decoder._encode_prompt_queries(
        latent=prompt_latent_source,
        prompt_masks=prompt_kwargs["prompt_masks"],
        prompt_points=prompt_kwargs["prompt_points"],
        prompt_valid_mask=prompt_kwargs["prompt_valid_mask"],
    )
    p_enriched, query_feats = decoder.cola3d(
        latent_cola3d,
        img_features=img_features,
        ext_features=ext_features,
        prompt_queries=prompt_queries,
        valid_query_mask=prompt_valid,
        prompt_q_coords=prompt_q_coords,
        prefused_input=prefused_input,
    )
    mask_logits = decoder.mask_decoder(
        p_enriched, query_feats, valid_query_mask=prompt_valid
    )
    return (p_enriched, query_feats, mask_logits, prompt_valid, rebuild_info)


def load_ss_flow_model(
    ss_flow_config: dict,
    ckpt_path: str,
    ema_ckpt_path: str = None,
    device: str = "cuda",
):

    from trellis2 import models

    model_cls = getattr(models, ss_flow_config["name"])
    model = model_cls(**ss_flow_config["args"])
    weights_path = (
        ema_ckpt_path if ema_ckpt_path and Path(ema_ckpt_path).exists() else ckpt_path
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.convert_to(torch.bfloat16)
    model.to(device)
    model.eval()
    logger.debug(f"  Loaded {ss_flow_config['name']} from {Path(weights_path).name} (bf16)")
    return model


def load_slat_flow_model(
    pipeline_config: dict,
    custom_ckpt: str = None,
    resolution_key: str = "512",
    device: str = "cuda",
):

    from trellis2 import models
    from safetensors.torch import load_file as load_safetensors

    slat_model_key = f"shape_slat_flow_model_{resolution_key}"
    model_path = str(
        Path("pretrained_ckpts") / pipeline_config["args"]["models"][slat_model_key]
    )
    config_path = f"{model_path}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model_cls = getattr(models, config["name"])
    model = model_cls(**config["args"])
    if custom_ckpt:
        state_dict = torch.load(custom_ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        label = f"custom: {Path(custom_ckpt).name}"
    else:
        weights_path = f"{model_path}.safetensors"
        state_dict = load_safetensors(weights_path, device="cpu")
        model.load_state_dict(state_dict)
        label = "pretrained"
    model.to(device)
    model.eval()
    logger.debug(f"  Loaded {config['name']} ({label} SLat flow)")
    return model


def load_tex_flow_model(
    pipeline_config: dict,
    resolution_key: str = "1024",
    custom_ckpt: str = None,
    device: str = "cuda",
):

    from trellis2 import models
    from safetensors.torch import load_file as load_safetensors

    tex_model_key = f"tex_slat_flow_model_{resolution_key}"
    model_path = str(
        Path("pretrained_ckpts") / pipeline_config["args"]["models"][tex_model_key]
    )
    config_path = f"{model_path}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model_cls = getattr(models, config["name"])
    model = model_cls(**config["args"])
    if custom_ckpt:
        state_dict = torch.load(custom_ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        label = f"custom: {Path(custom_ckpt).name}"
    else:
        weights_path = f"{model_path}.safetensors"
        state_dict = load_safetensors(weights_path, device="cpu")
        model.load_state_dict(state_dict)
        label = "pretrained"
    model.to(device)
    model.eval()
    logger.debug(f"  Loaded {config['name']} ({label} tex flow @ {resolution_key})")
    return model


def load_tex_decoder(
    pipeline_config: dict, custom_ckpt: str = None, device: str = "cuda"
):

    from trellis2 import models
    from safetensors.torch import load_file as load_safetensors

    model_path = str(
        Path("pretrained_ckpts") / pipeline_config["args"]["models"]["tex_slat_decoder"]
    )
    config_path = f"{model_path}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model_cls = getattr(models, config["name"])
    model = model_cls(**config["args"])
    weights_path = custom_ckpt or f"{model_path}.safetensors"
    state_dict = load_safetensors(weights_path, device="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    label = "custom" if custom_ckpt else "pretrained"
    logger.debug(f"  Loaded {config['name']} ({label} tex decoder)")
    return model


def load_tex_sampler(pipeline_config: dict):

    from trellis2.pipelines import samplers

    tex_cfg = pipeline_config["args"]["tex_slat_sampler"]
    tex_sampler = getattr(samplers, tex_cfg["name"])(**tex_cfg["args"])
    tex_params = dict(tex_cfg["params"])
    shape_norm = pipeline_config["args"]["shape_slat_normalization"]
    tex_norm = pipeline_config["args"]["tex_slat_normalization"]
    logger.debug(f"  Loaded tex sampler: {tex_cfg['name']}")
    return (tex_sampler, tex_params, shape_norm, tex_norm)


def load_dino_model(pipeline_config: dict, device: str = "cuda"):

    from trellis2.modules import image_feature_extractor
    import os
    import glob

    img_cond_cfg = pipeline_config["args"]["image_cond_model"]
    dino_cls = getattr(image_feature_extractor, img_cond_cfg["name"])
    dino_args = dict(img_cond_cfg.get("args", {}))
    try:
        dino_model = dino_cls(**dino_args)
    except (OSError, Exception) as e:
        model_name = dino_args.get("model_name", "")
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        repo_dir = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        snapshots = glob.glob(os.path.join(repo_dir, "snapshots", "*"))
        if snapshots:
            local_path = sorted(snapshots)[-1]
            print(
                f"  [INFO] HF hub failed ({type(e).__name__}), using local cache: {local_path}"
            )
            dino_args["model_name"] = local_path
            dino_model = dino_cls(**dino_args)
        else:
            raise
    dino_model.to(device)
    logger.debug(f"  Loaded {img_cond_cfg['name']}")
    return dino_model


def load_ss_decoder(device: str = "cuda", model_path: Optional[str] = None):

    from trellis2 import models

    model_path = (
        model_path or "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
    )
    try:
        ss_dec = models.from_pretrained(model_path)
    except Exception as exc:
        if Path(model_path).is_absolute():
            raise
        import glob

        pattern = str(
            Path.home()
            / ".cache/huggingface/hub/models--*TRELLIS-image-large/snapshots/*"
            / "ckpts/ss_dec_conv3d_16l8_fp16.json"
        )
        cands = sorted(glob.glob(pattern), key=lambda p: ("microsoft--" not in p, p))
        if not cands:
            raise
        local_path = cands[0][: -len(".json")]
        print(
            f"  [ss_dec] HF hub failed ({exc.__class__.__name__}), using local snapshot: {local_path}"
        )
        ss_dec = models.from_pretrained(local_path)
    ss_dec.to(device)
    ss_dec.eval()
    logger.debug(f"  Loaded SS decoder (pretrained)")
    return ss_dec


def load_samplers(pipeline_config: dict):

    from trellis2.pipelines import samplers

    ss_cfg = pipeline_config["args"]["sparse_structure_sampler"]
    ss_sampler = getattr(samplers, ss_cfg["name"])(**ss_cfg["args"])
    ss_params = ss_cfg["params"]
    slat_cfg = pipeline_config["args"]["shape_slat_sampler"]
    slat_sampler = getattr(samplers, slat_cfg["name"])(**slat_cfg["args"])
    slat_params = slat_cfg["params"]
    norm = pipeline_config["args"]["shape_slat_normalization"]
    norm_mean = torch.tensor(norm["mean"], dtype=torch.float32)
    norm_std = torch.tensor(norm["std"], dtype=torch.float32)
    logger.debug(f"  Loaded samplers: {ss_cfg['name']}, {slat_cfg['name']}")
    return (ss_sampler, ss_params, slat_sampler, slat_params, norm_mean, norm_std)


def sample_ss_from_flow(
    ss_flow_model, ss_decoder, ss_sampler, ss_params, img_feats, target_resolution=32
):

    reso = ss_flow_model.resolution
    in_channels = ss_flow_model.in_channels
    noise = torch.randn(1, in_channels, reso, reso, reso, device=img_feats.device)
    cond = {"cond": img_feats, "neg_cond": torch.zeros_like(img_feats)}
    z_s = ss_sampler.sample(
        ss_flow_model, noise, **cond, **ss_params, verbose=False, tqdm_desc="Sampling SS"
    ).samples
    decoded = ss_decoder(z_s) > 0
    if target_resolution != decoded.shape[2]:
        ratio = decoded.shape[2] // target_resolution
        decoded = F.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    if coords.shape[0] == 0:
        raise RuntimeError("SS flow produced empty occupancy; no scene can be decoded")
    return coords


def load_cached_shape_slat(
    load_dir: Path, instance_id: str, device: str = "cuda"
) -> sp.SparseTensor:

    p = Path(load_dir) / instance_id / "latents" / "shape_slat.pt"
    if not p.is_file():
        raise FileNotFoundError(f"cached shape_slat not found at {p}")
    blob = torch.load(p, map_location="cpu")
    feats = blob["feats"].to(device)
    coords = blob["coords"].to(device).contiguous()
    if coords.shape[1] != 4:
        raise ValueError(
            f"cached shape_slat coords expected (N, 4) [b,x,y,z], got {tuple(coords.shape)}"
        )
    if not (coords[:, 0] == 0).all():
        raise ValueError(
            f"cached shape_slat coords[:, 0] (batch idx) must all be 0; got {coords[:, 0].unique().tolist()}"
        )
    return sp.SparseTensor(feats=feats, coords=coords)


def load_cached_scene_tex_slat(
    load_dir: Path, instance_id: str, device: str = "cuda"
) -> Optional[sp.SparseTensor]:

    p = Path(load_dir) / instance_id / "latents" / "scene_tex_slat.pt"
    if not p.is_file():
        return None
    blob = torch.load(p, map_location="cpu")
    feats = blob["feats"].to(device)
    coords = blob["coords"].to(device).contiguous()
    return sp.SparseTensor(feats=feats, coords=coords)


def encode_prompt_for_eval(decoder, data, device, ext_features=None):

    qs = getattr(decoder, "query_source", "fixed")
    if qs not in ("prompt_mask", "prompt_point"):
        return (None, None, None)
    if qs == "prompt_mask":
        pm = data.get("prompt_masks")
        if pm is None:
            raise RuntimeError(
                "decoder.query_source='prompt_mask' but data['prompt_masks'] is None. Provide per-instance masks through the selected dataset."
            )
        if pm.ndim != 3:
            raise ValueError(
                f"prompt_masks must have shape (Q, H, W); got {tuple(pm.shape)}"
            )
        pm = pm.to(device=device, dtype=decoder.cola3d_dtype).unsqueeze(0)
        valid = torch.ones(1, pm.shape[1], dtype=torch.bool, device=device)
        latent_stub = data.get("shape_slat")
        prompt_queries, prompt_valid, prompt_q_coords = decoder._encode_prompt_queries(
            latent=latent_stub,
            prompt_masks=pm,
            prompt_points=None,
            prompt_valid_mask=valid,
        )
        return (prompt_queries, prompt_valid, prompt_q_coords)
    pp = data.get("prompt_points")
    if pp is None:
        raise RuntimeError(
            "decoder.query_source='prompt_point' but data['prompt_points'] is None. Provide prompt points via dataset emit or external prompt source."
        )
    if pp.ndim != 2 or pp.shape[-1] != 3:
        raise ValueError(f"prompt_points must have shape (Q, 3); got {tuple(pp.shape)}")
    pp = pp.to(device=device, dtype=torch.long).unsqueeze(0)
    valid = torch.ones(1, pp.shape[1], dtype=torch.bool, device=device)
    latent = data.get("shape_slat")
    if latent is None:
        raise RuntimeError(
            "decoder.query_source='prompt_point' requires data['shape_slat'] for latent feature gathering at the prompt coords."
        )
    prompt_latent_source = latent
    if getattr(decoder, "point_feature_source", "latent") == "latent_utonia":
        if ext_features is None:
            raise RuntimeError(
                "point_feature_source='latent_utonia' requires ext_features (fused Utonia feats) to encode prompt-point queries; pass it from the caller's Utonia prep stage via encode_prompt_for_eval(..., ext_features=...)."
            )
        latent_cola3d = latent.replace(latent.feats.to(decoder.cola3d_dtype))
        prompt_latent_source = decoder.cola3d.compute_input_features(
            latent_cola3d, ext_features
        )
    prompt_queries, prompt_valid, prompt_q_coords = decoder._encode_prompt_queries(
        latent=prompt_latent_source,
        prompt_masks=None,
        prompt_points=pp,
        prompt_valid_mask=valid,
    )
    return (prompt_queries, prompt_valid, prompt_q_coords)


def save_scene_latent_cache(
    *,
    scene_dir: Path,
    shape_slat: sp.SparseTensor,
    scene_tex_slat: Optional[sp.SparseTensor] = None,
) -> None:

    latents_dir = scene_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"feats": shape_slat.feats.cpu(), "coords": shape_slat.coords.cpu()},
        str(latents_dir / "shape_slat.pt"),
    )
    if scene_tex_slat is not None:
        torch.save(
            {
                "feats": scene_tex_slat.feats.cpu(),
                "coords": scene_tex_slat.coords.cpu(),
            },
            str(latents_dir / "scene_tex_slat.pt"),
        )


def scene_is_completed(scene_info_path: Path, *, prepare_manual: bool = False) -> bool:

    if not scene_info_path.exists():
        return False
    if prepare_manual:
        return True
    try:
        with open(scene_info_path, "r") as f:
            info = json.load(f)
    except Exception:
        return True
    return info.get("mode") != "prepare_manual"


PART_COLORS = torch.tensor(
    [
        [0.12, 0.47, 0.71],
        [1.0, 0.5, 0.05],
        [0.17, 0.63, 0.17],
        [0.84, 0.15, 0.16],
        [0.58, 0.4, 0.74],
        [0.55, 0.34, 0.29],
        [0.89, 0.47, 0.76],
        [0.5, 0.5, 0.5],
        [0.74, 0.74, 0.13],
        [0.09, 0.75, 0.81],
    ],
    dtype=torch.float32,
)


def _sum_numeric_timings(timings: Dict[str, Any]) -> float:

    total = 0.0
    for value in timings.values():
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def save_mesh_glb(mesh: Mesh, path: str, vertex_colors=None):

    import trimesh

    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    colors = vertex_colors
    if (
        colors is None
        and hasattr(mesh, "vertex_attrs")
        and (mesh.vertex_attrs is not None)
        and (mesh.vertex_attrs.shape[0] > 0)
    ):
        colors = mesh.vertex_attrs
    kwargs = {}
    if colors is not None:
        colors_np = colors.cpu().numpy() if isinstance(colors, torch.Tensor) else colors
        alpha = np.ones((colors_np.shape[0], 1), dtype=colors_np.dtype)
        rgba = np.concatenate([colors_np, alpha], axis=-1)
        kwargs["vertex_colors"] = (rgba * 255).astype(np.uint8)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, **kwargs)
    tm.export(path)


def merge_colored_parts(
    part_meshes: list, colors: torch.Tensor = None
) -> Optional[Mesh]:

    if colors is None:
        colors = PART_COLORS
    colors = colors.cpu()
    vertices_list = []
    faces_list = []
    colors_list = []
    vertex_offset = 0
    for part_idx, mesh in enumerate(part_meshes):
        if mesh is None:
            continue
        v = mesh.vertices.detach().cpu()
        f = mesh.faces.detach().cpu()
        if v.shape[0] == 0 or f.shape[0] == 0:
            continue
        vertices_list.append(v)
        faces_list.append(f + vertex_offset)
        color = colors[part_idx % len(colors)]
        colors_list.append(color.unsqueeze(0).expand(v.shape[0], 3))
        vertex_offset += v.shape[0]
    if not vertices_list:
        return None
    return Mesh(
        vertices=torch.cat(vertices_list, dim=0),
        faces=torch.cat(faces_list, dim=0),
        vertex_attrs=torch.cat(colors_list, dim=0),
    )


def _decode_parts_from_multihot(
    multihot: torch.Tensor,
    shape_slat: sp.SparseTensor,
    decoder,
    min_tokens: int,
    label: str = "",
    scene_tex_slat: Optional[sp.SparseTensor] = None,
    tex_decoder: Optional[nn.Module] = None,
    resolution: int = 1024,
    timings: Optional[dict] = None,
    timing_prefix: Optional[str] = None,
) -> Tuple[List[Optional[sp.SparseTensor]], List[Optional[Mesh]], List[Optional[Any]]]:

    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
    from trellis2.models.sc_vaes.sparse_unet_vae import EmptySubdivisionError
    from trellis2.pipelines.scene_texturing import decode_tex_voxel, wrap_textured_mesh

    z_start = shape_slat.layout[0].start
    z_end = shape_slat.layout[0].stop
    num_parts = multihot.shape[1]
    part_slats: List[Optional[sp.SparseTensor]] = []
    part_meshes: List[Optional[Mesh]] = []
    part_textured: List[Optional[Any]] = []
    slat_device = shape_slat.feats.device
    texture_enabled = scene_tex_slat is not None and tex_decoder is not None
    if texture_enabled:
        assert (
            scene_tex_slat.feats.shape[0] == shape_slat.feats.shape[0]
        ), "scene_tex_slat must share coords with shape_slat"
        tex_z_start = scene_tex_slat.layout[0].start
        tex_z_end = scene_tex_slat.layout[0].stop
    per_part_times: List[Optional[float]] = []
    n_tokens_list: List[Optional[int]] = []
    empty_part_ids: List[int] = []
    for part_id in range(num_parts):
        part_mask = multihot[:, part_id].to(slat_device)
        n_selected = int(part_mask.sum().item())
        if n_selected < min_tokens:
            logger.debug(f"  {label}Part {part_id}: {n_selected} tokens < {min_tokens} — SKIP")
            part_slats.append(None)
            part_meshes.append(None)
            part_textured.append(None)
            per_part_times.append(None)
            n_tokens_list.append(n_selected)
            continue
        part_feats = shape_slat.feats[z_start:z_end][part_mask]
        part_coords = shape_slat.coords[z_start:z_end][part_mask].clone()
        part_coords[:, 0] = 0
        part_slat = sp.SparseTensor(feats=part_feats, coords=part_coords)
        part_slat._scale = shape_slat._scale
        part_slats.append(part_slat)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t_part = time.time()
            if texture_enabled:
                result = FlexiDualGridVaeDecoder.forward(
                    decoder, part_slat, return_subs=True
                )
                if isinstance(result, tuple) and len(result) == 2:
                    mesh_list, subs = result
                else:
                    mesh_list, subs = (result, None)
            else:
                mesh_list = FlexiDualGridVaeDecoder.forward(decoder, part_slat)
                subs = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            per_part_times.append(round(time.time() - _t_part, 3))
            n_tokens_list.append(n_selected)
            part_slat.clear_spatial_cache()
            if not (
                isinstance(mesh_list, list)
                and len(mesh_list) > 0
                and (mesh_list[0] is not None)
            ):
                part_meshes.append(None)
                part_textured.append(None)
                print(f"  {label}Part {part_id}: empty mesh")
                continue
            mesh = mesh_list[0]
            part_meshes.append(mesh)
            logger.debug(
                f"  {label}Part {part_id}: {n_selected} tokens -> {mesh.vertices.shape[0]} verts"
            )
            if texture_enabled and subs is not None and (mesh.vertices.shape[0] > 0):
                part_tex_feats = scene_tex_slat.feats[tex_z_start:tex_z_end][part_mask]
                part_tex_coords = scene_tex_slat.coords[tex_z_start:tex_z_end][
                    part_mask
                ].clone()
                part_tex_coords[:, 0] = 0
                part_tex_slat = sp.SparseTensor(
                    feats=part_tex_feats, coords=part_tex_coords
                )
                part_tex_slat._scale = scene_tex_slat._scale
                try:
                    pbr_voxel = decode_tex_voxel(part_tex_slat, subs, tex_decoder)
                    part_tex_slat.clear_spatial_cache()
                    textured = wrap_textured_mesh(mesh, pbr_voxel, resolution)
                    part_textured.append(textured)
                    logger.debug(
                        f"  {label}Part {part_id}: textured ({pbr_voxel.feats.shape[0]} tex voxels)"
                    )
                except Exception as tex_e:
                    part_textured.append(None)
                    print(f"  {label}Part {part_id}: tex decode failed — {tex_e}")
            else:
                part_textured.append(None)
        except EmptySubdivisionError as e:
            part_meshes.append(None)
            part_textured.append(None)
            if len(per_part_times) < len(part_meshes):
                per_part_times.append(None)
                n_tokens_list.append(n_selected)
            empty_part_ids.append(part_id)
            print(f"  {label}Part {part_id}: {n_selected} tokens decode to empty geometry ({e}) — SKIP")
        except Exception as e:
            part_meshes.append(None)
            part_textured.append(None)
            if len(per_part_times) < len(part_meshes):
                per_part_times.append(None)
                n_tokens_list.append(n_selected)
            print(f"  {label}Part {part_id}: decode failed — {e}")
            if torch.cuda.is_available():

                torch.cuda.synchronize()
    if timings is not None and timing_prefix is not None:
        timings[f"{timing_prefix}_per_part_s"] = per_part_times
        timings[f"{timing_prefix}_per_part_tokens"] = n_tokens_list
        timings[f"{timing_prefix}_empty_part_ids"] = empty_part_ids
    return (part_slats, part_meshes, part_textured)


def run_single_scene(
    sha256: str,
    cond_image: torch.Tensor,
    dino_model,
    ss_flow_model,
    ss_decoder,
    slat_flow_model,
    decoder,
    ss_sampler,
    ss_params,
    slat_sampler,
    slat_params,
    norm_mean,
    norm_std,
    args,
    data: dict = None,
    tex_flow_model=None,
    tex_decoder=None,
    tex_sampler=None,
    tex_sampler_params: Optional[dict] = None,
    shape_slat_norm_dict: Optional[dict] = None,
    tex_slat_norm_dict: Optional[dict] = None,
    online_utonia_extractor=None,
    moge_model=None,
    scene_dir: Optional[Path] = None,
) -> SceneInferenceResult:

    device = "cuda"
    timer = StageTimer()
    timings = timer._t
    prompt_rebuild_info: Optional[dict] = None
    raw_multihot_full = None
    with torch.no_grad():
        _scene_seed = int(getattr(args, "seed", 42))
        torch.manual_seed(_scene_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_scene_seed)
        with _stage_timer(timings, "dino"):
            slat_cond_res = (
                int(args.slat_flow_resolution)
                if args.slat_flow_resolution != "512"
                else 512
            )
            dino_model.image_size = 512
            img_feats_512 = dino_model(cond_image.unsqueeze(0).to(device))
            if slat_cond_res > 512:
                dino_model.image_size = slat_cond_res
                img_feats_slat = dino_model(cond_image.unsqueeze(0).to(device))
                dino_model.image_size = 512
                logger.debug(
                    f"  DINO features: 512 -> {img_feats_512.shape[1]} patches, {slat_cond_res} -> {img_feats_slat.shape[1]} patches"
                )
            else:
                img_feats_slat = img_feats_512
            img_feats = img_feats_512
            slat_cond = {
                "cond": img_feats_slat,
                "neg_cond": torch.zeros_like(img_feats_slat),
            }
        use_cached_slat = bool(getattr(args, "load_shape_slats_from", None))
        use_gt_latent = (
            not use_cached_slat and getattr(args, "shape_latent_source", "flow") == "gt"
        )
        if use_cached_slat:
            shape_slat = load_cached_shape_slat(
                Path(args.load_shape_slats_from), sha256, device=device
            )
            coords = shape_slat.coords
            with _stage_timer(timings, "ss_sample"):
                pass
            with _stage_timer(timings, "slat_sample"):
                pass
        elif use_gt_latent:
            pz = (data or {}).get("precomputed_z")
            if pz is None:
                raise RuntimeError(
                    "--shape-latent-source=gt requires data['precomputed_z']; rebuild dataset_args with shape_latent_root pointing at the Test-side shape_latents/* dir."
                )
            shape_slat = pz
            shape_slat = shape_slat.replace(shape_slat.feats.to(device))
            shape_slat.coords = shape_slat.coords.to(device)
            coords = shape_slat.coords
            logger.debug(
                f"  GT shape latent: {shape_slat.feats.shape} (precomputed_z, SS+SLat-flow skipped)"
            )
        else:
            with _stage_timer(timings, "ss_sample"):
                coords = sample_ss_from_flow(
                    ss_flow_model,
                    ss_decoder,
                    ss_sampler,
                    ss_params,
                    img_feats,
                    args.ss_coord_resolution,
                )
                logger.debug(f"  SS coords: {coords.shape[0]} voxels")
            with _stage_timer(timings, "slat_sample"):
                noise = sp.SparseTensor(
                    feats=torch.randn(
                        coords.shape[0], slat_flow_model.in_channels, device=device
                    ),
                    coords=coords,
                )
                shape_slat = slat_sampler.sample(
                    slat_flow_model,
                    noise,
                    **slat_cond,
                    **slat_params,
                    verbose=False,
                    tqdm_desc="Sampling SLat",
                ).samples
                shape_slat = shape_slat.replace(
                    shape_slat.feats * norm_std.to(device) + norm_mean.to(device)
                )
                logger.debug(f"  Shape latent: {shape_slat.feats.shape}")
        scene_tex_slat = None
        if args.with_texture:
            from trellis2.pipelines.scene_texturing import sample_scene_tex_slat

            with _stage_timer(timings, "tex_slat_sample"):
                if use_cached_slat:
                    scene_tex_slat = load_cached_scene_tex_slat(
                        Path(args.load_shape_slats_from), sha256, device=device
                    )
                    if scene_tex_slat is None:
                        if getattr(args, "skip_tex_flow_when_cached", False):
                            raise RuntimeError(
                                f"--with-texture set, cached scene_tex_slat.pt missing under {args.load_shape_slats_from}/{sha256}/latents/, AND --skip-tex-flow-when-cached was passed. Drop --skip-tex-flow-when-cached or rerun the baseline with --with-texture."
                            )
                        assert (
                            tex_flow_model is not None and tex_decoder is not None
                        ), "--with-texture requires tex_flow_model + tex_decoder for fallback path"
                        print(
                            f"  [cache] scene_tex_slat.pt missing -> sampling tex_slat fresh from cached shape_slat"
                        )
                        scene_tex_slat = sample_scene_tex_slat(
                            cond=slat_cond,
                            tex_flow_model=tex_flow_model,
                            shape_slat=shape_slat,
                            shape_slat_normalization=shape_slat_norm_dict,
                            tex_slat_normalization=tex_slat_norm_dict,
                            sampler=tex_sampler,
                            sampler_params=tex_sampler_params,
                            verbose=False,
                        )
                        assert torch.equal(
                            scene_tex_slat.coords, shape_slat.coords
                        ), "freshly-sampled tex_slat must share coords with cached shape_slat"
                    else:
                        assert scene_tex_slat.coords.shape == shape_slat.coords.shape
                    logger.debug(
                        f"  [cache] scene_tex_slat: feats={tuple(scene_tex_slat.feats.shape)}"
                    )
                else:
                    assert (
                        tex_flow_model is not None and tex_decoder is not None
                    ), "--with-texture requires tex_flow_model + tex_decoder"
                    scene_tex_slat = sample_scene_tex_slat(
                        cond=slat_cond,
                        tex_flow_model=tex_flow_model,
                        shape_slat=shape_slat,
                        shape_slat_normalization=shape_slat_norm_dict,
                        tex_slat_normalization=tex_slat_norm_dict,
                        sampler=tex_sampler,
                        sampler_params=tex_sampler_params,
                        verbose=False,
                    )
                    assert scene_tex_slat.coords.shape == shape_slat.coords.shape
                    assert torch.equal(
                        scene_tex_slat.coords, shape_slat.coords
                    ), "tex_slat must share coords with shape_slat"
                    logger.debug(f"  Scene tex latent: {scene_tex_slat.feats.shape}")
        if getattr(args, "prepare_manual", False):
            save_scene_latent_cache(
                scene_dir=scene_dir if scene_dir is not None else Path("."),
                shape_slat=shape_slat,
                scene_tex_slat=scene_tex_slat,
            )
            n_latent = shape_slat.layout[0].stop - shape_slat.layout[0].start
            return SceneInferenceResult(
                sha256=sha256,
                coords=shape_slat.coords,
                shape_slat=shape_slat,
                img_feats=img_feats,
                multihot=torch.zeros(
                    (n_latent, 0), dtype=torch.bool, device=shape_slat.feats.device
                ),
                primary_labels=torch.zeros(
                    (n_latent,), dtype=torch.long, device=shape_slat.feats.device
                ),
                num_parts=0,
                scene_tex_slat=scene_tex_slat,
                timings=timings,
                prepare_manual=True,
            )
        x_cola3d = shape_slat.replace(shape_slat.feats.to(decoder.cola3d_dtype))
        ext_features = None
        _input_mode = getattr(decoder, "input_mode", "latent")
        if _input_mode in ("utonia", "latent_utonia"):
            with _stage_timer(timings, "utonia"):
                utonia_feats = decoder.utonia_extractor(
                    coords=x_cola3d.coords, batch_size=x_cola3d.shape[0]
                )
                ext_features = x_cola3d.replace(utonia_feats.to(decoder.cola3d_dtype))
        elif _input_mode == "latent_utonia_1024":
            precomputed_utonia = (
                (data or {}).get("precomputed_utonia") if use_gt_latent else None
            )
            if precomputed_utonia is not None:
                with _stage_timer(timings, "utonia"):
                    pu_expected_scale = Fraction(16, 1)
                    if precomputed_utonia._scale[0] != pu_expected_scale:
                        precomputed_utonia._scale = tuple(
                            (pu_expected_scale for _ in range(3))
                        )
                    pu_feats = (
                        precomputed_utonia.feats
                        if isinstance(precomputed_utonia, sp.SparseTensor)
                        else precomputed_utonia
                    )
                    ext_features = x_cola3d.replace(
                        pu_feats.to(device=device, dtype=decoder.cola3d_dtype)
                    )
                    logger.debug(f"  ext_features: precomputed_utonia ({pu_feats.shape})")
            elif online_utonia_extractor is not None:
                with _stage_timer(timings, "utonia"):
                    from trellis2.utils.online_utonia import (
                        compute_online_utonia_features,
                    )

                    _use_rgb = getattr(args, "online_utonia_rgb", False)
                    _use_normal = getattr(args, "online_utonia_normal", False)
                    _tex_dec_kw = tex_decoder if _use_rgb else None
                    _tex_slat_kw = scene_tex_slat if _use_rgb else None
                    if _use_rgb and (_tex_dec_kw is None or _tex_slat_kw is None):
                        print(
                            "  [online-utonia][WARN] --online-utonia-rgb requested but tex_decoder or scene_tex_slat is None (did you pass --with-texture?) — falling back to no RGB."
                        )
                        _tex_dec_kw = None
                        _tex_slat_kw = None
                    online_ext = compute_online_utonia_features(
                        shape_slat=shape_slat,
                        decoder=decoder,
                        extractor=online_utonia_extractor,
                        tex_decoder=_tex_dec_kw,
                        tex_slat=_tex_slat_kw,
                        pbr_attr_layout=(
                            getattr(tex_decoder, "pbr_attr_layout", None)
                            if _use_rgb
                            else None
                        ),
                        enable_normal=_use_normal,
                        timings=timings,
                    )
                    ext_features = x_cola3d.replace(
                        online_ext.feats.to(decoder.cola3d_dtype)
                    )
            else:
                raise RuntimeError(
                    "decoder.input_mode='latent_utonia_1024' but neither data['precomputed_utonia'] (gt mode) nor an online extractor (flow mode) is available."
                )
        prompt_data = dict(data) if data is not None else {}
        prompt_data.setdefault("shape_slat", x_cola3d)
        if getattr(decoder, "query_source", "fixed") == "prompt_point":
            _src = getattr(args, "prompt_point_source", "rebuild")
            if _src == "rebuild":
                new_pp, new_vm, prompt_rebuild_info = _rebuild_prompt_points_on_pred(
                    decoder,
                    latent=x_cola3d,
                    data=prompt_data,
                    fallback=getattr(args, "prompt_point_rebuild_fallback", "dilate"),
                    dilate_max=getattr(args, "prompt_point_dilate_max", 2),
                    align_pred=getattr(args, "prompt_point_align_pred", True),
                    align_method=getattr(
                        args, "prompt_point_align_method", "axis_search"
                    ),
                    selection=getattr(args, "prompt_point_selection", "center"),
                )
                prompt_data["prompt_points"] = new_pp.squeeze(0)[
                    new_vm.squeeze(0).bool()
                ].long()
            elif _src == "mask_moge2":
                new_pp, new_vm, prompt_rebuild_info = (
                    _build_prompt_points_from_mask_moge2(
                        decoder,
                        latent=x_cola3d,
                        data=prompt_data,
                        moge_model=moge_model,
                        align=getattr(args, "prompt_point_align_pred", True),
                        align_method=getattr(
                            args, "prompt_point_align_method", "axis_search"
                        ),
                        snap_radius=getattr(args, "prompt_point_snap_radius", 2),
                        moge_input_max_side=getattr(args, "moge2_input_max_side", 1024),
                    )
                )
                prompt_data["prompt_points"] = new_pp.squeeze(0)[
                    new_vm.squeeze(0).bool()
                ].long()
            elif _src == "manual":
                manual_root = getattr(args, "manual_prompt_points_from", None)
                if not manual_root:
                    raise ValueError(
                        "prompt_point_source=manual requires --manual-prompt-points-from"
                    )
                manual_path = Path(manual_root) / sha256 / "manual_prompt_points.pt"
                if not manual_path.exists():
                    raise FileNotFoundError(
                        f"manual prompt file not found: {manual_path}. Run scripts/inference/ui_manual_prompt_points.py to create it."
                    )
                blob = torch.load(manual_path, map_location="cpu")
                _pp = blob["prompt_points"]
                _vm = blob.get("valid_mask")
                if _pp.ndim != 3 or _pp.shape[0] != 1 or _pp.shape[-1] != 3:
                    raise ValueError(
                        f"manual prompt_points expected (1, Q, 3), got {tuple(_pp.shape)}"
                    )
                if _vm is not None:
                    _pp = _pp[:, _vm.bool().view(-1)]
                prompt_data["prompt_points"] = _pp.squeeze(0).long()
                prompt_rebuild_info = {
                    "strategy": "manual",
                    "num_objects": int(prompt_data["prompt_points"].shape[0]),
                    "num_valid": int(prompt_data["prompt_points"].shape[0]),
                    "manual_path": str(manual_path),
                    "dropped": [],
                }
        prompt_queries, prompt_valid, prompt_q_coords = encode_prompt_for_eval(
            decoder, prompt_data, device, ext_features=ext_features
        )
        with _stage_timer(timings, "cola3d_forward"):
            p_enriched, query_feats = decoder.cola3d(
                x_cola3d,
                img_features=img_feats,
                ext_features=ext_features,
                prompt_queries=prompt_queries,
                valid_query_mask=prompt_valid,
                prompt_q_coords=prompt_q_coords,
            )
            mask_logits = decoder.mask_decoder(
                p_enriched, query_feats, valid_query_mask=prompt_valid
            )
        n_points = shape_slat.layout[0].stop - shape_slat.layout[0].start
        logits_b0 = mask_logits[0, :n_points]
        coords_b0 = shape_slat.coords[
            shape_slat.layout[0].start : shape_slat.layout[0].stop, 1:
        ]
        with _stage_timer(timings, "seg_raw"):
            from trellis2.utils.segment_postprocess import (
                multihot_assign,
                compact_queries,
            )

            raw_probs = torch.sigmoid(logits_b0)
            raw_multihot_full = multihot_assign(
                raw_probs, threshold=args.assign_threshold
            )
            raw_masked_probs = raw_probs.clone()
            raw_masked_probs[~raw_multihot_full] = -1.0
            raw_primary_all = raw_masked_probs.argmax(dim=1)
            raw_multihot, raw_primary_labels, raw_num_parts, _ = compact_queries(
                raw_multihot_full, raw_primary_all
            )
            logger.debug(f"  Raw segmentation: {raw_num_parts} parts (no post-processing)")
        decoder.set_resolution(args.resolution)
        raw_part_meshes: List[Optional[Mesh]] = []
        if getattr(args, "skip_raw_decode", False):
            logger.debug("  [skip] raw decode (--skip-raw-decode)")
            timings["decode_raw"] = 0.0
        else:
            with _stage_timer(timings, "decode_raw"):
                logger.debug("  --- Raw decode ---")
                _, raw_part_meshes, _ = _decode_parts_from_multihot(
                    raw_multihot,
                    shape_slat,
                    decoder,
                    args.min_tokens,
                    label="[raw] ",
                    timings=timings,
                    timing_prefix="decode_raw",
                )
        mode = args.postprocess_mode
        run_standard = mode in ("standard", "both", "all")
        run_aggressive = mode in ("aggressive", "both", "all")
        run_mask_nms = mode in ("mask_nms", "all")
        agg_multihot, agg_primary_labels, agg_num_parts = (None, None, 0)
        agg_part_meshes = []
        mask_nms_multihot, mask_nms_primary_labels, mask_nms_num_parts = (None, None, 0)
        mask_nms_part_meshes = []
        _st_mh = _st_pl = _st_np = None
        _agg_mh = _agg_pl = _agg_np = None
        _mn_mh = _mn_pl = _mn_np = None
        with _stage_timer(timings, "seg_postprocess"):
            if run_standard:
                _st_mh, _st_pl, _st_np = postprocess_segmentation(
                    logits_b0,
                    coords_b0,
                    assign_threshold=args.assign_threshold,
                    nms_iou_threshold=args.nms_iou_threshold,
                    fragment_min_voxels=args.fragment_min_voxels,
                    fragment_ratio=args.fragment_ratio,
                    min_object_voxels=args.min_object_voxels,
                )
                logger.debug(f"  Post-processed segmentation (standard): {_st_np} parts")
            if run_aggressive:
                _agg_mh, _agg_pl, _agg_np = postprocess_segmentation_aggressive(
                    logits_b0,
                    coords_b0,
                    assign_threshold=args.assign_threshold,
                    nms_iou_threshold=args.nms_iou_threshold,
                    min_object_voxels=args.min_object_voxels,
                )
                logger.debug(f"  Post-processed segmentation (aggressive): {_agg_np} parts")
            if run_mask_nms:
                _mn_mh, _mn_pl, _mn_np = postprocess_segmentation_mask_nms(
                    logits_b0,
                    coords_b0,
                    assign_threshold=args.assign_threshold,
                    min_object_voxels=args.min_object_voxels,
                    containment_threshold=args.containment_threshold,
                )
                logger.debug(f"  Post-processed segmentation (mask_nms): {_mn_np} parts")
        if mode == "aggressive":
            multihot, primary_labels, num_parts = (_agg_mh, _agg_pl, _agg_np)
        elif mode == "mask_nms":
            multihot, primary_labels, num_parts = (_mn_mh, _mn_pl, _mn_np)
        else:
            multihot, primary_labels, num_parts = (_st_mh, _st_pl, _st_np)
        if mode in ("both", "all"):
            agg_multihot, agg_primary_labels, agg_num_parts = (
                _agg_mh,
                _agg_pl,
                _agg_np,
            )
        if mode == "all":
            mask_nms_multihot, mask_nms_primary_labels, mask_nms_num_parts = (
                _mn_mh,
                _mn_pl,
                _mn_np,
            )
        _part_tex_slat = (
            None if getattr(args, "skip_part_texture", False) else scene_tex_slat
        )
        _part_tex_dec = (
            None if getattr(args, "skip_part_texture", False) else tex_decoder
        )
        with _stage_timer(timings, "decode_post"):
            logger.debug("  --- Post-processed decode ---")
            part_slats, part_meshes, textured_post_meshes = _decode_parts_from_multihot(
                multihot,
                shape_slat,
                decoder,
                args.min_tokens,
                label="",
                scene_tex_slat=_part_tex_slat,
                tex_decoder=_part_tex_dec,
                resolution=args.resolution,
                timings=timings,
                timing_prefix="decode_post",
            )
        textured_agg_meshes: List[Optional[Any]] = []
        if mode in ("both", "all") and agg_num_parts > 0:
            with _stage_timer(timings, "decode_aggressive"):
                logger.debug("  --- Aggressive decode ---")
                _, agg_part_meshes, textured_agg_meshes = _decode_parts_from_multihot(
                    agg_multihot,
                    shape_slat,
                    decoder,
                    args.min_tokens,
                    label="[agg] ",
                    scene_tex_slat=_part_tex_slat,
                    tex_decoder=_part_tex_dec,
                    resolution=args.resolution,
                    timings=timings,
                    timing_prefix="decode_aggressive",
                )
        textured_mask_nms_meshes: List[Optional[Any]] = []
        if mode == "all" and mask_nms_num_parts > 0:
            with _stage_timer(timings, "decode_mask_nms"):
                logger.debug("  --- Mask-NMS decode ---")
                _, mask_nms_part_meshes, textured_mask_nms_meshes = (
                    _decode_parts_from_multihot(
                        mask_nms_multihot,
                        shape_slat,
                        decoder,
                        args.min_tokens,
                        label="[mask_nms] ",
                        scene_tex_slat=_part_tex_slat,
                        tex_decoder=_part_tex_dec,
                        resolution=args.resolution,
                        timings=timings,
                        timing_prefix="decode_mask_nms",
                    )
                )
        unseg_full_mesh = None
        textured_unseg_mesh: Optional[Any] = None
        if getattr(args, "skip_unseg_decode", False):
            logger.debug("  [skip] unsegmented full scene decode (--skip-unseg-decode)")
            timings["decode_unseg"] = 0.0
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t_unseg = time.time()
            try:
                from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder

                unseg_slat = sp.SparseTensor(
                    feats=shape_slat.feats.contiguous(),
                    coords=shape_slat.coords.contiguous(),
                )
                if (
                    args.with_texture
                    and scene_tex_slat is not None
                    and (not getattr(args, "skip_part_texture", False))
                ):
                    from trellis2.pipelines.scene_texturing import (
                        decode_tex_voxel,
                        wrap_textured_mesh,
                    )

                    result = FlexiDualGridVaeDecoder.forward(
                        decoder, unseg_slat, return_subs=True
                    )
                    if isinstance(result, tuple) and len(result) == 2:
                        full_mesh_list, unseg_subs = result
                    else:
                        full_mesh_list, unseg_subs = (result, None)
                    unseg_slat.clear_spatial_cache()
                    if (
                        isinstance(full_mesh_list, list)
                        and full_mesh_list
                        and (full_mesh_list[0] is not None)
                    ):
                        unseg_full_mesh = full_mesh_list[0]
                        logger.debug(
                            f"  Unsegmented full scene: {unseg_full_mesh.vertices.shape[0]} verts"
                        )
                        if (
                            unseg_subs is not None
                            and unseg_full_mesh.vertices.shape[0] > 0
                        ):
                            try:
                                unseg_tex_slat = sp.SparseTensor(
                                    feats=scene_tex_slat.feats.contiguous(),
                                    coords=scene_tex_slat.coords.contiguous(),
                                )
                                unseg_tex_slat._scale = scene_tex_slat._scale
                                pbr_voxel = decode_tex_voxel(
                                    unseg_tex_slat, unseg_subs, tex_decoder
                                )
                                unseg_tex_slat.clear_spatial_cache()
                                textured_unseg_mesh = wrap_textured_mesh(
                                    unseg_full_mesh, pbr_voxel, args.resolution
                                )
                                logger.debug(
                                    f"  Unsegmented full scene textured ({pbr_voxel.feats.shape[0]} tex voxels)"
                                )
                            except Exception as tex_e:
                                print(
                                    f"  Unsegmented full scene tex decode failed: {tex_e}"
                                )
                else:
                    full_mesh_list = FlexiDualGridVaeDecoder.forward(
                        decoder, unseg_slat
                    )
                    unseg_slat.clear_spatial_cache()
                    if isinstance(full_mesh_list, list) and len(full_mesh_list) > 0:
                        unseg_full_mesh = full_mesh_list[0]
                        logger.debug(
                            f"  Unsegmented full scene: {unseg_full_mesh.vertices.shape[0]} verts"
                        )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"  Unsegmented full scene decode failed: {e}")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
            timings["decode_unseg"] = round(time.time() - _t_unseg, 3)
    return SceneInferenceResult(
        sha256=sha256,
        coords=coords,
        shape_slat=shape_slat,
        img_feats=img_feats,
        multihot=multihot,
        primary_labels=primary_labels,
        num_parts=num_parts,
        part_slats=part_slats,
        part_meshes=part_meshes,
        unseg_full_mesh=unseg_full_mesh,
        raw_multihot=raw_multihot,
        raw_primary_labels=raw_primary_labels,
        raw_num_parts=raw_num_parts,
        raw_part_meshes=raw_part_meshes,
        agg_multihot=agg_multihot,
        agg_primary_labels=agg_primary_labels,
        agg_num_parts=agg_num_parts,
        agg_part_meshes=agg_part_meshes,
        mask_nms_multihot=mask_nms_multihot,
        mask_nms_primary_labels=mask_nms_primary_labels,
        mask_nms_num_parts=mask_nms_num_parts,
        mask_nms_part_meshes=mask_nms_part_meshes,
        scene_tex_slat=scene_tex_slat,
        textured_post_meshes=textured_post_meshes,
        textured_agg_meshes=textured_agg_meshes,
        textured_mask_nms_meshes=textured_mask_nms_meshes,
        textured_unseg_mesh=textured_unseg_mesh,
        prompt_masks=(
            (data or {}).get("prompt_masks")
            if getattr(decoder, "query_source", "fixed") == "prompt_mask"
            else None
        ),
        raw_multihot_full=(
            (raw_multihot_full.cpu() if raw_multihot_full is not None else None)
            if getattr(decoder, "query_source", "fixed") == "prompt_mask"
            else None
        ),
        prompt_rebuild_info=prompt_rebuild_info,
        timings=timings,
    )


def _compute_part_to_prompt_q(
    multihot: Optional[torch.Tensor], raw_multihot_full: Optional[torch.Tensor]
) -> Optional[List[int]]:

    if (
        multihot is None
        or raw_multihot_full is None
        or multihot.numel() == 0
        or (raw_multihot_full.numel() == 0)
    ):
        return None
    K = int(multihot.shape[1])
    Q = int(raw_multihot_full.shape[1])
    if K == 0 or Q == 0:
        return None
    mh_part = multihot.detach().cpu().float()
    mh_q = raw_multihot_full.detach().cpu().float()
    inter = mh_part.t() @ mh_q
    part_size = mh_part.sum(dim=0)
    q_size = mh_q.sum(dim=0)
    union = part_size[:, None] + q_size[None, :] - inter
    iou = inter / union.clamp_min(1.0)
    return iou.argmax(dim=1).tolist()


def _save_prompt_pngs_for_variant(
    variant_dir: Path,
    multihot: Optional[torch.Tensor],
    prompt_masks: Optional[torch.Tensor],
    raw_multihot_full: Optional[torch.Tensor],
) -> None:

    if prompt_masks is None or prompt_masks.numel() == 0:
        return
    best_q = _compute_part_to_prompt_q(multihot, raw_multihot_full)
    if best_q is None:
        return
    variant_dir.mkdir(parents=True, exist_ok=True)
    pm_cpu = prompt_masks.detach().cpu().float().clamp_(0.0, 1.0)
    for part_idx, q_idx in enumerate(best_q):
        if not (variant_dir / f"part_{part_idx}.glb").exists():
            continue
        mask_2d = pm_cpu[q_idx]
        rgb = mask_2d.unsqueeze(0).expand(3, -1, -1).contiguous()
        from torchvision.utils import save_image as _tv_save

        _tv_save(rgb.unsqueeze(0), str(variant_dir / f"prompt_{part_idx}.png"))


def _save_variant_meshes(
    variant_dir: Path,
    meshes: List[Optional[Mesh]],
    full_mesh: Optional[Mesh] = None,
    save_colored_merge: bool = True,
) -> None:

    if full_mesh is None and all(mesh is None for mesh in meshes):
        return
    variant_dir.mkdir(parents=True, exist_ok=True)
    for i, mesh in enumerate(meshes):
        if mesh is not None:
            save_mesh_glb(mesh, str(variant_dir / f"part_{i}.glb"))
    if full_mesh is not None:
        save_mesh_glb(full_mesh, str(variant_dir / "full_scene.glb"))
    if save_colored_merge:
        colored = merge_colored_parts(meshes)
        if colored is not None:
            save_mesh_glb(colored, str(variant_dir / "full_scene_colored.glb"))


def save_scene_results(
    result: SceneInferenceResult, scene_dir: Path, args, t0: float = None
):

    if not any(
        mesh is not None and mesh.vertices.numel() > 0 for mesh in result.part_meshes
    ):
        raise RuntimeError(
            "No valid object meshes were decoded; scene output is incomplete"
        )
    scene_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir = scene_dir / "geometry"
    _save_t0 = time.time()
    _save_variant_meshes(geometry_dir / "raw", result.raw_part_meshes)
    _save_prompt_pngs_for_variant(
        geometry_dir / "raw",
        result.raw_multihot,
        result.prompt_masks,
        result.raw_multihot_full,
    )
    _PRIMARY_DIR = {
        "standard": "post_standard",
        "aggressive": "post_aggressive",
        "mask_nms": "post_mask_nms",
        "both": "post_standard",
        "all": "post_standard",
    }
    primary_dir_name = _PRIMARY_DIR.get(args.postprocess_mode, "post_standard")
    _save_variant_meshes(geometry_dir / primary_dir_name, result.part_meshes)
    _save_prompt_pngs_for_variant(
        geometry_dir / primary_dir_name,
        result.multihot,
        result.prompt_masks,
        result.raw_multihot_full,
    )
    if result.agg_num_parts > 0:
        _save_variant_meshes(geometry_dir / "post_aggressive", result.agg_part_meshes)
        _save_prompt_pngs_for_variant(
            geometry_dir / "post_aggressive",
            result.agg_multihot,
            result.prompt_masks,
            result.raw_multihot_full,
        )
    if result.mask_nms_num_parts > 0:
        _save_variant_meshes(
            geometry_dir / "post_mask_nms", result.mask_nms_part_meshes
        )
        _save_prompt_pngs_for_variant(
            geometry_dir / "post_mask_nms",
            result.mask_nms_multihot,
            result.prompt_masks,
            result.raw_multihot_full,
        )
    if result.unseg_full_mesh is not None:
        unseg_dir = geometry_dir / "unseg"
        unseg_dir.mkdir(parents=True, exist_ok=True)
        save_mesh_glb(result.unseg_full_mesh, str(unseg_dir / "full_scene.glb"))
    result.raw_part_meshes = []
    result.part_meshes = []
    result.agg_part_meshes = []
    result.mask_nms_part_meshes = []
    result.unseg_full_mesh = None
    import gc as _gc

    _gc.collect()
    torch.cuda.empty_cache()
    seg_data = dict(
        coords=result.coords.cpu().numpy(),
        multihot=result.multihot.cpu().numpy(),
        primary_labels=result.primary_labels.cpu().numpy(),
        num_parts=result.num_parts,
    )
    if result.raw_multihot is not None:
        seg_data["raw_multihot"] = result.raw_multihot.cpu().numpy()
        seg_data["raw_primary_labels"] = result.raw_primary_labels.cpu().numpy()
        seg_data["raw_num_parts"] = result.raw_num_parts
    if result.agg_multihot is not None:
        seg_data["agg_multihot"] = result.agg_multihot.cpu().numpy()
        seg_data["agg_primary_labels"] = result.agg_primary_labels.cpu().numpy()
        seg_data["agg_num_parts"] = result.agg_num_parts
    if result.mask_nms_multihot is not None:
        seg_data["mask_nms_multihot"] = result.mask_nms_multihot.cpu().numpy()
        seg_data["mask_nms_primary_labels"] = (
            result.mask_nms_primary_labels.cpu().numpy()
        )
        seg_data["mask_nms_num_parts"] = result.mask_nms_num_parts
    np.savez(str(scene_dir / "segmentation.npz"), **seg_data)
    save_scene_latent_cache(
        scene_dir=scene_dir,
        shape_slat=result.shape_slat,
        scene_tex_slat=result.scene_tex_slat,
    )
    latents_dir = scene_dir / "latents"
    for i, slat in enumerate(result.part_slats):
        if slat is not None:
            torch.save(
                {"feats": slat.feats.cpu(), "coords": slat.coords.cpu()},
                str(latents_dir / f"part_{i}_slat.pt"),
            )
    if args.with_texture and not getattr(args, "skip_part_texture", False):
        from trellis2.pipelines.scene_texturing import save_textured_mesh_glb

        def write_textured(variant: str, meshes: List[Optional[Any]]) -> None:
            variant_dir = scene_dir / "texture" / variant
            for index, mesh in enumerate(meshes):
                if mesh is None:
                    continue
                variant_dir.mkdir(parents=True, exist_ok=True)
                save_textured_mesh_glb(
                    mesh,
                    str(variant_dir / f"part_{index}.glb"),
                    texture_size=args.tex_texture_size,
                    decimation_target=args.tex_decimation_target,
                )

        if not any(mesh is not None for mesh in result.textured_post_meshes):
            raise RuntimeError(
                "Texture output was requested but no textured objects were decoded"
            )
        with _stage_timer(result.timings, "tex_bake"):
            write_textured(primary_dir_name, result.textured_post_meshes)
            if primary_dir_name != "post_aggressive":
                write_textured("post_aggressive", result.textured_agg_meshes)
            if primary_dir_name != "post_mask_nms":
                write_textured("post_mask_nms", result.textured_mask_nms_meshes)
            if result.textured_unseg_mesh is not None:
                unseg_dir = scene_dir / "texture" / "unseg"
                unseg_dir.mkdir(parents=True, exist_ok=True)
                save_textured_mesh_glb(
                    result.textured_unseg_mesh,
                    str(unseg_dir / "full_scene.glb"),
                    texture_size=args.tex_texture_size,
                    decimation_target=args.tex_decimation_target,
                )
    timings = dict(getattr(result, "timings", {}) or {})
    timings["save"] = round(time.time() - _save_t0, 3)
    if t0 is not None:
        timings["total"] = round(time.time() - t0, 3)
    scene_info = {
        "sha256": result.sha256,
        "num_parts": result.num_parts,
        "raw_num_parts": result.raw_num_parts,
        "agg_num_parts": result.agg_num_parts,
        "mask_nms_num_parts": result.mask_nms_num_parts,
        "total_voxels": int(result.coords.shape[0]),
        "elapsed_s": round(time.time() - t0, 2) if t0 is not None else None,
        "timing": timings,
        "parts": [],
        "raw_parts": [],
        "agg_parts": [],
        "mask_nms_parts": [],
        "timings_s": dict(result.timings),
        "total_run_s": round(_sum_numeric_timings(result.timings), 4),
    }
    def _empty_ids(prefix: str) -> set:
        return set(result.timings.get(f"{prefix}_empty_part_ids", []) or [])

    empty_post, empty_raw = _empty_ids("decode_post"), _empty_ids("decode_raw")
    empty_agg, empty_mask_nms = _empty_ids("decode_aggressive"), _empty_ids("decode_mask_nms")
    for i in range(result.num_parts):
        part_info = {"part_id": i, "voxels": int(result.multihot[:, i].sum().item())}
        if i in empty_post:
            part_info["empty_decode"] = True
        if i < len(result.part_meshes) and result.part_meshes[i] is not None:
            part_info["vertices"] = int(result.part_meshes[i].vertices.shape[0])
            part_info["faces"] = int(result.part_meshes[i].faces.shape[0])
        scene_info["parts"].append(part_info)
    for i in range(result.raw_num_parts):
        raw_info = {"part_id": i, "voxels": int(result.raw_multihot[:, i].sum().item())}
        if i in empty_raw:
            raw_info["empty_decode"] = True
        if i < len(result.raw_part_meshes) and result.raw_part_meshes[i] is not None:
            raw_info["vertices"] = int(result.raw_part_meshes[i].vertices.shape[0])
            raw_info["faces"] = int(result.raw_part_meshes[i].faces.shape[0])
        scene_info["raw_parts"].append(raw_info)
    if result.agg_multihot is not None:
        for i in range(result.agg_num_parts):
            agg_info = {
                "part_id": i,
                "voxels": int(result.agg_multihot[:, i].sum().item()),
            }
            if i in empty_agg:
                agg_info["empty_decode"] = True
            if (
                i < len(result.agg_part_meshes)
                and result.agg_part_meshes[i] is not None
            ):
                agg_info["vertices"] = int(result.agg_part_meshes[i].vertices.shape[0])
                agg_info["faces"] = int(result.agg_part_meshes[i].faces.shape[0])
            scene_info["agg_parts"].append(agg_info)
    if result.mask_nms_multihot is not None:
        for i in range(result.mask_nms_num_parts):
            mn_info = {
                "part_id": i,
                "voxels": int(result.mask_nms_multihot[:, i].sum().item()),
            }
            if i in empty_mask_nms:
                mn_info["empty_decode"] = True
            if (
                i < len(result.mask_nms_part_meshes)
                and result.mask_nms_part_meshes[i] is not None
            ):
                mn_info["vertices"] = int(
                    result.mask_nms_part_meshes[i].vertices.shape[0]
                )
                mn_info["faces"] = int(result.mask_nms_part_meshes[i].faces.shape[0])
            scene_info["mask_nms_parts"].append(mn_info)
    if result.prompt_rebuild_info is not None:
        scene_info["prompt_rebuild"] = {
            "strategy": result.prompt_rebuild_info.get("strategy", "rebuild"),
            "num_objects": result.prompt_rebuild_info.get("num_objects", 0),
            "num_valid": result.prompt_rebuild_info.get("num_valid", 0),
            "dropped": result.prompt_rebuild_info.get("dropped", []),
            "valid_obj_ids": result.prompt_rebuild_info.get("valid_obj_ids", []),
            "dilated_radius": result.prompt_rebuild_info.get("dilated_radius", []),
            "matched_per_obj": result.prompt_rebuild_info.get("matched_per_obj", []),
            "collisions": result.prompt_rebuild_info.get("collisions", 0),
            "fallback": result.prompt_rebuild_info.get("fallback", "drop"),
        }
        if "align" in result.prompt_rebuild_info:
            scene_info["prompt_rebuild"]["align"] = result.prompt_rebuild_info["align"]
        if "per_obj" in result.prompt_rebuild_info:
            scene_info["prompt_rebuild"]["per_obj"] = result.prompt_rebuild_info[
                "per_obj"
            ]
        if "snap_radius" in result.prompt_rebuild_info:
            scene_info["prompt_rebuild"]["snap_radius"] = result.prompt_rebuild_info[
                "snap_radius"
            ]
    if result.prompt_masks is not None:
        scene_info["num_input_masks"] = int(result.prompt_masks.shape[0])
        scene_info["num_queries"] = int(result.raw_multihot_full.shape[1])
    if result.timings:
        scene_info["timings_s"] = result.timings
        scene_info["timings_s"]["total"] = round(
            _sum_numeric_timings(result.timings), 3
        )
    with open(scene_dir / "scene_info.json", "w") as f:
        json.dump(scene_info, f, indent=2)


def runtime_metadata() -> Dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version
    import os

    packages = {}
    for name in ("torch", "torchvision", "transformers", "flash-attn", "numpy", "Pillow", "flex-gemm", "o-voxel", "utonia"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "packages": packages,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "backends": {name: os.environ.get(name) for name in ("ATTN_BACKEND", "SPARSE_ATTN_BACKEND", "SPCONV_BACKEND", "CUBLAS_WORKSPACE_CONFIG")},
    }


def main():
    parser, args = _CLI_ARGUMENTS if _CLI_ARGUMENTS is not None else _parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.pipeline_config, "r") as f:
        pipeline_config = json.load(f)
    if args.decoder_config is None:
        run_dir = Path(args.cola3d_ckpt).resolve().parent.parent
        auto_config = run_dir / "config.json"
        if not auto_config.exists():
            parser.error(
                f"Cannot auto-resolve decoder config: {auto_config} not found. Use --decoder-config."
            )
        args.decoder_config = str(auto_config)
        logger.debug(f"Auto-resolved decoder config: {args.decoder_config}")
    with open(args.decoder_config, "r") as f:
        decoder_full_config = json.load(f)
    decoder_config = decoder_full_config["models"]["decoder"]
    query_source = decoder_config["args"].get("query_source", "fixed")
    if (
        args.dataset_class == "single_image"
        and query_source == "prompt_point"
        and (args.prompt_point_source in {"rebuild", "gt"})
        and (not args.prepare_manual)
    ):
        parser.error(
            "--dataset-class=single_image with a prompt_point decoder requires --prompt-point-source mask_moge2 (or manual with cached SLats)."
        )
    dec_resolution = decoder_config["args"].get("resolution", 512)
    if args.resolution is None:
        args.resolution = dec_resolution
        logger.debug(f"Auto-set --resolution to {args.resolution} (from decoder config)")
    if args.ss_coord_resolution is None:
        args.ss_coord_resolution = 64 if dec_resolution >= 1024 else 32
        logger.debug(
            f"Auto-set --ss-coord-resolution to {args.ss_coord_resolution} (from decoder config resolution={dec_resolution})"
        )
    if args.slat_flow_resolution == "auto":
        args.slat_flow_resolution = "1024" if dec_resolution >= 1024 else "512"
        logger.debug(
            f"Auto-detected SLat flow resolution: {args.slat_flow_resolution} (decoder config resolution={dec_resolution})"
        )
    with open(args.ss_flow_config, "r") as f:
        ss_flow_full_config = json.load(f)
    ss_flow_config = ss_flow_full_config["models"]["denoiser"]
    if args.dataset_class == "benchmark_views":
        from trellis2.datasets.benchmark_views import BenchmarkViewsDataset

        if args.benchmark_dataset == "InstPIFu":
            from trellis2.datasets.benchmark_instpifu import InstPIFuBenchmarkDataset
            dataset_class = InstPIFuBenchmarkDataset
        else:
            dataset_class = BenchmarkViewsDataset
        dataset = dataset_class(
            data_root=args.benchmark_root,
            dataset_name=args.benchmark_dataset,
            image_size=512,
            emit_prompt_masks=(query_source == "prompt_mask"),
            load_scene_voxels=(query_source == "prompt_point" and args.prompt_point_source == "rebuild"),
            instances=args.dataset_instances,
        )
        data_dir = None
        logger.debug(f"Loaded benchmark views: {len(dataset)}")
    elif args.dataset_class == "single_image":
        from trellis2.datasets.single_image import SingleImageDataset

        dataset = SingleImageDataset(
            input_rgb=args.input_rgb,
            input_mask=args.input_mask,
            input_rgba=args.input_rgba,
            prompt_mask_dir=args.input_prompt_mask_dir,
            image_size=512,
            instance_id=args.input_id,
            mask_mode=args.input_mask_mode,
            mask_threshold=args.input_mask_threshold,
        )
        data_dir = None
        logger.debug(f"Loaded SingleImageDataset: {dataset.instance_id}")
    else:
        from trellis2.datasets.flexi_dual_grid_multi_obj import (
            FlexiDualGridMultiObjDataset,
        )

        dataset_config = decoder_full_config["dataset"]
        dataset_args = dict(dataset_config["args"])
        dataset_args["instances"] = args.dataset_instances
        dataset_args["skip_mesh"] = True
        dataset_args["skip_per_obj"] = True
        dataset_args["preload_mode"] = None
        dataset_args["max_active_voxels"] = 999999999
        dataset_args["max_num_faces"] = 999999999
        dataset_args["max_objects"] = 999
        dataset_args["min_objects"] = None
        needs_dataset_prompt_points = (
            query_source == "prompt_point"
            and args.prompt_point_source in {"gt", "gt_native"}
        )
        if query_source == "prompt_mask":
            dataset_args["prompt_query_mode"] = "mask"
        elif needs_dataset_prompt_points:
            dataset_args["prompt_query_mode"] = "point"
        else:
            dataset_args["prompt_query_mode"] = "none"
        if needs_dataset_prompt_points:
            dataset_args.setdefault("shape_latent_root", {})
        else:
            dataset_args.pop("shape_latent_root", None)
        dataset_args.pop("utonia_feats_root", None)
        dataset_args.pop("dataset_repeat", None)
        if args.data_dir:
            import commentjson as _cj

            _parsed = (
                _cj.loads(args.data_dir)
                if isinstance(args.data_dir, str)
                else args.data_dir
            )
            if isinstance(dataset_args.get("render_root"), dict):
                dataset_args["render_root"] = {}
            if isinstance(dataset_args.get("shape_latent_root"), dict):
                dataset_args["shape_latent_root"] = {}
            for _dataset_key, _root_cfg in _parsed.items():
                _base = _root_cfg["base"] if isinstance(_root_cfg, dict) else _root_cfg
                if isinstance(dataset_args.get("render_root"), dict):
                    render_dir_name = (
                        "3D-FRONT-RENDER"
                        if _dataset_key == "MIDI_3D_FRONT"
                        else "renders"
                    )
                    dataset_args["render_root"][_dataset_key] = str(
                        Path(_base) / "raw" / render_dir_name
                    )
                else:
                    dataset_args["render_root"] = str(Path(_base) / "raw" / "renders")
                if "shape_latent_root" in dataset_args:
                    latent_root = (
                        Path(_base)
                        / "shape_latents"
                        / f"shape_enc_next_dc_f16c32_fp16_{dec_resolution}"
                    )
                    if isinstance(dataset_args["shape_latent_root"], dict):
                        dataset_args["shape_latent_root"][_dataset_key] = str(
                            latent_root
                        )
                    else:
                        dataset_args["shape_latent_root"] = str(latent_root)
            _utonia_map: Dict[str, str] = {}
            _slat_map: Dict[str, str] = {}
            _stripped_parsed: Dict[str, Any] = {}
            for _ds_key, _spec in _parsed.items():
                if isinstance(_spec, dict):
                    cleaned = {
                        k: v
                        for k, v in _spec.items()
                        if k not in ("utonia_feats", "shape_latents")
                    }
                    _stripped_parsed[_ds_key] = cleaned
                    if _spec.get("utonia_feats"):
                        _utonia_map[_ds_key] = str(_spec["utonia_feats"])
                    if _spec.get("shape_latents"):
                        _slat_map[_ds_key] = str(_spec["shape_latents"])
                else:
                    _stripped_parsed[_ds_key] = _spec
            if _utonia_map:
                dataset_args["utonia_feats_root"] = _utonia_map
            if _slat_map:
                dataset_args["shape_latent_root"] = _slat_map
            data_dir = json.dumps(_stripped_parsed)
        if not args.data_dir:
            data_dir = json.dumps(
                {
                    "MIDI_3D_FRONT": {
                        "base": "datasets/MIDI-3D-FRONT/Train",
                        "mesh_dump": "datasets/MIDI-3D-FRONT/Train/mesh_dumps",
                        "dual_grid": f"datasets/MIDI-3D-FRONT/Train/dual_grid_{dec_resolution}",
                        "asset_stats": "datasets/MIDI-3D-FRONT/Train/asset_stats",
                    }
                }
            )
        dataset = FlexiDualGridMultiObjDataset(roots=data_dir, **dataset_args)
    if args.dataset_class == "flexi_dual_grid_multi_obj" and args.render_idx >= 0:
        _orig_read_cond_image = dataset.read_cond_image

        def _fixed_read_cond_image(instance, **kwargs):
            if instance not in dataset._render_file_cache:
                _orig_read_cond_image(instance)
            candidates = dataset._render_file_cache.get(instance, [])
            if not candidates:
                return None
            candidates_sorted = sorted(candidates)
            idx = args.render_idx % len(candidates_sorted)
            image_path = candidates_sorted[idx]
            from PIL import Image as _Image

            image = _Image.open(image_path)
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            alpha = np.array(image.getchannel(3))
            nonzero = alpha.nonzero()
            if nonzero[0].size == 0:
                return None
            bbox = [
                nonzero[1].min(),
                nonzero[0].min(),
                nonzero[1].max(),
                nonzero[0].max(),
            ]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_bbox = [
                int(center[0] - hsize),
                int(center[1] - hsize),
                int(center[0] + hsize),
                int(center[1] + hsize),
            ]
            image = image.crop(aug_bbox)
            image = image.resize(
                (dataset.image_size, dataset.image_size), _Image.LANCZOS
            )
            import torchvision.transforms.functional as TF

            image = TF.to_tensor(image)
            image = image[:3] * image[3:4]
            logger.debug(f"  Using render: {Path(image_path).name}")
            return image

        dataset.read_cond_image = _fixed_read_cond_image
        logger.debug(f"Fixed render view index: {args.render_idx}")
    print("Loading models...")
    dino_model = load_dino_model(pipeline_config)
    if args.use_ss_flow:
        if args.ss_flow_ema_ckpt is None:
            ema_path = args.ss_flow_ckpt.replace(
                "denoiser_step", "denoiser_ema0.9999_step"
            )
            args.ss_flow_ema_ckpt = ema_path if Path(ema_path).exists() else None
        ss_flow_model = load_ss_flow_model(
            ss_flow_config, args.ss_flow_ckpt, args.ss_flow_ema_ckpt
        )
    else:
        ss_flow_model = None
    ss_decoder = load_ss_decoder(
        model_path=pipeline_config["args"]["models"].get("sparse_structure_decoder")
    )
    if args.load_shape_slats_from or args.shape_latent_source == "gt":
        slat_flow_model = None
        logger.debug("  [cache] skipping SLat flow model load (--load-shape-slats-from)")
    else:
        slat_flow_model = load_slat_flow_model(
            pipeline_config,
            custom_ckpt=args.slat_flow_ckpt,
            resolution_key=args.slat_flow_resolution,
        )
    decoder = load_decomposable_decoder(
        decoder_config,
        args.cola3d_ckpt,
        args.geo_ckpt,
        pretrained_geo_safetensors=args.pretrained_geo,
    )
    online_utonia_extractor = None
    if args.online_utonia:
        decoder_input_mode = getattr(decoder, "input_mode", "latent")
        if decoder_input_mode != "latent_utonia_1024":
            print(
                f"  [online-utonia] input_mode={decoder_input_mode!r} does not use precomputed Utonia features; flag ignored."
            )
        else:
            from trellis2.utils.online_utonia import build_utonia_extractor

            logger.debug("  Building online Utonia extractor (1024^3)...")
            online_utonia_extractor = build_utonia_extractor(resolution=1024)
            logger.debug(
                f"  Online Utonia extractor ready (bottleneck_dim={online_utonia_extractor.bottleneck_dim})."
            )
    ss_sampler, ss_params, slat_sampler, slat_params, norm_mean, norm_std = (
        load_samplers(pipeline_config)
    )
    moge_model = None
    if getattr(args, "prompt_point_source", "rebuild") == "mask_moge2":
        from trellis2.utils.prompt_point_moge2 import load_moge2_model

        logger.debug(
            f"Loading MoGe2 ({args.moge2_model_name}) for mask_moge2 prompt-point path..."
        )
        moge_model = load_moge2_model(args.moge2_model_name, device="cuda")
        logger.debug("  MoGe2 ready.")
    tex_flow_model = None
    tex_decoder = None
    tex_sampler = None
    tex_sampler_params = None
    shape_slat_norm_dict = None
    tex_slat_norm_dict = None
    if args.with_texture:
        logger.debug("Loading texture models...")
        skip_tex_flow = bool(args.load_shape_slats_from) and bool(
            getattr(args, "skip_tex_flow_when_cached", False)
        )
        if skip_tex_flow:
            tex_flow_model = None
            logger.debug(
                "  [cache] skipping tex_flow_model load (explicit --skip-tex-flow-when-cached)"
            )
        else:
            tex_flow_model = load_tex_flow_model(
                pipeline_config,
                resolution_key=args.slat_flow_resolution,
                custom_ckpt=args.tex_flow_ckpt,
            )
        tex_decoder = load_tex_decoder(
            pipeline_config, custom_ckpt=args.tex_decoder_ckpt
        )
        tex_sampler, tex_sampler_params, shape_slat_norm_dict, tex_slat_norm_dict = (
            load_tex_sampler(pipeline_config)
        )
    logger.debug("All models loaded.\n")
    scene_indices = list(range(len(dataset)))
    scene_indices = scene_indices[args.start_idx :]
    if args.end_idx is not None:
        scene_indices = scene_indices[: args.end_idx - args.start_idx]
    if args.max_scenes:
        scene_indices = scene_indices[: args.max_scenes]
    summary = {"scenes": [], "args": vars(args), "runtime": runtime_metadata()}
    skipped_resume = 0
    processed_count = 0
    for seq, idx in enumerate(scene_indices):
        root, sha256 = dataset.instances[idx]
        scene_dir = output_dir / sha256
        if scene_is_completed(
            scene_dir / "scene_info.json", prepare_manual=args.prepare_manual
        ):
            skipped_resume += 1
            continue
        print(f"[{seq + 1}/{len(scene_indices)}] {sha256}", flush=True)
        t0 = time.time()
        try:
            data = dataset.get_instance(root, sha256)
        except Exception as e:
            print(f"  [SKIP] Failed to load: {e}")
            summary["scenes"].append({"sha256": sha256, "error": str(e)})
            continue
        cond_image = data.get("cond_image")
        if cond_image is None:
            print("  [SKIP] No cond_image")
            summary["scenes"].append(
                {"sha256": sha256, "error": "No conditioning image"}
            )
            continue
        scene_dir.mkdir(parents=True, exist_ok=True)
        from torchvision.utils import save_image as tv_save

        tv_save(cond_image.unsqueeze(0), str(scene_dir / "cond_image.png"))
        result = None
        try:
            result = run_single_scene(
                sha256=sha256,
                cond_image=cond_image,
                dino_model=dino_model,
                ss_flow_model=ss_flow_model,
                ss_decoder=ss_decoder,
                slat_flow_model=slat_flow_model,
                decoder=decoder,
                ss_sampler=ss_sampler,
                ss_params=ss_params,
                slat_sampler=slat_sampler,
                slat_params=slat_params,
                norm_mean=norm_mean,
                norm_std=norm_std,
                args=args,
                data=data,
                tex_flow_model=tex_flow_model,
                tex_decoder=tex_decoder,
                tex_sampler=tex_sampler,
                tex_sampler_params=tex_sampler_params,
                shape_slat_norm_dict=shape_slat_norm_dict,
                tex_slat_norm_dict=tex_slat_norm_dict,
                online_utonia_extractor=online_utonia_extractor,
                moge_model=moge_model,
                scene_dir=scene_dir,
            )
            t_save_start = time.time()
            if result.prepare_manual:
                marker = {
                    "sha256": result.sha256,
                    "mode": "prepare_manual",
                    "max_parts": int(decoder.max_parts),
                    "query_source": getattr(decoder, "query_source", "fixed"),
                    "seed": args.seed,
                    "shape_slat": "latents/shape_slat.pt",
                    "num_latent_voxels": int(result.shape_slat.coords.shape[0]),
                    "next_step": "Run scripts/inference/ui_manual_prompt_points.py on this output dir, then rerun run_inference.py with --prompt-point-source manual --load-shape-slats-from <output-dir>.",
                }
                if result.scene_tex_slat is not None:
                    marker["scene_tex_slat"] = "latents/scene_tex_slat.pt"
                if result.timings:
                    marker["timings_s"] = dict(result.timings)
                    marker["timings_s"]["total"] = round(
                        _sum_numeric_timings(result.timings), 3
                    )
                with open(scene_dir / "scene_info.json", "w") as f:
                    json.dump(marker, f, indent=2)
            else:
                object_count = sum(
                    mesh is not None and mesh.vertices.numel() > 0
                    for mesh in result.part_meshes
                )
                save_scene_results(result, scene_dir, args, t0=t0)
            elapsed = time.time() - t0
            scene_summary = {
                "sha256": sha256,
                "mode": "prepare_manual" if result.prepare_manual else "inference",
                "num_parts": result.num_parts,
                "total_voxels": int(result.coords.shape[0]),
                "elapsed_s": round(elapsed, 2),
                "timing": dict(getattr(result, "timings", {}) or {}),
                "warmup": processed_count == 0,
            }
            if result.prompt_rebuild_info is not None:
                pri = result.prompt_rebuild_info
                pr_summary = {
                    "strategy": pri.get("strategy", "rebuild"),
                    "num_objects": pri.get("num_objects", 0),
                    "num_valid": pri.get("num_valid", 0),
                    "dropped": pri.get("dropped", []),
                    "dilated_radius": pri.get("dilated_radius", []),
                    "matched_per_obj": pri.get("matched_per_obj", []),
                    "collisions": pri.get("collisions", 0),
                }
                if "align" in pri:
                    pr_summary["align"] = pri["align"]
                if "snap_radius" in pri:
                    pr_summary["snap_radius"] = pri["snap_radius"]
                scene_summary["prompt_rebuild"] = pr_summary
            if result.timings:
                scene_summary["timings_s"] = dict(result.timings)
                scene_summary["timings_s"]["total"] = round(
                    _sum_numeric_timings(result.timings), 3
                )
            summary["scenes"].append(scene_summary)
            processed_count += 1
            if result.prepare_manual:
                print(f"  Prepared latent cache in {elapsed:.1f}s", flush=True)
            else:
                print(f"  Done: {object_count} objects in {elapsed:.1f}s", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(
                f"  [OOM-ABORT] CUDA OOM on {sha256}; exiting shard for clean restart"
            )
            print(f"             {e}")
            summary["scenes"].append({"sha256": sha256, "error": f"OOM: {e}"})
            with open(output_dir / args.summary_name, "w") as f:
                json.dump(summary, f, indent=2)
            import sys as _sys

            _sys.exit(99)
        except Exception as e:
            import traceback

            print(f"  [ERROR] {e}")
            traceback.print_exc()
            summary["scenes"].append({"sha256": sha256, "error": str(e)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with open(output_dir / args.summary_name, "w") as f:
        json.dump(summary, f, indent=2)
    failures = sum("error" in scene for scene in summary["scenes"])
    print(
        f"Completed {processed_count} scenes, skipped {skipped_resume}, failed {failures}: {output_dir}"
    )
    if failures or (processed_count == 0 and skipped_resume == 0):
        raise RuntimeError(
            f"Inference did not complete all requested scenes; see {args.summary_name}"
        )


if __name__ == "__main__":
    main()
