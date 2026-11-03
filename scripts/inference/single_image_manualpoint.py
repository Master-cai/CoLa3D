

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone single-image manual-point pipeline (prepare | decode).",
    )
    p.add_argument("--stage", required=True, choices=["prepare", "decode"])

    p.add_argument(
        "--input-rgb", default=None, help="RGB image path (needs --input-mask)."
    )
    p.add_argument("--input-mask", default=None, help="Foreground/semantic mask path.")
    p.add_argument(
        "--input-rgba",
        default=None,
        help="RGBA image (alpha = foreground); mutually exclusive with rgb+mask.",
    )
    p.add_argument(
        "--input-id", default=None, help="Scene id; also the output subdirectory name."
    )
    p.add_argument(
        "--input-mask-mode",
        default="auto",
        choices=["auto", "binary", "instance", "palette"],
    )
    p.add_argument("--input-mask-threshold", type=int, default=0)
    p.add_argument(
        "--instance-id",
        default=None,
        help="Override output instance dir (default: sanitized --input-id).",
    )

    p.add_argument(
        "--output-dir", required=True, help="Run dir; writes <output>/<instance_id>/..."
    )
    p.add_argument(
        "--load-shape-slats-from",
        default=None,
        help="Decode: dir with cached shape_slat (default: --output-dir).",
    )
    p.add_argument(
        "--manual-prompt-points-from",
        default=None,
        help="Decode: dir with manual_prompt_points.pt (default: --output-dir).",
    )

    p.add_argument(
        "--cola3d-ckpt", required=True, help="CoLa3D + MaskDecoder checkpoint (.pt)."
    )
    p.add_argument(
        "--geo-ckpt",
        default=None,
        help="Geometry decoder checkpoint (.pt). Either this or --pretrained-geo is required.",
    )
    p.add_argument(
        "--ss-flow-ckpt",
        default=None,
        help="SS flow denoiser checkpoint (.pt) — prepare only.",
    )
    p.add_argument(
        "--ss-flow-ema-ckpt", default=None, help="Optional SS flow EMA checkpoint."
    )
    p.add_argument(
        "--slat-flow-ckpt",
        default=None,
        help="SLat flow denoiser checkpoint (.pt) — prepare only.",
    )
    p.add_argument(
        "--pretrained-geo",
        default=None,
        help="Optional geo backbone .safetensors (fp16).",
    )

    p.add_argument("--pipeline-config", default="pretrained_ckpts/pipeline.json")
    p.add_argument("--ss-flow-config", default="configs/train/finetune/ss_flow.json")
    p.add_argument(
        "--decoder-config",
        default=None,
        help="Decoder training config JSON (default: auto from --cola3d-ckpt run dir).",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resolution", type=int, default=1024, help="Decoder mesh resolution."
    )
    p.add_argument("--ss-coord-resolution", type=int, default=64)
    p.add_argument(
        "--slat-flow-resolution",
        default="1024",
        help="SLat flow conditioning resolution key.",
    )

    p.add_argument("--min-tokens", type=int, default=3)
    p.add_argument("--assign-threshold", type=float, default=0.5)
    p.add_argument("--nms-iou-threshold", type=float, default=0.5)
    p.add_argument("--fragment-min-voxels", type=int, default=5)
    p.add_argument("--fragment-ratio", type=float, default=0.05)
    p.add_argument("--min-object-voxels", type=int, default=20)
    return p


def _parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.input_rgba:
        if args.input_rgb or args.input_mask:
            raise SystemExit(
                "--input-rgba is mutually exclusive with --input-rgb/--input-mask"
            )
    elif not (args.input_rgb and args.input_mask):
        raise SystemExit(
            "single-image input requires either --input-rgba or both --input-rgb and --input-mask"
        )
    if not args.geo_ckpt and (not args.pretrained_geo):
        raise SystemExit(
            "provide a geometry decoder via --geo-ckpt (.pt) or --pretrained-geo (.safetensors)"
        )
    if args.stage == "prepare":
        if not (args.ss_flow_ckpt and args.slat_flow_ckpt):
            raise SystemExit(
                "--stage prepare requires --ss-flow-ckpt and --slat-flow-ckpt"
            )
    return args


_CLI_ARGUMENTS = _parse_arguments() if __name__ == "__main__" else None

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "inference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_inference as E

import trellis2.modules.sparse as sp
from trellis2.datasets.single_image import SingleImageDataset
from trellis2.representations import Mesh
from trellis2.utils.segment_postprocess import postprocess_segmentation

DEVICE = "cuda"


def _resolve_decoder_config(args: argparse.Namespace) -> dict:

    if args.decoder_config is None:
        run_dir = Path(args.cola3d_ckpt).resolve().parent.parent
        auto_config = run_dir / "config.json"
        if not auto_config.exists():
            raise FileNotFoundError(
                f"Cannot auto-resolve decoder config: {auto_config} not found. "
                "Pass --decoder-config."
            )
        args.decoder_config = str(auto_config)
        print(f"Auto-resolved decoder config: {args.decoder_config}", flush=True)
    with open(args.decoder_config, "r") as f:
        decoder_full_config = json.load(f)
    return decoder_full_config["models"]["decoder"]


def _build_dataset(args: argparse.Namespace) -> SingleImageDataset:

    return SingleImageDataset(
        input_rgb=args.input_rgb,
        input_mask=args.input_mask,
        input_rgba=args.input_rgba,
        image_size=512,
        instance_id=args.input_id,
        mask_mode=args.input_mask_mode,
        mask_threshold=args.input_mask_threshold,
    )


def _dino_feats(dino_model, cond_image: torch.Tensor, image_size: int) -> torch.Tensor:

    dino_model.image_size = image_size
    feats = dino_model(cond_image.unsqueeze(0).to(DEVICE))
    dino_model.image_size = 512
    return feats


def _load_ss_decoder_offline_safe(pipeline_config: dict) -> torch.nn.Module:
    return E.load_ss_decoder(
        model_path=pipeline_config["args"]["models"].get("sparse_structure_decoder")
    )


def prepare(args: argparse.Namespace) -> None:
    decoder_config = _resolve_decoder_config(args)
    max_parts = int(decoder_config["args"]["max_parts"])
    if max_parts <= 0:
        raise ValueError("Decoder max_parts must be positive")
    with open(args.pipeline_config, "r") as f:
        pipeline_config = json.load(f)
    with open(args.ss_flow_config, "r") as f:
        ss_flow_config = json.load(f)["models"]["denoiser"]

    dataset = _build_dataset(args)
    instance_id = args.instance_id or dataset.instance_id
    root, sha256 = dataset.instances[0]
    data = dataset.get_instance(root, sha256)
    cond_image = data["cond_image"]
    print(f"Loaded SingleImageDataset: {instance_id}", flush=True)

    dino_model = E.load_dino_model(pipeline_config)
    ss_flow_model = E.load_ss_flow_model(
        ss_flow_config, args.ss_flow_ckpt, args.ss_flow_ema_ckpt
    )
    ss_decoder = _load_ss_decoder_offline_safe(pipeline_config)
    slat_flow_model = E.load_slat_flow_model(
        pipeline_config,
        custom_ckpt=args.slat_flow_ckpt,
        resolution_key=args.slat_flow_resolution,
    )
    ss_sampler, ss_params, slat_sampler, slat_params, norm_mean, norm_std = (
        E.load_samplers(pipeline_config)
    )

    with torch.no_grad():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        slat_cond_res = (
            int(args.slat_flow_resolution)
            if args.slat_flow_resolution != "512"
            else 512
        )
        img_feats = _dino_feats(dino_model, cond_image, 512)
        if slat_cond_res > 512:
            img_feats_slat = _dino_feats(dino_model, cond_image, slat_cond_res)
            print(
                f"  DINO features: 512 -> {img_feats.shape[1]} patches, "
                f"{slat_cond_res} -> {img_feats_slat.shape[1]} patches",
                flush=True,
            )
        else:
            img_feats_slat = img_feats
        slat_cond = {
            "cond": img_feats_slat,
            "neg_cond": torch.zeros_like(img_feats_slat),
        }

        coords = E.sample_ss_from_flow(
            ss_flow_model,
            ss_decoder,
            ss_sampler,
            ss_params,
            img_feats,
            args.ss_coord_resolution,
        )

        noise = sp.SparseTensor(
            feats=torch.randn(
                coords.shape[0], slat_flow_model.in_channels, device=DEVICE
            ),
            coords=coords,
        )
        shape_slat = slat_sampler.sample(
            slat_flow_model,
            noise,
            **slat_cond,
            **slat_params,
            verbose=True,
            tqdm_desc="Sampling SLat",
        ).samples
        shape_slat = shape_slat.replace(
            shape_slat.feats * norm_std.to(DEVICE) + norm_mean.to(DEVICE)
        )
        print(f"  Shape latent: {tuple(shape_slat.feats.shape)}", flush=True)

    scene_dir = Path(args.output_dir) / instance_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torchvision.utils import save_image as tv_save

        tv_save(cond_image.unsqueeze(0), str(scene_dir / "cond_image.png"))
    except Exception as e:
        print(f"  [prepare] cond_image dump skipped: {e}", flush=True)

    E.save_scene_latent_cache(scene_dir=scene_dir, shape_slat=shape_slat)
    marker = {
        "sha256": instance_id,
        "mode": "prepare_manual",
        "shape_slat": "latents/shape_slat.pt",
        "num_latent_voxels": int(shape_slat.coords.shape[0]),
        "max_parts": max_parts,
        "query_source": decoder_config["args"].get("query_source", "fixed"),
        "seed": args.seed,
    }
    (scene_dir / "scene_info.json").write_text(json.dumps(marker, indent=2) + "\n")
    print(
        f"[prepare] saved shape_slat -> {scene_dir / 'latents' / 'shape_slat.pt'}",
        flush=True,
    )


def decode(args: argparse.Namespace) -> None:
    with open(args.pipeline_config, "r") as f:
        pipeline_config = json.load(f)
    decoder_config = _resolve_decoder_config(args)

    dataset = _build_dataset(args)
    instance_id = args.instance_id or dataset.instance_id
    root, sha256 = dataset.instances[0]
    data = dataset.get_instance(root, sha256)
    cond_image = data["cond_image"]

    load_dir = Path(args.load_shape_slats_from or args.output_dir)
    shape_slat = E.load_cached_shape_slat(load_dir, instance_id, device=DEVICE)
    print(
        f"[decode] loaded shape_slat: feats={tuple(shape_slat.feats.shape)}", flush=True
    )

    dino_model = E.load_dino_model(pipeline_config)
    decoder = E.load_decomposable_decoder(
        decoder_config,
        args.cola3d_ckpt,
        args.geo_ckpt,
        pretrained_geo_safetensors=args.pretrained_geo,
    )

    manual_dir = Path(args.manual_prompt_points_from or args.output_dir)
    manual_points_path = manual_dir / instance_id / "manual_prompt_points.pt"

    with torch.no_grad():
        img_feats = _dino_feats(dino_model, cond_image, 512)

        x_cola3d = shape_slat.replace(shape_slat.feats.to(decoder.cola3d_dtype))
        ext_features = None
        input_mode = getattr(decoder, "input_mode", "latent")
        if input_mode == "latent_utonia_1024":
            from trellis2.utils.online_utonia import (
                build_utonia_extractor,
                compute_online_utonia_features,
            )

            print("  Building online Utonia extractor (1024^3)...", flush=True)
            extractor = build_utonia_extractor(resolution=1024)
            online_ext = compute_online_utonia_features(
                shape_slat=shape_slat,
                decoder=decoder,
                extractor=extractor,
                tex_decoder=None,
                tex_slat=None,
                pbr_attr_layout=None,
                enable_normal=False,
                timings=None,
            )
            ext_features = x_cola3d.replace(online_ext.feats.to(decoder.cola3d_dtype))
        elif input_mode in ("utonia", "latent_utonia"):
            utonia_feats = decoder.utonia_extractor(
                coords=x_cola3d.coords, batch_size=x_cola3d.shape[0]
            )
            ext_features = x_cola3d.replace(utonia_feats.to(decoder.cola3d_dtype))

        _, _, mask_logits, _, rebuild_info = E.run_prompted_cola3d_head(
            decoder,
            latent=x_cola3d,
            img_features=img_feats,
            data=None,
            ext_features=ext_features,
            prompt_strategy="manual",
            manual_points_path=manual_points_path,
            scene_dir=Path(args.output_dir) / instance_id,
        )
        print(f"  [manual] prompts: {rebuild_info.get('num_valid')} valid", flush=True)

        n_points = shape_slat.layout[0].stop - shape_slat.layout[0].start
        logits_b0 = mask_logits[0, :n_points]
        coords_b0 = shape_slat.coords[
            shape_slat.layout[0].start : shape_slat.layout[0].stop, 1:
        ]
        multihot, _primary_labels, num_parts = postprocess_segmentation(
            logits_b0,
            coords_b0,
            assign_threshold=args.assign_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            fragment_min_voxels=args.fragment_min_voxels,
            fragment_ratio=args.fragment_ratio,
            min_object_voxels=args.min_object_voxels,
        )
        print(f"  Segmentation (standard): {num_parts} parts", flush=True)

        decoder.set_resolution(args.resolution)
        _part_slats, part_meshes, _ = E._decode_parts_from_multihot(
            multihot,
            shape_slat,
            decoder,
            args.min_tokens,
            label="",
            resolution=args.resolution,
        )

    out_dir = Path(args.output_dir) / instance_id / "geometry" / "post"
    E._save_variant_meshes(out_dir, part_meshes)
    n_written = sum(1 for m in part_meshes if m is not None)
    print(
        f"[decode] wrote {n_written} part GLBs + colored merge -> {out_dir}", flush=True
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = (
        _CLI_ARGUMENTS
        if argv is None and _CLI_ARGUMENTS is not None
        else _parse_arguments(argv)
    )
    if args.stage == "prepare":
        prepare(args)
    else:
        decode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
