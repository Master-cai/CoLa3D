import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import torch
import numpy as np
import pandas as pd
import o_voxel
from tqdm import tqdm
from easydict import EasyDict as edict
import trellis2.models as models
import trellis2.modules.sparse as sp

torch.set_grad_enabled(False)


def clear_cuda_error() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _decode_intersected(attr_intersected: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [attr_intersected % 2, attr_intersected // 2 % 2, attr_intersected // 4 % 2],
        dim=-1,
    ).bool()


def merge_multi_obj_voxels(
    coords: torch.Tensor, vertices_feats: torch.Tensor, intersected_feats: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    unique_coords, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)
    n_unique = unique_coords.shape[0]
    if n_unique == coords.shape[0]:
        return (coords, vertices_feats, intersected_feats)
    count = torch.zeros(n_unique, 1, dtype=vertices_feats.dtype)
    count.scatter_add_(
        0,
        inverse_indices.unsqueeze(1),
        torch.ones(coords.shape[0], 1, dtype=vertices_feats.dtype),
    )
    merged_vertices = torch.zeros(
        n_unique, vertices_feats.shape[1], dtype=vertices_feats.dtype
    )
    merged_vertices.scatter_add_(
        0,
        inverse_indices.unsqueeze(1).expand(-1, vertices_feats.shape[1]),
        vertices_feats,
    )
    merged_vertices = merged_vertices / count.clamp(min=1.0)
    intersected_float = intersected_feats.float()
    merged_intersected = torch.zeros(
        n_unique, intersected_float.shape[1], dtype=intersected_float.dtype
    )
    merged_intersected.scatter_add_(
        0,
        inverse_indices.unsqueeze(1).expand(-1, intersected_float.shape[1]),
        intersected_float,
    )
    merged_intersected = merged_intersected > 0
    return (unique_coords, merged_vertices, merged_intersected)


def load_multi_obj_scene_sparse(
    dual_grid_root: Path, sha256: str
) -> Tuple[sp.SparseTensor, sp.SparseTensor]:
    obj_dir = dual_grid_root / sha256
    meta_path = obj_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json missing: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    all_coords: List[torch.Tensor] = []
    all_vertices: List[torch.Tensor] = []
    all_intersected: List[torch.Tensor] = []
    for obj_info in meta.get("objects", []):
        obj_path = obj_dir / obj_info["file"]
        if not obj_path.exists():
            continue
        coords, attr = o_voxel.io.read_vxz(str(obj_path), num_threads=2)
        if coords.shape[0] == 0:
            continue
        all_coords.append(coords.to(torch.int32))
        all_vertices.append((attr["vertices"] / 255.0).float())
        all_intersected.append(_decode_intersected(attr["intersected"]))
    if len(all_coords) == 0:
        raise RuntimeError(f"No valid voxel objects found under {obj_dir}")
    coords = torch.cat(all_coords, dim=0)
    vertices_feats = torch.cat(all_vertices, dim=0)
    intersected_feats = torch.cat(all_intersected, dim=0)
    coords, vertices_feats, intersected_feats = merge_multi_obj_voxels(
        coords, vertices_feats, intersected_feats
    )
    coords_with_batch = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1).to(
        torch.int32
    )
    vertices = sp.SparseTensor(vertices_feats, coords_with_batch)
    intersected = vertices.replace(intersected_feats)
    return (vertices, intersected)


def is_valid_sparse_tensor(tensor: sp.SparseTensor) -> bool:
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True, help="Directory containing metadata.csv"
    )
    parser.add_argument(
        "--dual_grid_root",
        type=str,
        default=None,
        help="Dual-grid root, defaults to --root",
    )
    parser.add_argument(
        "--shape_latent_root",
        type=str,
        default=None,
        help="Shape-latent root, defaults to --root",
    )
    parser.add_argument("--filter_low_aesthetic_score", type=float, default=None)
    parser.add_argument(
        "--resolution", type=int, default=1024, help="Dual-grid input resolution"
    )
    parser.add_argument(
        "--enc_pretrained",
        type=str,
        default="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16",
    )
    parser.add_argument("--model_root", type=str, default=None)
    parser.add_argument("--enc_model", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.dual_grid_root = opt.dual_grid_root or opt.root
    opt.shape_latent_root = opt.shape_latent_root or opt.root
    if opt.enc_model is None:
        latent_name = f"{opt.enc_pretrained.split('/')[-1]}_{opt.resolution}"
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f"{opt.enc_model.split('/')[-1]}_{opt.ckpt}_{opt.resolution}"
        cfg = edict(
            json.load(
                open(os.path.join(opt.model_root, opt.enc_model, "config.json"), "r")
            )
        )
        encoder = getattr(models, cfg.models.encoder.name)(
            **cfg.models.encoder.args
        ).cuda()
        ckpt_path = os.path.join(
            opt.model_root, opt.enc_model, "ckpts", f"encoder_{opt.ckpt}.pt"
        )
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f"Loaded model from {ckpt_path}")
    out_dir = Path(opt.shape_latent_root) / "shape_latents" / latent_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "new_records").mkdir(parents=True, exist_ok=True)
    metadata_path = Path(opt.root) / "metadata.csv"
    if not metadata_path.exists():
        raise ValueError(f"metadata.csv not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path).set_index("sha256")
    aesthetic_meta = Path(opt.root) / "aesthetic_scores" / "metadata.csv"
    if aesthetic_meta.exists():
        metadata = metadata.combine_first(
            pd.read_csv(aesthetic_meta).set_index("sha256")
        )
    dual_grid_meta = (
        Path(opt.dual_grid_root) / f"dual_grid_{opt.resolution}" / "metadata.csv"
    )
    if dual_grid_meta.exists():
        metadata = metadata.combine_first(
            pd.read_csv(dual_grid_meta).set_index("sha256")
        )
    shape_meta = out_dir / "metadata.csv"
    if shape_meta.exists():
        metadata = metadata.combine_first(pd.read_csv(shape_meta).set_index("sha256"))
    metadata = metadata.reset_index()
    if opt.instances is None:
        if (
            opt.filter_low_aesthetic_score is not None
            and "aesthetic_score" in metadata.columns
        ):
            metadata = metadata[
                metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score
            ]
        if "dual_grid_converted" not in metadata.columns:
            dual_grid_dir = Path(opt.dual_grid_root) / f"dual_grid_{opt.resolution}"
            if not dual_grid_dir.exists():
                raise ValueError(
                    f"dual_grid_{opt.resolution} not found under {opt.dual_grid_root}. Please run dual_grid_multi_obj first."
                )
            metadata["dual_grid_converted"] = metadata["sha256"].apply(
                lambda x: (dual_grid_dir / str(x) / "meta.json").exists()
            )
        metadata = metadata[metadata["dual_grid_converted"] == True]
        if "shape_latent_encoded" in metadata.columns:
            metadata = metadata[metadata["shape_latent_encoded"] != True]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]
    records: List[Dict[str, object]] = []
    existing_sha256s = set()
    for f in out_dir.iterdir():
        if f.suffix == ".npz":
            sha = f.stem
            existing_sha256s.add(sha)
            try:
                coords = np.load(f)["coords"]
                records.append(
                    {
                        "sha256": sha,
                        "shape_latent_encoded": True,
                        "shape_latent_tokens": int(coords.shape[0]),
                    }
                )
            except Exception:
                pass
    metadata = metadata[~metadata["sha256"].isin(existing_sha256s)]
    print(f"Found {len(existing_sha256s)} existing latents, {len(metadata)} remaining.")
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    print(f"Rank {opt.rank}/{opt.world_size}: processing {len(metadata)} instances...")
    dual_grid_root = Path(opt.dual_grid_root) / f"dual_grid_{opt.resolution}"
    for sha256 in tqdm(
        metadata["sha256"].values, desc="Encoding multi-object shape latents"
    ):
        try:
            vertices, intersected = load_multi_obj_scene_sparse(dual_grid_root, sha256)
            if not (
                is_valid_sparse_tensor(vertices) and is_valid_sparse_tensor(intersected)
            ):
                print(f"[Skip] {sha256}: non-finite sparse tensor")
                continue
            z = encoder(vertices.cuda(), intersected.cuda())
            torch.cuda.synchronize()
            if not torch.isfinite(z.feats).all():
                print(f"[Skip] {sha256}: non-finite latent feats")
                clear_cuda_error()
                continue
            pack = {
                "feats": z.feats.cpu().numpy().astype(np.float32),
                "coords": z.coords[:, 1:].cpu().numpy().astype(np.uint8),
            }
            np.savez_compressed(out_dir / f"{sha256}.npz", **pack)
            records.append(
                {
                    "sha256": sha256,
                    "shape_latent_encoded": True,
                    "shape_latent_tokens": int(pack["coords"].shape[0]),
                }
            )
        except Exception as e:
            print(f"[Error] {sha256}: {e}")
            clear_cuda_error()
            continue
    pd.DataFrame.from_records(
        records, columns=["sha256", "shape_latent_encoded", "shape_latent_tokens"]
    ).to_csv(out_dir / "new_records" / f"part_{opt.rank}.csv", index=False)


if __name__ == "__main__":
    main()
