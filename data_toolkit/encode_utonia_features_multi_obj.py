

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from encode_shape_latent_multi_obj import load_multi_obj_scene_sparse

torch.set_grad_enabled(False)


def clear_cuda_error() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Encode Utonia level-4 features for offline fusion"
    )
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
        required=True,
        help="Path to existing SLat shape_latent npz dir",
    )
    parser.add_argument(
        "--utonia_output_root",
        type=str,
        default=None,
        help="Output dir, defaults to {root}/utonia_feats_lvl4_fullres",
    )
    parser.add_argument("--utonia_model", type=str, default="Pointcept/Utonia")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    opt = parser.parse_args()
    opt.dual_grid_root = opt.dual_grid_root or opt.root
    if opt.utonia_output_root is None:
        out_dir = Path(opt.root) / "utonia_feats_lvl4_fullres"
    else:
        out_dir = Path(opt.utonia_output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "new_records").mkdir(parents=True, exist_ok=True)
    from trellis2.modules.ext_backbones import UtoniaFeatureExtractor

    extractor = UtoniaFeatureExtractor(
        model_name=opt.utonia_model, world_span=12.0, resolution=opt.resolution
    ).cuda()
    bottleneck_dim = extractor.bottleneck_dim
    print(f"Utonia bottleneck_dim={bottleneck_dim}")
    metadata_path = Path(opt.root) / "metadata.csv"
    metadata = pd.read_csv(metadata_path).set_index("sha256")
    dual_grid_meta = (
        Path(opt.dual_grid_root) / f"dual_grid_{opt.resolution}" / "metadata.csv"
    )
    if dual_grid_meta.exists():
        metadata = metadata.combine_first(
            pd.read_csv(dual_grid_meta).set_index("sha256")
        )
    utonia_meta = out_dir / "metadata.csv"
    if utonia_meta.exists():
        metadata = metadata.combine_first(pd.read_csv(utonia_meta).set_index("sha256"))
    metadata = metadata.reset_index()
    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]
    else:
        if "dual_grid_converted" not in metadata.columns:
            dual_grid_dir = Path(opt.dual_grid_root) / f"dual_grid_{opt.resolution}"
            metadata["dual_grid_converted"] = metadata["sha256"].apply(
                lambda x: (dual_grid_dir / str(x) / "meta.json").exists()
            )
        metadata = metadata[metadata["dual_grid_converted"] == True]
        if "utonia_encoded" in metadata.columns:
            metadata = metadata[metadata["utonia_encoded"] != True]
    existing = {f.stem for f in out_dir.iterdir() if f.suffix == ".npz"}
    metadata = metadata[~metadata["sha256"].isin(existing)]
    print(f"Found {len(existing)} existing, {len(metadata)} remaining.")
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    print(f"Rank {opt.rank}/{opt.world_size}: processing {len(metadata)} instances...")
    dual_grid_root = Path(opt.dual_grid_root) / f"dual_grid_{opt.resolution}"
    shape_latent_dir = Path(opt.shape_latent_root)
    records: List[Dict] = []
    skipped_spconv = 0
    desc = f"[R{opt.rank}/{opt.world_size}] Encoding"
    pbar = tqdm(
        metadata["sha256"].values,
        desc=desc,
        dynamic_ncols=True,
        position=opt.rank,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for sha256 in pbar:
        try:
            slat_path = shape_latent_dir / f"{sha256}.npz"
            if not slat_path.exists():
                continue
            vertices, _ = load_multi_obj_scene_sparse(dual_grid_root, sha256)
            raw_coords = vertices.coords.to(torch.int32)
            with np.load(str(slat_path)) as npz:
                slat_coords = npz["coords"].astype(np.int32)
            colors_t: torch.Tensor | None = None
            normals_t: torch.Tensor | None = None
            utonia_feats, utonia_gc = extractor.forward_level4(
                coords=raw_coords.cuda(),
                batch_size=1,
                colors=colors_t,
                normals=normals_t,
            )
            utonia_gc_np = utonia_gc.cpu().numpy().astype(np.int32)
            utonia_feats_np = utonia_feats.cpu().float().numpy()
            slat_set = set(map(tuple, slat_coords))
            utonia_set = set(map(tuple, utonia_gc_np))
            if slat_set != utonia_set:
                print(
                    f"[Skip] {sha256}: coord set mismatch (slat={len(slat_set)}, utonia={len(utonia_set)}, diff={len(slat_set ^ utonia_set)})"
                )
                continue
            utonia_idx = {tuple(c): i for i, c in enumerate(utonia_gc_np)}
            try:
                perm = np.array([utonia_idx[tuple(c)] for c in slat_coords])
            except KeyError as e:
                print(
                    f"[Skip] {sha256}: SLat coord {e} missing from Utonia output (alignment bug?)"
                )
                continue
            utonia_feats_aligned = utonia_feats_np[perm]
            record: Dict = {
                "sha256": sha256,
                "utonia_encoded": True,
                "utonia_tokens": int(slat_coords.shape[0]),
            }
            np.savez_compressed(
                out_dir / f"{sha256}.npz",
                feats=utonia_feats_aligned.astype(np.float16),
                coords=slat_coords.astype(np.uint8),
            )
            records.append(record)
        except RuntimeError as e:
            err_msg = str(e)
            if "int32 range" in err_msg or "int_max" in err_msg:
                skipped_spconv += 1
                print(
                    f"[Skip-spconv] {sha256}: spconv int32 overflow ({vertices.coords.shape[0]} voxels)"
                )
                clear_cuda_error()
                continue
            print(f"[Error] {sha256}: {e}")
            clear_cuda_error()
            continue
        except Exception as e:
            print(f"[Error] {sha256}: {e}")
            clear_cuda_error()
            continue
    if skipped_spconv > 0:
        print(
            f"\n[Summary] Skipped {skipped_spconv} instances due to spconv int32 overflow"
        )
    record_columns = ["sha256", "utonia_encoded", "utonia_tokens"]
    pd.DataFrame.from_records(records, columns=record_columns).to_csv(
        out_dir / "new_records" / f"part_{opt.rank}.csv", index=False
    )
    print(f"Done. Encoded {len(records)} instances.")


if __name__ == "__main__":
    main()
