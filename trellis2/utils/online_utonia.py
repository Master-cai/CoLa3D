

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch

from trellis2.modules import sparse as sp
from trellis2.utils.timing import stage_timer as _sub_timer

if TYPE_CHECKING:
    from trellis2.modules.ext_backbones import UtoniaFeatureExtractor
    from trellis2.representations.mesh.base import Mesh


logger = logging.getLogger(__name__)

CANONICAL_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
CANONICAL_FACE_WEIGHT = 1.0
CANONICAL_BOUNDARY_WEIGHT = 0.2
CANONICAL_REGULARIZATION_WEIGHT = 1e-2


def build_utonia_extractor(
    model_name: str = "Pointcept/Utonia",
    world_span: float = 12.0,
    resolution: int = 1024,
    device: str = "cuda",
) -> "UtoniaFeatureExtractor":

    from trellis2.modules.ext_backbones import UtoniaFeatureExtractor

    extractor = UtoniaFeatureExtractor(
        model_name=model_name,
        world_span=world_span,
        resolution=resolution,
    )
    return extractor.to(device)


@torch.no_grad()
def decode_scene_mesh(
    shape_slat: sp.SparseTensor,
    decoder: torch.nn.Module,
    resolution: int = 1024,
    return_subs: bool = False,
):


    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder

    prev_res = getattr(decoder, "resolution", None)
    decoder.set_resolution(resolution)
    try:
        out = FlexiDualGridVaeDecoder.forward(
            decoder,
            shape_slat,
            return_subs=return_subs,
        )
    finally:
        if prev_res is not None and prev_res != resolution:
            decoder.set_resolution(prev_res)

    if return_subs:
        if not isinstance(out, tuple) or len(out) != 2:
            raise RuntimeError(
                "FlexiDualGridVaeDecoder.forward(return_subs=True) must return (mesh_list, subs)"
            )
        mesh_list, subs = out
        if not mesh_list:
            raise RuntimeError("FlexiDualGridVaeDecoder.forward returned no mesh")
        return mesh_list[0], subs

    mesh_list = out if isinstance(out, list) else out[0]
    if not mesh_list:
        raise RuntimeError("FlexiDualGridVaeDecoder.forward returned no mesh")
    return mesh_list[0]


def voxelize_mesh_canonical(
    mesh: "Mesh",
    resolution: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:


    import o_voxel

    vertices = mesh.vertices.detach().cpu().float()
    faces = mesh.faces.detach().cpu().long()

    voxel_indices, dual_vertices, _ = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices,
        faces,
        grid_size=resolution,
        aabb=CANONICAL_AABB,
        face_weight=CANONICAL_FACE_WEIGHT,
        boundary_weight=CANONICAL_BOUNDARY_WEIGHT,
        regularization_weight=CANONICAL_REGULARIZATION_WEIGHT,
        timing=False,
    )
    return voxel_indices, dual_vertices


def compute_mesh_normals_at_voxels(
    mesh: "Mesh",
    voxel_indices: torch.Tensor,
    dual_vertices: torch.Tensor,
    resolution: int = 1024,
) -> torch.Tensor:


    import trimesh

    vertices = mesh.vertices.detach().cpu().float().numpy()
    faces = mesh.faces.detach().cpu().long().numpy()
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    face_normals = np.asarray(tri.face_normals, dtype=np.float32)

    vi = voxel_indices.detach().cpu().numpy().astype(np.float32)
    dv = dual_vertices.detach().cpu().numpy().astype(np.float32)
    p = (vi + dv) / float(resolution) - 0.5

    closest, _, face_ids = trimesh.proximity.closest_point(tri, p)
    normals = face_normals[face_ids]

    n = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.where(n > 1e-6, normals / np.clip(n, 1e-6, None), 0.0)
    return torch.from_numpy(normals.astype(np.float32))


def align_pbr_voxel_to_indices(
    pbr_voxel: sp.SparseTensor,
    voxel_indices: torch.Tensor,
    pbr_attr_layout: Optional[dict] = None,
) -> Tuple[torch.Tensor, float]:


    if pbr_attr_layout is None:
        pbr_attr_layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }
    base_color_slice = pbr_attr_layout["base_color"]
    assert pbr_voxel.feats.shape[1] == 6, (
        f"pbr_voxel.feats expected 6 channels, got {pbr_voxel.feats.shape[1]}. "
        "Check the texture decoder attribute layout."
    )
    assert base_color_slice == slice(
        0, 3
    ), f"base_color slice expected (0, 3), got {base_color_slice}"

    pbr_xyz = pbr_voxel.coords[:, 1:].detach().cpu().numpy().astype(np.int32)
    tgt = voxel_indices.detach().cpu().numpy().astype(np.int32)

    pbr_idx = {(int(c[0]), int(c[1]), int(c[2])): i for i, c in enumerate(pbr_xyz)}
    n = len(tgt)
    perm = np.full(n, -1, dtype=np.int64)
    for j, c in enumerate(tgt):
        hit = pbr_idx.get((int(c[0]), int(c[1]), int(c[2])))
        if hit is not None:
            perm[j] = hit

    hit_mask = perm >= 0
    hits = int(hit_mask.sum())
    hit_rate = hits / max(n, 1)

    base_color = (
        pbr_voxel.feats[:, base_color_slice].detach().float()
    )
    colors = torch.zeros(n, 3, dtype=torch.float32, device=base_color.device)
    if hits > 0:
        hit_rows = torch.from_numpy(np.nonzero(hit_mask)[0]).to(colors.device)
        perm_hit = torch.from_numpy(perm[hit_mask]).to(colors.device)
        colors[hit_rows] = base_color[perm_hit]
    return colors, hit_rate


def mesh_to_level4_features(
    mesh: "Mesh",
    extractor: "UtoniaFeatureExtractor",
    resolution: int = 1024,
    colors: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:


    voxel_indices, _ = voxelize_mesh_canonical(mesh, resolution=resolution)


    device = next(extractor.parameters()).device
    coords_with_batch = (
        torch.cat(
            [torch.zeros_like(voxel_indices[:, :1]), voxel_indices],
            dim=-1,
        )
        .to(torch.int32)
        .to(device)
    )

    colors_d = colors.to(device) if colors is not None else None
    normals_d = normals.to(device) if normals is not None else None

    return extractor.forward_level4(
        coords_with_batch,
        batch_size=1,
        colors=colors_d,
        normals=normals_d,
    )


def align_to_slat(
    utonia_feats: torch.Tensor,
    utonia_gc: torch.Tensor,
    slat_coords: torch.Tensor,
) -> Tuple[torch.Tensor, float]:


    utonia_gc_np = utonia_gc.detach().cpu().numpy().astype(np.int32)
    slat_np = slat_coords.detach().cpu().numpy()
    if slat_np.ndim != 2 or slat_np.shape[1] not in (3, 4):
        raise ValueError(f"slat_coords must be (N,3) or (N,4), got {slat_np.shape}")
    if slat_np.shape[1] == 4:
        slat_np = slat_np[:, 1:]
    slat_np = slat_np.astype(np.int32)

    utonia_idx = {tuple(c): i for i, c in enumerate(utonia_gc_np)}

    n = len(slat_np)
    perm = np.full(n, -1, dtype=np.int64)
    for j in range(n):
        key = (int(slat_np[j, 0]), int(slat_np[j, 1]), int(slat_np[j, 2]))
        match = utonia_idx.get(key)
        if match is not None:
            perm[j] = match

    hit_mask = perm >= 0
    hits = int(hit_mask.sum())
    hit_rate = hits / max(n, 1)

    D = utonia_feats.shape[1]
    aligned = torch.zeros(n, D, device=utonia_feats.device, dtype=utonia_feats.dtype)
    if hits > 0:
        hit_rows = torch.from_numpy(np.nonzero(hit_mask)[0]).to(aligned.device)
        perm_hit = torch.from_numpy(perm[hit_mask]).to(aligned.device)
        aligned[hit_rows] = utonia_feats[perm_hit]

    return aligned, hit_rate


@torch.no_grad()
def compute_online_utonia_features(
    shape_slat: sp.SparseTensor,
    decoder: torch.nn.Module,
    extractor: "UtoniaFeatureExtractor",
    *,
    vertices_override: Optional[sp.SparseTensor] = None,
    resolution: int = 1024,
    warn_hit_rate_below: float = 0.8,
    tex_decoder: Optional[torch.nn.Module] = None,
    tex_slat: Optional[sp.SparseTensor] = None,
    pbr_attr_layout: Optional[dict] = None,
    enable_normal: bool = False,
    timings: Optional[dict] = None,
) -> sp.SparseTensor:


    device = next(extractor.parameters()).device

    if vertices_override is not None:
        coords_with_batch = vertices_override.coords.to(torch.int32).to(device)
        with _sub_timer(timings, "utonia_forward"):
            utonia_feats, utonia_gc = extractor.forward_level4(
                coords_with_batch, batch_size=1
            )
        source_tag = "encoder-GT voxels"
        enrich_tag = ""
    else:


        need_subs = tex_decoder is not None and tex_slat is not None
        with _sub_timer(timings, "utonia_decode_mesh"):
            if need_subs:
                mesh, subs = decode_scene_mesh(
                    shape_slat,
                    decoder,
                    resolution=resolution,
                    return_subs=True,
                )
            else:
                mesh = decode_scene_mesh(shape_slat, decoder, resolution=resolution)
                subs = None
        with _sub_timer(timings, "utonia_voxelize"):
            voxel_indices, dual_vertices = voxelize_mesh_canonical(
                mesh, resolution=resolution
            )

        colors = None
        color_hit_rate = float("nan")
        if tex_decoder is not None and tex_slat is not None:
            with _sub_timer(timings, "utonia_color"):
                pbr_voxel = tex_decoder(tex_slat, guide_subs=subs)
                pbr_voxel = pbr_voxel.replace(pbr_voxel.feats * 0.5 + 0.5)
                colors, color_hit_rate = align_pbr_voxel_to_indices(
                    pbr_voxel,
                    voxel_indices,
                    pbr_attr_layout=pbr_attr_layout,
                )

        normals = None
        if enable_normal:
            with _sub_timer(timings, "utonia_normal"):
                normals = compute_mesh_normals_at_voxels(
                    mesh,
                    voxel_indices,
                    dual_vertices,
                    resolution=resolution,
                )

        coords_with_batch = (
            torch.cat(
                [torch.zeros_like(voxel_indices[:, :1]), voxel_indices],
                dim=-1,
            )
            .to(torch.int32)
            .to(device)
        )
        colors_d = colors.to(device) if colors is not None else None
        normals_d = normals.to(device) if normals is not None else None
        with _sub_timer(timings, "utonia_forward"):
            utonia_feats, utonia_gc = extractor.forward_level4(
                coords_with_batch,
                batch_size=1,
                colors=colors_d,
                normals=normals_d,
            )
        source_tag = "decoded mesh"
        parts = []
        if normals is not None:
            parts.append("normal")
        if colors is not None:
            parts.append(f"color(hit={color_hit_rate:.3f})")
        enrich_tag = f" enrich=[{','.join(parts)}]" if parts else ""

    with _sub_timer(timings, "utonia_align"):
        aligned, hit_rate = align_to_slat(utonia_feats, utonia_gc, shape_slat.coords)

    n = aligned.shape[0]
    hits = int(round(hit_rate * n))
    log = logger.warning if hit_rate < warn_hit_rate_below else logger.debug
    log(
        "[online-utonia] source=%s hits=%d/%d (%.3f)%s",
        source_tag, hits, n, hit_rate, enrich_tag,
    )

    return shape_slat.replace(aligned)
