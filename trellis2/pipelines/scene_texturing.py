


from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..modules.sparse import SparseTensor
from ..representations import Mesh, MeshWithVoxel
from . import samplers


PBR_ATTR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


def _filter_mesh_components_for_export(
    mesh_with_voxel: MeshWithVoxel,
    min_faces: int,
) -> Tuple[MeshWithVoxel, dict]:


    import trimesh

    vertices_np = mesh_with_voxel.vertices.detach().cpu().numpy()
    faces_np = mesh_with_voxel.faces.detach().cpu().numpy()
    if len(faces_np) == 0:
        return mesh_with_voxel, {
            "component_count": 0,
            "kept_components": 0,
            "faces_before": 0,
            "faces_after": 0,
            "vertices_before": int(len(vertices_np)),
            "vertices_after": int(len(vertices_np)),
            "min_faces": int(min_faces),
        }

    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
    face_nodes = np.arange(len(faces_np), dtype=np.int64)
    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        min_len=1,
        nodes=face_nodes,
        engine=None,
    )
    if len(components) <= 1:
        return mesh_with_voxel, {
            "component_count": int(len(components)),
            "kept_components": int(len(components)),
            "faces_before": int(len(faces_np)),
            "faces_after": int(len(faces_np)),
            "vertices_before": int(len(vertices_np)),
            "vertices_after": int(len(vertices_np)),
            "min_faces": int(min_faces),
        }

    sizes = np.array([len(c) for c in components], dtype=np.int64)
    largest = int(sizes.max())
    keep = [c for c, s in zip(components, sizes) if s >= min_faces or s == largest]
    keep_face_indices = np.concatenate(keep).astype(np.int64)
    keep_face_indices.sort()

    kept_faces = faces_np[keep_face_indices]
    unique_vertices, inverse = np.unique(kept_faces.reshape(-1), return_inverse=True)
    filtered_vertices = vertices_np[unique_vertices]
    filtered_faces = inverse.reshape(-1, 3).astype(faces_np.dtype, copy=False)

    device = mesh_with_voxel.vertices.device
    filtered = MeshWithVoxel(
        torch.as_tensor(
            filtered_vertices,
            dtype=mesh_with_voxel.vertices.dtype,
            device=device,
        ),
        torch.as_tensor(
            filtered_faces,
            dtype=mesh_with_voxel.faces.dtype,
            device=device,
        ),
        origin=mesh_with_voxel.origin.detach().cpu().tolist(),
        voxel_size=mesh_with_voxel.voxel_size,
        coords=mesh_with_voxel.coords,
        attrs=mesh_with_voxel.attrs,
        voxel_shape=mesh_with_voxel.voxel_shape,
        layout=mesh_with_voxel.layout,
    )
    stats = {
        "component_count": int(len(components)),
        "kept_components": int(len(keep)),
        "faces_before": int(len(faces_np)),
        "faces_after": int(len(filtered_faces)),
        "vertices_before": int(len(vertices_np)),
        "vertices_after": int(len(filtered_vertices)),
        "min_faces": int(min_faces),
    }
    return filtered, stats


@torch.no_grad()
def sample_scene_tex_slat(
    cond: dict,
    tex_flow_model: nn.Module,
    shape_slat: SparseTensor,
    shape_slat_normalization: dict,
    tex_slat_normalization: dict,
    sampler: samplers.Sampler,
    sampler_params: dict,
    verbose: bool = True,
) -> SparseTensor:


    device = shape_slat.feats.device


    shape_mean = torch.tensor(shape_slat_normalization["mean"], device=device)[None]
    shape_std = torch.tensor(shape_slat_normalization["std"], device=device)[None]
    shape_slat_norm = (shape_slat - shape_mean) / shape_std

    in_channels = tex_flow_model.in_channels
    noise_channels = in_channels - shape_slat_norm.feats.shape[1]
    noise = shape_slat_norm.replace(
        feats=torch.randn(
            shape_slat_norm.coords.shape[0], noise_channels, device=device
        )
    )

    tex_slat = sampler.sample(
        tex_flow_model,
        noise,
        concat_cond=shape_slat_norm,
        **cond,
        **sampler_params,
        verbose=verbose,
        tqdm_desc="Sampling scene tex SLat",
    ).samples

    tex_mean = torch.tensor(tex_slat_normalization["mean"], device=tex_slat.device)[
        None
    ]
    tex_std = torch.tensor(tex_slat_normalization["std"], device=tex_slat.device)[None]
    tex_slat = tex_slat * tex_std + tex_mean
    return tex_slat


@torch.no_grad()
def decode_tex_voxel(
    tex_slat: SparseTensor,
    subs: list,
    tex_decoder: nn.Module,
) -> SparseTensor:


    return tex_decoder(tex_slat, guide_subs=subs) * 0.5 + 0.5


@torch.no_grad()
def wrap_textured_mesh(
    mesh: Mesh,
    pbr_voxel: SparseTensor,
    resolution: int,
) -> MeshWithVoxel:


    mesh.fill_holes()
    return MeshWithVoxel(
        mesh.vertices,
        mesh.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / resolution,
        coords=pbr_voxel.coords[:, 1:],
        attrs=pbr_voxel.feats,
        voxel_shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
        layout=PBR_ATTR_LAYOUT,
    )


@torch.no_grad()
def decode_textured_mesh(
    shape_slat_subset: SparseTensor,
    tex_slat_subset: SparseTensor,
    shape_decoder: nn.Module,
    tex_decoder: nn.Module,
    resolution: int,
) -> Optional[MeshWithVoxel]:


    from ..models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder

    result = FlexiDualGridVaeDecoder.forward(
        shape_decoder, shape_slat_subset, return_subs=True
    )

    if not isinstance(result, tuple) or len(result) != 2:
        return None
    mesh_list, subs = result
    if not isinstance(mesh_list, list) or not mesh_list:
        return None
    mesh = mesh_list[0]
    if mesh is None or mesh.vertices.shape[0] == 0:
        return None

    pbr_voxel = decode_tex_voxel(tex_slat_subset, subs, tex_decoder)
    return wrap_textured_mesh(mesh, pbr_voxel, resolution)


def save_textured_mesh_glb(
    mesh_with_voxel: MeshWithVoxel,
    path: str,
    texture_size: int = 1024,
    decimation_target: int = 200_000,
    min_component_faces: int = 10,
) -> None:


    import o_voxel

    if min_component_faces > 0:
        mesh_with_voxel, stats = _filter_mesh_components_for_export(
            mesh_with_voxel,
            min_faces=min_component_faces,
        )
        if (
            stats["component_count"] > 1
            and stats["faces_before"] != stats["faces_after"]
        ):
            print(
                f"  [tex-export] component-filter: "
                f"components={stats['component_count']} kept={stats['kept_components']} "
                f"faces={stats['faces_before']}->{stats['faces_after']} "
                f"verts={stats['vertices_before']}->{stats['vertices_after']} "
                f"(min_faces={stats['min_faces']})"
            )

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh_with_voxel.vertices,
        faces=mesh_with_voxel.faces,
        attr_volume=mesh_with_voxel.attrs,
        coords=mesh_with_voxel.coords,
        attr_layout=mesh_with_voxel.layout,
        voxel_size=mesh_with_voxel.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    glb.export(path, extension_webp=True)
