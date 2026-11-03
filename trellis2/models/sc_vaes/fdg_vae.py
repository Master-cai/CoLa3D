from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from .sparse_unet_vae import (
    SparseResBlock3d,
    SparseConvNeXtBlock3d,
    SparseResBlockDownsample3d,
    SparseResBlockUpsample3d,
    SparseResBlockS2C3d,
    SparseResBlockC2S3d,
)
from .sparse_unet_vae import (
    SparseUnetVaeEncoder,
    SparseUnetVaeDecoder,
)
from ...representations import Mesh
from o_voxel.convert import flexible_dual_grid_to_mesh


def merge_overlapping_voxels(
    vertices: sp.SparseTensor,
    intersected: sp.SparseTensor,
    instance_id: Optional[torch.Tensor] = None,
    max_objects: int = 8,
) -> Tuple[sp.SparseTensor, sp.SparseTensor, Optional[torch.Tensor]]:


    coords = vertices.coords
    vert_feats = vertices.feats
    inter_feats = intersected.feats.float()

    n_original = coords.shape[0]
    device = coords.device


    unique_coords, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)
    n_unique = unique_coords.shape[0]


    if n_unique == n_original:
        if instance_id is not None:

            multi_hot = torch.zeros(
                n_original, max_objects, device=device, dtype=torch.float32
            )
            valid_mask = (instance_id >= 0) & (instance_id < max_objects)
            multi_hot[valid_mask, instance_id[valid_mask]] = 1.0
            return vertices, intersected, multi_hot
        return vertices, intersected, None


    count = torch.zeros(n_unique, 1, device=device, dtype=vert_feats.dtype)
    count.scatter_add_(
        0,
        inverse_indices.unsqueeze(1),
        torch.ones(n_original, 1, device=device, dtype=vert_feats.dtype),
    )


    merged_vert_feats = torch.zeros(n_unique, 3, device=device, dtype=vert_feats.dtype)
    merged_vert_feats.scatter_add_(
        0, inverse_indices.unsqueeze(1).expand(-1, 3), vert_feats
    )
    merged_vert_feats = merged_vert_feats / count.clamp(min=1)


    merged_inter_feats = torch.zeros(
        n_unique, 3, device=device, dtype=inter_feats.dtype
    )
    merged_inter_feats.scatter_add_(
        0, inverse_indices.unsqueeze(1).expand(-1, 3), inter_feats
    )
    merged_inter_feats = merged_inter_feats > 0


    multi_hot = None
    if instance_id is not None:
        multi_hot = torch.zeros(
            n_unique, max_objects, device=device, dtype=torch.float32
        )
        valid_mask = (instance_id >= 0) & (instance_id < max_objects)

        valid_inverse = inverse_indices[valid_mask]
        valid_ids = instance_id[valid_mask].long()
        multi_hot[valid_inverse, valid_ids] = 1.0


    sort_indices = torch.argsort(unique_coords[:, 0])
    unique_coords = unique_coords[sort_indices]
    merged_vert_feats = merged_vert_feats[sort_indices]
    merged_inter_feats = merged_inter_feats[sort_indices]
    if multi_hot is not None:
        multi_hot = multi_hot[sort_indices]


    merged_vertices = sp.SparseTensor(
        feats=merged_vert_feats,
        coords=unique_coords,
        shape=vertices.shape,
        layout=None,
    )
    merged_intersected = merged_vertices.replace(merged_inter_feats)

    return merged_vertices, merged_intersected, multi_hot


class FlexiDualGridVaeEncoder(SparseUnetVaeEncoder):
    def __init__(
        self,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        use_dtype: Optional[str] = None,
    ):
        super().__init__(
            6,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            down_block_type,
            block_args,
            use_fp16=use_fp16,
            use_dtype=use_dtype,
        )

    def forward(
        self,
        vertices: sp.SparseTensor,
        intersected: sp.SparseTensor,
        sample_posterior=False,
        return_raw=False,
    ):


        x = vertices.replace(
            torch.cat(
                [
                    vertices.feats - 0.5,
                    intersected.feats.float() - 0.5,
                ],
                dim=1,
            )
        )
        return super().forward(x, sample_posterior, return_raw)


class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
        use_dtype: Optional[str] = None,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin

        super().__init__(
            7,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            up_block_type,
            block_args,
            use_fp16=use_fp16,
            use_dtype=use_dtype,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution

    def forward(
        self, x: sp.SparseTensor, gt_intersected: sp.SparseTensor = None, **kwargs
    ):
        decoded = super().forward(x, **kwargs)
        if self.training:
            h, subs_gt, subs = decoded
            vertices = h.replace(
                (1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3])
                - self.voxel_margin
            )
            intersected_logits = h.replace(h.feats[..., 3:6])
            quad_lerp = h.replace(
                F.softplus(h.feats[..., 6:7])
            )
            mesh = [
                Mesh(
                    *flexible_dual_grid_to_mesh(
                        v.coords[:, 1:],
                        v.feats,
                        i.feats,
                        q.feats,
                        aabb=[
                            [-0.5, -0.5, -0.5],
                            [0.5, 0.5, 0.5],
                        ],
                        grid_size=self.resolution,
                        train=True,
                    )
                )
                for v, i, q in zip(vertices, gt_intersected, quad_lerp)
            ]
            return mesh, vertices, intersected_logits, subs_gt, subs
        else:
            out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
            h = out_list[0]
            vertices = h.replace(
                (1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3])
                - self.voxel_margin
            )
            intersected = h.replace(
                h.feats[..., 3:6] > 0
            )
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            mesh = [
                Mesh(
                    *flexible_dual_grid_to_mesh(
                        v.coords[:, 1:],
                        v.feats,
                        i.feats,
                        q.feats,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        grid_size=self.resolution,
                        train=False,
                    )
                )
                for v, i, q in zip(vertices, intersected, quad_lerp)
            ]
            out_list[0] = mesh
            return out_list[0] if len(out_list) == 1 else tuple(out_list)
