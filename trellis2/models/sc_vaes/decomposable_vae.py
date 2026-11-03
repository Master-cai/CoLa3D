


from typing import List, Dict, Any, Optional, Tuple, Union
from fractions import Fraction
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import sparse as sp
from ...modules.norm import GroupNorm32, LayerNorm32
from ...modules.utils import convert_module_to

from ...modules.sparse.transformer.cola3d_backbone import (
    LearnableFourierPE,
    CoLa3DBackbone,
    PartIndexEmbedder,
)
from ...representations import Mesh

from .fdg_vae import FlexiDualGridVaeDecoder
from .sparse_unet_vae import SparseUnetVaeDecoder

try:
    from o_voxel.convert import flexible_dual_grid_to_mesh
except ImportError:
    flexible_dual_grid_to_mesh = None


logger = logging.getLogger(__name__)

__all__ = [
    "MaskPromptEncoder",
    "PointPromptEncoder",
    "MaskDecoder",
    "DecomposableVaeDecoder",
]


class MaskPromptEncoder(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        pe_dim_per_axis: int = 128,
        encoder_type: str = "cnn",
    ):
        super().__init__()
        assert encoder_type in (
            "cnn",
            "cnn_centerpe",
        ), f"encoder_type must be 'cnn' or 'cnn_centerpe', got {encoder_type}"
        self.embed_dim = embed_dim
        self.pe_dim_per_axis = pe_dim_per_axis
        self.encoder_type = encoder_type
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 4, kernel_size=3, stride=2, padding=1),
            GroupNorm32(1, hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, padding=1
            ),
            GroupNorm32(1, hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            GroupNorm32(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
        )
        if encoder_type == "cnn_centerpe":
            self.cnn_norm = LayerNorm32(embed_dim, eps=1e-6)
            self.pe_mlp = nn.Sequential(
                nn.Linear(2 * pe_dim_per_axis, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.pe_norm = LayerNorm32(embed_dim, eps=1e-6)
            self.fuse = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

    @staticmethod
    def _mask_centroid(prompt_masks: torch.Tensor) -> torch.Tensor:


        _, _, height, width = prompt_masks.shape
        device = prompt_masks.device
        h_grid = torch.arange(height, device=device, dtype=torch.float32).view(
            1, 1, height, 1
        )
        w_grid = torch.arange(width, device=device, dtype=torch.float32).view(
            1, 1, 1, width
        )
        m = prompt_masks.to(torch.float32)
        mass = m.sum(dim=(-2, -1)).clamp_min(1.0)
        cy = (h_grid * m).sum(dim=(-2, -1)) / mass / max(height - 1, 1)
        cx = (w_grid * m).sum(dim=(-2, -1)) / mass / max(width - 1, 1)
        return torch.stack([cy, cx], dim=-1)

    @staticmethod
    def _sinusoidal_2d(centers: torch.Tensor, pe_dim_per_axis: int) -> torch.Tensor:

        emb_y = PartIndexEmbedder.sinusoidal_embedding(centers[..., 0], pe_dim_per_axis)
        emb_x = PartIndexEmbedder.sinusoidal_embedding(centers[..., 1], pe_dim_per_axis)
        return torch.cat([emb_y, emb_x], dim=-1)

    def forward(
        self,
        prompt_masks: torch.Tensor,
        valid_query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:


        batch_size, num_queries, height, width = prompt_masks.shape

        x = prompt_masks.reshape(batch_size * num_queries, 1, height, width)
        x = self.encoder(x)
        x = x.mean(dim=(-2, -1))
        cnn_feat = x.reshape(batch_size, num_queries, self.embed_dim)

        if self.encoder_type == "cnn":
            out = cnn_feat
            if valid_query_mask is not None:
                out = out * valid_query_mask.unsqueeze(-1).to(out.dtype)
            return out


        cnn_feat = self.cnn_norm(cnn_feat)


        centers = self._mask_centroid(prompt_masks)
        pe_in = self._sinusoidal_2d(centers, self.pe_dim_per_axis)
        pe_in = pe_in.to(self.pe_mlp[0].weight.dtype)
        pe_feat = self.pe_mlp(pe_in)
        pe_feat = self.pe_norm(pe_feat)


        out = self.fuse(cnn_feat + pe_feat)

        if valid_query_mask is not None:
            out = out * valid_query_mask.unsqueeze(-1).to(out.dtype)
        return out


class PointPromptEncoder(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        latent_dim: int,
        coord_resolution: int,
        num_fourier_feats: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.coord_encoder = LearnableFourierPE(
            hidden_dim=embed_dim,
            coord_dim=3,
            num_fourier_feats=num_fourier_feats,
            coord_resolution=coord_resolution,
        )
        self.latent_proj = nn.Linear(latent_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        prompt_points: torch.Tensor,
        prompt_latent_feats: torch.Tensor,
        valid_query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:


        batch_size, num_queries, _ = prompt_points.shape
        coord_embed = self.coord_encoder(
            prompt_points.reshape(batch_size * num_queries, 3)
        ).reshape(batch_size, num_queries, self.embed_dim)
        latent_embed = self.latent_proj(
            prompt_latent_feats.to(self.latent_proj.weight.dtype)
        )
        queries = coord_embed + latent_embed
        queries = self.output_norm(queries.to(self.output_norm.weight.dtype)).to(
            latent_embed.dtype
        )
        if valid_query_mask is not None:
            queries = queries * valid_query_mask.unsqueeze(-1).to(queries.dtype)
        return queries


class MaskDecoder(nn.Module):


    def __init__(
        self,
        point_dim: int,
        query_dim: int,
        max_parts: int,
        decoder_type: str = "cosine",
    ):
        super().__init__()
        self.point_dim = point_dim
        self.query_dim = query_dim
        self.max_parts = max_parts
        self.decoder_type = decoder_type

        if decoder_type == "cosine":

            self.point_proj = sp.SparseLinear(point_dim, query_dim)
            self.query_proj = nn.Linear(query_dim, query_dim)

            self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        elif decoder_type == "mlp":

            hidden_dim = query_dim * 4
            self.point_proj = sp.SparseLinear(point_dim, query_dim)
            self.mlp_decoder = nn.Sequential(
                nn.Linear(query_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(
                f"Unknown decoder_type: {decoder_type}. Expected 'cosine' or 'mlp'."
            )

    def forward(
        self,
        p: sp.SparseTensor,
        q: torch.Tensor,
        valid_query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:


        batch_size = p.shape[0]

        if self.decoder_type == "cosine":
            return self._forward_cosine(
                p, q, batch_size, valid_query_mask=valid_query_mask
            )
        else:
            return self._forward_mlp(
                p, q, batch_size, valid_query_mask=valid_query_mask
            )

    def _forward_cosine(
        self,
        p: sp.SparseTensor,
        q: torch.Tensor,
        batch_size: int,
        valid_query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:


        weight_dtype = self.query_proj.weight.dtype
        q = q.to(weight_dtype)


        p_proj = self.point_proj(p)
        q_proj = self.query_proj(q)


        p_feats = F.normalize(p_proj.feats, p=2, dim=-1)
        q_feats = F.normalize(q_proj, p=2, dim=-1)
        max_queries = q.shape[1]


        mask_logits_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for b in range(batch_size):
            start_idx = p.layout[b].start
            end_idx = p.layout[b].stop
            p_batch = p_feats[start_idx:end_idx]
            q_batch, query_indices = self._select_query_subset(
                q_feats,
                batch_idx=b,
                valid_query_mask=valid_query_mask,
            )

            if q_batch.shape[0] == 0:
                mask_logits_list.append(
                    (p_batch.new_zeros((p_batch.shape[0], 0)), query_indices)
                )
                continue


            similarity = torch.matmul(p_batch, q_batch.T)
            similarity = similarity / self.temperature.to(
                device=similarity.device,
                dtype=similarity.dtype,
            )
            mask_logits_list.append((similarity, query_indices))

        return self._pad_and_stack(
            mask_logits_list, batch_size, max_queries, p.device, p.dtype
        )

    def _forward_mlp(
        self,
        p: sp.SparseTensor,
        q: torch.Tensor,
        batch_size: int,
        valid_query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:


        weight_dtype = self.mlp_decoder[0].weight.dtype
        q = q.to(weight_dtype)


        p_proj = self.point_proj(p)
        max_queries = q.shape[1]


        mask_logits_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for b in range(batch_size):
            start_idx = p.layout[b].start
            end_idx = p.layout[b].stop
            p_batch = p_proj.feats[start_idx:end_idx]
            q_batch, query_indices = self._select_query_subset(
                q,
                batch_idx=b,
                valid_query_mask=valid_query_mask,
            )

            n_points = p_batch.shape[0]
            n_parts = q_batch.shape[0]
            if n_parts == 0:
                mask_logits_list.append(
                    (p_batch.new_zeros((n_points, 0)), query_indices)
                )
                continue


            p_expanded = p_batch.unsqueeze(1).expand(-1, n_parts, -1)
            q_expanded = q_batch.unsqueeze(0).expand(n_points, -1, -1)


            combined = torch.cat([p_expanded, q_expanded], dim=-1)


            logits = self.mlp_decoder(combined).squeeze(-1)
            mask_logits_list.append((logits, query_indices))

        return self._pad_and_stack(
            mask_logits_list, batch_size, max_queries, p.device, p.dtype
        )

    @staticmethod
    def _select_query_subset(
        q: torch.Tensor,
        batch_idx: int,
        valid_query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if valid_query_mask is None:
            query_indices = torch.arange(q.shape[1], device=q.device)
            return q[batch_idx], query_indices

        query_indices = (
            valid_query_mask[batch_idx].bool().nonzero(as_tuple=False).squeeze(1)
        )
        return q[batch_idx, query_indices], query_indices

    def _pad_and_stack(
        self,
        mask_logits_list: List[Tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
        max_queries: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        max_len = max(logits.shape[0] for logits, _ in mask_logits_list)
        mask_logits = torch.zeros(
            batch_size, max_len, max_queries, device=device, dtype=dtype
        )
        for b, (logits, query_indices) in enumerate(mask_logits_list):
            if logits.numel() == 0 or query_indices.numel() == 0:
                continue
            logits = logits.to(device=device, dtype=mask_logits.dtype)
            mask_logits[b, : logits.shape[0], query_indices] = logits

        return mask_logits


class DecomposableVaeDecoder(FlexiDualGridVaeDecoder):


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

        cola3d_hidden_dim: int = 768,
        cola3d_num_layers: int = 6,
        cola3d_num_heads: int = 12,
        max_parts: int = 8,
        mask_decoder_type: str = "cosine",
        query_init_mode: str = "learnable",
        query_source: str = "fixed",
        point_feature_source: str = "latent",

        routing_mode: str = "st",
        per_part_mode: str = "seg_remap",
        mesh_decode_latent: str = "original",

        use_image_ca: bool = False,
        img_feature_dim: int = 1024,

        use_coord_pe: bool = False,
        coord_pe_resolution: int = 32,

        pe_mode: str = "ape",

        num_fourier_feats: int = 64,

        input_mode: str = "latent",
        mask_prompt_encoder_type: str = "cnn",
        mask_prompt_pe_dim_per_axis: int = 128,

        coord_encoder_fourier_feats: int = 128,

        prompt_point_fourier_feats: int = 64,

        utonia_model: str = "Pointcept/Utonia",
        utonia_feature_dim: int = 576,
        utonia_world_span: float = 12.0,
    ):

        assert routing_mode in (
            "st",
            "gumbel",
            None,
        ), f"routing_mode must be 'st', 'gumbel', or None, got {routing_mode}"
        assert per_part_mode in (
            "gt_supervised",
            "seg_remap",
            "seg_pred",
        ), f"per_part_mode must be 'gt_supervised', 'seg_remap', or 'seg_pred', got {per_part_mode}"
        assert mesh_decode_latent in (
            "original",
            "enriched",
        ), f"mesh_decode_latent must be 'original' or 'enriched', got {mesh_decode_latent}"
        assert query_source in (
            "fixed",
            "prompt_mask",
            "prompt_point",
        ), f"query_source must be 'fixed', 'prompt_mask', or 'prompt_point', got {query_source}"
        assert mask_prompt_encoder_type in (
            "cnn",
            "cnn_centerpe",
        ), f"mask_prompt_encoder_type must be 'cnn' or 'cnn_centerpe', got {mask_prompt_encoder_type}"
        assert point_feature_source in (
            "latent",
            "latent_utonia",
        ), f"point_feature_source must be 'latent' or 'latent_utonia', got {point_feature_source}"
        if point_feature_source == "latent_utonia":
            assert query_source == "prompt_point", (
                "point_feature_source='latent_utonia' requires query_source='prompt_point', "
                f"got query_source='{query_source}'"
            )
            assert input_mode in ("latent_utonia", "latent_utonia_1024"), (
                "point_feature_source='latent_utonia' requires input_mode in "
                f"{{'latent_utonia','latent_utonia_1024'}}, got input_mode='{input_mode}'"
            )


        super().__init__(
            resolution=resolution,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            block_type=block_type,
            up_block_type=up_block_type,
            block_args=block_args,
            voxel_margin=voxel_margin,
            use_fp16=use_fp16,
            use_dtype=use_dtype,
        )

        self.max_parts = max_parts
        self.cola3d_hidden_dim = cola3d_hidden_dim
        self.routing_mode = routing_mode
        self.per_part_mode = per_part_mode
        self.mesh_decode_latent = mesh_decode_latent
        self.input_mode = input_mode
        self.query_source = query_source
        self.point_feature_source = point_feature_source
        self.latent_resolution = resolution // (2 ** len(up_block_type))
        logger.info(
            "routing_mode=%r, per_part_mode=%r, mesh_decode_latent=%r, input_mode=%r, "
            "query_source=%r, point_feature_source=%r",
            routing_mode,
            per_part_mode,
            mesh_decode_latent,
            input_mode,
            query_source,
            point_feature_source,
        )


        if input_mode in ("utonia", "latent_utonia"):
            from ...modules.ext_backbones import UtoniaFeatureExtractor

            self.utonia_extractor = UtoniaFeatureExtractor(
                model_name=utonia_model,
                world_span=utonia_world_span,
                resolution=self.latent_resolution,
            )
            utonia_feature_dim = self.utonia_extractor.output_dim
            logger.info("Utonia output_dim=%d", utonia_feature_dim)
        elif input_mode == "latent_utonia_1024":


            logger.info(
                "latent_utonia_1024: using precomputed Utonia feats (dim=%d)",
                utonia_feature_dim,
            )


        cola3d_query_init_mode = (
            "prompt_only"
            if query_source in ("prompt_mask", "prompt_point")
            else query_init_mode
        )
        self.cola3d = CoLa3DBackbone(
            latent_dim=latent_channels,
            hidden_dim=cola3d_hidden_dim,
            num_layers=cola3d_num_layers,
            num_heads=cola3d_num_heads,
            max_parts=max_parts,
            query_init_mode=cola3d_query_init_mode,
            use_image_ca=use_image_ca,
            img_feature_dim=img_feature_dim,
            use_coord_pe=use_coord_pe,
            coord_pe_resolution=coord_pe_resolution,
            pe_mode=pe_mode,
            num_fourier_feats=num_fourier_feats,
            input_mode=input_mode,
            coord_encoder_fourier_feats=coord_encoder_fourier_feats,
            utonia_feature_dim=utonia_feature_dim,
        )


        cola3d_point_dim = (
            cola3d_hidden_dim
            if input_mode
            in ("coord_only", "utonia", "latent_utonia", "latent_utonia_1024")
            else latent_channels
        )
        self.mask_decoder = MaskDecoder(
            point_dim=cola3d_point_dim,
            query_dim=cola3d_hidden_dim,
            max_parts=max_parts,
            decoder_type=mask_decoder_type,
        )
        if self.query_source == "prompt_mask":
            self.mask_prompt_encoder = MaskPromptEncoder(
                embed_dim=cola3d_hidden_dim,
                pe_dim_per_axis=mask_prompt_pe_dim_per_axis,
                encoder_type=mask_prompt_encoder_type,
            )
        else:
            self.mask_prompt_encoder = None
        if self.query_source == "prompt_point":
            point_query_latent_dim = (
                cola3d_hidden_dim
                if self.point_feature_source == "latent_utonia"
                else latent_channels
            )
            self.point_prompt_encoder = PointPromptEncoder(
                embed_dim=cola3d_hidden_dim,
                latent_dim=point_query_latent_dim,
                coord_resolution=self.latent_resolution,
                num_fourier_feats=prompt_point_fourier_feats,
            )
        else:
            self.point_prompt_encoder = None


        self.cola3d_dtype = self.dtype


        if self.cola3d_dtype != torch.float32:
            self.cola3d.apply(lambda l: convert_module_to(l, self.cola3d_dtype))
            self.mask_decoder.apply(lambda l: convert_module_to(l, self.cola3d_dtype))
            if self.mask_prompt_encoder is not None:
                self.mask_prompt_encoder.apply(
                    lambda l: convert_module_to(l, self.cola3d_dtype)
                )
            if self.point_prompt_encoder is not None:
                self.point_prompt_encoder.apply(
                    lambda l: convert_module_to(l, self.cola3d_dtype)
                )

    def _select_mesh_decode_latent(
        self,
        original_latent: sp.SparseTensor,
        enriched_latent: sp.SparseTensor,
    ) -> sp.SparseTensor:

        if self.mesh_decode_latent == "enriched":
            return enriched_latent
        return original_latent

    def _encode_prompt_queries(
        self,
        latent: sp.SparseTensor,
        prompt_masks: Optional[torch.Tensor],
        prompt_points: Optional[torch.Tensor],
        prompt_valid_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:


        if self.query_source not in ("prompt_mask", "prompt_point"):
            return None, None, None
        if prompt_valid_mask is None:
            raise ValueError(
                f"{self.query_source} query_source requires prompt_valid_mask"
            )

        if self.query_source == "prompt_mask":
            if prompt_masks is None:
                raise ValueError("prompt_mask query_source requires prompt_masks")
            if prompt_masks.ndim != 4:
                raise ValueError(
                    f"prompt_masks must have shape (B, Q, H, W), got {tuple(prompt_masks.shape)}"
                )
            if prompt_valid_mask.shape != prompt_masks.shape[:2]:
                raise ValueError(
                    "prompt_valid_mask must match the (B, Q) prefix of prompt_masks, "
                    f"got {tuple(prompt_valid_mask.shape)} vs {tuple(prompt_masks.shape[:2])}"
                )
            if self.mask_prompt_encoder is None:
                raise RuntimeError("mask_prompt_encoder is not initialized")
            device = self.mask_prompt_encoder.encoder[0].weight.device
            prompt_valid_mask = prompt_valid_mask.to(device=device, dtype=torch.bool)
            prompt_queries = self.mask_prompt_encoder(
                prompt_masks.to(device=device, dtype=self.cola3d_dtype),
                valid_query_mask=prompt_valid_mask,
            )
            return prompt_queries, prompt_valid_mask, None

        if prompt_points is None:
            raise ValueError("prompt_point query_source requires prompt_points")
        if prompt_points.ndim != 3 or prompt_points.shape[-1] != 3:
            raise ValueError(
                f"prompt_points must have shape (B, Q, 3), got {tuple(prompt_points.shape)}"
            )
        if prompt_valid_mask.shape != prompt_points.shape[:2]:
            raise ValueError(
                "prompt_valid_mask must match the (B, Q) prefix of prompt_points, "
                f"got {tuple(prompt_valid_mask.shape)} vs {tuple(prompt_points.shape[:2])}"
            )
        if self.point_prompt_encoder is None:
            raise RuntimeError("point_prompt_encoder is not initialized")

        device = self.point_prompt_encoder.latent_proj.weight.device
        prompt_valid_mask = prompt_valid_mask.to(device=device, dtype=torch.bool)
        prompt_points = prompt_points.to(device=device, dtype=torch.long)
        prompt_latent_feats = self._gather_prompt_latent_feats(
            latent=latent,
            prompt_points=prompt_points,
            prompt_valid_mask=prompt_valid_mask,
        )
        prompt_queries = self.point_prompt_encoder(
            prompt_points.to(dtype=self.cola3d_dtype),
            prompt_latent_feats=prompt_latent_feats,
            valid_query_mask=prompt_valid_mask,
        )
        return prompt_queries, prompt_valid_mask, prompt_points

    def _coords_to_linear_index(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:

        coords = coords.long()
        return (
            coords[..., 0] * self.latent_resolution * self.latent_resolution
            + coords[..., 1] * self.latent_resolution
            + coords[..., 2]
        )

    def _gather_prompt_latent_feats(
        self,
        latent: sp.SparseTensor,
        prompt_points: torch.Tensor,
        prompt_valid_mask: torch.Tensor,
    ) -> torch.Tensor:


        gathered = latent.feats.new_zeros(
            prompt_points.shape[0],
            prompt_points.shape[1],
            latent.feats.shape[1],
        )
        for batch_idx in range(prompt_points.shape[0]):
            sample_valid = prompt_valid_mask[batch_idx].bool()
            if not sample_valid.any():
                continue
            start_idx = latent.layout[batch_idx].start
            end_idx = latent.layout[batch_idx].stop
            sample_coords = latent.coords[start_idx:end_idx, 1:4].long()
            sample_linear = self._coords_to_linear_index(sample_coords)
            sorted_keys, sorted_order = torch.sort(sample_linear)

            sample_points = prompt_points[batch_idx, sample_valid]
            if (sample_points < 0).any() or (
                sample_points >= self.latent_resolution
            ).any():
                raise ValueError(
                    f"prompt_points for batch {batch_idx} exceed latent resolution "
                    f"{self.latent_resolution}: {sample_points.tolist()}"
                )

            point_linear = self._coords_to_linear_index(sample_points)
            hit_pos = torch.searchsorted(sorted_keys, point_linear)
            in_range = hit_pos < sorted_keys.numel()
            hit_pos_clamped = hit_pos.clamp_max(sorted_keys.numel() - 1)
            matched = in_range & (sorted_keys[hit_pos_clamped] == point_linear)
            if not bool(matched.all().item()):
                missing_points = sample_points[~matched].detach().cpu().tolist()
                raise ValueError(
                    f"Failed to gather latent feats for prompt_points in batch {batch_idx}: "
                    f"{missing_points}"
                )
            latent_indices = start_idx + sorted_order[hit_pos_clamped]
            gathered[batch_idx, sample_valid] = latent.feats[latent_indices]
        return gathered

    def forward(
        self,
        x: sp.SparseTensor,
        gt_intersected: sp.SparseTensor = None,
        parallel_decode: bool = True,
        return_intermediate: bool = False,
        also_parallel_decode: bool = False,
        per_obj_coords: Optional[List[List[torch.Tensor]]] = None,
        gt_multihot_latent: Optional[
            torch.Tensor
        ] = None,
        gt_num_objects: Optional[torch.Tensor] = None,
        per_obj_intersected: Optional[
            List[List[torch.Tensor]]
        ] = None,
        img_features: Optional[
            torch.Tensor
        ] = None,
        prompt_masks: Optional[torch.Tensor] = None,
        prompt_points: Optional[
            torch.Tensor
        ] = None,
        prompt_valid_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[
        Tuple[List[Mesh], torch.Tensor, torch.Tensor],
        Tuple[
            List[Mesh],
            torch.Tensor,
            torch.Tensor,
            sp.SparseTensor,
            sp.SparseTensor,
            List,
            List,
        ],
        Tuple[
            List[Mesh],
            torch.Tensor,
            torch.Tensor,
            sp.SparseTensor,
            sp.SparseTensor,
            List,
            List,
            List[List[Mesh]],
            Optional[torch.Tensor],
        ],
    ]:


        local_kwargs = dict(kwargs)
        routing_threshold = local_kwargs.pop("routing_threshold", None)
        multihot_threshold = local_kwargs.pop("multihot_threshold", None)
        if routing_threshold is None:
            if multihot_threshold is not None:
                p = float(multihot_threshold)
                p = max(1e-6, min(1 - 1e-6, p))
                routing_threshold = float(math.log(p / (1 - p)))
            else:
                routing_threshold = 0.0


        original_dtype = x.feats.dtype
        x_cola3d = x.replace(x.feats.type(self.cola3d_dtype))


        import os as _os
        import time as _time

        _timing_on = (
            _os.environ.get("TRELLIS2_TIMING", "0") == "1"
            and self.training
            and x.feats.is_cuda
        )
        if _timing_on:
            if not hasattr(self, "_fwd_timing_stats"):
                self._fwd_timing_stats = {
                    "utonia_ms": 0.0,
                    "cola3d_ms": 0.0,
                    "mask_ms": 0.0,
                    "count": 0,
                }
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()


        ext_features = None
        if self.input_mode == "latent_utonia_1024":
            precomputed_utonia = kwargs.get("precomputed_utonia")
            assert (
                precomputed_utonia is not None
            ), "latent_utonia_1024 requires 'precomputed_utonia' in kwargs"
            feats = (
                precomputed_utonia.feats
                if isinstance(precomputed_utonia, sp.SparseTensor)
                else precomputed_utonia
            )
            ext_features = x.replace(feats.to(self.cola3d_dtype))
        elif self.input_mode in ("utonia", "latent_utonia"):
            utonia_feats = self.utonia_extractor(
                coords=x.coords,
                batch_size=x.shape[0],
            )
            ext_features = x.replace(utonia_feats.to(self.cola3d_dtype))

        if _timing_on:
            torch.cuda.synchronize()
            _t1 = _time.perf_counter()


        prefused_input = None
        prompt_latent_source = x_cola3d
        if (
            self.query_source == "prompt_point"
            and self.point_feature_source == "latent_utonia"
        ):
            prefused_input = self.cola3d.compute_input_features(x_cola3d, ext_features)
            prompt_latent_source = prefused_input

        prompt_queries, prompt_valid, prompt_q_coords = self._encode_prompt_queries(
            latent=prompt_latent_source,
            prompt_masks=prompt_masks,
            prompt_points=prompt_points,
            prompt_valid_mask=prompt_valid_mask,
        )

        p_enriched, query_feats = self.cola3d(
            x_cola3d,
            img_features=img_features,
            ext_features=ext_features,
            prompt_queries=prompt_queries,
            valid_query_mask=prompt_valid,
            prompt_q_coords=prompt_q_coords,
            prefused_input=prefused_input,
        )

        if _timing_on:
            torch.cuda.synchronize()
            _t2 = _time.perf_counter()


        seg_only_flag = kwargs.get("seg_only", False)
        if (
            self.input_mode
            in ("coord_only", "utonia", "latent_utonia", "latent_utonia_1024")
            and not seg_only_flag
            and self.training
        ):
            raise NotImplementedError(
                f"{self.input_mode} input_mode currently only supports seg_only=True (CoLa3D training). "
                "Mesh decode requires latent features which are not available in coord_only/utonia/latent_utonia mode."
            )


        mask_logits = self.mask_decoder(
            p_enriched, query_feats, valid_query_mask=prompt_valid
        )

        if _timing_on:
            torch.cuda.synchronize()
            _t3 = _time.perf_counter()
            _s = self._fwd_timing_stats
            _s["utonia_ms"] += (_t1 - _t0) * 1000.0
            _s["cola3d_ms"] += (_t2 - _t1) * 1000.0
            _s["mask_ms"] += (_t3 - _t2) * 1000.0
            _s["count"] += 1
            if _s["count"] % 50 == 0:
                import torch.distributed as _dist

                _is_main = (
                    (not _dist.is_available())
                    or (not _dist.is_initialized())
                    or _dist.get_rank() == 0
                )
                if _is_main:
                    _n = _s["count"]
                    _avg_u = _s["utonia_ms"] / _n
                    _avg_p = _s["cola3d_ms"] / _n
                    _avg_m = _s["mask_ms"] / _n
                    _tot = _avg_u + _avg_p + _avg_m
                    _share = (_avg_u / _tot * 100.0) if _tot > 0 else 0.0
                    print(
                        f"[fwd-timing] n={_n} "
                        f"utonia={_avg_u:.2f}ms cola3d={_avg_p:.2f}ms mask={_avg_m:.2f}ms | "
                        f"sum={_tot:.2f}ms utonia_share={_share:.1f}% "
                        f"N_pts={x.coords.shape[0]} B={x.shape[0]}",
                        flush=True,
                    )


        p_enriched = p_enriched.replace(p_enriched.feats.type(original_dtype))
        mesh_decode_latent = self._select_mesh_decode_latent(x, p_enriched)


        if parallel_decode and not self.training:

            meshes, routing_gates, _, _, _, _ = self._decode_parts_parallel(
                x,
                p_enriched,
                mask_logits,
                routing_threshold=routing_threshold,
                per_obj_coords=per_obj_coords,
                gt_multihot_latent=gt_multihot_latent,
                gt_num_objects=gt_num_objects,
                **local_kwargs,
            )

            return meshes, mask_logits, query_feats
        else:

            seg_only = local_kwargs.pop("seg_only", False)

            if seg_only:

                if return_intermediate:
                    return (
                        None,
                        mask_logits,
                        query_feats,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                else:
                    return None, mask_logits, query_feats


            skip_unified = local_kwargs.pop("skip_unified_decode", False)

            if skip_unified and self.training:

                (
                    part_meshes,
                    routing_gates,
                    part_vertices,
                    part_intersected,
                    part_subs_gt_list,
                    part_subs_list,
                ) = self._decode_parts_parallel(
                    x,
                    p_enriched,
                    mask_logits,
                    routing_threshold=routing_threshold,
                    per_obj_coords=per_obj_coords,
                    gt_multihot_latent=gt_multihot_latent,
                    gt_num_objects=gt_num_objects,
                    per_obj_intersected=per_obj_intersected,
                    **local_kwargs,
                )


                if return_intermediate:
                    return (
                        None,
                        mask_logits,
                        query_feats,
                        None,
                        None,
                        None,
                        None,
                        part_meshes,
                        routing_gates,
                        part_vertices,
                        part_intersected,
                        part_subs_gt_list,
                        part_subs_list,
                    )
                else:
                    return (
                        None,
                        mask_logits,
                        query_feats,
                        part_meshes,
                        routing_gates,
                        part_vertices,
                        part_intersected,
                        part_subs_gt_list,
                        part_subs_list,
                    )


            decoded = super().forward(
                mesh_decode_latent, gt_intersected, **local_kwargs
            )

            if self.training:
                meshes, vertices, intersected_logits, subs_gt, subs = decoded


                if also_parallel_decode:
                    (
                        part_meshes,
                        routing_gates,
                        part_vertices,
                        part_intersected,
                        part_subs_gt_list,
                        part_subs_list,
                    ) = self._decode_parts_parallel(
                        x,
                        p_enriched,
                        mask_logits,
                        routing_threshold=routing_threshold,
                        per_obj_coords=per_obj_coords,
                        gt_multihot_latent=gt_multihot_latent,
                        gt_num_objects=gt_num_objects,
                        per_obj_intersected=per_obj_intersected,
                        **local_kwargs,
                    )
                    if return_intermediate:
                        return (
                            meshes,
                            mask_logits,
                            query_feats,
                            vertices,
                            intersected_logits,
                            subs_gt,
                            subs,
                            part_meshes,
                            routing_gates,
                            part_vertices,
                            part_intersected,
                            part_subs_gt_list,
                            part_subs_list,
                        )
                    else:
                        return (
                            meshes,
                            mask_logits,
                            query_feats,
                            part_meshes,
                            routing_gates,
                            part_vertices,
                            part_intersected,
                            part_subs_gt_list,
                            part_subs_list,
                        )

                if return_intermediate:
                    return (
                        meshes,
                        mask_logits,
                        query_feats,
                        vertices,
                        intersected_logits,
                        subs_gt,
                        subs,
                    )
                else:
                    return meshes, mask_logits, query_feats
            else:
                meshes = decoded if isinstance(decoded, list) else [decoded]
                return meshes, mask_logits, query_feats

    def _decode_parts_parallel(
        self,
        original_latent: sp.SparseTensor,
        enriched_latent: sp.SparseTensor,
        mask_logits: torch.Tensor,
        routing_tau: float = 1.0,
        routing_threshold: float = 0.0,
        per_obj_coords: Optional[List[List[torch.Tensor]]] = None,
        gt_multihot_latent: Optional[
            torch.Tensor
        ] = None,
        gt_num_objects: Optional[torch.Tensor] = None,
        per_obj_intersected: Optional[
            List[List[torch.Tensor]]
        ] = None,
        **kwargs,
    ) -> Tuple[List[List[Mesh]], Optional[torch.Tensor], List, List]:


        if self.routing_mode == "st":
            m_hard, g_st = sp.straight_through_routing(
                mask_logits, threshold=routing_threshold, tau=routing_tau
            )
            use_gates = g_st if self.training else None
        elif self.routing_mode == "gumbel":
            g_sample = sp.gumbel_sigmoid(
                mask_logits, temperature=routing_tau, hard=True
            )
            m_hard = g_sample > 0.5
            use_gates = g_sample if self.training else None
        else:
            m_hard = (
                (torch.sigmoid(mask_logits) > 0.5)
                if routing_threshold == 0.0
                else (mask_logits > routing_threshold)
            )
            use_gates = None


        all_gates = [] if use_gates is not None else None
        all_meshes = []
        all_vertices = []
        all_intersected = []
        all_subs_gt = []
        all_subs = []


        use_gt_supervision = (
            self.per_part_mode == "gt_supervised"
            and gt_multihot_latent is not None
            and gt_num_objects is not None
        )
        if (
            self.per_part_mode == "gt_supervised"
            and self.training
            and not use_gt_supervision
        ):
            raise ValueError(
                "gt_supervised mode requires gt_multihot_latent and gt_num_objects during training. "
                "Pass these from trainer via decoder.forward()."
            )

        decode_latent = self._select_mesh_decode_latent(
            original_latent,
            enriched_latent,
        )
        mesh_device = decode_latent.feats.device

        def _make_empty_mesh() -> Mesh:
            return Mesh(
                vertices=torch.zeros(0, 3, dtype=torch.float32, device=mesh_device),
                faces=torch.zeros(0, 3, dtype=torch.int32, device=mesh_device),
            )

        for b in range(decode_latent.shape[0]):
            batch_meshes = []
            batch_vertices = []
            batch_intersected = []
            batch_subs_gt = []
            batch_subs = []


            start_idx = decode_latent.layout[b].start
            end_idx = decode_latent.layout[b].stop
            n_points = end_idx - start_idx


            batch_hard = m_hard[b, :n_points]
            n_obj = int(gt_num_objects[b].item()) if use_gt_supervision else None
            num_queries = batch_hard.shape[1]

            for part_id in range(num_queries):

                part_mask = batch_hard[:, part_id]


                if use_gt_supervision:
                    if part_id >= n_obj:
                        continue
                    part_mask = gt_multihot_latent[b, :n_points, part_id].bool()

                if part_mask.sum() == 0:
                    if use_gt_supervision:

                        batch_meshes.append(_make_empty_mesh())
                    continue


                if all_gates is not None:
                    part_gate = use_gates[b, :n_points, part_id][part_mask]
                    all_gates.append(part_gate)


                part_feats = decode_latent.feats[start_idx:end_idx][part_mask]
                part_coords = decode_latent.coords[start_idx:end_idx][part_mask].clone()
                part_coords[:, 0] = 0


                part_latent = sp.SparseTensor(part_feats, part_coords)
                part_latent._scale = decode_latent._scale


                if self.per_part_mode == "seg_remap" and hasattr(
                    original_latent, "_spatial_cache"
                ):

                    full_mask = torch.zeros(
                        original_latent.coords.shape[0],
                        dtype=torch.bool,
                        device=part_mask.device,
                    )
                    full_mask[start_idx:end_idx] = part_mask
                    adapted_cache = sp.adapt_spatial_cache_for_part(
                        original_latent._spatial_cache, full_mask
                    )
                    part_latent._spatial_cache = adapted_cache

                    part_latent._scale = original_latent._scale

                elif use_gt_supervision:


                    if (
                        per_obj_coords is not None
                        and b < len(per_obj_coords)
                        and part_id < len(per_obj_coords[b])
                    ):
                        obj_coords = per_obj_coords[b][
                            part_id
                        ]


                        scene_shape = tuple(
                            int(s * int(original_latent._scale[i]))
                            for i, s in enumerate(original_latent.spatial_shape)
                        )
                        gt_cache, expected_latent_coords = (
                            sp.build_subdivision_cache_from_gt(
                                obj_coords,
                                global_shape=scene_shape,
                                return_latent_coords=True,
                            )
                        )

                        try:
                            part_latent = sp.reorder_latent_to_match_cache(
                                part_latent, expected_latent_coords
                            )
                        except ValueError as e:

                            print(
                                f"[WARNING] Per-part decode: {e}. Skipping part {part_id}."
                            )
                            continue
                        part_latent._spatial_cache = gt_cache


                was_training = self.training


                has_part_gt_intersected = (
                    per_obj_intersected is not None
                    and b < len(per_obj_intersected)
                    and part_id < len(per_obj_intersected[b])
                    and per_obj_intersected[b][part_id] is not None
                )


                if has_part_gt_intersected:

                    part_gt_inter = per_obj_intersected[b][
                        part_id
                    ]


                    if not isinstance(part_gt_inter, sp.SparseTensor):


                        if (
                            per_obj_coords is not None
                            and b < len(per_obj_coords)
                            and part_id < len(per_obj_coords[b])
                        ):
                            obj_coords = per_obj_coords[b][part_id]

                            if obj_coords.shape[1] == 3:

                                batch_idx = torch.zeros(
                                    (obj_coords.shape[0], 1),
                                    device=obj_coords.device,
                                    dtype=obj_coords.dtype,
                                )
                                obj_coords_4d = torch.cat(
                                    [batch_idx, obj_coords], dim=1
                                )
                            else:
                                obj_coords_4d = obj_coords.clone()
                                obj_coords_4d[:, 0] = 0

                            part_gt_inter_sp = sp.SparseTensor(
                                part_gt_inter, obj_coords_4d.int()
                            )
                        else:

                            has_part_gt_intersected = False
                    else:
                        part_gt_inter_sp = part_gt_inter

                if has_part_gt_intersected:

                    decode_output = super().forward(
                        part_latent, part_gt_inter_sp, **kwargs
                    )


                    if isinstance(decode_output, tuple) and len(decode_output) == 5:
                        part_meshes, part_verts, part_inter, part_subs_gt, part_subs = (
                            decode_output
                        )
                        if use_gt_supervision:
                            batch_meshes.append(
                                part_meshes[0]
                                if len(part_meshes) > 0
                                else _make_empty_mesh()
                            )
                        else:
                            batch_meshes.extend(part_meshes)
                        batch_vertices.append(part_verts)
                        batch_intersected.append(part_inter)
                        batch_subs_gt.append(part_subs_gt)
                        batch_subs.append(part_subs)
                    else:
                        part_meshes = (
                            decode_output
                            if isinstance(decode_output, list)
                            else [decode_output]
                        )
                        if use_gt_supervision:
                            batch_meshes.append(
                                part_meshes[0]
                                if len(part_meshes) > 0
                                else _make_empty_mesh()
                            )
                        else:
                            batch_meshes.extend(part_meshes)
                else:


                    self.eval()
                    try:
                        decode_output = super().forward(part_latent, **kwargs)
                    except (IndexError, RuntimeError) as e:


                        import warnings

                        warnings.warn(
                            f"Per-part eval decode failed for part {part_id} "
                            f"(batch {b}): {e}. Using empty mesh.",
                            stacklevel=2,
                        )


                        try:
                            torch.cuda.synchronize()
                        except RuntimeError:
                            pass
                        if was_training:
                            self.train()
                        batch_meshes.append(_make_empty_mesh())
                        continue
                    if was_training:
                        self.train()

                    part_meshes = (
                        decode_output
                        if isinstance(decode_output, list)
                        else [decode_output]
                    )
                    if use_gt_supervision:
                        batch_meshes.append(
                            part_meshes[0]
                            if len(part_meshes) > 0
                            else _make_empty_mesh()
                        )
                    else:
                        batch_meshes.extend(part_meshes)

            all_meshes.append(batch_meshes)
            all_vertices.append(batch_vertices)
            all_intersected.append(batch_intersected)
            all_subs_gt.append(batch_subs_gt)
            all_subs.append(batch_subs)


        routing_gates = torch.cat(all_gates, dim=0) if all_gates else None

        return (
            all_meshes,
            routing_gates,
            all_vertices,
            all_intersected,
            all_subs_gt,
            all_subs,
        )

    def forward_unified(
        self,
        x: sp.SparseTensor,
        gt_intersected: sp.SparseTensor = None,
        img_features: Optional[torch.Tensor] = None,
        prompt_masks: Optional[torch.Tensor] = None,
        prompt_points: Optional[torch.Tensor] = None,
        prompt_valid_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):


        prompt_queries, prompt_valid, prompt_q_coords = self._encode_prompt_queries(
            latent=x,
            prompt_masks=prompt_masks,
            prompt_points=prompt_points,
            prompt_valid_mask=prompt_valid_mask,
        )
        p_enriched, query_feats = self.cola3d(
            x,
            img_features=img_features,
            prompt_queries=prompt_queries,
            valid_query_mask=prompt_valid,
            prompt_q_coords=prompt_q_coords,
        )
        mesh_decode_latent = self._select_mesh_decode_latent(x, p_enriched)


        mask_logits = self.mask_decoder(
            p_enriched, query_feats, valid_query_mask=prompt_valid
        )


        decoded = super().forward(mesh_decode_latent, gt_intersected, **kwargs)

        if self.training:
            meshes, vertices, intersected_logits, subs_gt, subs = decoded
            return (
                meshes,
                mask_logits,
                query_feats,
                vertices,
                intersected_logits,
                subs_gt,
                subs,
            )
        else:
            return decoded, mask_logits, query_feats
