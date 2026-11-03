


from typing import Tuple, Optional, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic import SparseTensor, VarLenTensor
from ..linear import SparseLinear
from ..nonlinearity import SparseGELU
from ..attention import SparseMultiHeadAttention, SparseMultiHeadRMSNorm
from ..attention.full_attn import sparse_scaled_dot_product_attention
from ...attention.rope import RotaryPositionEmbedder
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet


__all__ = [
    "PartIndexEmbedder",
    "SparseCoordEmbedder",
    "LearnableFourierPE",
    "CoLa3DBlock",
    "CoLa3DBackbone",
]


class PartIndexEmbedder(nn.Module):


    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def sinusoidal_embedding(
        x: torch.Tensor, dim: int, max_period: float = 10000.0
    ) -> torch.Tensor:


        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period))
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=x.device)


        args = x.unsqueeze(-1).float() * freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[..., :1])], dim=-1
            )
        return embedding

    def forward(
        self, num_parts: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:


        part_indices = (
            torch.arange(num_parts, device=device, dtype=torch.float32) / num_parts
        )


        part_embed = self.sinusoidal_embedding(
            part_indices, self.frequency_embedding_size
        )


        weight_dtype = self.mlp[0].weight.dtype
        part_embed = part_embed.to(weight_dtype)
        query = self.mlp(part_embed)


        query = query.unsqueeze(0).expand(batch_size, -1, -1)

        return query


class SparseCoordEmbedder(nn.Module):


    def __init__(self, hidden_dim: int, resolution: int = 32):
        super().__init__()
        assert (
            hidden_dim % 6 == 0
        ), f"hidden_dim must be divisible by 6 (3 axes x sin/cos), got {hidden_dim}"
        self.resolution = resolution
        self.freq_dim = hidden_dim // 3

    def forward(self, coords: torch.Tensor) -> torch.Tensor:


        xyz = coords[:, 1:].float() / self.resolution
        embed_x = PartIndexEmbedder.sinusoidal_embedding(xyz[:, 0], self.freq_dim)
        embed_y = PartIndexEmbedder.sinusoidal_embedding(xyz[:, 1], self.freq_dim)
        embed_z = PartIndexEmbedder.sinusoidal_embedding(xyz[:, 2], self.freq_dim)
        return torch.cat([embed_x, embed_y, embed_z], dim=-1)


class LearnableFourierPE(nn.Module):


    def __init__(
        self,
        hidden_dim: int,
        coord_dim: int = 3,
        num_fourier_feats: int = 64,
        coord_resolution: int = 32,
    ):
        super().__init__()
        self.coord_resolution = coord_resolution

        self.Wr = nn.Linear(coord_dim, num_fourier_feats, bias=False)
        nn.init.normal_(self.Wr.weight, std=1.0)

        self.mlp = nn.Sequential(
            nn.Linear(num_fourier_feats * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:


        x = coords.float() / self.coord_resolution
        x = x.to(self.Wr.weight.dtype)
        h = self.Wr(x)
        fourier = torch.cat(
            [
                torch.cos(2 * math.pi * h),
                torch.sin(2 * math.pi * h),
            ],
            dim=-1,
        )
        return self.mlp(fourier)


class CoLa3DBlock(nn.Module):


    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = True,
        use_image_ca: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.use_rope = use_rope


        self.q_norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        if use_rope:

            self.q_to_qkv = nn.Linear(channels, channels * 3)
            self.q_self_attn_out = nn.Linear(channels, channels)
        else:
            self.q_self_attn = nn.MultiheadAttention(
                channels, num_heads, batch_first=True, dropout=0.0
            )


        if use_image_ca:

            self.img_norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
            self.img_to_kv = nn.Linear(channels, channels * 2)


            self.q_norm_img = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
            self.q_to_q_img = nn.Linear(channels, channels)
            self.q_img_out = nn.Linear(channels, channels)


            self.p_norm_img = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
            self.p_to_q_img = SparseLinear(channels, channels)
            self.p_img_out = SparseLinear(channels, channels)


        self.q_norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.p_norm_p2q = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.p_to_kv = SparseLinear(channels, channels * 2)
        self.q_to_q_p2q = nn.Linear(channels, channels)
        self.p2q_out = nn.Linear(channels, channels)


        self.p_norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        if use_rope:

            self.p_to_q_q2p = SparseLinear(channels, channels)
            self.q_to_kv_q2p = nn.Linear(channels, channels * 2)
            self.q2p_out = SparseLinear(channels, channels)
            self.q2p_qk_rms_norm = qk_rms_norm
            if qk_rms_norm:
                self.q2p_q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
                self.q2p_k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
        else:
            self.q2p_attn = SparseMultiHeadAttention(
                channels,
                num_heads=num_heads,
                ctx_channels=channels,
                type="cross",
                qk_rms_norm=qk_rms_norm,
            )


        self.p_norm_ff = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.q_norm_ff = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.p_mlp = SparseFeedForwardNet(channels, mlp_ratio)
        self.q_mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(
        self,
        p: SparseTensor,
        q: torch.Tensor,
        img_tokens: Optional[
            torch.Tensor
        ] = None,
        p_phases: Optional[
            torch.Tensor
        ] = None,
        q_phases: Optional[
            torch.Tensor
        ] = None,
        valid_query_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> Tuple[SparseTensor, torch.Tensor]:


        if valid_query_mask is not None:


            p_parts: List[SparseTensor] = []
            q_parts: List[torch.Tensor] = []
            max_queries = q.shape[1]
            for batch_idx in range(q.shape[0]):
                sample_valid = valid_query_mask[batch_idx].bool()
                query_indices = sample_valid.nonzero(as_tuple=False).squeeze(1)
                num_valid = int(query_indices.numel())
                p_sample = p[batch_idx]
                sample_p_phases = (
                    p_phases[p.layout[batch_idx]] if p_phases is not None else None
                )
                if num_valid == 0:
                    p_parts.append(p_sample)
                    q_parts.append(q.new_zeros((1, max_queries, q.shape[-1])))
                    continue

                q_sample = q[batch_idx : batch_idx + 1, query_indices]
                img_sample = (
                    img_tokens[batch_idx : batch_idx + 1]
                    if img_tokens is not None
                    else None
                )
                sample_q_phases = (
                    q_phases[batch_idx][query_indices] if q_phases is not None else None
                )
                p_updated, q_updated = self._forward_impl(
                    p_sample,
                    q_sample,
                    img_tokens=img_sample,
                    p_phases=sample_p_phases,
                    q_phases=sample_q_phases,
                )
                q_padded = q.new_zeros((1, max_queries, q.shape[-1]))
                q_padded[:, query_indices] = q_updated
                p_parts.append(p_updated)
                q_parts.append(q_padded)

            p_out = p.replace(
                torch.cat([part.feats for part in p_parts], dim=0).contiguous()
            )
            return p_out, torch.cat(q_parts, dim=0)

        return self._forward_impl(
            p,
            q,
            img_tokens=img_tokens,
            p_phases=p_phases,
            q_phases=q_phases,
        )

    def _forward_impl(
        self,
        p: SparseTensor,
        q: torch.Tensor,
        img_tokens: Optional[torch.Tensor] = None,
        p_phases: Optional[torch.Tensor] = None,
        q_phases: Optional[torch.Tensor] = None,
    ) -> Tuple[SparseTensor, torch.Tensor]:


        weight_dtype = self.q_to_q_p2q.weight.dtype
        original_q_dtype = q.dtype


        q = q.to(weight_dtype)


        q_normed = self.q_norm1(q)
        if self.use_rope:
            qkv = self.q_to_qkv(q_normed)
            B_q, P_max, _ = qkv.shape
            qkv = qkv.reshape(B_q, P_max, 3, self.num_heads, self.head_dim)
            q_q, q_k, q_v = qkv.unbind(dim=2)
            q_q = RotaryPositionEmbedder.apply_rotary_embedding(q_q, q_phases)
            q_k = RotaryPositionEmbedder.apply_rotary_embedding(q_k, q_phases)
            q_attn = (
                F.scaled_dot_product_attention(
                    q_q.transpose(1, 2), q_k.transpose(1, 2), q_v.transpose(1, 2)
                )
                .transpose(1, 2)
                .reshape(B_q, P_max, -1)
            )
            q = q + self.q_self_attn_out(q_attn)
        else:
            q_attn, _ = self.q_self_attn(q_normed, q_normed, q_normed)
            q = q + q_attn


        if img_tokens is not None and hasattr(self, "img_to_kv"):

            img_normed = self.img_norm(img_tokens)
            img_kv = self.img_to_kv(img_normed)
            B_img, N_patch, _ = img_kv.shape
            img_kv = img_kv.reshape(B_img, N_patch, 2, self.num_heads, self.head_dim)
            img_k, img_v = img_kv.unbind(dim=2)


            q_normed_img = self.q_norm_img(q)
            q_q = self.q_to_q_img(q_normed_img)
            q_q = q_q.reshape(B_img, q.shape[1], self.num_heads, self.head_dim)
            q_q = q_q.transpose(1, 2)
            ik = img_k.transpose(1, 2)
            iv = img_v.transpose(1, 2)
            q_attn_img = F.scaled_dot_product_attention(q_q, ik, iv)
            q_attn_img = q_attn_img.transpose(1, 2).reshape(B_img, q.shape[1], -1)
            q = q + self.q_img_out(q_attn_img)


            p_normed_img = p.replace(self.p_norm_img(p.feats))
            p_q = self.p_to_q_img(p_normed_img)
            p_q = p_q.reshape(self.num_heads, self.head_dim)
            p_attn_img = sparse_scaled_dot_product_attention(p_q, img_k, img_v)
            p_attn_img = p_attn_img.reshape(-1)
            p_attn_sp = p.replace(p_attn_img.feats)
            p = p + self.p_img_out(p_attn_sp)


        q_normed = self.q_norm2(q)
        q_query = self.q_to_q_p2q(q_normed)

        p_normed = p.replace(self.p_norm_p2q(p.feats))
        p_kv = self.p_to_kv(p_normed)


        batch_size = q.shape[0]
        q_query_reshaped = q_query.reshape(
            batch_size, q.shape[1], self.num_heads, self.head_dim
        )

        p_kv_reshaped = p_kv.reshape(2, self.num_heads, self.head_dim)

        if self.use_rope:

            p_k, p_v = p_kv_reshaped.unbind(dim=1)
            q_query_reshaped = RotaryPositionEmbedder.apply_rotary_embedding(
                q_query_reshaped, q_phases
            )
            p_k = p_k.replace(
                RotaryPositionEmbedder.apply_rotary_embedding(p_k.feats, p_phases)
            )
            q_attn = sparse_scaled_dot_product_attention(q_query_reshaped, p_k, p_v)
        else:
            q_attn = sparse_scaled_dot_product_attention(
                q_query_reshaped, p_kv_reshaped
            )

        q_attn = q_attn.reshape(batch_size, q.shape[1], -1)
        q = q + self.p2q_out(q_attn)


        p_normed = p.replace(self.p_norm(p.feats))
        if self.use_rope:

            p_q = self.p_to_q_q2p(p_normed)
            p_q = p_q.reshape(self.num_heads, self.head_dim)
            q_kv = self.q_to_kv_q2p(q)
            B_q2p = q.shape[0]
            q_kv = q_kv.reshape(B_q2p, q.shape[1], 2, self.num_heads, self.head_dim)
            q_k, q_v = q_kv.unbind(dim=2)
            if self.q2p_qk_rms_norm:
                p_q = self.q2p_q_rms_norm(p_q)
                q_k = self.q2p_k_rms_norm(q_k)
            p_q = p_q.replace(
                RotaryPositionEmbedder.apply_rotary_embedding(p_q.feats, p_phases)
            )
            q_k = RotaryPositionEmbedder.apply_rotary_embedding(q_k, q_phases)
            p_attn = sparse_scaled_dot_product_attention(p_q, q_k, q_v)
            p_attn = p_attn.reshape(-1)
            p = p + self.q2p_out(p_attn)
        else:
            p_attn = self.q2p_attn(p_normed, q)
            p = p + p_attn


        p_normed = p.replace(self.p_norm_ff(p.feats))
        p = p + self.p_mlp(p_normed)

        q = q + self.q_mlp(self.q_norm_ff(q))


        q = q.to(original_q_dtype)

        return p, q


class CoLa3DBackbone(nn.Module):


    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        max_parts: int = 8,
        mlp_ratio: float = 4.0,
        query_init_mode: str = "learnable",
        use_image_ca: bool = False,
        img_feature_dim: int = 1024,
        use_coord_pe: bool = False,
        coord_pe_resolution: int = 32,
        pe_mode: str = "ape",
        num_fourier_feats: int = 64,
        input_mode: str = "latent",
        coord_encoder_fourier_feats: int = 128,
        utonia_feature_dim: int = 512,
    ):
        super().__init__()
        assert pe_mode in (
            "ape",
            "rope",
            "lff",
            "none",
        ), f"pe_mode must be 'ape', 'rope', 'lff', or 'none', got {pe_mode}"
        _VALID_INPUT_MODES = (
            "latent",
            "coord_only",
            "utonia",
            "latent_utonia",
            "latent_utonia_1024",
        )
        assert (
            input_mode in _VALID_INPUT_MODES
        ), f"input_mode must be one of {_VALID_INPUT_MODES}, got {input_mode!r}"
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_parts = max_parts
        self.query_init_mode = query_init_mode
        self.use_image_ca = use_image_ca
        self.pe_mode = pe_mode
        self.input_mode = input_mode

        self.use_coord_pe = use_coord_pe and pe_mode == "ape" and input_mode == "latent"


        if input_mode == "latent":

            self.latent_proj = SparseLinear(latent_dim, hidden_dim)

            self.output_proj = SparseLinear(hidden_dim, latent_dim)
        elif input_mode == "coord_only":

            self.coord_encoder = LearnableFourierPE(
                hidden_dim,
                coord_dim=3,
                num_fourier_feats=coord_encoder_fourier_feats,
                coord_resolution=coord_pe_resolution,
            )

        elif input_mode == "utonia":

            self.utonia_proj = SparseLinear(utonia_feature_dim, hidden_dim)

        elif input_mode in ("latent_utonia", "latent_utonia_1024"):


            self.latent_proj = SparseLinear(latent_dim, hidden_dim)
            self.utonia_proj = SparseLinear(utonia_feature_dim, hidden_dim)
            self.latent_ln = LayerNorm32(hidden_dim, elementwise_affine=True, eps=1e-6)
            self.utonia_ln = LayerNorm32(hidden_dim, elementwise_affine=True, eps=1e-6)
            if input_mode == "latent_utonia_1024":
                self.fusion_proj = nn.Linear(2 * hidden_dim, hidden_dim)




        if self.use_coord_pe:
            self.coord_embedder = SparseCoordEmbedder(
                hidden_dim,
                resolution=coord_pe_resolution,
            )
        if pe_mode == "rope":
            head_dim = hidden_dim // num_heads
            self.rope_embedder = RotaryPositionEmbedder(head_dim, dim=3)
        elif pe_mode == "lff" and input_mode == "latent":

            self.lff_embedder = LearnableFourierPE(
                hidden_dim,
                coord_dim=3,
                num_fourier_feats=num_fourier_feats,
                coord_resolution=coord_pe_resolution,
            )


        if use_image_ca:
            self.img_proj = nn.Linear(img_feature_dim, hidden_dim)


        if query_init_mode == "learnable":


            self.query_embed = nn.Parameter(
                torch.randn(1, max_parts, hidden_dim) * 0.02
            )
            self.part_index_embedder = None
        elif query_init_mode == "positional":

            self.query_embed = None
            self.part_index_embedder = PartIndexEmbedder(hidden_dim)
        elif query_init_mode == "prompt_only":


            self.query_embed = None
            self.part_index_embedder = None
        else:
            raise ValueError(
                f"Unknown query_init_mode: {query_init_mode}. "
                f"Must be 'learnable', 'positional', or 'prompt_only'."
            )


        self.blocks = nn.ModuleList(
            [
                CoLa3DBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio,
                    use_image_ca=use_image_ca,
                    use_rope=(pe_mode == "rope"),
                )
                for _ in range(num_layers)
            ]
        )


        self.p_final_norm = LayerNorm32(hidden_dim, elementwise_affine=True, eps=1e-6)
        self.q_final_norm = LayerNorm32(hidden_dim, elementwise_affine=True, eps=1e-6)

        self._initialize_weights()

    @property
    def point_output_dim(self) -> int:


        if self.input_mode in (
            "coord_only",
            "utonia",
            "latent_utonia",
            "latent_utonia_1024",
        ):
            return self.hidden_dim
        return self.latent_dim

    def _initialize_weights(self) -> None:


        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if hasattr(self, "lff_embedder"):
            nn.init.normal_(self.lff_embedder.Wr.weight, std=1.0)
        if hasattr(self, "coord_encoder"):
            nn.init.normal_(self.coord_encoder.Wr.weight, std=1.0)

    def compute_input_features(
        self,
        latent: SparseTensor,
        ext_features: Optional[SparseTensor] = None,
    ) -> SparseTensor:


        if self.input_mode == "coord_only":
            xyz = latent.coords[:, 1:].float()
            coord_feats = self.coord_encoder(xyz)
            target_dtype = self.coord_encoder.mlp[0].weight.dtype
            p = latent.replace(coord_feats.to(target_dtype))
            return p

        if self.input_mode == "utonia":
            assert (
                ext_features is not None
            ), "ext_features must be provided when input_mode='utonia'"
            return self.utonia_proj(ext_features)

        if self.input_mode in ("latent_utonia", "latent_utonia_1024"):
            assert (
                ext_features is not None
            ), f"ext_features must be provided when input_mode='{self.input_mode}'"
            p_lat = self.latent_proj(latent)
            p_utonia = self.utonia_proj(ext_features)
            p_lat_feats = self.latent_ln(p_lat.feats)
            p_utonia_feats = self.utonia_ln(p_utonia.feats)
            if self.input_mode == "latent_utonia_1024":
                fused = self.fusion_proj(
                    torch.cat([p_lat_feats, p_utonia_feats], dim=-1)
                )
            else:

                fused = p_lat_feats + p_utonia_feats
            return p_lat.replace(fused)


        p = self.latent_proj(latent)
        if self.use_coord_pe:
            coord_pe = self.coord_embedder(latent.coords).to(p.feats.dtype)
            p = p.replace(p.feats + coord_pe)
        if self.pe_mode == "lff" and hasattr(self, "lff_embedder"):
            lff_pe = self.lff_embedder(latent.coords[:, 1:]).to(p.feats.dtype)
            p = p.replace(p.feats + lff_pe)
        return p

    def forward(
        self,
        latent: SparseTensor,
        img_features: Optional[torch.Tensor] = None,
        ext_features: Optional[SparseTensor] = None,
        prompt_queries: Optional[torch.Tensor] = None,
        valid_query_mask: Optional[torch.Tensor] = None,
        prompt_q_coords: Optional[torch.Tensor] = None,
        prefused_input: Optional[SparseTensor] = None,
    ) -> Tuple[SparseTensor, torch.Tensor]:


        batch_size = latent.shape[0]
        if prompt_queries is not None:
            if prompt_queries.ndim != 3:
                raise ValueError(
                    f"prompt_queries must have shape (B, Q, C), got {tuple(prompt_queries.shape)}"
                )
            if (
                prompt_queries.shape[0] != batch_size
                or prompt_queries.shape[2] != self.hidden_dim
            ):
                raise ValueError(
                    "prompt_queries must match decoder batch/hidden dims, "
                    f"got {tuple(prompt_queries.shape)} for batch_size={batch_size}, hidden_dim={self.hidden_dim}"
                )


        if prefused_input is not None:
            if self.input_mode not in ("latent_utonia", "latent_utonia_1024"):
                raise ValueError(
                    f"prefused_input is only valid when input_mode is 'latent_utonia' "
                    f"or 'latent_utonia_1024', got input_mode='{self.input_mode}'"
                )
            if prefused_input.feats.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"prefused_input feats must have last dim == hidden_dim ({self.hidden_dim}), "
                    f"got {prefused_input.feats.shape[-1]}"
                )
            p = prefused_input
        else:
            p = self.compute_input_features(latent, ext_features)


        query_len = (
            prompt_queries.shape[1] if prompt_queries is not None else self.max_parts
        )
        p_phases = q_phases = None
        if self.pe_mode == "rope":
            p_coords = latent.coords[:, 1:].float()
            p_phases = self.rope_embedder(p_coords)

            if prompt_q_coords is not None:


                if prompt_q_coords.shape[:2] != (batch_size, query_len):
                    raise ValueError(
                        "prompt_q_coords must match (B, Q), got "
                        f"{tuple(prompt_q_coords.shape)} vs {(batch_size, query_len)}"
                    )
                q_phases = self.rope_embedder(prompt_q_coords.float())
            else:
                q_indices = torch.arange(
                    query_len, device=latent.device, dtype=torch.float32
                )
                q_coords = q_indices.unsqueeze(-1).expand(-1, 3)
                q_phases = self.rope_embedder(q_coords)


                if valid_query_mask is not None:
                    q_phases = q_phases.unsqueeze(0).expand(batch_size, -1, -1)


        img_tokens = None
        if img_features is not None and self.use_image_ca:

            if self.input_mode == "coord_only":
                ref_dtype = self.coord_encoder.mlp[0].weight.dtype
            elif self.input_mode == "utonia":
                ref_dtype = self.utonia_proj.weight.dtype
            elif self.input_mode in ("latent_utonia", "latent_utonia_1024"):
                ref_dtype = self.latent_proj.weight.dtype
            else:
                ref_dtype = self.latent_proj.weight.dtype
            img_tokens = self.img_proj(img_features.to(ref_dtype))


        if prompt_queries is not None:
            q = prompt_queries
        elif self.query_init_mode == "learnable":
            q = self.query_embed.expand(
                batch_size, -1, -1
            )
        elif self.query_init_mode == "positional":
            q = self.part_index_embedder(self.max_parts, batch_size, latent.device)
        else:
            raise RuntimeError("prompt_only query_init_mode requires prompt_queries")
        if valid_query_mask is not None:
            valid_query_mask = valid_query_mask.to(device=q.device, dtype=torch.bool)
            if valid_query_mask.shape != q.shape[:2]:
                raise ValueError(
                    "valid_query_mask must match query shape (B, Q), "
                    f"got {tuple(valid_query_mask.shape)} vs {tuple(q.shape[:2])}"
                )


        for block in self.blocks:
            p, q = block(
                p,
                q,
                img_tokens=img_tokens,
                p_phases=p_phases,
                q_phases=q_phases,
                valid_query_mask=valid_query_mask,
            )


        p = p.replace(self.p_final_norm(p.feats))
        q = self.q_final_norm(q)
        if valid_query_mask is not None:
            q = q * valid_query_mask.unsqueeze(-1).to(q.dtype)

        if self.input_mode in (
            "coord_only",
            "utonia",
            "latent_utonia",
            "latent_utonia_1024",
        ):

            p_out = p
        else:

            p_out = self.output_proj(p)

        return p_out, q
