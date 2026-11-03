


from fractions import Fraction
from typing import Dict, List, Tuple, Optional, Any

import torch

from .basic import SparseTensor


__all__ = [

    "adapt_channel2spatial_cache",
    "adapt_upsample_cache",
    "adapt_subdivision_cache",
    "adapt_spatial_cache_for_part",
    "extract_part_latent",
    "split_latent_by_parts",

    "build_subdivision_cache_from_gt",
    "reorder_latent_to_match_cache",

    "straight_through_routing",
    "gumbel_sigmoid",

    "pack_part_latents_with_cache",
]


LAZY_COMPUTED_CACHE_KEYS = frozenset(
    ["layout", "seqlen", "cum_seqlen", "batch_boardcast_map"]
)


def adapt_channel2spatial_cache(
    cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    part_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


    original_coords, idx, subidx = cache


    num_original = part_mask.shape[0]
    old_to_new = torch.full(
        (num_original,), -1, dtype=torch.long, device=part_mask.device
    )
    old_to_new[part_mask] = torch.arange(
        part_mask.sum(), device=part_mask.device, dtype=torch.long
    )


    valid_mask = part_mask[idx]


    filtered_coords = original_coords[valid_mask]
    remapped_idx = old_to_new[idx[valid_mask]]
    filtered_subidx = subidx[valid_mask]

    return (filtered_coords, remapped_idx, filtered_subidx)


def adapt_upsample_cache(
    cache: Tuple[torch.Tensor, torch.Tensor],
    part_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:


    original_coords, idx = cache

    num_original = part_mask.shape[0]
    old_to_new = torch.full(
        (num_original,), -1, dtype=torch.long, device=part_mask.device
    )
    old_to_new[part_mask] = torch.arange(
        part_mask.sum(), device=part_mask.device, dtype=torch.long
    )

    valid_mask = part_mask[idx]
    filtered_coords = original_coords[valid_mask]
    remapped_idx = old_to_new[idx[valid_mask]]

    return (filtered_coords, remapped_idx)


def adapt_subdivision_cache(
    cache: torch.Tensor,
    part_mask: torch.Tensor,
) -> torch.Tensor:


    return cache[part_mask]


def adapt_spatial_cache_for_part(
    spatial_cache: Dict[str, Any],
    part_mask: torch.Tensor,
) -> Dict[str, Any]:


    if spatial_cache is None:
        return {}

    adapted = {}
    current_mask = part_mask

    for scale_key, scale_cache in spatial_cache.items():
        if scale_cache is None:
            continue

        adapted[scale_key] = {}
        next_mask = None

        for key, value in scale_cache.items():

            if key in LAZY_COMPUTED_CACHE_KEYS:
                continue

            if value is None:
                continue

            if "channel2spatial" in key.lower():

                adapted_cache = adapt_channel2spatial_cache(value, current_mask)
                adapted[scale_key][key] = adapted_cache

                if next_mask is None:
                    next_mask = torch.ones(
                        adapted_cache[0].shape[0],
                        dtype=torch.bool,
                        device=current_mask.device,
                    )
            elif (
                "upsample" in key.lower()
                and isinstance(value, tuple)
                and len(value) == 2
            ):

                adapted_cache = adapt_upsample_cache(value, current_mask)
                adapted[scale_key][key] = adapted_cache

                if next_mask is None:
                    next_mask = torch.ones(
                        adapted_cache[0].shape[0],
                        dtype=torch.bool,
                        device=current_mask.device,
                    )
            elif key == "subdivision":

                adapted[scale_key][key] = adapt_subdivision_cache(value, current_mask)
            elif key == "shape":

                adapted[scale_key][key] = value
            else:

                if (
                    isinstance(value, torch.Tensor)
                    and value.shape[0] == current_mask.shape[0]
                ):
                    adapted[scale_key][key] = value[current_mask]


        if next_mask is not None:
            current_mask = next_mask

    return adapted


def extract_part_latent(
    latent: SparseTensor,
    part_mask: torch.Tensor,
    adapt_cache: bool = True,
    batch_idx: int = 0,
) -> SparseTensor:


    part_feats = latent.feats[part_mask]


    part_coords = latent.coords[part_mask].clone()


    part_latent = SparseTensor(
        feats=part_feats,
        coords=part_coords,
    )


    part_latent._scale = latent._scale


    if adapt_cache and hasattr(latent, "_spatial_cache") and latent._spatial_cache:
        part_latent._spatial_cache = adapt_spatial_cache_for_part(
            latent._spatial_cache, part_mask
        )

    return part_latent


def split_latent_by_parts(
    latent: SparseTensor,
    part_assignments: torch.Tensor,
    max_parts: int,
    adapt_cache: bool = True,
) -> Dict[int, SparseTensor]:


    parts = {}

    for part_id in range(max_parts):
        part_mask = part_assignments == part_id

        if part_mask.sum() == 0:
            continue

        parts[part_id] = extract_part_latent(latent, part_mask, adapt_cache=adapt_cache)

    return parts


def build_subdivision_cache_from_gt(
    per_obj_coords: torch.Tensor,
    global_shape: Tuple[int, int, int] = (512, 512, 512),
    downsample_factor: int = 2,
    num_levels: int = 4,
    return_latent_coords: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:


    DIM = 3
    coords = per_obj_coords.clone()
    device = coords.device
    caches: Dict[str, Dict[str, torch.Tensor]] = {}


    if coords.shape[-1] == 3:

        has_batch_idx = False
        spatial_coords = coords
    elif coords.shape[-1] == 4:

        has_batch_idx = True
        spatial_coords = coords[:, 1:]
    else:
        raise ValueError(f"Expected coords shape (N, 3) or (N, 4), got {coords.shape}")


    current_global_shape = list(global_shape)


    current_scale = (Fraction(1, 1), Fraction(1, 1), Fraction(1, 1))

    for level in range(num_levels):

        batch_col = torch.zeros(
            (spatial_coords.shape[0], 1), device=device, dtype=spatial_coords.dtype
        )
        coords_4d = torch.cat([batch_col, spatial_coords], dim=1)


        down_coords_4d = coords_4d.clone()
        down_coords_4d[:, 1:] = coords_4d[:, 1:] // downsample_factor


        actual_bbox = (spatial_coords.max(dim=0).values + 1).tolist()
        actual_max = [
            (int(s) + downsample_factor - 1) // downsample_factor for s in actual_bbox
        ]


        MAX = [
            (s + downsample_factor - 1) // downsample_factor
            for s in current_global_shape
        ]


        OFFSET = torch.cumprod(torch.tensor(MAX[::-1], device=device), 0).tolist()[
            ::-1
        ] + [
            1
        ]


        coord_list = [down_coords_4d[:, i] for i in range(DIM + 1)]
        code = sum([c * o for c, o in zip(coord_list, OFFSET)])
        new_code, idx = code.unique(return_inverse=True)


        subidx = spatial_coords % downsample_factor
        subidx = sum([subidx[..., i] * downsample_factor**i for i in range(DIM)])
        subdivision = torch.zeros(
            new_code.shape[0], downsample_factor**DIM, device=device, dtype=torch.bool
        )
        subdivision[idx, subidx] = True


        new_coords_4d = torch.stack(
            [new_code // OFFSET[0]]
            + [(new_code // OFFSET[i + 1]) % MAX[i] for i in range(DIM)],
            dim=-1,
        )
        new_coords = new_coords_4d[:, 1:]


        post_downsample_scale = tuple([s * downsample_factor for s in current_scale])
        scale_key = str(post_downsample_scale)
        caches[scale_key] = {}


        batch_idx = torch.zeros(
            (spatial_coords.shape[0], 1), device=device, dtype=spatial_coords.dtype
        )
        spatial_coords_4d = torch.cat([batch_idx, spatial_coords], dim=1)


        caches[scale_key]["subdivision"] = subdivision
        caches[scale_key][f"channel2spatial_{downsample_factor}"] = (
            spatial_coords_4d,
            idx,
            subidx,
        )
        caches[scale_key]["shape"] = torch.Size(actual_max)


        spatial_coords = new_coords
        current_global_shape = MAX
        current_scale = post_downsample_scale

    if return_latent_coords:

        return caches, spatial_coords
    return caches


def reorder_latent_to_match_cache(
    part_latent: "SparseTensor",
    expected_coords: torch.Tensor,
) -> "SparseTensor":


    if part_latent.coords.shape[1] == 4:
        actual_coords = part_latent.coords[:, 1:4]
    else:
        actual_coords = part_latent.coords


    if actual_coords.shape[0] != expected_coords.shape[0]:
        raise ValueError(
            f"Coord count mismatch: part_latent has {actual_coords.shape[0]}, "
            f"expected {expected_coords.shape[0]}"
        )


    max_coord = max(actual_coords.max().item(), expected_coords.max().item()) + 1

    def encode_coords(c):
        return c[:, 0] * max_coord * max_coord + c[:, 1] * max_coord + c[:, 2]

    actual_encoded = encode_coords(actual_coords)
    expected_encoded = encode_coords(expected_coords)


    actual_set = set(actual_encoded.tolist())
    expected_set = set(expected_encoded.tolist())

    if actual_set != expected_set:
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        raise ValueError(
            f"Coord position mismatch: {len(missing)} missing, {len(extra)} extra. "
            f"This indicates data inconsistency between GT mask selection and per_obj_coords."
        )


    actual_lookup = {v.item(): i for i, v in enumerate(actual_encoded)}


    reorder_idx = torch.tensor(
        [actual_lookup[v.item()] for v in expected_encoded],
        device=part_latent.device,
        dtype=torch.long,
    )


    new_feats = part_latent.feats[reorder_idx]


    batch_idx = torch.zeros(
        (expected_coords.shape[0], 1),
        device=part_latent.device,
        dtype=part_latent.coords.dtype,
    )
    new_coords = torch.cat([batch_idx, expected_coords.to(part_latent.device)], dim=1)


    from .basic import SparseTensor

    new_latent = SparseTensor(new_feats, new_coords)
    new_latent._scale = part_latent._scale

    return new_latent


def straight_through_routing(
    logits: torch.Tensor,
    threshold: float = 0.0,
    tau: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:


    p_soft = torch.sigmoid((logits - threshold) / tau)


    m_hard = p_soft > 0.5


    g_st = m_hard.float() + p_soft - p_soft.detach()

    return m_hard, g_st


def gumbel_sigmoid(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:


    u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
    gumbel_noise = torch.log(u) - torch.log(1 - u)

    noisy_logits = (logits + gumbel_noise) / temperature
    soft_sample = torch.sigmoid(noisy_logits)

    if hard:
        hard_sample = (soft_sample > 0.5).float()

        return hard_sample - soft_sample.detach() + soft_sample
    else:
        return soft_sample


def pack_part_latents_with_cache(
    part_latents: List[SparseTensor],
    part_ids: List[int],
) -> Tuple[SparseTensor, torch.Tensor, Dict[int, slice]]:


    if len(part_latents) == 0:
        raise ValueError("Cannot pack empty list of latents")

    if len(part_latents) == 1:
        part_id_map = torch.full(
            (part_latents[0].feats.shape[0],),
            part_ids[0],
            device=part_latents[0].device,
            dtype=torch.long,
        )
        return (
            part_latents[0],
            part_id_map,
            {part_ids[0]: slice(0, part_latents[0].feats.shape[0])},
        )


    all_feats = []
    all_coords = []
    all_part_ids = []
    layout = {}
    offset = 0

    for lat, pid in zip(part_latents, part_ids):
        n = lat.feats.shape[0]
        all_feats.append(lat.feats)
        all_coords.append(lat.coords)
        all_part_ids.extend([pid] * n)
        layout[pid] = slice(offset, offset + n)
        offset += n

    packed_feats = torch.cat(all_feats, dim=0)
    packed_coords = torch.cat(all_coords, dim=0)
    part_id_map = torch.tensor(
        all_part_ids, device=packed_feats.device, dtype=torch.long
    )


    packed = SparseTensor(feats=packed_feats, coords=packed_coords)
    packed._scale = part_latents[0]._scale


    return packed, part_id_map, layout
