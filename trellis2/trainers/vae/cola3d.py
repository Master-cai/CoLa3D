

from typing import Dict, List, Tuple, Optional, Union
import functools
from fractions import Fraction
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import utils3d
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from .shape_vae import ShapeVaeTrainer
from ...modules import sparse as sp
from ...modules.hungarian_matcher import HungarianMatcher
from ...models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder, merge_overlapping_voxels
from ...representations import Mesh
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler

__all__ = ["CoLa3DTrainer", "DecomposableShapeVaeTrainer"]


def _restore_sparse_scale(
    tensor: sp.SparseTensor, device, expected_scale: Fraction
) -> sp.SparseTensor:
    tensor = tensor.to(device)
    if tensor._scale[0] != expected_scale:
        tensor._scale = tuple((expected_scale for _ in range(3)))
    return tensor


DOWNSAMPLE_RATIO = 16


def make_part_palette(n: int, device=None) -> torch.Tensor:

    import colorsys

    n = max(int(n), 1)
    cols = []
    for i in range(n):
        s = 0.85 if i % 2 == 0 else 0.6
        v = 0.95 if i % 2 == 0 else 0.78
        cols.append(colorsys.hsv_to_rgb(i / n, s, v))
    palette = torch.tensor(cols, dtype=torch.float32)
    return palette.to(device) if device is not None else palette


class CoLa3DTrainer(ShapeVaeTrainer):


    def __init__(
        self,
        *args,
        lambda_seg: float = 1.0,
        lambda_seg_ce: float = 1.0,
        lambda_seg_dice: float = 1.0,
        image_cond_model: Optional[dict] = None,
        use_diversity_loss: bool = False,
        lambda_diversity: float = 0.1,
        diversity_mode: str = "none",
        snapshot_master_only: bool = True,
        snapshot_use_distributed_collectives: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_seg = lambda_seg
        self.lambda_seg_ce = lambda_seg_ce
        self.lambda_seg_dice = lambda_seg_dice
        self.image_cond_model_config = image_cond_model
        self._image_cond_model = None
        self.use_diversity_loss = use_diversity_loss
        self.lambda_diversity = lambda_diversity
        self.diversity_mode = diversity_mode
        self.snapshot_master_only = snapshot_master_only
        self.snapshot_use_distributed_collectives = snapshot_use_distributed_collectives
        if diversity_mode not in {"none", "all", "matched"}:
            raise ValueError(f"Unknown diversity_mode: {diversity_mode}")
        self.matcher = HungarianMatcher()

    def _init_image_cond_model(self) -> None:

        from ...modules import image_feature_extractor
        from ...utils import dist_utils

        config = self.image_cond_model_config
        model_name = config.get("name", "DinoV3FeatureExtractor")
        if not hasattr(image_feature_extractor, model_name):
            raise ValueError(f"Unknown image_cond_model: {model_name}")
        model_cls = getattr(image_feature_extractor, model_name)
        model_args = dict(config.get("args", {}))
        dataset_image_size = getattr(self.dataset, "image_size", None)
        if dataset_image_size is not None:
            model_args["image_size"] = dataset_image_size
        with dist_utils.local_master_first():
            self._image_cond_model = model_cls(**model_args)
            self._image_cond_model.cuda()

    @torch.no_grad()
    def _extract_image_features(self, cond_image: torch.Tensor) -> torch.Tensor:

        if self._image_cond_model is None:
            self._init_image_cond_model()
        return self._image_cond_model(cond_image)

    def _randomize_camera_at(self, center: torch.Tensor) -> Dict:

        r_min, r_max = self.camera_randomization_config["radius_range"]
        k_min = 1 / r_max**2
        k_max = 1 / r_min**2
        ks = torch.rand(1, device=self.device) * (k_max - k_min) + k_min
        radius = 1 / torch.sqrt(ks)
        fov = 2 * torch.arcsin(0.5 / radius)
        direction = F.normalize(torch.randn(1, 3, device=self.device), dim=-1)
        origin = center.unsqueeze(0) + radius.unsqueeze(-1) * direction
        target = center.unsqueeze(0)
        up = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)
        extrinsics = utils3d.torch.extrinsics_look_at(origin, target, up)
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        near = [np.random.uniform(radius.item() - 1, radius.item())]
        return {"extrinsics": extrinsics, "intrinsics": intrinsics, "near": near}

    def _downsample_multihot_gt(
        self,
        multihot_gt: torch.Tensor,
        vertices: sp.SparseTensor,
        z: sp.SparseTensor,
        downsample_ratio: int = 16,
        max_parts: int = 8,
    ) -> torch.Tensor:


        device = z.device
        orig_res = self.models["decoder"].resolution
        latent_res = orig_res // DOWNSAMPLE_RATIO
        batch_size = z.shape[0]
        max_points = max(
            (z.layout[b].stop - z.layout[b].start for b in range(batch_size))
        )
        gt_masks = torch.zeros(
            batch_size, max_points, max_parts, dtype=torch.float32, device=device
        )
        for b in range(batch_size):
            v_start = vertices.layout[b].start
            v_end = vertices.layout[b].stop
            batch_orig_coords = vertices.coords[v_start:v_end, 1:4].long()
            batch_multihot = multihot_gt[v_start:v_end]
            multihot_grid = torch.zeros(
                latent_res,
                latent_res,
                latent_res,
                max_parts,
                dtype=torch.float32,
                device=device,
            )
            ds_coords = batch_orig_coords // downsample_ratio
            ds_coords = ds_coords.clamp(0, latent_res - 1)
            linear_idx = (
                ds_coords[:, 0] * latent_res * latent_res
                + ds_coords[:, 1] * latent_res
                + ds_coords[:, 2]
            )
            grid_flat = multihot_grid.view(-1, max_parts)
            linear_idx_expanded = linear_idx.unsqueeze(1).expand(-1, max_parts)
            grid_flat.scatter_reduce_(
                dim=0,
                index=linear_idx_expanded,
                src=batch_multihot.to(device),
                reduce="amax",
                include_self=True,
            )
            multihot_grid = grid_flat.view(
                latent_res, latent_res, latent_res, max_parts
            )
            z_start = z.layout[b].start
            z_end = z.layout[b].stop
            n_points = z_end - z_start
            batch_z_coords = z.coords[z_start:z_end, 1:4].long()
            batch_z_coords = batch_z_coords.clamp(0, latent_res - 1)
            gt_masks[b, :n_points] = multihot_grid[
                batch_z_coords[:, 0], batch_z_coords[:, 1], batch_z_coords[:, 2]
            ]
        return gt_masks

    def prepare_dataloader(self, **kwargs):

        self.data_sampler = BalancedResumableSampler(
            self.dataset, shuffle=True, batch_size=self.batch_size_per_gpu
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.dataloader_workers > 0,
            prefetch_factor=4 if self.dataloader_workers > 0 else None,
            collate_fn=functools.partial(
                self.dataset.collate_fn, split_size=self.batch_split
            ),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    @staticmethod
    def _is_prompt_query_source(query_source: str) -> bool:

        return query_source in {"prompt_mask", "prompt_point"}

    @staticmethod
    def _get_prompt_num_valid(
        prompt_num_queries: Union[torch.Tensor, List[int], Tuple[int, ...], int],
        batch_idx: int,
        available_queries: int,
    ) -> int:

        if isinstance(prompt_num_queries, torch.Tensor):
            if prompt_num_queries.ndim == 0:
                requested_queries = int(prompt_num_queries.item())
            else:
                requested_queries = int(prompt_num_queries[batch_idx].item())
        elif isinstance(prompt_num_queries, (list, tuple)):
            requested_queries = int(prompt_num_queries[batch_idx])
        else:
            requested_queries = int(prompt_num_queries)
        return max(0, min(requested_queries, available_queries))

    def _prepare_prompt_query_batch(
        self,
        prompt_queries: Optional[Union[List[torch.Tensor], torch.Tensor]],
        prompt_num_queries: Optional[
            Union[torch.Tensor, List[int], Tuple[int, ...], int]
        ],
        max_queries: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        if prompt_queries is None or prompt_num_queries is None:
            return (None, None)
        if isinstance(prompt_queries, torch.Tensor):
            if prompt_queries.ndim == 2:
                prompt_queries = prompt_queries.unsqueeze(0)
            target_queries = (
                prompt_queries.shape[1] if max_queries is None else max_queries
            )
            padded_queries = torch.zeros(
                prompt_queries.shape[0],
                target_queries,
                *prompt_queries.shape[2:],
                dtype=prompt_queries.dtype,
                device=prompt_queries.device,
            )
            prompt_valid_mask = torch.zeros(
                (prompt_queries.shape[0], target_queries),
                dtype=torch.bool,
                device=prompt_queries.device,
            )
            for batch_idx in range(prompt_queries.shape[0]):
                num_valid = self._get_prompt_num_valid(
                    prompt_num_queries,
                    batch_idx,
                    min(prompt_queries.shape[1], target_queries),
                )
                if num_valid > 0:
                    padded_queries[batch_idx, :num_valid] = prompt_queries[
                        batch_idx, :num_valid
                    ]
                prompt_valid_mask[batch_idx, :num_valid] = True
            return (padded_queries.to(self.device), prompt_valid_mask.to(self.device))
        batch_size = len(prompt_queries)
        sample_query = next(
            (query for query in prompt_queries if query is not None), None
        )
        if sample_query is None:
            return (None, None)
        available_queries = [
            query.shape[0] for query in prompt_queries if query is not None
        ]
        target_queries = max(available_queries) if max_queries is None else max_queries
        padded_queries = torch.zeros(
            batch_size,
            target_queries,
            *sample_query.shape[1:],
            dtype=sample_query.dtype,
            device=self.device,
        )
        prompt_valid_mask = torch.zeros(
            batch_size, target_queries, dtype=torch.bool, device=self.device
        )
        for batch_idx, query in enumerate(prompt_queries):
            if query is None:
                continue
            num_valid = self._get_prompt_num_valid(
                prompt_num_queries, batch_idx, min(query.shape[0], target_queries)
            )
            if num_valid > 0:
                padded_queries[batch_idx, :num_valid] = query[:num_valid].to(
                    self.device
                )
            prompt_valid_mask[batch_idx, :num_valid] = True
        return (padded_queries, prompt_valid_mask)

    def _prepare_prompt_object_ids(
        self,
        prompt_object_ids: Optional[Union[List[torch.Tensor], torch.Tensor]],
        prompt_num_queries: Optional[
            Union[torch.Tensor, List[int], Tuple[int, ...], int]
        ],
        max_queries: Optional[int] = None,
    ) -> Optional[torch.Tensor]:

        if prompt_object_ids is None or prompt_num_queries is None:
            return None
        if isinstance(prompt_object_ids, torch.Tensor):
            if prompt_object_ids.ndim == 1:
                prompt_object_ids = prompt_object_ids.unsqueeze(0)
            target_queries = (
                prompt_object_ids.shape[1] if max_queries is None else max_queries
            )
            padded_ids = torch.full(
                (prompt_object_ids.shape[0], target_queries),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            for batch_idx in range(prompt_object_ids.shape[0]):
                num_valid = self._get_prompt_num_valid(
                    prompt_num_queries,
                    batch_idx,
                    min(prompt_object_ids.shape[1], target_queries),
                )
                padded_ids[batch_idx, :num_valid] = prompt_object_ids[
                    batch_idx, :num_valid
                ].to(self.device, dtype=torch.long)
            return padded_ids
        batch_size = len(prompt_object_ids)
        available_queries = [
            ids.shape[0] for ids in prompt_object_ids if ids is not None
        ]
        target_queries = max(available_queries) if max_queries is None else max_queries
        padded_ids = torch.full(
            (batch_size, target_queries), -1, dtype=torch.long, device=self.device
        )
        for batch_idx, object_ids in enumerate(prompt_object_ids):
            if object_ids is None:
                continue
            num_valid = self._get_prompt_num_valid(
                prompt_num_queries, batch_idx, min(object_ids.shape[0], target_queries)
            )
            padded_ids[batch_idx, :num_valid] = object_ids[:num_valid].to(
                self.device, dtype=torch.long
            )
        return padded_ids

    def _prepare_prompt_inputs(
        self, query_source: str, batch_kwargs: Dict, max_queries: int
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, Optional[torch.Tensor]],
    ]:

        prompt_inputs = None
        prompt_valid_mask = None
        prompt_object_ids = None
        decoder_prompt_kwargs: Dict[str, Optional[torch.Tensor]] = {}
        if query_source == "prompt_point":
            prompt_inputs, prompt_valid_mask = self._prepare_prompt_query_batch(
                batch_kwargs.get("prompt_points"),
                batch_kwargs.get(
                    "prompt_point_num_queries", batch_kwargs.get("prompt_num_queries")
                ),
                max_queries=max_queries,
            )
            prompt_object_ids = self._prepare_prompt_object_ids(
                batch_kwargs.get(
                    "prompt_point_object_ids", batch_kwargs.get("prompt_object_ids")
                ),
                batch_kwargs.get(
                    "prompt_point_num_queries", batch_kwargs.get("prompt_num_queries")
                ),
                max_queries=max_queries,
            )
            decoder_prompt_kwargs["prompt_points"] = prompt_inputs
            decoder_prompt_kwargs["prompt_valid_mask"] = prompt_valid_mask
        elif query_source == "prompt_mask":
            prompt_inputs, prompt_valid_mask = self._prepare_prompt_query_batch(
                batch_kwargs.get("prompt_masks"),
                batch_kwargs.get(
                    "prompt_mask_num_queries", batch_kwargs.get("prompt_num_queries")
                ),
                max_queries=max_queries,
            )
            prompt_object_ids = self._prepare_prompt_object_ids(
                batch_kwargs.get(
                    "prompt_mask_object_ids", batch_kwargs.get("prompt_object_ids")
                ),
                batch_kwargs.get(
                    "prompt_mask_num_queries", batch_kwargs.get("prompt_num_queries")
                ),
                max_queries=max_queries,
            )
            decoder_prompt_kwargs["prompt_masks"] = prompt_inputs
            decoder_prompt_kwargs["prompt_valid_mask"] = prompt_valid_mask
        return (
            prompt_inputs,
            prompt_valid_mask,
            prompt_object_ids,
            decoder_prompt_kwargs,
        )

    @staticmethod
    def _sanitize_prompt_matching(
        prompt_object_ids: Optional[torch.Tensor],
        num_objects: Optional[torch.Tensor],
        num_queries: int,
        valid_query_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:

        if prompt_object_ids is None or num_objects is None:
            return None
        matching = torch.full(
            (prompt_object_ids.shape[0], num_queries),
            -1,
            dtype=torch.long,
            device=prompt_object_ids.device,
        )
        copy_queries = min(num_queries, prompt_object_ids.shape[1])
        matching[:, :copy_queries] = prompt_object_ids[:, :copy_queries]
        for batch_idx in range(matching.shape[0]):
            max_objects = int(num_objects[batch_idx].item())
            sample_match = matching[batch_idx]
            if valid_query_mask is not None:
                sample_valid_queries = valid_query_mask[batch_idx].to(
                    device=sample_match.device, dtype=torch.bool
                )
            else:
                sample_valid_queries = sample_match >= 0
            sample_match[~sample_valid_queries] = -1
        return matching

    def _compute_diversity_loss(
        self, query_feats: torch.Tensor, valid_query_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if valid_query_mask is None:
            q_norm = F.normalize(query_feats.float(), dim=-1)
            sim = torch.bmm(q_norm, q_norm.transpose(1, 2))
            num_queries = sim.shape[1]
            mask = (
                ~torch.eye(num_queries, device=sim.device, dtype=torch.bool)
                .unsqueeze(0)
                .expand_as(sim)
            )
            return sim[mask].abs().mean()
        losses: List[torch.Tensor] = []
        for batch_idx in range(query_feats.shape[0]):
            sample_valid = valid_query_mask[batch_idx].bool()
            num_valid = int(sample_valid.sum().item())
            if num_valid < 2:
                continue
            sample_queries = F.normalize(
                query_feats[batch_idx, sample_valid].float(), dim=-1
            )
            sim = sample_queries @ sample_queries.transpose(0, 1)
            off_diagonal_mask = ~torch.eye(
                num_valid, device=sim.device, dtype=torch.bool
            )
            losses.append(sim[off_diagonal_mask].abs().mean())
        if len(losses) == 0:
            return query_feats.new_zeros(())
        return torch.stack(losses).mean()

    def _compute_diversity_loss_matched(
        self, query_feats: torch.Tensor, matching: torch.Tensor
    ) -> torch.Tensor:

        losses: List[torch.Tensor] = []
        for batch_idx in range(query_feats.shape[0]):
            matched_mask = matching[batch_idx] != -1
            num_matched = int(matched_mask.sum().item())
            if num_matched < 2:
                continue
            matched_queries = query_feats[batch_idx, matched_mask]
            matched_queries = F.normalize(matched_queries.float(), dim=-1)
            sim = matched_queries @ matched_queries.transpose(0, 1)
            off_diagonal_mask = ~torch.eye(
                num_matched, device=sim.device, dtype=torch.bool
            )
            losses.append(sim[off_diagonal_mask].abs().mean())
        if len(losses) == 0:
            return query_feats.new_zeros(())
        return torch.stack(losses).mean()

    def _get_diversity_term(
        self,
        query_feats: Optional[torch.Tensor],
        matching: Optional[torch.Tensor] = None,
        valid_query_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:

        if query_feats is None or not self.use_diversity_loss:
            return None
        if self.diversity_mode == "none":
            return None
        if self.diversity_mode == "all":
            return self._compute_diversity_loss(
                query_feats, valid_query_mask=valid_query_mask
            )
        if self.diversity_mode == "matched":
            if matching is None:
                return None
            return self._compute_diversity_loss_matched(query_feats, matching)
        raise AssertionError(f"Unhandled diversity_mode: {self.diversity_mode}")

    def _compute_segmentation_loss(
        self,
        mask_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        num_objects: torch.Tensor,
        valid_mask: torch.Tensor,
        valid_query_mask: Optional[torch.Tensor] = None,
        direct_matching: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


        from ...modules.hungarian_matcher import dice_loss_per_class

        if direct_matching is not None:
            matching = direct_matching.to(mask_logits.device, dtype=torch.long)
        else:
            matching = self.matcher(
                mask_logits,
                gt_masks,
                num_objects,
                valid_mask,
                valid_queries=valid_query_mask,
            )
        gt_reordered = self.matcher.get_reordered_gt(matching, gt_masks)
        bce_terms: List[torch.Tensor] = []
        dice_terms: List[torch.Tensor] = []
        for batch_idx in range(mask_logits.shape[0]):
            sample_valid_pts = valid_mask[batch_idx].bool()
            if valid_query_mask is not None:
                sample_valid_queries = valid_query_mask[batch_idx].bool()
            else:
                sample_valid_queries = torch.ones(
                    mask_logits.shape[2], dtype=torch.bool, device=mask_logits.device
                )
            if int(sample_valid_queries.sum().item()) == 0:
                continue
            logits_valid = mask_logits[batch_idx][sample_valid_pts][
                :, sample_valid_queries
            ]
            gt_valid = gt_reordered[batch_idx][sample_valid_pts][
                :, sample_valid_queries
            ]
            bce_terms.append(F.binary_cross_entropy_with_logits(logits_valid, gt_valid))
            pred_probs = torch.sigmoid(logits_valid)
            dice_terms.append(dice_loss_per_class(pred_probs, gt_valid))
        if len(bce_terms) == 0:
            zero = mask_logits.new_zeros(())
            return (zero, zero, matching)
        bce_loss = torch.stack(bce_terms).mean()
        dice = torch.stack(dice_terms).mean()
        return (bce_loss, dice, matching)

    def training_losses(
        self,
        vertices: sp.SparseTensor,
        intersected: sp.SparseTensor,
        mesh: Optional[List[Mesh]] = None,
        instance_id: Optional[torch.Tensor] = None,
        num_objects: Optional[torch.Tensor] = None,
        per_obj_vertices: Optional[List] = None,
        per_obj_intersected: Optional[List] = None,
        per_obj_meshes: Optional[List] = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:


        max_parts = self.models["decoder"].max_parts
        if instance_id is not None:
            if isinstance(instance_id, list):
                instance_id_flat = torch.cat(
                    [x.to(vertices.device) for x in instance_id], dim=0
                )
            elif instance_id.dim() == 2:
                instance_id_flat = instance_id.flatten().to(vertices.device)
            else:
                instance_id_flat = instance_id.to(vertices.device)
        else:
            instance_id_flat = None
        vertices, intersected, multihot_gt_512 = merge_overlapping_voxels(
            vertices, intersected, instance_id_flat, max_objects=max_parts
        )
        expected_scale = Fraction(DOWNSAMPLE_RATIO, 1)
        precomputed_z = kwargs.get("precomputed_z")
        if precomputed_z is not None:
            z = _restore_sparse_scale(precomputed_z, vertices.device, expected_scale)
            mean = None
            logvar = None
        else:
            z, mean, logvar = self.training_models["encoder"](
                vertices, intersected, sample_posterior=True, return_raw=True
            )
        precomputed_utonia = kwargs.get("precomputed_utonia")
        if precomputed_utonia is not None:
            kwargs["precomputed_utonia"] = _restore_sparse_scale(
                precomputed_utonia, vertices.device, expected_scale
            )
        decoder = self.training_models["decoder"]
        decoder_model = self.models["decoder"]
        query_source = getattr(decoder_model, "query_source", "fixed")
        _, prompt_valid_mask, prompt_object_ids, decoder_prompt_kwargs = (
            self._prepare_prompt_inputs(query_source, kwargs, decoder_model.max_parts)
        )
        img_features = None
        cond_image = kwargs.get("cond_image")
        if cond_image is not None and self.image_cond_model_config is not None:
            img_features = self._extract_image_features(cond_image)
        output = decoder(
            z,
            intersected,
            parallel_decode=False,
            return_intermediate=True,
            also_parallel_decode=False,
            skip_unified_decode=True,
            seg_only=True,
            per_obj_coords=kwargs.get("per_obj_coords"),
            gt_multihot_latent=None,
            gt_num_objects=None,
            per_obj_intersected=per_obj_intersected,
            img_features=img_features,
            precomputed_utonia=kwargs.get("precomputed_utonia"),
            **decoder_prompt_kwargs,
        )
        _, mask_logits, query_feats, *_ = output
        terms = edict(loss=0.0)
        if mean is not None and logvar is not None:
            logvar_clamped = logvar.clamp(-20, 10)
            terms["kl"] = 0.5 * torch.mean(
                mean.pow(2) + logvar_clamped.exp() - logvar_clamped - 1
            )
            terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
        else:
            terms["kl"] = torch.tensor(0.0, device=vertices.device)
        matching = None
        if (
            instance_id is not None
            and num_objects is not None
            and (multihot_gt_512 is not None)
        ):
            batch_size = mask_logits.shape[0]
            max_points = mask_logits.shape[1]
            max_parts = decoder_model.max_parts
            gt_masks = self._downsample_multihot_gt(
                multihot_gt_512,
                vertices,
                z,
                downsample_ratio=DOWNSAMPLE_RATIO,
                max_parts=max_parts,
            )
            valid_mask = torch.zeros(
                batch_size, max_points, dtype=torch.bool, device=mask_logits.device
            )
            for b in range(batch_size):
                n_points = z.layout[b].stop - z.layout[b].start
                valid_mask[b, :n_points] = True
            if isinstance(num_objects, list):
                num_objects = torch.tensor(num_objects, device=mask_logits.device)
            elif not isinstance(num_objects, torch.Tensor):
                num_objects = torch.tensor([num_objects], device=mask_logits.device)
            else:
                num_objects = num_objects.to(mask_logits.device)
            direct_matching = None
            if self._is_prompt_query_source(query_source):
                direct_matching = self._sanitize_prompt_matching(
                    prompt_object_ids,
                    num_objects,
                    mask_logits.shape[-1],
                    valid_query_mask=prompt_valid_mask,
                )
            seg_bce, seg_dice, matching = self._compute_segmentation_loss(
                mask_logits,
                gt_masks,
                num_objects,
                valid_mask,
                valid_query_mask=prompt_valid_mask,
                direct_matching=direct_matching,
            )
            terms["seg/bce"] = seg_bce
            terms["seg/dice"] = seg_dice
            terms["loss"] = terms["loss"] + self.lambda_seg * (
                self.lambda_seg_ce * terms["seg/bce"]
                + self.lambda_seg_dice * terms["seg/dice"]
            )
        diversity_term = self._get_diversity_term(
            query_feats, matching=matching, valid_query_mask=prompt_valid_mask
        )
        if diversity_term is not None:
            terms["diversity"] = diversity_term
            terms["loss"] = terms["loss"] + self.lambda_diversity * terms["diversity"]
        return (terms, {})

    @torch.no_grad()
    def run_snapshot(
        self, num_samples: int, batch_size: int, verbose: bool = False
    ) -> Dict:

        import copy
        import warnings
        from tqdm import tqdm

        decoder = self.models["decoder"]
        part_colors = make_part_palette(decoder.max_parts, self.device)

        def _make_empty_mesh() -> Mesh:
            return Mesh(
                vertices=torch.zeros(0, 3, dtype=torch.float32, device=self.device),
                faces=torch.zeros(0, 3, dtype=torch.int32, device=self.device),
            )

        def _decode_part_from_latent(
            sample_latent: sp.SparseTensor, part_mask: torch.Tensor
        ) -> Mesh:
            if part_mask.sum() == 0:
                return _make_empty_mesh()
            part_feats = sample_latent.feats[part_mask]
            part_coords = sample_latent.coords[part_mask].clone()
            part_coords[:, 0] = 0
            part_latent = sp.SparseTensor(part_feats, part_coords)
            part_latent._scale = sample_latent._scale
            try:
                decoded = FlexiDualGridVaeDecoder.forward(
                    decoder, part_latent, gt_intersected=None
                )
            except (IndexError, RuntimeError) as e:
                warnings.warn(f"Seg-only part decode failed: {e}", stacklevel=2)
                try:
                    torch.cuda.synchronize()
                except RuntimeError:
                    pass
                return _make_empty_mesh()
            mesh_list = decoded if isinstance(decoded, list) else [decoded]
            return mesh_list[0] if len(mesh_list) > 0 else _make_empty_mesh()

        def _merge_colored_parts(
            batch_parts: List[List[Mesh]],
            color_indices: Optional[List[List[int]]] = None,
        ) -> List[Mesh]:
            merged_meshes: List[Mesh] = []
            for sample_idx, parts in enumerate(batch_parts):
                vertices: List[torch.Tensor] = []
                faces: List[torch.Tensor] = []
                colors: List[torch.Tensor] = []
                vertex_offset = 0
                sample_colors = (
                    color_indices[sample_idx] if color_indices is not None else None
                )
                for part_idx, part_mesh in enumerate(parts):
                    if (
                        part_mesh.vertices.shape[0] == 0
                        or part_mesh.faces.shape[0] == 0
                    ):
                        continue
                    vertices.append(part_mesh.vertices)
                    faces.append(part_mesh.faces + vertex_offset)
                    color_idx = (
                        sample_colors[part_idx]
                        if sample_colors is not None and part_idx < len(sample_colors)
                        else part_idx
                    )
                    color = part_colors[color_idx % len(part_colors)]
                    colors.append(
                        color.unsqueeze(0).expand(part_mesh.vertices.shape[0], 3)
                    )
                    vertex_offset += part_mesh.vertices.shape[0]
                if vertices:
                    merged_meshes.append(
                        Mesh(
                            vertices=torch.cat(vertices, dim=0),
                            faces=torch.cat(faces, dim=0),
                            vertex_attrs=torch.cat(colors, dim=0),
                        )
                    )
                else:
                    merged_meshes.append(
                        Mesh(
                            vertices=torch.zeros(0, 3, device=self.device),
                            faces=torch.zeros(
                                0, 3, dtype=torch.int32, device=self.device
                            ),
                            vertex_attrs=torch.zeros(0, 3, device=self.device),
                        )
                    )
            return merged_meshes

        def _render_part_strips(
            batch_parts: List[List[Mesh]], ref: torch.Tensor
        ) -> torch.Tensor:
            _, channels, height, width = ref.shape
            black = torch.zeros(channels, height, width, device=self.device)
            import math

            strips: List[torch.Tensor] = []
            num_slots = max((len(parts) for parts in batch_parts), default=0)
            if num_slots == 0:
                num_slots = 1
            ncols = max(1, int(math.ceil(math.sqrt(num_slots))))
            nrows = max(1, int(math.ceil(num_slots / ncols)))
            for parts in batch_parts:
                part_views: List[torch.Tensor] = []
                for part_idx in range(num_slots):
                    if (
                        part_idx < len(parts)
                        and parts[part_idx].vertices.shape[0] > 0
                        and (parts[part_idx].faces.shape[0] > 0)
                    ):
                        try:
                            center = parts[part_idx].vertices.detach().mean(dim=0)
                            cam = self._randomize_camera_at(center)
                            render_out = self._render_batch(
                                [parts[part_idx]], **cam, return_types=["normal"]
                            )
                            part_views.append(render_out["normal"][0])
                        except Exception:
                            try:
                                torch.cuda.synchronize()
                            except RuntimeError:
                                pass
                            part_views.append(black)
                    else:
                        part_views.append(black)
                while len(part_views) < nrows * ncols:
                    part_views.append(black)
                grid_rows = [
                    torch.cat(part_views[r * ncols : (r + 1) * ncols], dim=-1)
                    for r in range(nrows)
                ]
                strips.append(torch.cat(grid_rows, dim=-2))
            return torch.stack(strips, dim=0)

        def _render_prompt_point_placeholders(
            batch_prompt_points: List[Optional[torch.Tensor]],
            batch_prompt_object_ids: List[Optional[torch.Tensor]],
            batch_prompt_counts: List[int],
            num_slots: int,
        ) -> torch.Tensor:
            tile_size = min(128, getattr(self, "render_resolution", 512))
            tile_size = max(32, tile_size)
            num_slots = max(1, num_slots)
            import math

            grid_ncols = max(1, int(math.ceil(math.sqrt(num_slots))))
            grid_nrows = max(1, int(math.ceil(num_slots / grid_ncols)))
            strips: List[torch.Tensor] = []
            latent_res = max(
                1, self.models["decoder"].resolution // DOWNSAMPLE_RATIO - 1
            )
            axis_palette = torch.eye(3, dtype=torch.float32, device=self.device)
            for sample_points, sample_object_ids, num_valid in zip(
                batch_prompt_points, batch_prompt_object_ids, batch_prompt_counts
            ):
                prompt_views: List[torch.Tensor] = []
                for prompt_idx in range(num_slots):
                    tile = torch.zeros(3, tile_size, tile_size, device=self.device)
                    if sample_points is not None and prompt_idx < num_valid:
                        point = sample_points[prompt_idx].float()
                        point_xyz = torch.zeros(
                            3, dtype=torch.float32, device=self.device
                        )
                        coord_dims = min(3, point.shape[-1])
                        if coord_dims > 0:
                            point_xyz[:coord_dims] = point[:coord_dims]
                        coord_values = (point_xyz / float(latent_res)).clamp_(0, 1)
                        header_height = max(4, tile_size // 8)
                        object_id = -1
                        if (
                            sample_object_ids is not None
                            and prompt_idx < sample_object_ids.shape[0]
                        ):
                            object_id = int(sample_object_ids[prompt_idx].item())
                        if object_id >= 0:
                            tile[:, :header_height, :] = part_colors[
                                object_id % len(part_colors)
                            ].view(3, 1, 1)
                        body_height = max(1, tile_size - header_height)
                        band_height = max(1, body_height // 3)
                        for axis_idx in range(3):
                            row_start = header_height + axis_idx * band_height
                            row_end = (
                                header_height + (axis_idx + 1) * band_height
                                if axis_idx < 2
                                else tile_size
                            )
                            tile[:, row_start:row_end, :] = (
                                axis_palette[axis_idx] * coord_values[axis_idx]
                            ).view(3, 1, 1)
                    prompt_views.append(tile)
                while len(prompt_views) < grid_nrows * grid_ncols:
                    prompt_views.append(
                        torch.zeros(3, tile_size, tile_size, device=self.device)
                    )
                grid_rows = [
                    torch.cat(
                        prompt_views[r * grid_ncols : (r + 1) * grid_ncols], dim=-1
                    )
                    for r in range(grid_nrows)
                ]
                strips.append(torch.cat(grid_rows, dim=-2))
            return torch.stack(strips, dim=0)

        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=(
                self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None
            ),
        )
        data_iter = iter(dataloader)
        all_pred_parts: List[List[Mesh]] = []
        all_gt_parts: List[List[Mesh]] = []
        all_pred_color_indices: List[List[int]] = []
        all_prompt_points: List[Optional[torch.Tensor]] = []
        all_prompt_point_object_ids: List[Optional[torch.Tensor]] = []
        all_prompt_counts: List[int] = []
        all_case_ids: List[str] = []
        if "encoder" in self.models:
            self.models["encoder"].eval()
        decoder.eval()
        sample_iter = range(0, num_samples, batch_size)
        if self.is_master:
            sample_iter = tqdm(
                sample_iter, desc="Seg-only snapshot", unit="batch", leave=False
            )
        for i in sample_iter:
            batch = min(batch_size, num_samples - i)
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data = next(data_iter)
            batch_case_ids = data.get("sha256")
            if batch_case_ids is not None:
                if isinstance(batch_case_ids, str):
                    batch_case_ids = [batch_case_ids]
                ids = list(batch_case_ids)[:batch]
                batch_src = data.get("dataset_key")
                if isinstance(batch_src, str):
                    batch_src = [batch_src]
                srcs = (
                    list(batch_src)[:batch]
                    if batch_src is not None
                    else [""] * len(ids)
                )
                for s, src in zip(ids, srcs):
                    all_case_ids.append(f"{src}/{s}" if src else str(s))
            args = {
                k: v[:batch] if hasattr(v, "__getitem__") else v
                for k, v in data.items()
            }
            args = recursive_to_device(args, self.device)
            instance_id = args.get("instance_id")
            if instance_id is not None:
                if isinstance(instance_id, list):
                    instance_id = torch.cat(
                        [x.to(self.device) for x in instance_id], dim=0
                    )
                elif isinstance(instance_id, torch.Tensor) and instance_id.dim() == 2:
                    instance_id = instance_id.flatten().to(self.device)
                elif isinstance(instance_id, torch.Tensor):
                    instance_id = instance_id.to(self.device)
            max_objects = decoder.max_parts if instance_id is not None else 0
            vertices_merged, intersected_merged, multihot_gt_512 = (
                merge_overlapping_voxels(
                    args["vertices"],
                    args["intersected"],
                    instance_id=instance_id,
                    max_objects=max_objects,
                )
            )
            precomputed_z = args.get("precomputed_z")
            if precomputed_z is not None:
                z = _restore_sparse_scale(
                    precomputed_z, precomputed_z.device, Fraction(DOWNSAMPLE_RATIO, 1)
                )
            else:
                z = self.models["encoder"](vertices_merged, intersected_merged)
            img_features = None
            query_source = getattr(decoder, "query_source", "fixed")
            prompt_inputs, prompt_valid_mask, prompt_object_ids, _ = (
                self._prepare_prompt_inputs(query_source, args, decoder.max_parts)
            )
            if query_source == "prompt_point":
                for sample_idx in range(batch):
                    num_valid = 0
                    if prompt_valid_mask is not None:
                        num_valid = int(prompt_valid_mask[sample_idx].sum().item())
                    sample_points = (
                        None
                        if prompt_inputs is None
                        else prompt_inputs[sample_idx].detach()
                    )
                    sample_object_ids = (
                        None
                        if prompt_object_ids is None
                        else prompt_object_ids[sample_idx].detach()
                    )
                    all_prompt_points.append(sample_points)
                    all_prompt_point_object_ids.append(sample_object_ids)
                    all_prompt_counts.append(num_valid)
            if "cond_image" in args and self.image_cond_model_config is not None:
                img_features = self._extract_image_features(args["cond_image"])
            original_dtype = z.feats.dtype
            z_cola3d = z.replace(z.feats.type(decoder.cola3d_dtype))
            ext_features = None
            if decoder.input_mode == "latent_utonia_1024":
                precomputed_utonia = args.get("precomputed_utonia")
                if precomputed_utonia is not None:
                    pu = _restore_sparse_scale(
                        precomputed_utonia, z.device, Fraction(DOWNSAMPLE_RATIO, 1)
                    )
                    pu_feats = pu.feats if isinstance(pu, sp.SparseTensor) else pu
                    ext_features = z.replace(pu_feats.to(decoder.cola3d_dtype))
            elif decoder.input_mode in ("utonia", "latent_utonia"):
                utonia_feats = decoder.utonia_extractor(
                    coords=z.coords, batch_size=z.shape[0]
                )
                ext_features = z.replace(utonia_feats.to(decoder.cola3d_dtype))
            prefused_input = None
            prompt_latent_source = z_cola3d
            if (
                query_source == "prompt_point"
                and getattr(decoder, "point_feature_source", "latent")
                == "latent_utonia"
            ):
                prefused_input = decoder.cola3d.compute_input_features(
                    z_cola3d, ext_features
                )
                prompt_latent_source = prefused_input
            prompt_queries, prompt_valid, prompt_q_coords = (
                decoder._encode_prompt_queries(
                    latent=prompt_latent_source,
                    prompt_masks=(
                        prompt_inputs if query_source == "prompt_mask" else None
                    ),
                    prompt_points=(
                        prompt_inputs if query_source == "prompt_point" else None
                    ),
                    prompt_valid_mask=prompt_valid_mask,
                )
            )
            enriched_latent, query_feats = decoder.cola3d(
                z_cola3d,
                img_features=img_features,
                ext_features=ext_features,
                prompt_queries=prompt_queries,
                valid_query_mask=prompt_valid,
                prompt_q_coords=prompt_q_coords,
                prefused_input=prefused_input,
            )
            mask_logits = decoder.mask_decoder(
                enriched_latent, query_feats, valid_query_mask=prompt_valid
            )
            enriched_latent = enriched_latent.replace(
                enriched_latent.feats.type(original_dtype)
            )
            mesh_decode_latent = decoder._select_mesh_decode_latent(z, enriched_latent)
            num_objects = args.get("num_objects")
            has_gt = multihot_gt_512 is not None and num_objects is not None
            gt_masks = None
            num_objects_tensor = None
            matching = None
            if has_gt:
                gt_masks = self._downsample_multihot_gt(
                    multihot_gt_512,
                    vertices_merged,
                    z,
                    downsample_ratio=DOWNSAMPLE_RATIO,
                    max_parts=decoder.max_parts,
                )
                if isinstance(num_objects, list):
                    num_objects_tensor = torch.tensor(num_objects, device=self.device)
                elif isinstance(num_objects, torch.Tensor):
                    num_objects_tensor = num_objects.to(self.device)
                else:
                    num_objects_tensor = torch.tensor([num_objects], device=self.device)
                valid_mask = torch.zeros(
                    mask_logits.shape[0],
                    mask_logits.shape[1],
                    dtype=torch.bool,
                    device=self.device,
                )
                for b in range(batch):
                    n_points = z.layout[b].stop - z.layout[b].start
                    valid_mask[b, :n_points] = True
                direct_matching = None
                if self._is_prompt_query_source(query_source):
                    direct_matching = self._sanitize_prompt_matching(
                        prompt_object_ids,
                        num_objects_tensor,
                        mask_logits.shape[-1],
                        valid_query_mask=prompt_valid_mask,
                    )
                if direct_matching is not None:
                    matching = direct_matching
                else:
                    matching = self.matcher(
                        mask_logits,
                        gt_masks,
                        num_objects_tensor,
                        valid_mask,
                        valid_queries=prompt_valid_mask,
                    )
            for b in range(batch):
                n_points = z.layout[b].stop - z.layout[b].start
                probs = mask_logits[b, :n_points].sigmoid()
                sample_feats = mesh_decode_latent.feats[mesh_decode_latent.layout[b]]
                sample_coords = mesh_decode_latent.coords[
                    mesh_decode_latent.layout[b]
                ].clone()
                sample_coords[:, 0] = 0
                sample_latent = sp.SparseTensor(sample_feats, sample_coords)
                sample_latent._scale = mesh_decode_latent._scale
                pred_parts: List[Mesh] = []
                color_indices: List[int] = []
                num_pred_queries = mask_logits.shape[-1]
                for part_idx in range(num_pred_queries):
                    part_mask = probs[:, part_idx] > 0.5
                    pred_parts.append(
                        _decode_part_from_latent(sample_latent, part_mask)
                    )
                    if matching is not None:
                        matched_idx = int(matching[b, part_idx].item())
                        color_indices.append(
                            matched_idx if matched_idx >= 0 else part_idx
                        )
                    else:
                        color_indices.append(part_idx)
                all_pred_parts.append(pred_parts)
                all_pred_color_indices.append(color_indices)
                gt_parts: List[Mesh] = []
                if has_gt and gt_masks is not None and (num_objects_tensor is not None):
                    num_obj = int(num_objects_tensor[b].item())
                    for part_idx in range(num_pred_queries):
                        if part_idx >= num_obj:
                            gt_parts.append(_make_empty_mesh())
                            continue
                        gt_part_mask = gt_masks[b, :n_points, part_idx].bool()
                        gt_parts.append(
                            _decode_part_from_latent(sample_latent, gt_part_mask)
                        )
                else:
                    for _ in range(num_pred_queries):
                        gt_parts.append(_make_empty_mesh())
                all_gt_parts.append(gt_parts)
        if "encoder" in self.models:
            self.models["encoder"].train()
        decoder.train()

        def _sanitize_mesh_for_render(mesh: Mesh) -> Mesh:
            if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
                return mesh
            if not torch.isfinite(mesh.vertices).all():
                return _make_empty_mesh()
            if mesh.faces.numel() > 0:
                max_idx = int(mesh.faces.max().item())
                if (
                    max_idx >= mesh.vertices.shape[0]
                    or int(mesh.faces.min().item()) < 0
                ):
                    return _make_empty_mesh()
            return mesh

        all_pred_parts = [
            [_sanitize_mesh_for_render(m) for m in parts] for parts in all_pred_parts
        ]
        all_gt_parts = [
            [_sanitize_mesh_for_render(m) for m in parts] for parts in all_gt_parts
        ]
        pred_seg_meshes = _merge_colored_parts(
            all_pred_parts, color_indices=all_pred_color_indices
        )
        gt_seg_meshes = _merge_colored_parts(all_gt_parts)
        pred_seg_meshes = [_sanitize_mesh_for_render(m) for m in pred_seg_meshes]
        gt_seg_meshes = [_sanitize_mesh_for_render(m) for m in gt_seg_meshes]
        cameras = self._randomize_camera(len(pred_seg_meshes))
        render_size = getattr(self, "render_resolution", 512)
        black_placeholder = torch.zeros(
            len(pred_seg_meshes), 3, render_size, render_size, device=self.device
        )
        sample_dict: Dict = {}
        try:
            seg_attr = self._render_batch(
                pred_seg_meshes, **cameras, return_types=["attr"]
            )
            seg_normal = self._render_batch(
                pred_seg_meshes, **cameras, return_types=["normal"]
            )
            sample_dict["seg_pred"] = {"value": seg_attr["attr"], "type": "image"}
            sample_dict["seg_normal"] = {"value": seg_normal["normal"], "type": "image"}
        except Exception as e:
            print(
                f"  [CoLa3D snapshot] Pred render failed: {e}. Using black placeholders."
            )
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            sample_dict["seg_pred"] = {
                "value": black_placeholder.clone(),
                "type": "image",
            }
            sample_dict["seg_normal"] = {
                "value": black_placeholder.clone(),
                "type": "image",
            }
        try:
            gt_seg_attr = self._render_batch(
                gt_seg_meshes, **cameras, return_types=["attr"]
            )
            gt_seg_normal = self._render_batch(
                gt_seg_meshes, **cameras, return_types=["normal"]
            )
            sample_dict["gt_seg_pred"] = {"value": gt_seg_attr["attr"], "type": "image"}
            sample_dict["gt_seg_normal"] = {
                "value": gt_seg_normal["normal"],
                "type": "image",
            }
        except Exception as e:
            print(
                f"  [CoLa3D snapshot] GT-latent render failed: {e}. Using black placeholders."
            )
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            sample_dict["gt_seg_pred"] = {
                "value": black_placeholder.clone(),
                "type": "image",
            }
            sample_dict["gt_seg_normal"] = {
                "value": black_placeholder.clone(),
                "type": "image",
            }
        sample_dict["parts_vis"] = {
            "value": _render_part_strips(
                all_pred_parts, sample_dict["seg_normal"]["value"]
            ),
            "type": "image",
        }
        sample_dict["gt_parts_vis"] = {
            "value": _render_part_strips(
                all_gt_parts, sample_dict["gt_seg_normal"]["value"]
            ),
            "type": "image",
        }
        if all_prompt_points:
            sample_dict["prompt_points_vis"] = {
                "value": _render_prompt_point_placeholders(
                    all_prompt_points,
                    all_prompt_point_object_ids,
                    all_prompt_counts,
                    num_slots=decoder.max_parts,
                ),
                "type": "image",
            }
        if all_case_ids:
            print(
                f"  [CoLa3D snapshot][rank{self.rank}] step {self.step} cases ({len(all_case_ids)}): {all_case_ids}"
            )
            try:
                import os

                man_dir = os.path.join(
                    self.output_dir, "samples", f"step{self.step:07d}"
                )
                os.makedirs(man_dir, exist_ok=True)
                with open(
                    os.path.join(man_dir, f"snapshot_cases_rank{self.rank}.txt"), "w"
                ) as f:
                    f.write("\n".join(all_case_ids) + "\n")
            except Exception as e:
                print(f"  [CoLa3D snapshot] failed to write case manifest: {e}")
        return sample_dict


DecomposableShapeVaeTrainer = CoLa3DTrainer
