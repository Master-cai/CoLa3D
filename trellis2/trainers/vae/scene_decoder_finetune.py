from typing import Dict, List, Optional, Tuple
import copy
import functools
import numpy as np
import torch
import torch.nn.functional as F
import utils3d
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from .shape_vae import ShapeVaeTrainer
from ...modules import sparse as sp
from ...models.sc_vaes.fdg_vae import merge_overlapping_voxels
from ...representations import Mesh
from ...utils.data_utils import BalancedResumableSampler, cycle, recursive_to_device
from ...utils.loss_utils import l1_loss, lpips, ssim

__all__ = ["SceneDecoderFinetuneTrainer"]
DOWNSAMPLE_RATIO = 16


class SceneDecoderFinetuneTrainer(ShapeVaeTrainer):


    def __init__(self, *args, max_parts: int = 8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_parts = max_parts
        self.snapshot_master_only = True
        self.snapshot_use_distributed_collectives = False

    def prepare_dataloader(self, **kwargs) -> None:

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

    def _randomize_camera_at(self, center: torch.Tensor) -> Dict[str, torch.Tensor]:

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

    def _merge_part_meshes(self, part_meshes: List[Mesh]) -> Optional[Mesh]:

        all_vertices: List[torch.Tensor] = []
        all_faces: List[torch.Tensor] = []
        vertex_offset = 0
        for part_mesh in part_meshes:
            if part_mesh.vertices.shape[0] == 0 or part_mesh.faces.shape[0] == 0:
                continue
            all_vertices.append(part_mesh.vertices)
            all_faces.append(part_mesh.faces + vertex_offset)
            vertex_offset += part_mesh.vertices.shape[0]
        if len(all_vertices) == 0:
            return None
        return Mesh(
            vertices=torch.cat(all_vertices, dim=0), faces=torch.cat(all_faces, dim=0)
        )

    def _flatten_instance_id(
        self, instance_id: Optional[torch.Tensor], device: torch.device
    ) -> Optional[torch.Tensor]:

        if instance_id is None:
            return None
        if isinstance(instance_id, list):
            return torch.cat([x.to(device) for x in instance_id], dim=0)
        if instance_id.dim() == 2:
            return instance_id.flatten().to(device)
        return instance_id.to(device)

    def _num_objects_tensor(
        self, num_objects: torch.Tensor, device: torch.device
    ) -> torch.Tensor:

        if isinstance(num_objects, list):
            tensor = torch.tensor(num_objects, device=device)
        elif isinstance(num_objects, torch.Tensor):
            tensor = num_objects.to(device)
        else:
            tensor = torch.tensor([num_objects], device=device)
        if tensor.max().item() > self.max_parts:
            raise ValueError(
                f"num_objects exceeds max_parts: max(num_objects)={int(tensor.max().item())}, max_parts={self.max_parts}"
            )
        return tensor.long()

    def _require_supervision_inputs(
        self,
        *,
        instance_id: Optional[torch.Tensor],
        num_objects: Optional[torch.Tensor],
        per_obj_vertices: Optional[List[List[torch.Tensor]]],
        per_obj_intersected: Optional[List[List[torch.Tensor]]],
        per_obj_meshes: Optional[List[List[Mesh]]],
        per_obj_coords: Optional[List[List[torch.Tensor]]],
    ) -> None:

        required = {
            "instance_id": instance_id,
            "num_objects": num_objects,
            "per_obj_vertices": per_obj_vertices,
            "per_obj_intersected": per_obj_intersected,
            "per_obj_meshes": per_obj_meshes,
            "per_obj_coords": per_obj_coords,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"SceneDecoderFinetuneTrainer requires full multi-object supervision inputs. Missing: {missing}"
            )
        dataset_max_objects = getattr(self.dataset, "max_objects", None)
        if dataset_max_objects is not None and dataset_max_objects != self.max_parts:
            raise ValueError(
                f"trainer.max_parts={self.max_parts} must match dataset.max_objects={dataset_max_objects}"
            )

    def _downsample_multihot_gt(
        self,
        multihot_gt: torch.Tensor,
        vertices: sp.SparseTensor,
        z: sp.SparseTensor,
        downsample_ratio: int = DOWNSAMPLE_RATIO,
        max_parts: Optional[int] = None,
    ) -> torch.Tensor:

        max_parts = self.max_parts if max_parts is None else max_parts
        device = z.device
        latent_res = self.models["decoder"].resolution // DOWNSAMPLE_RATIO
        batch_size = z.shape[0]
        max_points = max(
            (z.layout[b].stop - z.layout[b].start for b in range(batch_size))
        )
        gt_masks = torch.zeros(
            batch_size, max_points, max_parts, dtype=torch.float32, device=device
        )
        for batch_idx in range(batch_size):
            v_start = vertices.layout[batch_idx].start
            v_end = vertices.layout[batch_idx].stop
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
            ds_coords = (batch_orig_coords // downsample_ratio).clamp(0, latent_res - 1)
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
            z_start = z.layout[batch_idx].start
            z_end = z.layout[batch_idx].stop
            n_points = z_end - z_start
            batch_z_coords = (
                z.coords[z_start:z_end, 1:4].long().clamp(0, latent_res - 1)
            )
            gt_masks[batch_idx, :n_points] = grid_flat.view(
                latent_res, latent_res, latent_res, max_parts
            )[batch_z_coords[:, 0], batch_z_coords[:, 1], batch_z_coords[:, 2]]
        return gt_masks

    def _extract_part_latent(
        self, z: sp.SparseTensor, batch_idx: int, part_mask: torch.Tensor
    ) -> sp.SparseTensor:

        start_idx = z.layout[batch_idx].start
        end_idx = z.layout[batch_idx].stop
        part_feats = z.feats[start_idx:end_idx][part_mask]
        part_coords = z.coords[start_idx:end_idx][part_mask].clone()
        part_coords[:, 0] = 0
        part_latent = sp.SparseTensor(part_feats, part_coords)
        part_latent._scale = z._scale
        return part_latent

    def _attach_gt_cache(
        self, part_latent: sp.SparseTensor, obj_coords: torch.Tensor, z: sp.SparseTensor
    ) -> sp.SparseTensor:

        scene_shape = tuple(
            (
                int(spatial_dim * int(z._scale[dim_idx]))
                for dim_idx, spatial_dim in enumerate(z.spatial_shape)
            )
        )
        gt_cache, expected_latent_coords = sp.build_subdivision_cache_from_gt(
            obj_coords, global_shape=scene_shape, return_latent_coords=True
        )
        part_latent = sp.reorder_latent_to_match_cache(
            part_latent, expected_latent_coords
        )
        part_latent._spatial_cache = gt_cache
        return part_latent

    def _build_gt_intersected_sparse(
        self, part_gt_inter: torch.Tensor, obj_coords: torch.Tensor
    ) -> sp.SparseTensor:

        if isinstance(part_gt_inter, sp.SparseTensor):
            return part_gt_inter
        if obj_coords.shape[1] == 3:
            batch_idx = torch.zeros(
                (obj_coords.shape[0], 1),
                device=obj_coords.device,
                dtype=obj_coords.dtype,
            )
            obj_coords_4d = torch.cat([batch_idx, obj_coords], dim=1)
        else:
            obj_coords_4d = obj_coords.clone()
            obj_coords_4d[:, 0] = 0
        return sp.SparseTensor(part_gt_inter, obj_coords_4d.int())

    def _render_mesh_triplet(
        self,
        gt_mesh: Mesh,
        pred_with_gt_sub: Optional[Mesh],
        pred_no_sub: Optional[Mesh],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        resolution = int(self.renderer.rendering_options["resolution"])
        height = resolution
        width = resolution
        black = torch.zeros(3, height, width, device=self.device)
        center = gt_mesh.vertices.detach().mean(dim=0)
        camera = self._randomize_camera_at(center)
        try:
            gt_render = self._render_batch(
                [gt_mesh], **camera, return_types=["normal"]
            )["normal"][0]
        except Exception:
            gt_render = black

        def render_pred(mesh: Optional[Mesh]) -> torch.Tensor:
            if mesh is None or mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
                return black
            try:
                return self._render_batch([mesh], **camera, return_types=["normal"])[
                    "normal"
                ][0]
            except Exception:
                return black

        return (gt_render, render_pred(pred_with_gt_sub), render_pred(pred_no_sub))

    def _render_normals_safe(
        self,
        meshes: List[Mesh],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        near: List[float],
    ) -> torch.Tensor:

        resolution = int(self.renderer.rendering_options["resolution"])
        black = torch.zeros(3, resolution, resolution, device=self.device)
        renders: List[torch.Tensor] = []
        for mesh_idx, mesh in enumerate(meshes):
            if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
                renders.append(black)
                continue
            try:
                self.renderer.rendering_options["near"] = near[mesh_idx]
                self.renderer.rendering_options["far"] = near[mesh_idx] + 2
                render_out = self.renderer.render(
                    mesh,
                    extrinsics[mesh_idx],
                    intrinsics[mesh_idx],
                    return_types=["normal"],
                )
                renders.append(render_out["normal"])
            except Exception:
                try:
                    torch.cuda.synchronize()
                except RuntimeError:
                    pass
                renders.append(black)
        return torch.stack(renders, dim=0)

    def _render_part_pair_losses(
        self, matched_pred_meshes: List[Mesh], matched_gt_meshes: List[Mesh]
    ) -> Dict[str, torch.Tensor]:

        mask_losses: List[torch.Tensor] = []
        depth_losses: List[torch.Tensor] = []
        normal_l1_losses: List[torch.Tensor] = []
        normal_ssim_losses: List[torch.Tensor] = []
        normal_lpips_losses: List[torch.Tensor] = []
        for pred_mesh, gt_mesh in zip(matched_pred_meshes, matched_gt_meshes):
            center = gt_mesh.vertices.detach().mean(dim=0)
            camera = self._randomize_camera_at(center)
            pred_render = self._render_batch(
                [pred_mesh], **camera, return_types=["mask", "normal", "depth"]
            )
            gt_render = self._render_batch(
                [gt_mesh], **camera, return_types=["mask", "normal", "depth"]
            )
            mask_losses.append(l1_loss(pred_render["mask"], gt_render["mask"]))
            depth_losses.append(l1_loss(pred_render["depth"], gt_render["depth"]))
            normal_l1_losses.append(l1_loss(pred_render["normal"], gt_render["normal"]))
            normal_ssim_losses.append(
                1 - ssim(pred_render["normal"], gt_render["normal"])
            )
            normal_lpips_losses.append(
                lpips(pred_render["normal"], gt_render["normal"])
            )
        return {
            "mask": torch.stack(mask_losses).mean(),
            "depth": torch.stack(depth_losses).mean(),
            "normal/l1": torch.stack(normal_l1_losses).mean(),
            "normal/ssim": torch.stack(normal_ssim_losses).mean(),
            "normal/lpips": torch.stack(normal_lpips_losses).mean(),
        }

    def _get_latent(
        self,
        vertices: sp.SparseTensor,
        intersected: sp.SparseTensor,
        precomputed_z: Optional[sp.SparseTensor],
    ) -> Tuple[sp.SparseTensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        if precomputed_z is not None:
            from fractions import Fraction

            expected_scale = tuple((Fraction(DOWNSAMPLE_RATIO, 1) for _ in range(3)))
            if precomputed_z._scale != expected_scale:
                precomputed_z._scale = expected_scale
            return (precomputed_z, None, None)
        if "encoder" in self.training_models:
            z, mean, logvar = self.training_models["encoder"](
                vertices, intersected, sample_posterior=True, return_raw=True
            )
            return (z, mean, logvar)
        raise RuntimeError(
            "No precomputed_z in batch and no encoder in models. Set shape_latent_root in dataset config or add encoder to model config."
        )

    def training_losses(
        self,
        vertices: sp.SparseTensor,
        intersected: sp.SparseTensor,
        mesh: Optional[List[Mesh]] = None,
        instance_id: Optional[torch.Tensor] = None,
        num_objects: Optional[torch.Tensor] = None,
        per_obj_vertices: Optional[List[List[torch.Tensor]]] = None,
        per_obj_intersected: Optional[List[List[torch.Tensor]]] = None,
        per_obj_meshes: Optional[List[List[Mesh]]] = None,
        per_obj_coords: Optional[List[List[torch.Tensor]]] = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:

        self._require_supervision_inputs(
            instance_id=instance_id,
            num_objects=num_objects,
            per_obj_vertices=per_obj_vertices,
            per_obj_intersected=per_obj_intersected,
            per_obj_meshes=per_obj_meshes,
            per_obj_coords=per_obj_coords,
        )
        instance_id_flat = self._flatten_instance_id(instance_id, vertices.device)
        num_objects_tensor = self._num_objects_tensor(num_objects, vertices.device)
        vertices, intersected, multihot_gt_512 = merge_overlapping_voxels(
            vertices, intersected, instance_id_flat, max_objects=self.max_parts
        )
        z, mean, logvar = self._get_latent(
            vertices, intersected, kwargs.get("precomputed_z")
        )
        gt_masks = self._downsample_multihot_gt(
            multihot_gt_512, vertices, z, max_parts=self.max_parts
        )
        terms = edict(loss=torch.tensor(0.0, device=self.device))
        matched_pred_meshes: List[Mesh] = []
        matched_gt_meshes: List[Mesh] = []
        per_part_vertices_losses: List[torch.Tensor] = []
        per_part_intersected_losses: List[torch.Tensor] = []
        per_part_sub_losses: Dict[int, List[torch.Tensor]] = {}
        batch_size = z.shape[0]
        for batch_idx in range(batch_size):
            n_obj = int(num_objects_tensor[batch_idx].item())
            start_idx = z.layout[batch_idx].start
            end_idx = z.layout[batch_idx].stop
            n_points = end_idx - start_idx
            for obj_id in range(n_obj):
                part_mask = gt_masks[batch_idx, :n_points, obj_id].bool()
                if int(part_mask.sum().item()) == 0:
                    continue
                part_latent = self._extract_part_latent(z, batch_idx, part_mask)
                part_latent = self._attach_gt_cache(
                    part_latent, per_obj_coords[batch_idx][obj_id], z
                )
                part_gt_inter_sp = self._build_gt_intersected_sparse(
                    per_obj_intersected[batch_idx][obj_id],
                    per_obj_coords[batch_idx][obj_id],
                )
                part_meshes, pred_verts, pred_inter, part_subs_gt, part_subs = (
                    self.training_models["decoder"](part_latent, part_gt_inter_sp)
                )
                gt_verts = per_obj_vertices[batch_idx][obj_id].to(pred_verts.device)
                gt_inter = (
                    per_obj_intersected[batch_idx][obj_id].to(pred_inter.device).float()
                )
                if pred_verts.feats.shape[0] != gt_verts.shape[0]:
                    raise AssertionError(
                        f"Per-part vertex count mismatch for batch={batch_idx}, obj={obj_id}: pred={pred_verts.feats.shape[0]}, gt={gt_verts.shape[0]}"
                    )
                if pred_inter.feats.shape[0] != gt_inter.shape[0]:
                    raise AssertionError(
                        f"Per-part intersected count mismatch for batch={batch_idx}, obj={obj_id}: pred={pred_inter.feats.shape[0]}, gt={gt_inter.shape[0]}"
                    )
                per_part_vertices_losses.append(F.mse_loss(pred_verts.feats, gt_verts))
                per_part_intersected_losses.append(
                    F.binary_cross_entropy_with_logits(
                        pred_inter.feats.flatten(), gt_inter.flatten()
                    )
                )
                for layer_idx, (sub_gt, sub_pred) in enumerate(
                    zip(part_subs_gt, part_subs)
                ):
                    per_part_sub_losses.setdefault(layer_idx, []).append(
                        F.binary_cross_entropy_with_logits(
                            sub_pred.feats, sub_gt.float()
                        )
                    )
                pred_mesh = part_meshes[0]
                gt_mesh = (
                    per_obj_meshes[batch_idx][obj_id]
                    if obj_id < len(per_obj_meshes[batch_idx])
                    else None
                )
                if (
                    gt_mesh is not None
                    and pred_mesh.vertices.shape[0] > 0
                    and (pred_mesh.faces.shape[0] > 0)
                    and (gt_mesh.vertices.shape[0] > 0)
                    and (gt_mesh.faces.shape[0] > 0)
                ):
                    matched_pred_meshes.append(pred_mesh)
                    matched_gt_meshes.append(gt_mesh)
        if per_part_vertices_losses:
            terms["per_part/direct/vertices"] = torch.stack(
                per_part_vertices_losses
            ).mean()
            terms["loss"] = (
                terms["loss"] + self.lambda_vertice * terms["per_part/direct/vertices"]
            )
        if per_part_intersected_losses:
            terms["per_part/direct/intersected"] = torch.stack(
                per_part_intersected_losses
            ).mean()
            terms["loss"] = (
                terms["loss"]
                + self.lambda_intersected * terms["per_part/direct/intersected"]
            )
        for layer_idx, losses in per_part_sub_losses.items():
            terms[f"per_part/bce_sub{layer_idx}"] = torch.stack(losses).mean()
            terms["loss"] = (
                terms["loss"]
                + self.lambda_subdiv * terms[f"per_part/bce_sub{layer_idx}"]
            )
        if matched_pred_meshes:
            render_terms = self._render_part_pair_losses(
                matched_pred_meshes, matched_gt_meshes
            )
            terms["per_part/mask"] = render_terms["mask"]
            terms["per_part/depth"] = render_terms["depth"]
            terms["per_part/normal/l1"] = render_terms["normal/l1"]
            terms["per_part/normal/ssim"] = render_terms["normal/ssim"]
            terms["per_part/normal/lpips"] = render_terms["normal/lpips"]
            per_part_loss = (
                self.lambda_mask * terms["per_part/mask"]
                + self.lambda_depth * terms["per_part/depth"]
                + self.lambda_normal
                * (
                    terms["per_part/normal/l1"]
                    + self.lambda_ssim * terms["per_part/normal/ssim"]
                    + self.lambda_lpips * terms["per_part/normal/lpips"]
                )
            )
            terms["per_part/recon"] = per_part_loss
            terms["loss"] = terms["loss"] + per_part_loss
        if mean is not None and logvar is not None:
            logvar_clamped = logvar.clamp(-20, 10)
            terms["kl"] = 0.5 * torch.mean(
                mean.pow(2) + logvar_clamped.exp() - logvar_clamped - 1
            )
            terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
        terms["per_part/num_matched"] = torch.tensor(
            len(matched_pred_meshes), dtype=torch.float32, device=self.device
        )
        return (terms, {})

    @torch.no_grad()
    def run_snapshot(
        self, num_samples: int, batch_size: int, verbose: bool = False
    ) -> Dict:

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
        has_encoder = "encoder" in self.models
        gt_scene_meshes: List[Mesh] = []
        whole_scene_with_gt_sub: List[Mesh] = []
        whole_scene_no_sub: List[Mesh] = []
        combined_scene_with_gt_sub: List[Mesh] = []
        combined_scene_no_sub: List[Mesh] = []
        part_gt_meshes_all: List[List[Mesh]] = []
        part_pred_with_gt_sub_all: List[List[Optional[Mesh]]] = []
        part_pred_no_sub_all: List[List[Optional[Mesh]]] = []
        if has_encoder:
            self.models["encoder"].eval()
        from tqdm import tqdm

        sample_iter = range(0, num_samples, batch_size)
        if self.is_master:
            sample_iter = tqdm(
                sample_iter, desc="Snapshot sampling", unit="batch", leave=False
            )
        for sample_start in sample_iter:
            batch = min(batch_size, num_samples - sample_start)
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data = next(data_iter)
            args = {
                key: value[:batch] if hasattr(value, "__getitem__") else value
                for key, value in data.items()
            }
            args = recursive_to_device(args, self.device)
            self._require_supervision_inputs(
                instance_id=args.get("instance_id"),
                num_objects=args.get("num_objects"),
                per_obj_vertices=args.get("per_obj_vertices"),
                per_obj_intersected=args.get("per_obj_intersected"),
                per_obj_meshes=args.get("per_obj_meshes"),
                per_obj_coords=args.get("per_obj_coords"),
            )
            instance_id_flat = self._flatten_instance_id(
                args["instance_id"], args["vertices"].device
            )
            num_objects_tensor = self._num_objects_tensor(
                args["num_objects"], args["vertices"].device
            )
            merged_vertices, merged_intersected, multihot_gt_512 = (
                merge_overlapping_voxels(
                    args["vertices"],
                    args["intersected"],
                    instance_id_flat,
                    max_objects=self.max_parts,
                )
            )
            precomputed_z = args.get("precomputed_z")
            if precomputed_z is not None:
                from fractions import Fraction

                expected_scale = tuple(
                    (Fraction(DOWNSAMPLE_RATIO, 1) for _ in range(3))
                )
                if precomputed_z._scale != expected_scale:
                    precomputed_z._scale = expected_scale
                z = precomputed_z
            elif has_encoder:
                z = self.models["encoder"](merged_vertices, merged_intersected)
            else:
                raise RuntimeError("Snapshot requires precomputed_z or encoder model.")
            gt_masks = self._downsample_multihot_gt(
                multihot_gt_512, merged_vertices, z, max_parts=self.max_parts
            )
            empty_mesh = Mesh(
                vertices=torch.zeros(0, 3, device=self.device),
                faces=torch.zeros(0, 3, dtype=torch.long, device=self.device),
            )
            self.models["decoder"].train()
            try:
                scene_with_output = self.models["decoder"](z, merged_intersected)
                scene_with_meshes = (
                    scene_with_output[0]
                    if isinstance(scene_with_output, tuple)
                    else scene_with_output
                )
            except Exception:
                scene_with_meshes = [empty_mesh for _ in range(batch)]
            z.clear_spatial_cache()
            self.models["decoder"].eval()
            try:
                scene_no_meshes = self.models["decoder"](z)
            except Exception:
                try:
                    torch.cuda.synchronize()
                except RuntimeError:
                    pass
                scene_no_meshes = [empty_mesh for _ in range(batch)]
            for batch_idx in range(batch):
                gt_scene_meshes.append(args["mesh"][batch_idx])
                whole_scene_with_gt_sub.append(scene_with_meshes[batch_idx])
                whole_scene_no_sub.append(scene_no_meshes[batch_idx])
                n_obj = int(num_objects_tensor[batch_idx].item())
                start_idx = z.layout[batch_idx].start
                end_idx = z.layout[batch_idx].stop
                n_points = end_idx - start_idx
                gt_parts_for_sample = list(args["per_obj_meshes"][batch_idx][:n_obj])
                pred_parts_with_gt_sub: List[Optional[Mesh]] = []
                pred_parts_no_sub: List[Optional[Mesh]] = []
                for obj_id in range(n_obj):
                    part_mask = gt_masks[batch_idx, :n_points, obj_id].bool()
                    if int(part_mask.sum().item()) == 0:
                        pred_parts_with_gt_sub.append(None)
                        pred_parts_no_sub.append(None)
                        continue
                    part_latent = self._extract_part_latent(z, batch_idx, part_mask)
                    part_latent = self._attach_gt_cache(
                        part_latent, args["per_obj_coords"][batch_idx][obj_id], z
                    )
                    part_gt_inter_sp = self._build_gt_intersected_sparse(
                        args["per_obj_intersected"][batch_idx][obj_id],
                        args["per_obj_coords"][batch_idx][obj_id],
                    )
                    self.models["decoder"].train()
                    with_gt_mesh = self.models["decoder"](
                        part_latent, part_gt_inter_sp
                    )[0][0]
                    part_latent.clear_spatial_cache()
                    self.models["decoder"].eval()
                    try:
                        no_sub_mesh = self.models["decoder"](part_latent)[0]
                    except Exception:
                        try:
                            torch.cuda.synchronize()
                        except RuntimeError:
                            pass
                        no_sub_mesh = None
                    pred_parts_with_gt_sub.append(with_gt_mesh)
                    pred_parts_no_sub.append(no_sub_mesh)
                combined_with = self._merge_part_meshes(
                    [mesh for mesh in pred_parts_with_gt_sub if mesh is not None]
                )
                combined_no = self._merge_part_meshes(
                    [mesh for mesh in pred_parts_no_sub if mesh is not None]
                )
                if combined_with is None:
                    combined_with = empty_mesh
                if combined_no is None:
                    combined_no = empty_mesh
                combined_scene_with_gt_sub.append(combined_with)
                combined_scene_no_sub.append(combined_no)
                part_gt_meshes_all.append(gt_parts_for_sample)
                part_pred_with_gt_sub_all.append(pred_parts_with_gt_sub)
                part_pred_no_sub_all.append(pred_parts_no_sub)
            self.models["decoder"].train()
        if has_encoder:
            self.models["encoder"].train()
        self.models["decoder"].train()
        cameras = self._randomize_camera(num_samples)
        gt_normals = self._render_normals_safe(gt_scene_meshes, **cameras)
        whole_with_normals = self._render_normals_safe(
            whole_scene_with_gt_sub, **cameras
        )
        whole_no_normals = self._render_normals_safe(whole_scene_no_sub, **cameras)
        combined_with_normals = self._render_normals_safe(
            combined_scene_with_gt_sub, **cameras
        )
        combined_no_normals = self._render_normals_safe(
            combined_scene_no_sub, **cameras
        )
        max_num_parts = max((len(parts) for parts in part_gt_meshes_all), default=0)
        part_gt_strips: List[torch.Tensor] = []
        part_with_strips: List[torch.Tensor] = []
        part_no_strips: List[torch.Tensor] = []
        if max_num_parts > 0:
            render_height = gt_normals.shape[-2]
            render_width = gt_normals.shape[-1]
            black = torch.zeros(3, render_height, render_width, device=self.device)
            for gt_parts, pred_parts_with, pred_parts_no in zip(
                part_gt_meshes_all, part_pred_with_gt_sub_all, part_pred_no_sub_all
            ):
                gt_views: List[torch.Tensor] = []
                with_views: List[torch.Tensor] = []
                no_views: List[torch.Tensor] = []
                for part_idx in range(max_num_parts):
                    if part_idx < len(gt_parts):
                        gt_render, with_render, no_render = self._render_mesh_triplet(
                            gt_parts[part_idx],
                            pred_parts_with[part_idx],
                            pred_parts_no[part_idx],
                        )
                    else:
                        gt_render = black
                        with_render = black
                        no_render = black
                    gt_views.append(gt_render)
                    with_views.append(with_render)
                    no_views.append(no_render)
                part_gt_strips.append(torch.cat(gt_views, dim=-1))
                part_with_strips.append(torch.cat(with_views, dim=-1))
                part_no_strips.append(torch.cat(no_views, dim=-1))
        else:
            black = torch.zeros_like(gt_normals)
            part_gt_strips = [black[i] for i in range(black.shape[0])]
            part_with_strips = [black[i] for i in range(black.shape[0])]
            part_no_strips = [black[i] for i in range(black.shape[0])]
        return {
            "gt": {"value": gt_normals, "type": "image"},
            "whole_with_GT_sub": {"value": whole_with_normals, "type": "image"},
            "whole_no_sub": {"value": whole_no_normals, "type": "image"},
            "parts_gt": {"value": torch.stack(part_gt_strips, dim=0), "type": "image"},
            "parts_with_GT_sub": {
                "value": torch.stack(part_with_strips, dim=0),
                "type": "image",
            },
            "parts_no_sub": {
                "value": torch.stack(part_no_strips, dim=0),
                "type": "image",
            },
            "combined_with_GT_sub": {"value": combined_with_normals, "type": "image"},
            "combined_no_sub": {"value": combined_no_normals, "type": "image"},
        }
