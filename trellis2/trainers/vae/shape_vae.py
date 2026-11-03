from typing import *
import os
import copy
import functools
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils3d
from easydict import EasyDict as edict
from ..basic import BasicTrainer
from ...modules import sparse as sp
from ...renderers import MeshRenderer
from ...representations import Mesh
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from ...utils.loss_utils import l1_loss, ssim, lpips


class ShapeVaeTrainer(BasicTrainer):


    def __init__(
        self,
        *args,
        lambda_subdiv: float = 0.1,
        lambda_intersected: float = 0.1,
        lambda_vertice: float = 0.01,
        lambda_mask: float = 1,
        lambda_depth: float = 10,
        lambda_normal: float = 1,
        lambda_kl: float = 1e-06,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        render_resolution: float = 1024,
        camera_randomization_config: dict = {"radius_range": [2, 100]},
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_subdiv = lambda_subdiv
        self.lambda_intersected = lambda_intersected
        self.lambda_mask = lambda_mask
        self.lambda_vertice = lambda_vertice
        self.lambda_depth = lambda_depth
        self.lambda_normal = lambda_normal
        self.lambda_kl = lambda_kl
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.camera_randomization_config = camera_randomization_config
        self.renderer = MeshRenderer(
            {"near": 1, "far": 3, "resolution": render_resolution}, device=self.device
        )

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
            collate_fn=functools.partial(
                self.dataset.collate_fn, split_size=self.batch_split
            ),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def _randomize_camera(self, num_samples: int):
        r_min, r_max = self.camera_randomization_config["radius_range"]
        k_min = 1 / r_max**2
        k_max = 1 / r_min**2
        ks = torch.rand(num_samples, device=self.device) * (k_max - k_min) + k_min
        radius = 1 / torch.sqrt(ks)
        fov = 2 * torch.arcsin(0.5 / radius)
        origin = radius.unsqueeze(-1) * F.normalize(
            torch.randn(num_samples, 3, device=self.device), dim=-1
        )
        extrinsics = utils3d.torch.extrinsics_look_at(
            origin,
            torch.zeros_like(origin),
            torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device),
        )
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        near = [np.random.uniform(r - 1, r) for r in radius.tolist()]
        return {"extrinsics": extrinsics, "intrinsics": intrinsics, "near": near}

    def _render_batch(
        self,
        reps: List[Mesh],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        near: List,
        return_types=["mask", "normal", "depth"],
    ) -> Dict[str, torch.Tensor]:

        ret = {k: [] for k in return_types}
        for i, rep in enumerate(reps):
            self.renderer.rendering_options["near"] = near[i]
            self.renderer.rendering_options["far"] = near[i] + 2
            out_dict = self.renderer.render(
                rep, extrinsics[i], intrinsics[i], return_types=return_types
            )
            for k in out_dict:
                ret[k].append(
                    out_dict[k][None] if k in ["mask", "depth"] else out_dict[k]
                )
        for k in ret:
            ret[k] = torch.stack(ret[k])
        return ret

    def training_losses(
        self, vertices: sp.SparseTensor, intersected: sp.SparseTensor, mesh: List[Mesh]
    ) -> Tuple[Dict, Dict]:

        z, mean, logvar = self.training_models["encoder"](
            vertices, intersected, sample_posterior=True, return_raw=True
        )
        recon, pred_vertice, pred_intersected, subs_gt, subs = self.training_models[
            "decoder"
        ](z, intersected)
        terms = edict(loss=0.0)
        if self.lambda_intersected > 0:
            terms["direct/intersected"] = F.binary_cross_entropy_with_logits(
                pred_intersected.feats.flatten(), intersected.feats.flatten().float()
            )
            terms["loss"] = (
                terms["loss"] + self.lambda_intersected * terms["direct/intersected"]
            )
        if self.lambda_vertice > 0:
            terms["direct/vertice"] = F.mse_loss(pred_vertice.feats, vertices.feats)
            terms["loss"] = (
                terms["loss"] + self.lambda_vertice * terms["direct/vertice"]
            )
        for i, (sub_gt, sub) in enumerate(zip(subs_gt, subs)):
            terms[f"bce_sub{i}"] = F.binary_cross_entropy_with_logits(
                sub.feats, sub_gt.float()
            )
            terms["loss"] = terms["loss"] + self.lambda_subdiv * terms[f"bce_sub{i}"]
        cameras = self._randomize_camera(len(mesh))
        gt_renders = self._render_batch(
            mesh, **cameras, return_types=["mask", "normal", "depth"]
        )
        pred_renders = self._render_batch(
            recon, **cameras, return_types=["mask", "normal", "depth"]
        )
        terms["render/mask"] = l1_loss(pred_renders["mask"], gt_renders["mask"])
        terms["render/depth"] = l1_loss(pred_renders["depth"], gt_renders["depth"])
        terms["render/normal/l1"] = l1_loss(
            pred_renders["normal"], gt_renders["normal"]
        )
        terms["render/normal/ssim"] = 1 - ssim(
            pred_renders["normal"], gt_renders["normal"]
        )
        terms["render/normal/lpips"] = lpips(
            pred_renders["normal"], gt_renders["normal"]
        )
        terms["loss"] = (
            terms["loss"]
            + self.lambda_mask * terms["render/mask"]
            + self.lambda_depth * terms["render/depth"]
            + self.lambda_normal
            * (
                terms["render/normal/l1"]
                + self.lambda_ssim * terms["render/normal/ssim"]
                + self.lambda_lpips * terms["render/normal/lpips"]
            )
        )
        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
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
        gts = []
        recons = []
        recons2 = []
        self.models["encoder"].eval()
        from tqdm import tqdm

        sample_iter = range(0, num_samples, batch_size)
        if self.is_master:
            sample_iter = tqdm(
                sample_iter, desc="Snapshot sampling", unit="batch", leave=False
            )
        for i in sample_iter:
            batch = min(batch_size, num_samples - i)
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data = next(data_iter)
            args = {k: v[:batch] for k, v in data.items()}
            args = recursive_to_device(args, self.device)
            z = self.models["encoder"](args["vertices"], args["intersected"])
            self.models["decoder"].train()
            y = self.models["decoder"](z, args["intersected"])[0]
            z.clear_spatial_cache()
            self.models["decoder"].eval()
            y2 = self.models["decoder"](z)
            gts.extend(args["mesh"])
            recons.extend(y)
            recons2.extend(y2)
        self.models["encoder"].train()
        self.models["decoder"].train()
        cameras = self._randomize_camera(num_samples)
        gt_renders = self._render_batch(gts, **cameras, return_types=["normal"])
        recons_renders = self._render_batch(recons, **cameras, return_types=["normal"])
        recons2_renders = self._render_batch(
            recons2, **cameras, return_types=["normal"]
        )
        sample_dict = {
            "gt": {"value": gt_renders["normal"], "type": "image"},
            "with_GT_sub": {"value": recons_renders["normal"], "type": "image"},
            "no_sub": {"value": recons2_renders["normal"], "type": "image"},
        }
        return sample_dict
