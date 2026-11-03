from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin


class SparseFlowMatchingTrainer(FlowMatchingTrainer):


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

    def training_losses(
        self, x_0: sp.SparseTensor, cond=None, **kwargs
    ) -> Tuple[Dict, Dict]:

        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        pred = self.training_models["denoiser"](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]
        mse_per_instance = np.array(
            [
                F.mse_loss(
                    pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]
                ).item()
                for i in range(x_0.shape[0])
            ]
        )
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}
        return (terms, {})

    @torch.no_grad()
    def run_snapshot(
        self, num_samples: int, batch_size: int, verbose: bool = False
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=num_samples,
            shuffle=True,
            num_workers=0,
            collate_fn=(
                self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None
            ),
        )
        data = next(iter(dataloader))
        sampler = self.get_sampler()
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch_data = {k: v[i : i + batch_size] for k, v in data.items()}
            batch_data = recursive_to_device(batch_data, "cuda")
            noise = batch_data["x_0"].replace(torch.randn_like(batch_data["x_0"].feats))
            cond_vis.append(self.vis_cond(**batch_data))
            del batch_data["x_0"]
            args = self.get_inference_cond(**batch_data)
            res = sampler.sample(
                self.models["denoiser"],
                noise=noise,
                **args,
                steps=12,
                guidance_strength=3.0,
                verbose=verbose,
            )
            sample.append(res.samples)
        sample = sp.sparse_cat(sample)
        sample_gt = {k: v for k, v in data.items()}
        sample = {k: v if k != "x_0" else sample for k, v in data.items()}
        sample_dict = {
            "sample_gt": {"value": sample_gt, "type": "sample"},
            "sample": {"value": sample, "type": "sample"},
        }
        sample_dict.update(
            dict_reduce(
                cond_vis,
                None,
                {"value": lambda x: torch.cat(x, dim=0), "type": lambda x: x[0]},
            )
        )
        return sample_dict


class SparseFlowMatchingCFGTrainer(
    ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer
):


    pass


class ImageConditionedSparseFlowMatchingCFGTrainer(
    ImageConditionedMixin, SparseFlowMatchingCFGTrainer
):


    pass
