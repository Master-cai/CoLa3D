from typing import List
from abc import abstractmethod
import os
import time
import json
import copy
import threading
from functools import partial
from contextlib import nullcontext
import torch
from safetensors.torch import load_file as load_safetensors
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tqdm import tqdm
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler
from ..utils.dist_utils import *
from ..utils import grad_clip_utils, elastic_utils


class BasicTrainer:


    def __init__(
        self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode=None,
        mix_precision_mode="inflat_all",
        mix_precision_dtype=None,
        fp16_scale_growth=0.001,
        parallel_mode="ddp",
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        snapshot_batch_size=4,
        snapshot_num_samples=16,
        dataloader_workers: int = None,
        freeze_modules: List[str] = None,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        **kwargs,
    ):
        assert (
            batch_size is not None or batch_size_per_gpu is not None
        ), "Either batch_size or batch_size_per_gpu must be specified."
        self.models = models
        self.dataset = dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        if fp16_mode is not None:
            mix_precision_mode = fp16_mode
            if mix_precision_dtype is None:
                mix_precision_dtype = "float16"
        if mix_precision_dtype is None:
            mix_precision_dtype = "float16"
        self.mix_precision_mode = mix_precision_mode
        self.mix_precision_dtype = str_to_dtype(mix_precision_dtype)
        self.fp16_scale_growth = fp16_scale_growth
        self.parallel_mode = parallel_mode
        self.finetune_ckpt = finetune_ckpt or {}
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        self.snapshot_batch_size = snapshot_batch_size
        self.snapshot_num_samples = snapshot_num_samples
        self.dataloader_workers = (
            dataloader_workers
            if dataloader_workers is not None
            else int(np.ceil(os.cpu_count() / torch.cuda.device_count()))
        )
        self.log = []
        self._logged_first_nonfinite_grad = False
        if self.prefetch_data:
            self._data_prefetched = None
        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck
        self.enable_ddp_check = os.environ.get("DDP_ENABLE_CHECK", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.is_master = self.rank == 0
        else:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True
        self.batch_size = (
            batch_size
            if batch_size_per_gpu is None
            else batch_size_per_gpu * self.world_size
        )
        self.batch_size_per_gpu = (
            batch_size_per_gpu
            if batch_size_per_gpu is not None
            else batch_size // self.world_size
        )
        assert (
            self.batch_size % self.world_size == 0
        ), "Batch size must be divisible by the number of GPUs."
        assert (
            self.batch_size_per_gpu % self.batch_split == 0
        ), "Batch size per GPU must be divisible by batch split."
        self.freeze_modules = freeze_modules or []
        if self.freeze_modules:
            self._freeze_modules(self.freeze_modules)
        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)
        self.step = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, "ckpts"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, "tb_logs"))
        if self.parallel_mode == "ddp" and self.world_size > 1:
            self.check_ddp()
        if self.is_master:
            print("\n\nTrainer initialized.")
            print(self)

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f"  - Models:")
        for name, model in self.models.items():
            lines.append(f"    - {name}: {model.__class__.__name__}")
        lines.append(f"  - Dataset: {indent(str(self.dataset), 2)}")
        lines.append(f"  - Dataloader:")
        lines.append(f"    - Sampler: {self.dataloader.sampler.__class__.__name__}")
        lines.append(f"    - Num workers: {self.dataloader.num_workers}")
        lines.append(f"  - Number of steps: {self.max_steps}")
        lines.append(f"  - Number of GPUs: {self.world_size}")
        lines.append(f"  - Batch size: {self.batch_size}")
        lines.append(f"  - Batch size per GPU: {self.batch_size_per_gpu}")
        lines.append(f"  - Batch split: {self.batch_split}")
        lines.append(f"  - Optimizer: {self.optimizer.__class__.__name__}")
        lines.append(f"  - Learning rate: {self.optimizer.param_groups[0]['lr']}")
        if self.lr_scheduler_config is not None:
            lines.append(f"  - LR scheduler: {self.lr_scheduler.__class__.__name__}")
        if self.elastic_controller_config is not None:
            lines.append(
                f"  - Elastic memory: {indent(str(self.elastic_controller), 2)}"
            )
        if self.grad_clip is not None:
            lines.append(f"  - Gradient clip: {indent(str(self.grad_clip), 2)}")
        lines.append(f"  - EMA rate: {self.ema_rate}")
        lines.append(f"  - Mixed precision dtype: {self.mix_precision_dtype}")
        lines.append(f"  - Mixed precision mode: {self.mix_precision_mode}")
        if (
            self.mix_precision_mode == "amp"
            and self.mix_precision_dtype == torch.float16
        ):
            lines.append(f"  - FP16 scale growth: {self.fp16_scale_growth}")
        lines.append(f"  - Parallel mode: {self.parallel_mode}")
        return "\n".join(lines)

    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, "device"):
                return model.device
        return next(list(self.models.values())[0].parameters()).device

    def _freeze_modules(self, freeze_modules: List[str]):


        for module_spec in freeze_modules:
            model_name = module_spec.split(".")[0]
            if model_name not in self.models:
                if self.is_master:
                    print(
                        f'[Trainer] Warning: model "{model_name}" not found, skipping freeze for "{module_spec}"'
                    )
                continue
            if "." in module_spec:
                submodule_path = ".".join(module_spec.split(".")[1:])
                try:
                    submodule = self.models[model_name]
                    for attr in submodule_path.split("."):
                        submodule = getattr(submodule, attr)
                    frozen_count = 0
                    for param in submodule.parameters():
                        param.requires_grad = False
                        frozen_count += 1
                    if self.is_master:
                        print(
                            f"[Trainer] Frozen {module_spec} ({frozen_count} parameters)"
                        )
                except AttributeError as e:
                    if self.is_master:
                        print(
                            f'[Trainer] Error: submodule path "{submodule_path}" not found in "{model_name}": {e}'
                        )
            else:
                frozen_count = 0
                for param in self.models[model_name].parameters():
                    param.requires_grad = False
                    frozen_count += 1
                if self.is_master:
                    print(
                        f'[Trainer] Frozen entire model "{model_name}" ({frozen_count} parameters)'
                    )
        if self.is_master:
            print("[Trainer] Trainable parameters after freezing:")
            for name, model in self.models.items():
                trainable = sum(
                    (p.numel() for p in model.parameters() if p.requires_grad)
                )
                total = sum((p.numel() for p in model.parameters()))
                frozen_pct = 100 * (1 - trainable / total) if total > 0 else 0
                print(
                    f"  - {name}: {trainable:,} / {total:,} ({frozen_pct:.1f}% frozen)"
                )

    def init_models_and_more(self, **kwargs):

        if self.world_size > 1:
            self.training_models = {}
            for name, model in self.models.items():
                has_trainable = any((p.requires_grad for p in model.parameters()))
                if has_trainable:
                    torch.cuda.synchronize()
                    try:
                        self.training_models[name] = DDP(
                            model,
                            device_ids=[self.local_rank],
                            output_device=self.local_rank,
                            bucket_cap_mb=128,
                            find_unused_parameters=False,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"[Rank {self.rank}] DDP init failed for model '{name}' on local_rank={self.local_rank}"
                        ) from e
                else:
                    self.training_models[name] = model
                    if self.is_master:
                        print(f'[Trainer] Skipping DDP for frozen model "{name}"')
        else:
            self.training_models = self.models
        model_param_pairs = []
        for model_name, model in self.models.items():
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    model_param_pairs.append((f"{model_name}.{param_name}", param))
        self.model_param_names = [name for name, _ in model_param_pairs]
        self.model_params = [param for _, param in model_param_pairs]
        if self.mix_precision_mode == "amp":
            self.master_params = self.model_params
            if self.mix_precision_dtype == torch.float16:
                self.scaler = torch.GradScaler()
        elif self.mix_precision_mode == "inflat_all":
            self.master_params = make_master_params(self.model_params)
            if self.mix_precision_dtype == torch.float16:
                self.log_scale = 20.0
        elif self.mix_precision_mode is None:
            self.master_params = self.model_params
        else:
            raise NotImplementedError(
                f"Mix precision mode {self.mix_precision_mode} is not implemented."
            )
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]
        if hasattr(torch.optim, self.optimizer_config["name"]):
            self.optimizer = getattr(torch.optim, self.optimizer_config["name"])(
                self.master_params, **self.optimizer_config["args"]
            )
        else:
            self.optimizer = globals()[self.optimizer_config["name"]](
                self.master_params, **self.optimizer_config["args"]
            )
        if self.lr_scheduler_config is not None:
            import inspect

            sched_args = dict(self.lr_scheduler_config["args"])
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config["name"]):
                sched_cls = getattr(
                    torch.optim.lr_scheduler, self.lr_scheduler_config["name"]
                )
            else:
                sched_cls = globals()[self.lr_scheduler_config["name"]]
            sig = inspect.signature(sched_cls)
            if "total_steps" in sig.parameters and "total_steps" not in sched_args:
                sched_args["total_steps"] = self.max_steps
            self.lr_scheduler = sched_cls(self.optimizer, **sched_args)
        if self.elastic_controller_config is not None:
            assert any(
                [
                    isinstance(
                        model,
                        (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin),
                    )
                    for model in self.models.values()
                ]
            ), "No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin"
            self.elastic_controller = getattr(
                elastic_utils, self.elastic_controller_config["name"]
            )(**self.elastic_controller_config["args"])
            for model in self.models.values():
                if isinstance(
                    model,
                    (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin),
                ):
                    model.register_memory_controller(self.elastic_controller)
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip["name"])(
                    **self.grad_clip["args"]
                )

    def prepare_dataloader(self, **kwargs):

        self.data_sampler = ResumableSampler(self.dataset, shuffle=True)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.dataloader_workers > 0,
            collate_fn=(
                self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None
            ),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def _master_params_to_state_dicts(self, master_params):

        if self.mix_precision_mode == "inflat_all":
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dicts = {name: model.state_dict() for name, model in self.models.items()}
        master_params_names = sum(
            [
                [(name, n) for n, p in model.named_parameters() if p.requires_grad]
                for name, model in self.models.items()
            ],
            [],
        )
        for i, (model_name, param_name) in enumerate(master_params_names):
            state_dicts[model_name][param_name] = master_params[i]
        return state_dicts

    def _state_dicts_to_master_params(self, master_params, state_dicts):

        master_params_names = sum(
            [
                [(name, n) for n, p in model.named_parameters() if p.requires_grad]
                for name, model in self.models.items()
            ],
            [],
        )
        params = []
        for name, param_name in master_params_names:
            if param_name in state_dicts[name]:
                params.append(state_dicts[name][param_name])
            else:
                params.append(self.models[name].state_dict()[param_name])
        if self.mix_precision_mode == "inflat_all":
            model_params_to_master_params(params, master_params)
        else:
            for i, param in enumerate(params):
                master_params[i].data.copy_(param.data)

    def _load_model_resume_state(self, load_dir: str, model_name: str, step: int):

        ckpt_path = os.path.join(load_dir, "ckpts", f"{model_name}_step{step:07d}.pt")
        if broadcast_file_exists(ckpt_path):
            return torch.load(
                read_file_dist(ckpt_path), map_location=self.device, weights_only=True
            )
        if not self._is_model_fully_frozen(model_name):
            raise FileNotFoundError(
                f'Missing resume checkpoint "{ckpt_path}" for non-frozen model "{model_name}".'
            )
        fallback_path = self.finetune_ckpt.get(model_name)
        if fallback_path is None:
            raise FileNotFoundError(
                f'Missing resume checkpoint "{ckpt_path}" for frozen model "{model_name}", and no finetune_ckpt fallback is configured.'
            )
        if self.is_master:
            print(
                f'\n[Trainer] Resume fallback for frozen model "{model_name}": {fallback_path}',
                end="",
            )
        if fallback_path.endswith(".safetensors"):
            return load_safetensors(fallback_path, device=str(self.device))
        return torch.load(
            read_file_dist(fallback_path), map_location=self.device, weights_only=True
        )

    def load(self, load_dir, step=0):


        if self.is_master:
            print(f"\nLoading checkpoint from step {step}...", end="")
        model_ckpts = {}
        for name, model in self.models.items():
            model_ckpt = remap_legacy_state_dict(
                self._load_model_resume_state(load_dir, name, step), model.state_dict()
            )
            frozen_prefixes = self._get_frozen_submodule_prefixes(name)
            if frozen_prefixes:
                if self.finetune_ckpt and name in self.finetune_ckpt:
                    ft_path = self.finetune_ckpt[name]
                    if ft_path.endswith(".safetensors"):
                        ft_ckpt = load_safetensors(ft_path, device=str(self.device))
                    else:
                        ft_ckpt = torch.load(
                            read_file_dist(ft_path),
                            map_location=self.device,
                            weights_only=True,
                        )
                    frozen_keys_loaded = 0
                    for k, v in ft_ckpt.items():
                        if any((k.startswith(p) for p in frozen_prefixes)):
                            model_ckpt[k] = v
                            frozen_keys_loaded += 1
                    del ft_ckpt
                    if self.is_master:
                        print(
                            f'\n[Trainer] Restored {frozen_keys_loaded} frozen sub-module keys for "{name}" from finetune_ckpt',
                            end="",
                        )
                missing, unexpected = model.load_state_dict(model_ckpt, strict=False)
                non_frozen_missing = missing
                if non_frozen_missing or unexpected:
                    raise RuntimeError(
                        f'Incomplete resume state for "{name}": missing={missing}, unexpected={unexpected}'
                    )
            else:
                model.load_state_dict(model_ckpt)
            if not self._is_model_fully_frozen(name):
                model_ckpts[name] = model_ckpt
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts
        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    if self._is_model_fully_frozen(name):
                        continue
                    ema_ckpt = torch.load(
                        os.path.join(
                            load_dir, "ckpts", f"{name}_ema{ema_rate}_step{step:07d}.pt"
                        ),
                        map_location=self.device,
                        weights_only=True,
                    )
                    ema_ckpt = remap_legacy_state_dict(ema_ckpt, model.state_dict())
                    expected = {
                        key
                        for key, parameter in model.named_parameters()
                        if parameter.requires_grad
                    }
                    missing_ema = expected - ema_ckpt.keys()
                    unexpected_ema = ema_ckpt.keys() - model.state_dict().keys()
                    if missing_ema or unexpected_ema:
                        raise RuntimeError(
                            f"Incomplete EMA state for {name}: missing={sorted(missing_ema)}, unexpected={sorted(unexpected_ema)}"
                        )
                    ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts
        misc_ckpt = torch.load(
            read_file_dist(os.path.join(load_dir, "ckpts", f"misc_step{step:07d}.pt")),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        saved_freeze = set(misc_ckpt.get("freeze_modules") or [])
        current_freeze = set(self.freeze_modules or [])
        if saved_freeze != current_freeze:
            if self.is_master:
                print(f"\n[Trainer] WARNING: freeze_modules mismatch!")
                print(
                    f"  - Saved checkpoint: {(sorted(saved_freeze) if saved_freeze else 'none')}"
                )
                print(
                    f"  - Current config:   {(sorted(current_freeze) if current_freeze else 'none')}"
                )
                print(f"  - Optimizer state will NOT be loaded to avoid crash")
        else:
            self.optimizer.load_state_dict(misc_ckpt["optimizer"])
        self.step = misc_ckpt["step"]
        self.data_sampler.load_state_dict(misc_ckpt["data_sampler"])
        if (
            self.mix_precision_mode == "amp"
            and self.mix_precision_dtype == torch.float16
        ):
            self.scaler.load_state_dict(misc_ckpt["scaler"])
        elif (
            self.mix_precision_mode == "inflat_all"
            and self.mix_precision_dtype == torch.float16
        ):
            self.log_scale = misc_ckpt["log_scale"]
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.load_state_dict(misc_ckpt["lr_scheduler"])
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt["elastic_controller"])
        if self.grad_clip is not None and (not isinstance(self.grad_clip, float)):
            self.grad_clip.load_state_dict(misc_ckpt["grad_clip"])
        del misc_ckpt
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(" Done.")
        if self.world_size > 1:
            self.check_ddp()

    def _is_model_fully_frozen(self, model_name: str) -> bool:

        return model_name in self.freeze_modules

    def _get_frozen_submodule_prefixes(self, model_name: str) -> list:

        prefixes = []
        for spec in self.freeze_modules:
            if "." in spec and spec.split(".")[0] == model_name:
                subpath = ".".join(spec.split(".")[1:])
                prefixes.append(subpath + ".")
        return prefixes

    def save(self, non_blocking=True):


        assert self.is_master, "save() should be called only by the rank 0 process."
        print(f"\nSaving checkpoint at step {self.step}...", end="")
        skipped_models = [
            name for name in self.models if self._is_model_fully_frozen(name)
        ]
        if skipped_models:
            print(f" (skipping frozen: {skipped_models})", end="")
        model_ckpts = self._master_params_to_state_dicts(self.master_params)
        for name, model_ckpt in model_ckpts.items():
            if self._is_model_fully_frozen(name):
                continue
            frozen_prefixes = self._get_frozen_submodule_prefixes(name)
            if frozen_prefixes:
                model_ckpt = {
                    k: v
                    for k, v in model_ckpt.items()
                    if not any((k.startswith(p) for p in frozen_prefixes))
                }
            model_ckpt = {k: v.cpu() for k, v in model_ckpt.items()}
            if non_blocking:
                threading.Thread(
                    target=torch.save,
                    args=(
                        model_ckpt,
                        os.path.join(
                            self.output_dir, "ckpts", f"{name}_step{self.step:07d}.pt"
                        ),
                    ),
                ).start()
            else:
                torch.save(
                    model_ckpt,
                    os.path.join(
                        self.output_dir, "ckpts", f"{name}_step{self.step:07d}.pt"
                    ),
                )
        for i, ema_rate in enumerate(self.ema_rate):
            ema_ckpts = self._master_params_to_state_dicts(self.ema_params[i])
            for name, ema_ckpt in ema_ckpts.items():
                if self._is_model_fully_frozen(name):
                    continue
                frozen_prefixes = self._get_frozen_submodule_prefixes(name)
                if frozen_prefixes:
                    ema_ckpt = {
                        k: v
                        for k, v in ema_ckpt.items()
                        if not any((k.startswith(p) for p in frozen_prefixes))
                    }
                ema_ckpt = {k: v.cpu() for k, v in ema_ckpt.items()}
                if non_blocking:
                    threading.Thread(
                        target=torch.save,
                        args=(
                            ema_ckpt,
                            os.path.join(
                                self.output_dir,
                                "ckpts",
                                f"{name}_ema{ema_rate}_step{self.step:07d}.pt",
                            ),
                        ),
                    ).start()
                else:
                    torch.save(
                        ema_ckpt,
                        os.path.join(
                            self.output_dir,
                            "ckpts",
                            f"{name}_ema{ema_rate}_step{self.step:07d}.pt",
                        ),
                    )
        misc_ckpt = {
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "data_sampler": self.data_sampler.state_dict(),
            "freeze_modules": self.freeze_modules,
        }
        if (
            self.mix_precision_mode == "amp"
            and self.mix_precision_dtype == torch.float16
        ):
            misc_ckpt["scaler"] = self.scaler.state_dict()
        elif (
            self.mix_precision_mode == "inflat_all"
            and self.mix_precision_dtype == torch.float16
        ):
            misc_ckpt["log_scale"] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt["elastic_controller"] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and (not isinstance(self.grad_clip, float)):
            misc_ckpt["grad_clip"] = self.grad_clip.state_dict()
        if non_blocking:
            threading.Thread(
                target=torch.save,
                args=(
                    misc_ckpt,
                    os.path.join(
                        self.output_dir, "ckpts", f"misc_step{self.step:07d}.pt"
                    ),
                ),
            ).start()
        else:
            torch.save(
                misc_ckpt,
                os.path.join(self.output_dir, "ckpts", f"misc_step{self.step:07d}.pt"),
            )
        print(" Done.")

    def finetune_from(self, finetune_ckpt):


        if self.is_master:
            print("\nFinetuning from:")
            for name, path in finetune_ckpt.items():
                print(f"  - {name}: {path}")
        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                ckpt_path_str = finetune_ckpt[name]
                if ckpt_path_str.endswith(".safetensors"):
                    model_ckpt = load_safetensors(
                        ckpt_path_str, device=str(self.device)
                    )
                else:
                    ckpt_file = read_file_dist(ckpt_path_str)
                    model_ckpt = torch.load(
                        ckpt_file, map_location=self.device, weights_only=True
                    )
                model_ckpt = remap_legacy_state_dict(model_ckpt, model_state_dict)
                for k, v in model_ckpt.items():
                    if k not in model_state_dict:
                        if self.is_master:
                            print(
                                f"Warning: {k} not found in model_state_dict, skipped."
                            )
                        model_ckpt[k] = None
                    elif model_ckpt[k].shape != model_state_dict[k].shape:
                        raise RuntimeError(
                            f"Checkpoint shape mismatch for {name}.{k}: {v.shape} vs {model_state_dict[k].shape}"
                        )
                model_ckpt = {k: v for k, v in model_ckpt.items() if v is not None}
                model_ckpts[name] = model_ckpt
                load_result = model.load_state_dict(model_ckpt, strict=False)
                frozen_prefixes = self._get_frozen_submodule_prefixes(name)
                missing_frozen = [
                    key
                    for key in load_result.missing_keys
                    if self._is_model_fully_frozen(name)
                    or any((key.startswith(prefix) for prefix in frozen_prefixes))
                ]
                if missing_frozen:
                    raise RuntimeError(
                        f"Pretrained checkpoint for {name} is missing frozen parameters: {missing_frozen}"
                    )
                if load_result.missing_keys and self.is_master:
                    missing_modules = {}
                    for key in load_result.missing_keys:
                        top_module = key.split(".")[0]
                        if top_module not in missing_modules:
                            missing_modules[top_module] = 0
                        missing_modules[top_module] += 1
                    print(
                        f"  [{name}] Randomly initialized modules (not in checkpoint):"
                    )
                    for mod, count in sorted(missing_modules.items()):
                        print(f"    - {mod}: {count} parameters")
            else:
                if self.is_master:
                    print(f"Warning: {name} not found in finetune_ckpt, skipped.")
                model_ckpts[name] = model_state_dict
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                self._state_dicts_to_master_params(self.ema_params[i], model_ckpts)
        del model_ckpts
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print("Done.")
        if self.world_size > 1:
            self.check_ddp()

    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):

        pass

    @torch.no_grad()
    def visualize_sample(self, sample):

        if hasattr(self.dataset, "visualize_sample"):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=16, batch_size=4):

        num_samples = min(num_samples, len(self.dataset))
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=True,
            collate_fn=(
                self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None
            ),
        )
        save_cfg = {}
        for i in range(0, num_samples, batch_size):
            data = next(iter(dataloader))
            data = {k: v[: min(num_samples - i, batch_size)] for k, v in data.items()}
            data = recursive_to_device(data, self.device)
            vis = self.visualize_sample(data)
            if isinstance(vis, dict):
                for k, v in vis.items():
                    if f"dataset_{k}" not in save_cfg:
                        save_cfg[f"dataset_{k}"] = []
                    save_cfg[f"dataset_{k}"].append(v)
            else:
                if "dataset" not in save_cfg:
                    save_cfg["dataset"] = []
                save_cfg["dataset"].append(vis)
        for name, image in save_cfg.items():
            utils.save_image(
                torch.cat(image, dim=0),
                os.path.join(self.output_dir, "samples", f"{name}.jpg"),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=16, batch_size=4, verbose=False):


        snapshot_master_only = (
            bool(getattr(self, "snapshot_master_only", False)) and self.world_size > 1
        )
        snapshot_use_distributed_collectives = self.world_size > 1 and bool(
            getattr(self, "snapshot_use_distributed_collectives", True)
        )
        if snapshot_master_only and (not self.is_master):
            if dist.is_initialized():
                dist.barrier()
            return
        num_samples = min(num_samples, len(self.dataset))
        if self.is_master:
            print(f"\nSampling {num_samples} images...", end="")
        if suffix is None:
            suffix = f"step{self.step:07d}"
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        amp_context = (
            partial(torch.autocast, device_type="cuda", dtype=self.mix_precision_dtype)
            if self.mix_precision_mode == "amp"
            else nullcontext
        )
        samples = None
        snapshot_error = None
        with amp_context():
            try:
                samples = self.run_snapshot(
                    num_samples_per_process, batch_size=batch_size, verbose=verbose
                )
            except Exception as e:
                snapshot_error = f"[Rank {self.rank}] {type(e).__name__}: {e}"
                for model in self.models.values():
                    model.train()
                try:
                    torch.cuda.synchronize()
                except RuntimeError:
                    pass
                torch.cuda.current_device()
        local_failed = 1 if snapshot_error is not None else 0
        failed_ranks = [self.rank] if local_failed else []
        if snapshot_use_distributed_collectives:
            fail_flag = torch.tensor(
                local_failed, dtype=torch.int32, device=self.device
            )
            all_fail_flags = [
                torch.zeros_like(fail_flag) for _ in range(self.world_size)
            ]
            dist.all_gather(all_fail_flags, fail_flag)
            failed_ranks = [
                r for r, flag in enumerate(all_fail_flags) if int(flag.item()) == 1
            ]
        if len(failed_ranks) > 0:
            if self.is_master:
                if snapshot_error is not None:
                    print(
                        f" Failed. Snapshot skipped to keep training running. {snapshot_error}"
                    )
                else:
                    print(
                        f" Failed. Snapshot skipped to keep training running. Failure ranks: {failed_ranks}"
                    )
            if snapshot_use_distributed_collectives:
                dist.barrier()
            elif snapshot_master_only and dist.is_initialized():
                dist.barrier()
            return
        for key in list(samples.keys()):
            if samples[key]["type"] == "sample":
                vis = self.visualize_sample(samples[key]["value"])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f"{key}_{k}"] = {"value": v, "type": "image"}
                    del samples[key]
                else:
                    samples[key] = {"value": vis, "type": "image"}
        if snapshot_use_distributed_collectives:
            for key in samples.keys():
                samples[key]["value"] = samples[key]["value"].contiguous()
                if self.is_master:
                    all_images = [
                        torch.empty_like(samples[key]["value"])
                        for _ in range(self.world_size)
                    ]
                else:
                    all_images = []
                dist.gather(samples[key]["value"], all_images, dst=0)
                if self.is_master:
                    samples[key]["value"] = torch.cat(all_images, dim=0)[:num_samples]
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, "samples", suffix), exist_ok=True)
            for key in samples.keys():
                if samples[key]["type"] == "image":
                    utils.save_image(
                        samples[key]["value"],
                        os.path.join(
                            self.output_dir, "samples", suffix, f"{key}_{suffix}.jpg"
                        ),
                        nrow=int(np.sqrt(num_samples)),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
                elif samples[key]["type"] == "number":
                    val_min = samples[key]["value"].min()
                    val_max = samples[key]["value"].max()
                    images = (samples[key]["value"] - val_min) / (val_max - val_min)
                    images = utils.make_grid(
                        images, nrow=int(np.sqrt(num_samples)), normalize=False
                    )
                    save_image_with_notes(
                        images,
                        os.path.join(
                            self.output_dir, "samples", suffix, f"{key}_{suffix}.jpg"
                        ),
                        notes=f"{key} min: {val_min}, max: {val_max}",
                    )
        if self.is_master:
            print(" Done.")
        if snapshot_use_distributed_collectives:
            dist.barrier()
        elif snapshot_master_only and dist.is_initialized():
            dist.barrier()

    def update_ema(self):


        assert (
            self.is_master
        ), "update_ema() should be called only by the rank 0 process."
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(
                    master_param, alpha=1.0 - ema_rate
                )

    def check_ddp(self):


        if not self.enable_ddp_check:
            if self.is_master:
                print(
                    "[DDP] Consistency check disabled by DDP_ENABLE_CHECK=0", flush=True
                )
            if dist.is_initialized():
                dist.barrier()
            return
        if self.is_master:
            print("\nPerforming DDP check...")
        if self.is_master:
            print("Checking if parameters are consistent across processes...")
        dist.barrier()
        try:
            for p in self.master_params:
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().view(-1)[i : i + sub_size]
                    sub_p_gather = [
                        torch.empty_like(sub_p) for _ in range(self.world_size)
                    ]
                    dist.all_gather(sub_p_gather, sub_p)
                    assert all(
                        [
                            torch.equal(sub_p, sub_p_gather[i])
                            for i in range(self.world_size)
                        ]
                    ), "parameters are not consistent across processes"
        except AssertionError as e:
            if self.is_master:
                print(f"\n\x1b[91mError: {e}\x1b[0m")
                print("DDP check failed.")
            raise e
        dist.barrier()
        if self.is_master:
            print("Done.")

    @abstractmethod
    def training_losses(**mb_data):

        pass

    def load_data(self):

        if self.prefetch_data:
            if self._data_prefetched is None:
                raw_data = next(self.data_iterator)
                self._data_prefetched = recursive_to_device(
                    raw_data, self.device, non_blocking=True
                )
            data = self._data_prefetched
            raw_data = next(self.data_iterator)
            self._data_prefetched = recursive_to_device(
                raw_data, self.device, non_blocking=True
            )
        else:
            data = recursive_to_device(
                next(self.data_iterator), self.device, non_blocking=True
            )
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                data_list = [
                    {
                        k: v[
                            i
                            * batch_size
                            // self.batch_split : (i + 1)
                            * batch_size
                            // self.batch_split
                        ]
                        for k, v in data.items()
                    }
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError("Data must be a dict or a list of dicts.")
        return data_list

    def run_step(self, data_list):

        step_log = {"loss": {}, "status": {}}
        amp_context = (
            partial(torch.autocast, device_type="cuda", dtype=self.mix_precision_dtype)
            if self.mix_precision_mode == "amp"
            else nullcontext
        )
        elastic_controller_context = (
            self.elastic_controller.record
            if self.elastic_controller_config is not None
            else nullcontext
        )
        losses = []
        statuses = []
        elastic_controller_logs = []
        zero_grad(self.model_params)
        for i, mb_data in enumerate(data_list):
            sync_contexts = []
            if i != len(data_list) - 1 and self.world_size > 1:
                for name, model in self.training_models.items():
                    if isinstance(model, DDP):
                        sync_contexts.append(model.no_sync)
            if not sync_contexts:
                sync_contexts = [nullcontext]
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    loss, status = self.training_losses(**mb_data)
                    l = loss["loss"] / len(data_list)
                if (
                    self.mix_precision_mode == "amp"
                    and self.mix_precision_dtype == torch.float16
                ):
                    self.scaler.scale(l).backward()
                elif (
                    self.mix_precision_mode == "inflat_all"
                    and self.mix_precision_dtype == torch.float16
                ):
                    scaled_l = l * 2**self.log_scale
                    scaled_l.backward()
                else:
                    l.backward()
            losses.append(
                dict_foreach(
                    loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x
                )
            )
            statuses.append(
                dict_foreach(
                    status, lambda x: x.item() if isinstance(x, torch.Tensor) else x
                )
            )
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())
        if self.grad_clip is not None:
            if (
                self.mix_precision_mode == "amp"
                and self.mix_precision_dtype == torch.float16
            ):
                self.scaler.unscale_(self.optimizer)
            elif self.mix_precision_mode == "inflat_all":
                model_grads_to_master_grads(self.model_params, self.master_params)
                if self.mix_precision_dtype == torch.float16:
                    self.master_params[0].grad.mul_(1.0 / 2**self.log_scale)
            if isinstance(self.grad_clip, float):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.master_params, self.grad_clip
                )
            else:
                grad_norm = self.grad_clip(self.master_params)
            if torch.isfinite(grad_norm):
                statuses[-1]["grad_norm"] = grad_norm.item()
        optimizer_stepped = False
        if (
            self.mix_precision_mode == "amp"
            and self.mix_precision_dtype == torch.float16
        ):
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_stepped = self.scaler.get_scale() >= prev_scale
            if not optimizer_stepped:
                self._log_first_nonfinite_grad(
                    self.model_params, self.model_param_names
                )
        elif self.mix_precision_mode == "inflat_all":
            if self.mix_precision_dtype == torch.float16:
                prev_scale = 2**self.log_scale
                if not any((not p.grad.isfinite().all() for p in self.model_params)):
                    if self.grad_clip is None:
                        model_grads_to_master_grads(
                            self.model_params, self.master_params
                        )
                        self.master_params[0].grad.mul_(1.0 / 2**self.log_scale)
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                    self.log_scale += self.fp16_scale_growth
                    optimizer_stepped = True
                else:
                    self._log_first_nonfinite_grad(
                        self.model_params, self.model_param_names
                    )
                    self.log_scale -= 1
            else:
                prev_scale = 1.0
                if self.grad_clip is None:
                    model_grads_to_master_grads(self.model_params, self.master_params)
                if not any((not p.grad.isfinite().all() for p in self.master_params)):
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                    optimizer_stepped = True
                else:
                    self._log_first_nonfinite_grad(self.master_params)
                    print(
                        "\n\x1b[93mWarning: NaN detected in gradients. Skipping update.\x1b[0m"
                    )
        else:
            prev_scale = 1.0
            if not any((not p.grad.isfinite().all() for p in self.model_params)):
                self.optimizer.step()
                optimizer_stepped = True
            else:
                self._log_first_nonfinite_grad(
                    self.model_params, self.model_param_names
                )
                print(
                    "\n\x1b[93mWarning: NaN detected in gradients. Skipping update.\x1b[0m"
                )
        statuses[-1]["lr"] = self.optimizer.param_groups[0]["lr"]
        statuses[-1]["optimizer_step_skipped"] = 0.0 if optimizer_stepped else 1.0
        if self.lr_scheduler_config is not None and optimizer_stepped:
            self.lr_scheduler.step()
        step_log["loss"] = dict_reduce(losses, lambda x: np.mean(x))
        step_log["status"] = dict_reduce(
            statuses,
            lambda x: np.mean(x),
            special_func={"min": lambda x: np.min(x), "max": lambda x: np.max(x)},
        )
        if self.elastic_controller_config is not None:
            step_log["elastic"] = dict_reduce(
                elastic_controller_logs, lambda x: np.mean(x)
            )
        if self.grad_clip is not None:
            step_log["grad_clip"] = (
                self.grad_clip
                if isinstance(self.grad_clip, float)
                else self.grad_clip.log()
            )
        if self.log_param_stats:
            param_norms = {}
            param_grads = {}
            for model_name, model in self.models.items():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param_norms[f"{model_name}.{name}"] = param.norm().item()
                        if param.grad is not None and torch.isfinite(param.grad).all():
                            param_grads[f"{model_name}.{name}"] = (
                                param.grad.norm().item() / prev_scale
                            )
            step_log["param_norms"] = param_norms
            step_log["param_grads"] = param_grads
        if self.is_master:
            self.update_ema()
        return step_log

    def save_logs(self):
        if not self.log:
            return
        log_str = "\n".join(
            [
                f"{step}: {json.dumps(dict_foreach(log, lambda x: float(x)))}"
                for step, log in self.log
            ]
        )
        with open(os.path.join(self.output_dir, "log.txt"), "a") as log_file:
            log_file.write(log_str + "\n")
        log_show = [l for _, l in self.log if not dict_any(l, lambda x: np.isnan(x))]
        if log_show:
            log_show = dict_reduce(log_show, lambda x: np.mean(x))
            log_show = dict_flatten(log_show, sep="/")
            for key, value in log_show.items():
                self.writer.add_scalar(key, value, self.step)
        self.log = []

    def _log_first_nonfinite_grad(self, params, param_names=None):
        if self._logged_first_nonfinite_grad or not self.is_master:
            return
        for idx, param in enumerate(params):
            grad = param.grad
            if grad is None:
                continue
            finite_mask = torch.isfinite(grad)
            isfinite = bool(finite_mask.all().item())
            if isfinite:
                continue
            finite_ratio = finite_mask.float().mean().item()
            finite_grad = grad[finite_mask]
            if finite_grad.numel() > 0:
                grad_min = finite_grad.min().item()
                grad_max = finite_grad.max().item()
            else:
                grad_min = float("nan")
                grad_max = float("nan")
            has_nan = bool(torch.isnan(grad).any().item())
            has_inf = bool(torch.isinf(grad).any().item())
            param_name = (
                param_names[idx]
                if param_names is not None and idx < len(param_names)
                else f"param_{idx}"
            )
            print(
                f"\n[Rank {self.rank}] First non-finite grad detected at step {self.step}: {param_name}, grad_min={grad_min:.6e}, grad_max={grad_max:.6e}, isfinite={isfinite}, finite_ratio={finite_ratio:.6f}, has_nan={has_nan}, has_inf={has_inf}",
                flush=True,
            )
            self._logged_first_nonfinite_grad = True
            return

    def check_abort(self):

        if (
            self.mix_precision_dtype == torch.float16
            and self.mix_precision_mode == "inflat_all"
            and (self.log_scale < 0)
        ):
            if self.is_master:
                print("\n\n\x1b[91m")
                print(
                    f"ABORT: log_scale in inflat_all mode is less than 0 at step {self.step}."
                )
                print(
                    "This indicates that the model is diverging. You should look into the model and the data."
                )
                print("\x1b[0m")
                self.save(non_blocking=False)
                self.save_logs()
            if self.world_size > 1:
                dist.barrier()
            raise ValueError("ABORT: log_scale in inflat_all mode is less than 0.")

    def run(self):

        if self.is_master:
            print("\nStarting training...")
            print(f"[Rank {self.rank}] Creating dataset snapshot...", flush=True)
            self.snapshot_dataset(
                num_samples=self.snapshot_num_samples,
                batch_size=self.snapshot_batch_size,
            )
            print(f"[Rank {self.rank}] Dataset snapshot saved.", flush=True)
        if self.is_master:
            print(f"[Rank {self.rank}] Entering training loop...", flush=True)
        time_last_print = 0.0
        time_elapsed = 0.0
        pbar = tqdm(
            initial=self.step,
            total=self.max_steps,
            desc="Training",
            disable=not self.is_master,
            dynamic_ncols=True,
            smoothing=0.1,
        )
        while self.step < self.max_steps:
            time_start = time.time()
            data_list = self.load_data()
            step_log = self.run_step(data_list)
            time_end = time.time()
            time_elapsed += time_end - time_start
            self.step += 1
            if self.is_master:
                pbar.update(1)
                if (
                    step_log is not None
                    and "loss" in step_log
                    and ("loss" in step_log["loss"])
                ):
                    loss_val = step_log["loss"]["loss"]
                    pbar.set_postfix(
                        {
                            "loss": f"{loss_val:.4f}",
                            "speed": f"{3600 / (time_end - time_start):.1f} steps/h",
                        }
                    )
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f"Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)",
                    f"Elapsed: {time_elapsed / 3600:.2f} h",
                    f"Speed: {speed:.2f} steps/h",
                    f"ETA: {(self.max_steps - self.step) / speed:.2f} h",
                ]
                tqdm.write(" | ".join([c.ljust(25) for c in columns]))
                time_last_print = time_elapsed
            if (
                self.parallel_mode == "ddp"
                and self.world_size > 1
                and (self.i_ddpcheck is not None)
                and (self.step % self.i_ddpcheck == 0)
            ):
                self.check_ddp()
            if self.step % self.i_sample == 0 and self.step < self.max_steps:
                self.snapshot(
                    num_samples=self.snapshot_num_samples,
                    batch_size=self.snapshot_batch_size,
                )
            if self.is_master:
                self.log.append((self.step, {}))
                self.log[-1][1]["time"] = {
                    "step": time_end - time_start,
                    "elapsed": time_elapsed,
                }
                if step_log is not None:
                    self.log[-1][1].update(step_log)
                if self.mix_precision_dtype == torch.float16:
                    if self.mix_precision_mode == "amp":
                        self.log[-1][1]["scale"] = self.scaler.get_scale()
                    elif self.mix_precision_mode == "inflat_all":
                        self.log[-1][1]["log_scale"] = self.log_scale
                if self.step % self.i_log == 0:
                    self.save_logs()
                if self.step % self.i_save == 0 and self.step < self.max_steps:
                    self.save()
            self.check_abort()
        pbar.close()
        if self.is_master:
            self.save(non_blocking=False)
            self.save_logs()
        if self.world_size > 1:
            dist.barrier()
        self.snapshot(
            suffix="final",
            num_samples=self.snapshot_num_samples,
            batch_size=self.snapshot_batch_size,
        )
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            self.writer.close()
            print("Training finished.")
