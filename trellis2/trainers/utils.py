import math
import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def str_to_dtype(dtype_str: str):
    return {
        "f16": torch.float16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "f32": torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[dtype_str]


def make_master_params(model_params):

    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def unflatten_master_params(model_params, master_params):

    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def model_params_to_master_params(model_params, master_params):

    master_params[0].detach().copy_(
        _flatten_dense_tensors([param.detach().float() for param in model_params])
    )


def master_params_to_model_params(model_params, master_params):

    for param, master_param in zip(
        model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)
    ):
        param.detach().copy_(master_param)


def model_grads_to_master_grads(model_params, master_params):


    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )


def zero_grad(model_params):
    for param in model_params:
        if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()


from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupLRScheduler(LambdaLR):

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLRScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 1) / self.warmup_steps
        return 1.0


class CosineWarmupLRScheduler(LambdaLR):


    def __init__(
        self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0, last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 1) / self.warmup_steps
        progress = (current_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)
        return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )


def remap_legacy_state_dict(
    state: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:

    result = {}
    for key, value in state.items():
        mapped = "cola3d." + key[4:] if key.startswith("pat.") else key
        candidate = mapped.replace(
            "mask_prompt_encoder.cnn.", "mask_prompt_encoder.encoder.", 1
        )
        if candidate in expected:
            mapped = candidate
        if mapped in result:
            raise RuntimeError(
                f"Duplicate checkpoint key after compatibility mapping: {mapped}"
            )
        result[mapped] = value
    return result
