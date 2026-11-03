

from collections.abc import Mapping

import torch
from torch import nn


def load_decoder_state_dict(
    decoder: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:

    current_state = decoder.state_dict()
    missing = current_state.keys() - state_dict.keys()
    allowed = set()
    extractor = getattr(decoder, "utonia_extractor", None)
    if missing and extractor is not None:
        from ..modules.ext_backbones import UtoniaFeatureExtractor

        if isinstance(extractor, UtoniaFeatureExtractor):
            pretrained_keys = getattr(extractor, "_pretrained_state_keys", frozenset())
            parameters = dict(extractor._model.named_parameters())
            allowed = {
                f"utonia_extractor._model.{key}"
                for key in pretrained_keys
                if key not in parameters or not parameters[key].requires_grad
            }

    required_missing = sorted(missing - allowed)
    if required_missing:
        raise RuntimeError(
            "Decoder checkpoint is missing required state; refusing to retain "
            "model initialization for: " + ", ".join(required_missing)
        )

    retained = tuple(sorted(missing))
    merged = dict(state_dict)
    merged.update({key: current_state[key] for key in retained})
    decoder.load_state_dict(merged, strict=True)
    return retained
