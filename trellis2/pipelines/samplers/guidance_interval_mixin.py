from typing import *


class GuidanceIntervalSamplerMixin:


    def _inference_model(
        self, model, x_t, t, cond, guidance_strength, guidance_interval, **kwargs
    ):
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return super()._inference_model(
                model, x_t, t, cond, guidance_strength=guidance_strength, **kwargs
            )
        else:
            return super()._inference_model(
                model, x_t, t, cond, guidance_strength=1, **kwargs
            )
