import importlib

__attributes = {
    "Trellis2ImageTo3DPipeline": "trellis2_image_to_3d",
}

__submodules = ["samplers", "rembg"]

__all__ = list(__attributes.keys()) + __submodules


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str):


    import os
    import json

    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download

        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, "r") as f:
        config = json.load(f)
    return __getattr__(config["name"]).from_pretrained(path)


if __name__ == "__main__":
    from . import samplers, rembg
    from .trellis2_image_to_3d import Trellis2ImageTo3DPipeline
