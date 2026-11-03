import importlib

__attributes = {

    "SparseStructureEncoder": "sparse_structure_vae",
    "SparseStructureDecoder": "sparse_structure_vae",
    "SparseStructureFlowModel": "sparse_structure_flow",

    "SLatFlowModel": "structured_latent_flow",
    "ElasticSLatFlowModel": "structured_latent_flow",

    "SparseUnetVaeEncoder": "sc_vaes.sparse_unet_vae",
    "SparseUnetVaeDecoder": "sc_vaes.sparse_unet_vae",
    "FlexiDualGridVaeEncoder": "sc_vaes.fdg_vae",
    "FlexiDualGridVaeDecoder": "sc_vaes.fdg_vae",
    "DecomposableVaeDecoder": "sc_vaes.decomposable_vae",
}

__submodules = []

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


def from_pretrained(path: str, **kwargs):


    import os
    import json
    from safetensors.torch import load_file

    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        path_parts = path.split("/")
        repo_id = f"{path_parts[0]}/{path_parts[1]}"
        model_name = "/".join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, "r") as f:
        config = json.load(f)
    model = __getattr__(config["name"])(**config["args"], **kwargs)
    state = load_file(model_file)
    if (
        config["name"] == "SparseStructureFlowModel"
        and type(model) is __getattr__("SparseStructureFlowModel")
        and model.pe_mode == "rope"
        and "rope_phases" not in state
    ):

        state = dict(state)
        state["rope_phases"] = model.rope_phases
    model.load_state_dict(state, strict=True)

    return model


if __name__ == "__main__":
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel

    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
