import importlib

__attributes = {
    "FlexiDualGridDataset": "flexi_dual_grid",
    "FlexiDualGridMultiObjDataset": "flexi_dual_grid_multi_obj",
    "SingleImageDataset": "single_image",
    "SparseStructureLatent": "sparse_structure_latent",
    "ImageConditionedSparseStructureLatent": "sparse_structure_latent",
    "ImageConditionedSparseStructureLatent3DFront": "sparse_structure_latent",
    "SLat": "structured_latent",
    "ImageConditionedSLat": "structured_latent",
    "SLatShape": "structured_latent_shape",
    "ImageConditionedSLatShape": "structured_latent_shape",
    "ImageConditionedSLatShape3DFront": "structured_latent_shape",
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


if __name__ == "__main__":
    from .flexi_dual_grid import FlexiDualGridDataset

    from .sparse_structure_latent import (
        SparseStructureLatent,
        ImageConditionedSparseStructureLatent,
        ImageConditionedSparseStructureLatent3DFront,
    )
    from .structured_latent import SLat, ImageConditionedSLat
    from .structured_latent_shape import SLatShape, ImageConditionedSLatShape
