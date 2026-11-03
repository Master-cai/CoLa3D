


from typing import Tuple, Union
import torch
from torch import Tensor


def intrinsics_from_fov_xy(
    fov_x: Union[float, Tensor],
    fov_y: Union[float, Tensor],
) -> Tensor:

    import utils3d.torch as u3d

    return u3d.intrinsics_from_fov(fov_x=fov_x, fov_y=fov_y)


def image_uv(
    height: int,
    width: int,
    left: int = None,
    top: int = None,
    right: int = None,
    bottom: int = None,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:


    if left is None:
        left = 0
    if top is None:
        top = 0
    if right is None:
        right = width
    if bottom is None:
        bottom = height
    u = torch.linspace(
        (left + 0.5) / width,
        (right - 0.5) / width,
        right - left,
        device=device,
        dtype=dtype,
    )
    v = torch.linspace(
        (top + 0.5) / height,
        (bottom - 0.5) / height,
        bottom - top,
        device=device,
        dtype=dtype,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def get_rays(
    extrinsics: Tensor,
    intrinsics: Tensor,
    uv: Tensor,
) -> Tuple[Tensor, Tensor]:


    uvz = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1).to(extrinsics)
    with torch.amp.autocast("cuda", enabled=False):
        inv_transformation = (intrinsics @ extrinsics[..., :3, :3]).inverse()
        inv_extrinsics = extrinsics.inverse()
    rays_d = uvz @ inv_transformation.transpose(-1, -2)
    rays_o = inv_extrinsics[..., None, :3, 3]
    return rays_o, rays_d


def get_image_rays(
    extrinsics: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:


    uv = image_uv(height, width).to(extrinsics).flatten(0, 1)
    rays_o, rays_d = get_rays(extrinsics, intrinsics, uv)
    rays_o = rays_o.unflatten(-2, (1, 1))
    rays_d = rays_d.unflatten(-2, (height, width))
    return rays_o, rays_d


def patch():

    import utils3d.torch as u3d

    _patches = {
        "intrinsics_from_fov_xy": intrinsics_from_fov_xy,
        "image_uv": image_uv,
        "get_rays": get_rays,
        "get_image_rays": get_image_rays,
    }

    for name, fn in _patches.items():
        if not hasattr(u3d, name):
            setattr(u3d, name, fn)


patch()
