


import sys
import types
import logging
from itertools import accumulate
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _segment_csr_torch(
    src: torch.Tensor,
    indptr: torch.Tensor,
    reduce: str = "sum",
) -> torch.Tensor:


    reduce_map = {"sum": "sum", "mean": "mean", "min": "min", "max": "max"}
    torch_reduce = reduce_map.get(reduce, reduce)
    return torch.segment_reduce(src, reduce=torch_reduce, offsets=indptr, unsafe=True)


def _ensure_torch_scatter_mock() -> None:

    if "torch_scatter" in sys.modules:
        return
    try:
        import torch_scatter

        return
    except ImportError:
        pass

    mock = types.ModuleType("torch_scatter")
    mock.segment_csr = _segment_csr_torch
    mock.__version__ = "0.0.0-torch-compat"
    sys.modules["torch_scatter"] = mock
    logger.debug(
        "Injected torch_scatter compatibility shim (using torch.segment_reduce)"
    )


_MIN_WORLD_SPAN_FOR_1024: float = 10.24


class UtoniaFeatureExtractor(nn.Module):


    def __init__(
        self,
        model_name: str = "Pointcept/Utonia",
        world_span: float = 12.0,
        resolution: int = 1024,
        enable_flash: bool = True,
    ):
        super().__init__()

        assert world_span >= _MIN_WORLD_SPAN_FOR_1024, (
            f"world_span={world_span} would cause GridSample(0.01) hash "
            f"collisions at 1024^3 (needs >= {_MIN_WORLD_SPAN_FOR_1024})"
        )
        assert resolution > 0, f"resolution must be positive, got {resolution}"

        self.model_name = model_name
        self.world_span = float(world_span)
        self.resolution = int(resolution)

        _ensure_torch_scatter_mock()

        import utonia as _utonia

        logger.debug(
            "Loading Utonia model from: %s (world_span=%.3f, resolution=%d)",
            model_name,
            self.world_span,
            self.resolution,
        )


        custom_config = dict(
            enable_flash=enable_flash,
            enc_mode=True,
            traceable=True,
            shuffle_orders=False,
        )

        import time

        t0 = time.time()

        ckpt = _utonia.load("utonia", repo_id=model_name, ckpt_only=True)
        enc_channels = ckpt["config"]["enc_channels"]
        for k, v in custom_config.items():
            ckpt["config"][k] = v

        from utonia.model import PointTransformerV3

        self._model = PointTransformerV3(**ckpt["config"])
        self._model.load_state_dict(ckpt["state_dict"], strict=True)
        self._pretrained_state_keys = frozenset(self._model.state_dict())
        del ckpt

        logger.debug(f"[Utonia] Model loaded in {time.time() - t0:.1f}s")

        def _force_no_shuffle(m: nn.Module) -> None:
            if hasattr(m, "shuffle_orders"):
                m.shuffle_orders = False

        self._model.apply(_force_no_shuffle)

        for param in self._model.parameters():
            param.requires_grad = False
        self._model.eval()


        self._transform = _utonia.transform.default(scale=1.0)


        self._output_dim: int = int(
            enc_channels[-3] + enc_channels[-2] + enc_channels[-1]
        )
        self._bottleneck_dim: int = int(enc_channels[-1])
        logger.debug(
            f"[Utonia] output_dim={self._output_dim} "
            f"bottleneck_dim={self._bottleneck_dim} "
            f"(enc_channels={enc_channels})",
        )

    def _idx_to_world(self, idx_np: np.ndarray) -> np.ndarray:


        return (
            (idx_np.astype(np.float32) + 0.5) / self.resolution - 0.5
        ) * self.world_span

    @staticmethod
    def _unpool_features(point: Any) -> torch.Tensor:


        for _ in range(2):
            if "pooling_parent" not in point.keys():
                break
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent

        return point.feat

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def bottleneck_dim(self) -> int:

        return self._bottleneck_dim

    def train(self, mode: bool = True):

        super().train(mode)
        self._model.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        coords: torch.Tensor,
        batch_size: int,
        colors: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:


        device = coords.device
        batch_idx = coords[:, 0].long()
        xyz_idx = coords[:, 1:]


        counts = torch.bincount(batch_idx, minlength=batch_size)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).cpu().tolist()


        per_sample: List[Optional[Dict[str, Any]]] = []
        for b in range(batch_size):
            s, e = ptr[b], ptr[b + 1]
            if s == e:
                per_sample.append(None)
                continue
            idx_np = xyz_idx[s:e].cpu().numpy().astype(np.int64)
            world = self._idx_to_world(idx_np)
            if colors is not None:


                color_np = colors[s:e].detach().cpu().numpy().astype(np.float32) * 255.0
            else:
                color_np = np.zeros_like(world, dtype=np.float32)
            if normals is not None:
                normal_np = normals[s:e].detach().cpu().numpy().astype(np.float32)
            else:
                normal_np = np.zeros_like(world, dtype=np.float32)
            point_in = {
                "coord": world.astype(np.float32),
                "color": color_np,
                "normal": normal_np,
            }
            point_out = self._transform(point_in)
            for k, v in list(point_out.items()):
                if isinstance(v, torch.Tensor):
                    point_out[k] = v.to(device, non_blocking=True)
            per_sample.append(point_out)


        Ms: List[int] = [
            int(p["coord"].shape[0]) if p is not None else 0 for p in per_sample
        ]
        if sum(Ms) == 0:
            return torch.zeros(
                coords.shape[0],
                self._output_dim,
                device=device,
                dtype=torch.float32,
            )


        coord_list = [p["coord"] for p in per_sample if p is not None]
        feat_list = [p["feat"] for p in per_sample if p is not None]
        grid_coord_list = [p["grid_coord"] for p in per_sample if p is not None]

        coord_cat = torch.cat(coord_list, dim=0)
        feat_cat = torch.cat(feat_list, dim=0)
        grid_coord_cat = torch.cat(grid_coord_list, dim=0)

        offset = torch.tensor(Ms, device=device, dtype=torch.long).cumsum(0)

        data: Dict[str, Any] = {
            "coord": coord_cat,
            "feat": feat_cat,
            "grid_coord": grid_coord_cat,
            "offset": offset,
        }


        point = self._model(data)
        feat_ds_cat = self._unpool_features(point)


        out = torch.zeros(
            coords.shape[0],
            self._output_dim,
            device=device,
            dtype=feat_ds_cat.dtype,
        )
        m_cum = list(accumulate(Ms, initial=0))
        for b in range(batch_size):
            pb = per_sample[b]
            if pb is None:
                continue
            s, e = ptr[b], ptr[b + 1]
            ms, me = m_cum[b], m_cum[b + 1]
            out[s:e] = feat_ds_cat[ms:me][pb["inverse"]]
        return out


    _DIRECT_GRID_SIZE: float = 0.01

    @staticmethod
    def _center_shift(coord: np.ndarray) -> np.ndarray:

        cmin = coord.min(0)
        cmax = coord.max(0)
        shift = np.array(
            [(cmin[0] + cmax[0]) / 2, (cmin[1] + cmax[1]) / 2, cmin[2]],
            dtype=np.float32,
        )
        return coord - shift

    def _transform_direct(
        self,
        idx_np: np.ndarray,
        color_np: Optional[np.ndarray] = None,
        normal_np: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:


        gs = self._DIRECT_GRID_SIZE
        coord = idx_np.astype(np.float32) * gs
        grid_coord = idx_np.astype(np.int32)
        centered = self._center_shift(coord.copy())

        n = len(coord)
        color = (
            color_np.astype(np.float32)
            if color_np is not None
            else np.zeros((n, 3), dtype=np.float32)
        )
        normal = (
            normal_np.astype(np.float32)
            if normal_np is not None
            else np.zeros((n, 3), dtype=np.float32)
        )
        assert color.shape == (
            n,
            3,
        ), f"color shape mismatch: expected ({n}, 3), got {color.shape}"
        assert normal.shape == (
            n,
            3,
        ), f"normal shape mismatch: expected ({n}, 3), got {normal.shape}"
        feat = np.concatenate([centered, color, normal], axis=1)
        return {
            "coord": torch.from_numpy(centered).float(),
            "grid_coord": torch.from_numpy(grid_coord).int(),
            "feat": torch.from_numpy(feat).float(),
        }

    @torch.no_grad()
    def forward_level4(
        self,
        coords: torch.Tensor,
        batch_size: int,
        colors: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
    ) -> tuple:


        device = coords.device
        batch_idx = coords[:, 0].long()
        xyz_idx = coords[:, 1:]

        counts = torch.bincount(batch_idx, minlength=batch_size)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).cpu().tolist()


        transformed: List[Optional[Dict[str, Any]]] = []
        for b in range(batch_size):
            s, e = ptr[b], ptr[b + 1]
            if s == e:
                transformed.append(None)
                continue
            idx_np = xyz_idx[s:e].cpu().numpy().astype(np.int64)
            color_np = (
                colors[s:e].detach().cpu().numpy().astype(np.float32)
                if colors is not None
                else None
            )
            normal_np = (
                normals[s:e].detach().cpu().numpy().astype(np.float32)
                if normals is not None
                else None
            )
            td = self._transform_direct(idx_np, color_np=color_np, normal_np=normal_np)
            for k, v in list(td.items()):
                if isinstance(v, torch.Tensor):
                    td[k] = v.to(device, non_blocking=True)
            transformed.append(td)

        Ms = [int(t["coord"].shape[0]) if t is not None else 0 for t in transformed]
        if sum(Ms) == 0:
            empty_f = torch.zeros(0, self._bottleneck_dim, device=device)
            empty_gc = torch.zeros(0, 3, device=device, dtype=torch.int32)
            return empty_f, empty_gc

        coord_cat = torch.cat([t["coord"] for t in transformed if t is not None])
        feat_cat = torch.cat([t["feat"] for t in transformed if t is not None])
        gc_cat = torch.cat([t["grid_coord"] for t in transformed if t is not None])

        offset = torch.tensor(Ms, device=device, dtype=torch.long).cumsum(0)

        data: Dict[str, Any] = {
            "coord": coord_cat,
            "feat": feat_cat,
            "grid_coord": gc_cat,
            "offset": offset,
        }


        point = self._model(data)
        return point.feat, point.grid_coord
