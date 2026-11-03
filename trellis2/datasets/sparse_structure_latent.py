import os
import json
from pathlib import Path
from typing import *
import numpy as np
import torch
from PIL import Image
from ..representations import Voxel
from ..renderers import VoxelRenderer
from .components import StandardDatasetBase, ImageConditionedMixin
from .. import models
from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


class SparseStructureLatentVisMixin:
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = "JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json",
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt

    def _loading_ss_dec(self):
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, "config.json"), "r"))
            decoder = getattr(models, cfg["models"]["decoder"]["name"])(
                **cfg["models"]["decoder"]["args"]
            )
            ckpt_path = os.path.join(
                self.ss_dec_path, "ckpts", f"decoder_{self.ss_dec_ckpt}.pt"
            )
            decoder.load_state_dict(
                torch.load(ckpt_path, map_location="cpu", weights_only=True)
            )
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_ss_dec()
        ss = []
        if self.normalization:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i : i + batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()
        return ss

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0["x_0"]
        x_0 = self.decode_latent(x_0.cuda())

        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4

        yaw = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        images = []

        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            color = coords / resolution
            rep = Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / resolution,
                coords=coords,
                attrs=color,
                layout={
                    "color": slice(0, 3),
                },
            )
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(rep, ext, intr, colors_overwrite=color)
                image[
                    :,
                    512 * (j // tile[1]) : 512 * (j // tile[1] + 1),
                    512 * (j % tile[1]) : 512 * (j % tile[1] + 1),
                ] = res["color"]
            images.append(image)

        return torch.stack(images)


class SparseStructureLatent(SparseStructureLatentVisMixin, StandardDatasetBase):

    def __init__(
        self,
        roots: str,
        *,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        instances: Optional[str] = None,
        max_samples: Optional[int] = None,
        sample_seed: int = 0,
        pretrained_ss_dec: str = "JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.instances = instances
        self.max_samples = max_samples
        self.sample_seed = sample_seed
        self.value_range = (0, 1)
        self._instance_whitelist = None

        if instances is not None:
            if os.path.exists(instances):
                with open(instances, "r") as f:
                    rows = [line.strip() for line in f.readlines()]
                self._instance_whitelist = set([x for x in rows if len(x) > 0])
            else:
                rows = [x.strip() for x in instances.split(",")]
                self._instance_whitelist = set([x for x in rows if len(x) > 0])

        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )

        self.loads = []
        load_keys = ["shape_latent_tokens", "dual_grid_size", "ss_latent_tokens"]
        for _, sha256 in self.instances:
            load_value = 1
            if sha256 in self.metadata.index:
                for key in load_keys:
                    if key not in self.metadata.columns:
                        continue
                    value = self.metadata.loc[sha256, key]
                    if hasattr(value, "iloc"):
                        value = value.iloc[0]
                    try:
                        numeric = float(value)
                        if np.isfinite(numeric) and numeric > 0:
                            load_value = int(numeric)
                            break
                    except Exception:
                        continue
            self.loads.append(load_value)

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization["mean"]).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization["std"]).reshape(-1, 1, 1, 1)

    def filter_metadata(self, metadata):
        stats = {}
        if self._instance_whitelist is not None:
            metadata = metadata[metadata.index.isin(self._instance_whitelist)]
            stats["Instance whitelist"] = len(metadata)
        metadata = metadata[metadata["ss_latent_encoded"] == True]
        stats["With latent"] = len(metadata)
        if (
            self.min_aesthetic_score is not None
            and "aesthetic_score" in metadata.columns
        ):
            metadata = metadata[metadata["aesthetic_score"] >= self.min_aesthetic_score]
            stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(metadata)
        if self.max_samples is not None and len(metadata) > self.max_samples:
            metadata = metadata.sample(
                n=self.max_samples, random_state=self.sample_seed
            )
            stats[f"Max samples <= {self.max_samples}"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        latent = np.load(os.path.join(root["ss_latent"], f"{instance}.npz"))
        z = torch.tensor(latent["z"]).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            "x_0": z,
        }
        return pack


class ImageConditionedSparseStructureLatent(
    ImageConditionedMixin, SparseStructureLatent
):

    pass


class ImageConditionedSparseStructureLatent3DFront(SparseStructureLatent):

    def __init__(
        self,
        roots: str,
        *,
        min_aesthetic_score: Optional[float] = None,
        normalization: Optional[dict] = None,
        instances: Optional[str] = None,
        max_samples: Optional[int] = None,
        sample_seed: int = 0,
        pretrained_ss_dec: str = "JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        render_root: Optional[Union[str, Dict[str, str]]] = None,
        image_size: int = 512,
        strict_cond_image: bool = True,
        skip_first_frame: Optional[Dict[str, bool]] = None,
        dataset_repeat: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        self.render_root = render_root
        self.image_size = image_size
        self.strict_cond_image = strict_cond_image
        self.skip_first_frame = skip_first_frame
        self.dataset_repeat = dataset_repeat
        self._render_file_cache: Dict[str, List[str]] = {}
        self._sha256_to_file_id: Dict[str, str] = {}
        super().__init__(
            roots,
            min_aesthetic_score=min_aesthetic_score,
            normalization=normalization,
            instances=instances,
            max_samples=max_samples,
            sample_seed=sample_seed,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
            **kwargs,
        )

        if self.render_root is not None and "file_identifier" in self.metadata.columns:
            for sha256 in self.metadata.index:
                file_id = self.metadata.loc[sha256, "file_identifier"]
                if isinstance(file_id, str) and len(file_id) > 0:
                    self._sha256_to_file_id[sha256] = file_id

        if self.dataset_repeat and isinstance(self.roots, dict):
            repeated_instances, repeated_loads = [], []
            for (root, sha256), load in zip(self.instances, self.loads):
                n = 1
                for key, root_dict in self.roots.items():
                    if root is root_dict:
                        n = self.dataset_repeat.get(key, 1)
                        break
                repeated_instances.extend([(root, sha256)] * n)
                repeated_loads.extend([load] * n)
            self.instances = repeated_instances
            self.loads = repeated_loads

        if self.render_root is not None:
            self._prewarm_render_file_cache()
            if self.strict_cond_image:
                before = len(self.instances)
                kept = [
                    (inst, load)
                    for inst, load in zip(self.instances, self.loads)
                    if len(self._render_file_cache.get(inst[1], [])) > 0
                ]
                dropped = before - len(kept)
                if dropped > 0:
                    self.instances = [k[0] for k in kept]
                    self.loads = [k[1] for k in kept]
                    print(
                        f"[{self.__class__.__name__}] Dropped {dropped}/{before} "
                        f"instances with empty render dir (image-cond needs a render)"
                    )

    def _resolve_render_root(self, root) -> Optional[str]:
        if isinstance(self.render_root, dict):
            if isinstance(self.roots, dict):
                for key, root_dict in self.roots.items():
                    if root is root_dict:
                        return self.render_root.get(key)
            return None
        return self.render_root

    def _should_skip_first_frame(self, root) -> bool:
        if not self.skip_first_frame or not isinstance(self.roots, dict):
            return False
        for key, root_dict in self.roots.items():
            if root is root_dict:
                return self.skip_first_frame.get(key, False)
        return False

    def _prewarm_render_file_cache(self):
        for root, sha256 in self.instances:
            if sha256 in self._render_file_cache:
                continue
            file_id = self._sha256_to_file_id.get(sha256)
            if file_id is None:
                self._render_file_cache[sha256] = []
                continue
            rr = self._resolve_render_root(root)
            if rr is None:
                self._render_file_cache[sha256] = []
                continue
            render_dir = Path(rr) / file_id
            if not render_dir.is_dir():
                self._render_file_cache[sha256] = []
            else:
                self._render_file_cache[sha256] = [
                    str(p)
                    for p in render_dir.glob("render_*")
                    if p.is_file()
                    and p.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg"}
                ]
        print(
            f"[{self.__class__.__name__}] Pre-warmed render cache for "
            f"{len(self._render_file_cache)} instances"
        )

    def _load_cond_image(self, instance: str, root=None) -> Optional[torch.Tensor]:
        render_root = self._resolve_render_root(root)
        if render_root is None:
            return None
        file_id = self._sha256_to_file_id.get(instance)
        if file_id is None:
            return None

        render_dir = Path(render_root) / file_id
        if instance not in self._render_file_cache:
            if not render_dir.is_dir():
                self._render_file_cache[instance] = []
            else:
                self._render_file_cache[instance] = [
                    str(p)
                    for p in render_dir.glob("render_*")
                    if p.is_file()
                    and p.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg"}
                ]

        view_candidates = self._render_file_cache[instance]
        if len(view_candidates) == 0:
            return None

        if self._should_skip_first_frame(root) and len(view_candidates) > 1:
            view_candidates = [
                p for p in view_candidates if "render_0000" not in os.path.basename(p)
            ]

        image_path = view_candidates[np.random.randint(0, len(view_candidates))]
        image = Image.open(image_path)
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        alpha = np.array(image.getchannel(3))
        nonzero = alpha.nonzero()
        if nonzero[0].size == 0:
            return None
        bbox = [nonzero[1].min(), nonzero[0].min(), nonzero[1].max(), nonzero[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_bbox = [
            int(center[0] - hsize),
            int(center[1] - hsize),
            int(center[0] + hsize),
            int(center[1] + hsize),
        ]
        image = image.crop(aug_bbox)
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )

        alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
        image = image.convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        image = image * alpha.unsqueeze(0)
        return image

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        cond = self._load_cond_image(instance, root)
        if self.render_root is not None and self.strict_cond_image and cond is None:
            raise RuntimeError(
                f"Missing cond image for instance={instance} under "
                f"render_root={self._resolve_render_root(root)}"
            )
        if cond is not None:
            pack["cond"] = cond
        return pack
