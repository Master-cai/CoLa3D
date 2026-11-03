import os
import json
from pathlib import Path
from typing import *
import numpy as np
import torch
from PIL import Image
from .. import models
from .components import ImageConditionedMixin
from ..modules.sparse import SparseTensor
from .structured_latent import SLatVisMixin, SLat
from ..utils.render_utils import get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics


class SLatShapeVisMixin(SLatVisMixin):
    def _loading_slat_dec(self):
        if self.slat_dec is not None:
            return
        if self.slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.slat_dec_path, "config.json"), "r"))
            decoder = getattr(models, cfg["models"]["decoder"]["name"])(
                **cfg["models"]["decoder"]["args"]
            )
            ckpt_path = os.path.join(
                self.slat_dec_path, "ckpts", f"decoder_{self.slat_dec_ckpt}.pt"
            )
            decoder.load_state_dict(
                torch.load(ckpt_path, map_location="cpu", weights_only=True)
            )
        else:
            decoder = models.from_pretrained(self.pretrained_slat_dec)
        decoder.set_resolution(self.resolution)
        self.slat_dec = decoder.cuda().eval()

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[SparseTensor, dict]):
        x_0 = x_0 if isinstance(x_0, SparseTensor) else x_0["x_0"]
        reps = self.decode_latent(x_0.cuda())

        yaw = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        renderer = get_renderer(reps[0])
        images = []
        for representation in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                image[
                    :,
                    512 * (j // tile[1]) : 512 * (j // tile[1] + 1),
                    512 * (j % tile[1]) : 512 * (j % tile[1] + 1),
                ] = res["normal"]
            images.append(image)
        images = torch.stack(images)
        return images


class SLatShape(SLatShapeVisMixin, SLat):

    def __init__(
        self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        super().__init__(
            roots,
            min_aesthetic_score=min_aesthetic_score,
            max_tokens=max_tokens,
            latent_key="shape_latent",
            normalization=normalization,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )
        self.resolution = resolution


class ImageConditionedSLatShape(ImageConditionedMixin, SLatShape):

    pass


class ImageConditionedSLatShape3DFront(SLatShape):

    def __init__(
        self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: Optional[float] = None,
        max_tokens: int = 8192,
        normalization: Optional[dict] = None,
        instances: Optional[str] = None,
        pretrained_slat_dec: str = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        render_root: Optional[Union[str, Dict[str, str]]] = None,
        image_size: int = 512,
        strict_cond_image: bool = True,
        skip_first_frame: Optional[Dict[str, bool]] = None,
        dataset_repeat: Optional[Dict[str, int]] = None,
    ):
        self.render_root = render_root
        self.image_size = image_size
        self.strict_cond_image = strict_cond_image
        self.skip_first_frame = skip_first_frame
        self.dataset_repeat = dataset_repeat
        self._render_file_cache: Dict[str, List[str]] = {}
        self._sha256_to_file_id: Dict[str, str] = {}

        self._instance_whitelist = None
        if instances is not None:
            if Path(instances).is_file():
                with open(instances) as f:
                    self._instance_whitelist = set(
                        line.strip() for line in f if line.strip()
                    )
            else:
                self._instance_whitelist = set(
                    x.strip() for x in instances.split(",") if x.strip()
                )

        super().__init__(
            roots,
            resolution=resolution,
            min_aesthetic_score=min_aesthetic_score,
            max_tokens=max_tokens,
            normalization=normalization,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )

        if self.render_root is not None and "file_identifier" in self.metadata.columns:
            for sha256 in self.metadata.index:
                file_id = self.metadata.loc[sha256, "file_identifier"]
                if isinstance(file_id, str) and len(file_id) > 0:
                    self._sha256_to_file_id[sha256] = file_id

        if self.dataset_repeat and isinstance(self.roots, dict):
            original = list(zip(self.instances, self.loads))
            repeated_instances, repeated_loads = [], []
            for (root, sha256), load in original:
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
            f"[{self.__class__.__name__}] Pre-warmed render file cache for "
            f"{len(self._render_file_cache)} instances"
        )

    def filter_metadata(self, metadata):
        stats = {}

        if self._instance_whitelist is not None:
            metadata = metadata[metadata.index.isin(self._instance_whitelist)]
            stats["Instance whitelist"] = len(metadata)

        latent_col = f"{self.latent_key}_encoded"
        if latent_col in metadata.columns:
            metadata = metadata[metadata[latent_col] == True]
            stats["With latent"] = len(metadata)

        token_col = f"{self.latent_key}_tokens"
        if token_col in metadata.columns:
            metadata = metadata[metadata[token_col] <= self.max_tokens]
            stats[f"Num tokens <= {self.max_tokens}"] = len(metadata)

        if (
            "aesthetic_score" in metadata.columns
            and self.min_aesthetic_score is not None
        ):
            metadata = metadata[metadata["aesthetic_score"] >= self.min_aesthetic_score]
            stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(metadata)

        return metadata, stats

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

        alpha_t = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
        image = image.convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        image = image * alpha_t.unsqueeze(0)
        return image

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        cond = self._load_cond_image(instance, root=root)
        render_root = self._resolve_render_root(root)
        if render_root is not None and self.strict_cond_image and cond is None:
            raise RuntimeError(
                f"Missing cond image for instance={instance} under render_root={render_root}"
            )
        if cond is not None:
            pack["cond"] = cond
        return pack
