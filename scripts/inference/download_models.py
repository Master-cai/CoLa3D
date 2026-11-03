
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def ensure_file(path: Path, repo_id: str | None, filename: str, revision: str) -> Path:
    if path.is_file() and path.stat().st_size > 0:
        return path
    if path.exists() or path.is_symlink():
        raise ValueError(f"Invalid existing model file; inspect it before retrying: {path}")
    if not repo_id:
        raise FileNotFoundError(
            f"Missing {path}. Set COLA3D_MODEL_REPO to the Hugging Face model repository "
            "or place the checkpoint locally."
        )
    from huggingface_hub import hf_hub_download


    return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision,
                               local_dir=path.parents[len(Path(filename).parts) - 1]))


def ensure_models(config: dict[str, Any], needed: list[str], checkpoint_root: Path) -> None:
    registry = json.loads((ROOT / "configs/downloads.json").read_text())
    repo = os.environ.get("COLA3D_MODEL_REPO") or registry["model_repo"]
    revision = os.environ.get("COLA3D_MODEL_REVISION") or registry["model_revision"]
    for key in needed:
        name = config["checkpoints"][key]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Checkpoint filenames must be relative to checkpoint_root: {name}")
        ensure_file(checkpoint_root / name, repo, name, revision)

    pipeline = ROOT / config["pipeline_config"]
    if not pipeline.is_file():
        if pipeline != ROOT / "pretrained_ckpts/pipeline.json":
            raise FileNotFoundError(f"Provide the custom pipeline config locally: {pipeline}")
        ensure_file(pipeline, registry["base_repo"], "pipeline.json", registry["base_revision"])
    models = json.loads(pipeline.read_text())["args"]["models"]

    for key in ("shape_slat_decoder", "shape_slat_flow_model_1024"):
        name = models[key] + ".json"
        if not name.startswith("ckpts/") or ".." in Path(name).parts:
            raise ValueError(f"Unsupported base model config path: {name}")
        ensure_file(ROOT / "pretrained_ckpts" / name, registry["base_repo"], name, registry["base_revision"])
