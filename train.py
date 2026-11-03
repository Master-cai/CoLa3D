"""Train CoLa3D or fine-tune a generator/geometry decoder from a JSON config."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import random
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path, required=True
    )
    parser.add_argument("--load-dir", "--load_dir", dest="load_dir", type=Path)
    parser.add_argument(
        "--ckpt", default="latest", help="latest, none, or a saved step number"
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        help="Override config data_dir (path or JSON mapping)",
    )
    parser.add_argument("--seed", type=int, help="Override config seed (default: 0)")
    parser.add_argument("--max-steps", type=int, help="Override trainer max_steps")
    parser.add_argument(
        "--tryrun",
        action="store_true",
        help="Initialize models and data without training",
    )
    return parser.parse_args(argv)


def resolve_seed(config: dict[str, Any], cli_seed: int | None) -> int:
    seed = config.get("seed", 0) if cli_seed is None else cli_seed
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    return seed


def build_config_snapshot(
    config: dict[str, Any], roots: str, seed: int, world_size: int
) -> dict[str, Any]:
    snapshot = dict(config)
    snapshot["data_dir"] = json.loads(roots) if roots.startswith("{") else roots
    snapshot["seed"] = seed
    snapshot["world_size"] = world_size
    return snapshot


def find_checkpoint(
    load_dir: Path, checkpoint: str, *, require_existing: bool = False
) -> int | None:

    if checkpoint == "none":
        return None
    if checkpoint != "latest":
        if not checkpoint.isdecimal():
            raise ValueError(
                "--ckpt must be latest, none, or a nonnegative step number."
            )
        step = int(checkpoint)
        path = load_dir / "ckpts" / f"misc_step{step:07d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        return step
    steps = []
    for path in (load_dir / "ckpts").glob("misc_step*.pt"):
        suffix = path.stem.removeprefix("misc_step")
        if (
            path.is_file()
            and suffix.isdecimal()
            and path.name == f"misc_step{int(suffix):07d}.pt"
        ):
            steps.append(int(suffix))
    step = max(steps, default=None)
    if step is None and require_existing:
        raise FileNotFoundError(
            f'No resume checkpoint found in {load_dir / "ckpts"}. '
            "Correct --load-dir or use --ckpt none to start a new run."
        )
    return step


def is_json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and is_json_safe(item) for key, item in value.items()
        )
    return False


def main(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    load_dir = args.load_dir or args.output_dir
    step = find_checkpoint(
        load_dir, args.ckpt, require_existing=args.load_dir is not None
    )
    seed = resolve_seed(config, args.seed)

    import numpy as np
    import torch
    import torch.distributed as dist
    from trellis2 import datasets, models, trainers

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.multiprocessing.set_sharing_strategy("file_system")
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        roots = args.data_dir if args.data_dir is not None else config.get("data_dir")
        if roots is None:
            raise ValueError("Set data_dir in the config or pass --data-dir.")
        if isinstance(roots, dict):
            roots = json.dumps(roots)
        dataset = getattr(datasets, config["dataset"]["name"])(
            roots, **config["dataset"]["args"]
        )
        if len(dataset) == 0:
            raise ValueError("No training samples remain after dataset filtering.")
        if rank == 0:
            print(dataset)

        model_dict = {}
        for name, model_config in config["models"].items():
            cls = getattr(models, model_config["name"])
            explicit_args = model_config.get("args", {})
            defaults = {
                key: parameter.default
                for key, parameter in inspect.signature(cls.__init__).parameters.items()
                if key != "self"
                and parameter.default is not inspect.Parameter.empty
                and is_json_safe(parameter.default)
            }
            model_config["args"] = {**defaults, **explicit_args}
            model_dict[name] = cls(**explicit_args).cuda()
        if args.max_steps is not None:
            config["trainer"]["args"]["max_steps"] = args.max_steps
        if rank == 0:
            snapshot = build_config_snapshot(config, roots, seed, world_size)
            snapshot_path = args.output_dir / (
                "config.json" if step is None else f"config_resume_step{step:07d}.json"
            )
            snapshot_path.write_text(json.dumps(snapshot, indent=4) + "\n")
        trainer = getattr(trainers, config["trainer"]["name"])(
            model_dict,
            dataset,
            **config["trainer"]["args"],
            output_dir=str(args.output_dir),
            load_dir=str(load_dir),
            step=step,
        )
        if not args.tryrun:
            trainer.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
