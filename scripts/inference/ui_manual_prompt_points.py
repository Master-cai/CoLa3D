

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

PART_COLORS_RGB: List[Tuple[int, int, int]] = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]
COORD_RESOLUTION = 64


def discover_instances(run_dir: Path, glob_pattern: str = "*") -> List[str]:

    out: List[str] = []
    for child in sorted(run_dir.glob(glob_pattern)):
        if not child.is_dir():
            continue
        if (child / "latents" / "shape_slat.pt").is_file():
            out.append(child.name)
    return out


def _load_shape_slat_coords(run_dir: Path, iid: str) -> np.ndarray:

    blob = torch.load(run_dir / iid / "latents" / "shape_slat.pt", map_location="cpu")
    coords = blob["coords"]
    if coords.shape[1] != 4:
        raise ValueError(
            f"shape_slat coords expected (N, 4) [b,x,y,z], got {tuple(coords.shape)}"
        )
    return coords[:, 1:].numpy().astype(np.int32)


def _load_segmentation_hint(
    run_dir: Path, iid: str, num_voxels: int
) -> Tuple[Optional[np.ndarray], Optional[int]]:

    raw_num_parts: Optional[int] = None
    scene_info_path = run_dir / iid / "scene_info.json"
    if scene_info_path.is_file():
        try:
            scene_info = json.loads(scene_info_path.read_text())
            candidate = int(scene_info.get("raw_num_parts", 0))
            raw_num_parts = candidate if candidate > 0 else None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            raw_num_parts = None

    p = run_dir / iid / "segmentation.npz"
    if not p.is_file():
        return None, raw_num_parts
    try:
        data = np.load(p)
    except Exception:
        return None, raw_num_parts
    if "primary_labels" not in data or "num_parts" not in data:
        return None, raw_num_parts
    pl = np.asarray(data["primary_labels"]).astype(np.int64)
    np_count = int(data["num_parts"])
    k_hint = raw_num_parts if raw_num_parts is not None else np_count
    if pl.shape[0] != num_voxels:

        return None, k_hint
    return pl, k_hint


def _load_existing_annotation(
    run_dir: Path, iid: str
) -> Tuple[List[Tuple[int, int, int]], int]:

    p = run_dir / iid / "manual_prompt_points.pt"
    if not p.is_file():
        return [], 0
    blob = torch.load(p, map_location="cpu")
    pp = blob["prompt_points"]
    vm = blob["valid_mask"]
    if pp.dim() != 3 or pp.shape[-1] != 3:
        return [], 0
    pp0 = pp[0]
    vm0 = vm[0]
    picked: List[Tuple[int, int, int]] = []
    for i in range(pp0.shape[0]):
        if bool(vm0[i].item()):
            x, y, z = int(pp0[i, 0]), int(pp0[i, 1]), int(pp0[i, 2])
            picked.append((x, y, z))
    k_hint = int(blob.get("metadata", {}).get("max_parts", pp0.shape[0]))
    return picked, k_hint


def save_annotation(
    run_dir: Path,
    iid: str,
    picked: List[Tuple[int, int, int]],
    max_parts: int,
    shape_slat_mtime: float,
) -> Path:

    if max_parts <= 0:
        raise ValueError(f"max_parts must be positive, got {max_parts}")
    if len(picked) > max_parts:
        raise ValueError(f"picked {len(picked)} points exceeds max_parts={max_parts}")
    pp = torch.zeros((1, max_parts, 3), dtype=torch.long)
    vm = torch.zeros((1, max_parts), dtype=torch.bool)
    for i, (x, y, z) in enumerate(picked):
        pp[0, i, 0] = int(x)
        pp[0, i, 1] = int(y)
        pp[0, i, 2] = int(z)
        vm[0, i] = True
    blob = {
        "prompt_points": pp,
        "valid_mask": vm,
        "object_ids": list(range(len(picked))),
        "metadata": {
            "tool": "manual",
            "version": 1,
            "num_picked": len(picked),
            "max_parts": int(max_parts),
            "coord_resolution": COORD_RESOLUTION,
            "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "shape_slat_mtime": shape_slat_mtime,
        },
    }
    out_path = run_dir / iid / "manual_prompt_points.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out_path)
    return out_path


def _make_blank_state() -> Dict[str, Any]:
    return {
        "iid": None,
        "coords": None,
        "primary_labels": None,
        "picked": [],
        "max_parts": 0,
        "shape_slat_mtime": 0.0,
        "k_hint": None,
    }


def _snap_to_existing_coord(
    coords: np.ndarray, xyz: Tuple[int, int, int]
) -> Tuple[int, int, int]:

    target = np.asarray(xyz, dtype=np.int32)
    diff = coords.astype(np.int32) - target[None, :]
    dist2 = np.sum(diff * diff, axis=1)
    nearest_idx = int(np.argmin(dist2))
    return tuple(int(c) for c in coords[nearest_idx])


def _parse_manual_point(text: str, coords: np.ndarray) -> Tuple[int, int, int]:

    clean = text.strip()
    if not clean:
        raise ValueError("empty point")

    nums = [int(v) for v in re.findall(r"-?\d+", clean)]
    if len(nums) == 1:
        idx = nums[0]
        if idx < 0 or idx >= coords.shape[0]:
            raise ValueError(f"idx={idx} outside [0, {coords.shape[0] - 1}]")
        return tuple(int(c) for c in coords[idx])

    if len(nums) >= 3:
        xyz = (nums[0], nums[1], nums[2])
        if any(v < 0 or v >= COORD_RESOLUTION for v in xyz):
            raise ValueError(f"xyz={xyz} outside [0, {COORD_RESOLUTION - 1}]^3")
        return _snap_to_existing_coord(coords, xyz)

    raise ValueError(f"could not parse point from {text!r}")


def decoder_point_capacity(config_path: Path) -> int:

    config = json.loads(config_path.read_text())
    decoder = config.get("models", {}).get("decoder", config)
    capacity = int(decoder["args"]["max_parts"])
    if capacity <= 0:
        raise ValueError("Decoder max_parts must be positive")
    return capacity


def resolve_point_capacity(
    run_dir: Path,
    iid: str,
    max_parts_default: Optional[int],
    annotation_capacity: int = 0,
) -> int:

    capacity = max_parts_default
    if capacity is None:
        metadata_path = run_dir / iid / "scene_info.json"
        if metadata_path.is_file():
            capacity = json.loads(metadata_path.read_text()).get("max_parts")
    if capacity is None and annotation_capacity > 0:
        capacity = annotation_capacity
    if capacity is None or int(capacity) <= 0:
        raise ValueError(
            f"No valid decoder capacity for {iid}; provide --decoder-config or --max-parts"
        )
    return int(capacity)


def load_state_for_instance(
    run_dir: Path, iid: str, max_parts_default: Optional[int]
) -> Dict[str, Any]:
    state = _make_blank_state()
    state["iid"] = iid
    state["coords"] = _load_shape_slat_coords(run_dir, iid)
    pl, k_hint = _load_segmentation_hint(
        run_dir, iid, num_voxels=state["coords"].shape[0]
    )
    state["primary_labels"] = pl
    state["k_hint"] = k_hint
    state["shape_slat_mtime"] = (
        (run_dir / iid / "latents" / "shape_slat.pt").stat().st_mtime
    )
    existing, ann_max = _load_existing_annotation(run_dir, iid)
    state["picked"] = existing
    capacity = resolve_point_capacity(run_dir, iid, max_parts_default, ann_max)
    state["max_parts"] = max(capacity, len(existing))
    return state


class ConnectionManager:


    def __init__(self) -> None:
        self._clients: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, iid: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients[iid].add(ws)

    async def disconnect(self, iid: str, ws: WebSocket) -> None:
        async with self._lock:
            self._clients[iid].discard(ws)

    async def broadcast(self, iid: str, payload: Dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._clients[iid])
        dead: List[WebSocket] = []
        for w in targets:
            try:
                await w.send_json(payload)
            except Exception:
                dead.append(w)
        if dead:
            async with self._lock:
                for w in dead:
                    self._clients[iid].discard(w)


def _serialize_instance(state: Dict[str, Any]) -> Dict[str, Any]:
    coords = state["coords"]
    pl = state["primary_labels"]
    return {
        "iid": state["iid"],
        "coords": coords.astype(np.int32).tolist(),
        "primary_labels": pl.astype(np.int64).tolist() if pl is not None else None,
        "picked": [list(p) for p in state["picked"]],
        "k": state["max_parts"],
        "k_hint": state["k_hint"],
        "num_voxels": int(coords.shape[0]),
    }


def _state_payload(state: Dict[str, Any], msg: Optional[str] = None) -> Dict[str, Any]:
    return {
        "type": "state",
        "picked": [list(p) for p in state["picked"]],
        "k": state["max_parts"],
        "k_hint": state["k_hint"],
        "msg": msg,
    }


def build_app(
    run_dir: Path,
    instance_ids: List[str],
    max_parts_default: Optional[int],
    monochrome: bool = False,
) -> FastAPI:
    if not instance_ids:
        raise SystemExit(
            f"No cached instances found under {run_dir}/*/latents/shape_slat.pt"
        )

    app = FastAPI(title="Manual CoLa3D Prompt-Point Picker")
    state_cache: Dict[str, Dict[str, Any]] = {}
    instance_set = set(instance_ids)
    manager = ConnectionManager()

    def _ensure_state(iid: str) -> Dict[str, Any]:
        if iid not in instance_set:
            raise HTTPException(status_code=404, detail=f"unknown instance {iid!r}")
        if iid not in state_cache:
            state_cache[iid] = load_state_for_instance(run_dir, iid, max_parts_default)
            if monochrome:
                state_cache[iid]["primary_labels"] = None
        return state_cache[iid]

    @app.get("/api/instances")
    def api_instances() -> Dict[str, Any]:
        return {
            "instance_ids": instance_ids,
            "run_dir": str(run_dir),
            "k_default": max_parts_default,
            "coord_resolution": COORD_RESOLUTION,
            "part_colors": PART_COLORS_RGB,
            "monochrome": monochrome,
        }

    @app.get("/api/instance/{iid}")
    def api_instance(iid: str) -> Dict[str, Any]:
        return _serialize_instance(_ensure_state(iid))

    @app.get("/api/cond/{iid}")
    def api_cond(iid: str):
        if iid not in instance_set:
            raise HTTPException(status_code=404, detail="unknown iid")
        p = run_dir / iid / "cond_image.png"
        if not p.is_file():
            raise HTTPException(status_code=404, detail="no cond_image.png")
        return FileResponse(str(p), media_type="image/png")

    @app.websocket("/ws/{iid}")
    async def ws_iid(ws: WebSocket, iid: str) -> None:
        if iid not in instance_set:
            await ws.close(code=1008)
            return
        state = _ensure_state(iid)
        await manager.connect(iid, ws)
        try:
            await ws.send_json(_state_payload(state))
            while True:
                data = await ws.receive_text()
                try:
                    msg = json.loads(data)
                except Exception:
                    continue
                ack: Optional[str] = None
                t = msg.get("type")
                if t == "click":
                    try:
                        idx = int(msg["idx"])
                    except Exception:
                        continue
                    n = state["coords"].shape[0]
                    if not (0 <= idx < n):
                        ack = f"idx {idx} out of range [0,{n - 1}]"
                    elif len(state["picked"]) >= state["max_parts"]:
                        ack = f"already at K={state['max_parts']} - undo or raise K"
                    else:
                        xyz = tuple(int(v) for v in state["coords"][idx])
                        state["picked"].append(xyz)
                elif t == "undo":
                    if state["picked"]:
                        state["picked"].pop()
                elif t == "clear":
                    state["picked"] = []
                elif t == "remove_at":
                    try:
                        i = int(msg["i"])
                    except Exception:
                        continue
                    if 0 <= i < len(state["picked"]):
                        state["picked"].pop(i)
                elif t == "set_k":
                    try:
                        k = max(1, int(msg["k"]))
                    except Exception:
                        k = state["max_parts"]
                    state["max_parts"] = k
                elif t == "manual_add":
                    try:
                        xyz = _parse_manual_point(
                            str(msg.get("text", "")), state["coords"]
                        )
                        if len(state["picked"]) >= state["max_parts"]:
                            ack = f"already at K={state['max_parts']}"
                        else:
                            state["picked"].append(xyz)
                            ack = f"added {xyz[0]},{xyz[1]},{xyz[2]}"
                    except Exception as e:
                        ack = f"add failed: {e}"
                elif t == "save":
                    try:
                        out_path = save_annotation(
                            run_dir=run_dir,
                            iid=iid,
                            picked=state["picked"],
                            max_parts=state["max_parts"],
                            shape_slat_mtime=state["shape_slat_mtime"],
                        )
                        ack = f"saved {len(state['picked'])} -> {out_path.name}"
                        await manager.broadcast(
                            iid, {"type": "saved", "path": str(out_path)}
                        )
                    except Exception as e:
                        ack = f"save failed: {e}"
                else:
                    continue

                await manager.broadcast(iid, _state_payload(state, msg=ack))
        except WebSocketDisconnect:
            pass
        finally:
            await manager.disconnect(iid, ws)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    return app


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Prompt-Point Picker</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
:root {
    --bg: #0a0a0c;
    --bg-pane: rgba(18, 18, 22, 0.78);
    --bg-pane-solid: #121216;
    --border: #2a2a30;
    --border-strong: #3d3d46;
    --fg: #e6e6ea;
    --fg-dim: #8a8a92;
    --fg-faint: #4a4a52;
    --accent: #f59e0b;
    --accent-dim: rgba(245, 158, 11, 0.22);
    --danger: #ef4444;
    --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
    --display: 'Space Mono', ui-monospace, SFMono-Regular, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; background: var(--bg); color: var(--fg); font-family: var(--mono); overflow: hidden; }
button, input, select { font: inherit; color: inherit; }

#stage {
    position: fixed; inset: 0;
    background: radial-gradient(ellipse at center, #101015 0%, #06060a 90%);
}
#canvas { width: 100%; height: 100%; display: block; cursor: crosshair; }
.label-layer { position: absolute; top: 0; left: 0; pointer-events: none; }
.picked-label {
    color: var(--accent);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    text-shadow: 0 0 6px rgba(0,0,0,0.9), 0 0 1px #000;
    transform: translate(6px, -6px);
    letter-spacing: 0.04em;
}

/* ── top bar ── */
#topbar {
    position: fixed; top: 0; left: 0; right: 0;
    height: 56px;
    display: flex; align-items: center; gap: 16px;
    padding: 0 20px;
    background: linear-gradient(to bottom, rgba(8,8,12,0.92), rgba(8,8,12,0.6) 70%, transparent);
    z-index: 10;
    border-bottom: 1px solid transparent;
    pointer-events: none;
}
#topbar > * { pointer-events: auto; }
#title {
    font-family: var(--display);
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--fg);
    border-left: 2px solid var(--accent);
    padding-left: 12px;
    margin-right: 8px;
}
#title small {
    display: block;
    font-size: 9px;
    letter-spacing: 0.3em;
    color: var(--fg-dim);
    margin-top: 2px;
}
.iid-control {
    display: flex; align-items: center; gap: 6px;
    background: var(--bg-pane);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 4px 6px;
    backdrop-filter: blur(8px);
}
.iid-control select {
    background: transparent;
    border: 0;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--fg);
    min-width: 240px;
    outline: none;
    cursor: pointer;
    padding: 4px 6px;
}
.iid-control select option { background: var(--bg-pane-solid); }
.nav-btn {
    background: transparent;
    border: 1px solid transparent;
    color: var(--fg-dim);
    font-family: var(--mono);
    font-size: 11px;
    padding: 4px 8px;
    cursor: pointer;
    border-radius: 2px;
    transition: color 120ms, border-color 120ms;
}
.nav-btn:hover { color: var(--fg); border-color: var(--border-strong); }

#status-pill {
    background: var(--bg-pane);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 6px 12px;
    font-size: 10px;
    letter-spacing: 0.1em;
    color: var(--fg-dim);
    backdrop-filter: blur(8px);
    font-variant-numeric: tabular-nums;
}
#status-pill .num { color: var(--fg); font-weight: 500; }
#status-pill .accent { color: var(--accent); font-weight: 700; }

#topbar .spacer { flex: 1; }

#save-btn {
    background: var(--accent);
    color: #0a0a0c;
    border: 0;
    padding: 9px 22px;
    font-family: var(--display);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    cursor: pointer;
    border-radius: 2px;
    transition: filter 120ms, transform 60ms;
}
#save-btn:hover { filter: brightness(1.1); }
#save-btn:active { transform: translateY(1px); }
#save-btn:disabled { background: var(--border); color: var(--fg-faint); cursor: not-allowed; filter: none; }

/* ── left drawer ── */
#drawer {
    position: fixed; top: 72px; left: 16px;
    width: 240px;
    background: var(--bg-pane);
    border: 1px solid var(--border);
    backdrop-filter: blur(10px);
    z-index: 9;
    display: flex; flex-direction: column;
    padding: 14px;
    gap: 12px;
    border-radius: 2px;
    transition: opacity 200ms cubic-bezier(.2,.8,.2,1), transform 200ms cubic-bezier(.2,.8,.2,1);
}
#drawer .label {
    font-size: 9px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--fg-dim);
    margin-bottom: 4px;
}
#cond-wrap {
    width: 100%; aspect-ratio: 1 / 1;
    background: #050507;
    border: 1px solid var(--border);
    overflow: hidden;
    position: relative;
}
#cond-img {
    width: 100%; height: 100%;
    object-fit: contain;
    display: block;
}
#cond-wrap.empty::after {
    content: "no cond_image.png";
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; color: var(--fg-faint);
    letter-spacing: 0.1em;
}
.k-row { display: flex; gap: 6px; align-items: center; }
.k-row input {
    width: 56px;
    background: #050507;
    border: 1px solid var(--border);
    padding: 6px 8px;
    text-align: center;
    color: var(--fg);
    font-variant-numeric: tabular-nums;
    border-radius: 2px;
}
.k-row input:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
.k-row .hint { font-size: 9px; color: var(--fg-faint); letter-spacing: 0.1em; }

#manual-text {
    width: 100%;
    background: #050507;
    border: 1px solid var(--border);
    padding: 7px 9px;
    color: var(--fg);
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 11px;
}
#manual-text:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
#manual-text::placeholder { color: var(--fg-faint); }

.row-btns { display: flex; gap: 6px; }
.row-btns button {
    flex: 1;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-dim);
    padding: 7px;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    cursor: pointer;
    border-radius: 2px;
    transition: color 120ms, border-color 120ms, background 120ms;
}
.row-btns button:hover { color: var(--fg); border-color: var(--border-strong); }
.row-btns button.danger:hover { color: var(--danger); border-color: var(--danger); background: rgba(239,68,68,0.06); }

/* ── HUD top-right ── */
#hud {
    position: fixed; top: 72px; right: 16px;
    background: var(--bg-pane);
    border: 1px solid var(--border);
    backdrop-filter: blur(10px);
    padding: 10px 14px;
    font-size: 12px;
    color: var(--fg);
    border-radius: 2px;
    z-index: 9;
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.05em;
    transition: opacity 120ms;
    pointer-events: none;
    min-width: 280px;
    text-align: right;
}
.hud-key {
    color: var(--fg-faint);
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-right: 4px;
    margin-left: 6px;
}
.hud-key.accent { color: var(--accent); }

/* ── picked list bottom-right ── */
#picked-panel {
    position: fixed; bottom: 16px; right: 16px;
    width: 280px;
    max-height: 50vh;
    background: var(--bg-pane);
    border: 1px solid var(--border);
    backdrop-filter: blur(10px);
    z-index: 9;
    display: flex; flex-direction: column;
    border-radius: 2px;
}
#picked-panel .head {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    font-size: 9px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--fg-dim);
    display: flex; justify-content: space-between; align-items: center;
}
#picked-panel .head .count {
    color: var(--accent);
    font-family: var(--display);
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.1em;
}
#picked-list {
    list-style: none;
    overflow-y: auto;
    max-height: calc(50vh - 32px);
}
#picked-list li {
    display: grid;
    grid-template-columns: 36px 1fr 28px;
    align-items: center;
    padding: 7px 12px;
    font-size: 11px;
    border-bottom: 1px solid #1a1a1f;
    font-variant-numeric: tabular-nums;
    transition: background 100ms;
}
#picked-list li:last-child { border-bottom: 0; }
#picked-list li:hover { background: rgba(245, 158, 11, 0.04); }
#picked-list .ord {
    color: var(--accent);
    font-weight: 700;
    letter-spacing: 0.05em;
}
#picked-list .coord { color: var(--fg); letter-spacing: 0.04em; }
#picked-list .x {
    background: transparent;
    border: 0;
    color: var(--fg-faint);
    font-size: 16px;
    line-height: 1;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 2px;
}
#picked-list .x:hover { color: var(--danger); background: rgba(239,68,68,0.08); }
#picked-list .empty {
    text-align: center;
    padding: 20px;
    color: var(--fg-faint);
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ── toast bottom-center ── */
#toast {
    position: fixed; bottom: 24px; left: 50%;
    transform: translateX(-50%) translateY(20px);
    background: var(--bg-pane-solid);
    border: 1px solid var(--accent);
    color: var(--fg);
    padding: 9px 18px;
    font-size: 11px;
    letter-spacing: 0.06em;
    border-radius: 2px;
    z-index: 20;
    opacity: 0;
    transition: opacity 200ms, transform 200ms cubic-bezier(.2,.8,.2,1);
    pointer-events: none;
    box-shadow: 0 0 24px rgba(245, 158, 11, 0.2);
}
#toast.show {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
}

/* ── corner crosshair ticks (Blender feel) ── */
.crosshair {
    position: fixed;
    width: 20px; height: 20px;
    border: 0 solid var(--border-strong);
    pointer-events: none;
    z-index: 5;
}
.crosshair.tl { top: 64px; left: 0; border-top-width: 1px; border-left-width: 1px; }
.crosshair.tr { top: 64px; right: 0; border-top-width: 1px; border-right-width: 1px; }
.crosshair.bl { bottom: 0; left: 0; border-bottom-width: 1px; border-left-width: 1px; }
.crosshair.br { bottom: 0; right: 0; border-bottom-width: 1px; border-right-width: 1px; }

/* ── conn pip ── */
#conn {
    position: fixed; bottom: 16px; left: 16px;
    font-size: 9px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--fg-faint);
    z-index: 8;
    display: flex; align-items: center; gap: 6px;
}
#conn .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--fg-faint);
    transition: background 200ms, box-shadow 200ms;
}
#conn.live .dot { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
#conn.live { color: #22c55e; }
</style>
</head>
<body>
<div id="stage">
    <canvas id="canvas"></canvas>
</div>

<div class="crosshair tl"></div>
<div class="crosshair tr"></div>
<div class="crosshair bl"></div>
<div class="crosshair br"></div>

<header id="topbar">
    <div id="title">Prompt-Points<small>manual picker · 64³ latent grid</small></div>

    <div class="iid-control">
        <button id="prev-btn" class="nav-btn" title="previous instance (←)">◀</button>
        <select id="iid-select"></select>
        <button id="next-btn" class="nav-btn" title="next instance (→)">▶</button>
    </div>

    <div id="status-pill">
        <span>INST</span> <span class="num" id="st-cur">0</span>/<span id="st-tot">0</span>
        &nbsp;·&nbsp;
        <span>VOX</span> <span class="num" id="st-vox">—</span>
        &nbsp;·&nbsp;
        <span>K</span> <span class="num" id="st-k">—</span>
        &nbsp;·&nbsp;
        <span>PICKED</span> <span class="accent" id="st-picked">0</span>
    </div>

    <div class="spacer"></div>

    <button id="save-btn" title="save manual_prompt_points.pt (S)">Save</button>
</header>

<aside id="drawer">
    <div>
        <div class="label">Cond Image</div>
        <div id="cond-wrap" class="empty"><img id="cond-img" alt="cond_image" style="display:none" /></div>
    </div>
    <div>
        <div class="label">K · max objects</div>
        <div class="k-row">
            <input id="k-input" type="number" min="1" max="64" step="1" />
            <span class="hint" id="k-hint-txt">hint: —</span>
        </div>
    </div>
    <div>
        <div class="label">Add by xyz / idx (fallback)</div>
        <input id="manual-text" type="text" placeholder="12,34,56  or  idx=1234" />
    </div>
    <div class="row-btns">
        <button id="undo-btn" title="undo last (U)">Undo</button>
        <button id="clear-btn" class="danger" title="clear all">Clear</button>
    </div>
</aside>

<div id="hud" style="opacity:0.4;">— hover a voxel —</div>

<section id="picked-panel">
    <div class="head">
        <span>Picked Points</span>
        <span class="count"><span id="head-cur">0</span>/<span id="head-max">0</span></span>
    </div>
    <ol id="picked-list"></ol>
</section>

<div id="conn"><span class="dot"></span><span id="conn-txt">connecting</span></div>
<div id="toast"></div>

<script type="importmap">
{
    "imports": {
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
    }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const RES = 64;
const CENTER = (RES - 1) / 2;

// Per-pick palette: each picked point #i gets its own color (3D point,
// #i label and the right-rail list row all stay in sync). Hand-tuned for
// high contrast on a near-black background and visually distinct from
// PART_COLORS_RGB (which uses the muted tab10 palette for voxels).
const PICKED_COLORS = [
    [245, 158,  11],  // amber
    [239,  68,  68],  // red
    [236,  72, 153],  // pink
    [217,  70, 239],  // fuchsia
    [168,  85, 247],  // purple
    [ 59, 130, 246],  // blue
    [  6, 182, 212],  // cyan
    [ 16, 185, 129],  // emerald
    [132, 204,  22],  // lime
    [234, 179,   8],  // yellow
    [249, 115,  22],  // orange
    [244, 114, 182],  // rose
];
function pickedColor(i) {
    if (cfg && cfg.monochrome) return [255, 255, 255];
    return PICKED_COLORS[((i % PICKED_COLORS.length) + PICKED_COLORS.length) % PICKED_COLORS.length];
}
function pickedColorCss(i) {
    const c = pickedColor(i);
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

// ── DOM refs ──
const canvas = document.getElementById('canvas');
const stage = document.getElementById('stage');
const dropdown = document.getElementById('iid-select');
const saveBtn = document.getElementById('save-btn');
const undoBtn = document.getElementById('undo-btn');
const clearBtn = document.getElementById('clear-btn');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const kInput = document.getElementById('k-input');
const kHintTxt = document.getElementById('k-hint-txt');
const manualText = document.getElementById('manual-text');
const condImg = document.getElementById('cond-img');
const condWrap = document.getElementById('cond-wrap');
const hud = document.getElementById('hud');
const pickedListEl = document.getElementById('picked-list');
const headCur = document.getElementById('head-cur');
const headMax = document.getElementById('head-max');
const stCur = document.getElementById('st-cur');
const stTot = document.getElementById('st-tot');
const stVox = document.getElementById('st-vox');
const stK = document.getElementById('st-k');
const stPicked = document.getElementById('st-picked');
const conn = document.getElementById('conn');
const connTxt = document.getElementById('conn-txt');
const toastEl = document.getElementById('toast');

// ── App state ──
let cfg = null;             // /api/instances response (cached partColors, ids)
let curIid = null;
let curIdx = 0;
let coords = null;          // Float32Array (N*3)
let labels = null;          // Int32Array (N) or null
let picked = [];            // [[x,y,z], ...]
let kVal = 8;
let kHint = null;
let ws = null;
let savedCameraState = null;

// ── three.js scene ──
const scene = new THREE.Scene();
scene.background = null;

const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 5000);
camera.up.set(0, 0, 1);
camera.position.set(CENTER + 90, -90, CENTER + 90);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);

const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.className = 'label-layer';
stage.appendChild(labelRenderer.domElement);

const controls = new OrbitControls(camera, canvas);
controls.target.set(CENTER, CENTER, CENTER);
controls.enableDamping = true;
controls.dampingFactor = 0.09;
controls.zoomSpeed = 0.8;
controls.rotateSpeed = 0.7;

// Wireframe cube outlining the [0, RES-1]^3 grid
{
    const cubeGeom = new THREE.BoxGeometry(RES - 1, RES - 1, RES - 1);
    const wf = new THREE.LineSegments(
        new THREE.EdgesGeometry(cubeGeom),
        new THREE.LineBasicMaterial({ color: 0x222226 })
    );
    wf.position.set(CENTER, CENTER, CENTER);
    scene.add(wf);
}
// Axes at origin
{
    const ax = new THREE.AxesHelper(8);
    scene.add(ax);
}

// ── Materials ──
const voxelMat = new THREE.PointsMaterial({
    size: 1.5,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.92,
    depthWrite: true,
});
const pickedMat = new THREE.PointsMaterial({
    size: 4.2,
    color: 0xffffff,        // white base; vertexColors below paints per-pick hue
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 1.0,
    depthTest: false,        // always on top
});
const highlightMat = new THREE.PointsMaterial({
    size: 5.0,
    color: 0xffffff,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.85,
    depthTest: false,
});

let voxelPoints = null;
let pickedPoints = null;
let pickedLabelObjs = [];
let highlightDot = null;

const raycaster = new THREE.Raycaster();
raycaster.params.Points = { threshold: 0.65 };
const ndc = new THREE.Vector2();
let hoveredIdx = -1;
let dragStart = null;

// ── Render loop ──
function fitCanvas() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false);
    labelRenderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}
new ResizeObserver(fitCanvas).observe(stage);
fitCanvas();

function tick() {
    requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
}
tick();

// ── Voxel rendering ──
function rebuildVoxels() {
    if (voxelPoints) {
        scene.remove(voxelPoints);
        voxelPoints.geometry.dispose();
        voxelPoints = null;
    }
    if (!coords) return;
    const n = coords.length / 3;
    const colors = new Float32Array(n * 3);
    const palette = cfg.part_colors;
    for (let i = 0; i < n; i++) {
        let r = 0.45, g = 0.45, b = 0.5;
        if (labels) {
            const lab = labels[i];
            if (lab >= 0) {
                const c = palette[((lab % palette.length) + palette.length) % palette.length];
                r = c[0] / 255; g = c[1] / 255; b = c[2] / 255;
            }
        }
        colors[i * 3] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(coords, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geom.computeBoundingSphere();
    voxelPoints = new THREE.Points(geom, voxelMat);
    voxelPoints.renderOrder = 0;
    scene.add(voxelPoints);
}

function rebuildPicked() {
    if (pickedPoints) {
        scene.remove(pickedPoints);
        pickedPoints.geometry.dispose();
        pickedPoints = null;
    }
    for (const o of pickedLabelObjs) scene.remove(o);
    pickedLabelObjs = [];

    if (picked.length > 0) {
        const positions = new Float32Array(picked.length * 3);
        const colors = new Float32Array(picked.length * 3);
        picked.forEach((p, i) => {
            positions[i * 3] = p[0];
            positions[i * 3 + 1] = p[1];
            positions[i * 3 + 2] = p[2];
            const c = pickedColor(i);
            colors[i * 3]     = c[0] / 255;
            colors[i * 3 + 1] = c[1] / 255;
            colors[i * 3 + 2] = c[2] / 255;
        });
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        pickedPoints = new THREE.Points(geom, pickedMat);
        pickedPoints.renderOrder = 10;
        scene.add(pickedPoints);

        picked.forEach((p, i) => {
            const div = document.createElement('div');
            div.className = 'picked-label';
            div.textContent = '#' + i;
            div.style.color = pickedColorCss(i);
            const obj = new CSS2DObject(div);
            obj.position.set(p[0], p[1], p[2]);
            scene.add(obj);
            pickedLabelObjs.push(obj);
        });
    }
    renderPickedList();
    renderStatus();
}

function setHighlight(idx) {
    if (highlightDot) {
        scene.remove(highlightDot);
        highlightDot.geometry.dispose();
        highlightDot = null;
    }
    if (idx < 0 || !coords) return;
    const x = coords[idx * 3], y = coords[idx * 3 + 1], z = coords[idx * 3 + 2];
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([x, y, z]), 3));
    highlightDot = new THREE.Points(geom, highlightMat);
    highlightDot.renderOrder = 20;
    scene.add(highlightDot);
}

// ── Pointer ──
canvas.addEventListener('pointerdown', (e) => {
    dragStart = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener('pointerup', (e) => {
    if (!dragStart) return;
    const dx = Math.abs(e.clientX - dragStart.x);
    const dy = Math.abs(e.clientY - dragStart.y);
    dragStart = null;
    if (dx + dy > 4) return; // drag, not click
    if (hoveredIdx < 0) return;
    send({ type: 'click', idx: hoveredIdx });
});

let pointerScheduled = false;
let lastPointer = null;
canvas.addEventListener('pointermove', (e) => {
    lastPointer = e;
    if (!pointerScheduled) {
        pointerScheduled = true;
        requestAnimationFrame(processPointer);
    }
});
canvas.addEventListener('pointerleave', () => {
    hoveredIdx = -1;
    setHighlight(-1);
    hud.style.opacity = '0.4';
    hud.textContent = '— hover a voxel —';
});

function processPointer() {
    pointerScheduled = false;
    if (!voxelPoints || !lastPointer) return;
    const e = lastPointer;
    const rect = canvas.getBoundingClientRect();
    ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObject(voxelPoints, false);
    if (hits.length === 0) {
        hoveredIdx = -1;
        setHighlight(-1);
        hud.style.opacity = '0.4';
        hud.textContent = '— hover a voxel —';
        return;
    }
    hits.sort((a, b) => a.distanceToRay - b.distanceToRay);
    const idx = hits[0].index;
    hoveredIdx = idx;
    setHighlight(idx);
    const x = coords[idx * 3], y = coords[idx * 3 + 1], z = coords[idx * 3 + 2];
    const lab = labels ? labels[idx] : -1;
    const labStr = lab >= 0 ? `part_${lab}` : 'unassigned';
    hud.style.opacity = '1';
    hud.innerHTML =
        `<span class="hud-key">idx</span>${String(idx).padStart(5)}` +
        `<span class="hud-key">x</span>${String(x).padStart(2)}` +
        `<span class="hud-key">y</span>${String(y).padStart(2)}` +
        `<span class="hud-key">z</span>${String(z).padStart(2)}` +
        `<span class="hud-key accent">${labStr}</span>`;
}

// ── WebSocket ──
function setConn(state, txt) {
    conn.classList.toggle('live', state === 'live');
    connTxt.textContent = txt;
}

function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(obj));
    }
}

let wsRetryTimer = null;
function openWs(iid) {
    if (ws) {
        ws.onclose = null;
        ws.onerror = null;
        try { ws.close(); } catch {}
        ws = null;
    }
    if (wsRetryTimer) { clearTimeout(wsRetryTimer); wsRetryTimer = null; }
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    setConn('pending', 'connecting');
    ws = new WebSocket(`${proto}://${location.host}/ws/${encodeURIComponent(iid)}`);
    ws.onopen = () => setConn('live', `live · ${iid.slice(0, 18)}`);
    ws.onmessage = (e) => {
        let msg;
        try { msg = JSON.parse(e.data); } catch { return; }
        if (msg.type === 'state') {
            picked = msg.picked || [];
            kVal = msg.k;
            kHint = msg.k_hint;
            kInput.value = kVal;
            kHintTxt.textContent = (kHint != null) ? `hint: K=${kHint}` : 'hint: —';
            rebuildPicked();
            if (msg.msg) toast(msg.msg);
        } else if (msg.type === 'saved') {
            // state will follow with msg=saved...; nothing extra to do here
        }
    };
    ws.onclose = () => {
        setConn('down', 'disconnected · retrying');
        wsRetryTimer = setTimeout(() => { if (curIid === iid) openWs(iid); }, 1200);
    };
}

// ── Toast ──
let toastTimer = null;
function toast(text) {
    toastEl.textContent = text;
    toastEl.classList.add('show');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove('show'), 2500);
}

// ── Picked list / status ──
function renderPickedList() {
    pickedListEl.innerHTML = '';
    if (picked.length === 0) {
        const li = document.createElement('li');
        li.className = 'empty';
        li.textContent = 'nothing picked yet';
        pickedListEl.appendChild(li);
        return;
    }
    picked.forEach((p, i) => {
        const li = document.createElement('li');
        const hue = pickedColorCss(i);
        li.innerHTML =
            `<span class="ord" style="color:${hue}">#${i}</span>` +
            `<span class="coord">${p[0]}, ${p[1]}, ${p[2]}</span>` +
            `<button class="x" data-i="${i}" title="remove">×</button>`;
        pickedListEl.appendChild(li);
    });
}

pickedListEl.addEventListener('click', (e) => {
    const btn = e.target.closest('button.x');
    if (!btn) return;
    const i = parseInt(btn.dataset.i, 10);
    if (Number.isFinite(i)) send({ type: 'remove_at', i });
});

function renderStatus() {
    stCur.textContent = String(curIdx + 1);
    stTot.textContent = String(cfg ? cfg.instance_ids.length : 0);
    stVox.textContent = coords ? String(coords.length / 3) : '—';
    stK.textContent = String(kVal);
    stPicked.textContent = `${picked.length}/${kVal}`;
    headCur.textContent = String(picked.length);
    headMax.textContent = String(kVal);
    saveBtn.disabled = picked.length === 0;
}

// ── Instance loading ──
async function loadInstance(iid) {
    if (!cfg) return;
    const idx = cfg.instance_ids.indexOf(iid);
    if (idx < 0) return;
    curIid = iid;
    curIdx = idx;
    dropdown.value = iid;

    // cond image: try to load directly; onerror keeps the "no cond" placeholder
    condImg.style.display = 'none';
    condWrap.classList.add('empty');
    condImg.onload = () => {
        condImg.style.display = 'block';
        condWrap.classList.remove('empty');
    };
    condImg.onerror = () => {
        condImg.style.display = 'none';
        condWrap.classList.add('empty');
    };
    condImg.src = `/api/cond/${encodeURIComponent(iid)}?t=` + Date.now();

    // instance data
    const r = await fetch(`/api/instance/${encodeURIComponent(iid)}`);
    if (!r.ok) { toast('failed to load ' + iid); return; }
    const data = await r.json();
    const flat = new Float32Array(data.coords.length * 3);
    data.coords.forEach((c, i) => {
        flat[i * 3] = c[0];
        flat[i * 3 + 1] = c[1];
        flat[i * 3 + 2] = c[2];
    });
    coords = flat;
    labels = data.primary_labels ? Int32Array.from(data.primary_labels) : null;
    rebuildVoxels();

    // open ws (state msg will populate picked + k)
    openWs(iid);
    renderStatus();
}

// ── Buttons / inputs ──
saveBtn.addEventListener('click', () => send({ type: 'save' }));
undoBtn.addEventListener('click', () => send({ type: 'undo' }));
clearBtn.addEventListener('click', () => {
    if (picked.length === 0) return;
    if (confirm('Clear all picked points for this scene?')) send({ type: 'clear' });
});
prevBtn.addEventListener('click', () => {
    if (curIdx > 0) loadInstance(cfg.instance_ids[curIdx - 1]);
});
nextBtn.addEventListener('click', () => {
    if (curIdx < cfg.instance_ids.length - 1) loadInstance(cfg.instance_ids[curIdx + 1]);
});
dropdown.addEventListener('change', () => loadInstance(dropdown.value));

kInput.addEventListener('change', () => {
    const k = Math.max(1, parseInt(kInput.value, 10) || 1);
    send({ type: 'set_k', k });
});

manualText.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        const text = manualText.value.trim();
        if (!text) return;
        send({ type: 'manual_add', text });
        manualText.value = '';
    }
});

// ── Keyboard shortcuts ──
window.addEventListener('keydown', (e) => {
    if (document.activeElement && ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    if (e.key === 'ArrowLeft') { prevBtn.click(); }
    else if (e.key === 'ArrowRight') { nextBtn.click(); }
    else if (e.key === 'u' || e.key === 'U') { undoBtn.click(); }
    else if (e.key === 's' || e.key === 'S') { saveBtn.click(); }
});

// ── Boot ──
async function boot() {
    try {
        const r = await fetch('/api/instances');
        if (!r.ok) throw new Error('boot fetch failed');
        cfg = await r.json();
    } catch (e) {
        toast('failed to load /api/instances');
        return;
    }
    dropdown.innerHTML = '';
    cfg.instance_ids.forEach((iid) => {
        const opt = document.createElement('option');
        opt.value = iid;
        opt.textContent = iid;
        dropdown.appendChild(opt);
    });
    if (cfg.instance_ids.length > 0) {
        await loadInstance(cfg.instance_ids[0]);
    }
}
boot();
</script>
</body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manual prompt-point picker for cached CoLa3D SLat coords (FastAPI + Three.js).",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Cached run directory containing <instance_id>/latents/shape_slat.pt",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18080,
        help="Server port (default 18080; cluster reserves 8080).",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=None,
        help="Override the decoder/prepare metadata prompt capacity.",
    )
    parser.add_argument(
        "--decoder-config",
        type=Path,
        default=None,
        help="Decoder training config used to infer the prompt capacity.",
    )
    parser.add_argument(
        "--display-stride",
        type=int,
        default=1,
        help="Accepted for launcher compatibility but currently unused; the "
        "Three.js renderer handles 7700-voxel scenes without subsampling.",
    )
    parser.add_argument(
        "--instance-glob",
        type=str,
        default="*",
        help="Optional glob for filtering instance subdirs (default: '*').",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Bind address (default 127.0.0.1; reverse tunnels connect locally).",
    )
    parser.add_argument(
        "--monochrome",
        action="store_true",
        help="Render all latent voxels in neutral gray and all picked points in white. "
        "Segmentation labels remain available for K hints but do not affect color.",
    )
    args = parser.parse_args(argv)
    if args.max_parts is not None and args.max_parts <= 0:
        parser.error("--max-parts must be positive")
    if args.max_parts is None and args.decoder_config is not None:
        args.max_parts = decoder_point_capacity(args.decoder_config)

    if not args.run_dir.is_dir():
        print(f"--run-dir not a directory: {args.run_dir}", file=sys.stderr)
        return 2

    instance_ids = discover_instances(args.run_dir, args.instance_glob)
    print(f"Discovered {len(instance_ids)} instances under {args.run_dir}", flush=True)
    if not instance_ids:
        print("No instances with cached shape_slat.pt found.", file=sys.stderr)
        return 1
    if args.display_stride != 1:
        print(
            f"NOTE: --display-stride={args.display_stride} ignored (Three.js renders all voxels).",
            file=sys.stderr,
        )

    app = build_app(
        args.run_dir,
        instance_ids,
        args.max_parts,
        monochrome=args.monochrome,
    )
    uvicorn.run(
        app,
        host=args.server_name,
        port=args.port,
        log_level="info",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
