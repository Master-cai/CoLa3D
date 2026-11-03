

from __future__ import annotations

import contextlib
import time
from typing import Dict, Optional

import torch


@contextlib.contextmanager
def stage_timer(timings: Optional[Dict[str, float]], key: str):
    if timings is None:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings[key] = round(timings.get(key, 0.0) + (time.time() - t), 3)
