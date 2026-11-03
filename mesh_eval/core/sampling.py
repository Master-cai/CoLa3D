import hashlib
import os
import numpy as np
import trimesh
_ENV = 'MESH_EVAL_SAMPLE_SEED'

def sample_seed_base():
    raw = os.environ.get(_ENV)
    if raw is None or raw == '':
        return None
    try:
        return int(raw)
    except ValueError:
        return None

def _mesh_seed(mesh, base):
    h = hashlib.blake2b(digest_size=8)
    for arr in (getattr(mesh, 'vertices', None), getattr(mesh, 'faces', None)):
        if arr is None:
            continue
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return (base ^ int.from_bytes(h.digest(), 'little')) % 2 ** 32

def surface_points(mesh, count):
    base = sample_seed_base()
    seed = None if base is None else _mesh_seed(mesh, base)
    return trimesh.sample.sample_surface(mesh, count, seed=seed)[0]
