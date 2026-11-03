#!/usr/bin/env python3

import os
import sys
from pathlib import Path


for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mesh_eval.cli import main

if __name__ == "__main__":
    main()
