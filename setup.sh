#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${COLA3D_ENV_PREFIX:-$ROOT/.env}"
BUILD_DIR="${COLA3D_BUILD_DIR:-$ROOT/.build}"
CUMESH_REV="${CUMESH_REV:-12289e1062f0603f2f0d0771b02e1395d247f26f}"
NVDIFFREC_REV="${NVDIFFREC_REV:-b296927cc7fd01c2ac1087c8065c4d7248f72da4}"
EIGEN_REV="21e4582d1739107337a03460c81412981130373e"
NEW_ENV=0 CHECK=0 PREFLIGHT=0 BASIC=0 NATIVE=0 UTONIA=0 MOGE=0 UI=0 DATA=0 EVAL=0
FLASH_ATTN=0 NVDIFFRAST=0 NVDIFFREC=0 CUMESH=0 FLEXGEMM=0 OVOXEL=0
usage() {
    cat <<'EOF'
Usage: ./setup.sh [OPTIONS]
  --new-env       Create or reuse COLA3D_ENV_PREFIX (default: .env), Python 3.11
  --all           Install training/inference stack: basic, native, Utonia
  --basic         Install PyTorch 2.7.1 / CUDA 12.8 and Python dependencies
  --native        Build FlashAttention, nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, O-Voxel
  --utonia        Install Utonia and its sparse-point dependencies
  --data          Install data-preparation dependencies (Blender installed separately)
  --eval          Install benchmark evaluation dependencies (Open3D, probreg)
  --ui            Install the browser point-picker dependencies
  --moge          Install optional MoGe point prediction
  --preflight     Check Python/build prerequisites without installing anything
  --check         Check installed package metadata without allocating a GPU
  -h, --help      Show this message

Individual native flags: --flash-attn --nvdiffrast --nvdiffrec --cumesh --flexgemm --o-voxel
Requires Linux, a CUDA 12.8 toolkit and a compatible C++ compiler. Conda is only
required with --new-env. Otherwise activate your Python 3.11 environment first.
COLA3D_BUILD_DIR selects persistent build storage (default: .build).
MAX_JOBS limits native build parallelism (default: 4).
CUMESH_REV and NVDIFFREC_REV override the pinned source commits.
EOF
}
if [[ $# -eq 0 ]]; then usage; exit 0; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --new-env) NEW_ENV=1 ;;
        --all) BASIC=1; NATIVE=1; UTONIA=1 ;;
        --basic) BASIC=1 ;;
        --native) NATIVE=1 ;;
        --utonia) UTONIA=1 ;;
        --data) DATA=1 ;;
        --eval) EVAL=1 ;;
        --moge) MOGE=1 ;;
        --ui) UI=1 ;;
        --check) CHECK=1 ;;
        --preflight) PREFLIGHT=1 ;;
        --flash-attn) FLASH_ATTN=1 ;;
        --nvdiffrast) NVDIFFRAST=1 ;;
        --nvdiffrec) NVDIFFREC=1 ;;
        --cumesh) CUMESH=1 ;;
        --flexgemm) FLEXGEMM=1 ;;
        --o-voxel) OVOXEL=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done
if [[ $CHECK -eq 1 || $PREFLIGHT -eq 1 ]]; then
    if [[ $NEW_ENV -eq 1 || $BASIC -eq 1 || $NATIVE -eq 1 || $UTONIA -eq 1 || $DATA -eq 1 || $EVAL -eq 1 || $MOGE -eq 1 || $UI -eq 1 || $FLASH_ATTN -eq 1 || $NVDIFFRAST -eq 1 || $NVDIFFREC -eq 1 || $CUMESH -eq 1 || $FLEXGEMM -eq 1 || $OVOXEL -eq 1 ]]; then
        echo "Use --check/--preflight separately from installation options." >&2
        exit 2
    fi
fi
if [[ $NEW_ENV -eq 1 ]]; then
    command -v conda >/dev/null || { echo "Install Conda first, or use an activated environment." >&2; exit 1; }
    set +u
    eval "$(conda shell.bash hook)"
    set -u
    if [[ ! -e "$ENV_PREFIX" ]]; then
        conda create --yes --prefix "$ENV_PREFIX" python=3.11 pip
    elif [[ ! -d "$ENV_PREFIX/conda-meta" ]]; then
        echo "Refusing to reuse a non-Conda directory: $ENV_PREFIX" >&2
        exit 1
    fi
    set +u
    conda activate "$ENV_PREFIX"
    set -u
fi
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit("Use Python 3.11, or pass --new-env.")
PY
if [[ $PREFLIGHT -eq 1 || $NATIVE -eq 1 || $FLASH_ATTN -eq 1 || $NVDIFFRAST -eq 1 || $NVDIFFREC -eq 1 || $CUMESH -eq 1 || $FLEXGEMM -eq 1 || $OVOXEL -eq 1 ]]; then
    [[ "$(uname -s)" == Linux ]] || { echo "This build targets Linux." >&2; exit 1; }
    command -v git >/dev/null
    command -v "${CXX:-c++}" >/dev/null
    NVCC="${CUDA_HOME:+$CUDA_HOME/bin/}nvcc"
    command -v "$NVCC" >/dev/null || { echo "Set CUDA_HOME to a CUDA 12.8 toolkit." >&2; exit 1; }
    "$NVCC" --version | grep -q 'release 12\.8' || { echo "A CUDA 12.8 toolkit is required by this build." >&2; exit 1; }
    echo "Python and CUDA build prerequisites found. No GPU runtime check was performed."
    if [[ $PREFLIGHT -eq 1 && $CHECK -eq 0 ]]; then exit 0; fi
fi
if [[ $CHECK -eq 1 ]]; then
    python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
packages = {
    "torch": "2.7.1", "torchvision": "0.22.1", "flash-attn": "2.7.3",
    "nvdiffrast": "0.4.0", "nvdiffrec-render": None, "cumesh": None,
    "flex-gemm": None, "o-voxel": None, "utonia": None,
    "spconv-cu121": "2.3.8", "torch-scatter": "2.1.2",
}
errors = []
for package, expected in packages.items():
    try:
        installed = version(package)
        print(f"{package}: {installed}")
        if expected and installed.split("+")[0] != expected:
            errors.append(f"{package}: expected {expected}")
    except PackageNotFoundError:
        errors.append(f"{package}: missing")
if errors:
    raise SystemExit("\n".join(errors))
print("Package metadata checked; imports and GPU kernels still require runtime validation.")
PY
    exit 0
fi
mkdir -p "$BUILD_DIR/tmp"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
export TMPDIR="$BUILD_DIR/tmp"
export MAX_JOBS="${MAX_JOBS:-4}"
export PIP_CONSTRAINT="$ROOT/requirements/base.txt"
checkout_source() {
    local name="$1" url="$2" revision="$3"
    [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo "Expected a full source commit for $name." >&2; return 1; }
    local source="$BUILD_DIR/$name-$revision"
    if [[ -e "$source" ]]; then
        [[ -d "$source/.git" ]] || { echo "Not a source checkout: $source" >&2; return 1; }
        [[ "$(git -C "$source" remote get-url origin)" == "$url" ]] || { echo "Unexpected origin: $source" >&2; return 1; }
        [[ -z "$(git -C "$source" status --porcelain --untracked-files=no)" ]] || { echo "Source has local changes: $source" >&2; return 1; }
        [[ "$(git -C "$source" rev-parse HEAD)" == "$revision" ]] || { echo "Unexpected revision: $source" >&2; return 1; }
    else
        git clone --no-checkout "$url" "$source"
        git -C "$source" checkout --detach "$revision"
    fi
    git -C "$source" submodule update --init --recursive
}
install_source() {
    checkout_source "$@"
    python -m pip install --no-build-isolation "$BUILD_DIR/$1-$3"
}
if [[ $BASIC -eq 1 ]]; then
    python -m pip install setuptools==80.9.0 wheel==0.45.1 packaging==25.0
    python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r "$ROOT/requirements/base.txt"
    install_source utils3d https://github.com/EasternJournalist/utils3d.git 3fab839f0be9931dac7c8488eb0e1600c236e183
elif [[ $DATA -eq 1 ]]; then
    python -m pip install -r "$ROOT/requirements/base.txt"
fi
if [[ $NATIVE -eq 1 ]]; then
    FLASH_ATTN=1; NVDIFFRAST=1; NVDIFFREC=1; CUMESH=1; FLEXGEMM=1; OVOXEL=1
fi
if [[ $FLASH_ATTN -eq 1 ]]; then python -m pip install flash-attn==2.7.3 --no-build-isolation; fi
if [[ $NVDIFFRAST -eq 1 ]]; then
    install_source nvdiffrast https://github.com/NVlabs/nvdiffrast.git 253ac4fcea7de5f396371124af597e6cc957bfae
fi
if [[ $NVDIFFREC -eq 1 ]]; then
    install_source nvdiffrec https://github.com/JeffreyXiang/nvdiffrec.git "$NVDIFFREC_REV"
fi
if [[ $CUMESH -eq 1 ]]; then
    install_source CuMesh https://github.com/JeffreyXiang/CuMesh.git "$CUMESH_REV"
fi
if [[ $FLEXGEMM -eq 1 ]]; then
    install_source FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git b1965ad0ae4bc8488d13b5280afb54439ce27359
fi
if [[ $OVOXEL -eq 1 ]]; then
    EIGEN_DIR="$ROOT/o-voxel/third_party/eigen"
    if [[ ! -f "$EIGEN_DIR/Eigen/Core" ]]; then
        if [[ -e "$EIGEN_DIR" && ( ! -d "$EIGEN_DIR" || -n "$(ls -A "$EIGEN_DIR")" ) ]]; then
            echo "Refusing to replace incomplete Eigen contents: $EIGEN_DIR" >&2
            exit 1
        fi
        checkout_source eigen https://gitlab.com/libeigen/eigen.git "$EIGEN_REV"
        mkdir -p "$EIGEN_DIR"
        git -C "$BUILD_DIR/eigen-$EIGEN_REV" archive "$EIGEN_REV" | tar -x -C "$EIGEN_DIR"
    fi
    [[ -f "$EIGEN_DIR/Eigen/Core" ]] || { echo "Eigen headers are missing after extraction." >&2; exit 1; }
    python -m pip install --no-build-isolation --no-deps "$ROOT/o-voxel"
fi
if [[ $UTONIA -eq 1 ]]; then
    python -m pip install spconv-cu121==2.3.8
    python -m pip install torch-scatter==2.1.2+pt27cu128 --no-deps --no-index \
        --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html
    install_source Utonia https://github.com/Pointcept/Utonia.git 6113b827341c1e025068c1f3a2601f5de6d7637f
fi
if [[ $EVAL -eq 1 ]]; then python -m pip install -r "$ROOT/requirements/eval.txt"; fi
if [[ $UI -eq 1 ]]; then python -m pip install -r "$ROOT/requirements/ui.txt"; fi
if [[ $MOGE -eq 1 ]]; then
    python -m pip install -r "$ROOT/requirements/moge.txt"
    install_source MoGe https://github.com/microsoft/MoGe.git 07444410f1e33f402353b99d6ccd26bd31e469e8
fi
python -m pip check
echo "Setup complete. Activate your environment before running training or inference."
if [[ $NEW_ENV -eq 1 ]]; then printf 'conda activate %q\n' "$ENV_PREFIX"; fi
