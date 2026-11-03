#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [3dfront metascenes replica scannetpp metascenes_hard]"
    echo "Default: all five datasets, all modes, sequentially on one GPU."
    echo "Optional: BENCHMARK_ROOT, CHECKPOINT_ROOT, OUTPUT_ROOT."
    exit 0
fi
if [[ $# -eq 0 ]]; then set -- 3dfront metascenes replica scannetpp metascenes_hard; fi
for dataset in "$@"; do
    [[ "$dataset" == instpifu ]] && dataset=3dfront
    case "$dataset" in
        3dfront|metascenes|replica|scannetpp) config=configs/inference/q8.json ;;
        metascenes_hard) config=configs/inference/q20.json ;;
        *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
    esac
    for mode in free point mask; do
        python scripts/inference/run_benchmark.py --config "$config" --dataset "$dataset" --mode "$mode" \
            --benchmark-root "${BENCHMARK_ROOT:-benchmarks}" --checkpoint-root "${CHECKPOINT_ROOT:-checkpoints}" \
            --output-root "${OUTPUT_ROOT:-outputs}"
    done
done
