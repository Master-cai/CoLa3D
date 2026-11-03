#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [3dfront metascenes replica scannetpp metascenes_hard]"
    echo "Default: all five datasets, all modes, evaluation seed 42."
    echo "Optional: BENCHMARK_ROOT, OUTPUT_ROOT, RESULTS_ROOT, NUM_WORKERS (default 1)."
    exit 0
fi
if [[ $# -eq 0 ]]; then set -- 3dfront metascenes replica scannetpp metascenes_hard; fi
for dataset in "$@"; do
    [[ "$dataset" == instpifu ]] && dataset=3dfront
    case "$dataset" in
        3dfront|metascenes|replica|scannetpp|metascenes_hard) ;;
        *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
    esac
    for mode in free point mask; do
        python scripts/eval/evaluate.py --dataset "$dataset" --mode "$mode" \
            --benchmark-root "${BENCHMARK_ROOT:-benchmarks}" \
            --pred-root "${OUTPUT_ROOT:-outputs}/$dataset/$mode" \
            --output "${RESULTS_ROOT:-results}/$dataset/$mode" \
            --evaluation-seed 42 --num-workers "${NUM_WORKERS:-1}"
    done
done
