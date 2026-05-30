#!/bin/bash
# =============================================================================
# Lowest 2 thresholds — REALPHA only (gpt2, llama1B, llama8B)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_lowest_realpha.sh
# =============================================================================
set -euo pipefail
# Activate a conda env if CONDA_ENV is set; otherwise use the active env.
if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook 2>/dev/null)"
  conda activate "$CONDA_ENV"
fi

export PYTHONUNBUFFERED=TRUE
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

RUN="python -B -m transduced_lm.benchmark.run"
THRESHOLDS="0.000003 0.000001"
PARAGRAPHS=10
MAX_ATTEMPTS=5
SEGFAULT_SLEEP=300

cd "$(dirname "$0")"
mkdir -p logs results

run_one() {
    local tag="$1" transducer="$2" model="$3" preset="$4" ths="$5" para="$6"
    shift 6
    local extra_args=("$@")

    local output="results/${tag}_ths${ths}_pg${para}.pkl"
    local logfile="logs/${tag}_ths${ths}_pg${para}.log"

    if [[ -f "$output" ]]; then
        echo "  [SKIP] $output (already exists)"
        return 0
    fi

    for attempt in $(seq 1 $MAX_ATTEMPTS); do
        echo "  [RUN]  $tag ths=$ths para=$para (attempt $attempt/$MAX_ATTEMPTS)"
        local status=0
        $RUN \
            --transducer "$transducer" \
            --model "$model" \
            --preset "$preset" \
            --paragraphs $PARAGRAPHS \
            --para-index "$para" \
            --prune-threshold "$ths" \
            --use-vllm --dtype bf16 \
            --cleanup-interval 5000 \
            --max-retries 20 \
            --output "$output" \
            "${extra_args[@]}" \
            &> "$logfile" || status=$?

        if [[ $status -eq 0 ]]; then
            echo "  [OK]   $output"
            return 0
        elif [[ $status -eq 139 ]]; then
            echo "  [SEGV] $tag ths=$ths para=$para attempt=$attempt. Sleeping ${SEGFAULT_SLEEP}s..."
            sleep $SEGFAULT_SLEEP
        else
            echo "  [FAIL] $tag ths=$ths para=$para exit=$status. Skipping."
            return 0
        fi
    done
    echo "  [GIVE UP] $tag ths=$ths para=$para after $MAX_ATTEMPTS attempts"
}

EXPERIMENTS=(
    "realpha_gpt2      hf_realpha   gpt2-large                   realpha"
    "realpha_llama1B   hf_realpha   meta-llama/Llama-3.2-1B      realpha"
    "realpha_llama8B   hf_realpha   meta-llama/Llama-3.1-8B      realpha  --max-model-len 4096"
)

echo "============================================================"
echo "  REALPHA — lowest thresholds (0.000003, 0.000001)"
echo "  $(date)"
echo "============================================================"

for exp in "${EXPERIMENTS[@]}"; do
    read -r tag transducer model preset extra <<< "$exp"

    echo ""
    echo "============================================================"
    echo "  $tag  ($transducer / $model / $preset)"
    echo "  $(date)"
    echo "============================================================"

    for ths in $THRESHOLDS; do
        echo ""
        echo "  --- threshold=$ths ---"
        for para in $(seq 0 9); do
            # shellcheck disable=SC2086
            run_one "$tag" "$transducer" "$model" "$preset" "$ths" "$para" $extra
        done
    done
done

echo ""
echo "============================================================"
echo "  REALPHA ALL DONE: $(date)"
echo "============================================================"
