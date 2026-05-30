#!/bin/bash
# =============================================================================
# Experiment 1: Tokens -> Characters (hf_realpha)
# Paper Tables 4/7: JSD + throughput for 3 models x 11 thresholds
# =============================================================================
#
# Each threshold runs in its own process for full isolation (no shared
# LLM/genlm state, clean timing, no OOM from accumulated caches).
#
# Transducer: hf_realpha (HF tokenizer -> bytes FST)
#   - GPT-2 Large: 75,723 states, all universal
#   - Llama 3.2-1B: 176,990 states, all universal
#   - Llama 3.1-8B: same as Llama 1B (shared tokenizer)
#
# Data: 10 paragraphs from WikiText-2 test split
# Backend: vLLM, bf16
# Preset: old-realpha (matches paper parameters)
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate a conda env if CONDA_ENV is set; otherwise use the active env.
if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook 2>/dev/null)"
  conda activate "$CONDA_ENV"
fi

RUN="python -B -m transduced_lm.benchmark.run"

THRESHOLDS="0.1 0.03 0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.000003 0.000001"

if [[ "${1:-}" == "--quick" ]]; then
    THRESHOLDS="0.01 0.001"
    PARAGRAPHS=1
    MAX_BYTES="--max-bytes 100"
    echo "[QUICK MODE] 1 paragraph, 100 bytes, 2 thresholds"
else
    PARAGRAPHS=10
    MAX_BYTES=""
fi

echo "============================================================"
echo "  Experiment 1: Tokens -> Characters (hf_realpha)"
echo "  Paragraphs: $PARAGRAPHS"
echo "============================================================"

# Retry count per threshold: aggressive thresholds need more retries
# (prune_factor=0.5 halves each retry: 10 retries from 0.1 → 0.0001).
retries_for() {
    local ths="$1"
    case "$ths" in
        0.1|0.03) echo 10 ;;
        *)        echo 5  ;;
    esac
}

run_model() {
    local MODEL="$1"
    local OUTPUT="$2"
    local EXTRA_ARGS="${3:-}"

    for ths in $THRESHOLDS; do
        local retries
        retries=$(retries_for "$ths")
        echo ""
        echo "  >>> ths=$ths  retries=$retries"
        $RUN \
          --transducer hf_realpha \
          --model "$MODEL" \
          --paragraphs $PARAGRAPHS \
          $MAX_BYTES \
          --prune-threshold "$ths" \
          --preset old-realpha \
          --use-vllm --dtype bf16 \
          --cleanup-interval 5000 \
          --max-retries "$retries" \
          $EXTRA_ARGS \
          --output "$OUTPUT" \
        || echo "  [WARN] ths=$ths exited with code $?"
    done
}

# --- GPT-2 Large ---
echo ""
echo ">>> GPT-2 Large"
run_model "gpt2-large" "results/realpha_gpt2_large.pkl" ""

# --- Llama 3.2-1B ---
echo ""
echo ">>> Llama 3.2-1B"
run_model "meta-llama/Llama-3.2-1B" "results/realpha_llama_1b.pkl" "--max-model-len 8192"

# --- Llama 3.1-8B ---
echo ""
echo ">>> Llama 3.1-8B"
run_model "meta-llama/Llama-3.1-8B" "results/realpha_llama_8b.pkl" "--max-model-len 8192"

echo ""
echo "Experiment 1 complete. Results in results/realpha_*.pkl"
