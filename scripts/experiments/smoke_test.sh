#!/bin/bash
# =============================================================================
# Smoke test: reduced experiments for quick validation
#   - Thresholds: 0.01, 0.001, 0.0001, 0.00001, 0.000001
#   - Realpha + PTB: 3 paragraphs each
#   - DNA: 65 proteins, single max_candidates=10000
#   - GPT-2 Large only (realpha + PTB)
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

THRESHOLDS="0.01 0.001 0.0001 0.00001 0.000001"

echo "============================================================"
echo "  Smoke Test"
echo "  Thresholds: $THRESHOLDS"
echo "============================================================"

# --- Realpha: GPT-2 Large, 3 paragraphs ---
echo ""
echo ">>> Realpha / GPT-2 Large (3 paragraphs)"
for ths in $THRESHOLDS; do
    echo "  >>> ths=$ths"
    $RUN \
      --transducer hf_realpha \
      --model gpt2-large \
      --paragraphs 3 \
      --prune-threshold "$ths" \
      --preset old-realpha \
      --use-vllm --dtype bf16 \
      --cleanup-interval 5000 \
      --max-retries 5 \
      --output "results/realpha_gpt2_large.pkl" \
    || echo "  [WARN] ths=$ths exited with code $?"
done

# --- PTB: GPT-2 Large, 3 paragraphs ---
echo ""
echo ">>> PTB / GPT-2 Large (3 paragraphs)"
for ths in $THRESHOLDS; do
    echo "  >>> ths=$ths"
    $RUN \
      --transducer ptb_ported \
      --model gpt2-large \
      --paragraphs 3 \
      --prune-threshold "$ths" \
      --preset old-ptb \
      --use-vllm --dtype bf16 \
      --genlm-K 8 --genlm-prune 0.001 \
      --cleanup-interval 5000 \
      --max-retries 3 \
      --output "results/ptb_gpt2_large.pkl" \
    || echo "  [WARN] ths=$ths exited with code $?"
done

# --- DNA: gpt2-dna, 65 proteins, max_candidates=10000 ---
echo ""
echo ">>> DNA / gpt2-dna (65 proteins, max_candidates=10000)"
for ths in $THRESHOLDS; do
    echo "  >>> ths=$ths"
    $RUN \
      --transducer hf_dna2aa \
      --model vesteinn/gpt2-dna \
      --paragraphs 65 \
      --prune-threshold "$ths" \
      --preset old-dna2aa \
      --max-candidates 10000 \
      --use-vllm --dtype bf16 \
      --cleanup-interval 5000 \
      --max-retries 5 \
      --output "results/dna2aa_maxcand_10000.pkl" \
    || echo "  [WARN] ths=$ths exited with code $?"
done

echo ""
echo "Smoke test complete. Results in results/"
