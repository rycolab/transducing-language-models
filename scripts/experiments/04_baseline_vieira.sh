#!/bin/bash
# =============================================================================
# Experiment 4: Vieira et al. Baseline (genlm bytes, no transduction)
# Paper Table 3/6: JSD comparison between transduced LM and raw byte-level LM
# =============================================================================
#
# This runs the GenLMRealpha byte-level language model WITHOUT any FST
# transduction.  The distributions are stored in the same pickle format
# as the transduced experiments, allowing direct JSD comparison via
# analyze.py.
#
# The Vieira baseline uses genlm's ByteBeamState to decompose token-level
# models into byte-level distributions.  Parameters K and prune_threshold
# control the beam search quality (not FST pruning).
#
# Data: 10 paragraphs from WikiText-2 test split (same as realpha/PTB)
# Backend: vLLM, bf16
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

# For baseline, prune-threshold controls genlm's internal beam pruning,
# not FST pruning.  We use a single conservative value.
THRESHOLDS="0.001"

if [[ "${1:-}" == "--quick" ]]; then
    PARAGRAPHS=1
    MAX_BYTES="--max-bytes 100"
    echo "[QUICK MODE] 1 paragraph, 100 bytes"
else
    PARAGRAPHS=10
    MAX_BYTES=""
fi

echo "============================================================"
echo "  Experiment 4: Vieira Baseline (genlm bytes, no FST)"
echo "  Paragraphs: $PARAGRAPHS"
echo "============================================================"

# --- GPT-2 Large ---
echo ""
echo ">>> GPT-2 Large (baseline)"
$RUN \
  --transducer baseline \
  --model gpt2-large \
  --paragraphs $PARAGRAPHS \
  $MAX_BYTES \
  --prune-threshold $THRESHOLDS \
  --use-vllm --dtype bf16 \
  --genlm-K 8 --genlm-prune 0.001 \
  --output results/baseline_gpt2_large.pkl

# --- Llama 3.2-1B ---
echo ""
echo ">>> Llama 3.2-1B (baseline)"
$RUN \
  --transducer baseline \
  --model meta-llama/Llama-3.2-1B \
  --paragraphs $PARAGRAPHS \
  $MAX_BYTES \
  --prune-threshold $THRESHOLDS \
  --use-vllm --dtype bf16 \
  --max-model-len 8192 \
  --genlm-K 8 --genlm-prune 0.001 \
  --output results/baseline_llama_1b.pkl

# --- Llama 3.1-8B ---
echo ""
echo ">>> Llama 3.1-8B (baseline)"
$RUN \
  --transducer baseline \
  --model meta-llama/Llama-3.1-8B \
  --paragraphs $PARAGRAPHS \
  $MAX_BYTES \
  --prune-threshold $THRESHOLDS \
  --use-vllm --dtype bf16 \
  --max-model-len 8192 \
  --genlm-K 8 --genlm-prune 0.001 \
  --output results/baseline_llama_8b.pkl

echo ""
echo "Experiment 4 complete. Results in results/baseline_*.pkl"
