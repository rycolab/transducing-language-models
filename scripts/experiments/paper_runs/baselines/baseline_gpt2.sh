#!/bin/bash
#SBATCH -J baseline-gpt2
#SBATCH -o logs/baseline_gpt2_%j.out
#SBATCH -e logs/baseline_gpt2_%j.err
#SBATCH -t 14:00:00
#SBATCH -n 1
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=120000
# =============================================================================
# genlm-bytes baseline — GPT-2 Large
#
# Runs Vieira et al. baseline (genlm bytes, no FST) to produce byte-level
# distributions for JSD comparison with the transduced LM.
#
# Usage:
#   sbatch scripts/experiments/cr/baselines/baseline_gpt2.sh
# =============================================================================

set -euo pipefail
mkdir -p logs results
export PYTHONUNBUFFERED=TRUE

# Activate a conda env if CONDA_ENV is set; otherwise use the active env.
if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook 2>/dev/null)"
  conda activate "$CONDA_ENV"
fi

RUN="srun python -B -m transduced_lm.benchmark.run"

echo "============================================================"
echo "  genlm-bytes baseline — GPT-2 Large"
echo "  Job: $SLURM_JOB_ID"
echo "  $(date)"
echo "============================================================"

for K in 64; do
    echo ""
    echo ">>> GPT-2 Large (baseline, K=$K)"
    $RUN \
      --transducer baseline \
      --model gpt2-large \
      --paragraphs 10 \
      --prune-threshold 0.001 \
      --use-vllm --dtype bf16 \
      --genlm-K $K --genlm-prune 0.0 \
      --output "results/baseline_gpt2_large_K${K}.pkl" \
    || echo "  [WARN] K=$K exited with code $?"
done

echo ""
echo "Finished: $(date)"
