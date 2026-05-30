#!/bin/bash
#SBATCH -J realpha-phi4
#SBATCH -o logs/realpha_phi4_%j.out
#SBATCH -e logs/realpha_phi4_%j.err
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH --gpus=rtx_4090:2
#SBATCH --mem-per-cpu=60000
# =============================================================================
# Tokens -> Characters (hf_realpha) — Phi-4 14B
#
# Usage:
#   sbatch scripts/experiments/cr/realpha_phi4.sh
#   sbatch scripts/experiments/cr/realpha_phi4.sh --quick
#   TLM_LOCAL=1 bash realpha_phi4.sh [--quick]  # local run (tlm env)
# =============================================================================

set -euo pipefail
mkdir -p logs results
export PYTHONUNBUFFERED=TRUE

# Activate a conda env if CONDA_ENV is set; otherwise use the active env.
if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook 2>/dev/null)"
  conda activate "$CONDA_ENV"
fi
if [[ "${TLM_LOCAL:-}" == "1" ]]; then
    RUN="python -B -m transduced_lm.benchmark.run"
else
    RUN="srun python -B -m transduced_lm.benchmark.run"
fi

THRESHOLDS="0.1 0.03 0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.000003 0.000001"

if [[ "${1:-}" == "--quick" ]]; then
    THRESHOLDS="0.01 0.001"
    PARAGRAPHS=1
    MAX_BYTES="--max-bytes 850"
    echo "[QUICK MODE] 1 paragraph, 850 bytes, 2 thresholds"
else
    PARAGRAPHS=10
    MAX_BYTES=""
fi

echo "============================================================"
echo "  Realpha — Phi-4 14B"
echo "  Job: ${SLURM_JOB_ID:-local}"
echo "  Paragraphs: $PARAGRAPHS"
echo "  $(date)"
echo "============================================================"

for ths in $THRESHOLDS; do
    echo ""
    echo "  >>> ths=$ths"
    $RUN \
      --transducer hf_realpha \
      --model microsoft/phi-4 \
      --preset realpha \
      --paragraphs $PARAGRAPHS \
      $MAX_BYTES \
      --prune-threshold "$ths" \
      --use-vllm --dtype bf16 --max-model-len 4096 --tensor-parallel-size 2 \
      --cleanup-interval 5000 \
      --max-retries 20 \
      --output "results/realpha_phi4.pkl" \
    || echo "  [WARN] ths=$ths exited with code $?"
done

echo ""
echo "Finished: $(date)"
