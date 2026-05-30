#!/bin/bash
#SBATCH -J du-realpha-llama8B
#SBATCH -o logs/du_realpha_llama8B_%j.out
#SBATCH -e logs/du_realpha_llama8B_%j.err
#SBATCH -t 72:00:00
#SBATCH -n 1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=200000
# =============================================================================
# Drop Universals — hf_realpha + Llama 3.1 8B
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
    RUN="python -B -m transduced_lm.benchmark.drop_universals"
else
    RUN="srun python -B -m transduced_lm.benchmark.drop_universals"
fi

echo "============================================================"
echo "  Drop Universals — hf_realpha + Llama 3.1 8B"
echo "  Job: ${SLURM_JOB_ID:-local}"
echo "  $(date)"
echo "============================================================"

$RUN \
  --transducer hf_realpha \
  --model meta-llama/Llama-3.1-8B \
  --paragraphs 10 \
  --steps 11 \
  --fraction-us 2 \
  --repeats 5 \
  --prune-threshold 0.0001 \
  --use-vllm --dtype bf16 \
  --max-model-len 8192 \
  --cleanup-interval 5000 \
  --output "results/drop_universals_llama_8B_realpha.pkl"

echo ""
echo "Finished: $(date)"
