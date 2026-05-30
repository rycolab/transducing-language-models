#!/bin/bash
#SBATCH -J dna2aa
#SBATCH -o logs/dna2aa_%j.out
#SBATCH -e logs/dna2aa_%j.err
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=80000
# =============================================================================
# DNA -> Amino Acids (hf_dna2aa)
# Paper Tables 6/9: JSD + throughput for max_candidates x thresholds
#
# Usage:
#   sbatch src/transduced_lm/benchmark/gpu_scripts/dna2aa.sh
#   sbatch src/transduced_lm/benchmark/gpu_scripts/dna2aa.sh --quick
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

THRESHOLDS="0.1 0.03 0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.000003 0.000001"

if [[ "${1:-}" == "--quick" ]]; then
    THRESHOLDS="0.01 0.001"
    PARAGRAPHS=1
    MAX_BYTES="--max-bytes 100"
    MAX_CANDS="5000"
    echo "[QUICK MODE] 1 protein, 100 bytes, 2 thresholds, 1 max_candidates"
else
    PARAGRAPHS=65
    MAX_BYTES=""
    MAX_CANDS="5000 10000 15000 20000"
fi

echo "============================================================"
echo "  DNA->AA benchmark (RTX 4090)"
echo "  Job: $SLURM_JOB_ID"
echo "  Proteins: $PARAGRAPHS"
echo "  Max candidates: $MAX_CANDS"
echo "  $(date)"
echo "============================================================"

for MAX_CAND in $MAX_CANDS; do
    echo ""
    echo ">>> max_candidates = $MAX_CAND"
    OUTPUT="results/dna2aa_maxcand_${MAX_CAND}.pkl"

    for ths in $THRESHOLDS; do
        echo "  >>> ths=$ths"
        $RUN \
          --transducer hf_dna2aa \
          --model vesteinn/gpt2-dna \
          --preset dna2aa \
          --paragraphs $PARAGRAPHS \
          $MAX_BYTES \
          --prune-threshold "$ths" \
          --max-candidates $MAX_CAND \
          --use-vllm --dtype bf16 \
          --cleanup-interval 5000 \
          --max-retries 20 \
          --output "$OUTPUT" \
        || echo "  [WARN] ths=$ths exited with code $?"
    done
done

echo ""
echo "Finished: $(date)"
