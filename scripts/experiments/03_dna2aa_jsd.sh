#!/bin/bash
# =============================================================================
# Experiment 3: DNA -> Amino Acids (hf_dna2aa)
# Paper Tables 6/9: JSD + throughput for 4 max_candidates x thresholds
# =============================================================================
#
# Each threshold runs in its own process for full isolation.
#
# Transducer: hf_dna2aa (DNA codon -> amino acid FST)
#   - 21 states, all universal
#   - Model: vesteinn/gpt2-dna (custom DNA tokenizer)
#
# Data: all 65 proteins from FASTA file
# Backend: vLLM, bf16
# Preset: old-dna2aa
#
# Aggressive thresholds (0.1, 0.03) omitted — too slow and mostly -inf.
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

THRESHOLDS="0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.000003 0.000001"

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
echo "  Experiment 3: DNA -> Amino Acids (hf_dna2aa)"
echo "  Proteins: $PARAGRAPHS"
echo "  Max candidates: $MAX_CANDS"
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
          --paragraphs $PARAGRAPHS \
          $MAX_BYTES \
          --prune-threshold "$ths" \
          --preset old-dna2aa \
          --max-candidates $MAX_CAND \
          --use-vllm --dtype bf16 \
          --cleanup-interval 5000 \
          --max-retries 5 \
          --output "$OUTPUT" \
        || echo "  [WARN] ths=$ths exited with code $?"
    done
done

echo ""
echo "Experiment 3 complete. Results in results/dna2aa_*.pkl"
