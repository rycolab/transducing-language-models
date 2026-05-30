#!/usr/bin/env bash
# =============================================================================
# Quick setup smoke test.
#
# Loads each transducer + model, scores a few symbols, and confirms the data
# loads. Use this to verify a fresh install end-to-end: package import, GPU /
# vLLM, HuggingFace model download/load, and dataset download (WikiText / FASTA).
#
# It is NOT an experiment (1 paragraph, a handful of symbols). For paper
# reproduction see scripts/experiments/.
#
# Usage:
#   bash scripts/smoke_test.sh                 # gpt2-large + gpt2-dna (no login)
#   CONDA_ENV=tlm bash scripts/smoke_test.sh   # activate a named conda env first
#   MODEL=meta-llama/Llama-3.2-1B bash scripts/smoke_test.sh   # gated (needs login)
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate a conda env if CONDA_ENV is set; otherwise use the active env.
if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook 2>/dev/null)"
  conda activate "$CONDA_ENV"
fi

RUN="python -B -m transduced_lm.benchmark.run"
MODEL="${MODEL:-gpt2-large}"   # token-level model for the realpha / PTB checks
fail=0

smoke() {  # <transducer> <model> <preset> [extra args...]
  local t=$1 m=$2 p=$3; shift 3
  echo ""
  echo "=================================================================="
  echo "  smoke: transducer=$t  model=$m"
  echo "=================================================================="
  rm -f "results/smoke_${t}.pkl"   # always run fresh (this is a setup check)
  $RUN --transducer "$t" --model "$m" --paragraphs 1 --max-bytes 8 \
       --prune-threshold 0.001 --preset "$p" --use-vllm --dtype bf16 "$@" \
       --output "results/smoke_${t}.pkl" || { echo "  [FAIL] $t / $m"; fail=1; }
}

# Same CLI + presets as scripts/run_{realpha,ptb,dna2aa}.sh; only --max-bytes
# (set in smoke() below) is smaller, for speed.
# tokens -> bytes, and tokens -> words (Penn Treebank): both use $MODEL
smoke hf_realpha "$MODEL"          old-realpha
smoke ptb_ported "$MODEL"          old-ptb
# DNA -> amino acids: uses the custom DNA model
smoke hf_dna2aa  vesteinn/gpt2-dna old-dna2aa    --max-candidates 5000

echo ""
if [[ $fail -eq 0 ]]; then
  echo "smoke test: ALL PASSED"
else
  echo "smoke test: FAILURES (see above)"
fi
exit $fail
