#!/bin/bash
# Submit all CR benchmark jobs to SLURM.
#
# Usage:
#   bash scripts/experiments/cr/thresholds/submit_all.sh
#   bash scripts/experiments/cr/thresholds/submit_all.sh --quick   # quick mode for all jobs

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ARGS="${*:-}"

echo "Submitting CR benchmark jobs ..."
echo ""

for script in \
    "$DIR/realpha_gpt2.sh" \
    "$DIR/realpha_llama1B.sh" \
    "$DIR/realpha_llama8B.sh" \
    "$DIR/realpha_phi4.sh" \
    "$DIR/ptb_gpt2.sh" \
    "$DIR/ptb_llama1B.sh" \
    "$DIR/ptb_llama8B.sh" \
    "$DIR/ptb_phi4.sh" \
    "$DIR/dna2aa.sh" \
; do
    name="$(basename "$script")"
    jobid=$(sbatch $ARGS "$script" | awk '{print $4}')
    echo "  $name -> job $jobid"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
