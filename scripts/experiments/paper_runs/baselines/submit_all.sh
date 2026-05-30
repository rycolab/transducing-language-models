#!/bin/bash
# Submit all genlm-bytes baseline jobs to SLURM.
#
# Usage:
#   bash scripts/experiments/cr/baselines/submit_all.sh

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting genlm-bytes baseline jobs ..."
echo ""

for script in \
    "$DIR/baseline_gpt2.sh" \
    "$DIR/baseline_llama1B.sh" \
    "$DIR/baseline_llama8B.sh" \
    "$DIR/baseline_phi4.sh" \
; do
    name="$(basename "$script")"
    jobid=$(sbatch "$script" | awk '{print $4}')
    echo "  $name -> job $jobid"
done

echo ""
echo "All baseline jobs submitted. Monitor with: squeue -u \$USER"
