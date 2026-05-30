#!/bin/bash
# Submit all drop-universals experiments
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

for script in "$DIR"/realpha_*.sh; do
    echo "Submitting: $(basename "$script")"
    sbatch "$script"
done
