#!/bin/bash
# Submit all CE-only realpha jobs
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting CE-only realpha jobs..."
sbatch "$DIR/realpha_gpt2.sh"
sbatch "$DIR/realpha_llama1B.sh"
sbatch "$DIR/realpha_llama8B.sh"
sbatch "$DIR/realpha_phi4.sh"
echo "Done."
