#!/bin/bash
# =============================================================================
# Master script: run all experiments, analyze, generate tables and plots.
#
# Usage:
#   bash run_all.sh           # Full run (all models, all thresholds)
#   bash run_all.sh --quick   # Quick smoke test (1 para, 2 thresholds, small)
#
# Steps:
#   1. Run experiments (01-04): writes pickle files to results/
#   2. Analyze pickles -> CSV: writes csv/ files
#   3. Generate LaTeX tables: writes tables/ files
#   4. Generate plots: writes figures/ files
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

QUICK_FLAG="${1:-}"

echo "============================================================"
echo "  Transduced LM Experiment Suite"
echo "  $(date)"
echo "============================================================"

# --- Step 1: Run experiments ---
echo ""
echo "=== Step 1: Run experiments ==="
echo ""

echo "--- 01: Tokens -> Characters (hf_realpha) ---"
bash 01_realpha_jsd.sh $QUICK_FLAG

echo ""
echo "--- 02: Tokens -> Words (PTB) ---"
bash 02_ptb_jsd.sh $QUICK_FLAG

echo ""
echo "--- 03: DNA -> Amino Acids ---"
bash 03_dna2aa_jsd.sh $QUICK_FLAG

echo ""
echo "--- 04: Vieira Baseline ---"
bash 04_baseline_vieira.sh $QUICK_FLAG

# --- Step 2: Analyze ---
echo ""
echo "=== Step 2: Analyze results ==="
python analyze.py

# --- Step 3: Generate tables ---
echo ""
echo "=== Step 3: Generate LaTeX tables ==="
python generate_tables.py

# --- Step 4: Generate plots ---
echo ""
echo "=== Step 4: Generate plots ==="
python plot_figures.py

# --- Done ---
echo ""
echo "============================================================"
echo "  All done!"
echo "============================================================"
echo "  Results:  results/*.pkl"
echo "  CSVs:     csv/*.csv"
echo "  Tables:   tables/*.tex"
echo "  Figures:  figures/*.pdf"
echo "============================================================"
