# Experiment Reproduction Suite

Reproduces the experiments from "Transducing Language Models" (ICLR 2026).

## Quick Start

```bash
# Smoke test: 1 paragraph, 100 bytes, 2 thresholds, GPT-2 only
bash run_all.sh --quick

# Full paper reproduction (all models, all thresholds)
bash run_all.sh
```

## Experiments

| Script | Paper Table | Description | Models | Data |
|--------|------------|-------------|--------|------|
| `01_realpha_jsd.sh` | Table 4/7 | Tokens -> Characters | GPT-2 Large, Llama 1B, Llama 8B | 10 paras WikiText |
| `02_ptb_jsd.sh` | Table 5/8 | Tokens -> Words (PTB) | GPT-2 Large, Llama 1B, Llama 8B | 10 paras WikiText |
| `03_dna2aa_jsd.sh` | Table 6/9 | DNA -> Amino Acids | gpt2-dna | 65 proteins FASTA |
| `04_baseline_vieira.sh` | Table 3/6 | Vieira baseline (no FST) | GPT-2 Large, Llama 1B, Llama 8B | 10 paras WikiText |

Cross-entropy (Table 8/11) is extracted from the same realpha pickles by `analyze.py` -- no separate run needed.

## Pipeline

```
run_all.sh
  |
  +--> 01-04 bash scripts --> results/*.pkl    (raw benchmark output)
  +--> analyze.py          --> csv/*.csv        (JSD, CE, throughput with bootstrap CI)
  +--> generate_tables.py  --> tables/*.tex     (LaTeX tables matching paper format)
  +--> plot_figures.py     --> figures/*.pdf    (JSD-vs-throughput scatter plots)
```

## Directory Layout

```
experiments/
  results/           Raw pickle outputs from benchmark/run.py
  csv/               Processed CSV files (one per experiment)
  tables/            Generated LaTeX table fragments
  figures/           Generated PDF plots
```

## Parameters

All experiments use the paper's parameters (Appendix I):

- **Pruning thresholds**: 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.000003, 0.000001
- **Reference**: lowest threshold (1e-6), used as JSD denominator
- **Pruning**: candidate_threshold=100, prune_threshold_alpha=0.7, max_prune_mass=0.4
- **GenLM** (PTB/baseline): K=8, prune=0.001
- **Backend**: vLLM, bf16
- **DNA max_candidates**: 5000, 10000, 15000, 20000

## Running Individual Steps

```bash
# Just one experiment
bash 01_realpha_jsd.sh

# Just post-processing (pickles already exist)
python analyze.py
python generate_tables.py
python plot_figures.py
```

Each experiment script accepts `--quick` for a fast smoke test. Runs are incremental -- already-scored paragraphs are skipped on re-run.

## Notes

- Llama models require `huggingface-cli login` for gated access.
- Numbers will differ from the paper due to hardware, RNG, and library versions.
- The Vieira baseline is GenLMRealpha (genlm bytes) without any FST transduction.
