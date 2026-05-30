python -B -m transduced_lm.benchmark.run \
  --transducer ptb_ported  \
  --model gpt2-large \
  --paragraphs 1 \
  --prune-threshold 0.001 \
  --max-bytes 850 \
  --preset old-ptb \
  --use-vllm \
  --dtype bf16 \
  --output results/realpha_gpt2_ptb_ported_0p001.pkl
