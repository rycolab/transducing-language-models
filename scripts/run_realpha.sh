python -B -m transduced_lm.benchmark.run \
  --transducer hf_realpha  \
  --model gpt2-large \
  --paragraphs 1 \
  --prune-threshold 0.001 \
  --max-bytes 850 \
  --preset old-realpha \
  --use-vllm \
  --dtype bf16 \
  --output results/realpha_gpt2_0p001.pkl

