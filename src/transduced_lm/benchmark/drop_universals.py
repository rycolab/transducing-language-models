#!/usr/bin/env python3
"""
Drop-universals experiment: progressively remove universal states from an FST
and measure the impact on scoring quality (JSD) and speed.

For each "drop count" K, randomly samples K universal states to mark as
non-universal, then scores paragraphs with the modified FST. Repeats over
multiple random seeds to produce averaged results.

Port of the old tokenizer_conversion/benchmarking/prefix_probs_drop_universals.py
to the new transduced_lm module.

Usage:
    python -m transduced_lm.benchmark.drop_universals \
        --transducer hf_realpha --model gpt2-large \
        --paragraphs 10 --output results/drop_universals_gpt2.pkl

    python -m transduced_lm.benchmark.drop_universals \
        --transducer hf_realpha --model gpt2-large \
        --steps 11 --fraction-us 2 --repeats 5 \
        --output results/drop_universals_gpt2.pkl
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import random
import sys
import time

import numpy as np

from .sequence import sequence_logp_next
from .transducer import (
    TransducerSetup,
    build_cleanup_fn,
    build_logp_next_fn,
    cleanup_genlm,
    load_transducer,
    setup_genlm,
)
from .data import load_wikitext_sequences
from .storage import safe_load_pickle, file_lock, atomic_pickle_dump

try:
    profile
except NameError:
    def profile(fn):
        return fn


async def run_drop_universals(
    model_name: str,
    transducer_name: str,
    split: str = "test",
    paragraphs: int = 10,
    max_bytes: int | None = None,
    prune_threshold: float = 0.0001,
    steps: int = 11,
    fraction_us: int = 2,
    repeats: int = 5,
    output_file: str = "results/drop_universals.pkl",
    seed: int = 80808,
    use_vllm: bool = False,
    torch_dtype: str = "bf16",
    max_model_len: int | None = None,
    tensor_parallel_size: int | None = None,
    genlm_K: int = 8,
    genlm_prune: float = 0.001,
    cleanup_interval: int = 5000,
    max_retries: int = 5,
    verbose: bool = True,
):
    """Run the drop-universals experiment."""
    if not sys.stdout.line_buffering:
        sys.stdout.reconfigure(line_buffering=True)

    random.seed(seed)
    np.random.seed(seed)
    wall_t0 = time.time()

    needs_genlm = transducer_name in ("ptb_ported",)

    # ── Banner ────────────────────────────────────────────────────────────
    print()
    print("=" * 66)
    print("  Drop Universals Experiment")
    print("=" * 66)
    print(f"  Model       : {model_name}")
    print(f"  Transducer  : {transducer_name}")
    print(f"  Backend     : {'vLLM' if use_vllm else 'HuggingFace'}")
    print(f"  Dtype       : {torch_dtype}")
    print(f"  Paragraphs  : {paragraphs}")
    print(f"  Threshold   : {prune_threshold}")
    print(f"  Steps       : {steps}")
    print(f"  Fraction US : 1/{fraction_us}")
    print(f"  Repeats     : {repeats}")
    print(f"  Output      : {output_file}")
    print("=" * 66)

    # ── 1. Load LLM ──────────────────────────────────────────────────────
    print(f"\n[1/4] Loading LLM: {model_name} ...", end=" ", flush=True)
    t0 = time.time()

    import torch
    from genlm.backend import load_model_by_name

    _dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    _torch_dtype = _dtype_map[torch_dtype]
    _vllm_dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}

    if use_vllm:
        engine_opts = {"dtype": _vllm_dtype_map[torch_dtype]}
        if max_model_len is not None:
            engine_opts["max_model_len"] = max_model_len
        if tensor_parallel_size is not None:
            engine_opts["tensor_parallel_size"] = tensor_parallel_size
        llm = load_model_by_name(model_name, llm_opts={"engine_opts": engine_opts})
    else:
        llm = load_model_by_name(
            model_name, backend="hf",
            llm_opts={"hf_opts": {"torch_dtype": _torch_dtype}},
        )
    print(f"done ({time.time() - t0:.1f}s)")

    # ── 2. Load transducer ────────────────────────────────────────────────
    print(f"[2/4] Loading transducer: {transducer_name} ...", end=" ", flush=True)
    t0 = time.time()
    setup = load_transducer(transducer_name, llm=llm, model_name=model_name)

    n_states = setup.vfst.num_states
    n_arcs = sum(setup.vfst.fst.num_arcs(s) for s in setup.vfst.fst.states())
    original_universals = {
        s for s, is_u in setup.vfst.universal_states.items() if is_u
    }
    n_univ = len(original_universals)
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"       FST: {n_states:,} states, {n_arcs:,} arcs")
    print(f"       Universal: {n_univ:,}/{n_states:,}")

    # ── 3. Load data ──────────────────────────────────────────────────────
    print(f"[3/4] Loading data ...", end=" ", flush=True)
    t0 = time.time()
    data_seqs, original_texts, total_len = load_wikitext_sequences(
        setup, split=split, n=paragraphs, verbose=False,
    )
    if max_bytes is not None:
        for i in range(len(data_seqs)):
            data_seqs[i] = data_seqs[i][:max_bytes]
        total_len = sum(len(s) for s in data_seqs)
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"       {len(data_seqs)} paragraph(s), {total_len} total output symbols")
    for j, txt in enumerate(original_texts):
        preview = txt[:80] + ("..." if len(txt) > 80 else "")
        print(f"       [{j}] ({len(data_seqs[j]):,} sym) {preview!r}")

    # ── 4. Compute drop schedule ──────────────────────────────────────────
    max_K = int(n_univ / fraction_us)
    drop_counts = np.unique(np.linspace(0, max_K, steps, dtype=int)).tolist()

    print(f"\n[4/4] Drop schedule: {drop_counts}")
    print(f"       Max drop: {max_K} ({n_univ} universal / {fraction_us})")

    # ── 5. Load existing results for resume ───────────────────────────────
    metadata = {
        "model_name": model_name,
        "transducer_name": transducer_name,
        "split": split,
        "paragraphs": len(data_seqs),
        "text_length": total_len,
        "prune_threshold": prune_threshold,
        "steps": steps,
        "fraction_us": fraction_us,
        "repeats": repeats,
        "org_num_universal_states": n_univ,
        "drop_counts": drop_counts,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    default_payload = {
        "metadata": metadata,
        "stats": {},
        "p_nexts": {},
        "drop_counts": drop_counts,
        "num_universal_states": n_univ,
    }
    saved = safe_load_pickle(output_file, default_payload)
    completed_Ks = set(saved.get("p_nexts", {}).keys())

    # Set up genlm if needed (shared across all K)
    if needs_genlm and setup.lm is None:
        await setup_genlm(
            setup, model_name, llm=llm,
            K=genlm_K, prune_threshold=genlm_prune,
        )

    # ── Main loop ─────────────────────────────────────────────────────────
    for K in drop_counts:
        if K in completed_Ks:
            print(f"\n  Skipping K={K}, already processed")
            continue

        print(f"\n{'━'*66}")
        print(f"  K = {K}  (dropping {K}/{n_univ} universal states)")
        print(f"{'━'*66}")

        all_times_K = []
        all_p_nexts_K = []
        stats_K = {"times": [], "log_probs": []}

        for r in range(repeats):
            # Sample which universals to drop
            drop_set = set(random.sample(list(original_universals), K)) if K > 0 else set()
            new_universals = original_universals - drop_set
            print(f"\n  Repeat {r+1}/{repeats}: "
                  f"keeping {len(new_universals)}/{n_univ} universal states")

            for pi, para in enumerate(data_seqs):
                preview = original_texts[pi][:50] + ("..." if len(original_texts[pi]) > 50 else "")
                print(f"    Para {pi+1}/{len(data_seqs)} ({len(para)} sym) {preview!r}")

                # Modify universality on the shared VectorizedFST
                setup.vfst.universal_states = {
                    s: (s in new_universals)
                    for s in setup.vfst.universal_states
                }
                setup.vfst._universal_set_cache = {}
                setup.vfst._universal_sets_by_size.clear()
                # Invalidate the all_universal cached property
                if "all_universal" in setup.vfst.__dict__:
                    del setup.vfst.__dict__["all_universal"]

                logp_next_fn = fallback_fn = on_fallback = on_recover = None
                probe_fn = score_single_fn = None
                try:
                    logp_next_fn, fallback_fn, on_fallback, on_recover, probe_fn, score_single_fn = build_logp_next_fn(
                        setup, impl="logp_next",
                        prune_threshold=prune_threshold,
                        verbose=False,
                    )

                    genlm_cleanup = build_cleanup_fn(setup) if cleanup_interval > 0 else None

                    t0 = time.time()
                    stats = await sequence_logp_next(
                        logp_next_fn, para,
                        out_id_to_sym=setup.out_id_to_sym,
                        verbose=verbose,
                        cleanup_fn=genlm_cleanup,
                        cleanup_interval=cleanup_interval,
                        fallback_fn=fallback_fn,
                        on_fallback=on_fallback,
                        on_recover=on_recover,
                        probe_fn=probe_fn,
                        score_single_fn=score_single_fn,
                        max_retries=max_retries,
                    )
                    elapsed = time.time() - t0

                    ce = -stats['total_logp'] / len(para)
                    bpb = ce / np.log(2)
                    bps = len(para) / elapsed if elapsed > 0 else 0
                    print(f"      {elapsed:.1f}s | CE={ce:.4f} | "
                          f"bpb={bpb:.4f} | {bps:.1f} byte/s")

                    stats_K["times"].extend(stats["times"])
                    stats_K["log_probs"].extend(stats["log_probs"])

                    all_p_nexts_K.append([
                        dict(d) for d in stats["distributions"]
                    ])

                except RuntimeError as e:
                    if "Unrecoverable -inf" in str(e):
                        print(f"      ERROR: {e}")
                        print(f"      Skipping paragraph {pi+1}")
                        continue
                    raise

                finally:
                    _tlm = getattr(logp_next_fn, '__self__', None)
                    if _tlm is not None and hasattr(_tlm, 'clear_cache'):
                        _tlm.clear_cache()
                    del _tlm
                    logp_next_fn = fallback_fn = on_fallback = on_recover = None
                    probe_fn = score_single_fn = None
                    if needs_genlm:
                        _gc_fn = build_cleanup_fn(setup)
                        if _gc_fn is not None:
                            _gc_fn()
                    gc.collect()
                    try:
                        import torch as _torch
                        _torch.cuda.empty_cache()
                    except ImportError:
                        pass

        # Restore original universality
        setup.vfst.universal_states = {
            s: (s in original_universals)
            for s in setup.vfst.universal_states
        }
        setup.vfst._universal_set_cache = {}
        setup.vfst._universal_sets_by_size.clear()
        if "all_universal" in setup.vfst.__dict__:
            del setup.vfst.__dict__["all_universal"]

        # ── Save after each K ─────────────────────────────────────────────
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        lock_path = output_file + ".lock"

        with file_lock(lock_path):
            prev = safe_load_pickle(output_file, default_payload)
            prev_stats = prev.get("stats", {})
            prev_pnexts = prev.get("p_nexts", {})

            prev_stats[K] = stats_K
            prev_pnexts[K] = all_p_nexts_K

            atomic_pickle_dump(
                {
                    "metadata": metadata,
                    "stats": prev_stats,
                    "p_nexts": prev_pnexts,
                    "drop_counts": drop_counts,
                    "num_universal_states": n_univ,
                },
                output_file,
            )
        print(f"  Saved K={K} to {output_file}")
        del prev, prev_stats, prev_pnexts
        gc.collect()

    # Final cleanup
    if needs_genlm:
        await cleanup_genlm(setup)

    wall_elapsed = time.time() - wall_t0
    print()
    print("=" * 66)
    print("  Drop Universals — Complete")
    print("=" * 66)
    print(f"  Model       : {model_name}")
    print(f"  Transducer  : {transducer_name}")
    print(f"  Wall time   : {wall_elapsed:.1f}s")
    print(f"  Drop counts : {drop_counts}")
    print(f"  Output      : {output_file}")
    print("=" * 66)


async def main():
    parser = argparse.ArgumentParser(
        description="Drop-universals experiment: measure JSD impact of "
                    "removing universal states from the FST"
    )
    parser.add_argument(
        "--transducer", required=True,
        choices=["hf_realpha", "ptb", "ptb_ported"],
        help="Transducer type",
    )
    parser.add_argument("--model", default="gpt2-large", help="HF model name")
    parser.add_argument("--split", default="test", help="WikiText split")
    parser.add_argument("--paragraphs", type=int, default=10)
    parser.add_argument("--max-bytes", type=int, default=None)

    parser.add_argument(
        "--prune-threshold", type=float, default=0.0001,
        help="Prune threshold for scoring (default: 0.0001)",
    )
    parser.add_argument(
        "--steps", type=int, default=11,
        help="Number of linearly spaced drop counts (default: 11)",
    )
    parser.add_argument(
        "--fraction-us", type=int, default=2,
        help="Denominator for max drop: drops up to N_universal/fraction-us states "
             "(default: 2, i.e. drop up to half)",
    )
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="Number of random samples per drop count (default: 5)",
    )

    parser.add_argument("--output", default=None, help="Output pickle file path")
    parser.add_argument("--seed", type=int, default=80808)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument(
        "--dtype", default="bf16", choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--genlm-K", type=int, default=8)
    parser.add_argument("--genlm-prune", type=float, default=0.001)
    parser.add_argument("--cleanup-interval", type=int, default=5000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.output is None:
        model_short = args.model.split("/")[-1].lower().replace("-", "_")
        args.output = f"results/drop_universals_{model_short}_{args.transducer}.pkl"

    await run_drop_universals(
        model_name=args.model,
        transducer_name=args.transducer,
        split=args.split,
        paragraphs=args.paragraphs,
        max_bytes=args.max_bytes,
        prune_threshold=args.prune_threshold,
        steps=args.steps,
        fraction_us=args.fraction_us,
        repeats=args.repeats,
        output_file=args.output,
        seed=args.seed,
        use_vllm=args.use_vllm,
        torch_dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        genlm_K=args.genlm_K,
        genlm_prune=args.genlm_prune,
        cleanup_interval=args.cleanup_interval,
        max_retries=args.max_retries,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    asyncio.run(main())
