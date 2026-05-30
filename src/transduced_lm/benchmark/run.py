#!/usr/bin/env python3
"""
Main benchmark entry point for the refactored transduced_lm module.

Each invocation scores all paragraphs at a **single** prune threshold and
appends results to the output pickle.  Shell scripts loop over thresholds,
giving full process isolation (no shared LLM/genlm state between thresholds).

Usage examples:

    # PTB ported + GPT-2, default (logp_next), first 100 bytes of 1 paragraph
    python -m transduced_lm.benchmark.run \
        --transducer ptb_ported --model gpt2-large \
        --max-bytes 100 --paragraphs 1

    # Single threshold — shell scripts call this once per threshold
    python -m transduced_lm.benchmark.run \
        --transducer ptb_ported --model gpt2-large \
        --prune-threshold 0.001 --output results/ptb_gpt2.pkl

    # Reproduce the old benchmark config
    python -m transduced_lm.benchmark.run \
        --transducer ptb_ported --model gpt2-large --preset old-benchmark

Parameter presets:
    --preset old-benchmark:
        Reproduce the old benchmarking/prefix_probs_bytes_threshold.py behavior:
        ignore_remainder=True, max_steps=3 (old code used expand_threshold=3 as
        iteration cap; in the new code that role is max_steps),
        skip_combined_univ=True, skip_target_universal=True, expand_threshold=3,
        stop_epsilon_mass=0.01, candidate_threshold=100, prune_threshold_alpha=0.0,
        max_prune_mass=1.0.
    --preset old-ptb / old-realpha / old-dna2aa:
        Per-FST aliases of old-benchmark (identical settings for now; provides
        a place for FST-specific tuning later).
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import sys
import time

import numpy as np

from .sequence import sequence_logp_next, sequence_ce_only
from .transducer import (
    TransducerSetup,
    build_cleanup_fn,
    build_logp_next_fn,
    cleanup_genlm,
    load_transducer,
    setup_genlm,
)
from .data import load_wikitext_sequences, load_fasta_sequences
from .storage import safe_load_pickle, file_lock, atomic_pickle_dump

# line_profiler support: no-op when not running under kernprof
try:
    profile
except NameError:
    def profile(fn):
        return fn


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

@profile
async def run_eval(
    model_name: str,
    transducer_name: str,
    impl: str,
    split: str = "test",
    paragraphs: int = 4,
    max_bytes: int | None = None,
    prune_threshold: float = 0.001,
    max_steps: int | None = None,
    skip_combined_univ: bool = True,
    skip_target_universal: bool = True,
    stop_epsilon_mass: float = 0.01,
    candidate_threshold: int = 100,
    prune_threshold_alpha: float = 0.0,
    max_prune_mass: float = 1.0,
    max_candidates: int | None = None,
    max_logp_next_beams: int | None = None,
    ignore_remainder: bool | None = None,
    expand_threshold: int = 3,  # shared by old + new presets
    output_file: str | None = None,
    seed: int = 80808,
    use_vllm: bool = False,
    torch_dtype: str = "fp32",
    max_model_len: int | None = None,
    tensor_parallel_size: int | None = None,
    fasta_file: str | None = None,
    genlm_K: int = 8,  # genlm-bytes beam size (matches the --genlm-K CLI default)
    genlm_prune: float = 0.001,
    genlm_beam_cache_mb: int = 20_000,
    cleanup_interval: int = 0,
    max_retries: int = 5,
    para_index: int | None = None,
    verbose: bool = True,
    ce_only: bool = False,
):
    """Run a benchmark: load model + transducer + data, score, save results."""
    # Force line-buffered stdout so output appears immediately under conda run / pipes
    if not sys.stdout.line_buffering:
        sys.stdout.reconfigure(line_buffering=True)

    np.random.seed(seed)

    ths = prune_threshold

    if output_file is None:
        output_file = f"results/{transducer_name}_{impl}.pkl"

    is_baseline = transducer_name == "baseline"
    needs_genlm = transducer_name in ("ptb_ported", "baseline")
    wall_t0 = time.time()

    # ── Early exit: check if all paragraphs already scored ──────────────
    # Do this BEFORE loading the LLM / transducer / data to avoid
    # spending minutes on model loading only to discover there's nothing
    # to do.  We need the output file, threshold, and paragraph count.
    _n_expected = paragraphs if para_index is None else 1
    _early_saved = safe_load_pickle(output_file, {})
    _early_stats = _early_saved.get("stats", {}).get(ths, {})
    _early_done = set(_early_stats.get("para_indices", []))
    _all_done = (
        para_index in _early_done if para_index is not None
        else len(_early_done) >= _n_expected
    )
    if _all_done:
        print()
        print("=" * 66)
        print(f"  Transduced LM Benchmark — SKIP (already complete)")
        print("=" * 66)
        print(f"  Threshold   : {ths}")
        print(f"  Paragraphs  : {sorted(_early_done)} "
              f"({len(_early_done)}/{_n_expected} done)")
        print(f"  Output      : {output_file}")
        print("=" * 66)
        return
    del _early_saved, _early_stats, _early_done, _all_done

    # ── Banner ────────────────────────────────────────────────────────────
    print()
    print("=" * 66)
    print(f"  Transduced LM Benchmark")
    print("=" * 66)
    print(f"  Model       : {model_name}")
    print(f"  Transducer  : {transducer_name}")
    print(f"  Impl        : {impl}")
    print(f"  Backend     : {'vLLM' if use_vllm else 'HuggingFace'}")
    print(f"  Dtype       : {torch_dtype}")
    print(f"  Max bytes   : {max_bytes or 'unlimited'}")
    print(f"  Paragraphs  : {paragraphs}")
    print(f"  Threshold   : {ths}")
    from .transducer import _TRANSDUCER_DEFAULTS
    effective_ms = max_steps if max_steps is not None else _TRANSDUCER_DEFAULTS.get(transducer_name, {}).get("max_steps")
    effective_ir = (
        ignore_remainder if ignore_remainder is not None
        else _TRANSDUCER_DEFAULTS.get(transducer_name, {}).get("ignore_remainder", True)
    )
    print(f"  Max steps   : {effective_ms or 'unlimited'}")
    print(f"  Combined U  : {not skip_combined_univ}")
    print(f"  Stop eps    : {stop_epsilon_mass}")
    print(f"  Cand thresh : {candidate_threshold}")
    print(f"  Prune alpha : {prune_threshold_alpha}")
    print(f"  Max prune m : {max_prune_mass}")
    print(f"  Max cands   : {max_candidates or 'unlimited'}")
    print(f"  Max lnx bms : {max_logp_next_beams or 'unlimited'}")
    print(f"  Ignore R    : {effective_ir}")
    print(f"  Expand thr  : {expand_threshold}")
    has_fallback = impl in ("logp_next", "clean_v5")
    print(f"  Fallback    : {'logp_next_clean' if has_fallback else 'none'}")
    print(f"  Max retries : {max_retries}")
    if needs_genlm:
        print(f"  Beam $ MB   : {genlm_beam_cache_mb}")
    if ce_only:
        print(f"  CE-only     : True (single-symbol scoring, no distributions)")
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

    # genlm's AsyncLM.__init__ calls decode_vocab() which requires a standard
    # GPT-2 byte tokenizer.  Non-standard tokenizers (e.g. vesteinn/gpt2-dna)
    # fail because their vocabulary can't be mapped to bytes.  Since we only
    # use the raw next_token_logprobs interface (via AsyncLMAdapter), we can
    # safely patch decode_vocab to return dummy values for these models.
    _needs_vocab_patch = transducer_name == "hf_dna2aa"
    if _needs_vocab_patch:
        import genlm.backend.llm.base as _genlm_base
        _orig_decode_vocab = _genlm_base.decode_vocab
        _genlm_base.decode_vocab = lambda tok, **kw: (
            [b""] * len(tok.get_vocab()), [""] * len(tok.get_vocab())
        )

    try:
        if use_vllm:
            engine_opts = {"dtype": _vllm_dtype_map[torch_dtype]}
            if max_model_len is not None:
                engine_opts["max_model_len"] = max_model_len
            if tensor_parallel_size is not None:
                engine_opts["tensor_parallel_size"] = tensor_parallel_size
            llm = load_model_by_name(model_name,
                                     llm_opts={"engine_opts": engine_opts})
        else:
            llm = load_model_by_name(model_name, backend="hf",
                                     llm_opts={"hf_opts": {"torch_dtype": _torch_dtype}})
    finally:
        if _needs_vocab_patch:
            _genlm_base.decode_vocab = _orig_decode_vocab

    print(f"done ({time.time() - t0:.1f}s)")

    # ── 2. Load transducer ────────────────────────────────────────────────
    if is_baseline:
        print(f"[2/4] Baseline mode (no FST) ...", end=" ", flush=True)
        t0 = time.time()
        setup = load_transducer(transducer_name, llm=llm, model_name=model_name)
        print(f"done ({time.time() - t0:.1f}s)")
        print(f"       Vieira baseline: genlm bytes → 256 output bytes")
    else:
        print(f"[2/4] Loading transducer: {transducer_name} ...", end=" ", flush=True)
        t0 = time.time()
        setup = load_transducer(transducer_name, llm=llm, model_name=model_name)

        n_states = setup.vfst.num_states
        n_arcs = sum(setup.vfst.fst.num_arcs(s) for s in setup.vfst.fst.states())
        n_out = len(setup.out_sym_to_id)
        n_univ = sum(1 for v in setup.vfst.universal_states.values() if v)
        all_univ = setup.vfst.all_universal
        print(f"done ({time.time() - t0:.1f}s)")
        print(f"       FST: {n_states:,} states, {n_arcs:,} arcs, "
              f"{n_out} output symbols")
        print(f"       Universal: {n_univ:,}/{n_states:,} "
              f"({'all' if all_univ else 'partial'})")

    # ── 3. Load data ──────────────────────────────────────────────────────
    print(f"[3/4] Loading data ...", end=" ", flush=True)
    t0 = time.time()
    if transducer_name == "hf_dna2aa":
        if fasta_file is None:
            from pathlib import Path
            _benchmark_dir = Path(__file__).resolve().parent
            fasta_file = str(
                _benchmark_dir / "dna_data"
                / "uniprotkb_accession_A0A0A0MT78_OR_access_2025_08_20.fasta"
            )
        data_seqs, total_len = load_fasta_sequences(setup, fasta_file, n=paragraphs, verbose=False)
        original_texts = [f"<protein {i+1}>" for i in range(len(data_seqs))]
    elif is_baseline:
        # Baseline: load WikiText as raw UTF-8 bytes (no FST encoding)
        from .data import load_wikitext_bytes
        data_seqs, original_texts, total_len = load_wikitext_bytes(
            split=split, n=paragraphs, max_len=max_bytes, verbose=False,
        )
    else:
        data_seqs, original_texts, total_len = load_wikitext_sequences(
            setup, split=split, n=paragraphs, verbose=False,
        )

    # Truncate if needed
    if max_bytes is not None:
        for i in range(len(data_seqs)):
            data_seqs[i] = data_seqs[i][:max_bytes]
        total_len = sum(len(s) for s in data_seqs)

    # Filter to a single paragraph if requested
    if para_index is not None:
        if para_index < 0 or para_index >= len(data_seqs):
            raise ValueError(
                f"--para-index {para_index} out of range "
                f"(loaded {len(data_seqs)} paragraphs, indices 0..{len(data_seqs)-1})"
            )
        data_seqs = [data_seqs[para_index]]
        original_texts = [original_texts[para_index]]
        total_len = len(data_seqs[0])
        _para_index_offset = para_index
        print(f"done ({time.time() - t0:.1f}s)")
        print(f"       Targeting paragraph {para_index}, {total_len} output symbols")
    else:
        _para_index_offset = 0
        print(f"done ({time.time() - t0:.1f}s)")
        print(f"       {len(data_seqs)} paragraph(s), {total_len} total output symbols")

    # Print text previews
    for j, txt in enumerate(original_texts):
        preview = txt[:80] + ("..." if len(txt) > 80 else "")
        print(f"       [{j}] ({len(data_seqs[j]):,} sym) {preview!r}")

    # ── 4. Metadata ───────────────────────────────────────────────────────
    print(f"[4/4] Scoring sequences")
    metadata = {
        "model_name": model_name,
        "transducer_name": transducer_name,
        "impl": impl,
        "split": split,
        "paragraphs": len(data_seqs),
        "max_bytes": max_bytes,
        "max_steps": max_steps,
        "text_length": total_len,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    default_payload = {"metadata": metadata, "stats": {}, "p_nexts": {}}
    saved = safe_load_pickle(output_file, default_payload)

    # ── 5. Determine which paragraphs still need scoring ──────────────────
    # Use para_indices (the actual data_seqs indices that were scored) to
    # decide which paragraphs to skip.  This is robust to skipped paragraphs:
    # if paragraphs 0,1,3 were scored (2 failed), para_indices=[0,1,3] and
    # we know to attempt 2,4,5,...  on the next run.
    saved_stats = saved.get("stats", {}).get(ths, {})
    done_indices = set(saved_stats.get("para_indices", []))
    # Which paragraph indices are we targeting in this run?
    target_indices = {i + _para_index_offset for i in range(len(data_seqs))}
    remaining_indices = target_indices - done_indices
    if done_indices & target_indices:
        print(f"\n  [ths={ths}] Already scored paragraphs: "
              f"{sorted(done_indices & target_indices)} "
              f"({len(done_indices & target_indices)}/{len(target_indices)})")

    all_run_times = []
    all_run_logps = []

    if not remaining_indices:
        print(f"  All {len(target_indices)} paragraphs done, nothing to do.")
    else:
        print(f"\n{'─'*66}")
        print(f"  Prune threshold = {ths}")
        print(f"{'─'*66}")

        # Set up GenLMRealpha (for ptb/baseline transducers).
        # For hf_realpha/hf_dna2aa, setup.lm is already an AsyncLMAdapter.
        if needs_genlm and setup.lm is None:
            await setup_genlm(
                setup, model_name, llm=llm,
                K=genlm_K, prune_threshold=genlm_prune,
                beam_cache_mb=genlm_beam_cache_mb,
            )

        # ── Warmup pass ────────────────────────────────────────────────
        # Score a short sequence from the validation split (not the
        # benchmark data) to warm up eps_closure LRU, vLLM KV cache,
        # and Python runtime before timed scoring begins.
        if not is_baseline:
            _WARMUP_SYMS = 20
            warmup_seq = []
            if transducer_name != "hf_dna2aa":
                _warmup_seqs, _, _ = load_wikitext_sequences(
                    setup, split="validation", n=1, verbose=False)
                warmup_seq = _warmup_seqs[0][:_WARMUP_SYMS] if _warmup_seqs else []
            if warmup_seq:
                print(f"\n  Warming up ({len(warmup_seq)} symbols from validation) ...",
                      end=" ", flush=True)
                t_warmup = time.time()
                _wfn, *_ = build_logp_next_fn(
                    setup, impl=impl,
                    prune_threshold=ths,
                    max_steps=max_steps,
                    skip_combined_univ=skip_combined_univ,
                    skip_target_universal=skip_target_universal,
                    stop_epsilon_mass=stop_epsilon_mass,
                    candidate_threshold=candidate_threshold,
                    prune_threshold_alpha=prune_threshold_alpha,
                    max_prune_mass=max_prune_mass,
                    max_candidates=max_candidates,
                    max_logp_next_beams=max_logp_next_beams,
                    ignore_remainder=ignore_remainder,
                    expand_threshold=expand_threshold,
                    verbose=False,
                )
                _ctx: tuple = ()
                for _sym in warmup_seq:
                    await _wfn(_ctx)
                    _ctx = _ctx + (_sym,)
                del _wfn, _ctx
                print(f"done ({time.time() - t_warmup:.1f}s)")

        for i in range(len(data_seqs)):
            orig_i = i + _para_index_offset
            if orig_i in done_indices:
                continue

            para = data_seqs[i]
            preview = original_texts[i][:60] + ("..." if len(original_texts[i]) > 60 else "")

            print(f"\n  ▸ Paragraph {orig_i} ({i+1}/{len(data_seqs)})  "
                  f"({len(para)} symbols)")
            print(f"    Text: {preview!r}")

            # Build logp_next creates a fresh TransducedLM (with clean caches)
            # from the reusable setup.  No need to reload the FST per paragraph.
            t_init = time.time()

            try:
                if is_baseline:
                    from .transducer import build_baseline_logp_next_fn
                    logp_next_fn = build_baseline_logp_next_fn(setup)
                    fallback_fn = on_fallback = on_recover = probe_fn = score_single_fn = None
                else:
                    logp_next_fn, fallback_fn, on_fallback, on_recover, probe_fn, score_single_fn = build_logp_next_fn(
                        setup, impl=impl,
                        prune_threshold=ths,
                        max_steps=max_steps,
                        skip_combined_univ=skip_combined_univ,
                        skip_target_universal=skip_target_universal,
                        stop_epsilon_mass=stop_epsilon_mass,
                        candidate_threshold=candidate_threshold,
                        prune_threshold_alpha=prune_threshold_alpha,
                        max_prune_mass=max_prune_mass,
                        max_candidates=max_candidates,
                        max_logp_next_beams=max_logp_next_beams,
                        ignore_remainder=ignore_remainder,
                        expand_threshold=expand_threshold,
                        verbose=verbose,
                    )

                if ce_only and score_single_fn is None and not is_baseline:
                    raise ValueError(
                        "--ce-only requires score_single_fn (use impl='logp_next'). "
                        "Baseline mode does not support --ce-only."
                    )

                init_time = time.time() - t_init
                print(f"    Init: {init_time:.1f}s")

                # Build cleanup function for long sequences
                genlm_cleanup = build_cleanup_fn(setup) if cleanup_interval > 0 else None

                # Run sequence scoring
                if ce_only and not is_baseline:
                    print(f"    Scoring {len(para)} positions (CE-only, single-symbol) ...")
                    if cleanup_interval > 0:
                        print(f"    (genlm cleanup every {cleanup_interval} positions)")
                    t0 = time.time()
                    try:
                        stats = await sequence_ce_only(
                            score_single_fn, para,
                            out_id_to_sym=setup.out_id_to_sym,
                            verbose=verbose,
                            cleanup_fn=genlm_cleanup,
                            cleanup_interval=cleanup_interval,
                            on_fallback=on_fallback,
                            on_recover=on_recover,
                            probe_fn=probe_fn,
                            max_retries=max_retries,
                        )
                    except RuntimeError as e:
                        if "Unrecoverable -inf" in str(e):
                            print(f"\n    ERROR: {e}")
                            print(f"    Skipping paragraph {i+1} for threshold {ths}.")
                            continue
                        raise
                else:
                    fb_msg = " (with fallback)" if fallback_fn else ""
                    print(f"    Scoring {len(para)} positions{fb_msg} ...")
                    if cleanup_interval > 0:
                        print(f"    (genlm cleanup every {cleanup_interval} positions)")
                    t0 = time.time()
                    try:
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
                    except RuntimeError as e:
                        if "Unrecoverable -inf" in str(e):
                            print(f"\n    ERROR: {e}")
                            print(f"    Skipping paragraph {i+1} for threshold {ths}.")
                            continue
                        raise
                elapsed = time.time() - t0

                # Per-paragraph summary
                times_arr = np.array(stats["times"])
                logps_arr = np.array(stats["log_probs"])
                throughput = len(para) / elapsed

                para_ce = -stats['total_logp'] / len(para)
                para_bpb = para_ce / np.log(2)
                para_bps = len(para) / elapsed if elapsed > 0 else 0

                print(f"\n    {'─'*50}")
                print(f"    Paragraph {i+1} summary:")
                print(f"      Symbols scored : {len(para)}")
                print(f"      Wall time      : {elapsed:.2f}s")
                print(f"      Bytes/sec      : {para_bps:.1f}")
                print(f"      Cross-entropy  : {para_ce:.4f} nats")
                print(f"      Bits/byte      : {para_bpb:.4f}")
                print(f"      Total log-prob : {stats['total_logp']:+.4f}")
                print(f"      Mean log-prob  : {np.mean(logps_arr):+.4f}")
                print(f"      Per-step time  : "
                      f"mean={np.mean(times_arr):.4f}s, "
                      f"std={np.std(times_arr):.4f}s, "
                      f"min={np.min(times_arr):.4f}s, "
                      f"max={np.max(times_arr):.4f}s")
                n_finite = np.sum(np.isfinite(logps_arr))
                n_inf = len(logps_arr) - n_finite
                if n_inf > 0:
                    print(f"      WARNING: {n_inf} positions with -inf log-prob")
                fb_count = stats.get("fallback_count", 0)
                if fb_count > 0:
                    print(f"      Fallbacks      : {fb_count}")
                print(f"    {'─'*50}")

                all_run_times.extend(stats["times"])
                all_run_logps.extend(stats["log_probs"])

            finally:
                # Clear TransducedLM caches (beam, cover, logp) before GC
                # so the memory is reclaimable immediately.
                _tlm = getattr(logp_next_fn, '__self__', None)
                if _tlm is not None and hasattr(_tlm, 'clear_cache'):
                    _tlm.clear_cache()
                del _tlm
                # Drop bound-method references so the TransducedLM instance
                # becomes unreachable and GC can reclaim it fully.
                logp_next_fn = fallback_fn = on_fallback = on_recover = probe_fn = score_single_fn = None
                # Clear genlm state between paragraphs to prevent
                # memory accumulation from TokenTrie / beam caches.
                if needs_genlm:
                    _gc_fn = build_cleanup_fn(setup)
                    if _gc_fn is not None:
                        _gc_fn()
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass

            # ── Incremental save ──────────────────────────────────────
            # Load previous results, merge, save, then immediately free
            # the loaded data to avoid accumulating all paragraphs' full
            # distributions in memory.
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            lock_path = output_file + ".lock"

            with file_lock(lock_path):
                prev = safe_load_pickle(output_file, default_payload)
                meta_out = prev.get("metadata", metadata)
                stats_out = prev.get("stats", {})
                pnext_out = prev.get("p_nexts", {})

                if ths not in pnext_out:
                    pnext_out[ths] = []
                if ths not in stats_out:
                    stats_out[ths] = {"times": [], "log_probs": [], "para_indices": []}

                if not ce_only:
                    pnext_out[ths].append([
                        dict(d) for d in stats["distributions"]
                    ])
                stats_out[ths].setdefault("para_indices", []).append(orig_i)
                stats_out[ths]["times"].extend(stats["times"])
                stats_out[ths]["log_probs"].extend(stats["log_probs"])

                atomic_pickle_dump(
                    {"metadata": meta_out, "stats": stats_out, "p_nexts": pnext_out},
                    output_file,
                )

            print(f"    Saved to {output_file}")
            del prev, meta_out, stats_out, pnext_out, stats
            gc.collect()

    # Final genlm cleanup
    if needs_genlm:
        await cleanup_genlm(setup)

    # ── Final summary ─────────────────────────────────────────────────────
    wall_elapsed = time.time() - wall_t0
    print()
    print("=" * 66)
    print(f"  Benchmark complete")
    print("=" * 66)
    print(f"  Transducer  : {transducer_name}")
    print(f"  Impl        : {impl}")
    print(f"  Model       : {model_name}")
    print(f"  Threshold   : {ths}")
    if all_run_times:
        total_syms = len(all_run_times)
        scoring_time = sum(all_run_times)
        _all_logps = np.array(all_run_logps)
        ce = -np.sum(_all_logps) / total_syms
        bpb = ce / np.log(2)
        bps = total_syms / scoring_time if scoring_time > 0 else 0
        print(f"  Symbols     : {total_syms}")
        print(f"  Scoring time: {scoring_time:.2f}s")
        print(f"  Wall time   : {wall_elapsed:.1f}s")
        print(f"  Bytes/sec   : {bps:.1f}")
        print(f"  Cross-entrop: {ce:.4f} nats")
        print(f"  Bits/byte   : {bpb:.4f}")
        print(f"  Total logp  : {np.sum(_all_logps):+.4f}")
        print(f"  Mean logp   : {np.mean(_all_logps):+.4f}")
    print(f"  Output      : {output_file}")
    print("=" * 66)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark logp_next implementations on transduced LMs"
    )
    parser.add_argument(
        "--transducer", required=True,
        choices=["hf_realpha", "ptb_ported", "hf_dna2aa", "baseline"],
        help="Transducer type. 'baseline' = Vieira et al. (genlm bytes, no FST).",
    )
    parser.add_argument(
        "--impl", default="logp_next",
        help="logp_next implementation: 'logp_next' (default, batched), "
             "'clean' (per-symbol reference), or 'clean_v5' (alias for logp_next).",
    )
    parser.add_argument("--model", default="gpt2-large", help="HF model name")
    parser.add_argument("--split", default="test", help="WikiText split")
    parser.add_argument("--paragraphs", type=int, default=1, help="Number of paragraphs")
    parser.add_argument(
        "--para-index", type=int, default=None,
        help="Score only this paragraph index (0-based). "
             "Still loads --paragraphs paragraphs, then filters to this one.",
    )
    parser.add_argument("--max-bytes", type=int, default=None, help="Max output symbols per paragraph")
    parser.add_argument(
        "--prune-threshold", type=float, default=0.001,
        help="Prune threshold (single value; shell scripts loop over thresholds)",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Max beam expansion steps")
    parser.add_argument(
        "--skip-combined-univ", action="store_true", default=None,
        help="Skip combined universality check (default: per-preset)",
    )
    parser.add_argument(
        "--combined-univ", action="store_true",
        help="Enable combined universality check (override --skip-combined-univ)",
    )
    parser.add_argument(
        "--target-universal", action="store_true", default=False,
        help="Enable target-constrained universality (mini-DFA BFS). "
             "Off by default for speed. Only affects FSTs with non-covering "
             "universal beams (rare for PTB).",
    )
    parser.add_argument("--output", default=None, help="Output pickle file path")
    parser.add_argument("--seed", type=int, default=80808, help="Random seed")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM backend")
    parser.add_argument(
        "--dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
        help="Model precision (default: fp32). bf16 halves memory with "
             "fp32-range exponents. fp16 may cause NaN in genlm byte-beam.",
    )
    parser.add_argument("--fasta-file", default=None, help="FASTA file for dna2aa")
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="vLLM max_model_len (limits KV cache allocation). "
             "Required for Llama models to avoid OOM on 128K context.",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=None,
        help="vLLM tensor_parallel_size for multi-GPU inference. "
             "Splits model across N GPUs. Default: 1 (single GPU).",
    )
    parser.add_argument("--genlm-K", type=int, default=8, help="GenLMRealpha K parameter")
    parser.add_argument("--genlm-prune", type=float, default=0.001, help="GenLMRealpha prune threshold")
    parser.add_argument(
        "--genlm-beam-cache-mb", type=int, default=20_000,
        help="Memory budget in MB for the genlm beam LRU cache. "
             "The actual entry count is computed at init from the measured "
             "per-beam size (K × trie_nodes × 8 bytes). "
             "Default 20000 (20GB). Lower values reduce peak RSS. "
             "Evicted beams are rebuilt on demand via recursive _beam_for.",
    )
    parser.add_argument(
        "--stop-epsilon-mass", type=float, default=None,
        help="Early-stop expansion when frontier mass is this fraction of "
             "resolved mass. Lower = more expansion steps. Default: per-preset "
             "(uses fallback for -inf positions).",
    )
    parser.add_argument(
        "--cleanup-interval", type=int, default=5000,
        help="Clear genlm caches every N positions to prevent GPU memory "
             "exhaustion on long sequences. 0 = disabled. Default 5000. "
             "Typical value: 500.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-position output")
    parser.add_argument(
        "--ce-only", action="store_true",
        help="Single-symbol scoring mode: score only the actual next symbol "
             "at each position using one decomposition (~256x faster). "
             "Reports CE loss and bits/char but does NOT save full distributions. "
             "Incompatible with --transducer baseline.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="Max retry cycles per -inf position (default: 5). "
             "Each retry tightens expansion then retries. "
             "Set to 0 for aggressive thresholds (e.g. 0.1) where "
             "-inf is expected and retries won't help.",
    )

    # ── Pruning / decomposition parameters ────────────────────────────────
    parser.add_argument(
        "--candidate-threshold", type=int, default=None,
        help="Pivot point for adaptive pruning ramp-up (default: per-preset). "
             "When frontier < this, pruning is gentler; above, it ramps up.",
    )
    parser.add_argument(
        "--prune-threshold-alpha", type=float, default=None,
        help="Steepness of pruning increase above candidate-threshold "
             "(default: per-preset). Higher = more aggressive pruning.",
    )
    parser.add_argument(
        "--max-prune-mass", type=float, default=None,
        help="Cap on fraction of total mass pruned per step (default: per-preset).",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=None,
        help="Hard cap on frontier size (default: unlimited). "
             "If set, keeps only top-N by weight.",
    )
    parser.add_argument(
        "--max-logp-next-beams", type=int, default=None,
        help="Cap on unscored Q beams per logp_next call (fast path). "
             "Keeps top-K by logp.  Default: unlimited (exact).",
    )
    parser.add_argument(
        "--ignore-remainder", action="store_true", default=None,
        help="Ignore remainder (R) contributions. "
             "Default: per-transducer (True for PTB/hf_realpha).",
    )
    parser.add_argument(
        "--no-ignore-remainder", action="store_true",
        help="Include remainder contributions (override per-transducer default).",
    )
    parser.add_argument(
        "--expand-threshold", type=int, default=None,
        help="Skip further expansion when covering beams are this many "
             "symbols past the target and Q is non-empty (default: per-preset).",
    )

    # ── Presets ────────────────────────────────────────────────────────────
    _OLD_PRESET = dict(
        ignore_remainder=True,
        expand_threshold=3,
        stop_epsilon_mass=0.01,
        candidate_threshold=100,
        prune_threshold_alpha=0.0,
        max_prune_mass=1.0,
        skip_combined_univ=True,
    )
    _NEW_PRESET = dict(
        ignore_remainder=True,
        expand_threshold=5,
        stop_epsilon_mass=0.01,
        candidate_threshold=100,
        prune_threshold_alpha=0.0,
        max_prune_mass=1.0,
        skip_combined_univ=True,
    )
    _PRESET_DEFAULTS = {
        # New presets
        "realpha": dict(_NEW_PRESET),
        "ptb": dict(_NEW_PRESET, ignore_remainder=False, skip_combined_univ=False),
        "dna2aa": dict(_NEW_PRESET),
        # Old presets — reproducing legacy benchmarking defaults
        "old-benchmark": dict(_OLD_PRESET),
        "old-ptb": dict(_OLD_PRESET),
        "old-realpha": dict(_OLD_PRESET),
        "old-dna2aa": dict(_OLD_PRESET),
    }

    parser.add_argument(
        "--preset",
        choices=list(_PRESET_DEFAULTS.keys()),
        help="Load a named parameter preset. "
             "'realpha'/'ptb'/'dna2aa' use production-tuned adaptive pruning. "
             "'old-*' variants reproduce the legacy benchmarking defaults.",
    )

    args = parser.parse_args()

    # ── Apply presets ─────────────────────────────────────────────────────
    # Preset values fill in any CLI field the user didn't explicitly set
    # (detected via default=None on preset-affected argparse fields).
    # When no preset is given, use production-tuned defaults.
    # Use --preset old-* to reproduce legacy benchmarking behavior.
    preset = _PRESET_DEFAULTS.get(args.preset, _NEW_PRESET)

    _PRESET_FIELDS = [
        "expand_threshold", "stop_epsilon_mass", "candidate_threshold",
        "prune_threshold_alpha", "max_prune_mass",
    ]
    for field in _PRESET_FIELDS:
        if getattr(args, field) is None:
            setattr(args, field, preset[field])

    if args.ignore_remainder is None and not args.no_ignore_remainder:
        args.ignore_remainder = preset["ignore_remainder"]

    if args.skip_combined_univ is None:
        args.skip_combined_univ = preset["skip_combined_univ"]

    skip_combined = args.skip_combined_univ and not args.combined_univ

    # Resolve ignore_remainder: --ignore-remainder / --no-ignore-remainder / None (per-transducer)
    ignore_remainder = args.ignore_remainder
    if args.no_ignore_remainder:
        ignore_remainder = False

    await run_eval(
        model_name=args.model,
        transducer_name=args.transducer,
        impl=args.impl,
        split=args.split,
        paragraphs=args.paragraphs,
        max_bytes=args.max_bytes,
        prune_threshold=args.prune_threshold,
        max_steps=args.max_steps,
        skip_combined_univ=skip_combined,
        skip_target_universal=not args.target_universal,
        stop_epsilon_mass=args.stop_epsilon_mass,
        candidate_threshold=args.candidate_threshold,
        prune_threshold_alpha=args.prune_threshold_alpha,
        max_prune_mass=args.max_prune_mass,
        max_candidates=args.max_candidates,
        max_logp_next_beams=args.max_logp_next_beams,
        ignore_remainder=ignore_remainder,
        expand_threshold=args.expand_threshold,
        output_file=args.output,
        seed=args.seed,
        use_vllm=args.use_vllm,
        torch_dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        fasta_file=args.fasta_file,
        genlm_K=args.genlm_K,
        genlm_prune=args.genlm_prune,
        genlm_beam_cache_mb=args.genlm_beam_cache_mb,
        cleanup_interval=args.cleanup_interval,
        max_retries=args.max_retries,
        para_index=args.para_index,
        verbose=not args.quiet,
        ce_only=args.ce_only,
    )


if __name__ == "__main__":
    asyncio.run(main())
