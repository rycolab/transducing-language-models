"""
Core benchmarking loop: score a sequence position-by-position.

The logp_next function is injected as a callable, so this works with:
  - New TransducedLM's built-in logp_next
  - Old-style variants wrapped via make_variant_fn
  - Any async (context: tuple) -> Dict[int, float] callable
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np


def _deep_size(obj, seen=None) -> int:
    """Recursively measure memory of obj, deduplicating by id().

    Handles the beam cache structures: dicts, tuples, lists, ints, floats,
    and numpy arrays.  Stops recursion at numpy array contents (counted
    via .nbytes) and at types that don't contain references (int, float,
    str, bytes).
    """
    import sys
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_size(k, seen)
            size += _deep_size(v, seen)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            size += _deep_size(item, seen)
    elif hasattr(obj, 'nbytes'):
        # numpy array: getsizeof gives the object header,
        # nbytes gives the data buffer
        size += obj.nbytes
    # int, float, str, bytes, None — no children to recurse into
    return size


def _deep_cache_bytes(cache, seen=None) -> int:
    """Measure total memory of a cache dict, sharing a seen-set across caches."""
    return _deep_size(cache, seen)


def _memory_report(tlm=None) -> str:
    """One-line memory report: RSS + GPU + cache sizes + deep byte estimates."""
    parts = []

    # Process RSS (from /proc on Linux)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    parts.append(f"RSS={rss_kb / 1024:.0f}MB")
                    break
    except (OSError, ValueError):
        pass

    # GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            parts.append(f"GPU={alloc:.0f}/{reserved:.0f}MB")
    except ImportError:
        pass

    # TransducedLM cache sizes
    if tlm is not None:
        bc_dict = getattr(tlm, '_beam_cache', {})
        cbc_dict = getattr(tlm, '_cover_beam_cache', {})
        lc_dict = getattr(tlm, '_logp_cache', {})
        parts.append(f"beam_cache={len(bc_dict)}")
        parts.append(f"cover_cache={len(cbc_dict)}")
        parts.append(f"logp_cache={len(lc_dict)}")

        # Deep byte estimates (shared seen-set deduplicates across caches)
        try:
            seen = set()
            bc_mb = _deep_cache_bytes(bc_dict, seen) / 1024**2
            cbc_mb = _deep_cache_bytes(cbc_dict, seen) / 1024**2
            lc_mb = _deep_cache_bytes(lc_dict, seen) / 1024**2
            parts.append(f"beam_MB={bc_mb:.0f}")
            parts.append(f"cover_MB={cbc_mb:.0f}")
            parts.append(f"logp_MB={lc_mb:.0f}")
        except Exception:
            pass

        # PSCache size (interned powerstates)
        ps = getattr(tlm, 'ps_cache', None)
        if ps is not None:
            parts.append(f"ps_cache={len(ps._ps2id)}")

        # GenLMRealpha beam cache (often the largest memory consumer)
        _lm = getattr(tlm, 'lm', None)
        # Unwrap RemappedGenLMRealpha or ByteIndexedLMAdapter
        _inner = getattr(_lm, 'inner', _lm)
        # GenLMRealpha has _beams directly; some wrappers nest further
        _genlm = _inner
        if not hasattr(_genlm, '_beams'):
            _genlm = getattr(_inner, '_genlm', None) or getattr(_inner, 'genlm', None)
        _beams = getattr(_genlm, '_beams', None) if _genlm else None
        if _beams is not None:
            parts.append(f"genlm_beams={len(_beams)}")

    return "  ".join(parts)


async def sequence_logp_next(
    logp_next_fn: Callable,
    sequence: list[int],
    out_id_to_sym: Optional[dict[int, str]] = None,
    verbose: bool = False,
    cleanup_fn: Optional[Callable] = None,
    cleanup_interval: int = 0,
    fallback_fn: Optional[Callable] = None,
    on_fallback: Optional[Callable] = None,
    on_recover: Optional[Callable] = None,
    probe_fn: Optional[Callable] = None,
    score_single_fn: Optional[Callable] = None,
    max_retries: int = 5,
) -> dict:
    """Score a sequence position-by-position using a logp_next function.

    For each position i in the sequence, calls logp_next_fn with the
    output context (sequence[:i]) and records the full distribution,
    the log-prob assigned to the actual next symbol, and wall-clock time.

    Args:
        logp_next_fn: Async callable ``(context: tuple) -> Dict[int, float]``.
            Returns a dict mapping output symbol IDs to log-probabilities.
            Can be ``new_tlm.logp_next``, a wrapped old variant, or any
            custom function matching this signature.
        sequence: List of output symbol IDs (ints) representing the
            target sequence to score.  These are pynini output label IDs,
            not raw bytes — the caller is responsible for encoding.
        out_id_to_sym: Optional mapping from output symbol IDs to string
            labels.  If provided, distributions are also stored with
            string keys for human-readable output.  If None, only
            integer-keyed distributions are stored.
        verbose: If True, print per-position timing and log-prob info.
        cleanup_fn: Optional callable for periodic genlm cache cleanup.
            Called every ``cleanup_interval`` positions to prevent
            unbounded GPU memory growth from the genlm TokenTrie.
        cleanup_interval: How often (in positions) to call cleanup_fn.
            0 means no periodic cleanup.  Typical values: 200-500.
        fallback_fn: Unused (kept for API compatibility).  Previously
            used for per-symbol fallback; retries now use the batched
            logp_next_fn which is faster at relaxed thresholds.
        on_fallback: Optional callable invoked before each retry.
            Receives the current ``context`` tuple as its sole argument.
            Typically ``tlm.tighten_expansion`` — adaptively lowers
            stop_epsilon_mass, raises max_steps, temporarily relaxes
            pruning thresholds, and evicts beam/cover-beam caches for
            the failing context so the retry has a better chance.
        on_recover: Optional no-arg callable invoked after the retry block
            completes (whether it succeeded or not).  Typically
            ``tlm.restore_pruning`` — restores the original pruning
            thresholds so subsequent positions use normal parameters.
        probe_fn: Optional async callable ``(context, sym_id) -> bool``.
            Cheap reachability check: returns True if the target symbol
            has Q or R beams (single decomposition, no expansion loop).
            When provided, each retry probes first and only calls the
            full logp_next_fn when the probe succeeds.  This makes
            failed retries O(1 decomposition) instead of O(expansion loop).
            Typically ``tlm.probe_target``.
        score_single_fn: Optional async callable
            ``(context, sym_id) -> float``.  Targeted single-symbol
            scoring: decomposes only ``context + (sym_id,)`` instead of
            the full expansion loop over all ~256 symbols.  When provided,
            retries use this instead of ``logp_next_fn`` for ~256x speedup.
            The result is patched into the original distribution.
            Typically ``tlm.score_single_symbol``.
        max_retries: Maximum number of tighten + retry cycles per position.
            Each retry calls on_fallback() then logp_next_fn().

    Returns:
        Dict with keys:
          - ``"log_probs"``: List[float] — log-prob of each actual symbol
          - ``"distributions"``: List[Dict] — full distribution per position
              (string-keyed if out_id_to_sym given, else int-keyed)
          - ``"times"``: List[float] — wall-clock seconds per position
          - ``"total_time"``: float — total wall-clock seconds
          - ``"total_logp"``: float — sum of per-position log-probs
          - ``"fallback_count"``: int — number of positions where fallback was used
    """
    log_probs: List[float] = []
    distributions: List[Dict] = []
    times: List[float] = []

    context: tuple = ()
    total_logp = 0.0
    fallback_count = 0

    # Get TransducedLM reference for memory tracking
    _fn = logp_next_fn
    _tlm_ref = getattr(_fn, '__self__', None)
    _mem_interval = 50  # report memory every N positions

    for i, sym_id in enumerate(sequence):
        # Periodic memory report
        if verbose and i > 0 and i % _mem_interval == 0:
            print(f"    [mem @{i}] {_memory_report(_tlm_ref)}", flush=True)

        # Periodic cleanup to prevent unbounded memory growth.
        # Only clear _logp_cache (large: ~512KB/entry) and genlm beams.
        # beam_cache and cover_beam_cache are tiny (~2KB/entry) and essential
        # for incremental decomposition — clearing them forces a full
        # re-decomposition that often produces -inf and cascading failures.
        if cleanup_interval > 0 and i > 0 and i % cleanup_interval == 0:
            if _tlm_ref is not None:
                if hasattr(_tlm_ref, '_logp_cache'):
                    _tlm_ref._logp_cache.clear()
            if cleanup_fn is not None:
                cleanup_fn()
            if verbose:
                print(f"    [cleanup at position {i}]", flush=True)
        t0 = time.time()
        dist = await logp_next_fn(context)

        # Fallback: if the actual next symbol got -inf, progressively
        # tighten expansion parameters and retry with logp_next_fn itself.
        # Only fall back to fallback_fn (slow per-symbol) as a last resort.
        #
        # If after all retries the target still has -inf, raise an error:
        # scoring cannot continue because decompose(context + (sym,)) will
        # have no beams, making all subsequent distributions meaningless.
        lp = dist.get(sym_id, float("-inf"))
        used_fallback = False
        if on_fallback is not None and not np.isfinite(lp):
            for retry in range(max_retries):
                # Dual-track: first half keeps cover beams (extends from
                # nearby, works for position 780-style failures), second
                # half evicts them (rebuilds from further back, works for
                # position 148-style failures where beams are too sparse).
                evict_cover = retry >= min(3, max_retries // 2)
                on_fallback(context, evict_cover=evict_cover)
                fallback_count += 1
                used_fallback = True

                # Evict stale extended-context cache so retries recompute
                # with relaxed thresholds instead of hitting cached -inf.
                if _tlm_ref is not None and score_single_fn is not None:
                    ext_key = tuple(context) + (sym_id,)
                    _tlm_ref._beam_cache.pop(ext_key, None)
                    _tlm_ref._cover_beam_cache.pop(ext_key, None)

                # Probe: cheap reachability check (single decomposition,
                # no expansion loop).  If the target is not yet reachable
                # at this threshold, skip the expensive logp_next call.
                if probe_fn is not None:
                    reachable = await probe_fn(context, sym_id)
                    if not reachable:
                        if verbose:
                            print(f"    [{i:4d}] -inf, probe miss {retry+1}/{max_retries}",
                                  flush=True)
                        continue

                if verbose:
                    print(f"    [{i:4d}] -inf, retry {retry+1}/{max_retries}...",
                          flush=True)
                # Targeted retry: score only the target symbol (1 decomposition)
                # instead of the full expansion loop (~256 symbols).  The
                # probe already called decompose(context, cache_result=True),
                # so score_single_symbol extends from cached beams.
                # With cache_result=True, the extended beams survive
                # restore_pruning and provide continuity for the next position.
                if score_single_fn is not None:
                    single_lp = await score_single_fn(context, sym_id)
                    if np.isfinite(single_lp):
                        # score_single_fn returns PREFIX probability
                        # log p_Y→(ctx·sym); convert to conditional via
                        # prefix_mass(ctx) so it's compatible with the
                        # normalized dist from logp_next_fn.
                        _ssf_tlm = getattr(score_single_fn, '__self__', None)
                        if _ssf_tlm is not None and hasattr(_ssf_tlm, 'prefix_mass'):
                            ctx_mass = await _ssf_tlm.prefix_mass(context)
                            lp = single_lp - ctx_mass
                        else:
                            lp = single_lp  # fallback: use raw prefix prob
                        dist[sym_id] = lp
                else:
                    dist = await logp_next_fn(context)
                    lp = dist.get(sym_id, float("-inf"))
                if np.isfinite(lp):
                    break
            # Last resort: re-run full logp_next with relaxed pruning
            # (before on_recover restores original thresholds).  The
            # expansion loop in logp_next discovers paths through the FST
            # via input symbols — this finds output-reachable paths that
            # the single-step score_single_fn decomposition can miss.
            if not np.isfinite(lp):
                if verbose:
                    print(f"    [{i:4d}] targeted retries exhausted, "
                          f"falling back to full logp_next...", flush=True)
                dist = await logp_next_fn(context)
                lp = dist.get(sym_id, float("-inf"))
                if np.isfinite(lp):
                    used_fallback = True
                    fallback_count += 1
                    if verbose:
                        print(f"    [{i:4d}] logp_next recovered: logp={lp:+.4f}",
                              flush=True)

            # Restore pruning thresholds after retry block (temporary relaxation)
            if on_recover is not None:
                on_recover()

            # After recovery, evict _beam_cache entries (R,Q computed
            # with relaxed thresholds) but KEEP all _cover_beam_cache
            # entries.  Cover beams are starting points for incremental
            # decomposition — evicting ctx_key forces the next position
            # to walk back past the problematic position and re-traverse
            # it with original thresholds, causing the same pruning
            # failure → cascading -inf.  Keeping relaxed-threshold cover
            # beams is safe: they're a superset, and the next step's
            # original-threshold pruning naturally trims excess beams.
            if np.isfinite(lp) and _tlm_ref is not None:
                ctx_key = tuple(context)
                _tlm_ref._beam_cache.pop(ctx_key, None)
                ext_key = ctx_key + (sym_id,)
                _tlm_ref._beam_cache.pop(ext_key, None)
                if verbose:
                    print(f"    [{i:4d}] evicted fallback caches", flush=True)

        # Fatal: if target symbol is still -inf after all retries, scoring
        # cannot continue — decompose(context+(sym,)) would have no beams.
        if not np.isfinite(lp):
            sym_name = (out_id_to_sym.get(sym_id, str(sym_id))
                        if out_id_to_sym else str(sym_id))
            raise RuntimeError(
                f"Unrecoverable -inf at position {i} (symbol {sym_name!r}, "
                f"id={sym_id}) after {max_retries} retries. "
                f"Context length={len(context)}. "
                f"The pruning threshold is too aggressive for this sequence — "
                f"try a lower --prune-threshold or more --max-retries."
            )

        elapsed = time.time() - t0

        context = context + (sym_id,)
        times.append(elapsed)

        # Sliding-window cache eviction: bound memory by removing entries
        # for contexts older than the window.  Safe because:
        #   _beam_cache: exact-match only; old entries are dead (ce_only
        #       reuses 1 position back, covered by beam_window=2)
        #   _cover_beam_cache: prefix-walk; retries walk back up to 32
        #       positions (covered by cover_window=40)
        if _tlm_ref is not None and hasattr(_tlm_ref, 'evict_old_caches'):
            _tlm_ref.evict_old_caches(len(context))

        log_probs.append(lp)
        total_logp += lp

        # Store distribution (string-keyed if mapping available)
        if out_id_to_sym is not None:
            str_dist = {
                out_id_to_sym.get(sid, str(sid)): v
                for sid, v in dist.items()
            }
            distributions.append(str_dist)
        else:
            distributions.append(dict(dist))

        if verbose:
            sym_name = out_id_to_sym.get(sym_id, str(sym_id)) if out_id_to_sym else str(sym_id)
            bps = (i + 1) / sum(times)
            fb_tag = " [fallback]" if used_fallback else ""
            # Try to get |Q|/|R| from engine timer
            _fn = logp_next_fn
            _tlm_v = getattr(_fn, '__self__', None)
            _tmr = getattr(_tlm_v, 'timer', None) if _tlm_v else None
            qr_tag = ""
            if _tmr is not None:
                qr_tag = f"  |Q|={_tmr.last_n_q} |R|={_tmr.last_n_r}"
            print(
                f"    [{i:4d}] {sym_name:>6s}  logp={lp:+.4f}  "
                f"cumul={total_logp:+.4f}  {elapsed:.3f}s  ({bps:.1f} sym/s){qr_tag}{fb_tag}",
                flush=True,
            )

    total_time = sum(times)

    # Final memory report
    if verbose:
        print(f"\n    [mem final] {_memory_report(_tlm_ref)}", flush=True)

    # Print engine timer summary if available
    if verbose:
        # Try to find the timer on the TransducedLM via the bound method
        _fn = logp_next_fn
        _tlm = getattr(_fn, '__self__', None)
        _timer = getattr(_tlm, 'timer', None) if _tlm else None
        if _timer is not None:
            print(f"\n    {_timer.summary()}", flush=True)

        # Print LM timer (cache hit/miss breakdown)
        _lm_timer = getattr(_tlm, '_lm_timer', None) if _tlm else None
        if _lm_timer is not None:
            _lt = _lm_timer
            _avg = f" ({1000*_lt['t_lm']/_lt['n_scored']:.1f}ms/call)" if _lt['n_scored'] else ""
            print(f"    lm_timer: {_lt['n_scored']} scored, "
                  f"{_lt['n_cached']} cached, "
                  f"t_lm={_lt['t_lm']:.2f}s{_avg}",
                  flush=True)

        # Print GenLMRealpha internal timer (may be wrapped in RemappedGenLMRealpha)
        _lm_obj = getattr(_tlm, 'lm', None) if _tlm else None
        _remap_obj = _lm_obj if (_lm_obj and hasattr(_lm_obj, 'inner')) else None
        if _remap_obj is not None:
            _lm_obj = _remap_obj.inner  # unwrap RemappedGenLMRealpha
        _gt = getattr(_lm_obj, '_genlm_timer', None) if _lm_obj else None
        if _gt is not None:
            _total = _gt['t_beam'] + _gt['t_logp_next'] + _gt['t_materialize']
            print(f"    genlm_timer ({_gt['n_calls']} calls, {_total:.2f}s): "
                  f"beam={_gt['t_beam']:.2f}s({_gt['n_beam_miss']} misses) "
                  f"logp_next={_gt['t_logp_next']:.2f}s "
                  f"materialize={_gt['t_materialize']:.2f}s",
                  flush=True)

        # Print RemappedGenLMRealpha remap timer
        _rt = getattr(_remap_obj, '_remap_timer', None) if _remap_obj else None
        if _rt is not None:
            _total = _rt['t_ctx_remap'] + _rt['t_inner'] + _rt['t_arr_remap']
            print(f"    remap_timer ({_rt['n_calls']} calls, {_total:.2f}s): "
                  f"ctx_remap={_rt['t_ctx_remap']:.2f}s "
                  f"inner={_rt['t_inner']:.2f}s "
                  f"arr_remap={_rt['t_arr_remap']:.2f}s",
                  flush=True)

        # Print decompose phase breakdown if available
        _dpt = getattr(_tlm, '_decompose_phase_timer', None) if _tlm else None
        if _dpt is not None and _dpt['n_steps'] > 0:
            _total = sum(_dpt[k] for k in ['t_collect', 't_score', 't_build', 't_prune', 't_materialize'])
            if _total > 0:
                print(f"    decompose_phases ({_dpt['n_steps']} steps, {_total:.2f}s): "
                      f"collect={_dpt['t_collect']:.2f}s({_dpt['t_collect']/_total*100:.0f}%) "
                      f"score={_dpt['t_score']:.2f}s({_dpt['t_score']/_total*100:.0f}%) "
                      f"build={_dpt['t_build']:.2f}s({_dpt['t_build']/_total*100:.0f}%) "
                      f"prune={_dpt['t_prune']:.2f}s({_dpt['t_prune']/_total*100:.0f}%) "
                      f"materialize={_dpt['t_materialize']:.2f}s({_dpt['t_materialize']/_total*100:.0f}%)",
                      flush=True)
                print(f"      beams={_dpt['n_beams_total']} candidates={_dpt['n_candidates']} survivors={_dpt['n_survivors']}",
                      flush=True)

        # Print eps_closure cache stats
        _vfst = getattr(_tlm, 'vfst', None) if _tlm else None
        if _vfst is not None and hasattr(_vfst.eps_closure, 'cache_info'):
            _ci = _vfst.eps_closure.cache_info()
            _total_calls = _ci.hits + _ci.misses
            _hit_rate = 100 * _ci.hits / _total_calls if _total_calls else 0
            print(f"    eps_closure_cache: {_ci.hits} hits, {_ci.misses} misses "
                  f"({_hit_rate:.1f}% hit rate), size={_ci.currsize}/{_ci.maxsize}",
                  flush=True)

    return {
        "log_probs": log_probs,
        "distributions": distributions,
        "times": times,
        "total_time": total_time,
        "total_logp": total_logp,
        "fallback_count": fallback_count,
    }


async def sequence_ce_only(
    score_single_fn: Callable,
    sequence: list[int],
    out_id_to_sym: Optional[dict[int, str]] = None,
    verbose: bool = False,
    cleanup_fn: Optional[Callable] = None,
    cleanup_interval: int = 0,
    on_fallback: Optional[Callable] = None,
    on_recover: Optional[Callable] = None,
    probe_fn: Optional[Callable] = None,
    max_retries: int = 5,
) -> dict:
    """Score a sequence position-by-position using single-symbol decomposition.

    Instead of computing the full next-symbol distribution (~256 decompositions
    per position), this calls ``score_single_fn(context, sym_id)`` which
    performs a single decomposition per position.  Only the conditional
    log-probability of the actual next symbol is recorded — no distributions
    are saved.

    ``score_single_symbol`` returns the **prefix** probability
    ``log p_Y→(context·sym_id)`` — the total mass of all output strings
    with that prefix.  To get the **conditional** log-probability needed
    for cross-entropy, we compute::

        log p(y_{t+1} | y_{1:t}) = log p_Y→(y_{1:t+1}) - log p_Y→(y_{1:t})

    At each position, ``prefix_mass(context)`` supplies the denominator
    (a cache hit since ``score_single_symbol`` already decomposes
    ``context`` internally) and ``score_single_symbol`` supplies the
    numerator.

    This is ~256x faster than ``sequence_logp_next`` but only produces
    cross-entropy loss, not full distributions for JSD comparisons.

    Args:
        score_single_fn: Async callable ``(context, sym_id) -> float``.
            Typically ``tlm.score_single_symbol``.
        sequence: List of output symbol IDs to score.
        out_id_to_sym: Optional mapping for verbose logging.
        verbose: Print per-position info.
        cleanup_fn: Optional periodic cache cleanup callable.
        cleanup_interval: How often to call cleanup_fn (0 = disabled).
        on_fallback: Optional callable for retry tightening (receives context).
        on_recover: Optional callable to restore pruning after retries.
        probe_fn: Optional async ``(context, sym_id) -> bool`` reachability check.
        max_retries: Max retry cycles per -inf position.

    Returns:
        Dict with keys: ``log_probs``, ``distributions`` (empty list),
        ``times``, ``total_time``, ``total_logp``, ``fallback_count``.
    """
    log_probs: List[float] = []
    times: List[float] = []

    context: tuple = ()
    total_logp = 0.0
    fallback_count = 0

    _fn = score_single_fn
    _tlm_ref = getattr(_fn, '__self__', None)
    _mem_interval = 50

    # prefix_mass_fn computes log p_Y→(context) — the denominator for
    # conditional probabilities.  At position 0 we must call it explicitly;
    # for subsequent positions we reuse the previous ext_mass (which IS
    # the prefix mass of the extended context — same decomposition, same
    # R+Q sum).  This eliminates one decompose + LM-call round per position.
    if _tlm_ref is not None and hasattr(_tlm_ref, 'prefix_mass'):
        _prefix_mass_fn = _tlm_ref.prefix_mass
    else:
        _prefix_mass_fn = None

    # logp_next_fn: full-distribution fallback for unrecoverable -inf.
    # score_single_symbol only decomposes one extension — it depends on
    # cover beams surviving pruning at earlier positions.  logp_next runs
    # an expansion loop that discovers paths through the FST via input
    # symbols, finding output reachable paths that the single-step BFS
    # decomposition misses.  ~256x slower but almost always recovers.
    _logp_next_fn = getattr(_tlm_ref, 'logp_next', None) if _tlm_ref else None

    # Carry forward: previous ext_mass becomes next ctx_mass, avoiding a
    # redundant prefix_mass() call.  None means "must compute fresh".
    _prev_ext_mass: Optional[float] = None

    for i, sym_id in enumerate(sequence):
        if verbose and i > 0 and i % _mem_interval == 0:
            print(f"    [mem @{i}] {_memory_report(_tlm_ref)}", flush=True)

        if cleanup_interval > 0 and i > 0 and i % cleanup_interval == 0:
            if _tlm_ref is not None:
                if hasattr(_tlm_ref, '_logp_cache'):
                    _tlm_ref._logp_cache.clear()
            if cleanup_fn is not None:
                cleanup_fn()
            if verbose:
                print(f"    [cleanup at position {i}]", flush=True)
            # After cleanup, cached decompositions are gone — must recompute
            _prev_ext_mass = None

        t0 = time.time()

        # Compute prefix mass for context (denominator) and extended
        # (numerator).  score_single_fn returns log p_Y→(context·sym_id);
        # prefix_mass returns log p_Y→(context).  The conditional is
        # the difference.
        #
        # Optimization: at position i>0, the previous score_single_symbol
        # returned log p_Y→(prev_context·prev_sym) = log p_Y→(context),
        # which is exactly the ctx_mass we need.  Reuse it instead of
        # calling prefix_mass (which re-decomposes + re-scores R beams).
        if _prev_ext_mass is not None:
            ctx_mass = _prev_ext_mass
        elif _prefix_mass_fn is not None:
            ctx_mass = await _prefix_mass_fn(context)
        else:
            ctx_mass = 0.0  # assume total mass ≈ 1 if no TLM reference

        ext_mass = await score_single_fn(context, sym_id)

        # Conditional log-probability
        lp = ext_mass - ctx_mass

        used_fallback = False
        if on_fallback is not None and not np.isfinite(ext_mass):
            for retry in range(max_retries):
                evict_cover = retry >= min(3, max_retries // 2)
                on_fallback(context, evict_cover=evict_cover)
                fallback_count += 1
                used_fallback = True

                # Evict stale extended-context cache so retries recompute
                # with relaxed thresholds instead of hitting cached -inf.
                if _tlm_ref is not None:
                    ext_key = tuple(context) + (sym_id,)
                    _tlm_ref._beam_cache.pop(ext_key, None)
                    _tlm_ref._cover_beam_cache.pop(ext_key, None)

                if probe_fn is not None:
                    reachable = await probe_fn(context, sym_id)
                    if not reachable:
                        if verbose:
                            print(f"    [{i:4d}] -inf, probe miss {retry+1}/{max_retries}",
                                  flush=True)
                        continue

                if verbose:
                    print(f"    [{i:4d}] -inf, retry {retry+1}/{max_retries}...",
                          flush=True)
                ext_mass = await score_single_fn(context, sym_id)
                if np.isfinite(ext_mass):
                    # Recompute context mass with same pruning for consistency
                    if _prefix_mass_fn is not None:
                        ctx_mass = await _prefix_mass_fn(context)
                    lp = ext_mass - ctx_mass
                    break

            if on_recover is not None:
                on_recover()

            if np.isfinite(ext_mass) and _tlm_ref is not None:
                ctx_key = tuple(context)
                _tlm_ref._beam_cache.pop(ctx_key, None)
                ext_key = ctx_key + (sym_id,)
                _tlm_ref._beam_cache.pop(ext_key, None)
                if verbose:
                    print(f"    [{i:4d}] evicted fallback caches", flush=True)

        # Last resort: full-distribution fallback.  score_single_symbol
        # only decomposes one extension and depends on cover beams
        # surviving pruning at earlier steps.  logp_next runs the
        # expansion loop which discovers paths through the FST via input
        # symbols — this finds reachable paths that the single-step BFS
        # decomposition misses.  ~256x slower but almost always recovers.
        if not np.isfinite(ext_mass) and _logp_next_fn is not None:
            if verbose:
                print(f"    [{i:4d}] targeted retries exhausted, "
                      f"falling back to full logp_next...", flush=True)
            dist = await _logp_next_fn(context)
            lp = dist.get(sym_id, float("-inf"))
            if np.isfinite(lp):
                ext_mass = 0.0  # sentinel: lp already holds the conditional
                used_fallback = True
                fallback_count += 1
                if verbose:
                    print(f"    [{i:4d}] logp_next recovered: logp={lp:+.4f}",
                          flush=True)

        if not np.isfinite(ext_mass) or not np.isfinite(lp):
            sym_name = (out_id_to_sym.get(sym_id, str(sym_id))
                        if out_id_to_sym else str(sym_id))
            raise RuntimeError(
                f"Unrecoverable -inf at position {i} (symbol {sym_name!r}, "
                f"id={sym_id}) after {max_retries} retries + logp_next fallback. "
                f"Context length={len(context)}. "
                f"The pruning threshold is too aggressive for this sequence — "
                f"try a lower --prune-threshold or more --max-retries."
            )

        elapsed = time.time() - t0
        context = context + (sym_id,)
        times.append(elapsed)

        # Carry forward ext_mass as next position's ctx_mass.
        # After a logp_next fallback, ext_mass is a sentinel (0.0) and
        # lp already holds the conditional — don't carry it forward.
        # After retry-with-eviction, beam caches were modified so the
        # cached decomposition may not match — recompute to be safe.
        if used_fallback:
            _prev_ext_mass = None
        else:
            _prev_ext_mass = ext_mass

        # Sliding-window cache eviction (see sequence_logp_next for details)
        if _tlm_ref is not None and hasattr(_tlm_ref, 'evict_old_caches'):
            _tlm_ref.evict_old_caches(len(context))

        log_probs.append(lp)
        total_logp += lp

        if verbose:
            sym_name = out_id_to_sym.get(sym_id, str(sym_id)) if out_id_to_sym else str(sym_id)
            bps = (i + 1) / sum(times)
            fb_tag = " [fallback]" if used_fallback else ""
            print(
                f"    [{i:4d}] {sym_name:>6s}  logp={lp:+.4f}  "
                f"cumul={total_logp:+.4f}  {elapsed:.3f}s  ({bps:.1f} sym/s){fb_tag}",
                flush=True,
            )

    total_time = sum(times)

    if verbose:
        print(f"\n    [mem final] {_memory_report(_tlm_ref)}", flush=True)

    return {
        "log_probs": log_probs,
        "distributions": [],
        "times": times,
        "total_time": total_time,
        "total_logp": total_logp,
        "fallback_count": fallback_count,
    }
