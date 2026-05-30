"""
Refactored TransducedLM: composes VectorizedFST + LM scorer + Config.

Provides decomposition and logp_next using the class-based BeamDecomposition.
The LM is injected as any object with ``async logp_next_for(ctx) -> ndarray``
and an ``EOS_LM_IDX`` attribute.

logp_next uses the fast_next_dist algorithm (paper Figure 13 right):
  - Single decomposition for context y
  - Expansion of Q elements that haven't produced the next output symbol
  - Locked symbol tracking ensures mass goes to the correct output symbol
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

import numpy as np
from scipy.special import logsumexp

from .config import Config

# Beam tuple indices (inlined from transducedLM.utils.constants)
POWERSTATE, BEAM_OUT = (0, 1)
YS, LOGP = (0, 1)

from .beam_decomposition import Beam, BeamDecomposition
from .vectorized_fst import VectorizedFST

# FST advance helpers — re-exported for backward compatibility
from .fst_advance import _advance_ps, _advance_ps_unlabeled, _is_combined_universal_for_sym


def _flatten_grouped_cache(grouped_cache):
    """Unwrap grouped cover beam cache to flat {beam_key: [beam_items]} dict.

    The cache stores {beam_key: {None: [beams]}} — the None bucket holds
    boundary beams (len == n_out) and, for slow-path Q keys, behind beams
    (len < n_out) needed by is_target_universal.

    Returns:
        Flat {beam_key: [beam_items]} dict, or None if empty.
    """
    flat = {}
    for beam_key, groups in grouped_cache.items():
        beams = groups.get(None)
        if beams:
            flat[beam_key] = beams
    return flat or None


# ---------------------------------------------------------------------------
# TransducedLM
# ---------------------------------------------------------------------------


class TransducedLM:
    """Transduced language model: VectorizedFST + LM scorer + Config.

    Usage::

        vfst = VectorizedFST(pynini_fst)
        vfst.compute_universal_states()
        tlm = TransducedLM(vfst, mock_lm, config)
        dist = await tlm.logp_next(context)
    """

    def __init__(
        self,
        vfst: VectorizedFST,
        lm,
        config: Config | None = None,
    ):
        """
        Args:
            vfst: VectorizedFST with precomputed universality.
            lm: LM object with ``async logp_next_for(ctx: tuple) -> ndarray``
                and ``EOS_LM_IDX: int`` attribute.
            config: Decomposition/pruning configuration.
        """
        self.vfst = vfst
        self.lm = lm
        self.config = config or Config()

        # Caches
        self._logp_cache: dict[tuple, np.ndarray] = {}
        self._beam_cache: dict[tuple, tuple[Beam, Beam]] = {}
        self._cover_beam_cache: dict[tuple, dict] = {}

        # EOS index in LM distribution arrays
        self.EOS_LM_IDX: int = getattr(lm, "EOS_LM_IDX", 0)

        # Valid non-epsilon input symbols (precomputed for expansion loop).
        # Exclude special symbols (EOS, separators) tracked on vfst.
        _special = set(vfst.special_symbols) if vfst.special_symbols else set()
        self._input_syms: list[int] = [
            s for s in vfst.input_syms if s not in _special
        ]
        self._input_syms_arr: np.ndarray = np.array(
            self._input_syms, dtype=np.int32
        )

    # ── LM scoring (passed to BeamDecomposition) ──────────────────────────

    async def scorer(self, contexts: set[tuple]) -> dict[tuple, np.ndarray]:
        """Score a set of source contexts via the LM, with caching.

        This method is passed to BeamDecomposition as the scoring callable.
        Uses asyncio.gather for concurrent LM calls (matching the old
        gather_ctx2logdist behavior).

        Args:
            contexts: Set of source context tuples to score.

        Returns:
            Dict mapping each context to its LM log-prob distribution array.
        """
        import time as _time
        result = {}
        to_score = []
        for ctx in contexts:
            if ctx in self._logp_cache:
                result[ctx] = self._logp_cache[ctx]
            else:
                to_score.append(ctx)

        if to_score:
            _t0 = _time.perf_counter()
            if hasattr(self.lm, 'batch_logp_next_for'):
                dists = await self.lm.batch_logp_next_for(to_score)
            else:
                dists = await asyncio.gather(
                    *(self.lm.logp_next_for(ctx) for ctx in to_score)
                )
            _dt = _time.perf_counter() - _t0
            for ctx, dist in zip(to_score, dists):
                self._logp_cache[ctx] = dist
                result[ctx] = dist

            # Accumulate LM timing stats
            if not hasattr(self, '_lm_timer'):
                self._lm_timer = {'n_calls': 0, 'n_scored': 0,
                                  'n_cached': 0, 't_lm': 0.0}
            self._lm_timer['n_calls'] += 1
            self._lm_timer['n_scored'] += len(to_score)
            self._lm_timer['n_cached'] += len(result) - len(to_score)
            self._lm_timer['t_lm'] += _dt
        else:
            if not hasattr(self, '_lm_timer'):
                self._lm_timer = {'n_calls': 0, 'n_scored': 0,
                                  'n_cached': 0, 't_lm': 0.0}
            self._lm_timer['n_calls'] += 1
            self._lm_timer['n_cached'] += len(result)

        return result

    # ── LM helper (single context) ────────────────────────────────────────

    async def _lm_dist(self, ctx: tuple) -> np.ndarray:
        """Get LM distribution for a single context, with caching."""
        import time as _time
        if ctx in self._logp_cache:
            return self._logp_cache[ctx]
        _t0 = _time.perf_counter()
        dist = await self.lm.logp_next_for(ctx)
        _dt = _time.perf_counter() - _t0
        self._logp_cache[ctx] = dist
        if not hasattr(self, '_lm_timer'):
            self._lm_timer = {'n_calls': 0, 'n_scored': 0,
                              'n_cached': 0, 't_lm': 0.0}
        self._lm_timer['n_scored'] += 1
        self._lm_timer['t_lm'] += _dt
        return dist

    async def _batch_lm_dist(self, ctxs: list[tuple]) -> list[np.ndarray]:
        """Batched LM scoring with caching.

        Uses batch_logp_next_for when available (true GPU batch), otherwise
        falls back to asyncio.gather of individual calls.
        """
        import time as _time
        results = [None] * len(ctxs)
        to_score_idx = []
        to_score_ctx = []
        for i, ctx in enumerate(ctxs):
            if ctx in self._logp_cache:
                results[i] = self._logp_cache[ctx]
            else:
                to_score_idx.append(i)
                to_score_ctx.append(ctx)

        if not hasattr(self, '_lm_timer'):
            self._lm_timer = {'n_calls': 0, 'n_scored': 0,
                              'n_cached': 0, 't_lm': 0.0}
        self._lm_timer['n_cached'] += len(ctxs) - len(to_score_ctx)

        if to_score_ctx:
            _t0 = _time.perf_counter()
            if hasattr(self.lm, 'batch_logp_next_for'):
                dists = await self.lm.batch_logp_next_for(to_score_ctx)
            else:
                dists = await asyncio.gather(
                    *(self.lm.logp_next_for(ctx) for ctx in to_score_ctx)
                )
            _dt = _time.perf_counter() - _t0
            for idx, ctx, dist in zip(to_score_idx, to_score_ctx, dists):
                self._logp_cache[ctx] = dist
                results[idx] = dist
            self._lm_timer['n_scored'] += len(to_score_ctx)
            self._lm_timer['t_lm'] += _dt

        return results

    # ── Decomposition ─────────────────────────────────────────────────────

    @profile
    async def decompose(
        self,
        out_tokens,
        cache_result: bool = True,
    ) -> tuple[Beam, Beam]:
        """Run beam-search decomposition with incremental beam caching.

        When use_beam_cache is enabled and a cached beam exists for a prefix
        of out_tokens, the decomposition starts from those cached beams
        instead of from scratch.  This makes extending by one symbol O(1 step)
        instead of O(n_ctx steps).

        Returns (remainder, quotient) beam dicts.
        """
        key = tuple(out_tokens)
        if cache_result and key in self._beam_cache:
            return self._beam_cache[key]

        # Look for cached cover beams from a prefix
        initial_buckets = None
        if self.config.use_beam_cache and key:
            for i in range(len(key) - 1, -1, -1):
                prefix = key[:i]
                if prefix in self._cover_beam_cache:
                    initial_buckets = _flatten_grouped_cache(
                        self._cover_beam_cache[prefix])
                    break

        decomp = BeamDecomposition(self.vfst, self.scorer, self.config)
        R, Q, cover = await decomp.decompose(out_tokens, initial_buckets)

        # Accumulate decompose phase timing for profiling
        if hasattr(decomp, '_phase_timer'):
            if not hasattr(self, '_decompose_phase_timer'):
                self._decompose_phase_timer = {
                    'n_steps': 0,
                    't_collect': 0.0, 't_score': 0.0, 't_build': 0.0,
                    't_prune': 0.0, 't_materialize': 0.0,
                    'n_beams_total': 0, 'n_candidates': 0, 'n_survivors': 0,
                }
            dpt = self._decompose_phase_timer
            pt = decomp._phase_timer
            for k in pt:
                dpt[k] += pt[k]

        if cache_result:
            self._beam_cache[key] = (R, Q)
            if self.config.use_beam_cache and cover:
                self._cover_beam_cache[key] = cover
        return R, Q

    # ── _logp_next_unbatched (expansion loop without PSCache/batching) ────

    async def _logp_next_unbatched(
        self,
        context: Optional[tuple] = None,
    ) -> Dict[int, float]:
        """Expansion loop WITHOUT batched LM calls or PSCache.

        Kept as a reference implementation for correctness tests.
        For production use, prefer ``logp_next`` (batched loop).

        Delegates to UnbatchedFastNextDist (fast_next_dist.py).
        """
        if not hasattr(self, '_unbatched_engine'):
            from .fast_next_dist import UnbatchedFastNextDist
            self._unbatched_engine = UnbatchedFastNextDist(self)
        return await self._unbatched_engine.compute(context)

    # ── Expansion-based EOS helper ────────────────────────────────────────


    # ── logp_next_clean (per-symbol decomposition) ────────────────────────

    async def logp_next_clean(
        self,
        context: Optional[tuple] = None,
    ) -> Dict[int, float]:
        """Compute logp_next via per-symbol decomposition (slow but simple).

        For each output symbol y: decompose context+(y,), collect Q+R mass.
        """
        context = () if context is None else tuple(context)
        n_ctx = len(context)
        eos_lm_idx = self.EOS_LM_IDX

        all_ids = self.vfst.all_output_labels
        scores: Dict[int, float] = {}

        # Pre-decompose context to populate _cover_beam_cache[context].
        # Without this, each per-symbol decompose(context+(sid,)) walks
        # further back in the cache and redundantly re-decomposes the
        # same prefix — 250× slower when cover beams were evicted.
        remainder_ctx, quotient_ctx = await self.decompose(
            context, cache_result=True
        )

        for sid in all_ids:
            extended = context + (int(sid),)
            remainder, quotient = await self.decompose(extended, cache_result=True)

            q_logp = (
                logsumexp([key[LOGP] for key in quotient])
                if quotient
                else -np.inf
            )

            r_logp = -np.inf
            if remainder and not self.config.ignore_remainder:
                r_parts = []
                for key in remainder:
                    src_ctx = key[YS]
                    beam_logp = key[LOGP]
                    lm_dist = await self.lm.logp_next_for(src_ctx)
                    logp_eos = float(lm_dist[eos_lm_idx])
                    r_parts.append(beam_logp + logp_eos)
                r_logp = logsumexp(r_parts) if r_parts else -np.inf

            scores[int(sid)] = float(logsumexp([q_logp, r_logp]))

        # EOS via residual: p_Y(context) = p_Y→(context) - Σ_y p_Y→(context·y)
        # where p_Y→(context) = Q prefix mass + R string mass from decompose(context).
        # This avoids double-counting (samuel) and missing eps-output EOS (delete_b).
        eos_sym_id = int(self.vfst.eos_out)
        if self.config.ignore_remainder:
            scores[eos_sym_id] = -np.inf
        else:
            # Reuse remainder_ctx, quotient_ctx from pre-decompose above.
            # p_Y→(context) = Σ_Q p_X→(x) + Σ_R p_X(x)
            prefix_parts = []
            for key in quotient_ctx:
                prefix_parts.append(key[LOGP])  # Q: prefix probability
            for key in remainder_ctx:
                src_ctx = key[YS]
                lm_dist = await self.lm.logp_next_for(src_ctx)
                logp_eos = float(lm_dist[eos_lm_idx])
                prefix_parts.append(key[LOGP] + logp_eos)  # R: string probability
            logp_prefix = logsumexp(prefix_parts) if prefix_parts else -np.inf
            # Σ_y score[y] (unnormalized, before EOS)
            sym_vals = [v for k, v in scores.items() if np.isfinite(v)]
            logp_syms = logsumexp(sym_vals) if sym_vals else -np.inf
            # Residual in log-space: log(exp(a) - exp(b)) where a >= b
            if not np.isfinite(logp_prefix) or not np.isfinite(logp_syms):
                scores[eos_sym_id] = logp_prefix if np.isfinite(logp_prefix) else -np.inf
            else:
                diff = logp_syms - logp_prefix
                if diff > 0:
                    # Numerical noise: syms slightly exceed prefix, clamp to ~0
                    scores[eos_sym_id] = -np.inf
                else:
                    scores[eos_sym_id] = float(logp_prefix + np.log1p(-np.exp(diff)))

        vals = np.array(list(scores.values()), dtype=np.float64)
        finite_mask = np.isfinite(vals)
        if finite_mask.any():
            logZ = logsumexp(vals[finite_mask])
        else:
            logZ = 0.0

        return {sid: float(lp - logZ) for sid, lp in scores.items()}

    # ── Sequence scoring ──────────────────────────────────────────────────
    @profile
    async def sequence_logp_next(
        self,
        sequence: list[int],
        logp_next_fn=None,
        verbose: bool = False,
    ) -> dict:
        """Score a sequence position-by-position.

        Convenience wrapper around ``benchmark.sequence_logp_next`` that
        binds ``self.logp_next`` (or a custom function) and the FST's
        output symbol mapping.

        Args:
            sequence: List of output symbol IDs to score.
            logp_next_fn: Optional custom logp_next function.  If None,
                uses ``self.logp_next``.  Must have signature
                ``async (context: tuple) -> Dict[int, float]``.
            verbose: Print per-position timing info.

        Returns:
            Stats dict (see ``benchmark.sequence_logp_next``).
        """
        from .benchmark.sequence import sequence_logp_next as _seq_logp_next

        fn = logp_next_fn if logp_next_fn is not None else self.logp_next
        return await _seq_logp_next(
            fn,
            sequence,
            out_id_to_sym=self.vfst._out_id_to_sym,
            verbose=verbose,
        )

    # ── FST application ───────────────────────────────────────────────────

    def apply(
        self,
        input_str: str = None,
        input_tokens: list[str] = None,
        tokenizer=None,
    ) -> list[str]:
        """Transduce input through the FST to get output symbol strings.

        Args:
            input_str: Text to transduce.  If tokenizer is given, text is
                first tokenized; otherwise UTF-8 bytes are used as input.
            input_tokens: Pre-tokenized input symbol strings.  Mutually
                exclusive with input_str.
            tokenizer: Optional HF tokenizer.  If given, ``input_str`` is
                encoded with it; otherwise bytes are used.

        Returns:
            List of output symbol string labels.
        """
        from .benchmark.setup import apply_fst

        if input_str is not None and input_tokens is not None:
            raise ValueError("Provide input_str or input_tokens, not both.")
        if input_tokens is None:
            if input_str is None:
                raise ValueError("Provide input_str or input_tokens.")
            if tokenizer is not None:
                ids = tokenizer.encode(input_str, add_special_tokens=False)
                input_tokens = [str(tid) for tid in ids]
            else:
                input_tokens = [str(b) for b in input_str.encode("utf-8")]

        return apply_fst(self.vfst.fst, input_tokens, eps_id=self.vfst.eps_id)

    # ── logp_next (batched expansion with PSCache) ──────────────────────
    @profile
    async def logp_next(
        self,
        context: Optional[tuple] = None,
    ) -> Dict[int, float]:
        """Default logp_next: PSCache + dense mass + batched LM + dedup.

        Uses asyncio.gather for concurrent LM calls and a persistent
        PSCache for cached powerstate transitions across calls.

        Delegates to BatchedFastNextDist (fast_next_dist_batched.py).

        If the result is all-inf (no valid mass), internally retries with
        progressively relaxed pruning thresholds before backtracking.
        """
        if not hasattr(self, '_batched_engine'):
            from .fast_next_dist_batched import BatchedFastNextDist
            self._batched_engine = BatchedFastNextDist(self)

        return await self._batched_engine.compute(context)

    @property
    def timer(self):
        """Access the batched engine's wall-clock timer."""
        if hasattr(self, '_batched_engine'):
            return self._batched_engine.timer
        return None

    @timer.setter
    def timer(self, value):
        """Set the batched engine's timer."""
        if not hasattr(self, '_batched_engine'):
            from .fast_next_dist_batched import BatchedFastNextDist
            self._batched_engine = BatchedFastNextDist(self)
        self._batched_engine.timer = value

    @property
    def ps_cache(self):
        """Access the batched engine's PSCache."""
        if hasattr(self, '_batched_engine'):
            return self._batched_engine.psc
        return None

    @ps_cache.setter
    def ps_cache(self, value):
        """Set the batched engine's PSCache."""
        if not hasattr(self, '_batched_engine'):
            from .fast_next_dist_batched import BatchedFastNextDist
            self._batched_engine = BatchedFastNextDist(self)
        self._batched_engine.psc = value

    @ps_cache.deleter
    def ps_cache(self):
        """Delete the batched engine's PSCache."""
        if hasattr(self, '_batched_engine'):
            self._batched_engine.psc = None

    # Backward-compatible aliases
    logp_next_v5 = logp_next
    _v5_timer = timer
    _v5_ps_cache = ps_cache

    # ── Probe (cheap target reachability check for retries) ─────────────

    async def probe_target(self, context, sym_id) -> bool:
        """Check if target symbol is reachable with current pruning params.

        Two-step approach:
        1. ``decompose(context, cache_result=True)`` — populates beam and
           cover-beam caches so the subsequent ``logp_next`` (if probe
           succeeds) gets a cache hit on decompose(context).
        2. ``decompose(context + (sym_id,), cache_result=False)`` — checks
           reachability without caching (the probe result becomes stale
           after each ``tighten_expansion()`` call).  Uses the cover-beam
           cache from step 1 so only one extension step is needed.

        Returns True if Q or R is non-empty for context + (sym_id,).

        Used during backtracking retries: probe is O(2 decompositions)
        vs full logp_next which is O(expansion loop over all symbols).
        Failed retries skip the expansion loop entirely.
        """
        ctx = tuple(context)
        # Step 1: populate cache for context (logp_next will reuse this)
        await self.decompose(ctx, cache_result=True)
        # Step 2: check if target is reachable (don't cache stale result)
        extended = ctx + (int(sym_id),)
        R, Q = await self.decompose(extended, cache_result=False)
        return bool(Q) or bool(R)

    # ── Prefix mass computation ──────────────────────────────────────────

    async def prefix_mass(self, context: tuple) -> float:
        """Compute total probability mass for an output context.

        Returns ``log p_Y→(context)`` = Q prefix mass + R string mass,
        i.e. the total probability of all output strings having *context*
        as a prefix (including exact matches that terminate at *context*).

        This is the denominator needed to convert prefix probabilities
        from ``score_single_symbol`` into conditional probabilities::

            log p(y_{t+1} | y_{1:t}) = score_single_symbol(ctx, y_{t+1})
                                      - prefix_mass(ctx)
        """
        context = tuple(context)
        remainder, quotient = await self.decompose(context, cache_result=True)

        parts = []
        for key in quotient:
            parts.append(key[LOGP])

        if remainder and not self.config.ignore_remainder:
            for key in remainder:
                src_ctx = key[YS]
                beam_logp = key[LOGP]
                lm_dist = await self._lm_dist(src_ctx)
                logp_eos = float(lm_dist[self.EOS_LM_IDX])
                parts.append(beam_logp + logp_eos)

        return float(logsumexp(parts)) if parts else -np.inf

    # ── Targeted single-symbol scoring (for fallback retries) ────────────

    async def score_single_symbol(self, context: tuple, sym_id: int,
                                   cache_result: bool = True) -> float:
        """Score a single output symbol via one decomposition.

        Unlike ``logp_next`` which runs the full expansion loop over all
        output symbols, or ``logp_next_clean`` which decomposes all ~256
        symbols, this only decomposes ``context + (sym_id,)`` — one
        decomposition instead of ~256.

        When ``cache_result=True`` (default), the decomposition result is
        cached in ``_beam_cache`` and ``_cover_beam_cache``.  This is
        important for fallback retries: the cached beams at the extended
        context become the starting point for the next position's
        decomposition.  ``restore_pruning()`` only evicts entries at
        ``context`` length and shorter, so ``context + (sym_id,)``
        entries survive and provide continuity.

        Args:
            context: Output context tuple (symbol IDs).
            sym_id: Target output symbol ID to score.
            cache_result: Whether to cache the decomposition result.

        Returns:
            Log prefix probability ``log p_Y→(context·sym_id)`` — the
            total mass of all output strings with prefix ``context·sym_id``.
            To get the conditional ``log p(sym_id | context)``, subtract
            ``prefix_mass(context)``.
        """
        extended = tuple(context) + (int(sym_id),)
        remainder, quotient = await self.decompose(extended,
                                                    cache_result=cache_result)

        # Quotient contribution: sum of beam log-probs (prefix probability)
        q_logp = (logsumexp([key[LOGP] for key in quotient])
                  if quotient else -np.inf)

        # Remainder contribution: beam_logp + logp(EOS) for each R beam
        r_logp = -np.inf
        if remainder and not self.config.ignore_remainder:
            r_parts = []
            for key in remainder:
                src_ctx = key[YS]
                beam_logp = key[LOGP]
                lm_dist = await self._lm_dist(src_ctx)
                logp_eos = float(lm_dist[self.EOS_LM_IDX])
                r_parts.append(beam_logp + logp_eos)
            r_logp = logsumexp(r_parts) if r_parts else -np.inf

        result = float(logsumexp([q_logp, r_logp]))

        # Evict stale cache when the result is -inf.  decompose() above
        # cached the (empty Q, empty R) result in _beam_cache[extended].
        # Without eviction, retries with relaxed pruning thresholds hit
        # the stale cached -inf and never actually recompute — making the
        # entire retry loop ineffective.
        if cache_result and not np.isfinite(result):
            self._beam_cache.pop(extended, None)
            self._cover_beam_cache.pop(extended, None)

        return result

    # ── Cache management ──────────────────────────────────────────────────

    def tighten_expansion(self, context=None,
                           eps_factor: float = 0.7, eps_floor: float = 1e-10,
                           steps_factor: float = 1.5, steps_ceil: int = 50,
                           prune_factor: float = 0.5,
                           evict_cover: bool = False):
        """Relax limits and evict caches after a fallback produces -inf.

        Reduces ``stop_epsilon_mass`` (so early-stop fires later),
        increases ``max_steps`` (so the hard cap allows more iterations),
        and temporarily relaxes pruning thresholds (so more beams survive).

        Pruning relaxation is temporary: call ``restore_pruning()`` after
        recovery to revert to the original thresholds.  This matches the
        old code's ``backtrack()`` which used a config copy.

        When ``context`` is provided (the failing output context), also
        performs backward walking of the cover beam cache — the key
        mechanism for recovering from cascading -inf:

        1. Evicts ``_beam_cache[context]`` so the next decompose() call
           recomputes instead of returning the cached (failing) result.
           This is needed even when Q is non-empty: the Q beams may not
           produce the required target symbol.

        2. Walks ``_cover_beam_cache`` backward from ``context``, evicting
           entries progressively.  Each retry evicts one more step back
           (tracked via ``_backtrack_depth``).  This forces decompose() to
           fall back to an earlier cached prefix and re-decompose with
           relaxed pruning, potentially finding paths that were previously
           pruned away.

        This matches the old code's ``backtrack()`` function in
        ``logp_next/backtracking.py`` which walked ``_cover_beam_cache``
        backward by removing tokens one at a time, copying beams from
        shorter prefixes, and retrying with relaxed thresholds.

        Args:
            context: The output context tuple that produced -inf.  If None,
                only relaxes parameters without cache eviction.
        """
        # Save ALL original params on first tighten call
        if not hasattr(self, '_saved_prune_params'):
            self._saved_prune_params = (
                self.config.prune_threshold,
                self.config.prune_threshold_alpha,
                self.config.max_prune_mass,
                self.config.stop_epsilon_mass,
                self.config.max_steps,
            )

        # Expansion parameters — temporary (restored by restore_pruning)
        self.config.stop_epsilon_mass = max(
            self.config.stop_epsilon_mass * eps_factor, eps_floor)
        if self.config.max_steps is not None:
            self.config.max_steps = min(
                int(self.config.max_steps * steps_factor), steps_ceil)

        # Pruning parameters — temporary (like old backtrack's config copy)
        # Floor at 1e-10 to prevent unlimited beam growth at near-zero thresholds.
        self.config.prune_threshold = max(self.config.prune_threshold * prune_factor, 1e-10)
        self.config.prune_threshold_alpha = max(self.config.prune_threshold_alpha * prune_factor, 1e-10)
        self.config.max_prune_mass = max(self.config.max_prune_mass * prune_factor, 1e-10)

        # ── Context-aware cache eviction ──────────────────────────────
        # When context is provided, evict the failing position's caches
        # and progressively more predecessor entries on each retry.
        # This forces decompose() to recompute from an earlier (less-pruned)
        # prefix with the relaxed threshold, recovering beams that were
        # pruned at intermediate steps.
        #
        # Progressive depth (tracked via _backtrack_depth):
        #   retry 1: evict 1 position back
        #   retry 2: evict 2 positions back
        #   retry 3: evict 4 positions back (exponential growth)
        #   retry 4: evict 8, etc.
        #
        # Exponential growth is critical for coarse thresholds: the
        # cover_beam_cache at nearby positions was built with the original
        # (coarse) threshold, so even near-zero retry thresholds can't
        # find the target if extending from those sparse beams.  Going
        # back further reaches positions with richer beam sets.
        if context is not None:
            key = tuple(context)
            self._tighten_context = key  # for restore_pruning cache eviction

            # Separate depth tracking for beam-only vs cover eviction.
            # _backtrack_depth doubles each call (for _beam_cache eviction).
            # _cover_backtrack_depth tracks cover eviction independently —
            # reset to 1 on the first evict_cover=True call so cover
            # eviction starts shallow and ramps up gradually.  Without
            # this, _backtrack_depth is already at 32 by the time
            # evict_cover=True starts (from 10 prior evict_cover=False
            # calls), making the first cover eviction unnecessarily
            # expensive (re-decomposing 32+ BFS steps).
            if not hasattr(self, '_backtrack_depth'):
                self._backtrack_depth = 1
            else:
                self._backtrack_depth = min(self._backtrack_depth * 2, 32)

            if evict_cover:
                if not hasattr(self, '_cover_backtrack_depth'):
                    self._cover_backtrack_depth = 1
                else:
                    self._cover_backtrack_depth = min(
                        self._cover_backtrack_depth * 2, 32)
                cover_depth = self._cover_backtrack_depth
            else:
                cover_depth = 0

            # Evict _beam_cache using _backtrack_depth (always).
            # Evict _cover_beam_cache using cover_depth (only when
            # evict_cover=True, starting from 1 and doubling).
            evict_depth = max(self._backtrack_depth, cover_depth)
            for d in range(evict_depth):
                if d > len(key):
                    break
                prefix = key[:len(key) - d] if d < len(key) else ()
                self._beam_cache.pop(prefix, None)
                if evict_cover and d < cover_depth:
                    self._cover_beam_cache.pop(prefix, None)

    def restore_pruning(self):
        """Restore all params saved by ``tighten_expansion()``.

        Called after a successful retry (or after all retries) so that
        subsequent positions use the original parameters.  Restores
        pruning thresholds, stop_epsilon_mass, and max_steps.
        """
        if hasattr(self, '_saved_prune_params'):
            (self.config.prune_threshold,
             self.config.prune_threshold_alpha,
             self.config.max_prune_mass,
             self.config.stop_epsilon_mass,
             self.config.max_steps) = self._saved_prune_params
            del self._saved_prune_params
        # Evict _beam_cache entries re-populated with relaxed thresholds.
        # Keep _cover_beam_cache — relaxed-threshold cover beams are a
        # superset of original-threshold beams; pruning trims naturally.
        ctx = getattr(self, '_tighten_context', None)
        depth = getattr(self, '_backtrack_depth', 0)
        if ctx is not None:
            for d in range(depth + 1):
                if d > len(ctx):
                    break
                prefix = ctx[:len(ctx) - d] if d < len(ctx) else ()
                self._beam_cache.pop(prefix, None)

        for attr in ('_backtrack_depth', '_cover_backtrack_depth',
                     '_tighten_context'):
            if hasattr(self, attr):
                delattr(self, attr)

    def evict_old_caches(self, current_context_len: int,
                          beam_window: int = 2,
                          cover_window: int = 40,
                          logp_interval: int = 100) -> tuple[int, int, int]:
        """Evict cache entries for contexts shorter than the sliding window.

        During sequential scoring, old cache entries are never re-accessed:
        - ``_beam_cache`` uses exact-match lookup; entries from completed
          positions are dead (except in ce_only mode where the previous
          position's entry is reused by prefix_mass).
        - ``_cover_beam_cache`` uses prefix walk; only entries within
          ``_backtrack_depth`` (max 32) of the current position are needed
          during retry backtracking.
        - ``_logp_cache`` keys are source-space contexts with no clean
          position mapping, so it is cleared entirely every
          ``logp_interval`` positions.  This is safe because incremental
          decomposition via cover beams only scores NEW source contexts;
          old ones are dead weight.  A retry that needs an evicted context
          simply re-scores it via the LM.

        Args:
            current_context_len: Length of the output context AFTER the
                current position was scored (i.e., position index + 1).
            beam_window: Keep ``_beam_cache`` entries with
                ``len(key) >= current_context_len - beam_window``.
                Default 2 covers ce_only reuse (1 back) + margin.
            cover_window: Keep ``_cover_beam_cache`` entries with
                ``len(key) >= current_context_len - cover_window``.
                Default 40 covers max backtrack depth (32) + margin.
            logp_interval: Clear ``_logp_cache`` every N positions.
                Default 100.  Set to 0 to disable.

        Returns:
            (n_beam_evicted, n_cover_evicted, n_logp_evicted) counts.
        """
        beam_min = max(0, current_context_len - beam_window)
        cover_min = max(0, current_context_len - cover_window)

        n_beam = 0
        if beam_min > 0:
            to_del = [k for k in self._beam_cache if len(k) < beam_min]
            for k in to_del:
                del self._beam_cache[k]
            n_beam = len(to_del)

        n_cover = 0
        if cover_min > 0:
            to_del = [k for k in self._cover_beam_cache if len(k) < cover_min]
            for k in to_del:
                del self._cover_beam_cache[k]
            n_cover = len(to_del)

        n_logp = 0
        if logp_interval > 0 and current_context_len % logp_interval == 0:
            n_logp = len(self._logp_cache)
            self._logp_cache.clear()

        return n_beam, n_cover, n_logp

    def clear_cache(self):
        """Clear all caches."""
        self._logp_cache.clear()
        self._beam_cache.clear()
        self._cover_beam_cache.clear()
        if hasattr(self, 'ps_cache'):
            del self.ps_cache
