"""
Batched fast_next_dist: PSCache + dense mass + batched LM + dedup + early stop.

Production implementation of fast_next_dist (paper Figure 16) with optimizations:
- PSCache: powerstate interning + cached transitions
- Dense mass array: np.logaddexp.at instead of dict[list].append
- Batched LM scoring: GPU batch via scorer() instead of sequential _lm_dist()
- Universality lookahead: skip expansion when next step is universal
- Combined universality: scored ∪ unscored universal check
- Single-output early termination: all mass to one symbol → skip expansion
- Frontier aggregation: collapse large frontiers with mixture LM
- Eps-input mid-token resolution: skip LM for mid-token flower FST beams
- M_total early stopping: stop when frontier mass is negligible

Paper notation → code mapping: see fast_next_dist.py docstring.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, TYPE_CHECKING

import time as _time

import numpy as np
from scipy.special import logsumexp

from .pruning import _prune_by_logweights

from .config import Config
from .fast_next_dist import FastNextDist
from .ps_cache import PSCache
from .transduced_lm import POWERSTATE, BEAM_OUT, YS, LOGP
from .fst_advance import _advance_ps, _advance_ps_unlabeled, _is_combined_universal_for_sym, _split_at_boundary
from .vectorized_fst import VectorizedFST

if TYPE_CHECKING:
    from .transduced_lm import TransducedLM

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Lightweight wall-clock timing for async calls
# ---------------------------------------------------------------------------

class _Timer:
    """Accumulates wall-clock time across batched logp_next calls."""
    __slots__ = (
        "n_calls", "t_decompose", "t_lm_dist", "t_scorer",
        "t_python", "t_total",
        "n_decompose", "n_lm_dist", "n_scorer",
        "n_lookahead", "n_lookahead_hit",
        "last_n_q", "last_n_r",
    )

    def __init__(self):
        self.n_calls = 0
        self.t_decompose = 0.0   # await tlm.decompose()
        self.t_lm_dist = 0.0     # await tlm._lm_dist()
        self.t_scorer = 0.0      # await tlm.scorer()
        self.t_python = 0.0      # everything else (sync Python)
        self.t_total = 0.0       # wall clock per call
        self.n_decompose = 0
        self.n_lm_dist = 0
        self.n_scorer = 0
        self.n_lookahead = 0     # total lookahead checks
        self.n_lookahead_hit = 0 # resolved without expansion
        self.last_n_q = 0        # |Q| from last call
        self.last_n_r = 0        # |R| from last call

    def summary(self) -> str:
        if self.n_calls == 0:
            return "timer: no calls"
        other = self.t_total - self.t_decompose - self.t_lm_dist - self.t_scorer
        la_str = ""
        if self.n_lookahead > 0:
            la_str = (f" lookahead={self.n_lookahead_hit}/{self.n_lookahead}"
                      f"({100*self.n_lookahead_hit/self.n_lookahead:.0f}%)")
        return (
            f"timer ({self.n_calls} calls, {self.t_total:.2f}s total): "
            f"decompose={self.t_decompose:.2f}s({self.n_decompose}x) "
            f"lm_dist={self.t_lm_dist:.2f}s({self.n_lm_dist}x) "
            f"scorer={self.t_scorer:.2f}s({self.n_scorer}x) "
            f"python={self.t_python:.2f}s "
            f"other/unaccounted={other:.2f}s"
            f"{la_str}"
        )


# ---------------------------------------------------------------------------
# Per-call state
# ---------------------------------------------------------------------------

class _S:
    """Per-call mutable state for BatchedFastNextDist."""
    __slots__ = ('context', 'n_ctx', 'mass', 'expand_queue', 'eos_parts',
                 'async_accum', 'wall_t0')

    def __init__(self, context, n_ctx, mass, wall_t0):
        self.context = context
        self.n_ctx = n_ctx
        self.mass = mass
        self.expand_queue = []
        self.eos_parts = []
        self.async_accum = 0.0
        self.wall_t0 = wall_t0


# ---------------------------------------------------------------------------
# Helper functions (module-level)
# ---------------------------------------------------------------------------

def _unlocked_lookahead(psc, ps_id: int, input_syms_arr: np.ndarray) -> bool:
    """Check if ALL single-token transitions produce scored output leading to
    universal states, with no unscored residuals and no multi-output tokens.

    Returns True iff the unlocked item can be fully resolved in one step.
    """
    first_out_tbl, scored_tbl, unscored_tbl, multi_tbl = psc.batch_advance(ps_id)
    fo = first_out_tbl[input_syms_arr]
    sp = scored_tbl[input_syms_arr]
    up = unscored_tbl[input_syms_arr]
    mm = multi_tbl[input_syms_arr]

    if mm.any():
        return False
    has_scored = fo >= 0
    if not has_scored.any():
        return False
    has_unscored = up >= 0
    if has_unscored.any():
        return False

    safe_sp = np.where(sp >= 0, sp, 0)
    all_univ = (sp >= 0) & (psc._univ_np[safe_sp] == 1)
    return bool(all_univ.sum() == has_scored.sum())


def _resolve_unlocked_lookahead(
    psc, ps_id: int, input_syms_arr: np.ndarray,
    lm_dist: np.ndarray, beam_logp: float,
) -> dict[int, float] | None:
    """Compute per-output-symbol mass for a fully resolvable unlocked item.

    Returns {out_sym_id: logp_contribution} or None if resolution fails.
    """
    first_out_tbl, scored_tbl, _, _ = psc.batch_advance(ps_id)
    fo = first_out_tbl[input_syms_arr]
    sp = scored_tbl[input_syms_arr]

    has_scored = fo >= 0
    if not has_scored.any():
        return None

    lps = lm_dist[input_syms_arr].astype(np.float64)
    finite = np.isfinite(lps)
    ok = has_scored & finite
    if not ok.any():
        return None

    out_syms = fo[ok].astype(np.intp)
    tok_lps = lps[ok]

    result: dict[int, float] = {}
    uniq_out = np.unique(out_syms)
    for osym in uniq_out:
        mask = out_syms == osym
        total = float(logsumexp(tok_lps[mask]))
        result[int(osym)] = beam_logp + total
    return result


def _ensure_mass(mass: np.ndarray, idx: int) -> np.ndarray:
    """Grow dense mass array if needed to accommodate idx."""
    if idx >= len(mass):
        new_mass = np.full(idx + 1, -np.inf, dtype=np.float64)
        new_mass[:len(mass)] = mass
        return new_mass
    return mass


# ---------------------------------------------------------------------------
# BatchedFastNextDist
# ---------------------------------------------------------------------------

class BatchedFastNextDist(FastNextDist):
    """Production implementation of fast_next_dist with all optimizations.

    Overrides compute() with the batched algorithm. Each step of the paper's
    algorithm is a separate method:

    compute()                         fast_next_dist(y, p_X_arrow)  Fig 16
      _process_quotient()             for (F,x) in Q    lines 6-12
        _try_quotient_fast_path()       all-universal optimization
        _process_quotient_per_key()     per-key scoring + lookahead
      _process_remainder()            for (F,x) in R    lines 13-18
      _process_eos()                  EOS contributions  (implicit in paper)
      _try_single_output_termination  optimization (not in paper)
      _expand()                       while |queue|>0   lines 19-28
      _normalize()                    return p_bar
    """

    def __init__(self, tlm: TransducedLM):
        super().__init__(tlm)
        self.timer = _Timer()
        self.psc: PSCache | None = None
        self._out_labels_arr: np.ndarray | None = None

    # -- Scaffold abstract methods (implemented but called via compute()) ----

    def _init_state(self, context):
        raise NotImplementedError("Use compute() directly")

    # The following are implemented as proper methods and called from compute().
    # They override the ABC to satisfy the abstract method contract.

    async def _process_quotient(self, s, quotient):
        """Step 2: Process Q elements — fast path or per-key."""
        if self.tlm.vfst.all_universal and not self.tlm.vfst.has_output_epsilon:
            if await self._try_quotient_fast_path(s, quotient):
                return
        await self._process_quotient_per_key(s, quotient)

    async def _process_remainder(self, s, remainder):
        """Step 3: Score R elements with EOS weight."""
        if self.tlm.config.ignore_remainder:
            return

        tlm = self.tlm
        n_ctx = s.n_ctx
        mass = s.mass
        eos_lm_idx = tlm.EOS_LM_IDX
        _timer = self.timer

        for key, beam_list in remainder.items():
            beam_logp = key[LOGP]
            src_ctx = key[YS]
            _t0 = _time.perf_counter()
            lm_dist = await tlm._lm_dist(src_ctx)
            _dt = _time.perf_counter() - _t0
            _timer.t_lm_dist += _dt; _timer.n_lm_dist += 1; s.async_accum += _dt
            logp_eos = float(lm_dist[eos_lm_idx])

            next_syms = set()
            for b in beam_list:
                if len(b[BEAM_OUT]) > n_ctx:
                    ns = int(b[BEAM_OUT][n_ctx])
                    next_syms.add(ns)
                    mass = _ensure_mass(mass, ns)

            for ns in next_syms:
                mass[ns] = np.logaddexp(mass[ns], beam_logp + logp_eos)

        s.mass = mass

    async def _process_eos(self, s, quotient, remainder):
        """Step 3b: EOS contributions from beams producing exactly context."""
        if self.tlm.config.ignore_remainder:
            return

        tlm = self.tlm
        n_ctx = s.n_ctx
        vfst = tlm.vfst
        eos_lm_idx = tlm.EOS_LM_IDX
        _timer = self.timer

        for key, beam_list in quotient.items():
            exact_ps = frozenset(
                int(b[POWERSTATE]) for b in beam_list
                if len(b[BEAM_OUT]) == n_ctx
            )
            if exact_ps and vfst.has_final(exact_ps):
                _t0 = _time.perf_counter()
                lm_dist = await tlm._lm_dist(key[YS])
                _dt = _time.perf_counter() - _t0
                _timer.t_lm_dist += _dt; _timer.n_lm_dist += 1; s.async_accum += _dt
                s.eos_parts.append(key[LOGP] + float(lm_dist[eos_lm_idx]))
        for key, beam_list in remainder.items():
            exact_ps = frozenset(
                int(b[POWERSTATE]) for b in beam_list
                if len(b[BEAM_OUT]) == n_ctx
            )
            if exact_ps and vfst.has_final(exact_ps):
                _t0 = _time.perf_counter()
                lm_dist = await tlm._lm_dist(key[YS])
                _dt = _time.perf_counter() - _t0
                _timer.t_lm_dist += _dt; _timer.n_lm_dist += 1; s.async_accum += _dt
                s.eos_parts.append(key[LOGP] + float(lm_dist[eos_lm_idx]))

    async def _expand(self, s):
        """Step 4: Expansion loop — resolve queued items via batched LM scoring.

        Paper Figure 16, lines 19-28: while |queue| > 0, advance each frontier
        item by one input symbol, check universality/finality, score or re-queue.

        The loop processes three kinds of frontier items per step:
          - Locked:      already committed to an output symbol; advance ignoring
                         output labels (batch_advance_unlabeled). Universal
                         successors → mass, non-universal → re-queue.
          - Unlocked:    haven't chosen an output symbol yet; advance with label
                         tracking (batch_advance). Single-output tokens get the
                         fast path; multi-output tokens fall back to per-arc.
          - Catching-up: behind on output (len(beam_out) < n_ctx); advance
                         per-token matching context symbols until caught up.

        Optimizations over the paper's algorithm:
          - Batched LM:  all frontier contexts scored in one GPU call per step.
          - Dedup:       items with same (ps_id, ctx, locked_sym, out_pos)
                         are merged via logaddexp before pruning.
          - Pruning:     _prune_by_logweights removes low-probability items.
          - Aggregation: when frontier > 500 items, items with the same
                         (ps_id, locked_sym) are compressed into weighted
                         mixture distributions.
          - Early stop:  when remaining frontier mass is negligible relative
                         to M_total (scored mass so far), stop iterating.
          - Deferred EOS: finality-based EOS contributions are batched across
                         the step rather than scored one at a time.
        """
        tlm = self.tlm
        vfst = tlm.vfst
        config = tlm.config
        psc = self.psc
        eos_lm_idx = tlm.EOS_LM_IDX
        input_syms = tlm._input_syms
        _timer = self.timer
        context = s.context
        n_ctx = s.n_ctx
        mass = s.mass

        max_expand = config.max_steps or 100
        log_eps_rel = np.float32(np.log(max(config.stop_epsilon_mass, 1e-30)))

        # M_total tracks total scored mass (resolved + EOS) for early stopping
        _initial_parts = list(mass[np.isfinite(mass)])
        _initial_parts.extend(s.eos_parts)
        M_total = np.float32(logsumexp(_initial_parts)) if _initial_parts else np.float32(NEG_INF)

        prune_kwargs = dict(
            thld=config.prune_threshold,
            cand_thld=config.candidate_threshold,
            alpha=config.prune_threshold_alpha,
            max_prune_mass=config.max_prune_mass,
            max_cand=config.max_candidates,
        )

        # Reorder expand_queue items to frontier format: (ps_id, ctx, logp, locked_sym, out_pos)
        frontier = [
            (item[2], item[0], item[1], item[3], item[4])
            for item in s.expand_queue
        ]

        # Mixture weights for aggregated (compressed) frontier items
        _ctx_weights: dict = {}

        # Preimage-stop EOS: catching-up items that reach the context boundary
        # at a final powerstate need P(EOS|new_ctx) scored in the next step.
        preimage_eos_pending: list[tuple] = []

        for _step in range(max_expand):
            if not frontier and not preimage_eos_pending:
                break

            # Early stop: if remaining frontier mass is negligible, stop
            if _step > 0 and frontier:
                frontier_lps = np.array([it[2] for it in frontier], dtype=np.float32)
                R_bound = float(logsumexp(frontier_lps))
                if R_bound - np.logaddexp(R_bound, M_total) <= log_eps_rel:
                    if not preimage_eos_pending:
                        break

            # ── Score all frontier contexts in one batched LM call ──
            ctx_set = set(it[1] for it in frontier)
            for pe_ctx, _ in preimage_eos_pending:
                ctx_set.add(pe_ctx)
            for cw_dict in _ctx_weights.values():
                ctx_set.update(cw_dict.keys())
            _t0 = _time.perf_counter()
            ctx2dist = await tlm.scorer(ctx_set)
            _dt = _time.perf_counter() - _t0
            _timer.t_scorer += _dt; _timer.n_scorer += 1; s.async_accum += _dt

            # ── Score pending preimage-stop EOS from catching-up items ──
            # When a catching-up item reached the context boundary at a final
            # powerstate in the PREVIOUS step, it deferred its EOS scoring
            # because P(EOS|new_ctx) wasn't available yet.  Score them now.
            step_preimage_eos_lps: list[float] = []
            if preimage_eos_pending and not config.ignore_remainder:
                for pe_ctx, pe_logp in preimage_eos_pending:
                    pe_dist = ctx2dist.get(pe_ctx)
                    if pe_dist is not None:
                        logp_eos = float(pe_dist[eos_lm_idx])
                        scored_lp = pe_logp + logp_eos
                        s.eos_parts.append(scored_lp)
                        step_preimage_eos_lps.append(scored_lp)
                preimage_eos_pending.clear()

            if not frontier:
                # Only had preimage_eos_pending; no frontier items to expand
                if step_preimage_eos_lps:
                    M_total = np.logaddexp(
                        M_total, np.float32(logsumexp(step_preimage_eos_lps)))
                break

            # ── Compute mixture distributions for compressed/aggregated items ──
            # When frontier aggregation compressed multiple contexts into one
            # representative, we need a weighted mixture of their LM distributions
            mix_dists: dict = {}
            for (cw_ps_id, cw_rep_ctx), cw_dict in _ctx_weights.items():
                Z = logsumexp(list(cw_dict.values()))
                mix = None
                for c, clp in cw_dict.items():
                    cd = ctx2dist.get(c)
                    if cd is None:
                        continue
                    weighted = clp + cd
                    if mix is None:
                        mix = weighted.copy()
                    else:
                        mix = np.logaddexp(mix, weighted)
                if mix is not None:
                    mix -= Z
                    mix_dists[(cw_ps_id, cw_rep_ctx)] = mix
            _ctx_weights.clear()

            # ── Group frontier items by (ps_id, ctx) to share advance calls ──
            groups = defaultdict(list)
            for ps_id, ctx, logp, locked_sym, out_pos in frontier:
                groups[(ps_id, ctx)].append((logp, locked_sym, out_pos))

            next_frontier = []
            step_univ_lps = []     # log-probs resolved as universal this step
            eos_deferred = []      # (ctx, out_sym, logp) for batched EOS scoring

            for (ps_id, ctx), items in groups.items():
                # Use mixture distribution if available, else raw LM distribution
                dist = mix_dists.get((ps_id, ctx))
                if dist is None:
                    dist = ctx2dist.get(ctx)
                if dist is None:
                    continue

                # ── Extract finite-probability tokens (vectorized) ──
                _tok_arr = tlm._input_syms_arr
                lps_all = dist[_tok_arr]
                finite_mask = np.isfinite(lps_all)
                if not finite_mask.any():
                    continue

                finite_tok_arr = _tok_arr[finite_mask]
                lp_arr = lps_all[finite_mask].astype(np.float32)

                # Classify items: locked (committed to output sym), unlocked
                # (haven't chosen yet), or catching up (behind on context)
                locked_items = []
                unlocked_items = []
                catching_up = []
                for lp, sym, op in items:
                    if op is not None:
                        catching_up.append((lp, op))
                    elif sym is not None:
                        locked_items.append((lp, sym, op))
                    else:
                        unlocked_items.append((lp, sym, op))

                # ── Catching-up items: advance through context positions ──
                if catching_up:
                    tok_advance_cu = {}
                    for tok_sym in finite_tok_arr:
                        tb = int(tok_sym)
                        tok_advance_cu[tb] = psc.advance(ps_id, tb)

                    for base_logp, out_pos in catching_up:
                        for j in range(len(finite_tok_arr)):
                            tok_sym = int(finite_tok_arr[j])
                            scored_dict, unscored_id = tok_advance_cu[tok_sym]
                            new_logp = base_logp + float(lp_arr[j])
                            new_ctx = ctx + (tok_sym,)

                            for out_lab, new_ps_id in scored_dict.items():
                                if out_pos < n_ctx and out_lab == context[out_pos]:
                                    new_out_pos = out_pos + 1
                                    if new_out_pos >= n_ctx:
                                        # Split: eps_closure may have added
                                        # output past the context boundary.
                                        parent_ps = psc.get_ps(ps_id)
                                        at_bnd, beyond = _split_at_boundary(
                                            vfst, parent_ps, tok_sym, out_lab)

                                        if beyond:
                                            for bsym, bst in beyond.items():
                                                b_id = psc.intern(bst)
                                                if psc.is_universal(b_id):
                                                    mass = _ensure_mass(
                                                        mass, bsym)
                                                    mass[bsym] = np.logaddexp(
                                                        mass[bsym], new_logp)
                                                    step_univ_lps.append(
                                                        new_logp)
                                                else:
                                                    if (psc.is_final(b_id)
                                                            and not config
                                                            .ignore_remainder):
                                                        eos_deferred.append((
                                                            new_ctx, bsym,
                                                            new_logp))
                                                    next_frontier.append((
                                                        b_id, new_ctx,
                                                        new_logp, bsym, None))
                                            if at_bnd:
                                                # States with only eps-input
                                                # arcs are fully captured by
                                                # the beyond states (their
                                                # eps_closure leads there).
                                                # Skip both frontier AND EOS:
                                                # the beyond contribution uses
                                                # the full prefix probability
                                                # p_X→(prefix), which already
                                                # includes EOS. Adding at_bnd
                                                # EOS would double-count.
                                                _EPS = vfst.EPS_LABEL
                                                _has_real = any(
                                                    (vfst.arcs(st).in_sym
                                                     != _EPS).any()
                                                    for st in at_bnd
                                                )
                                                if _has_real:
                                                    at_id = psc.intern(at_bnd)
                                                    if (psc.is_final(at_id)
                                                            and not config
                                                            .ignore_remainder):
                                                        preimage_eos_pending\
                                                            .append((new_ctx,
                                                                     new_logp))
                                                    next_frontier.append((
                                                        at_id, new_ctx,
                                                        new_logp, None, None))
                                        else:
                                            if (psc.is_final(new_ps_id)
                                                    and not config
                                                    .ignore_remainder):
                                                preimage_eos_pending.append(
                                                    (new_ctx, new_logp))
                                            next_frontier.append(
                                                (new_ps_id, new_ctx, new_logp,
                                                 None, None))
                                    else:
                                        next_frontier.append(
                                            (new_ps_id, new_ctx, new_logp,
                                             None, new_out_pos))

                            if unscored_id != -1:
                                next_frontier.append(
                                    (unscored_id, new_ctx, new_logp,
                                     None, out_pos))

                # ── Locked items: advance ignoring output labels ──
                # These items already know their output symbol. We only need
                # to check if the successor powerstate is universal (→ resolve)
                # or needs further expansion (→ re-queue).
                if locked_items:
                    table = psc.batch_advance_unlabeled(ps_id)
                    fin_next = table[finite_tok_arr]
                    valid = fin_next >= 0
                    safe = np.where(valid, fin_next, 0)
                    is_univ = valid & (psc._univ_np[safe] == 1)
                    is_final_arr = valid & ~is_univ & (psc._final_np[safe] == 1)

                    if is_univ.any():
                        univ_total = float(logsumexp(lp_arr[is_univ]))
                        for base_logp, locked_sym, _ in locked_items:
                            mass = _ensure_mass(mass, locked_sym)
                            new_logp = base_logp + univ_total
                            mass[locked_sym] = np.logaddexp(
                                mass[locked_sym], new_logp)
                            step_univ_lps.append(new_logp)

                    if is_final_arr.any() and not config.ignore_remainder:
                        fin_idx = np.flatnonzero(is_final_arr)
                        for base_logp, locked_sym, _ in locked_items:
                            for j in fin_idx:
                                eos_deferred.append(
                                    (ctx + (int(finite_tok_arr[j]),),
                                     locked_sym,
                                     float(base_logp + lp_arr[j])))

                    nonu_idx = np.flatnonzero(valid & ~is_univ)
                    if nonu_idx.size:
                        for base_logp, locked_sym, _ in locked_items:
                            for j in nonu_idx:
                                next_frontier.append(
                                    (int(fin_next[j]),
                                     ctx + (int(finite_tok_arr[j]),),
                                     float(base_logp + lp_arr[j]),
                                     locked_sym, None))

                # ── Unlocked items: advance with output label tracking ──
                # These items haven't committed to an output symbol yet.
                # batch_advance returns per-token tables:
                #   first_out_tbl[tok] = first output label produced (-1 if none)
                #   scored_tbl[tok]    = ps_id for the scored (output-producing) successor
                #   unscored_tbl[tok]  = ps_id for the unscored (no output yet) successor
                #   multi_tbl[tok]     = True if token produces multiple distinct output labels
                # Single-output tokens (>99% for PTB) use the vectorized fast path.
                # Multi-output tokens fall back to per-arc PSCache.advance().
                if unlocked_items:
                    first_out_tbl, scored_tbl, unscored_tbl, multi_tbl = \
                        psc.batch_advance(ps_id)

                    fo = first_out_tbl[finite_tok_arr]
                    s_ps = scored_tbl[finite_tok_arr]
                    u_ps = unscored_tbl[finite_tok_arr]
                    mm = multi_tbl[finite_tok_arr]

                    # Single-output fast path: tokens that produce exactly one
                    # distinct output symbol. Handles >99% of tokens for PTB.
                    # Universal successors → mass, combined-universal → mass,
                    # non-universal → re-queue as locked items.
                    single = ~mm & (fo >= 0)
                    if single.any():
                        s_out = fo[single]
                        s_sp = s_ps[single]
                        s_up = u_ps[single]
                        s_lps = lp_arr[single]
                        s_toks = finite_tok_arr[single]
                        n_s = len(s_out)

                        safe_sp = np.where(s_sp >= 0, s_sp, 0)
                        s_univ = (s_sp >= 0) & (psc._univ_np[safe_sp] == 1)
                        s_final = (s_sp >= 0) & ~s_univ & (psc._final_np[safe_sp] == 1)

                        # Combined universality for non-universal tokens
                        s_comb = np.zeros(n_s, dtype=bool)
                        if not config.skip_combined_univ:
                            for ji in np.flatnonzero(
                                    ~s_univ & (s_sp >= 0) & (s_up >= 0)):
                                if _is_combined_universal_for_sym(
                                        vfst, psc.get_ps(int(s_sp[ji])),
                                        psc.get_ps(int(s_up[ji])),
                                        int(s_out[ji]), input_syms):
                                    s_comb[ji] = True

                        any_u = s_univ | s_comb
                        if any_u.any():
                            univ_out = s_out[any_u].astype(np.intp)
                            univ_lps = s_lps[any_u].astype(np.float64)
                            max_ol = int(univ_out.max())
                            accum = np.full(max_ol + 1, -np.inf, dtype=np.float64)
                            np.logaddexp.at(accum, univ_out, univ_lps)
                            active = np.flatnonzero(np.isfinite(accum))
                            if active.size:
                                mass = _ensure_mass(mass, max_ol)
                                active_totals = accum[active]
                                for base_logp, _, _ in unlocked_items:
                                    contrib = base_logp + active_totals
                                    np.logaddexp.at(mass, active, contrib)
                                    step_univ_lps.extend(contrib.tolist())

                        not_u = ~any_u & (s_sp >= 0)
                        if not_u.any():
                            eos_ok = not_u & s_final
                            for ji in np.flatnonzero(not_u):
                                new_ctx = ctx + (int(s_toks[ji]),)
                                ol = int(s_out[ji])
                                nps = int(s_sp[ji])
                                tok_lp = float(s_lps[ji])
                                is_eos = bool(eos_ok[ji])
                                for base_logp, _, _ in unlocked_items:
                                    new_logp = base_logp + tok_lp
                                    if is_eos and not config.ignore_remainder:
                                        eos_deferred.append(
                                            (new_ctx, ol, new_logp))
                                    next_frontier.append(
                                        (nps, new_ctx, new_logp, ol, None))

                        # Unscored: only consumed by combined universality
                        unsc_elig = (s_up >= 0) & ~s_comb
                        if unsc_elig.any():
                            for ji in np.flatnonzero(unsc_elig):
                                new_ctx = ctx + (int(s_toks[ji]),)
                                uid = int(s_up[ji])
                                tok_lp = float(s_lps[ji])
                                is_eos = (not config.ignore_remainder
                                          and psc.is_final(uid))
                                for base_logp, _, _ in unlocked_items:
                                    new_logp = float(base_logp + tok_lp)
                                    if is_eos:
                                        preimage_eos_pending.append(
                                            (new_ctx, new_logp))
                                    next_frontier.append(
                                        (uid, new_ctx, new_logp,
                                         None, None))

                    # No-scored tokens with unscored output
                    no_scored_unsc = ~mm & (fo < 0) & (u_ps >= 0)
                    if no_scored_unsc.any():
                        for ji in np.flatnonzero(no_scored_unsc):
                            tok_sym = int(finite_tok_arr[ji])
                            uid = int(u_ps[ji])
                            tok_lp = float(lp_arr[ji])
                            new_ctx = ctx + (tok_sym,)
                            is_eos = (not config.ignore_remainder
                                      and psc.is_final(uid))
                            for base_logp, _, _ in unlocked_items:
                                new_logp = float(base_logp + tok_lp)
                                if is_eos:
                                    preimage_eos_pending.append(
                                        (new_ctx, new_logp))
                                next_frontier.append(
                                    (uid, new_ctx, new_logp,
                                     None, None))

                    # Multi-output fallback (rare)
                    if mm.any():
                        for ji in np.flatnonzero(mm):
                            tok_sym = int(finite_tok_arr[ji])
                            scored_dict, unscored_id = psc.advance(
                                ps_id, tok_sym)
                            tok_lp = float(lp_arr[ji])
                            new_ctx = ctx + (tok_sym,)

                            for base_logp, _, _ in unlocked_items:
                                new_logp = base_logp + tok_lp
                                unscored_consumed = False
                                for out_lab, new_ps_id in scored_dict.items():
                                    mass = _ensure_mass(mass, out_lab)
                                    if psc.is_universal(new_ps_id):
                                        mass[out_lab] = np.logaddexp(
                                            mass[out_lab], new_logp)
                                        step_univ_lps.append(new_logp)
                                        continue
                                    if (unscored_id != -1
                                            and not config.skip_combined_univ
                                            and _is_combined_universal_for_sym(
                                                vfst, psc.get_ps(new_ps_id),
                                                psc.get_ps(unscored_id),
                                                out_lab, input_syms)):
                                        mass[out_lab] = np.logaddexp(
                                            mass[out_lab], new_logp)
                                        step_univ_lps.append(new_logp)
                                        unscored_consumed = True
                                        continue
                                    if psc.is_final(new_ps_id):
                                        eos_deferred.append(
                                            (new_ctx, int(out_lab), new_logp))
                                    next_frontier.append(
                                        (new_ps_id, new_ctx, new_logp,
                                         int(out_lab), None))
                                if unscored_id != -1 and not unscored_consumed:
                                    if (not config.ignore_remainder
                                            and psc.is_final(unscored_id)):
                                        preimage_eos_pending.append(
                                            (new_ctx, new_logp))
                                    next_frontier.append(
                                        (unscored_id, new_ctx, new_logp,
                                         None, None))

            # ── Deferred EOS: batch-score all final states from this step ──
            # Instead of scoring EOS one-by-one during advance, we collect
            # all (ctx, out_sym, logp) tuples and score them in one batched call.
            step_eos_lps = []
            if eos_deferred and not config.ignore_remainder:
                eos_ctxs = list(set(d[0] for d in eos_deferred))
                _t0 = _time.perf_counter()
                eos_dists = await tlm.scorer(set(eos_ctxs))
                _dt = _time.perf_counter() - _t0
                _timer.t_scorer += _dt; _timer.n_scorer += 1; s.async_accum += _dt
                for (new_ctx, out_sym, new_logp) in eos_deferred:
                    eos_dist = eos_dists.get(new_ctx)
                    if eos_dist is not None:
                        logp_eos = float(eos_dist[eos_lm_idx])
                        scored_lp = new_logp + logp_eos
                        mass = _ensure_mass(mass, out_sym)
                        mass[out_sym] = np.logaddexp(mass[out_sym], scored_lp)
                        step_eos_lps.append(scored_lp)

            # Update M_total with all mass resolved this step (for early stopping)
            step_scored_lps = step_univ_lps + step_eos_lps + step_preimage_eos_lps
            if step_scored_lps:
                M_total = np.logaddexp(
                    M_total, np.float32(logsumexp(step_scored_lps))
                )

            if not next_frontier:
                break

            # ── Dedup: merge items with same (ps_id, ctx, locked, out_pos) ──
            dedup = {}
            for (ps_id, ctx, logp, locked_sym, out_pos) in next_frontier:
                locked_key = locked_sym if locked_sym is not None else -1
                op_key = out_pos if out_pos is not None else -1
                key = (ps_id, ctx, locked_key, op_key)
                if key in dedup:
                    dedup[key] = float(np.logaddexp(dedup[key], logp))
                else:
                    dedup[key] = logp

            frontier_rebuild = []
            frontier_lps = []
            for (ps_id, ctx, locked_key, op_key), lp in dedup.items():
                locked_sym = locked_key if locked_key >= 0 else None
                out_pos = op_key if op_key >= 0 else None
                frontier_rebuild.append((ps_id, ctx, lp, locked_sym, out_pos))
                frontier_lps.append(lp)

            lps_arr = np.array(frontier_lps, dtype=np.float32)
            keep_idx = _prune_by_logweights(lps_arr, **prune_kwargs)

            frontier = [frontier_rebuild[i] for i in keep_idx]

            # ── Frontier aggregation: compress large frontiers ──
            # When frontier grows too large, group items by (ps_id, locked_sym)
            # and represent each group by its highest-probability context.
            # The other contexts' LM distributions will be mixed in as a
            # weighted combination at the start of the next step.
            _MAX_FRONTIER = 500
            if len(frontier) > _MAX_FRONTIER:
                agg = {}
                for (ps_id, ctx, logp, locked_sym, out_pos) in frontier:
                    agg_key = (ps_id, locked_sym, out_pos)
                    if agg_key not in agg:
                        agg[agg_key] = (logp, ctx, logp, {ctx: logp})
                    else:
                        old_total, old_ctx, old_best, cw = agg[agg_key]
                        new_total = float(np.logaddexp(old_total, logp))
                        if ctx in cw:
                            cw[ctx] = float(np.logaddexp(cw[ctx], logp))
                        else:
                            cw[ctx] = logp
                        if logp > old_best:
                            agg[agg_key] = (new_total, ctx, logp, cw)
                        else:
                            agg[agg_key] = (new_total, old_ctx, old_best, cw)

                frontier = []
                for (ps_id, locked_sym, out_pos), (total_lp, rep_ctx, _, cw) in agg.items():
                    frontier.append((ps_id, rep_ctx, total_lp, locked_sym, out_pos))
                    if len(cw) > 1:
                        _ctx_weights[(ps_id, rep_ctx)] = cw

            if frontier:
                R_tight = float(logsumexp([it[2] for it in frontier]))
                if R_tight - np.logaddexp(R_tight, M_total) <= log_eps_rel:
                    break

        s.mass = mass

    def _normalize(self, s):
        """Step 5+6: Aggregate scores and normalize to probability distribution."""
        vfst = self.tlm.vfst
        mass = s.mass
        eos_parts = s.eos_parts

        eos_sym_id = int(vfst.eos_out)
        eos_val = float(logsumexp(eos_parts)) if eos_parts else NEG_INF

        if self._out_labels_arr is None:
            self._out_labels_arr = np.array(
                sorted(set(vfst.all_output_labels) | {eos_sym_id}), dtype=np.intp
            )
        out_labels = self._out_labels_arr
        max_label = int(out_labels[-1])

        mass = _ensure_mass(mass, max_label)
        mass[eos_sym_id] = eos_val

        raw_vals = mass[out_labels]

        finite_mask = np.isfinite(raw_vals)
        if finite_mask.any():
            logZ = logsumexp(raw_vals[finite_mask])
        else:
            logZ = 0.0
        normed = raw_vals - logZ

        return dict(zip(out_labels.tolist(), normed.tolist()))

    # -- Main entry point ---------------------------------------------------

    @profile
    async def compute(self, context: tuple | None = None) -> Dict[int, float]:
        """fast_next_dist(y, p_X_arrow) — batched implementation.

        Steps map to the paper's algorithm (Figure 16):
          1. Decompose context → (R, Q)             [prefix_prob]
          2. Process Q: score resolved, queue rest   [fast_next_dist lines 6-12]
          3. Process R: score with EOS weight        [fast_next_dist lines 13-18]
          3b. EOS contributions                      [implicit in paper]
          *  Single-output early termination         [optimization]
          4. Expansion loop: resolve queued items    [fast_next_dist lines 19-28]
          5+6. Aggregate + normalize → p_bar         [return p_bar]
        """
        tlm = self.tlm
        context = () if context is None else tuple(context)
        n_ctx = len(context)
        vfst = tlm.vfst
        config = tlm.config
        _timer = self.timer
        _wall_t0 = _time.perf_counter()

        # ── Step 1: Decompose (timed) ──
        _t0 = _time.perf_counter()
        remainder, quotient = await tlm.decompose(context, cache_result=True)
        _dt = _time.perf_counter() - _t0
        _timer.t_decompose += _dt; _timer.n_decompose += 1
        _timer.last_n_q = len(quotient)
        _timer.last_n_r = len(remainder)

        # Initialize per-call state with pre-sized mass array
        max_out_sym = 0
        for key, beam_list in quotient.items():
            for b in beam_list:
                if len(b[BEAM_OUT]) > n_ctx:
                    max_out_sym = max(max_out_sym, int(b[BEAM_OUT][n_ctx]))
        if not config.ignore_remainder:
            for key, beam_list in remainder.items():
                for b in beam_list:
                    if len(b[BEAM_OUT]) > n_ctx:
                        max_out_sym = max(max_out_sym, int(b[BEAM_OUT][n_ctx]))

        s = _S(context, n_ctx,
               np.full(max(max_out_sym + 1, 1), -np.inf, dtype=np.float64),
               _wall_t0)
        s.async_accum = _dt  # decompose time

        # Ensure PSCache is initialized
        if self.psc is None:
            self.psc = PSCache(vfst)

        # ── Step 2: Process Q elements ──
        await self._process_quotient(s, quotient)

        # ── Step 3: Process R elements ──
        await self._process_remainder(s, remainder)

        # ── Step 3b: EOS contributions ──
        await self._process_eos(s, quotient, remainder)

        # ── Optimization: single-output early termination ──
        self._try_single_output_termination(s)

        # ── Step 4: Expansion loop ──
        await self._expand(s)

        # ── Step 5+6: Normalize ──
        scores = self._normalize(s)

        # ── Timing finalization ──
        _wall_total = _time.perf_counter() - _wall_t0
        _timer.t_total += _wall_total
        _timer.t_python += (_wall_total - s.async_accum)
        _timer.n_calls += 1

        return scores

    # -- Quotient processing sub-methods ------------------------------------

    async def _try_quotient_fast_path(self, s, quotient) -> bool:
        """All-universal fast path for FSTs where every state is universal.

        When all FST states are universal AND no output-epsilon arcs, the
        entire Q processing can be done without PSCache/expansion:
          1. Scored items → direct mass accumulation
          2. Eps-input mid-token beams → resolved via _first_eps_output
          3. Token-boundary beams → resolved via first_output table + batched LM

        Returns True if the fast path handled all Q processing.
        """
        tlm = self.tlm
        vfst = tlm.vfst
        n_ctx = s.n_ctx
        mass = s.mass
        _timer = self.timer
        _input_syms_arr = tlm._input_syms_arr

        # Quick pre-check: any catching-up items? (rare for all-universal FSTs)
        for key, beam_list in quotient.items():
            for b in beam_list:
                if len(b[BEAM_OUT]) < n_ctx:
                    return False

        _fp_eps_out = vfst._first_eps_output
        _fp_fo = None        # first_output table (lazily computed once)
        _fp_unscored = []    # keys needing LM scoring

        for key, beam_list in quotient.items():
            beam_logp = key[LOGP]
            needs_lm = False
            # Collect unique output symbols for this key to avoid
            # counting beam_logp multiple times for the same output.
            # Multiple beams under one key represent different FST
            # powerstates reached by the same tokenization prefix;
            # the key probability should be counted once per output.
            _key_scored_syms = set()
            _key_mid_syms = set()
            for b in beam_list:
                blen = len(b[BEAM_OUT])
                if blen > n_ctx:
                    _key_scored_syms.add(int(b[BEAM_OUT][n_ctx]))
                else:
                    eps_out = int(_fp_eps_out[int(b[POWERSTATE])])
                    if eps_out >= 0:
                        _key_mid_syms.add(eps_out)
                    else:
                        needs_lm = True
                        if _fp_fo is None:
                            ps = frozenset(
                                int(bb[POWERSTATE]) for bb in beam_list
                                if (len(bb[BEAM_OUT]) == n_ctx
                                    and int(_fp_eps_out[int(bb[POWERSTATE])]) < 0)
                            )
                            if ps:
                                _fp_fo = vfst.first_output_for_ps(
                                    ps, _input_syms_arr)
            # Union scored and mid syms: a key should contribute beam_logp
            # at most once per output symbol, regardless of whether it came
            # from a scored beam or a mid-token (eps-output) beam.
            for out_sym in (_key_scored_syms | _key_mid_syms):
                mass = _ensure_mass(mass, out_sym)
                mass[out_sym] = np.logaddexp(mass[out_sym], beam_logp)
            if needs_lm:
                _fp_unscored.append((key, beam_logp))

        # For token-boundary beams: use first_output table + batched LM to
        # determine which output symbol each input token produces, then
        # weight by the LM probability of that input token.
        if _fp_unscored and _fp_fo is not None:
            _fp_ok = _fp_fo >= 0
            if not _fp_ok.any():
                # Multi-state powerstate with disagreeing outputs for ALL tokens.
                # Fast path can't resolve — fall back to per-key expansion.
                return False
            _fp_ok_fo = _fp_fo[_fp_ok].astype(np.intp)
            _fp_ok_syms = _input_syms_arr[_fp_ok]
            _fp_max_out = int(_fp_ok_fo.max())
            mass = _ensure_mass(mass, _fp_max_out)

            _t0 = _time.perf_counter()
            _fp_ctxs = [key[YS] for key, _ in _fp_unscored]
            _fp_dists = await tlm._batch_lm_dist(_fp_ctxs)
            _dt = _time.perf_counter() - _t0
            _timer.t_lm_dist += _dt
            _timer.n_lm_dist += len(_fp_unscored)
            s.async_accum += _dt

            for (_, beam_logp), lm_dist in zip(_fp_unscored, _fp_dists):
                lps = lm_dist[_fp_ok_syms].astype(np.float64)
                finite = np.isfinite(lps)
                if not finite.any():
                    continue
                np.logaddexp.at(
                    mass, _fp_ok_fo[finite], beam_logp + lps[finite]
                )

        s.mass = mass
        return True

    async def _process_quotient_per_key(self, s, quotient):
        """Per-key Q processing with PSCache, universality, and lookahead.

        For each beam key (source context, logp) in Q:
          - Beams with len(output) > n_ctx have produced the next symbol
          - Beams with len(output) == n_ctx are at the context boundary
          - Beams with len(output) < n_ctx are catching up

        Universal scored beams → direct mass accumulation.
        Lookahead-resolvable beams → resolved via one-step LM call.
        Non-universal → queued for expansion in Step 4.
        """
        tlm = self.tlm
        vfst = tlm.vfst
        config = tlm.config
        psc = self.psc
        eos_lm_idx = tlm.EOS_LM_IDX
        input_syms = tlm._input_syms
        _input_syms_arr = tlm._input_syms_arr
        _timer = self.timer
        n_ctx = s.n_ctx
        mass = s.mass
        expand_queue = s.expand_queue

        for key, beam_list in quotient.items():
            beam_logp = key[LOGP]

            scored_items = defaultdict(list)
            unscored_items = []
            catching_up_items = []
            for b in beam_list:
                blen = len(b[BEAM_OUT])
                if blen > n_ctx:
                    scored_items[int(b[BEAM_OUT][n_ctx])].append(b)
                elif blen == n_ctx:
                    unscored_items.append(b)
                else:
                    catching_up_items.append(b)

            if catching_up_items:
                by_pos = defaultdict(list)
                for b in catching_up_items:
                    by_pos[len(b[BEAM_OUT])].append(b)
                for out_pos, items in by_pos.items():
                    cu_ps = frozenset(int(b[POWERSTATE]) for b in items)
                    cu_ps_id = psc.intern(cu_ps)
                    expand_queue.append(
                        (key[YS], beam_logp, cu_ps_id, None, out_pos)
                    )

            unscored_ps = (
                frozenset(int(b[POWERSTATE]) for b in unscored_items)
                if unscored_items else frozenset()
            )

            if scored_items:
                unscored_consumed = False  # whether unscored ps was used in combined universality
                _eos_cache = None          # cached EOS logp (shared across output syms for same key)
                _la_lm_cache = None        # cached LM dist for lookahead (same key)
                # Track output symbols fully resolved by scored-universal.
                # These must be excluded from unscored lookahead to avoid
                # double-counting: the scored-universal contribution already
                # covers the full cylinder p_X_arrow(key), which includes
                # all extensions that the unscored beams would add.
                _scored_universal_syms = set()

                for out_sym, items in scored_items.items():
                    scored_ps = frozenset(int(b[POWERSTATE]) for b in items)
                    scored_ps_id = psc.intern(scored_ps)

                    mass = _ensure_mass(mass, out_sym)

                    # Check 1: Is the scored powerstate universal?
                    # If so, this beam accepts all continuations → resolve immediately.
                    if psc.is_universal(scored_ps_id):
                        mass[out_sym] = np.logaddexp(mass[out_sym], beam_logp)
                        _scored_universal_syms.add(out_sym)
                        continue

                    # Check 2: Combined universality — scored ∪ unscored universal?
                    # Even if scored alone isn't universal, adding the unscored
                    # states may make the combined set universal for this output sym.
                    if (unscored_ps
                            and not config.skip_combined_univ
                            and _is_combined_universal_for_sym(
                                vfst, scored_ps, unscored_ps,
                                out_sym, input_syms)):
                        mass[out_sym] = np.logaddexp(mass[out_sym], beam_logp)
                        unscored_consumed = True
                        continue

                    # Check 3: EOS contribution — if ps has final states and we're
                    # not ignoring remainder, add the EOS-weighted mass.
                    if not config.ignore_remainder and psc.is_final(scored_ps_id):
                        if _eos_cache is None:
                            _t0 = _time.perf_counter()
                            _lm_dist = await tlm._lm_dist(key[YS])
                            _dt = _time.perf_counter() - _t0
                            _timer.t_lm_dist += _dt; _timer.n_lm_dist += 1; s.async_accum += _dt
                            _eos_cache = float(_lm_dist[eos_lm_idx])
                        mass[out_sym] = np.logaddexp(
                            mass[out_sym], beam_logp + _eos_cache
                        )

                    # Check 4: One-step universality lookahead for locked items.
                    # If ALL single-token transitions from this ps lead to
                    # universal states, we can resolve without the expansion loop:
                    # just weight the LM distribution by the beam probability.
                    _timer.n_lookahead += 1
                    _la_tbl = psc.batch_advance_unlabeled(scored_ps_id)
                    _la_next = _la_tbl[_input_syms_arr]
                    _la_valid = _la_next >= 0
                    if _la_valid.any():
                        _la_safe = np.where(_la_valid, _la_next, 0)
                        _la_univ = _la_valid & (psc._univ_np[_la_safe] == 1)
                        if _la_univ.sum() == _la_valid.sum():
                            if _la_lm_cache is None:
                                _t0 = _time.perf_counter()
                                _la_lm_cache = await tlm._lm_dist(key[YS])
                                _dt = _time.perf_counter() - _t0
                                _timer.t_lm_dist += _dt; _timer.n_lm_dist += 1
                                s.async_accum += _dt
                            _la_lps = _la_lm_cache[_input_syms_arr]
                            _la_finite = np.isfinite(_la_lps)
                            _la_ok = _la_univ & _la_finite
                            if _la_ok.any():
                                _la_total = float(logsumexp(_la_lps[_la_ok]))
                                mass[out_sym] = np.logaddexp(
                                    mass[out_sym], beam_logp + _la_total)
                            _timer.n_lookahead_hit += 1
                            continue

                    expand_queue.append(
                        (key[YS], beam_logp, scored_ps_id, int(out_sym), None)
                    )

                # Handle remaining unscored items (not consumed by combined universality).
                # Try unlocked lookahead: if all transitions produce scored universal
                # output, resolve the full LM distribution grouped by output symbol.
                if unscored_ps and not unscored_consumed:
                    unscored_ps_id = psc.intern(unscored_ps)
                    if _unlocked_lookahead(psc, unscored_ps_id, _input_syms_arr):
                        _timer.n_lookahead += 1
                        if _la_lm_cache is None:
                            _t0 = _time.perf_counter()
                            _la_lm_cache = await tlm._lm_dist(key[YS])
                            _dt = _time.perf_counter() - _t0
                            _timer.t_lm_dist += _dt; _timer.n_lm_dist += 1
                            s.async_accum += _dt
                        _la_mass = _resolve_unlocked_lookahead(
                            psc, unscored_ps_id, _input_syms_arr,
                            _la_lm_cache, beam_logp)
                        if _la_mass is not None:
                            for _la_sym, _la_val in _la_mass.items():
                                # Skip output symbols already resolved by
                                # scored-universal: beam_logp already covers
                                # the full cylinder including these extensions.
                                if _la_sym in _scored_universal_syms:
                                    continue
                                mass = _ensure_mass(mass, _la_sym)
                                mass[_la_sym] = np.logaddexp(mass[_la_sym], _la_val)
                            _timer.n_lookahead_hit += 1
                        else:
                            expand_queue.append(
                                (key[YS], beam_logp, unscored_ps_id, None, None))
                    else:
                        expand_queue.append(
                            (key[YS], beam_logp, unscored_ps_id, None, None))
            else:
                if unscored_items:
                    ps = frozenset(int(b[POWERSTATE]) for b in unscored_items)
                    ps_id = psc.intern(ps)
                    if _unlocked_lookahead(psc, ps_id, _input_syms_arr):
                        _timer.n_lookahead += 1
                        _t0 = _time.perf_counter()
                        _la_dist = await tlm._lm_dist(key[YS])
                        _dt = _time.perf_counter() - _t0
                        _timer.t_lm_dist += _dt; _timer.n_lm_dist += 1
                        s.async_accum += _dt
                        _la_mass = _resolve_unlocked_lookahead(
                            psc, ps_id, _input_syms_arr, _la_dist, beam_logp)
                        if _la_mass is not None:
                            for _la_sym, _la_val in _la_mass.items():
                                mass = _ensure_mass(mass, _la_sym)
                                mass[_la_sym] = np.logaddexp(mass[_la_sym], _la_val)
                            _timer.n_lookahead_hit += 1
                        else:
                            expand_queue.append(
                                (key[YS], beam_logp, ps_id, None, None))
                    else:
                        expand_queue.append(
                            (key[YS], beam_logp, ps_id, None, None))

        s.mass = mass

    # -- Single-output early termination ------------------------------------

    def _try_single_output_termination(self, s):
        """Optimization: if all mass goes to one output symbol, skip expansion.

        When all scored mass and all queued items target the same symbol,
        the expansion loop would just confirm what we already know.
        """
        mass = s.mass
        expand_queue = s.expand_queue

        scored_syms = set(np.where(np.isfinite(mass))[0])
        queue_locked_syms = set()
        has_unlocked = False
        has_catching_up = False
        for _, _, _, locked_sym, out_pos in expand_queue:
            if out_pos is not None:
                has_catching_up = True
                break
            if locked_sym is None:
                has_unlocked = True
                break
            queue_locked_syms.add(locked_sym)
        all_syms = scored_syms | queue_locked_syms
        if not has_unlocked and not has_catching_up and len(all_syms) == 1 and all_syms:
            only_sym = next(iter(all_syms))
            mass = _ensure_mass(mass, only_sym)
            for src_ctx, beam_logp, ps_id, locked_sym, out_pos in expand_queue:
                mass[only_sym] = np.logaddexp(mass[only_sym], beam_logp)
            s.expand_queue = []
            s.mass = mass
