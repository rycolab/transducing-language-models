"""
fast_next_dist: paper Figure 16 algorithm as a class hierarchy.

Base class FastNextDist implements the paper's algorithm structure.
UnbatchedFastNextDist is the reference implementation without optimizations.

Paper notation → code mapping:
    y (target string)       → context (tuple of output label IDs)
    N = |y|                 → n_ctx = len(context)
    (R, Q)                  → (remainder, quotient) from decompose()
    F (frontier)            → beam_list (list of BeamItems with powerstate + output)
    x (source string)       → key[YS] (source context tuple)
    score(x, p_X_arrow)     → key[LOGP] (cumulative source log-probability)
    p_bar                   → mass (dict or dense array of output symbol → logp)
    queue                   → expand_queue (list of frontier items to expand)
    is_universal(f, S)      → vfst.is_universal(powerstate)
    S ∩ F ≠ ∅               → vfst.has_final(powerstate)
    next_frontier(F, x')    → _advance_ps(vfst, ps, tok)
    prune(candidates)       → threshold pruning on log-probabilities
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Dict

import numpy as np
from scipy.special import logsumexp

from .transduced_lm import POWERSTATE, BEAM_OUT, YS, LOGP
from .fst_advance import _advance_ps, _advance_ps_unlabeled, _is_combined_universal_for_sym, _split_at_boundary

if TYPE_CHECKING:
    from .transduced_lm import TransducedLM


class FastNextDist(ABC):
    """Base class implementing fast_next_dist (paper Figure 16).

    The compute() method is the paper's algorithm with clear step boundaries.
    Subclasses override individual steps to add optimizations (batching,
    vectorization, caching) without obscuring the algorithm structure.
    """

    def __init__(self, tlm: TransducedLM):
        self.tlm = tlm

    async def compute(self, context: tuple | None = None) -> Dict[int, float]:
        """fast_next_dist(y, p_X_arrow) — paper Figure 16.

        Steps map directly to the paper's algorithm:
          1. Decompose context → (R, Q)             [prefix_prob]
          2. Process Q: score resolved, queue rest   [fast_next_dist lines 6-12]
          3. Process R: score with EOS weight        [fast_next_dist lines 13-18]
          3b. EOS contributions                      [implicit in paper]
          4. Expansion loop: resolve queued items    [fast_next_dist lines 19-28]
          5. Normalize → output distribution         [return p_bar]
        """
        ctx = () if context is None else tuple(context)
        state = self._init_state(ctx)

        # ── Step 1: Decompose (paper: prefix_prob → decomposition) ──
        remainder, quotient = await self._decompose(state)

        # ── Step 2: Process Q elements (paper: lines 6-12) ──
        #   For each (F, x) in Q:
        #     Y = {y'[N+1] : (s,y') in F, y'>y}  — next output symbols
        #     If |Y|=1 and is_universal → score directly
        #     Else → add to expansion queue
        await self._process_quotient(state, quotient)

        # ── Step 3: Process R elements (paper: lines 13-18) ──
        #   For each (F, x) in R with S∩F≠∅:
        #     score(x, p_X)/Z  (full string prob via EOS)
        await self._process_remainder(state, remainder)

        # ── Step 3b: EOS contributions ──
        #   (Not in paper's pseudocode but required: beams producing
        #   exactly y with final states contribute to p_bar[EOS])
        await self._process_eos(state, quotient, remainder)

        # ── Step 4: Expansion loop (paper: lines 19-28) ──
        #   while |queue| > 0:
        #     for (F, x) in queue: for x' in X:
        #       F' = next_frontier(F, x')
        #       check universality/finality, score or re-queue
        #     queue = prune(candidates)
        await self._expand(state)

        # ── Step 5: Normalize (paper: return p_bar) ──
        return self._normalize(state)

    @abstractmethod
    def _init_state(self, context: tuple) -> dict:
        """Initialize per-call mutable state."""

    async def _decompose(self, state: dict):
        """Step 1: Decompose context into (R, Q)."""
        remainder, quotient = await self.tlm.decompose(
            state['context'], cache_result=True
        )
        return remainder, quotient

    @abstractmethod
    async def _process_quotient(self, state: dict, quotient) -> None:
        """Step 2: Process Q elements."""

    @abstractmethod
    async def _process_remainder(self, state: dict, remainder) -> None:
        """Step 3: Process R elements."""

    @abstractmethod
    async def _process_eos(self, state: dict, quotient, remainder) -> None:
        """Step 3b: EOS contributions."""

    @abstractmethod
    async def _expand(self, state: dict) -> None:
        """Step 4: Expansion loop."""

    @abstractmethod
    def _normalize(self, state: dict) -> Dict[int, float]:
        """Step 5: Normalize and return distribution."""


class UnbatchedFastNextDist(FastNextDist):
    """Reference implementation: per-item LM calls, no PSCache, no batching.

    This is the simplest correct implementation of fast_next_dist.
    Each step maps directly to the paper without optimization.
    Kept as the correctness reference for test_opt_equivalence.py.
    """

    def _init_state(self, context: tuple) -> dict:
        return {
            'context': context,
            'n_ctx': len(context),
            'score_parts': defaultdict(list),  # output_sym → [logp, ...]
            'eos_parts': [],                   # [logp, ...]
            'expand_queue': [],                # (src_ctx, logp, ps, locked_sym, out_pos)
        }

    async def _process_quotient(self, state: dict, quotient) -> None:
        """Step 2: Classify Q beams as resolved (universal) or queued.

        For each beam key (source context, logp) in Q:
          - Beams with len(output) > n_ctx have produced the next symbol
          - Beams with len(output) == n_ctx are at the context boundary
          - Beams with len(output) < n_ctx are catching up (non-covering)

        Universal scored beams → direct mass accumulation.
        Non-universal → queued for expansion in Step 4.
        """
        n_ctx = state['n_ctx']
        context = state['context']
        vfst = self.tlm.vfst
        input_syms = self.tlm._input_syms
        score_parts = state['score_parts']
        expand_queue = state['expand_queue']
        eos_lm_idx = self.tlm.EOS_LM_IDX

        for key, beam_list in quotient.items():
            beam_logp = key[LOGP]

            # Classify beams by output length
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

            # Catching-up beams: queue with out_pos < n_ctx
            if catching_up_items:
                by_pos = defaultdict(list)
                for b in catching_up_items:
                    by_pos[len(b[BEAM_OUT])].append(b)
                for out_pos, items in by_pos.items():
                    cu_ps = frozenset(int(b[POWERSTATE]) for b in items)
                    expand_queue.append(
                        (key[YS], beam_logp, cu_ps, None, out_pos)
                    )

            unscored_ps = (
                frozenset(int(b[POWERSTATE]) for b in unscored_items)
                if unscored_items else frozenset()
            )

            if scored_items:
                unscored_consumed = False
                _eos_cache = None

                for out_sym, items in scored_items.items():
                    scored_ps = frozenset(int(b[POWERSTATE]) for b in items)

                    if vfst.is_universal(scored_ps):
                        score_parts[out_sym].append(beam_logp)
                        continue

                    if unscored_ps and _is_combined_universal_for_sym(
                            vfst, scored_ps, unscored_ps,
                            out_sym, input_syms):
                        score_parts[out_sym].append(beam_logp)
                        unscored_consumed = True
                        continue

                    if vfst.has_final(scored_ps):
                        if _eos_cache is None:
                            _lm = await self.tlm._lm_dist(key[YS])
                            _eos_cache = float(_lm[eos_lm_idx])
                        score_parts[out_sym].append(beam_logp + _eos_cache)

                    expand_queue.append(
                        (key[YS], beam_logp, scored_ps, int(out_sym), None)
                    )

                if unscored_ps and not unscored_consumed:
                    expand_queue.append(
                        (key[YS], beam_logp, unscored_ps, None, None)
                    )
            else:
                if unscored_items:
                    ps = frozenset(int(b[POWERSTATE]) for b in unscored_items)
                    expand_queue.append(
                        (key[YS], beam_logp, ps, None, None)
                    )

    async def _process_remainder(self, state: dict, remainder) -> None:
        """Step 3: Score R elements with EOS weight.

        For each (F, x) in R with S∩F≠∅:
            p_bar[y_hat] += score(x, p_X)/Z
        """
        if self.tlm.config.ignore_remainder:
            return

        n_ctx = state['n_ctx']
        score_parts = state['score_parts']
        eos_lm_idx = self.tlm.EOS_LM_IDX

        for key, beam_list in remainder.items():
            beam_logp = key[LOGP]
            src_ctx = key[YS]
            lm_dist = await self.tlm._lm_dist(src_ctx)
            logp_eos = float(lm_dist[eos_lm_idx])

            next_syms = set()
            for b in beam_list:
                if len(b[BEAM_OUT]) > n_ctx:
                    next_syms.add(int(b[BEAM_OUT][n_ctx]))

            for ns in next_syms:
                score_parts[ns].append(beam_logp + logp_eos)

    async def _process_eos(self, state: dict, quotient, remainder) -> None:
        """Step 3b: EOS contributions from beams producing exactly context.

        Source strings at final states producing exactly y contribute
        to p_bar[EOS] via p_X(x) = p_X→(x) · p_LM(EOS|x).
        """
        if self.tlm.config.ignore_remainder:
            return

        n_ctx = state['n_ctx']
        eos_parts = state['eos_parts']
        vfst = self.tlm.vfst
        eos_lm_idx = self.tlm.EOS_LM_IDX

        for key, beam_list in quotient.items():
            exact_ps = frozenset(
                int(b[POWERSTATE]) for b in beam_list
                if len(b[BEAM_OUT]) == n_ctx
            )
            if exact_ps and vfst.has_final(exact_ps):
                lm_dist = await self.tlm._lm_dist(key[YS])
                eos_parts.append(key[LOGP] + float(lm_dist[eos_lm_idx]))

        if not self.tlm.config.ignore_remainder:
            for key, beam_list in remainder.items():
                exact_ps = frozenset(
                    int(b[POWERSTATE]) for b in beam_list
                    if len(b[BEAM_OUT]) == n_ctx
                )
                if exact_ps and vfst.has_final(exact_ps):
                    lm_dist = await self.tlm._lm_dist(key[YS])
                    eos_parts.append(key[LOGP] + float(lm_dist[eos_lm_idx]))

    async def _expand(self, state: dict) -> None:
        """Step 4: Expansion loop — resolve queued items.

        while |queue| > 0:
            for (F, x) in queue: for x' in X:
                F' = next_frontier(F, x')
                check universality/finality, score or re-queue
            queue = prune(candidates)
        """
        context = state['context']
        n_ctx = state['n_ctx']
        score_parts = state['score_parts']
        expand_queue = state['expand_queue']
        vfst = self.tlm.vfst
        input_syms = self.tlm._input_syms
        config = self.tlm.config
        eos_lm_idx = self.tlm.EOS_LM_IDX

        max_expand = config.max_steps or 100
        _advance_cache = {}

        for _step in range(max_expand):
            if not expand_queue:
                break

            next_queue = []

            for src_ctx, beam_logp, ps, locked_sym, out_pos in expand_queue:
                lm_dist = await self.tlm._lm_dist(src_ctx)

                for tok in input_syms:
                    logp_tok = float(lm_dist[tok])
                    if not np.isfinite(logp_tok):
                        continue

                    new_logp = beam_logp + logp_tok
                    new_src = src_ctx + (tok,)

                    if out_pos is not None:
                        # Catching up: advance and check output matches context
                        cache_key = (ps, tok, False)
                        if cache_key in _advance_cache:
                            scored_groups, unscored_new = _advance_cache[cache_key]
                        else:
                            scored_groups, unscored_new = _advance_ps(
                                vfst, ps, tok
                            )
                            _advance_cache[cache_key] = (scored_groups, unscored_new)

                        for out_lab, new_ps in scored_groups.items():
                            if out_pos < n_ctx and out_lab == context[out_pos]:
                                new_out_pos = out_pos + 1
                                if new_out_pos >= n_ctx:
                                    # Split: eps_closure may have added
                                    # output past the context boundary.
                                    at_bnd, beyond = _split_at_boundary(
                                        vfst, ps, tok, out_lab)

                                    if beyond:
                                        for bsym, bst in beyond.items():
                                            if vfst.is_universal(bst):
                                                score_parts[bsym].append(
                                                    new_logp)
                                            else:
                                                if (vfst.has_final(bst)
                                                        and not config
                                                        .ignore_remainder):
                                                    new_lm = (
                                                        await self.tlm
                                                        ._lm_dist(new_src))
                                                    score_parts[bsym].append(
                                                        new_logp + float(
                                                            new_lm[eos_lm_idx]
                                                        ))
                                                next_queue.append((
                                                    new_src, new_logp, bst,
                                                    int(bsym), None))
                                        if at_bnd:
                                            # States with only eps-input
                                            # arcs are fully captured by
                                            # the beyond states. Skip both
                                            # frontier AND EOS: the beyond
                                            # contribution uses the full
                                            # prefix probability, which
                                            # already includes EOS.
                                            _EPS = vfst.EPS_LABEL
                                            _has_real = any(
                                                (vfst.arcs(st).in_sym
                                                 != _EPS).any()
                                                for st in at_bnd
                                            )
                                            if _has_real:
                                                if (vfst.has_final(at_bnd)
                                                        and not config
                                                        .ignore_remainder):
                                                    new_lm = (
                                                        await self.tlm
                                                        ._lm_dist(new_src))
                                                    state['eos_parts'].append(
                                                        new_logp + float(
                                                            new_lm[eos_lm_idx]
                                                        ))
                                                next_queue.append((
                                                    new_src, new_logp,
                                                    at_bnd, None, None))
                                    else:
                                        if (vfst.has_final(new_ps)
                                                and not config
                                                .ignore_remainder):
                                            new_lm = (
                                                await self.tlm
                                                ._lm_dist(new_src))
                                            state['eos_parts'].append(
                                                new_logp + float(
                                                    new_lm[eos_lm_idx]))
                                        next_queue.append(
                                            (new_src, new_logp, new_ps,
                                             None, None))
                                else:
                                    next_queue.append(
                                        (new_src, new_logp, new_ps, None, new_out_pos)
                                    )

                        if unscored_new:
                            next_queue.append(
                                (new_src, new_logp, unscored_new, None, out_pos)
                            )

                    elif locked_sym is not None:
                        # Already locked — advance without output tracking
                        cache_key = (ps, tok, True)
                        if cache_key in _advance_cache:
                            new_ps = _advance_cache[cache_key]
                        else:
                            new_ps = _advance_ps_unlabeled(vfst, ps, tok)
                            _advance_cache[cache_key] = new_ps

                        if not new_ps:
                            continue

                        if vfst.is_universal(new_ps):
                            score_parts[locked_sym].append(new_logp)
                            continue

                        if vfst.has_final(new_ps):
                            new_lm = await self.tlm._lm_dist(new_src)
                            score_parts[locked_sym].append(
                                new_logp + float(new_lm[eos_lm_idx])
                            )

                        next_queue.append(
                            (new_src, new_logp, new_ps, locked_sym, None)
                        )

                    else:
                        # Not locked — discover next output symbol
                        cache_key = (ps, tok, False)
                        if cache_key in _advance_cache:
                            scored_groups, unscored_ps = _advance_cache[cache_key]
                        else:
                            scored_groups, unscored_ps = _advance_ps(
                                vfst, ps, tok
                            )
                            _advance_cache[cache_key] = (scored_groups, unscored_ps)

                        unscored_consumed = False
                        for out_lab, new_ps in scored_groups.items():
                            if vfst.is_universal(new_ps):
                                score_parts[out_lab].append(new_logp)
                                continue

                            if unscored_ps and _is_combined_universal_for_sym(
                                    vfst, new_ps, unscored_ps,
                                    out_lab, input_syms):
                                score_parts[out_lab].append(new_logp)
                                unscored_consumed = True
                                continue

                            if vfst.has_final(new_ps):
                                new_lm = await self.tlm._lm_dist(new_src)
                                score_parts[out_lab].append(
                                    new_logp + float(new_lm[eos_lm_idx])
                                )

                            next_queue.append(
                                (new_src, new_logp, new_ps, int(out_lab), None)
                            )

                        if unscored_ps and not unscored_consumed:
                            if (not config.ignore_remainder
                                    and vfst.has_final(unscored_ps)):
                                new_lm = await self.tlm._lm_dist(new_src)
                                state['eos_parts'].append(
                                    new_logp + float(new_lm[eos_lm_idx])
                                )
                            next_queue.append(
                                (new_src, new_logp, unscored_ps, None, None)
                            )

            # Prune
            if next_queue:
                logps = np.array([item[1] for item in next_queue])
                max_logp = logps.max()
                threshold = max_logp + np.log(
                    max(config.prune_threshold, 1e-10)
                )
                keep = logps >= threshold
                expand_queue[:] = [
                    next_queue[i]
                    for i in range(len(next_queue))
                    if keep[i]
                ]
            else:
                expand_queue.clear()

    def _normalize(self, state: dict) -> Dict[int, float]:
        """Step 5: Aggregate scores and normalize to probability distribution."""
        score_parts = state['score_parts']
        eos_parts = state['eos_parts']
        vfst = self.tlm.vfst

        scores = {}
        for sid in vfst.all_output_labels:
            sid_int = int(sid)
            if sid_int in score_parts:
                scores[sid_int] = float(logsumexp(score_parts[sid_int]))
            else:
                scores[sid_int] = -np.inf

        eos_sym_id = int(vfst.eos_out)
        scores[eos_sym_id] = (
            float(logsumexp(eos_parts)) if eos_parts else -np.inf
        )

        if scores:
            vals = np.array(list(scores.values()), dtype=np.float64)
            finite_mask = np.isfinite(vals)
            if finite_mask.any():
                logZ = logsumexp(vals[finite_mask])
            else:
                logZ = 0.0
            scores = {sid: float(lp - logZ) for sid, lp in scores.items()}

        return scores
