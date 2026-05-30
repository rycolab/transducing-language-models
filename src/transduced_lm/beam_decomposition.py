"""
Class-based beam-search decomposition following the paper's algorithm hierarchy.

This is a standalone implementation that maps the paper's AbstractDecomp -> FSTDecomp ->
LazyDecomp class hierarchy onto the rational-tokenizers beam-search approach.

Uses VectorizedFST for all FST operations, replacing the scattered attribute access
on the old TransducedLM.  LM scoring is injected as an async callable.

Paper algorithms referenced:
  - Algorithm 1 (AbstractDecomp): BFS loop + continuous/discontinuous/candidate checks
  - Algorithm 3 (LazyDecomp): On-the-fly powerset determinization via power states
  - Algorithm 5 (prob_mass_prune): Probability-mass-based adaptive pruning
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from itertools import repeat
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .config import Config

# Beam tuple indices (inlined from transducedLM.utils.constants)
POWERSTATE, BEAM_OUT = (0, 1)
YS, LOGP = (0, 1)

# Pre-allocated constant for single-element RLE
_ZERO_ARR = np.array([0], dtype=np.intp)


def _output_matches(beam_out, target_np, n_out, _target_bytes=None):
    """Check beam output at positions 0..n_out-1 matches target.

    target_np should be a numpy array (int16) for fast comparison.
    _target_bytes: optional pre-computed target_np.tobytes() for reuse.
    Uses bytes comparison for numpy arrays (~8x faster than np.array_equal
    for small arrays due to less function call overhead).
    """
    if _target_bytes is not None:
        return beam_out[:n_out].tobytes() == _target_bytes
    return beam_out[:n_out].tobytes() == target_np[:n_out].tobytes()

from .vectorized_fst import ArcArrays, VectorizedFST


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Block:
    """Stores the arc arrays for a single beam's accepted candidates.

    When we process a beam item, we vectorize its outgoing arcs and filter
    to accepted candidates.  Rather than copying per-candidate, we store
    the full arc arrays plus an index (acc_idx) into the accepted positions.
    Multiple candidates from the same beam share this block.
    """
    nstate: np.ndarray    # next-state ids for all arcs
    adv: np.ndarray       # bool: does this arc advance the output? (non-eps output)
    olab: np.ndarray      # output labels for all arcs
    acc_idx: np.ndarray   # indices into the above arrays for accepted candidates
    bpref: np.ndarray     # beam output prefix (shared across all candidates from this beam)
    L: int                # len(bpref) — cached for fast access


# Type aliases
Packed = Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
BeamKey = Tuple[Tuple[int, ...], float]
BeamItem = Tuple[int, np.ndarray]  # (powerstate, beam_out_array)
Beam = Dict[BeamKey, List[BeamItem]]


# ---------------------------------------------------------------------------
# Abstract base — Paper's Algorithm 1
# ---------------------------------------------------------------------------

class AbstractDecomp(ABC):
    """Paper's Algorithm 1: generic decomposition loop.

    Subclasses implement:
      - continuous(key, covering_beams): Is this a cylinder? -> quotient
      - discontinuous(key, covering_beams): Is this in P(y) but not continuous? -> remainder
      - expand_and_score(buckets, out_tokens): Collect candidates, score, prune, materialize
    """

    def __init__(self, config: Config):
        self._config = config

    @property
    def config(self) -> Config:
        return self._config

    @abstractmethod
    def initialize(self, out_tokens) -> dict:
        """Return initial buckets."""
        ...

    @abstractmethod
    def continuous(self, key, covering_beams, all_beams=None) -> bool:
        """Paper's continuity check: is the powerstate universal?"""
        ...

    @abstractmethod
    def discontinuous(self, key, covering_beams) -> bool:
        """Paper's discontinuity check: does the powerstate have a final state?"""
        ...

    @abstractmethod
    async def expand_and_score(self, buckets, out_tokens) -> dict:
        """Collect candidates, score via LM, prune, materialize."""
        ...

    @profile
    async def decompose(self, out_tokens, initial_buckets=None):
        """The main decomposition loop (Algorithm 1 + pruning).

        Args:
            out_tokens: Target output token sequence.
            initial_buckets: If provided, start from these cached beams
                instead of from the start state.  Used by beam caching.

        Returns (remainder, quotient, cover_beams):
            remainder, quotient: beam dicts as before.
            cover_beams: dict of beams whose output exactly matches the target
                (used as starting point for extending to longer targets).
        """
        self._out_tokens = tuple(out_tokens)
        out_tokens_np = np.asarray(out_tokens, dtype=np.int16)
        _target_bytes = out_tokens_np.tobytes()  # pre-compute for fast comparison
        n_out = len(out_tokens)
        max_steps = self.config.max_steps
        expand_threshold = self.config.expand_threshold
        cover_opt = self.config.cover_opt

        quotient = defaultdict(list)
        remainder = defaultdict(list)
        cover_beams = defaultdict(dict)

        if initial_buckets is not None:
            buckets = initial_buckets
        else:
            buckets = self.initialize(out_tokens)

        step = 0
        while buckets:
            step += 1
            if max_steps is not None and step > max_steps:
                break

            # Phase 1: Classify covering beams
            to_expand = {}
            _is_univ = self.vfst.is_universal
            for key, beam_list in buckets.items():
                # Single pass: compute covering + boundary together
                covering = []
                boundary = []
                for b in beam_list:
                    bout = b[BEAM_OUT]
                    blen = len(bout)
                    if cover_opt or (blen >= n_out
                            and _output_matches(bout, out_tokens_np, n_out, _target_bytes)):
                        covering.append(b)
                        if blen == n_out:
                            boundary.append(b)

                if covering:
                    if self.continuous(key, covering, beam_list):
                        quotient[key] = beam_list
                        # Cache boundary beams (len == n_out).
                        # For slow-path Q keys (is_target_universal),
                        # also cache behind beams (len < n_out) that
                        # contributed to the universality check.
                        covering_ps = frozenset(b[POWERSTATE] for b in covering)
                        if _is_univ(covering_ps):
                            if boundary:
                                cover_beams[key] = {None: boundary}
                        else:
                            # Slow path: behind beams needed for
                            # is_target_universal at the next target.
                            needed = [b for b in beam_list
                                      if len(b[BEAM_OUT]) <= n_out]
                            if needed:
                                cover_beams[key] = {None: needed}
                        continue

                    # Non-Q: boundary beams already computed above
                    if boundary:
                        cover_beams[key] = {None: boundary}

                    if self.discontinuous(key, covering):
                        remainder[key] = beam_list

                    # Skip expansion for buckets whose covering beams have
                    # all gone well past the target without converging,
                    # provided we already have some quotient mass.
                    if (all(len(b[BEAM_OUT]) >= n_out + expand_threshold
                           for b in covering)
                            and len(quotient)):
                        continue

                # Drop beams past the target with wrong output — they are
                # dead ends (output is append-only, mismatch can't be fixed).
                # Without this, wrong-output beams expand indefinitely and
                # crowd the pruning budget, starving correct beams.
                live = [b for b in beam_list
                        if len(b[BEAM_OUT]) < n_out
                        or _output_matches(b[BEAM_OUT], out_tokens_np, n_out, _target_bytes)]
                if live:
                    to_expand[key] = live

            # Phase 2: Expand, score, prune, materialize
            if to_expand:
                buckets = await self.expand_and_score(to_expand, out_tokens)
            else:
                buckets = {}

        return remainder, quotient, dict(cover_beams)


# ---------------------------------------------------------------------------
# Beam-based implementation — Paper's LazyDecomp + pruning
# ---------------------------------------------------------------------------

class BeamDecomposition(AbstractDecomp):
    """Paper's LazyDecomp (Algorithm 3) + probability-mass pruning (Algorithm 5).

    All decomposition logic is self-contained in this class.  The algorithm
    has six phases per iteration, each a method:

      Phase 3: collect_candidates — vectorize arcs, filter, pack
      Phase 4: score             — batched LM scoring
      Phase 4b: build_candidate_arrays — flat scored arrays for pruning
      Phase 5: prune             — adaptive probability-mass pruning
      Phase 6: materialize       — unpack survivors + epsilon closure
    """

    def __init__(
        self,
        vfst: VectorizedFST,
        scorer: Callable,
        config: Config,
    ):
        """
        Args:
            vfst: VectorizedFST with precomputed universality.
            scorer: Async callable ``set[tuple] -> dict[tuple, ndarray]``
                that returns LM log-prob distributions for source contexts.
            config: Decomposition configuration.
        """
        super().__init__(config)
        self.vfst = vfst
        self._scorer = scorer

    # -- Initialization -----------------------------------------------------

    def initialize(self, out_tokens) -> dict:
        """Initialize buckets from start state eps-closure."""
        eps_closure = self.vfst.eps_closure(self.vfst.start_state, missing_out=None)
        start = {((), 0.0): []}
        for state, out in eps_closure:
            start[((), 0.0)].append(
                (state, np.array(out, dtype=np.int16))
            )
        return start

    # -- Classification (paper's Algorithm 2, Lines 6-8) --------------------
    @profile
    def continuous(self, key, covering_beams, all_beams=None) -> bool:
        """Is the union powerstate of covering beams universal?

        If yes, every source string with this prefix is in P(y) —
        this bucket is a cylinder -> quotient.

        Uses two-level check:
          1. Fast path: covering beams' powerstates are universal.
          2. Slow path: if non-covering beams exist, perform target-constrained
             mini-DFA BFS (is_target_universal) which mirrors the reference
             Precover DFA's universality check.
        """
        union_ps = frozenset(b[POWERSTATE] for b in covering_beams)
        if self.vfst.is_universal(union_ps):
            return True
        # Slow path: target-constrained universality via mini-DFA BFS
        if (all_beams is not None and len(all_beams) > len(covering_beams)
                and not self._config.skip_target_universal):
            return self.vfst.is_target_universal(all_beams, self._out_tokens)
        return False

    def discontinuous(self, key, covering_beams) -> bool:
        """Does the union powerstate contain a final state?

        If yes, at least one source string reaches acceptance —
        this bucket contributes to the remainder.
        """
        union_ps = frozenset(b[POWERSTATE] for b in covering_beams)
        return self.vfst.has_final(union_ps)

    # -- Phase 3: Candidate collection (paper's candidate check) -----------

    @profile
    def collect_candidates(self, buckets, out_tokens):
        """Vectorize arcs, filter by output-label match, pack into blocks.

        For each beam item, we:
          1. vfst.arcs(state): retrieve all outgoing arcs as ArcArrays
          2. Filter: if output hasn't covered the target yet, only keep arcs
             whose output label matches the expected next target token
          3. Pack: split accepted arcs into non-epsilon (need LM scoring)
             and epsilon (inherit parent's score) groups

        Returns:
            (blocks, cand_pack, cand_eps_idx, pending_ys)
        """
        out_tokens_np = np.asarray(out_tokens, dtype=np.int16)
        n_out = len(out_tokens)
        cover_opt = self.config.cover_opt
        _vfst_arcs = self.vfst.arcs
        _EPS_SYM = VectorizedFST.EPS_SYM

        blocks: List[Block] = []
        cand_pack: Dict[BeamKey, List[Packed]] = {}
        cand_eps_idx: Dict[BeamKey, List[Tuple[int, np.ndarray]]] = {}
        pending_ys: set = set()

        # Cache per-(state, output_position) filtering results.
        # For the same FST state and output position, the accepted arc
        # indices are identical across different beams.
        _filter_cache: dict = {}

        for key, beam_list in buckets.items():
            for b in beam_list:
                beamout_pref = b[BEAM_OUT]
                len_beamout = len(beamout_pref)
                state = b[POWERSTATE]
                covered_target = len_beamout >= n_out or cover_opt

                # Cache key: filtering depends only on state + output position
                _fk = (state, -1) if covered_target else (state, len_beamout)
                cached = _filter_cache.get(_fk)
                if cached is not None:
                    acc_idx, insym_v, ne_pos, has_ne = cached
                    if acc_idx.size == 0:
                        continue
                else:
                    # Step 1: Vectorized arcs for this state
                    arc = _vfst_arcs(state)

                    eps_mask = arc.in_sym != _EPS_SYM
                    acc_m = arc.in_valid.copy()

                    # Step 2: Output-label filtering
                    if covered_target:
                        # Past the target — only take non-epsilon input arcs
                        acc_m &= eps_mask
                    else:
                        # Still building toward the target — filter advancing arcs
                        # to only those matching the expected output token
                        adv_mask = acc_m & arc.advances
                        in_range_m = adv_mask & (len_beamout < n_out)
                        if in_range_m.any():
                            acc_m[in_range_m] = (
                                arc.out_label[in_range_m] == out_tokens_np[len_beamout]
                            )

                    # Step 3: Compute accepted indices
                    acc_idx = np.flatnonzero(acc_m)
                    if acc_idx.size == 0:
                        _filter_cache[_fk] = (acc_idx, None, None, False)
                        continue
                    insym_v = arc.in_sym[acc_idx]
                    ne_mask = eps_mask[acc_idx]
                    ne_pos = np.flatnonzero(ne_mask)
                    has_ne = ne_pos.size > 0
                    _filter_cache[_fk] = (acc_idx, insym_v, ne_pos, has_ne)

                # Register block — stores arc arrays + accepted index
                arc = _vfst_arcs(state)
                blk_id = len(blocks)
                blocks.append(Block(
                    arc.next_state, arc.advances, arc.out_label,
                    acc_idx, beamout_pref, int(len_beamout),
                ))

                # Epsilon candidates: inherit parent score, no LM query needed
                if ne_pos.size < acc_idx.size:
                    eps_pos = np.ones(acc_idx.size, dtype=bool)
                    eps_pos[ne_pos] = False
                    eps_take = np.flatnonzero(eps_pos)
                    if eps_take.size:
                        cand_eps_idx.setdefault(key, []).append((blk_id, eps_take))

                # Non-epsilon candidates: need LM scoring per input symbol
                if has_ne:
                    pending_ys.add(key[YS])
                    ins_ne = insym_v[ne_pos]
                    order = np.argsort(ins_ne, kind="stable")
                    sym_ord = ins_ne[order]
                    pos_ord = ne_pos[order]
                    # Run-length encode by input symbol for efficient lookup
                    n_ne = sym_ord.size
                    if n_ne == 1:
                        starts = _ZERO_ARR
                        ends = np.array([1], dtype=np.intp)
                        sym_runs = sym_ord[:1].astype(np.int32, copy=False)
                    else:
                        changes = np.empty(n_ne, dtype=bool)
                        changes[0] = True
                        np.not_equal(sym_ord[1:], sym_ord[:-1], out=changes[1:])
                        starts = np.flatnonzero(changes)
                        ends = np.empty(starts.size, dtype=np.intp)
                        ends[:-1] = starts[1:]
                        ends[-1] = n_ne
                        sym_runs = sym_ord[starts].astype(np.int32, copy=False)
                    cand_pack.setdefault(key, []).append(
                        (blk_id, sym_runs, starts, ends, pos_ord)
                    )

        return blocks, cand_pack, cand_eps_idx, pending_ys

    # -- Phase 4: LM scoring -----------------------------------------------

    async def score(self, pending_ys):
        """Batch LM scoring for all pending source contexts.

        Returns {ctx_tuple: logdist_array}.
        """
        return await self._scorer(pending_ys)

    # -- Phase 4b: Build flat candidate arrays ------------------------------

    @profile
    def build_candidate_arrays(self, cand_pack, cand_eps_idx, ctx2logdist):
        """Combine packed candidates with LM scores into flat arrays for pruning.

        For each source context (ys, logp):
          - Non-epsilon: look up LM dist for ys, compute logp + dist[sym]
            for each unique input symbol
          - Epsilon: keep parent logp unchanged

        Returns:
            (kind, sym, owner_idx, key_lp, owners) — flat numpy arrays
        """
        owners = []
        kind_chunks, sym_chunks, lp_chunks, owner_chunks = [], [], [], []
        owner_id = 0

        # Non-epsilon candidates: score = base_logp + LM_logp(input_sym | context)
        for (ys, logp), records in cand_pack.items():
            dist = ctx2logdist[ys]
            if len(records) == 1:
                all_syms = records[0][1]
            else:
                all_syms = np.concatenate([r[1] for r in records])
            uniq = np.unique(all_syms).astype(np.int32, copy=False)

            # Score: base_logp + LM dist lookup
            if isinstance(dist, dict):
                dget = dist.get
                _neginf = float("-inf")
                base_lp = float(logp)
                new_lp = np.empty(uniq.size, dtype=np.float32)
                for j in range(uniq.size):
                    new_lp[j] = base_lp + dget(int(uniq[j]), _neginf)
            else:
                dist = np.asarray(dist, dtype=np.float32, order="C")
                new_lp = (np.float32(logp) + dist[uniq])

            owners.append(("ne", ys, float(logp), records))
            n = uniq.size
            kind_chunks.append(np.zeros(n, dtype=np.uint8))
            sym_chunks.append(uniq)
            lp_chunks.append(new_lp)
            owner_chunks.append(np.full(n, owner_id, dtype=np.int32))
            owner_id += 1

        # Epsilon candidates: keep parent logp
        # Pre-allocate shared constant arrays for repeated allocation patterns
        _kind_eps = np.ones(1, dtype=np.uint8)
        _sym_eps = np.array([-1], dtype=np.int32)
        for (ys, logp), eps_chunks in cand_eps_idx.items():
            owners.append(("eps", ys, float(logp), eps_chunks))
            kind_chunks.append(_kind_eps)
            sym_chunks.append(_sym_eps)
            lp_chunks.append(np.array([float(logp)], dtype=np.float32))
            owner_chunks.append(np.array([owner_id], dtype=np.int32))
            owner_id += 1

        if not lp_chunks:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty, empty, empty.astype(np.float32), owners

        kind = np.concatenate(kind_chunks)
        sym = np.concatenate(sym_chunks)
        key_lp = np.concatenate(lp_chunks)
        owner_idx = np.concatenate(owner_chunks)
        return kind, sym, owner_idx, key_lp, owners

    # -- Phase 5: Pruning (paper's Algorithm 5) ----------------------------
    @profile
    def prune(self, key_lp):
        """Adaptive probability-mass pruning.

        Paper's Algorithm 5 (prob_mass_prune):
          1. Compute adaptive threshold based on candidate count
          2. Sort candidates by score, accumulate mass from the top
          3. Keep candidates until the kept mass reaches (1 - threshold) * total

        Returns survivor indices (numpy array).
        """
        key_lp = np.asarray(key_lp, dtype=np.float32, order="C")
        n = key_lp.size
        if n == 0:
            return np.empty(0, dtype=np.int64)

        thld = self.config.prune_threshold
        cand_thld = self.config.candidate_threshold
        alpha = self.config.prune_threshold_alpha
        max_prune_mass = self.config.max_prune_mass
        max_cand = self.config.max_candidates

        # Convert log-probs to stable weights (needed for both paths)
        m = float(np.max(key_lp))

        # Fast paths: all -inf (max is -inf) or no pruning requested
        if m == float("-inf") or thld <= 0:
            if max_cand is not None and n > max_cand:
                return np.argpartition(key_lp, -max_cand)[-max_cand:].astype(
                    np.int64, copy=False
                )
            return np.arange(n, dtype=np.int64)

        # Adaptive threshold: grows with candidate count
        thr = thld
        if n > cand_thld:
            thr = min(thld * (n / cand_thld) ** alpha, max_prune_mass)
        thr = float(min(max(thr, 0.0), 1.0))

        w = np.exp(key_lp - m, dtype=np.float32)
        total = float(w.sum(dtype=np.float64))

        # Keep largest mass >= (1 - thr) * total
        target_keep = total - min(thr * total, np.nextafter(total, 0.0))
        if not (target_keep > 0.0):
            keep = np.array([int(np.argmax(key_lp))], dtype=np.int64)
            if max_cand is not None and keep.size > max_cand:
                kk = np.argpartition(key_lp[keep], -max_cand)[-max_cand:]
                keep = keep[kk]
            return keep

        # Grow a top-K block until it holds enough mass
        cap = n if max_cand is None else min(max_cand, n)
        K = min(32, cap) if cap > 0 else 0
        if K == 0:
            return np.array([np.argmax(key_lp)], dtype=np.int64)

        while True:
            idx = np.argpartition(key_lp, -K)[-K:]
            if w[idx].sum(dtype=np.float64) >= target_keep or K >= cap:
                break
            K = min(K * 2, cap)

        # Find the minimal prefix of sorted candidates that covers target_keep
        order_desc = np.argsort(key_lp[idx])[::-1]
        idx_sorted = idx[order_desc]
        cum = np.cumsum(w[idx_sorted], dtype=np.float32)
        k_star = min(int(np.searchsorted(cum, target_keep, side="left")) + 1, K)
        keep = idx_sorted[:k_star]

        if max_cand is not None and keep.size > cap:
            sel = np.argpartition(key_lp[keep], -cap)[-cap:]
            keep = keep[sel]
        if keep.size == 0:
            keep = np.array([np.argmax(key_lp)], dtype=np.int64)

        return keep.astype(np.int64, copy=False)

    # -- Phase 6: Materialization -------------------------------------------
    @profile
    def materialize(self, survivors, kind, sym, owner_idx, key_lp,
                    owners, blocks, out_tokens):
        """Unpack surviving candidates from blocks, then apply epsilon closure.

        Two steps:
          1. For each survivor, look up its block and extract the raw
             (next_state, beam_prefix, prefix_len, advances, output_label) tuples
          2. For each raw tuple, compute the input-epsilon closure to get
             all reachable states and their accumulated outputs -> powerstates

        Step 2 is LazyDecomp's on-the-fly powerset construction.
        """
        n_out = len(out_tokens)

        # Step 1: Unpack survivors from compressed block representation
        raw_beams: Dict[BeamKey, list] = {}
        for i in survivors:
            _, ys, _, ref = owners[int(owner_idx[i])]

            if kind[i] == 0:  # non-epsilon: extend source context by input symbol
                s = sym[i]
                lst = raw_beams.setdefault((ys + (s,), key_lp[i]), [])
                for blk_id, sym_runs, starts, ends, pos_ord in ref:
                    j = bisect_left(sym_runs, s)
                    if j < len(sym_runs) and sym_runs[j] == s:
                        take_pos = pos_ord[starts[j]:ends[j]]
                        blk = blocks[blk_id]
                        abs_idx = blk.acc_idx[take_pos]
                        lst.extend(zip(
                            blk.nstate[abs_idx],
                            repeat(blk.bpref, abs_idx.size),
                            repeat(blk.L, abs_idx.size),
                            blk.adv[abs_idx],
                            blk.olab[abs_idx],
                        ))

            else:  # epsilon: source context unchanged
                lst = raw_beams.setdefault((ys, key_lp[i]), [])
                for blk_id, eps_pos in ref:
                    blk = blocks[blk_id]
                    abs_idx = blk.acc_idx[eps_pos]
                    lst.extend(zip(
                        blk.nstate[abs_idx],
                        repeat(blk.bpref, abs_idx.size),
                        repeat(blk.L, abs_idx.size),
                        blk.adv[abs_idx],
                        blk.olab[abs_idx],
                    ))

        # Step 2: Epsilon closure -> powerstates
        #
        # Always run eps_closure — skipping it loses output labels from
        # input-epsilon chains (e.g., bytes emitted via epsilon arcs in
        # the realpha flower FST).
        #
        # Covered beams (output >= target length) use missing_out=None
        # since no further target filtering is needed.  Uncovered beams
        # always filter by the remaining target suffix to avoid beam
        # proliferation from non-matching epsilon-output paths.
        # Pre-compute missing_out tuples per output position to avoid
        # recreating them for each candidate at the same position.
        _missing_out_cache: dict = {}
        _eps_closure = self.vfst.eps_closure

        new_buckets = {}
        for beam_key, candidate_outs in raw_beams.items():
            outs = []
            for tgt, beam_out_prefix, len_beam_out, adv, olabel in candidate_outs:
                new_len = len_beam_out + adv
                if new_len >= n_out:
                    missing_out = None
                else:
                    if new_len not in _missing_out_cache:
                        _missing_out_cache[new_len] = tuple(out_tokens[new_len:])
                    missing_out = _missing_out_cache[new_len]
                eps_closure = _eps_closure(int(tgt), missing_out=missing_out)
                # Fast path: most states have trivial eps_closure {(state, ())}
                if len(eps_closure) == 1:
                    (state, closure_out), = eps_closure
                    clen = len(closure_out)
                    total = len_beam_out + adv + clen
                    new = np.empty(total, dtype=np.int16)
                    if len_beam_out:
                        new[:len_beam_out] = beam_out_prefix
                    if adv:
                        new[len_beam_out] = olabel
                        if clen:
                            new[len_beam_out + 1:total] = closure_out
                    elif clen:
                        new[len_beam_out:total] = closure_out
                    outs.append((state, new))
                else:
                    for state, closure_out in eps_closure:
                        clen = len(closure_out)
                        total = len_beam_out + adv + clen
                        new = np.empty(total, dtype=np.int16)
                        if len_beam_out:
                            new[:len_beam_out] = beam_out_prefix
                        if adv:
                            new[len_beam_out] = olabel
                            if clen:
                                new[len_beam_out + 1:total] = closure_out
                        elif clen:
                            new[len_beam_out:total] = closure_out
                        outs.append((state, new))

            new_buckets[beam_key] = outs

        return new_buckets

    # -- Composed: expand_and_score -----------------------------------------
    @profile
    async def expand_and_score(self, buckets, out_tokens) -> dict:
        """Phases 3-6 composed."""
        import time as _time

        # Phase 3: Candidate collection
        _t0 = _time.perf_counter()
        blocks, cand_pack, cand_eps_idx, pending_ys = self.collect_candidates(
            buckets, out_tokens
        )
        _dt_collect = _time.perf_counter() - _t0
        if not cand_pack and not cand_eps_idx:
            return {}

        # Phase 4: LM scoring
        _t0 = _time.perf_counter()
        ctx2logdist = await self.score(pending_ys)
        _dt_score = _time.perf_counter() - _t0

        # Phase 4b: Build flat candidate arrays
        _t0 = _time.perf_counter()
        kind, sym, owner_idx, key_lp, owners = self.build_candidate_arrays(
            cand_pack, cand_eps_idx, ctx2logdist
        )
        _dt_build = _time.perf_counter() - _t0

        # Phase 5: Pruning
        _t0 = _time.perf_counter()
        survivors = self.prune(key_lp)
        _dt_prune = _time.perf_counter() - _t0
        if not survivors.size:
            return {}

        # Phase 6: Materialization + epsilon closure
        _t0 = _time.perf_counter()
        result = self.materialize(
            survivors, kind, sym, owner_idx, key_lp, owners, blocks, out_tokens
        )
        _dt_mat = _time.perf_counter() - _t0

        # Accumulate phase timing
        if not hasattr(self, '_phase_timer'):
            self._phase_timer = {
                'n_steps': 0,
                't_collect': 0.0, 't_score': 0.0, 't_build': 0.0,
                't_prune': 0.0, 't_materialize': 0.0,
                'n_beams_total': 0, 'n_candidates': 0, 'n_survivors': 0,
            }
        pt = self._phase_timer
        pt['n_steps'] += 1
        pt['t_collect'] += _dt_collect
        pt['t_score'] += _dt_score
        pt['t_build'] += _dt_build
        pt['t_prune'] += _dt_prune
        pt['t_materialize'] += _dt_mat
        pt['n_beams_total'] += sum(len(bl) for bl in buckets.values())
        pt['n_candidates'] += len(key_lp)
        pt['n_survivors'] += len(survivors)

        return result


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

async def beam_decompose(vfst, scorer, out_tokens, config):
    """Drop-in wrapper: run BeamDecomposition and return (remainder, quotient)."""
    decomp = BeamDecomposition(vfst, scorer, config)
    return await decomp.decompose(out_tokens)
