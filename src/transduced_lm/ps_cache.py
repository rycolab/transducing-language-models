"""
Powerstate interning and cached transition table for VectorizedFST.

Adapted from logp_next_variants/v3_transition_cache.PSCache to work with
VectorizedFST instead of the old TransducedLM.
"""

from collections import defaultdict

import numpy as np

from .vectorized_fst import VectorizedFST
from .fst_advance import _advance_ps, _advance_ps_unlabeled


class PSCache:
    """Powerstate interning and cached transition table.

    Assigns integer IDs to frozenset powerstates, caches universality/finality
    checks and advance results by (ps_id, tok_byte).

    Dense numpy arrays (_univ_np, _final_np) indexed by ps_id enable
    vectorized batch lookups in the expansion loop.
    """

    def __init__(self, vfst: VectorizedFST):
        self.vfst = vfst
        self._ps2id: dict[frozenset, int] = {}
        self._id2ps: list[frozenset] = []
        self._univ: list[bool] = []
        self._final: list[bool] = []
        self._advance_cache: dict[tuple, tuple] = {}
        self._advance_unlab_cache: dict[tuple, int] = {}

        # Dense numpy arrays for vectorized batch lookups
        self._arr_cap = 64
        self._univ_np = np.zeros(self._arr_cap, dtype=np.int8)
        self._final_np = np.zeros(self._arr_cap, dtype=np.int8)

        # Batch transition table size: covers all pynini input labels
        self._sym_table_size = (
            max(vfst._in_sym_table.keys()) + 1
            if vfst._in_sym_table else 1
        )

        # Batch transition tables (cached per ps_id)
        self._batch_unlab: dict[int, np.ndarray] = {}
        self._batch_adv: dict[int, tuple] = {}

    def _sync_np(self, ps_id: int):
        """Ensure numpy arrays cover ps_id and are up to date."""
        if ps_id >= self._arr_cap:
            new_cap = max(ps_id * 2, self._arr_cap * 2)
            u = np.zeros(new_cap, dtype=np.int8)
            f = np.zeros(new_cap, dtype=np.int8)
            u[:self._arr_cap] = self._univ_np
            f[:self._arr_cap] = self._final_np
            self._univ_np = u
            self._final_np = f
            self._arr_cap = new_cap
        self._univ_np[ps_id] = 1 if self._univ[ps_id] else 0
        self._final_np[ps_id] = 1 if self._final[ps_id] else 0

    def intern(self, ps: frozenset) -> int:
        """Intern a powerstate, returning its integer ID."""
        if ps in self._ps2id:
            return self._ps2id[ps]
        ps_id = len(self._id2ps)
        self._ps2id[ps] = ps_id
        self._id2ps.append(ps)
        self._univ.append(self.vfst.is_universal(ps))
        self._final.append(self.vfst.has_final(ps))
        self._sync_np(ps_id)
        return ps_id

    def is_universal(self, ps_id: int) -> bool:
        return self._univ[ps_id]

    def is_final(self, ps_id: int) -> bool:
        return self._final[ps_id]

    def get_ps(self, ps_id: int) -> frozenset:
        return self._id2ps[ps_id]

    # ── Per-token advance (existing) ──────────────────────────────────────

    def advance(self, ps_id: int, tok_byte: int):
        """Cached _advance_ps: returns ({out_label: ps_id}, unscored_ps_id).

        unscored_ps_id is -1 if unscored set is empty.
        """
        key = (ps_id, tok_byte)
        if key in self._advance_cache:
            return self._advance_cache[key]

        ps = self._id2ps[ps_id]
        scored_raw, unscored_raw = _advance_ps(self.vfst, ps, tok_byte)

        scored = {
            out_label: self.intern(next_ps)
            for out_label, next_ps in scored_raw.items()
        }
        unscored_id = self.intern(unscored_raw) if unscored_raw else -1

        self._advance_cache[key] = (scored, unscored_id)
        return scored, unscored_id


    # ── Batch advance (vectorized) ────────────────────────────────────────

    def batch_advance_unlabeled(self, ps_id: int) -> np.ndarray:
        """Batch unlabeled advance for ALL input bytes at once.

        Returns int32 array of shape [258] mapping sym_value -> next_ps_id
        (-1 if no transition).  Computed once per ps_id by iterating over
        the (few) states in the powerstate using VectorizedFST's cached
        ArcArrays, then eps_closure + intern.

        Usage in expansion loop::

            table = psc.batch_advance_unlabeled(ps_id)
            fin_next = table[finite_tok_arr]       # one numpy fancy-index
            valid = fin_next >= 0
            is_univ = valid & (psc._univ_np[np.where(valid, fin_next, 0)] == 1)
        """
        cached = self._batch_unlab.get(ps_id)
        if cached is not None:
            return cached

        ps = self._id2ps[ps_id]
        vfst = self.vfst

        # Collect next states per input byte from ArcArrays
        by_sym: dict[int, set] = defaultdict(set)
        for state in ps:
            a = vfst.arcs(state)
            non_eps = a.in_sym != VectorizedFST.EPS_SYM
            for i in np.where(non_eps)[0]:
                by_sym[int(a.in_sym[i])].add(int(a.next_state[i]))

        # eps_closure + intern for each byte
        table = np.full(self._sym_table_size, -1, dtype=np.int32)
        for sym_val, next_states in by_sym.items():
            expanded = set()
            for s in next_states:
                for cs, _ in vfst.eps_closure(s):
                    expanded.add(cs)
            if expanded:
                table[sym_val] = self.intern(frozenset(expanded))

        self._batch_unlab[ps_id] = table
        return table

    def batch_advance(self, ps_id: int):
        """Batch labeled advance for ALL input bytes at once.

        Returns (first_out, scored_ps, unscored_ps, multi_mask):
            first_out:    int32[258] — first/only output label per byte (-1 if none)
            scored_ps:    int32[258] — ps_id of scored successor (-1 if none)
            unscored_ps:  int32[258] — ps_id of unscored successor (-1 if none)
            multi_mask:   bool[258]  — True if >1 scored output (needs fallback)

        For PTB, >99% of transitions have exactly one output label, so
        first_out + scored_ps covers nearly all cases.

        Usage in expansion loop::

            first_out, scored_ps, unscored_ps, multi_mask = psc.batch_advance(ps_id)
            # Single-output fast path:
            single = ~multi_mask & (first_out[tok_arr] >= 0)
            # Multi-output fallback: use per-token psc.advance()
        """
        cached = self._batch_adv.get(ps_id)
        if cached is not None:
            return cached

        ps = self._id2ps[ps_id]
        vfst = self.vfst

        # Collect per-byte: {out_label: set_of_next_states} and eps_reached
        per_byte_scored: dict[int, dict[int, set]] = defaultdict(lambda: defaultdict(set))
        per_byte_eps: dict[int, set] = defaultdict(set)

        for state in ps:
            a = vfst.arcs(state)
            non_eps = a.in_sym != VectorizedFST.EPS_SYM
            for i in np.where(non_eps)[0]:
                sym_val = int(a.in_sym[i])
                if a.advances[i]:
                    out_label = int(a.out_label[i])
                    next_state = int(a.next_state[i])
                    # Apply eps_closure — ALL closure states join
                    # scored[out_label]. The arc already committed out_label
                    # as the first output; eps_closure output is at deeper
                    # positions and doesn't change the grouping key.
                    for cls_state, out_tuple in vfst.eps_closure(next_state):
                        per_byte_scored[sym_val][out_label].add(int(cls_state))
                else:
                    per_byte_eps[sym_val].add(int(a.next_state[i]))

        bts = self._sym_table_size
        first_out = np.full(bts, -1, dtype=np.int32)
        scored_ps = np.full(bts, -1, dtype=np.int32)
        unscored_ps = np.full(bts, -1, dtype=np.int32)
        multi_mask = np.zeros(bts, dtype=bool)

        all_bytes = set(per_byte_scored.keys()) | set(per_byte_eps.keys())

        for sym_val in all_bytes:
            # Process scored outputs (non-eps output arcs + eps_closure deferred)
            scored_dict = per_byte_scored.get(sym_val, {})

            # Process eps-reached states via eps_closure
            eps_states = per_byte_eps.get(sym_val, set())
            unscored_set = set()
            for s in eps_states:
                for cs, out_tuple in vfst.eps_closure(s):
                    if out_tuple:
                        # Deferred output from eps_closure
                        scored_dict.setdefault(int(out_tuple[0]), set()).add(cs)
                    else:
                        unscored_set.add(cs)

            # Unscored successor
            if unscored_set:
                unscored_ps[sym_val] = self.intern(frozenset(unscored_set))

            # Scored successors
            if scored_dict:
                labels = list(scored_dict.keys())
                if len(labels) == 1:
                    out_lab = labels[0]
                    next_states = scored_dict[out_lab]
                    first_out[sym_val] = out_lab
                    scored_ps[sym_val] = self.intern(frozenset(next_states))
                else:
                    multi_mask[sym_val] = True
                    # Store first label for reference, but caller must use
                    # per-token psc.advance() for multi-output bytes
                    first_out[sym_val] = labels[0]

                    # Populate per-token advance cache as a side effect so
                    # the fallback path hits cache instead of recomputing
                    all_scored = {
                        ol: self.intern(frozenset(ns))
                        for ol, ns in scored_dict.items()
                    }
                    unsc_id = unscored_ps[sym_val]
                    key = (ps_id, sym_val)
                    if key not in self._advance_cache:
                        self._advance_cache[key] = (all_scored, int(unsc_id))

        result = (first_out, scored_ps, unscored_ps, multi_mask)
        self._batch_adv[ps_id] = result
        return result
