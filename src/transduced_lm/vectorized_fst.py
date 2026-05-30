"""
VectorizedFST: pynini FST with precomputed vectorized access and universality checking.

Owns all FST-derived state and presents it through a typed, documented interface.
All internal indexing uses pynini arc labels directly:

  - Input symbols: pynini input label IDs in ArcArrays.in_sym (eps = 0)
  - Output labels: pynini output label IDs in ArcArrays.out_label (eps = 0)
  - State IDs: pynini state IDs in ArcArrays.next_state
  - Symbol table: _in_sym_table maps pynini labels to symbol strings, used by
    LM adapters to bridge between pynini labels and external ID spaces (byte
    values, HF token IDs, etc.)

Replaces ~30 scattered attributes on the old TransducedLM with a single class.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import FrozenSet, List

import numpy as np
import pynini

from collections import deque


# ---------------------------------------------------------------------------
# Universality helpers (inlined from transducedLM.fst.universality +
# transducedLM.fst.properties — makes this module fully self-contained)
# ---------------------------------------------------------------------------

def _input_epsilon_closure(fst_obj, start_state: int) -> set:
    """Plain input-epsilon closure (state set only, no output tracking).

    Used by universality computation.  Follows arcs with ilabel==0.
    """
    closure = set()
    queue = deque([start_state])
    while queue:
        state = queue.popleft()
        if state in closure:
            continue
        closure.add(state)
        for arc in fst_obj.fst.arcs(state):
            if arc.ilabel == 0:
                queue.append(arc.nextstate)
    return closure


def _get_sigma_star(fst_obj):
    """Build sigma* FST: single-state loop over all non-epsilon input symbols."""
    sig = pynini.Fst()
    s = sig.add_state()
    sig.set_start(s)
    sig.set_final(s)
    for ilabel, _ in fst_obj.fst.input_symbols():
        if ilabel == fst_obj.eps_id:
            continue
        sig.add_arc(
            s,
            pynini.Arc(
                int(ilabel),
                int(ilabel),
                pynini.Weight.one(fst_obj.fst.weight_type()),
                s,
            ),
        )
    return sig


def _closure_input_reads_all_symbols(fst_obj, state: int, closure: set) -> bool:
    """Check if the input closure reads all input symbols (quick precheck)."""
    symbols = fst_obj.fst.input_symbols()
    read_symbols = set()
    for s in closure:
        for arc in fst_obj.fst.arcs(s):
            read_symbols.add(arc.ilabel)
    all_symbols = set(sym_id for sym_id, _ in symbols)
    if fst_obj.is_final(state):
        all_symbols.discard(fst_obj.eps_id)
        read_symbols.discard(fst_obj.eps_id)
    return read_symbols == all_symbols

@profile
def _is_universal_single(fst_obj, start_state: int, closure=None) -> bool:
    """Check if a single state is universal via pynini DFA operations.

    Steps: copy FST, set start, project input, rmepsilon, determinize,
    minimize, difference with sigma*.  Universal iff difference is empty.
    """
    if closure is None:
        closure = _input_epsilon_closure(fst_obj, start_state)
    if not _closure_input_reads_all_symbols(fst_obj, start_state, closure):
        return False
    sub = fst_obj.fst.copy()
    sub.set_start(start_state)
    sub.project("input")
    sub = pynini.rmepsilon(sub)
    sub = pynini.determinize(sub)
    sub = pynini.minimize(sub, allow_nondet=False)
    diff = pynini.difference(fst_obj.sigma_star, sub)
    diff = pynini.connect(diff)
    return diff.num_states() == 0

@profile
def _is_universal_set(fst_obj, start_states) -> bool:
    """Universality for a set of states.

    Creates a new start state with epsilon arcs to each state in the set,
    then checks universality of the resulting sub-FST.
    """
    if isinstance(start_states, int):
        start_states = frozenset({start_states})
    sub = fst_obj.fst.copy()
    new_start = sub.add_state()
    sub.set_start(new_start)
    one = pynini.Weight.one(sub.weight_type())
    for s in start_states:
        sub.add_arc(
            new_start, pynini.Arc(fst_obj.eps_id, fst_obj.eps_id, one, s)
        )
    sub.project("input")
    sub = pynini.rmepsilon(sub)
    sub = pynini.determinize(sub)
    sub = pynini.minimize(sub, allow_nondet=False)
    diff = pynini.difference(fst_obj.sigma_star, sub)
    diff = pynini.connect(diff)
    return diff.num_states() == 0


def _sort_states_by_incoming_arcs(fst_obj) -> list:
    """Sort states by number of incoming arcs (descending)."""
    incoming = defaultdict(int)
    for state in fst_obj.fst.states():
        for arc in fst_obj.fst.arcs(state):
            incoming[arc.nextstate] += 1
    all_states = list(fst_obj.fst.states())
    return sorted(all_states, key=lambda s: incoming[s], reverse=True)

@profile
def _compute_universal_states(fst_obj):
    """Compute single-state universality for all states.

    Sorts by incoming arc count (states with more incoming arcs are more
    likely to be in other states' closures).  Uses closure-based shortcut:
    if any state in the input-epsilon closure is already known universal,
    the current state is also universal.
    """
    sorted_states = _sort_states_by_incoming_arcs(fst_obj)
    for state in sorted_states:
        if state in fst_obj.universal_states:
            continue
        input_closure = _input_epsilon_closure(fst_obj, state)
        for closure_state in input_closure:
            if fst_obj.universal_states.get(closure_state):
                fst_obj.universal_states[state] = True
                break
        else:
            fst_obj.universal_states[state] = _is_universal_single(
                fst_obj, state, closure=input_closure
            )


# ---------------------------------------------------------------------------
# ArcArrays — typed return value for arcs()
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class ArcArrays:
    """Vectorized representation of all outgoing arcs from a single state.

    Each field is a 1-D numpy array of length n_arcs.  The ID namespace
    is documented per field so consumers never need to guess.
    """
    in_sym: np.ndarray
    """Pynini input label IDs.  Epsilon arcs have value 0."""

    out_label: np.ndarray
    """Pynini output label IDs.  Epsilon = EPS_LABEL (0).  dtype=int16."""

    advances: np.ndarray
    """Bool array: True where out_label != EPS_LABEL (arc produces output)."""

    next_state: np.ndarray
    """Pynini state IDs of arc destinations.  dtype=int32."""

    in_valid: np.ndarray
    """Bool array: True where the input symbol is not a special symbol."""


# ---------------------------------------------------------------------------
# VectorizedFST
# ---------------------------------------------------------------------------

class VectorizedFST:
    """Pynini FST with precomputed vectorized access and universality checking.

    Usage::

        vfst = VectorizedFST(pynini_fst)
        arcs = vfst.arcs(state)          # -> ArcArrays
        cl = vfst.eps_closure(state)     # -> set[(state, out_tuple)]
        u = vfst.is_universal(ps)        # -> bool
        f = vfst.has_final(ps)           # -> bool
    """

    # ── Constants ──────────────────────────────────────────────────────────
    EPS_SYM = 0
    """Pynini input epsilon label (always 0 by OpenFst convention)."""

    EPS_LABEL = 0
    """Epsilon in pynini's label ID space.  Always 0 by OpenFst convention."""

    # ── Construction ───────────────────────────────────────────────────────

    def __init__(self, fst: pynini.Fst, eos_out: str = "257"):
        # --- Core FST ---
        self.fst = fst

        # --- Epsilon IDs ---
        # Pynini ID 0 is always epsilon.  The symbol table maps it to a
        # string label (e.g. "257").  int(label) gives the byte value.
        self.eps_id: int = 0
        self._eps_label_str: str = fst.input_symbols().find(0)  # e.g. "257"

        # --- EOS ---
        self.eos_out = eos_out

        # --- Symbol maps ---
        # _in_sym_table: pynini input label ID -> symbol string (from FST symbol table)
        # Used by LM adapters to bridge pynini labels ↔ external IDs (bytes, HF tokens)
        self._in_sym_table: dict[int, str] = {}
        for pynini_id, label_str in fst.input_symbols():
            self._in_sym_table[int(pynini_id)] = label_str

        # _out_id_to_sym: pynini output label ID -> string label
        self._out_id_to_sym: dict[int, str] = {}
        for pynini_id, label_str in fst.output_symbols():
            self._out_id_to_sym[int(pynini_id)] = label_str

        # Reverse maps (used by external code)
        self._out_sym_to_id: dict[str, int] = {v: k for k, v in self._out_id_to_sym.items()}

        # Derived
        self.num_in_syms: int = len(self._in_sym_table)
        self.num_out_syms: int = len(self._out_id_to_sym)
        self.special_symbols: list = []

        # All valid output label IDs (non-epsilon), used for logp_next enumeration
        self._all_output_labels = np.array(
            [pid for pid in self._out_id_to_sym if pid != self.EPS_LABEL],
            dtype=np.int16,
        )

        # --- Precomputed graphs ---
        self._eps_graph = self._build_eps_graph()
        self._arcs_cache: dict[int, ArcArrays] = {}

        # --- Universality ---
        self.universal_states: dict[int, bool] = {}
        self._universal_set_cache: dict[FrozenSet[int], bool] = {}
        self._universal_sets_by_size: defaultdict[int, set] = defaultdict(set)
        self.sigma_star = _get_sigma_star(self)

    # ── Arc access (collapses 3-layer _arcs → _vectorized_arcs → get_vectorized) ──

    def arcs(self, state: int) -> ArcArrays:
        """Return vectorized arc arrays for all outgoing arcs from *state*.

        Cached per state.  Collapses the original 3-layer pipeline
        (_arcs → _vectorized_arcs → get_vectorized) into a single call.
        """
        cached = self._arcs_cache.get(state)
        if cached is not None:
            return cached

        arc_list = list(self.fst.arcs(state))
        n = len(arc_list)

        if n == 0:
            result = ArcArrays(
                in_sym=np.empty(0, dtype=np.int32),
                out_label=np.empty(0, dtype=np.int16),
                advances=np.empty(0, dtype=bool),
                next_state=np.empty(0, dtype=np.int32),
                in_valid=np.empty(0, dtype=bool),
            )
            self._arcs_cache[state] = result
            return result

        # Raw pynini arrays — single pass over arc_list instead of 3 generators
        ilabels = np.empty(n, dtype=np.int32)
        olabels = np.empty(n, dtype=np.int16)
        nextstates = np.empty(n, dtype=np.int32)
        for i, a in enumerate(arc_list):
            ilabels[i] = a.ilabel
            olabels[i] = a.olabel
            nextstates[i] = a.nextstate

        # Derived boolean arrays
        advances = olabels != self.EPS_LABEL
        in_valid = ~np.isin(ilabels, self.special_symbols, assume_unique=True)

        result = ArcArrays(
            in_sym=ilabels,
            out_label=olabels,
            advances=advances,
            next_state=nextstates,
            in_valid=in_valid,
        )
        self._arcs_cache[state] = result
        return result

    # ── Epsilon closure ────────────────────────────────────────────────────

    @lru_cache(maxsize=65536)
    def eps_closure(
        self,
        start_state: int,
        missing_out: tuple[int, ...] | None = None,
    ) -> frozenset[tuple[int, tuple[int, ...]]]:
        """Input-epsilon closure with output label tracking.

        Starting from *start_state*, follows all arcs with input label = epsilon.
        Tracks the accumulated output labels along the path.

        Args:
            start_state: Starting FST state.
            missing_out: If given, prune paths whose output diverges from
                this expected suffix.  Used during decomposition to avoid
                exploring epsilon paths that cannot match the target.

        Returns:
            Frozenset of (state, output_label_tuple) pairs.
        """
        closure: set[tuple[int, tuple[int, ...]]] = set()
        stack = [(start_state, ())]
        eps_graph = self._eps_graph

        while stack:
            state, out = stack.pop()
            key = (state, out)
            if key in closure:
                continue
            closure.add(key)

            for nxt, olabel in eps_graph[state]:
                if olabel == self.EPS_LABEL:
                    # Both-epsilon arc: advance state, don't extend output
                    stack.append((nxt, out))
                    continue
                if missing_out is None:
                    # No constraint: accumulate output label
                    stack.append((nxt, out + (olabel,)))
                    continue
                # Constrained: skip if output would diverge from expected
                if len(out) < len(missing_out) and olabel != missing_out[-1]:
                    continue
                stack.append((nxt, out + (olabel,)))

        return frozenset(closure)

    # ── Universality ───────────────────────────────────────────────────────

    def compute_universal_states(self):
        """Compute single-state universality for all states.

        Delegates to the existing (heavy) pynini-based implementation.
        """
        _compute_universal_states(self)

    @profile
    def is_universal(self, powerstate: FrozenSet[int]) -> bool:
        """Check if a powerstate (set of states) is universal.

        A powerstate is universal if the union of the right-input-languages
        of its states equals Sigma*.  Uses multi-level caching:
          1. Direct cache lookup
          2. Single-state shortcut
          3. Any single universal state in set
          4. Known universal subset
          5. Full pynini computation (cached on return)
        """
        S = powerstate

        # Level 1: direct cache
        if S in self._universal_set_cache:
            return self._universal_set_cache[S]

        # Level 2: single state
        if len(S) == 1:
            (state,) = S
            return self.universal_states[state]

        # Level 3: any single state is universal → set is universal
        if any(self.universal_states[state] for state in S):
            return True

        # Level 4: known universal subset
        for k in range(1, len(S) + 1):
            for U in self._universal_sets_by_size.get(k, ()):
                if U.issubset(S):
                    self._universal_set_cache[S] = True
                    return True

        # Level 5: full pynini computation
        val = _is_universal_set(self, S)
        self._universal_set_cache[S] = val
        if val:
            self._universal_sets_by_size[len(S)].add(S)
        return val

    # ── Finality ───────────────────────────────────────────────────────────

    def is_final(self, state: int) -> bool:
        """Check if a state is final (accepting) in the FST."""
        w = self.fst.final(state)
        return w != pynini.Weight.zero(self.fst.weight_type())

    def has_final(self, powerstate: FrozenSet[int]) -> bool:
        """Check if any state in the powerstate is final."""
        return any(self.is_final(s) for s in powerstate)

    # ── Target-constrained universality ──────────────────────────────────

    @cached_property
    def input_syms(self) -> List[int]:
        """Sorted non-epsilon pynini input label IDs."""
        return sorted(k for k in self._in_sym_table if k != self.eps_id)

    @profile
    def is_target_universal(self, beam_items, target_tokens) -> bool:
        """Check if beam items form a universal DFA state for the target.

        Performs a mini-DFA BFS mirroring the reference Precover's universality
        check.  Uses beam items (covering + non-covering) as the initial DFA
        state, tracking (fst_state, output_position) pairs.

        An output position tracks how many target tokens the beam has produced.
        Positions >= len(target_tokens) mean the beam has matched the full target.

        Args:
            beam_items: List of (powerstate, beam_out_array) tuples.
            target_tokens: Target output token sequence (tuple of ints).

        Returns:
            True if the DFA state is universal (all input extensions lead to
            states that can produce the target).
        """
        n_target = len(target_tokens)

        # Build initial DFA state: {(fst_state, output_position)}
        init_state = set()
        for b_ps, b_out in beam_items:
            pos = min(len(b_out), n_target)
            init_state.add((int(b_ps), pos))
        init_state = frozenset(init_state)

        # BFS
        visited = set()
        queue = deque([init_state])
        in_syms = self.input_syms

        while queue:
            dfa_state = queue.popleft()
            if dfa_state in visited:
                continue
            visited.add(dfa_state)

            # Check finality: exists (s, pos) with pos >= n_target and s is final
            is_final = any(
                pos >= n_target and self.is_final(s)
                for s, pos in dfa_state
            )
            if not is_final:
                return False

            # Check completeness: every input symbol has a non-empty successor
            for tok in in_syms:
                successor = set()
                for s, pos in dfa_state:
                    # Follow non-eps arcs matching this input symbol
                    for arc in self.fst.arcs(s):
                        if arc.ilabel == 0:
                            continue  # skip eps-input arcs (handled by eps_closure)
                        if arc.ilabel != tok:
                            continue
                        # Arc matches this input symbol
                        nxt = arc.nextstate
                        if arc.olabel == self.EPS_LABEL:
                            # Non-advancing: position unchanged
                            new_pos = pos
                        else:
                            # Advancing: check output matches target
                            if pos < n_target:
                                if arc.olabel != target_tokens[pos]:
                                    continue  # output diverges from target
                                new_pos = pos + 1
                            else:
                                new_pos = pos  # past target, any output ok
                        # Apply eps_closure to successor
                        for cs, cout in self.eps_closure(nxt, missing_out=None):
                            final_pos = new_pos
                            for olabel in cout:
                                if final_pos < n_target:
                                    if olabel != target_tokens[final_pos]:
                                        break  # diverges
                                    final_pos += 1
                                # past target: any output ok
                            else:
                                successor.add((cs, min(final_pos, n_target)))

                if not successor:
                    return False
                successor = frozenset(successor)
                if successor not in visited:
                    queue.append(successor)

        return True

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def start_state(self) -> int:
        """Start state of the FST."""
        return self.fst.start()

    @property
    def num_states(self) -> int:
        """Number of states in the FST."""
        return self.fst.num_states()

    @property
    def all_output_labels(self) -> np.ndarray:
        """All non-epsilon output label IDs.  dtype=int16."""
        return self._all_output_labels

    @cached_property
    def all_universal(self) -> bool:
        """True if every reachable state is universal.

        When True, universality checks can be skipped entirely — every
        powerstate is trivially universal.  This is a computed FST fact,
        not a config knob.
        """
        return all(
            self.universal_states.get(s, False)
            for s in self.fst.states()
        )

    @cached_property
    def has_output_epsilon(self) -> bool:
        """True if any non-eps-input arc produces epsilon output.

        Such arcs are "non-advancing" — the input token is consumed but
        no output symbol is produced (e.g., delete_b's ``b/ε`` arc).
        When True, the fast logp_next path (first_output_for_ps) cannot
        be used because it assumes every input token produces exactly
        one output symbol.
        """
        for s in range(self.num_states):
            a = self.arcs(s)
            if len(a.in_sym) == 0:
                continue
            non_eps_input = a.in_sym != self.EPS_SYM
            if (non_eps_input & ~a.advances).any():
                return True
        return False

    # ── Eps-input output (all-universal fast path, mid-token beams) ──────

    @cached_property
    def _first_eps_output(self) -> np.ndarray:
        """Dense lookup: state → first output label from eps-input arcs.

        For "flower" FSTs (e.g. hf_realpha), each LM token maps to a
        multi-byte output via chains of eps-input arcs.  During
        decomposition, eps_closure filtering truncates beams at the target
        boundary, leaving beams at mid-token FST states.  These mid-token
        beams have their next output determined by the FST's eps-input
        arcs, not the LM — so they can be resolved without an LM call.

        Returns int16 array of shape (num_states,), where result[s] is the
        first non-epsilon output label produced by following eps-input arcs
        from state s, or -1 if state s has no eps-input output (i.e., it's
        a true token boundary like state 0 in a flower FST).

        Analogous to the old code's ``first_symbol_epsin_set(T, ps)``
        which checked eps-input output to skip LM calls for mid-token
        beams.  Used by the all-universal fast path in BatchedFastNextDist.
        """
        n = self.num_states
        result = np.full(n, -1, dtype=np.int16)
        eps_graph = self._eps_graph
        for s in range(n):
            for _nxt, olabel in eps_graph[s]:
                if olabel != self.EPS_LABEL:
                    result[s] = olabel
                    break
        return result

    # ── First-output table (all-universal fast path) ────────────────────

    @lru_cache(maxsize=None)
    def first_output_table(self, state: int) -> np.ndarray:
        """Dense lookup: pynini_label → first output label for advancing arcs.

        Returns int16 array of shape (max_pynini_label+1,), where
        result[tok] is the output label produced by reading tok at state,
        or -1 if no advancing arc exists for tok.

        Only includes advancing arcs (non-eps-input, output-producing).
        Cached per state.
        """
        a = self.arcs(state)
        size = max(self._in_sym_table.keys()) + 1 if self._in_sym_table else 1
        result = np.full(size, -1, dtype=np.int16)
        adv_mask = a.advances & (a.in_sym != self.EPS_SYM)
        if adv_mask.any():
            result[a.in_sym[adv_mask]] = a.out_label[adv_mask].astype(np.int16)
        return result

    def first_output_for_ps(
        self, ps: FrozenSet[int], tok_arr: np.ndarray,
    ) -> np.ndarray:
        """Vectorized first-output lookup for a powerstate.

        Like old first_symbol_vectorized: for each token in tok_arr, returns
        the output label produced by the powerstate, or -1 if no arc / conflict.
        """
        if len(ps) == 1:
            s = next(iter(ps))
            return self.first_output_table(s)[tok_arr]

        # Multi-state: stack per-state tables, check agreement
        states = sorted(ps)
        rows = np.stack([self.first_output_table(s)[tok_arr] for s in states])
        valid = rows >= 0
        if not valid.any():
            return np.full(len(tok_arr), -1, dtype=np.int16)

        HI = np.iinfo(np.int16).max
        min_v = np.min(np.where(valid, rows, HI), axis=0)
        max_v = np.max(np.where(valid, rows, -1), axis=0)
        has_any = min_v != HI
        agree = has_any & (min_v == max_v)
        out = np.full(len(tok_arr), -1, dtype=np.int16)
        out[agree] = min_v[agree]
        return out

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_eps_graph(self) -> list[list[tuple[int, int]]]:
        """Build adjacency list for input-epsilon arcs.

        Returns:
            List indexed by state.  Each entry is a list of
            (next_state, output_label_id) for arcs with ilabel == 0.
        """
        graph = []
        for s in self.fst.states():
            eps_edges = []
            for arc in self.fst.arcs(s):
                if arc.ilabel == 0:
                    eps_edges.append((arc.nextstate, arc.olabel))
            graph.append(eps_edges)
        return graph
