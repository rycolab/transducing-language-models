"""
FST advance helpers for powerstate transitions.

These are pure FST operations that advance a powerstate (frozenset of state IDs)
by one input symbol, tracking output labels produced. Used by:
  - PSCache (cached transitions)
  - UnbatchedFastNextDist / BatchedFastNextDist (expansion loops)

Paper notation:
  next_frontier(F, x') → _advance_ps(vfst, ps, tok)
  F' ignoring labels   → _advance_ps_unlabeled(vfst, ps, tok)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .vectorized_fst import VectorizedFST


def _advance_ps(vfst: VectorizedFST, ps: frozenset, tok: int):
    """Advance powerstate by one non-epsilon input symbol.

    Follows all arcs with in_sym == tok from each state in ps.
    For arcs producing epsilon output, does eps_closure to find deferred
    output symbols.

    Returns:
        scored: dict[int, frozenset[int]] — non-eps output label → next states
        unscored: frozenset[int] — states reached with no output produced
    """
    scored = defaultdict(set)
    eps_reached = set()

    for state in ps:
        a = vfst.arcs(state)
        mask = a.in_sym == tok

        # Non-epsilon output arcs → scored by output label, then eps_closure
        scored_mask = mask & a.advances
        for i in np.where(scored_mask)[0]:
            out_label = int(a.out_label[i])
            next_state = int(a.next_state[i])
            # Apply eps_closure to the reached state — ALL closure states
            # join scored[out_label] regardless of additional eps output.
            # The arc already committed out_label as the FIRST output symbol;
            # any eps_closure output is at deeper positions and doesn't change
            # which output symbol this state is grouped under.
            for cls_state, out_tuple in vfst.eps_closure(next_state):
                scored[out_label].add(int(cls_state))

        # Epsilon output arcs → need eps closure to find deferred output
        eps_mask = mask & ~a.advances
        for i in np.where(eps_mask)[0]:
            eps_reached.add(int(a.next_state[i]))

    # Input-epsilon closure for states reached with epsilon output
    unscored = set()
    for s in eps_reached:
        closure = vfst.eps_closure(s)
        for cls_state, out_tuple in closure:
            if out_tuple:
                scored[int(out_tuple[0])].add(int(cls_state))
            else:
                unscored.add(int(cls_state))

    return (
        {k: frozenset(v) for k, v in scored.items()},
        frozenset(unscored),
    )


def _split_at_boundary(vfst: VectorizedFST, ps: frozenset, tok: int, out_lab: int):
    """Split a scored advance by whether eps_closure adds output beyond out_lab.

    When _advance_ps groups states under a single output label, some states
    may have been reached via eps_closure with additional output symbols.
    This function separates:

      at_boundary: states with no eps output beyond the scored label
      beyond: dict mapping the first additional output symbol to states

    Used by catching-up beams that reach the context boundary: states with
    additional eps output are already committed to that output symbol and
    should be scored/locked for it, not treated as unlocked at the boundary.
    """
    at_boundary = set()
    beyond = defaultdict(set)

    for state in ps:
        a = vfst.arcs(state)
        mask = a.in_sym == tok

        # Non-eps-output arcs producing out_lab
        scored_mask = mask & a.advances & (a.out_label == out_lab)
        for i in np.where(scored_mask)[0]:
            next_state = int(a.next_state[i])
            for cls_state, out_tuple in vfst.eps_closure(next_state):
                if out_tuple:
                    beyond[int(out_tuple[0])].add(int(cls_state))
                else:
                    at_boundary.add(int(cls_state))

        # Eps-output arcs where eps_closure first output == out_lab
        eps_mask = mask & ~a.advances
        for i in np.where(eps_mask)[0]:
            next_state = int(a.next_state[i])
            for cls_state, out_tuple in vfst.eps_closure(next_state):
                if out_tuple and int(out_tuple[0]) == out_lab:
                    if len(out_tuple) > 1:
                        beyond[int(out_tuple[1])].add(int(cls_state))
                    else:
                        at_boundary.add(int(cls_state))

    return frozenset(at_boundary), {k: frozenset(v) for k, v in beyond.items()}


def _advance_ps_unlabeled(vfst: VectorizedFST, ps: frozenset, tok: int):
    """Advance powerstate by one input symbol, ignoring output labels.

    Used for states already locked to an output symbol — we only need
    the new powerstate, not the output produced.

    Returns:
        frozenset of all reachable next states (including eps closure)
    """
    result = set()

    for state in ps:
        a = vfst.arcs(state)
        mask = a.in_sym == tok
        for i in np.where(mask)[0]:
            result.add(int(a.next_state[i]))

    # Input-epsilon closure on all reached states
    expanded = set()
    for s in result:
        for cls_state, _ in vfst.eps_closure(s):
            expanded.add(int(cls_state))

    return frozenset(expanded)


def _is_combined_universal_for_sym(
    vfst: VectorizedFST,
    scored_ps: frozenset,
    unscored_ps: frozenset,
    out_sym: int,
    input_syms: list[int],
) -> bool:
    """Check if scored_ps ∪ unscored_ps is effectively universal for out_sym.

    Conditions:
    1. Combined powerstate has a final state
    2. Combined powerstate is input-universal (accepts Σ*)
    3. For inputs where scored_ps has no arcs, unscored_ps must produce
       only output out_sym (not a different symbol)
    """
    combined = scored_ps | unscored_ps

    if not vfst.has_final(combined):
        return False

    if not vfst.is_universal(combined):
        return False

    for tok in input_syms:
        scored_succ = _advance_ps_unlabeled(vfst, scored_ps, tok)
        if scored_succ:
            continue

        unscored_groups, unscored_unscored = _advance_ps(
            vfst, unscored_ps, tok
        )
        for produced_sym in unscored_groups:
            if produced_sym != out_sym:
                return False
        if not unscored_groups and unscored_unscored:
            return False

    return True
