"""
Convert FSTs between the bundled FST library format and pynini format.

The bundled FST library uses:
  - Custom FST class with dict-of-dicts storage
  - Arbitrary hashable state names (ints, tuples, strings)
  - EPSILON = '' (empty string)

The rational-tokenizers library (pynini) uses:
  - OpenFst C++ backend with integer state IDs
  - Symbol tables mapping string labels <-> integer IDs
  - Epsilon = key 0 in symbol tables

Symbol table convention (matches bad_ungood.py, hf_realpha.py, dna2aa.py):
  - Key 0 = epsilon.  Its NAME is str(eps_name_id) (e.g. "257").
  - Keys 1..N = real symbols, named str(key) so init_symbols() can parse with int().
  - eps_name_id is NOT added as a separate key; it exists only as the name at key 0.
    TransducedLM.init_symbols() finds it via input_symbols().find(eos_out).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, FrozenSet, Optional, Set, Tuple

import pynini
from .fst import FST, EPSILON


def transduction_fst_to_pynini(
    fst: FST,
    *,
    eps_name_id: int = 257,
) -> Tuple[pynini.Fst, Dict, Dict]:
    """Convert a bundled-library FST to a pynini Fst.

    Args:
        fst: A FST instance.
        eps_name_id: Integer used as the name for the epsilon slot (key 0)
            in pynini symbol tables. Must not collide with any real symbol label.

    Returns:
        (pynini_fst, input_sym_map, output_sym_map) where:
          - pynini_fst is a pynini.Fst in the tropical semiring
          - input_sym_map  maps original symbol -> int label (1-based)
          - output_sym_map maps original symbol -> int label (1-based)
    """
    # ── 1. Collect symbols ────────────────────────────────────────────────
    input_syms: Set = fst.A - {EPSILON}
    output_syms: Set = fst.B - {EPSILON}

    # Build deterministic orderings for reproducibility
    in_sorted = sorted(input_syms, key=str)
    out_sorted = sorted(output_syms, key=str)

    # Assign integer labels starting at 1 (0 is epsilon).
    # Input and output symbol tables are independent in pynini,
    # so they get separate label sequences.  This keeps output labels
    # dense in [1, |B|] which the optimized code requires for array indexing.
    in_map: Dict = {}
    out_map: Dict = {}
    for i, s in enumerate(in_sorted, start=1):
        in_map[s] = i
    for i, s in enumerate(out_sorted, start=1):
        out_map[s] = i

    # Ensure eps_name_id doesn't collide
    used_labels = set(in_map.values()) | set(out_map.values())
    assert eps_name_id not in used_labels, (
        f"eps_name_id {eps_name_id} collides with a symbol label"
    )

    # ── 2. Build pynini symbol tables ─────────────────────────────────────
    isyms = pynini.SymbolTable()
    isyms.add_symbol(str(eps_name_id), 0)  # key 0 = epsilon
    for sym, lid in in_map.items():
        isyms.add_symbol(str(lid), lid)

    osyms = pynini.SymbolTable()
    osyms.add_symbol(str(eps_name_id), 0)  # key 0 = epsilon
    for sym, lid in out_map.items():
        osyms.add_symbol(str(lid), lid)

    # ── 3. Renumber states ────────────────────────────────────────────────
    state_list = sorted(fst.states, key=str)
    state_to_id: Dict = {s: i for i, s in enumerate(state_list)}

    # ── 4. Build pynini FST ───────────────────────────────────────────────
    pf = pynini.Fst()
    one = pynini.Weight.one(pf.weight_type())

    # Pre-create all states
    for _ in state_list:
        pf.add_state()

    # Handle start states: pynini supports only a single start state.
    # If there are multiple, create a fresh super-start with epsilon arcs.
    starts = fst.start
    if len(starts) == 1:
        (s0,) = starts
        pf.set_start(state_to_id[s0])
    elif len(starts) > 1:
        super_start = pf.add_state()
        pf.set_start(super_start)
        for s in starts:
            pf.add_arc(super_start, pynini.Arc(0, 0, one, state_to_id[s]))
    else:
        raise ValueError("FST has no start states")

    # Set final states
    for s in fst.stop:
        pf.set_final(state_to_id[s])

    # Add arcs
    for s in fst.states:
        sid = state_to_id[s]
        for a, b, t in fst.arcs(s):
            il = 0 if a == EPSILON else in_map[a]
            ol = 0 if b == EPSILON else out_map[b]
            pf.add_arc(sid, pynini.Arc(il, ol, one, state_to_id[t]))

    pf.set_input_symbols(isyms)
    pf.set_output_symbols(osyms)

    return pf, in_map, out_map






