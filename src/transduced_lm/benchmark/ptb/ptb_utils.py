"""
Shared PTB FST utilities for tests and benchmarks.

Provides:
  - Building the ported PTB FST from the bundled FST library
  - Relabeling byte-value decimal strings to single characters
  - RemappedGenLMRealpha: adapter bridging byte-indexed GenLMRealpha
    to the ported FST's pynini symbol IDs
  - Applying the FST to text to get output symbol sequences
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .fst import FST as NativeFST, EPSILON
from .ptb_fst_builder import build_ptb_fst_pynini

from .fst_converter import transduction_fst_to_pynini


# -- Constants ----------------------------------------------------------------

SEP_CHAR = chr(258)      # Word boundary marker in the relabeled PTB FST
EPS_NAME_ID = 257        # Integer ID used for epsilon in pynini symbol tables


# -- FST Construction ---------------------------------------------------------


def relabel_ptb_fst(native_fst: NativeFST) -> NativeFST:
    """Relabel byte-value decimal strings to single characters.

    The raw PTB FST from build_ptb_fst_pynini() uses decimal strings like '84'
    as symbol names. This relabels them to chr(84) = 'T', making each byte a
    single symbol that Precover/decomposition algorithms can process correctly.
    """
    new_fst = NativeFST()
    for state in native_fst.states:
        new_fst.states.add(state)
    for state in native_fst.start:
        new_fst.add_start(state)
    for state in native_fst.stop:
        new_fst.add_stop(state)
    for state in native_fst.states:
        for (inp, out, nxt) in native_fst.arcs(state):
            new_inp = chr(int(inp)) if inp != EPSILON else EPSILON
            new_out = chr(int(out)) if out != EPSILON else EPSILON
            new_fst.add_arc(state, new_inp, new_out, nxt)
    return new_fst


def build_ported_ptb() -> Tuple[NativeFST, "pynini.Fst", Dict[str, int], Dict[str, int]]:
    """Build the ported PTB FST from the bundled FST library.

    Returns:
        (ptb_fst, pynini_fst, in_map, out_map)
        - ptb_fst: Relabeled native FST (single-char symbols)
        - pynini_fst: Converted pynini FST
        - in_map: {char -> pynini_input_label_id}
        - out_map: {char -> pynini_output_label_id}
    """
    ptb_raw = build_ptb_fst_pynini()
    ptb_fst = relabel_ptb_fst(ptb_raw)
    pynini_fst, in_map, out_map = transduction_fst_to_pynini(
        ptb_fst, eps_name_id=EPS_NAME_ID
    )
    return ptb_fst, pynini_fst, in_map, out_map


# -- GenLMRealpha Remapping Adapter -------------------------------------------


class RemappedGenLMRealpha:
    """Adapter bridging a byte-indexed GenLMRealpha to the ported FST's pynini IDs.

    GenLMRealpha works with byte values (0-255, EOT=256):
      - Context: tuple of byte values
      - Output: np.ndarray of shape (257,) indexed by byte value

    The ported PTB FST uses arbitrary pynini label IDs assigned by
    transduction_fst_to_pynini(). This adapter remaps between the two:
      - Input contexts: pynini ID -> byte value (via ord(in_rev[pynini_id]))
      - Output arrays: byte value -> pynini ID (via in_map[chr(byte)])
    """

    EOT_IDX = 256
    EOS_IDX = 257    # EOS index in genlm-bytes 0.1.2 arrays
    EOS_LM_IDX = 0   # EOS probability stored at index 0 (remapped from EOS_IDX)

    def __init__(self, inner, in_map: Dict[str, int], max_id: int = None):
        """
        Args:
            inner: Real GenLMRealpha instance (byte-indexed)
            in_map: {char -> pynini_input_label_id} from build_ported_ptb()
            max_id: Array size for output (default: max value in in_map + 1)
        """
        self.inner = inner
        self.in_map = in_map
        self.in_rev = {v: k for k, v in in_map.items()}
        self.max_id = max_id or max(in_map.values(), default=256)

        # Precompute mappings for fast remapping
        # pynini_id -> byte_value
        self._pynini_to_byte = {}
        for char, pynini_id in in_map.items():
            byte_val = ord(char)
            if byte_val <= 255:
                self._pynini_to_byte[pynini_id] = byte_val

        # byte_value -> pynini_id
        self._byte_to_pynini = {v: k for k, v in self._pynini_to_byte.items()}

        # Vectorized remap arrays (numpy fancy indexing replaces Python loop)
        # _remap_byte_idx[i] = byte value for the i-th mapping
        # _remap_pynini_idx[i] = pynini ID for the i-th mapping
        _byte_vals = sorted(self._byte_to_pynini.keys())
        self._remap_byte_idx = np.array(_byte_vals, dtype=np.intp)
        self._remap_pynini_idx = np.array(
            [self._byte_to_pynini[b] for b in _byte_vals], dtype=np.intp
        )

    async def logp_next_for(self, ctx, **kwargs) -> np.ndarray:
        """Return log-prob array indexed by pynini IDs, given a pynini-ID context."""
        import time as _time
        _t0 = _time.perf_counter()

        # Convert pynini-ID context to byte-value context
        byte_ctx = tuple(
            self._pynini_to_byte.get(int(s), int(s))
            for s in ctx
        )

        _t1 = _time.perf_counter()
        # Get byte-indexed distribution from real GenLMRealpha
        byte_arr = await self.inner.logp_next_for(byte_ctx, **kwargs)

        _t2 = _time.perf_counter()
        # Remap to pynini-indexed array (vectorized)
        arr = np.full(self.max_id + 1, -np.inf, dtype=byte_arr.dtype)
        valid = self._remap_byte_idx < len(byte_arr)
        arr[self._remap_pynini_idx[valid]] = byte_arr[self._remap_byte_idx[valid]]

        # EOS probability at index 0 (epsilon slot) — from genlm EOS, not EOT
        if len(byte_arr) > self.EOS_IDX:
            arr[0] = byte_arr[self.EOS_IDX]

        _t3 = _time.perf_counter()

        if not hasattr(self, '_remap_timer'):
            self._remap_timer = {
                'n_calls': 0, 't_ctx_remap': 0.0,
                't_inner': 0.0, 't_arr_remap': 0.0,
            }
        self._remap_timer['n_calls'] += 1
        self._remap_timer['t_ctx_remap'] += (_t1 - _t0)
        self._remap_timer['t_inner'] += (_t2 - _t1)
        self._remap_timer['t_arr_remap'] += (_t3 - _t2)

        return arr

    # Delegate cleanup/cache methods to inner
    def empty_cache(self):
        self.inner.empty_cache()

    @property
    def llm(self):
        return self.inner.llm

    @property
    def root_beam(self):
        return self.inner.root_beam


# -- Text-to-Sequence Conversion ----------------------------------------------


def text_to_ptb_output_sequence(
    ptb_fst: NativeFST,
    text: str,
    out_map: Dict[str, int],
) -> list:
    """Apply the PTB FST to text and return pynini output label IDs as strings.

    Composes the input text with the PTB FST and reads off the output sequence,
    converting each output character to its pynini label ID (as string, for
    compatibility with sequence_logp_next).

    Args:
        ptb_fst: Relabeled native PTB FST (single-char symbols)
        text: Input text (e.g., "The cat sat on the mat.")
        out_map: {char -> pynini_output_label_id}

    Returns:
        List of string-encoded pynini output label IDs
    """
    input_fst = NativeFST.from_string(tuple(chr(b) for b in text.encode('utf-8')))
    output_fsa = (input_fst @ ptb_fst).project(1)
    lang_iter = output_fsa.language()
    output_chars = next(lang_iter, None)
    if output_chars is None:
        raise ValueError(
            f"PTB FST produced empty output for text of length {len(text)}."
        )
    return [str(out_map[c]) for c in output_chars]
