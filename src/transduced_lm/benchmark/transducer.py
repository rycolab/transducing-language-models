"""
Transducer loading and logp_next function construction for benchmarks.

Supports four transducer types:
  - hf_realpha:  HF tokenizer → bytes FST (GENLM_ASYNC, no powerstates)
  - ptb:         Penn Treebank tokenization FST (GENLM_BYTES, powerstates)
  - ptb_ported:  PTB FST ported from the transduction library (GENLM_BYTES, powerstates)
  - hf_dna2aa:   DNA codon → amino acid FST (GENLM_ASYNC, no powerstates)

Each loader returns a TransducerSetup that bundles a VectorizedFST (for the
refactored module) and symbol mappings needed for data encoding and result
interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..vectorized_fst import VectorizedFST
from ..config import Config


# ---------------------------------------------------------------------------
# AsyncLMAdapter — wraps a raw genlm LLM for the new module's interface
# ---------------------------------------------------------------------------


class AsyncLMAdapter:
    """Wrap a genlm LLM backend to provide ``logp_next_for`` for the new module.

    For GENLM_ASYNC transducers (hf_realpha, hf_dna2aa) the raw LLM already
    produces distributions over the FST's input symbol vocabulary.  This
    adapter mirrors the old ``gather_ctx2logdist`` GENLM_ASYNC path: prepend
    ``bos_id``, call ``llm.next_token_logprobs``, convert to numpy.

    The new module indexes everything by pynini arc labels, so this adapter
    remaps:
      - Context (pynini labels) → HF token IDs for the LLM call
      - Output (HF-token-ID-indexed distribution) → pynini-label-indexed array
    """

    def __init__(self, llm, in_sym_table: dict[int, str]):
        self.llm = llm
        self.bos_id = llm.tokenizer.bos_token_id

        # Build pynini_label <-> HF token ID mappings from the symbol table.
        # Symbol strings are HF token ID strings (e.g. "0", "1", ..., "50256").
        # Epsilon (pynini label 0) is excluded.
        self._pynini_to_ext = {}  # pynini label → HF token ID
        self._ext_to_pynini = {}  # HF token ID → pynini label
        for pynini_label, sym_str in in_sym_table.items():
            if pynini_label == 0:  # skip epsilon
                continue
            ext_id = int(sym_str)
            self._pynini_to_ext[pynini_label] = ext_id
            self._ext_to_pynini[ext_id] = pynini_label

        # Precompute numpy arrays for fast remapping
        max_pynini = max(in_sym_table.keys()) + 1 if in_sym_table else 1
        vocab_size = len(llm.tokenizer)

        # pynini_label → ext_id lookup array (for context conversion)
        self._pynini_to_ext_arr = np.zeros(max_pynini, dtype=np.int64)
        for pl, ext in self._pynini_to_ext.items():
            self._pynini_to_ext_arr[pl] = ext

        # ext_id → pynini_label lookup array (for distribution remapping)
        # Non-FST tokens map to -1 (will get -inf)
        self._ext_to_pynini_arr = np.full(vocab_size, -1, dtype=np.int32)
        for ext, pl in self._ext_to_pynini.items():
            if ext < vocab_size:
                self._ext_to_pynini_arr[ext] = pl

        # Precompute valid mapping pairs for vectorized scatter
        valid_ext = np.array(
            [ext for ext, pl in self._ext_to_pynini.items() if ext < vocab_size],
            dtype=np.int64,
        )
        valid_pynini = np.array(
            [self._ext_to_pynini[ext] for ext in valid_ext],
            dtype=np.int64,
        )
        self._valid_ext = valid_ext
        self._valid_pynini = valid_pynini
        self._out_size = max_pynini

    async def logp_next_for(self, ctx):
        # Convert pynini-label context to HF token ID context
        ext_ctx = [int(self._pynini_to_ext_arr[t]) for t in ctx]
        result = await self.llm.next_token_logprobs([self.bos_id, *ext_ctx])
        raw_dist = result.detach().float().cpu().numpy()

        # Remap HF-token-ID-indexed distribution to pynini-label-indexed
        out = np.full(self._out_size, -np.inf, dtype=np.float32)
        out[self._valid_pynini] = raw_dist[self._valid_ext]
        return out

    async def batch_logp_next_for(
        self, ctxs: list[tuple], chunk_size: int = 512,
    ) -> list[np.ndarray]:
        """Batched LM scoring: chunked GPU calls for multiple contexts.

        Uses llm.batch_next_token_logprobs for true GPU-parallel processing
        with KV cache sharing, matching the old GENLM_ASYNC path.

        Chunks large batches to avoid GPU OOM (each chunk produces a
        (chunk_size, vocab_size) tensor on GPU).
        """
        in_ids = [
            [self.bos_id] + [int(self._pynini_to_ext_arr[t]) for t in ctx]
            for ctx in ctxs
        ]

        results = []
        for start in range(0, len(in_ids), chunk_size):
            chunk = in_ids[start:start + chunk_size]
            scores = await self.llm.batch_next_token_logprobs(chunk)
            raw_dists = scores.detach().float().cpu().numpy()
            for raw_dist in raw_dists:
                out = np.full(self._out_size, -np.inf, dtype=np.float32)
                out[self._valid_pynini] = raw_dist[self._valid_ext]
                results.append(out)
        return results


# ---------------------------------------------------------------------------
# TransducerSetup
# ---------------------------------------------------------------------------


@dataclass
class TransducerSetup:
    """Everything needed to run a benchmark on a given transducer."""

    old_tlm: Any
    """Old TransducedLM instance (for old variant dispatch)."""

    vfst: VectorizedFST
    """New module's VectorizedFST (for clean logp_next / future implementations)."""

    out_sym_to_id: Dict[str, int]
    """Output symbol string → pynini output label ID."""

    out_id_to_sym: Dict[int, str]
    """Pynini output label ID → output symbol string."""

    transducer_name: str
    """One of: 'hf_realpha', 'ptb', 'ptb_ported', 'hf_dna2aa'."""

    genlm_realpha: Any = None
    """GenLMRealpha instance (only for GENLM_BYTES backend, i.e. ptb)."""

    lm: Any = None
    """LM object for the new TransducedLM (must have async logp_next_for)."""

    # Extra metadata stored by specific loaders
    extra: Dict[str, Any] = field(default_factory=dict)
    """Loader-specific extras (e.g. ptb_fst, in_map, out_map, aa_map)."""


# ---------------------------------------------------------------------------
# Transducer loaders
# ---------------------------------------------------------------------------


def load_transducer(
    name: str,
    llm=None,
    model_name: str | None = None,
    *,
    fasta_file: str | None = None,
) -> TransducerSetup:
    """Load a transducer by name, returning a TransducerSetup.

    Args:
        name: Transducer name ('hf_realpha', 'ptb_ported', 'hf_dna2aa', 'baseline').
        llm: Pre-loaded genlm LLM backend (required for hf_realpha/dna2aa).
        model_name: HF model name string.

    Returns:
        TransducerSetup with old_tlm, vfst, symbol mappings, etc.
    """
    if name == "hf_realpha":
        return _load_hf_realpha(llm, model_name)
    elif name == "ptb_ported":
        return _load_ptb_ported()
    elif name == "hf_dna2aa":
        return _load_dna2aa(llm, model_name)
    elif name == "baseline":
        return _load_baseline(llm, model_name)
    else:
        raise ValueError(f"Unknown transducer: {name!r}")


def _load_hf_realpha(llm, model_name) -> TransducerSetup:
    from .fst_loaders import build_hf_realpha_fst

    pynini_fst, eos_out, out_sym_to_id, out_id_to_sym, extra = (
        build_hf_realpha_fst(model_name, llm=llm)
    )

    vfst = VectorizedFST(pynini_fst, eos_out=eos_out)
    # All states are universal by construction (star on tokenizer).
    vfst.universal_states = {s: True for s in pynini_fst.states()}

    return TransducerSetup(
        old_tlm=None,
        vfst=vfst,
        out_sym_to_id=out_sym_to_id,
        out_id_to_sym=out_id_to_sym,
        transducer_name="hf_realpha",
        lm=AsyncLMAdapter(llm, vfst._in_sym_table),
    )


_ptb_ported_univ_cache: dict | None = None
_ptb_ported_set_cache: dict | None = None


def _load_ptb_ported() -> TransducerSetup:
    global _ptb_ported_univ_cache, _ptb_ported_set_cache
    from .ptb.ptb_utils import build_ported_ptb

    ptb_fst, pynini_fst, in_map, out_map = build_ported_ptb()

    # Push output labels to eliminate most output-epsilon arcs (352 → ~31).
    # This makes the beam search more effective: each input-consuming arc
    # is more likely to also produce output, so fewer BFS steps are "silent".
    import pynini as _pynini
    pynini_fst = _pynini.push(pynini_fst, push_labels=True,
                               push_weights=False, remove_total_weight=False)
    pynini_fst.rmepsilon()

    vfst = VectorizedFST(pynini_fst, eos_out=str(257))
    if _ptb_ported_univ_cache is not None:
        vfst.universal_states = dict(_ptb_ported_univ_cache)
        if _ptb_ported_set_cache is not None:
            vfst._universal_set_cache = dict(_ptb_ported_set_cache)
    else:
        vfst.compute_universal_states()
        _ptb_ported_univ_cache = dict(vfst.universal_states)

    # For ported PTB, out_map is {char → pynini_id}
    out_sym_to_id = {str(v): v for k, v in out_map.items()}
    out_id_to_sym = {v: str(v) for k, v in out_map.items()}

    return TransducerSetup(
        old_tlm=None,
        vfst=vfst,
        out_sym_to_id=out_sym_to_id,
        out_id_to_sym=out_id_to_sym,
        transducer_name="ptb_ported",
        extra={
            "ptb_fst": ptb_fst,
            "pynini_fst": pynini_fst,
            "in_map": in_map,
            "out_map": out_map,
        },
    )


def _load_dna2aa(llm, model_name) -> TransducerSetup:
    from .fst_loaders import build_dna2aa_fst

    pynini_fst, eos_out, out_sym_to_id, out_id_to_sym, extra = (
        build_dna2aa_fst(llm=llm, llm_name=model_name)
    )

    vfst = VectorizedFST(pynini_fst, eos_out=eos_out)
    # All states are universal by construction.
    vfst.universal_states = {s: True for s in pynini_fst.states()}

    return TransducerSetup(
        old_tlm=None,
        vfst=vfst,
        out_sym_to_id=out_sym_to_id,
        out_id_to_sym=out_id_to_sym,
        transducer_name="hf_dna2aa",
        lm=AsyncLMAdapter(llm, vfst._in_sym_table),
        extra={"aa_map": extra.get("aa_map", {})},
    )


def _load_baseline(llm, model_name) -> TransducerSetup:
    """Load a Vieira et al. baseline: no FST, just GenLMRealpha byte-level scoring.

    Returns a minimal TransducerSetup with vfst=None.  GenLMRealpha is attached
    later via setup_genlm().  Output symbols are raw bytes (0-255).
    """
    out_sym_to_id = {str(i): i for i in range(256)}
    out_id_to_sym = {i: str(i) for i in range(256)}

    return TransducerSetup(
        old_tlm=None,
        vfst=None,
        out_sym_to_id=out_sym_to_id,
        out_id_to_sym=out_id_to_sym,
        transducer_name="baseline",
    )


def build_baseline_logp_next_fn(setup: TransducerSetup):
    """Build a logp_next function for the Vieira baseline (genlm bytes, no FST).

    Returns an async callable (context: tuple[int,...]) -> dict[int, float]
    that queries GenLMRealpha for the byte-level next-byte distribution.
    """
    raw_genlm = setup.extra.get("raw_genlm")
    if raw_genlm is None:
        raise ValueError("GenLMRealpha not attached. Call setup_genlm() first.")

    from .genlm_realpha import EOS_IDX

    async def baseline_logp_next(context: tuple) -> dict:
        # GenLMRealpha.logp_next_for returns a numpy array of shape (258,)
        # indexed by byte value (0-255) + EOT (256) + EOS (257).
        raw = await raw_genlm.logp_next_for(context)
        raw = np.asarray(raw, dtype=np.float32)
        # Bytes 0-255 + EOS (257); drop EOT (256, internal to genlm)
        dist = {i: float(raw[i]) for i in range(min(256, len(raw)))}
        if len(raw) > EOS_IDX:
            dist[EOS_IDX] = float(raw[EOS_IDX])
        return dist

    baseline_logp_next.__name__ = "baseline_genlm_bytes"
    return baseline_logp_next


# ---------------------------------------------------------------------------
# GenLMRealpha setup / cleanup
# ---------------------------------------------------------------------------


class ByteIndexedLMAdapter:
    """Adapter bridging byte-indexed GenLMRealpha to pynini-label-indexed arrays.

    GenLMRealpha returns distributions indexed by byte value (0-255 + 256=EOT + 257=EOS).
    The new module indexes by pynini arc label. This adapter remaps between the
    two using the VectorizedFST's _in_sym_table.
    """

    EOS_LM_IDX = 0  # EOS at pynini label 0 (epsilon position, unused otherwise)
    GENLM_EOS_IDX = 257  # EOS index in genlm-bytes 0.1.2 arrays

    def __init__(self, inner, in_sym_table: dict[int, str]):
        self.inner = inner

        # Build pynini_label <-> byte_value mappings.
        # Symbol strings for PTB are decimal byte value strings ("0"-"255").
        self._pynini_to_byte = {}
        self._byte_to_pynini = {}
        for pynini_label, sym_str in in_sym_table.items():
            if pynini_label == 0:  # skip epsilon
                continue
            try:
                byte_val = int(sym_str)
            except ValueError:
                continue
            if 0 <= byte_val <= 255:
                self._pynini_to_byte[pynini_label] = byte_val
                self._byte_to_pynini[byte_val] = pynini_label

        # Output array size
        max_pynini = max(in_sym_table.keys()) + 1 if in_sym_table else 1
        self._out_size = max_pynini

        # Vectorized remap arrays
        _byte_vals = sorted(self._byte_to_pynini.keys())
        self._remap_byte_idx = np.array(_byte_vals, dtype=np.intp)
        self._remap_pynini_idx = np.array(
            [self._byte_to_pynini[b] for b in _byte_vals], dtype=np.intp
        )

    async def logp_next_for(self, ctx, **kwargs) -> np.ndarray:
        # Convert pynini-label context to byte-value context
        byte_ctx = tuple(
            self._pynini_to_byte.get(int(s), int(s)) for s in ctx
        )
        # Get byte-indexed distribution from GenLMRealpha
        raw = await self.inner.logp_next_for(byte_ctx, **kwargs)
        raw = np.asarray(raw, dtype=np.float32)

        # Remap to pynini-label-indexed array
        out = np.full(self._out_size, -np.inf, dtype=np.float32)
        out[self._remap_pynini_idx] = raw[self._remap_byte_idx]
        # Map genlm EOS (byte index 257) to EOS_LM_IDX (pynini label 0)
        if len(raw) > self.GENLM_EOS_IDX:
            out[self.EOS_LM_IDX] = raw[self.GENLM_EOS_IDX]
        return out


async def setup_genlm(
    setup: TransducerSetup,
    model_name: str,
    llm=None,
    K: int = 8,
    prune_threshold: float = 0.001,
    beam_cache_mb: int = 20_000,
) -> None:
    """Create GenLMRealpha and attach it to both old and new TransducedLM.

    For both PTB variants: wraps in a remapping adapter to bridge byte-indexed
    distributions to the FST's pynini symbol IDs.

    Mutates *setup* in place (sets .genlm_realpha, .lm, .old_tlm.genlm_realpha).
    """
    from .genlm_realpha import GenLMRealpha

    raw_genlm = await GenLMRealpha.create(
        model_name, llm=llm, K=K, prune_threshold=prune_threshold,
        beam_cache_mb=beam_cache_mb,
    )

    if setup.transducer_name == "baseline":
        # Baseline: no FST remapping, use raw GenLMRealpha directly
        setup.genlm_realpha = raw_genlm
        setup.lm = raw_genlm
    elif setup.transducer_name == "ptb_ported":
        from .ptb.ptb_utils import (
            RemappedGenLMRealpha,
        )

        in_map = setup.extra["in_map"]
        remapped = RemappedGenLMRealpha(raw_genlm, in_map)
        setup.genlm_realpha = remapped
        setup.lm = remapped
    else:
        # Production PTB: wrap in ByteIndexedLMAdapter for pynini remapping
        adapted = ByteIndexedLMAdapter(raw_genlm, setup.vfst._in_sym_table)
        setup.genlm_realpha = raw_genlm
        setup.lm = adapted

    # Store the raw genlm for cleanup
    setup.extra["raw_genlm"] = raw_genlm


async def cleanup_genlm(setup: TransducerSetup) -> None:
    """Clean up GenLMRealpha: empty caches, clear KV cache, cleanup beams."""
    raw_genlm = setup.extra.get("raw_genlm")
    if raw_genlm is None:
        return

    raw_genlm.empty_cache()
    raw_genlm.llm.clear_cache()
    if hasattr(raw_genlm.llm, "clear_kv_cache"):
        raw_genlm.llm.clear_kv_cache()
    await raw_genlm.root_beam.cleanup()

    setup.genlm_realpha = None
    setup.lm = None
    setup.extra.pop("raw_genlm", None)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

# Per-transducer default settings
_TRANSDUCER_DEFAULTS = {
    "hf_realpha": dict(
        ignore_remainder=True,
    ),
    "ptb_ported": dict(
        ignore_remainder=True,
        max_steps=5,
    ),
    "hf_dna2aa": dict(
        ignore_remainder=True,
    ),
    "baseline": dict(
        ignore_remainder=True,
    ),
}


def get_default_config(
    transducer_name: str,
    prune_threshold: float = 0.001,
    candidate_threshold: int = 100,
    prune_threshold_alpha: float = 0.7,
    max_prune_mass: float = 0.4,
    max_candidates: int = None,
    max_logp_next_beams: int = None,
    max_steps: int = None,
    stop_epsilon_mass: float = 0.1,
    ignore_remainder: bool = None,
    expand_threshold: int = 3,
    verbose: bool = False,
) -> Config:
    """Return a new-module Config with per-transducer defaults.

    Note: stop_epsilon_mass defaults to 0.1 (Config default) for speed.
    This can cause -inf at rare positions; use fallback_fn in the sequence
    loop to recover (see sequence_logp_next).

    Note: max_steps defaults to per-transducer setting (5 for PTB) when
    not explicitly provided. Without this limit, the frontier can grow
    exponentially at certain positions, causing the loop to stall.

    Note: ignore_remainder defaults to the per-transducer setting (True
    for PTB/hf_realpha) when not explicitly provided.
    """
    defaults = _TRANSDUCER_DEFAULTS.get(transducer_name, {})
    effective_max_steps = max_steps if max_steps is not None else defaults.get("max_steps")
    effective_ignore_remainder = (
        ignore_remainder if ignore_remainder is not None
        else defaults.get("ignore_remainder", True)
    )
    return Config(
        prune_threshold=prune_threshold,
        candidate_threshold=candidate_threshold,
        prune_threshold_alpha=prune_threshold_alpha,
        max_prune_mass=max_prune_mass,
        max_candidates=max_candidates,
        max_logp_next_beams=max_logp_next_beams,
        ignore_remainder=effective_ignore_remainder,
        use_beam_cache=True,
        expand_threshold=expand_threshold,
        stop_epsilon_mass=stop_epsilon_mass,
        max_steps=effective_max_steps,
        verbose=verbose,
    )



# ---------------------------------------------------------------------------
# logp_next function construction
# ---------------------------------------------------------------------------


def build_logp_next_fn(
    setup: TransducerSetup,
    impl: str = "logp_next",
    prune_threshold: float = 0.001,
    max_steps: int = None,
    skip_combined_univ: bool = True,
    skip_target_universal: bool = True,
    stop_epsilon_mass: float = 0.1,
    candidate_threshold: int = 100,
    prune_threshold_alpha: float = 0.7,
    max_prune_mass: float = 0.4,
    max_candidates: int = None,
    max_logp_next_beams: int = None,
    ignore_remainder: bool = None,
    expand_threshold: int = 3,
    verbose: bool = False,
    **config_kwargs,
) -> tuple[Callable, Callable | None, Callable | None, Callable | None, Callable | None, Callable | None]:
    """Build an async logp_next callable (and optional fallback) for the benchmark loop.

    Args:
        setup: TransducerSetup from load_transducer().
        impl: Implementation name. 'logp_next' (default) uses the batched
            logp_next; 'clean' uses the per-symbol decomposition (slower
            reference); 'clean_v5' is an alias for 'logp_next'.
        prune_threshold: Prune threshold for decomposition.
        max_steps: Max beam expansion steps (None = unlimited).
        skip_combined_univ: Skip combined universality check (safe for PTB).
        stop_epsilon_mass: Early-stop threshold for the expansion loop.
        candidate_threshold: Pivot point for adaptive pruning ramp-up.
        prune_threshold_alpha: Steepness of pruning increase above candidate_threshold.
        max_prune_mass: Cap on fraction of total mass pruned per step.
        max_candidates: Hard cap on frontier size (None = unlimited).
        max_logp_next_beams: Cap on unscored Q beams per logp_next call (fast path).
        ignore_remainder: Ignore remainder contributions (None = per-transducer default).
        expand_threshold: Skip further expansion when covering beams are this many
            symbols past the target and Q is non-empty.
        verbose: Print per-step info.

    Returns:
        Tuple of (logp_next_fn, fallback_fn, on_fallback, on_recover,
        probe_fn, score_single_fn).
        fallback_fn is logp_next_clean on the same TransducedLM (shares
        caches) for 'logp_next' impl, or None otherwise.
        on_fallback is tlm.tighten_expansion — adaptively lowers
        stop_epsilon_mass, raises max_steps, and temporarily relaxes
        pruning thresholds.
        on_recover is tlm.restore_pruning — restores original pruning
        thresholds after the retry block.
        probe_fn is tlm.probe_target — cheap target reachability check
        (single decomposition, no expansion loop).
        score_single_fn is tlm.score_single_symbol — targeted single-symbol
        scoring (one decomposition instead of full expansion loop).
    """
    config_params = dict(
        prune_threshold=prune_threshold,
        max_steps=max_steps,
        stop_epsilon_mass=stop_epsilon_mass,
        candidate_threshold=candidate_threshold,
        prune_threshold_alpha=prune_threshold_alpha,
        max_prune_mass=max_prune_mass,
        max_candidates=max_candidates,
        max_logp_next_beams=max_logp_next_beams,
        ignore_remainder=ignore_remainder,
        expand_threshold=expand_threshold,
        verbose=verbose,
    )
    if impl == "clean":
        return _build_clean_fn(setup, **config_params), None, None, None, None, None
    elif impl in ("logp_next", "clean_v5"):
        return _build_clean_v5_fn(
            setup, skip_combined_univ=skip_combined_univ,
            skip_target_universal=skip_target_universal, **config_params,
        )
    else:
        raise ValueError(f"Unknown impl: {impl!r}. Use 'logp_next' or 'clean'.")


def _build_clean_fn(setup, **config_params):
    """Build logp_next using the new module's per-symbol decomposition (slow, correct)."""
    from ..transduced_lm import TransducedLM as NewTransducedLM

    if setup.lm is None:
        raise ValueError(
            "No LM attached to setup. Call setup_genlm() first for PTB, "
            "or provide an LM for other transducers."
        )

    config = get_default_config(setup.transducer_name, **config_params)
    new_tlm = NewTransducedLM(setup.vfst, setup.lm, config)
    return new_tlm.logp_next_clean


def _build_clean_v5_fn(setup, *, skip_combined_univ=True, skip_target_universal=True, **config_params):
    """Build logp_next (batched) + fallback (logp_next_clean) on same TLM.

    Both functions share the same TransducedLM instance, so beam caches
    and LM caches are shared.  The fallback (logp_next_clean) does
    per-symbol decomposition and doesn't use the expansion loop at all,
    so stop_epsilon_mass is irrelevant for it.

    Returns:
        Tuple of (logp_next_fn, fallback_fn, on_fallback, on_recover).
    """
    from ..transduced_lm import TransducedLM as NewTransducedLM

    if setup.lm is None:
        raise ValueError(
            "No LM attached to setup. Call setup_genlm() first for PTB, "
            "or provide an LM for other transducers."
        )

    config = get_default_config(setup.transducer_name, **config_params)
    config.skip_combined_univ = skip_combined_univ
    config.skip_target_universal = skip_target_universal
    new_tlm = NewTransducedLM(setup.vfst, setup.lm, config)
    return (new_tlm.logp_next, new_tlm.logp_next_clean,
            new_tlm.tighten_expansion, new_tlm.restore_pruning,
            new_tlm.probe_target, new_tlm.score_single_symbol)


def build_cleanup_fn(setup: TransducerSetup) -> callable:
    """Build a cleanup function that reclaims GPU memory.

    The genlm HuggingFace backend accumulates a TokenTrie with cached
    logprobs (on CPU) and processes queries via model forward passes that
    allocate GPU tensors.  Over long sequences, unreleased GPU tensors
    can accumulate, eventually causing CUDA errors that are silently
    caught by GenLMRealpha (returning all-inf distributions) and cause
    the expansion loop to stall.

    The cleanup function forces garbage collection and releases unused
    GPU memory.  If the lightweight cleanup isn't sufficient, it also
    clears genlm's TokenTrie and beam caches (forcing re-computation).

    Returns:
        A no-arg callable, or None if genlm is not attached.
    """
    raw_genlm = setup.extra.get("raw_genlm")
    if raw_genlm is None:
        return None

    llm = raw_genlm.llm
    _call_count = [0]

    def _cleanup():
        import gc

        _call_count[0] += 1

        # Always: GC + release unused CUDA memory
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

        # Clear genlm's token-level cache every cleanup
        # (the TokenTrie accumulates logprobs nodes monotonically).
        # This forces re-computation but prevents unbounded growth.
        if hasattr(llm, 'clear_cache'):
            llm.clear_cache()
        # Clear beam LRU cache (candidates reference the old trie
        # for walk_cache lookups; stale references are harmless for
        # correctness but clearing avoids confusion).
        raw_genlm.empty_cache()
        raw_genlm._beams[()] = raw_genlm.root_beam

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

        # Force glibc to return freed pages to the OS.  Without this,
        # Python's allocator holds heap pages from evicted/cleared beams
        # indefinitely (heap fragmentation), causing RSS to ratchet up
        # monotonically across paragraphs even though the live set shrinks.
        try:
            import ctypes
            _libc = ctypes.CDLL("libc.so.6")
            _libc.malloc_trim(0)
        except (OSError, AttributeError):
            pass

    return _cleanup
