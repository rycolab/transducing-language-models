"""
GenLMRealpha: genlm-bytes beam search adapter for byte-level LM scoring.

Provides a cached async interface for computing byte-level next-token
log-probabilities using the genlm library's beam search.

Self-contained copy of transducers/genlm_realpha.py with no old-code deps.
"""

from typing import Dict, Tuple, Any
from cachetools import LRUCache
from collections import defaultdict
from genlm.bytes import ByteBeamState, BeamParams
from genlm.backend import load_model_by_name

import numpy as np

NEG_INF = float("-inf")

NestedCtx = Tuple[Any, ...]
EOT_IDX = 256
EOS_IDX = 257
GENLM_ARRAY_SIZE = 258  # 256 bytes + EOT (256) + EOS (257)


class GenLMRealpha:
    @classmethod
    async def create(
        cls,
        model_name: str,
        llm=None,
        backend="hf",
        K: int = 5,
        prune_threshold: float = 0.01,
        verbose: bool = False,
        beam_cache_mb: int = 20_000,
    ) -> "GenLMRealpha":
        if llm is None:
            llm = load_model_by_name(model_name, backend=backend)

        # Derive LLM EOS token bytes so genlm-bytes can compute EOS probability.
        eos_token_bytes = []
        if hasattr(llm, 'tokenizer') and hasattr(llm.tokenizer, 'eos_token_id'):
            eos_id = llm.tokenizer.eos_token_id
            if eos_id is not None and hasattr(llm, 'byte_vocab'):
                # genlm-bytes >=0.2 returns a Token (subclass of bytes) here.
                # Token.__hash__ is by token_id, not byte content, so set
                # membership against the trie's plain-bytes vocabulary mismatches
                # even when the bytes are equal, and TokenByteTrie._build_trie
                # raises "EOS byte string ... not in vocabulary".  Coerce to
                # plain bytes before passing to BeamParams.
                eos_bytes = bytes(llm.byte_vocab[eos_id])
                eos_token_bytes = [eos_bytes]
                if verbose:
                    print(f"  GenLMRealpha: EOS token bytes = {eos_bytes!r}")

        root_beam = await ByteBeamState.initial(
            llm, BeamParams(
                K=K, prune_threshold=prune_threshold,
                eos_tokens=eos_token_bytes or None,
            )
        )
        return cls(llm, root_beam, K, prune_threshold, verbose,
                   beam_cache_mb=beam_cache_mb)

    def __init__(
        self,
        llm: Any,
        root_beam: ByteBeamState,
        K: int,
        prune_threshold: float,
        verbose: bool = False,
        beam_cache_mb: int = 20_000,
    ):
        self.llm = llm
        self.K = K
        self.prune_threshold = prune_threshold
        self.root_beam = root_beam

        # Compute per-beam memory from the actual trie structure.
        # Each ByteBeamState holds up to K LazyTrieState objects, each with a
        # _mass numpy array of shape (num_trie_nodes,) in float64.
        # The trie is shared; only the _mass arrays are per-beam.
        per_beam_bytes = self._estimate_beam_bytes(root_beam, K)
        budget_bytes = beam_cache_mb * 1024 * 1024
        max_entries = max(100, budget_bytes // max(per_beam_bytes, 1))
        self._beam_size_estimate = per_beam_bytes
        self._beams: LRUCache = LRUCache(maxsize=max_entries)
        self._beams[()] = root_beam
        self._ctx: NestedCtx = ()
        self.verbose = verbose
        if verbose:
            print(f"  GenLMRealpha: beam cache budget={beam_cache_mb}MB, "
                  f"per_beam≈{per_beam_bytes / 1024 / 1024:.1f}MB, "
                  f"max_entries={max_entries}")

    @staticmethod
    def _estimate_beam_bytes(beam: ByteBeamState, K: int) -> int:
        """Estimate memory footprint of a single ByteBeamState with K states.

        The root beam only has 1 state, but child beams (after prune/step)
        hold up to K states. We use max(len(beam.states), K) so that the
        budget is computed against the worst-case child size, not the
        artificially small root beam.
        """
        n_actual = len(beam.states)
        if n_actual == 0:
            return 0

        n_states = max(n_actual, K)

        # Try to measure from actual materialized _mass arrays
        measured = 0
        n_measured = 0
        for state in beam.states:
            mass = getattr(state, '_mass', None)
            if mass is not None and hasattr(mass, 'nbytes'):
                measured += mass.nbytes
                n_measured += 1

        if n_measured > 0:
            per_state = measured // n_measured
            return n_states * per_state

        # Fallback: estimate from trie structure
        trie = beam.states[0].trie
        inner_trie = getattr(trie, 'trie', trie)
        n_nodes = len(inner_trie.children)
        return n_states * n_nodes * 8

    def empty_cache(self):
        self._beams.clear()


    async def logp_next_for(
        self, ctx, *, dtype=np.float32
    ) -> np.ndarray:
        import time as _time
        try:
            _t0 = _time.perf_counter()
            beam = await self._beam_for(ctx)
            _t1 = _time.perf_counter()
            try:
                lbp = await beam.logp_next()
            except RuntimeError as e:
                if "CUDA error" in str(e) or "illegal memory access" in str(e).lower():
                    if self.verbose:
                        print("WARNING: CUDA illegal memory access in logp_next:", e)
                return np.full(GENLM_ARRAY_SIZE, NEG_INF, dtype=dtype)
            _t2 = _time.perf_counter()

            ps = getattr(lbp, "ps", None)
            if ps is not None:
                arr = np.asarray(ps, dtype=dtype)
                if arr.shape != (GENLM_ARRAY_SIZE,):
                    raise ValueError(
                        f"Unexpected LazyByteProbs.ps shape {arr.shape}, "
                        f"expected ({GENLM_ARRAY_SIZE},)"
                    )
            else:
                Q = lbp.materialize()
                arr = np.full(GENLM_ARRAY_SIZE, NEG_INF, dtype=dtype)
                for k, v in Q.items():
                    if k is None:
                        arr[EOT_IDX] = v
                    elif isinstance(k, int) and 0 <= k < GENLM_ARRAY_SIZE:
                        arr[k] = v
            _t3 = _time.perf_counter()

            # Accumulate timing
            if not hasattr(self, '_genlm_timer'):
                self._genlm_timer = {
                    'n_calls': 0, 'n_beam_miss': 0,
                    't_beam': 0.0, 't_logp_next': 0.0, 't_materialize': 0.0,
                }
            self._genlm_timer['n_calls'] += 1
            self._genlm_timer['t_beam'] += (_t1 - _t0)
            self._genlm_timer['t_logp_next'] += (_t2 - _t1)
            self._genlm_timer['t_materialize'] += (_t3 - _t2)

        except (AssertionError, ValueError) as e:
            if self.verbose:
                print("WARNING: Caught genlm", e, "…")
            arr = np.full(GENLM_ARRAY_SIZE, NEG_INF, dtype=dtype)
        return arr

    async def _beam_for(self, ctx: NestedCtx) -> ByteBeamState:
        """Recursively build (and cache) a beam for ctx."""
        if ctx in self._beams:
            return self._beams[ctx]

        if ctx == ():
            self._beams[()] = self.root_beam
            return self.root_beam

        parent = ctx[:-1]
        ch = ctx[-1]
        parent_beam = await self._beam_for(parent)
        beam = await (parent_beam.prune() << int(ch))
        self._beams[ctx] = beam
        if hasattr(self, '_genlm_timer'):
            self._genlm_timer['n_beam_miss'] += 1
        return beam

    @staticmethod
    def _materialize(beam: ByteBeamState) -> Dict[str, float]:
        logp_next = beam.logp_next_sync()
        return (
            logp_next.materialize()
            .map_keys(lambda x: bytes([x]).decode("utf-8") if x is not None else "EOT")
            .to_dict()
        )

    async def cleanup(self):
        await self.root_beam.cleanup()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.cleanup()
