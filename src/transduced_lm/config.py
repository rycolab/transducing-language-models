"""
Standalone Config for decomposition — no torch or _LMBackend dependency.

This is a self-contained version of transducedLM.utils.config.Config,
keeping only the fields used by BeamDecomposition and TransducedLM.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class Config:
    """Configuration for decomposition."""

    prune_threshold: float = 0.001
    candidate_threshold: int = 500
    prune_threshold_alpha: float = 0.01
    max_prune_mass: float = 0.01
    max_candidates: int = None
    ignore_remainder: bool = False
    cover_opt: bool = False
    use_beam_cache: bool = True
    expand_threshold: int = 5
    max_steps: int = None
    skip_combined_univ: bool = False
    skip_target_universal: bool = False
    # Early stopping: stop expansion when unresolved frontier mass is less than
    # this fraction of resolved mass.  0.1 = fast (stops early, ~0.04 max error
    # on pathological FSTs like newspeak2).  1e-10 = exact (runs to convergence).
    stop_epsilon_mass: float = 0.1
    # Cap on unscored quotient beams per logp_next call (fast path only).
    # Keeps the top-K beams by logp, dropping low-probability tokenizations.
    # None = unlimited (exact).  500 is a good default for large-vocab FSTs.
    max_logp_next_beams: int = None
    verbose: bool = False
