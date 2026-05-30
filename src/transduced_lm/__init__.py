"""
Refactored transduced language model module.

Clean reimplementation of the transducedLM package, built alongside the original
for equivalence testing. Key improvements:
  - VectorizedFST: owns all FST state, typed interface, clear ID namespaces
  - BeamDecomposition: class-based decomposition using VectorizedFST
  - TransducedLM: composes VectorizedFST + LM scorer + Config
"""

import builtins
if not hasattr(builtins, 'profile'):
    builtins.profile = lambda f: f

from .config import Config
from .vectorized_fst import VectorizedFST, ArcArrays
from .beam_decomposition import BeamDecomposition, AbstractDecomp, beam_decompose
from .transduced_lm import TransducedLM

__all__ = [
    "Config",
    "VectorizedFST",
    "ArcArrays",
    "BeamDecomposition",
    "AbstractDecomp",
    "beam_decompose",
    "TransducedLM",
]
