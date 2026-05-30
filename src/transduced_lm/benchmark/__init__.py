"""
Benchmarking utilities for the refactored transduced_lm module.

Provides:
  - sequence_logp_next: score a sequence position-by-position with any logp_next fn
  - make_variant_fn: wrap old-style logp_next variants for use with sequence_logp_next
  - apply_fst: transduce input through a pynini FST to get output symbols
  - encode_text_as_bytes: convert text to byte-level output symbol IDs
  - TransducerSetup: bundles transducer state for benchmarks
  - load_transducer: load any supported transducer type
  - setup_genlm / cleanup_genlm: GenLMRealpha lifecycle management
  - build_logp_next_fn: create a pluggable logp_next callable
  - get_default_config / build_cleanup_fn: per-transducer Config defaults and cache cleanup
  - load_wikitext_sequences / load_fasta_sequences: data loading
  - safe_load_pickle / atomic_pickle_dump: incremental result storage
"""

from .sequence import sequence_logp_next
from .setup import make_variant_fn, apply_fst, encode_text_as_bytes
from .transducer import (
    TransducerSetup,
    load_transducer,
    setup_genlm,
    cleanup_genlm,
    build_cleanup_fn,
    build_logp_next_fn,
    get_default_config,
)
from .data import load_wikitext_sequences, load_fasta_sequences
from .storage import safe_load_pickle, atomic_pickle_dump

__all__ = [
    "sequence_logp_next",
    "make_variant_fn",
    "apply_fst",
    "encode_text_as_bytes",
    "TransducerSetup",
    "load_transducer",
    "setup_genlm",
    "cleanup_genlm",
    "build_cleanup_fn",
    "build_logp_next_fn",
    "get_default_config",
    "load_wikitext_sequences",
    "load_fasta_sequences",
    "safe_load_pickle",
    "atomic_pickle_dump",
]
