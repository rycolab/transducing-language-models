"""
Setup helpers for benchmarking: variant wrapping, FST application, text encoding.

These are thin adapters that let the benchmark loop work with both the new
self-contained module and old logp_next variants from the optimization registry.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import pynini


# ---------------------------------------------------------------------------
# Variant wrapping
# ---------------------------------------------------------------------------

def make_variant_fn(
    variant_fn: Callable,
    old_tlm,
    old_config,
) -> Callable:
    """Wrap an old-style logp_next variant for use with sequence_logp_next.

    Old variants have signature::

        async def variant(transducedLM, config, context=..., **kw) -> Dict[int, float]

    This returns an async callable with the simple signature::

        async def logp_next(context: tuple) -> Dict[int, float]

    Args:
        variant_fn: An old-style logp_next function (e.g. _logp_next_v9).
        old_tlm: The old TransducedLM instance the variant expects.
        old_config: The old Config instance the variant expects.

    Returns:
        Async callable suitable for ``sequence_logp_next``.
    """
    async def wrapped(context: tuple):
        return await variant_fn(old_tlm, old_config, context=context)
    wrapped.__name__ = getattr(variant_fn, "__name__", "wrapped_variant")
    wrapped.__qualname__ = f"make_variant_fn.<locals>.wrapped[{wrapped.__name__}]"
    return wrapped


# ---------------------------------------------------------------------------
# FST application (transduce input → output symbols)
# ---------------------------------------------------------------------------

def apply_fst(
    fst: pynini.Fst,
    input_tokens: List[str],
    eps_id: int = 0,
) -> List[str]:
    """Transduce a sequence of input tokens through a pynini FST.

    Composes the input token sequence with the FST, extracts the shortest
    path, and returns the output symbol string labels (excluding epsilon).

    This is a standalone version of the old TransducedLM.apply(), with no
    dependency on the TransducedLM class.

    Args:
        fst: The pynini FST to apply.
        input_tokens: List of input symbol string labels (e.g. token ID
            strings like "123", or byte strings like "97").
        eps_id: Epsilon label ID in pynini space (always 0).

    Returns:
        List of output symbol string labels (e.g. ["97", "98", "99"]).

    Raises:
        ValueError: If an input token is not in the FST's input symbol table.
    """
    isyms = fst.input_symbols()
    acceptors = []
    for token in input_tokens:
        label = isyms.find(token)
        if label == -1:
            raise ValueError(f"Token {token!r} not found in FST input symbols.")
        acceptors.append(pynini.accep(token, token_type=isyms))

    if not acceptors:
        input_fst = pynini.accep("", token_type=isyms)
    else:
        input_fst = acceptors[0]
        for a in acceptors[1:]:
            input_fst = input_fst + a

    lattice = input_fst @ fst
    path = pynini.shortestpath(lattice, nshortest=1, unique=True)

    output_labels = []
    state = path.start()
    zero = pynini.Weight.zero(path.weight_type())
    while path.final(state) == zero:
        for arc in path.arcs(state):
            if arc.olabel != eps_id:
                output_labels.append(arc.olabel)
            state = arc.nextstate
            break

    osyms = path.output_symbols()
    if osyms is not None:
        return [osyms.find(lab) for lab in output_labels]
    return [str(lab) for lab in output_labels]


def encode_text_as_bytes(
    text: str,
    out_sym_to_id: dict[str, int],
) -> List[int]:
    """Encode a text string as a sequence of output symbol IDs (byte-level).

    For byte-level FSTs where output symbols are named "0"-"255", this
    converts each byte of the UTF-8 encoding to the corresponding output
    symbol ID.

    Args:
        text: The text to encode.
        out_sym_to_id: Mapping from output symbol string labels to pynini
            output label IDs.  Typically ``vfst._out_sym_to_id``.

    Returns:
        List of output symbol IDs suitable for ``sequence_logp_next``.

    Raises:
        KeyError: If a byte value has no corresponding output symbol.
    """
    return [out_sym_to_id[str(b)] for b in text.encode("utf-8")]


def encode_tokens(
    text: str,
    tokenizer,
    fst: pynini.Fst,
    out_sym_to_id: dict[str, int],
    eps_id: int = 0,
) -> List[int]:
    """Encode text by tokenizing then transducing through the FST.

    For token-level FSTs (e.g. hf_realpha) where the input side is HF
    token IDs and the output side is bytes:
      1. Tokenize text with the HF tokenizer
      2. Transduce through the FST to get output symbol strings
      3. Map output symbol strings to output label IDs

    Args:
        text: The text to encode.
        tokenizer: HF tokenizer with ``.encode()`` method.
        fst: The pynini FST.
        out_sym_to_id: Mapping from output symbol strings to IDs.
        eps_id: Epsilon label ID (default 0).

    Returns:
        List of output symbol IDs suitable for ``sequence_logp_next``.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    input_tokens = [str(tid) for tid in token_ids]
    output_strs = apply_fst(fst, input_tokens, eps_id=eps_id)
    return [out_sym_to_id[s] for s in output_strs]
