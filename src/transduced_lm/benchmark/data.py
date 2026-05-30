"""
Data loading utilities for benchmarks.

Loads text/sequence data and encodes it as lists of output symbol IDs,
ready for sequence_logp_next.

Supports:
  - WikiText paragraphs (for hf_realpha, ptb, ptb_ported)
  - FASTA amino acid sequences (for hf_dna2aa)
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .transducer import TransducerSetup


# ---------------------------------------------------------------------------
# WikiText utilities (inlined from benchmarking/utils/data_utils.py)
# ---------------------------------------------------------------------------


def wikitext_detokenize(string: str) -> str:
    """Remove WikiText whitespace tokenization artifacts.

    Taken from NVIDIA Megatron-LM detokenizer.
    """
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace("  . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def _load_wikitext_dataset(split: str = "test"):
    """Load WikiText-2 raw dataset."""
    from datasets import load_dataset
    return load_dataset("wikitext", "wikitext-2-raw-v1", split=split)


# ---------------------------------------------------------------------------
# WikiText sequence loading
# ---------------------------------------------------------------------------


def load_wikitext_sequences(
    setup: TransducerSetup,
    split: str = "test",
    n: int = 4,
    max_len: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[List[int]], List[str], int]:
    """Load WikiText paragraphs as output symbol ID sequences.

    Loads n non-heading paragraphs from WikiText, detokenizes them,
    applies the transducer FST to get output symbols, and converts
    to pynini output label IDs.

    Args:
        setup: TransducerSetup from load_transducer().
        split: Dataset split ('test', 'train', 'validation').
        n: Number of paragraphs to load.
        max_len: If set, truncate total output to this many symbols.
        verbose: Print paragraph lengths.

    Returns:
        (sequences, original_texts, total_len):
          - sequences: List of paragraphs, each a list of output symbol IDs (int).
          - original_texts: The original detokenized text strings.
          - total_len: Total number of output symbols across all paragraphs.
    """
    dataset = _load_wikitext_dataset(split)
    sequences = []
    original_texts = []
    total_len = 0

    for item in dataset:
        text = item["text"].strip()
        if not text or text.startswith("="):
            continue

        text = wikitext_detokenize(text)
        seq = _encode_text(setup, text)

        if max_len is not None:
            remaining = max_len - total_len
            if remaining <= 0:
                break
            seq = seq[:remaining]

        sequences.append(seq)
        original_texts.append(text)
        total_len += len(seq)

        if verbose:
            print(
                f"Paragraph {len(sequences)} len {len(seq)} "
                f"cumulative length {total_len}"
            )

        if len(sequences) >= n:
            break

    return sequences, original_texts, total_len


# ---------------------------------------------------------------------------
# WikiText raw byte loading (for Vieira baseline)
# ---------------------------------------------------------------------------


def load_wikitext_bytes(
    split: str = "test",
    n: int = 4,
    max_len: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[List[int]], List[str], int]:
    """Load WikiText paragraphs as raw UTF-8 byte sequences.

    For the Vieira baseline (no FST): text → UTF-8 bytes directly,
    matching GenLMRealpha's byte-level interface.

    Args:
        split: Dataset split ('test', 'train', 'validation').
        n: Number of paragraphs to load.
        max_len: If set, truncate total output to this many bytes.
        verbose: Print paragraph lengths.

    Returns:
        (sequences, original_texts, total_len):
          - sequences: List of paragraphs, each a list of byte values (0-255).
          - original_texts: The original detokenized text strings.
          - total_len: Total number of bytes across all paragraphs.
    """
    dataset = _load_wikitext_dataset(split)
    sequences = []
    original_texts = []
    total_len = 0

    for item in dataset:
        text = item["text"].strip()
        if not text or text.startswith("="):
            continue

        text = wikitext_detokenize(text)
        seq = list(text.encode("utf-8"))

        if max_len is not None:
            remaining = max_len - total_len
            if remaining <= 0:
                break
            seq = seq[:remaining]

        sequences.append(seq)
        original_texts.append(text)
        total_len += len(seq)

        if verbose:
            print(
                f"Paragraph {len(sequences)} len {len(seq)} "
                f"cumulative length {total_len}"
            )

        if len(sequences) >= n:
            break

    return sequences, original_texts, total_len


# ---------------------------------------------------------------------------
# FASTA sequence loading
# ---------------------------------------------------------------------------


def load_fasta_sequences(
    setup: TransducerSetup,
    file_path: str,
    n: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[List[int]], int]:
    """Load FASTA amino acid sequences as output symbol ID sequences.

    Args:
        setup: TransducerSetup from load_transducer() with hf_dna2aa.
        file_path: Path to FASTA file.
        n: Max number of protein sequences to load (None = all).
        verbose: Print sequence lengths.

    Returns:
        (sequences, total_len):
          - sequences: List of protein sequences, each a list of output symbol IDs.
          - total_len: Total number of output symbols.
    """
    aa_map = setup.extra.get("aa_map", {})
    if not aa_map:
        raise ValueError("No aa_map in setup.extra — use hf_dna2aa transducer.")

    out_sym_to_id = setup.out_sym_to_id
    sequences = []
    current_seq_chars = []
    total_len = 0

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq_chars:
                    seq = _encode_fasta_chars(current_seq_chars, aa_map, out_sym_to_id)
                    sequences.append(seq)
                    total_len += len(seq)
                    current_seq_chars = []
                    if n is not None and len(sequences) >= n:
                        break
            else:
                current_seq_chars.append(line)

        if current_seq_chars and (n is None or len(sequences) < n):
            seq = _encode_fasta_chars(current_seq_chars, aa_map, out_sym_to_id)
            sequences.append(seq)
            total_len += len(seq)

    if verbose:
        for i, seq in enumerate(sequences):
            print(f"Protein {i+1} len {len(seq)}")
        print(f"Total length: {total_len}")

    return sequences, total_len


def _encode_fasta_chars(
    seq_lines: List[str],
    aa_map: dict,
    out_sym_to_id: dict,
) -> List[int]:
    """Convert FASTA amino acid characters to output symbol IDs."""
    aa_str = "".join(seq_lines)
    result = []
    for c in aa_str:
        aa_sym = aa_map.get(c)
        if aa_sym is not None:
            result.append(int(out_sym_to_id.get(aa_sym, aa_sym)))
    return result


# ---------------------------------------------------------------------------
# Text-to-sequence encoding
# ---------------------------------------------------------------------------


def _encode_text(setup: TransducerSetup, text: str) -> List[int]:
    """Encode text as a list of output symbol IDs using the transducer FST.

    Uses the appropriate encoding strategy for each transducer type:
      - hf_realpha: tokenize → transduce through FST → output symbol IDs
      - ptb_ported: Uses the native transduction library FST to compute output
      - hf_dna2aa: Should not reach here (use load_fasta_sequences instead)
    """
    if setup.transducer_name == "ptb_ported":
        return _encode_text_ptb_ported(setup, text)
    elif setup.transducer_name == "hf_realpha":
        return _encode_text_via_fst(setup, text)
    else:
        return _encode_text_via_fst(setup, text)


def _encode_text_via_fst(
    setup: TransducerSetup,
    text: str,
) -> List[int]:
    """Encode text by tokenizing then transducing through the FST."""
    from .setup import encode_tokens
    llm = setup.extra.get("llm") or (setup.lm.llm if hasattr(setup.lm, "llm") else None)
    if llm is None:
        raise ValueError("No LLM available for tokenization. Attach LLM via setup.extra['llm'].")
    return encode_tokens(text, llm.tokenizer, setup.vfst.fst, setup.out_sym_to_id)


def _encode_text_ptb_ported(setup: TransducerSetup, text: str) -> List[int]:
    """Encode text using the ported PTB FST from the transduction library."""
    from .ptb.ptb_utils import (
        text_to_ptb_output_sequence,
    )

    ptb_fst = setup.extra["ptb_fst"]
    out_map = setup.extra["out_map"]
    seq_strs = text_to_ptb_output_sequence(ptb_fst, text, out_map)
    return [int(s) for s in seq_strs]
