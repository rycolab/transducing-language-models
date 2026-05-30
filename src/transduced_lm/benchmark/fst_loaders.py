"""
Self-contained FST construction for benchmarks.

Builds pynini FSTs for each transducer type without depending on the old
TransducedLM class. Each builder returns the raw pynini FST and symbol
mappings needed to construct a VectorizedFST.

Extracted from transducers/hf_realpha.py, transducers/ptb.py,
transducers/dna2aa.py.
"""

from __future__ import annotations

import os
import operator
from functools import reduce
from typing import Dict, Tuple

import pynini
from pynini import cross, cdrewrite, union


# ---------------------------------------------------------------------------
# Helpers (from transducers/utils/construction.py)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# HF Realpha FST
# ---------------------------------------------------------------------------


def build_hf_realpha_fst(
    model_name: str,
    llm=None,
) -> Tuple[pynini.Fst, str, Dict[str, int], Dict[int, str], dict]:
    """Build the HF tokenizer → bytes FST.

    Args:
        model_name: HuggingFace model name.
        llm: Pre-loaded genlm LLM backend (required).

    Returns:
        (pynini_fst, eos_out, out_sym_to_id, out_id_to_sym, extra)
        extra contains: {"llm": llm}
    """
    from genlm.backend.tokenization.bytes import get_byte_vocab

    tokenizer = llm.tokenizer
    input_symtab = pynini.SymbolTable(name="input_symtab")
    output_symtab = pynini.SymbolTable(name="output_symtab")

    byte_vocab = get_byte_vocab(tokenizer)
    tokens = sorted(tokenizer.vocab.items(), key=lambda x: x[1])

    max_token_id = len(tokens) - 1
    EPS = str(max_token_id + 1)
    input_symtab.add_symbol(EPS, 0)
    output_symtab.add_symbol(EPS, 0)

    EOS_IN = str(llm.tokenizer.eos_token_id)
    EOS_OUT = str(max_token_id + 2)

    # Reserve all 256 byte values on the output side (labels 1-256),
    # then EOS output (label 257).  Order matters: adding EOS_OUT first
    # would auto-assign it label 1, colliding with byte 0.
    for b in range(256):
        output_symtab.add_symbol(str(b), b + 1)
    output_symtab.add_symbol(EOS_OUT, 257)

    T = pynini.Fst()
    start_state = T.add_state()
    T.set_start(start_state)
    T.set_final(start_state)

    def add_arc(src, ilabel, olabel, dst):
        i_id = (
            0
            if ilabel in (EPS, "")
            else (
                input_symtab.find(ilabel)
                if input_symtab.find(ilabel) >= 0
                else input_symtab.add_symbol(ilabel)
            )
        )
        o_id = 0 if olabel in (EPS, "") else output_symtab.find(olabel)
        arc = pynini.Arc(i_id, o_id, pynini.Weight.one("tropical"), dst)
        T.add_arc(src, arc)

    add_arc(start_state, EOS_IN, EOS_OUT, start_state)

    for bytes_vals, (token_str, token) in zip(byte_vocab, tokens):
        if token_str in tokenizer.all_special_tokens:
            continue

        current_state = start_state
        current_input = token
        for idx, byte_val in enumerate(bytes_vals):
            next_state = start_state if idx == len(bytes_vals) - 1 else T.add_state()
            add_arc(
                current_state,
                str(current_input) if idx == 0 else EPS,
                str(byte_val),
                next_state,
            )
            current_state = next_state
            current_input = EPS

    T.set_input_symbols(input_symtab)
    T.set_output_symbols(output_symtab)
    T.optimize()

    # Build symbol maps from the pynini symbol tables
    out_sym_to_id = {}
    out_id_to_sym = {}
    for label_id in range(output_symtab.num_symbols()):
        sym_str = output_symtab.find(label_id)
        if sym_str != "<eps>" and label_id != 0:
            out_sym_to_id[sym_str] = label_id
            out_id_to_sym[label_id] = sym_str

    return T, EOS_OUT, out_sym_to_id, out_id_to_sym, {"llm": llm}


# ---------------------------------------------------------------------------
# DNA → Amino Acid FST
# ---------------------------------------------------------------------------

GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def build_dna2aa_fst(
    llm=None,
    llm_name: str | None = None,
) -> Tuple[pynini.Fst, str, Dict[str, int], Dict[int, str], dict]:
    """Build the DNA codon → amino acid FST.

    Args:
        llm: Pre-loaded genlm LLM backend. If None, loads model_name.
        llm_name: Model name for loading LLM if llm is None.

    Returns:
        (pynini_fst, eos_out, out_sym_to_id, out_id_to_sym, extra)
        extra contains: {"aa_map": dict, "llm": llm}
    """
    if llm is None:
        import genlm.backend.llm.base as _genlm_base
        _orig = _genlm_base.decode_vocab
        _genlm_base.decode_vocab = lambda tok, **kw: (
            [b""] * len(tok.get_vocab()), [""] * len(tok.get_vocab())
        )
        try:
            from genlm.backend import load_model_by_name
            llm = load_model_by_name(llm_name, backend="hf")
        finally:
            _genlm_base.decode_vocab = _orig

    EPS = 0
    EOS_STR = "0"

    in_sym = pynini.SymbolTable("bytes_in")
    all_ins = list("ACGT")
    max_in = len(all_ins)
    in_sym.add_symbol(str(max_in + 1), EPS)
    codon_map = {}
    in_counter = 1
    for ch in "ACGT":
        in_sym.add_symbol(str(in_counter), in_counter)
        codon_map[ch] = str(in_counter)
        in_counter += 1
    in_sym.add_symbol(EOS_STR, max_in + 1)

    out_sym = pynini.SymbolTable("aa_out")
    all_outs = list("ACDEFGHIKLMNPQRSTVWY") + ["*"]
    max_out = len(all_outs)

    out_sym.add_symbol(str(max_out + 1), EPS)
    out_counter = 1
    aa_map = {}
    for aa in list("ACDEFGHIKLMNPQRSTVWY") + ["*"]:
        out_sym.add_symbol(str(out_counter), out_counter)
        aa_map[aa] = str(out_counter)
        out_counter += 1
    out_sym.add_symbol(EOS_STR, max_out + 1)

    CODON2AA_BYTE = {codon: aa_map[aa] for codon, aa in GENETIC_CODE.items()}

    fst = pynini.Fst()
    fst.set_input_symbols(in_sym)
    fst.set_output_symbols(out_sym)
    one = pynini.Weight.one(fst.weight_type())

    start = fst.add_state()
    fst.set_start(start)
    fst.set_final(start, one)

    A_id = in_sym.find(str(codon_map["A"]))
    C_id = in_sym.find(str(codon_map["C"]))
    G_id = in_sym.find(str(codon_map["G"]))
    T_id = in_sym.find(str(codon_map["T"]))
    EPS_OUT = EPS

    first = {}
    for ch, lbl in (("A", A_id), ("C", C_id), ("G", G_id), ("T", T_id)):
        s1 = fst.add_state()
        fst.set_final(s1, one)
        fst.add_arc(start, pynini.Arc(lbl, EPS_OUT, one, s1))
        first[ch] = s1

    second = {}
    for b1, s1 in first.items():
        for ch, lbl in (("A", A_id), ("C", C_id), ("G", G_id), ("T", T_id)):
            s2 = fst.add_state()
            fst.set_final(s2, one)
            fst.add_arc(s1, pynini.Arc(lbl, EPS_OUT, one, s2))
            second[b1 + ch] = s2

    for two, s2 in second.items():
        for ch, lbl in (("A", A_id), ("C", C_id), ("G", G_id), ("T", T_id)):
            codon = two + ch
            aa_byte = CODON2AA_BYTE[codon]
            fst.add_arc(s2, pynini.Arc(lbl, int(aa_byte), one, start))

    fst.rmepsilon()
    fst.optimize()

    # Build symbol maps
    out_sym_to_id = {}
    out_id_to_sym = {}
    for label_id in range(out_sym.num_symbols()):
        sym_str = out_sym.find(label_id)
        if label_id != 0:
            out_sym_to_id[sym_str] = label_id
            out_id_to_sym[label_id] = sym_str

    return fst, EOS_STR, out_sym_to_id, out_id_to_sym, {"aa_map": aa_map, "llm": llm}
