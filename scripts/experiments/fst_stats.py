#!/usr/bin/env python
"""Print FST statistics for all model/transducer combinations.

Usage:
    python -m transduced_lm.benchmark.gpu_scripts.fst_stats
"""

import sys
import time


def count_arcs(pynini_fst):
    """Count total arcs in a pynini FST."""
    return sum(pynini_fst.num_arcs(s) for s in pynini_fst.states())


def count_universal(vfst):
    """Count universal states from VectorizedFST."""
    return sum(1 for v in vfst.universal_states.values() if v)


def load_realpha(model_name):
    """Load hf_realpha FST for a given model."""
    from transduced_lm.benchmark.transducer import load_transducer
    from genlm.backend import load_model_by_name

    llm = load_model_by_name(model_name, backend="hf")
    setup = load_transducer("hf_realpha", llm=llm, model_name=model_name)
    return setup.vfst


def load_ptb():
    """Load ptb_ported FST (model-independent)."""
    from transduced_lm.benchmark.transducer import load_transducer

    setup = load_transducer("ptb_ported")
    return setup.vfst


def load_dna2aa(model_name="vesteinn/gpt2-dna"):
    """Load hf_dna2aa FST."""
    from transduced_lm.benchmark.transducer import load_transducer
    from genlm.backend import load_model_by_name

    llm = load_model_by_name(model_name, backend="hf")
    setup = load_transducer("hf_dna2aa", llm=llm, model_name=model_name)
    return setup.vfst


COMBINATIONS = [
    ("hf_realpha", "gpt2-large", load_realpha),
    ("hf_realpha", "meta-llama/Llama-3.2-1B", load_realpha),
    ("hf_realpha", "meta-llama/Llama-3.1-8B", load_realpha),
    ("ptb_ported", "(any)", lambda _: load_ptb()),
    ("hf_dna2aa", "vesteinn/gpt2-dna", load_dna2aa),
]


def main():
    header = f"{'Transducer':<14} {'Model':<30} {'States':>8} {'Arcs':>10} {'Universal':>10} {'All Univ':>10}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for transducer, model, loader in COMBINATIONS:
        try:
            t0 = time.time()
            vfst = loader(model)
            dt = time.time() - t0

            n_states = vfst.num_states
            n_arcs = count_arcs(vfst.fst)
            n_univ = count_universal(vfst)
            all_univ = vfst.all_universal

            print(
                f"{transducer:<14} {model:<30} {n_states:>8,} {n_arcs:>10,} "
                f"{n_univ:>10,} {'yes' if all_univ else 'no':>10}"
                f"  ({dt:.1f}s)"
            )
        except Exception as e:
            print(f"{transducer:<14} {model:<30} ERROR: {e}")

    print(sep)


if __name__ == "__main__":
    main()
