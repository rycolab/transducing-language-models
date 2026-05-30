#!/usr/bin/env python3
"""
Compute JSD, cross-entropy, and bits/byte between baseline (genlm-bytes)
and transduced LM (realpha) results across thresholds.

Loads paired pkl files from results/ and prints a summary table per model.

Usage:
    python scripts/experiments/cr/analyze_jsd.py
    python scripts/experiments/cr/analyze_jsd.py --results-dir results/
    python scripts/experiments/cr/analyze_jsd.py --baseline-K 16
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


# ── JSD helpers ──────────────────────────────────────────────────────────────

BYTE_KEYS = [str(i) for i in range(256)]


def dist_to_prob_vec(d: dict) -> np.ndarray:
    """Extract 256-dim probability vector from a {str_id: logp} dict."""
    logps = np.array([d.get(k, -300.0) for k in BYTE_KEYS], dtype=np.float64)
    probs = np.exp(logps)
    s = probs.sum()
    if s > 0:
        probs /= s
    return probs


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    nz = p > 0
    if not nz.any():
        return 0.0
    return float(np.sum(p[nz] * (np.log2(p[nz]) - np.log2(q[nz]))))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    if len(values) < 2:
        m = float(np.mean(values)) if len(values) else float("nan")
        return m, m, m
    values = np.asarray(values)
    boot = np.array([
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    return float(np.mean(values)), float(np.percentile(boot, lo)), float(np.percentile(boot, hi))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Per-model comparison ─────────────────────────────────────────────────────

def compare_baseline_vs_realpha(
    bl_data: dict,
    rl_data: dict,
    n_bootstrap: int = 1000,
) -> list[dict]:
    """Compare baseline distributions vs realpha at each threshold.

    Returns a list of result dicts (one per realpha threshold).
    """
    bl_stats = bl_data.get("stats", {})
    bl_pnexts = bl_data.get("p_nexts", {})
    bl_ths = list(bl_pnexts.keys())
    if not bl_ths:
        return []
    bl_ths_key = bl_ths[0]
    bl_paras = bl_pnexts[bl_ths_key]
    bl_indices = bl_stats.get(bl_ths_key, {}).get("para_indices", list(range(len(bl_paras))))

    bl_lps = np.array(bl_stats.get(bl_ths_key, {}).get("log_probs", []))
    bl_times = np.array(bl_stats.get(bl_ths_key, {}).get("times", []))
    bl_n = len(bl_lps)
    bl_ce = float(-np.sum(bl_lps) / bl_n) if bl_n else float("nan")
    bl_bpb = bl_ce / np.log(2) if bl_n else float("nan")
    bl_throughput = bl_n / np.sum(bl_times) if len(bl_times) and np.sum(bl_times) > 0 else float("nan")

    rl_stats = rl_data.get("stats", {})
    rl_pnexts = rl_data.get("p_nexts", {})

    results = []

    for ths in sorted(rl_pnexts.keys()):
        rl_paras = rl_pnexts[ths]
        rl_indices = rl_stats.get(ths, {}).get("para_indices", list(range(len(rl_paras))))

        common = sorted(set(bl_indices) & set(rl_indices))
        if not common:
            continue

        # Per-position JSD
        per_para_jsds = {}
        all_jsds = []
        for idx in common:
            bl_i = bl_indices.index(idx)
            rl_i = rl_indices.index(idx)
            bl_dists = bl_paras[bl_i]
            rl_dists = rl_paras[rl_i]
            n = min(len(bl_dists), len(rl_dists))
            para_jsds = []
            for j in range(n):
                p = dist_to_prob_vec(bl_dists[j])
                q = dist_to_prob_vec(rl_dists[j])
                jsd = jensen_shannon_divergence(p, q)
                if np.isfinite(jsd):
                    para_jsds.append(max(0.0, jsd))
            per_para_jsds[idx] = para_jsds
            all_jsds.extend(para_jsds)

        # Realpha stats
        rl_lps = np.array(rl_stats.get(ths, {}).get("log_probs", []))
        rl_times = np.array(rl_stats.get(ths, {}).get("times", []))
        rl_n = len(rl_lps)
        rl_ce = float(-np.sum(rl_lps) / rl_n) if rl_n else float("nan")
        rl_bpb = rl_ce / np.log(2) if rl_n else float("nan")
        rl_throughput = rl_n / np.sum(rl_times) if len(rl_times) and np.sum(rl_times) > 0 else float("nan")

        jsd_mean, jsd_lo, jsd_hi = bootstrap_ci(all_jsds, n_bootstrap) if all_jsds else (float("nan"),) * 3
        n_positions = len(all_jsds)

        results.append({
            "threshold": ths,
            "jsd_mean": jsd_mean,
            "jsd_ci_lo": jsd_lo,
            "jsd_ci_hi": jsd_hi,
            "ce_nats": rl_ce,
            "bits_per_byte": rl_bpb,
            "bytes_per_sec": rl_throughput,
            "n_paras": len(common),
            "n_positions": n_positions,
        })

    return results, {
        "ce_nats": bl_ce,
        "bits_per_byte": bl_bpb,
        "bytes_per_sec": bl_throughput,
        "n_paras": len(bl_indices),
        "n_symbols": bl_n,
    }


# ── Model definitions ────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "GPT-2 Large",
        "baseline": "baseline_gpt2_large_K{K}.pkl",
        "realpha": "realpha_gpt2_large.pkl",
    },
    {
        "name": "Llama 3.2 1B",
        "baseline": "baseline_llama_1B_K{K}.pkl",
        "realpha": "realpha_llama_1B.pkl",
    },
    {
        "name": "Llama 3.1 8B",
        "baseline": "baseline_llama_8B_K{K}.pkl",
        "realpha": "realpha_llama_8B.pkl",
    },
    {
        "name": "Phi-4 14B",
        "baseline": "baseline_phi4_K{K}.pkl",
        "realpha": "realpha_phi4.pkl",
    },
]


BASELINE_K_VALUES = [8, 16, 32, 64]


def main():
    parser = argparse.ArgumentParser(description="Baseline vs Realpha JSD analysis")
    parser.add_argument("--results-dir", default="results", help="Directory with pkl files")
    parser.add_argument("--baseline-K", type=int, default=None,
                        help="Baseline K value (default: all available)")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--csv", default=None,
                        help="Write CSV output to this path (default: csv/realpha_vs_baseline_jsd.csv)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    np.random.seed(42)

    k_values = [args.baseline_K] if args.baseline_K else BASELINE_K_VALUES
    all_csv_rows = []

    for baseline_K in k_values:
        for model in MODELS:
            bl_path = results_dir / model["baseline"].format(K=baseline_K)
            rl_path = results_dir / model["realpha"]

            if not bl_path.exists():
                print(f"[SKIP] {model['name']} K={baseline_K}: baseline not found ({bl_path.name})")
                continue
            if not rl_path.exists():
                print(f"[SKIP] {model['name']}: realpha not found ({rl_path.name})")
                continue

            bl_data = load_pkl(str(bl_path))
            rl_data = load_pkl(str(rl_path))

            rows, bl_info = compare_baseline_vs_realpha(bl_data, rl_data, args.n_bootstrap)

            print()
            print(f"{'='*80}")
            print(f"  {model['name']}  (baseline K={baseline_K})")
            print(f"{'='*80}")
            print(f"  Baseline: CE={bl_info['ce_nats']:.4f} nats, "
                  f"bits/byte={bl_info['bits_per_byte']:.4f}, "
                  f"bytes/sec={bl_info['bytes_per_sec']:.1f}, "
                  f"{bl_info['n_paras']} paras, {bl_info['n_symbols']} symbols")
            print()
            print(f"  {'Threshold':<12s} {'JSD':>10s} {'JSD 95% CI':>20s} "
                  f"{'CE (nats)':>10s} {'bits/byte':>10s} {'bytes/sec':>10s} "
                  f"{'paras':>6s} {'positions':>10s}")
            print(f"  {'-'*90}")

            for r in rows:
                print(f"  {r['threshold']:<12g} "
                      f"{r['jsd_mean']:>10.6f} "
                      f"[{r['jsd_ci_lo']:.6f}, {r['jsd_ci_hi']:.6f}] "
                      f"{r['ce_nats']:>10.4f} "
                      f"{r['bits_per_byte']:>10.4f} "
                      f"{r['bytes_per_sec']:>10.1f} "
                      f"{r['n_paras']:>6d} "
                      f"{r['n_positions']:>10d}")

                all_csv_rows.append({
                    "model": model["name"],
                    "baseline_K": baseline_K,
                    "threshold": r["threshold"],
                    "jsd_mean": r["jsd_mean"],
                    "jsd_ci_lower": r["jsd_ci_lo"],
                    "jsd_ci_upper": r["jsd_ci_hi"],
                    "ce_nats": r["ce_nats"],
                    "bits_per_byte": r["bits_per_byte"],
                    "realpha_bytes_per_sec": r["bytes_per_sec"],
                    "baseline_bytes_per_sec": bl_info["bytes_per_sec"],
                    "baseline_bits_per_byte": bl_info["bits_per_byte"],
                    "n_paras": r["n_paras"],
                    "n_positions": r["n_positions"],
                })

            print()

    # Write CSV
    if all_csv_rows:
        import csv as csv_mod

        csv_path = args.csv
        if csv_path is None:
            csv_dir = Path(__file__).resolve().parent.parent / "csv"
            csv_dir.mkdir(exist_ok=True)
            csv_path = csv_dir / "realpha_vs_baseline_jsd.csv"

        fieldnames = list(all_csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"\nWrote {csv_path} ({len(all_csv_rows)} rows)")


if __name__ == "__main__":
    main()
