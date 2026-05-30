#!/usr/bin/env python3
"""
Post-process benchmark pickle files into CSV tables.

Reads pickle files from results/ and produces:
  - csv/realpha_jsd.csv        (Experiment 1: realpha JSD + throughput)
  - csv/ptb_jsd.csv            (Experiment 2: PTB JSD + throughput)
  - csv/dna2aa_jsd.csv         (Experiment 3: DNA JSD + throughput)
  - csv/realpha_ce.csv         (Experiment 4: realpha cross-entropy + throughput)
  - csv/baseline_jsd.csv       (Experiment 7: Vieira baseline comparison)

Each CSV has columns:
  K, mean_metric, metric_ci_lower, metric_ci_upper,
  chars_per_sec, speed_ci_lower, speed_ci_upper, model

Adapted from:
  - benchmarking/jsd.py
  - benchmarking/utils/scoring_utils.py
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core statistical functions (ported from scoring_utils.py)
# ---------------------------------------------------------------------------


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in bits.  Skips zero entries in p."""
    nz = p.nonzero()
    p = p[nz]
    q = q[nz]
    if len(p) == 0:
        return 0.0
    return float(p.dot(np.log(p) - np.log(q)) / np.log(2))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Per-position JSD in bits: 0.5 * (KL(p||m) + KL(q||m))."""
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def compute_bootstrap_stats(
    metrics: np.ndarray | None,
    speeds: np.ndarray,
    text_length: int,
    n_bootstrap: int = 1000,
) -> dict:
    """Bootstrap 95% CI for metric mean and throughput."""
    if metrics is not None and len(metrics) > 0:
        mean_metric = float(np.mean(metrics))
        boot = [
            np.mean(np.random.choice(metrics, size=len(metrics), replace=True))
            for _ in range(n_bootstrap)
        ]
        metric_ci_lower = float(np.percentile(boot, 2.5))
        metric_ci_upper = float(np.percentile(boot, 97.5))
    else:
        mean_metric = np.nan
        metric_ci_lower = np.nan
        metric_ci_upper = np.nan

    if len(speeds) > 0:
        mean_speed = text_length / np.sum(speeds)
        speed_boot = [
            text_length / np.sum(np.random.choice(speeds, size=len(speeds), replace=True))
            for _ in range(n_bootstrap)
        ]
        speed_ci_lower = float(np.percentile(speed_boot, 2.5))
        speed_ci_upper = float(np.percentile(speed_boot, 97.5))
    else:
        mean_speed = np.nan
        speed_ci_lower = np.nan
        speed_ci_upper = np.nan

    return {
        "mean_metric": mean_metric,
        "metric_ci_lower": metric_ci_lower,
        "metric_ci_upper": metric_ci_upper,
        "chars_per_sec": float(mean_speed),
        "speed_ci_lower": speed_ci_lower,
        "speed_ci_upper": speed_ci_upper,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pickle(path: str) -> dict:
    """Load a benchmark pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _identify_paragraph_indices(stored_lens: list[int], ref_lens: list[int]) -> list[int]:
    """Greedy forward match of stored paragraph lengths to reference lengths.

    run.py iterates paragraphs in order and skips failures, so stored paragraphs
    are a subsequence of the reference paragraph list.  This function recovers
    which reference paragraph each stored entry corresponds to.

    Returns a list of reference paragraph indices (one per stored entry).
    Unmatched entries get index -1.
    """
    indices = []
    ref_idx = 0
    for stored_len in stored_lens:
        matched = False
        while ref_idx < len(ref_lens):
            if stored_len == ref_lens[ref_idx]:
                indices.append(ref_idx)
                ref_idx += 1
                matched = True
                break
            ref_idx += 1
        if not matched:
            indices.append(-1)

    n_unmatched = indices.count(-1)
    if n_unmatched > 0:
        warnings.warn(
            f"Paragraph alignment: {n_unmatched}/{len(stored_lens)} stored "
            f"paragraphs could not be matched to reference paragraphs "
            f"(stored_lens={stored_lens[:5]}{'...' if len(stored_lens) > 5 else ''}, "
            f"ref has {len(ref_lens)} paragraphs). "
            f"These paragraphs will be DROPPED from analysis."
        )

    return indices


def convert_logprobs_to_probs_by_paragraph(
    p_nexts: dict, metadata: dict, stats_raw: dict | None = None
) -> tuple[dict, list[int]]:
    """Convert log-prob arrays to probability dicts, keeping paragraph structure.

    Args:
        p_nexts: {K: [list of paragraph arrays]}
        metadata: benchmark metadata dict
        stats_raw: raw stats dict; if present, uses para_indices when available

    Returns:
        (result, ref_lens) where:
        result[K] = {para_idx: [list of prob dicts]}  (keyed by reference paragraph index)
        ref_lens = canonical paragraph lengths from the most complete threshold
    """
    # Find the most complete threshold to use as reference for paragraph lengths
    best_K = max(p_nexts.keys(), key=lambda k: sum(len(p) for p in p_nexts[k]))
    ref_lens = [len(p_nexts[best_K][i]) for i in range(len(p_nexts[best_K]))]

    # Check paragraph count consistency across thresholds
    para_counts = {K: len(p_nexts[K]) for K in p_nexts}
    unique_counts = set(para_counts.values())
    if len(unique_counts) > 1:
        warnings.warn(
            f"Paragraph count differs across thresholds: "
            + ", ".join(f"K={K}: {n} paragraphs" for K, n in sorted(para_counts.items()))
            + f". Reference (K={best_K}) has {len(ref_lens)} paragraphs."
        )

    result = {}
    for K in p_nexts:
        # Use explicit para_indices from stats when available (new runs)
        explicit = None
        if stats_raw is not None:
            explicit = stats_raw.get(K, {}).get("para_indices", None)

        if explicit is not None and len(explicit) == len(p_nexts[K]):
            para_indices = explicit
        else:
            stored_lens = [len(p_nexts[K][i]) for i in range(len(p_nexts[K]))]
            para_indices = _identify_paragraph_indices(stored_lens, ref_lens)

        result[K] = {}
        for j, para_idx in enumerate(para_indices):
            if para_idx < 0:
                continue
            para_data = p_nexts[K][j]
            if len(para_data) == 0:
                result[K][para_idx] = []
                continue
            # Data can be a list of dicts (log-prob dicts) or a 2D array
            if isinstance(para_data[0], dict):
                # Already dict format — convert log-probs to probs
                result[K][para_idx] = [
                    {k: np.exp(v) for k, v in d.items()} for d in para_data
                ]
            else:
                arr = np.asarray(para_data, dtype=np.float32)
                np.exp(arr, out=arr)
                result[K][para_idx] = [
                    {i: v for i, v in enumerate(row)} for row in arr
                ]

    return result, ref_lens


def _split_stats_by_paragraph(
    stats_K: dict, p_nexts_K: list, para_indices: list[int]
) -> dict[int, dict]:
    """Split flat stats (times, log_probs) into per-paragraph dicts.

    Returns {para_idx: {"times": [...], "log_probs": [...]}}
    keyed by reference paragraph index.
    """
    lens = [len(p_nexts_K[i]) for i in range(len(p_nexts_K))]
    times = stats_K.get("times", [])
    logps = stats_K.get("log_probs", [])

    total_positions = sum(lens)
    if len(times) > 0 and len(times) != total_positions:
        warnings.warn(
            f"Stats/distribution length mismatch: "
            f"{len(times)} timing entries but {total_positions} positions "
            f"across {len(lens)} paragraphs. "
            f"Some paragraphs may have missing timing data."
        )

    result = {}
    n_skipped = 0
    offset = 0
    for j, length in enumerate(lens):
        para_idx = para_indices[j] if j < len(para_indices) else -1
        if para_idx >= 0 and offset + length <= len(times):
            result[para_idx] = {
                "times": times[offset:offset + length],
                "log_probs": logps[offset:offset + length],
            }
        elif para_idx >= 0:
            n_skipped += 1
        offset += length

    if n_skipped > 0:
        warnings.warn(
            f"Stats split: {n_skipped} matched paragraphs skipped because "
            f"stats ran out (offset {offset} > stats length {len(times)})."
        )

    return result


# ---------------------------------------------------------------------------
# JSD computation
# ---------------------------------------------------------------------------


def compute_jsd_table(
    p_nexts_by_para: dict,
    Ks: list,
    stats_by_para: dict,
    metadata: dict,
    ref_K: float | None = None,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Compute JSD for each threshold vs reference.

    Uses paragraph-level alignment: only compares byte positions from
    paragraphs that are present in BOTH the test and reference thresholds.

    Args:
        p_nexts_by_para: {K: {para_idx: [prob_dicts]}} from convert_logprobs_to_probs_by_paragraph
        stats_by_para: {K: {para_idx: {"times": [...], "log_probs": [...]}}}
    """
    if ref_K is None:
        ref_K = min(Ks)

    text_length = metadata.get("text_length", 0)
    results = []

    ref_paras = p_nexts_by_para.get(ref_K, {})

    for K in Ks:
        if K <= ref_K:
            continue

        test_paras = p_nexts_by_para.get(K, {})
        # Only compare paragraphs present in both thresholds
        common_para_ids = sorted(set(test_paras.keys()) & set(ref_paras.keys()))

        n_test_only = len(set(test_paras.keys()) - set(ref_paras.keys()))
        n_ref_only = len(set(ref_paras.keys()) - set(test_paras.keys()))
        if n_test_only > 0 or n_ref_only > 0:
            warnings.warn(
                f"K={K}: paragraph mismatch vs reference (K={ref_K}): "
                f"{len(common_para_ids)} common, "
                f"{n_test_only} in test only, {n_ref_only} in ref only. "
                f"JSD computed over {len(common_para_ids)} common paragraphs."
            )

        jds = []
        speeds = []
        n_truncated_positions = 0
        n_nonfinite_jsd = 0
        for para_idx in common_para_ids:
            P_list = test_paras[para_idx]
            Q_list = ref_paras[para_idx]
            if len(P_list) != len(Q_list):
                n_truncated_positions += abs(len(P_list) - len(Q_list))
            # Times for this paragraph at this threshold
            para_times = stats_by_para.get(K, {}).get(para_idx, {}).get("times", [])

            for pos, (P, Q) in enumerate(zip(P_list, Q_list)):
                xs = set(P.keys()) | set(Q.keys())
                p = np.array([P.get(x, 0.0) for x in xs])
                q = np.array([Q.get(x, 0.0) for x in xs])

                jd = jensen_shannon_divergence(p, q)
                if np.isfinite(jd):
                    if jd < 0:
                        jd = 0.0
                    jds.append(jd)
                    if pos < len(para_times):
                        speeds.append(para_times[pos])
                else:
                    n_nonfinite_jsd += 1

        if n_truncated_positions > 0:
            warnings.warn(
                f"K={K}: {n_truncated_positions} positions lost due to "
                f"paragraph length mismatch between test and reference "
                f"(zip truncation). This may indicate incomplete scoring."
            )
        if n_nonfinite_jsd > 0:
            warnings.warn(
                f"K={K}: {n_nonfinite_jsd} non-finite JSD values dropped "
                f"(out of {len(jds) + n_nonfinite_jsd} total positions)."
            )

        jds = np.array(jds)
        speeds = np.array(speeds)
        # Fall back to all times if per-paragraph matching failed
        if len(speeds) == 0:
            all_times = []
            for para_stats in stats_by_para.get(K, {}).values():
                all_times.extend(para_stats.get("times", []))
            speeds = np.array(all_times)

        n_common = sum(
            min(len(test_paras.get(pi, [])), len(ref_paras.get(pi, [])))
            for pi in common_para_ids
        )

        results.append({
            "K": K,
            **compute_bootstrap_stats(
                metrics=jds,
                speeds=speeds,
                text_length=n_common if n_common > 0 else text_length,
                n_bootstrap=n_bootstrap,
            ),
        })

    # Add reference row (no JSD metric, just speed)
    ref_all_times = []
    for para_stats in stats_by_para.get(ref_K, {}).values():
        ref_all_times.extend(para_stats.get("times", []))
    ref_speeds = np.array(ref_all_times)
    ref_text_len = sum(len(probs) for probs in ref_paras.values())
    results.append({
        "K": ref_K,
        **compute_bootstrap_stats(
            metrics=None,
            speeds=ref_speeds,
            text_length=ref_text_len if ref_text_len > 0 else text_length,
            n_bootstrap=n_bootstrap,
        ),
    })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Cross-entropy computation
# ---------------------------------------------------------------------------


def compute_ce_table(
    stats_by_para: dict,
    Ks: list,
    metadata: dict,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Compute bits/byte (cross-entropy) for each threshold.

    Args:
        stats_by_para: {K: {para_idx: {"times": [...], "log_probs": [...]}}}
    """
    text_length = metadata.get("text_length", 0)
    results = []

    for K in Ks:
        if K not in stats_by_para:
            continue

        # Collect all log_probs and times across paragraphs
        all_logps = []
        all_times = []
        for para_stats in stats_by_para[K].values():
            all_logps.extend(para_stats.get("log_probs", []))
            all_times.extend(para_stats.get("times", []))

        logps = np.array(all_logps, dtype=np.float64)
        times = np.array(all_times, dtype=np.float64)

        # Bits per byte: -sum(log_probs) / (n * ln(2))
        finite_mask = np.isfinite(logps)
        n_nonfinite = int(np.sum(~finite_mask))
        logps_clean = logps[finite_mask]
        n = len(logps_clean)
        if n_nonfinite > 0:
            warnings.warn(
                f"CE K={K}: {n_nonfinite} non-finite log-prob values dropped "
                f"(out of {len(logps)} total). "
                f"{n} positions remain for bits/byte computation."
            )
        if n == 0:
            warnings.warn(
                f"CE K={K}: ALL {len(logps)} log-probs are non-finite. "
                f"Skipping this threshold entirely."
            )
            continue

        bpb = float(-np.sum(logps_clean) / (n * np.log(2)))

        # Bootstrap CI for bits/byte
        boot_bpb = [
            float(-np.sum(np.random.choice(logps_clean, size=n, replace=True)) / (n * np.log(2)))
            for _ in range(n_bootstrap)
        ]

        speed_stats = compute_bootstrap_stats(
            metrics=None,
            speeds=times,
            text_length=n,
            n_bootstrap=n_bootstrap,
        )
        results.append({
            "K": K,
            "mean_metric": bpb,
            "metric_ci_lower": float(np.percentile(boot_bpb, 2.5)),
            "metric_ci_upper": float(np.percentile(boot_bpb, 97.5)),
            "chars_per_sec": speed_stats["chars_per_sec"],
            "speed_ci_lower": speed_stats["speed_ci_lower"],
            "speed_ci_upper": speed_stats["speed_ci_upper"],
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Pipeline: process one pickle
# ---------------------------------------------------------------------------


def process_pickle(
    pkl_path: str,
    model_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Process a single pickle file.

    Returns:
        (jsd_df, ce_df, metadata)
        jsd_df: JSD results DataFrame
        ce_df: Cross-entropy results DataFrame
    """
    data = load_pickle(pkl_path)
    metadata = data.get("metadata", {})
    stats_raw = data.get("stats", {})
    p_nexts_raw = data.get("p_nexts", {})
    # Ks: union of thresholds from p_nexts and stats (CE-only runs have
    # empty p_nexts but valid stats)
    Ks = sorted(set(p_nexts_raw.keys()) | set(stats_raw.keys()))

    if not Ks:
        print(f"  WARNING: No thresholds found in {pkl_path}")
        return pd.DataFrame(), pd.DataFrame(), metadata

    # Detect CE-only pickles: p_nexts exists but all entries are empty lists
    is_ce_only = all(len(p_nexts_raw.get(K, [])) == 0 for K in Ks)

    if is_ce_only:
        # CE-only: no distributions stored, only flat stats with para_indices.
        # We don't need paragraph-level splitting for CE — just pass the
        # entire flat array as a single entry so compute_ce_table can use it.
        stats_by_para = {}
        for K in Ks:
            s = stats_raw.get(K, {})
            times = s.get("times", [])
            logps = s.get("log_probs", [])
            if not times:
                stats_by_para[K] = {}
                continue
            # Store as a single pseudo-paragraph (index 0) containing all data
            stats_by_para[K] = {
                0: {"times": times, "log_probs": logps}
            }

        # Summary for CE-only
        for K in sorted(Ks):
            n_paras = len(stats_raw.get(K, {}).get("para_indices", []))
            n_positions = len(stats_raw.get(K, {}).get("times", []))
            print(f"    K={K}: {n_paras} paragraphs, {n_positions} positions (CE-only)")

        # No JSD for CE-only
        jsd_df = pd.DataFrame()
        ce_df = compute_ce_table(stats_by_para, Ks, metadata)

    else:
        # Normal mode: has distributions in p_nexts
        # Convert logprobs to probabilities (paragraph-aligned)
        p_nexts_by_para, ref_lens = convert_logprobs_to_probs_by_paragraph(
            p_nexts_raw, metadata, stats_raw=stats_raw
        )

        # Build paragraph-aligned stats.
        # Use explicit para_indices from stats when available (new runs),
        # fall back to length-based matching (old runs).
        stats_by_para = {}
        for K in Ks:
            explicit_indices = stats_raw.get(K, {}).get("para_indices", None)
            if explicit_indices is not None and len(explicit_indices) == len(p_nexts_raw.get(K, [])):
                para_indices = explicit_indices
            else:
                stored_lens = [len(p_nexts_raw[K][i]) for i in range(len(p_nexts_raw.get(K, [])))]
                para_indices = _identify_paragraph_indices(stored_lens, ref_lens)
            stats_by_para[K] = _split_stats_by_paragraph(
                stats_raw.get(K, {}), p_nexts_raw.get(K, []), para_indices
            )

        # Print alignment summary and warn on mismatches
        expected_paras = metadata.get("n_paragraphs", metadata.get("paragraphs", None))
        for K in sorted(Ks):
            n_stored = len(p_nexts_raw.get(K, []))
            n_matched = len(p_nexts_by_para.get(K, {}))
            n_positions = sum(len(v) for v in p_nexts_by_para.get(K, {}).values())
            if n_stored != n_matched:
                warnings.warn(
                    f"{pkl_path} K={K}: {n_stored} stored paragraphs but only "
                    f"{n_matched} matched to reference ({n_positions} positions). "
                    f"{n_stored - n_matched} paragraphs DROPPED."
                )
            if expected_paras is not None and n_stored < expected_paras:
                warnings.warn(
                    f"{pkl_path} K={K}: only {n_stored}/{expected_paras} expected "
                    f"paragraphs present. Experiment may be incomplete."
                )

        # Check that all thresholds have the same paragraph count
        para_counts = {K: len(p_nexts_raw.get(K, [])) for K in Ks}
        counts_set = set(para_counts.values())
        if len(counts_set) > 1:
            max_count = max(para_counts.values())
            incomplete = {K: n for K, n in para_counts.items() if n < max_count}
            warnings.warn(
                f"{pkl_path}: inconsistent paragraph counts across thresholds. "
                f"Max is {max_count}. Incomplete: "
                + ", ".join(f"K={K}: {n}" for K, n in sorted(incomplete.items()))
                + ". Results may not be directly comparable."
            )

        # JSD
        jsd_df = compute_jsd_table(p_nexts_by_para, Ks, stats_by_para, metadata)

        # Cross-entropy
        ce_df = compute_ce_table(stats_by_para, Ks, metadata)

    # Add model label
    label = model_label or metadata.get("model_name", "unknown")
    jsd_df["model"] = label
    ce_df["model"] = label

    return jsd_df, ce_df, metadata


# ---------------------------------------------------------------------------
# Main: process all experiments
# ---------------------------------------------------------------------------


# Map from pickle filename pattern to (experiment_name, model_label)
EXPERIMENTS = {
    # Experiment 1: realpha (tokens -> characters)
    "realpha_gpt2_large.pkl": ("realpha", "GPT-2 Large"),
    "realpha_llama_1B.pkl": ("realpha", "Llama 3.2 1B"),
    "realpha_llama_8B.pkl": ("realpha", "Llama 3.1 8B"),
    "realpha_phi4.pkl": ("realpha", "Phi-4 14B"),
    # Experiment 2: PTB (tokens -> words)
    "ptb_gpt2_large.pkl": ("ptb", "GPT-2 Large"),
    "ptb_llama_1B.pkl": ("ptb", "Llama 3.2 1B"),
    "ptb_llama_8B.pkl": ("ptb", "Llama 3.1 8B"),
    "ptb_phi4.pkl": ("ptb", "Phi-4 14B"),
    # Experiment 3: DNA (6 max_candidates settings)
    "dna2aa_maxcand_5000.pkl": ("dna2aa", "gpt2-dna (max_cand=5000)"),
    "dna2aa_maxcand_10000.pkl": ("dna2aa", "gpt2-dna (max_cand=10000)"),
    "dna2aa_maxcand_15000.pkl": ("dna2aa", "gpt2-dna (max_cand=15000)"),
    "dna2aa_maxcand_20000.pkl": ("dna2aa", "gpt2-dna (max_cand=20000)"),
    "dna2aa_maxcand_25000.pkl": ("dna2aa", "gpt2-dna (max_cand=25000)"),
    "dna2aa_maxcand_30000.pkl": ("dna2aa", "gpt2-dna (max_cand=30000)"),
    # Experiment 7: Vieira baseline (multiple K values)
    "baseline_gpt2_large_K8.pkl": ("baseline", "GPT-2 Large (K=8)"),
    "baseline_gpt2_large_K16.pkl": ("baseline", "GPT-2 Large (K=16)"),
    "baseline_gpt2_large_K32.pkl": ("baseline", "GPT-2 Large (K=32)"),
    "baseline_gpt2_large_K64.pkl": ("baseline", "GPT-2 Large (K=64)"),
    "baseline_llama_1B_K8.pkl": ("baseline", "Llama 3.2 1B (K=8)"),
    "baseline_llama_1B_K16.pkl": ("baseline", "Llama 3.2 1B (K=16)"),
    "baseline_llama_1B_K32.pkl": ("baseline", "Llama 3.2 1B (K=32)"),
    "baseline_llama_1B_K64.pkl": ("baseline", "Llama 3.2 1B (K=64)"),
    "baseline_llama_8B_K8.pkl": ("baseline", "Llama 3.1 8B (K=8)"),
    "baseline_llama_8B_K16.pkl": ("baseline", "Llama 3.1 8B (K=16)"),
    "baseline_llama_8B_K32.pkl": ("baseline", "Llama 3.1 8B (K=32)"),
    "baseline_llama_8B_K64.pkl": ("baseline", "Llama 3.1 8B (K=64)"),
    "baseline_phi4_K8.pkl": ("baseline", "Phi-4 14B (K=8)"),
    "baseline_phi4_K16.pkl": ("baseline", "Phi-4 14B (K=16)"),
    "baseline_phi4_K32.pkl": ("baseline", "Phi-4 14B (K=32)"),
    "baseline_phi4_K64.pkl": ("baseline", "Phi-4 14B (K=64)"),
}


def main():
    # Show all warnings (don't suppress after first occurrence)
    warnings.simplefilter("always")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    results_dir = repo_root / "results"
    csv_dir = script_dir / "csv"
    csv_dir.mkdir(exist_ok=True)

    # Collect results by experiment
    jsd_by_exp: dict[str, list[pd.DataFrame]] = defaultdict(list)
    ce_by_exp: dict[str, list[pd.DataFrame]] = defaultdict(list)

    # Process known experiment files
    for filename, (exp_name, model_label) in EXPERIMENTS.items():
        pkl_path = results_dir / filename
        if not pkl_path.exists():
            continue

        print(f"Processing {filename} -> {exp_name} / {model_label}")
        jsd_df, ce_df, metadata = process_pickle(str(pkl_path), model_label)

        if not jsd_df.empty:
            jsd_by_exp[exp_name].append(jsd_df)
        if not ce_df.empty:
            ce_by_exp[exp_name].append(ce_df)

    # Also process any unknown .pkl files
    if results_dir.exists():
        for pkl_file in sorted(results_dir.glob("*.pkl")):
            if pkl_file.name in EXPERIMENTS:
                continue
            print(f"Processing (unknown) {pkl_file.name}")
            jsd_df, ce_df, metadata = process_pickle(str(pkl_file))
            exp_name = metadata.get("transducer_name", "unknown")
            if not jsd_df.empty:
                jsd_by_exp[exp_name].append(jsd_df)
            if not ce_df.empty:
                ce_by_exp[exp_name].append(ce_df)

    # Write CSVs
    for exp_name, dfs in jsd_by_exp.items():
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(["model", "K"])
        out_path = csv_dir / f"{exp_name}_jsd.csv"
        combined.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(combined)} rows)")

    for exp_name, dfs in ce_by_exp.items():
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(["model", "K"])
        out_path = csv_dir / f"{exp_name}_ce.csv"
        combined.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(combined)} rows)")

    if not jsd_by_exp and not ce_by_exp:
        print("No pickle files found in results/. Run experiments first.")
        sys.exit(1)

    print("\nDone. CSVs written to csv/")


if __name__ == "__main__":
    main()
