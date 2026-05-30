#!/usr/bin/env python3
"""Analyze benchmark logs: speed, errors, retries, and projected run time.

Usage:
    python analyze_speed.py [results_dir]

Reads ptb_pg*_th*.pkl.log files, extracts throughput/wall-time/symbols,
counts errors and retries, and projects how long the full sweep would take.
"""
import os
import re
import sys
from collections import defaultdict
import numpy as np

results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

# ── Parse logs ────────────────────────────────────────────────────────
# filename pattern: ptb_pg{para}_th{threshold}.pkl.log
fname_re = re.compile(r"ptb_pg(\d+)_th([\d.eE+-]+)\.pkl\.log")

# From summary block:
sym_re = re.compile(r"Symbols scored\s*:\s*(\d+)")
wall_re = re.compile(r"Wall time\s*:\s*([\d.]+)s")
tp_re = re.compile(r"Throughput\s*:\s*([\d.]+)\s*sym/s")

# Error/retry patterns
fallback_re = re.compile(r"\[fallback\]")
skip_re = re.compile(r"Skipping paragraph")
unrecov_re = re.compile(r"Unrecoverable -inf")
rebuild_re = re.compile(r"-inf, rebuilding covers")
oom_re = re.compile(r"CUDA out of memory|MemoryError|Killed|MemoryMax")
probe_miss_re = re.compile(r"\[probe miss\]")

data = []  # (para, threshold, symbols, wall_time, throughput, errors_dict)

for fname in sorted(os.listdir(results_dir)):
    m = fname_re.match(fname)
    if not m:
        continue
    para = int(m.group(1))
    threshold = float(m.group(2))

    path = os.path.join(results_dir, fname)
    text = open(path).read()

    sm = sym_re.search(text)
    wm = wall_re.search(text)
    tm = tp_re.search(text)

    # Count errors/retries
    n_fallback = len(fallback_re.findall(text))
    n_skip = len(skip_re.findall(text))
    n_unrecov = len(unrecov_re.findall(text))
    n_rebuild = len(rebuild_re.findall(text))
    n_oom = len(oom_re.findall(text))
    n_probe_miss = len(probe_miss_re.findall(text))

    # Check if process was killed (no summary block)
    was_killed = sm is None and ("Killed" in text or "MemoryMax" in text
                                  or "CUDA out of memory" in text)

    errors = {
        "fallbacks": n_fallback,
        "skipped": n_skip,
        "unrecoverable": n_unrecov,
        "rebuilds": n_rebuild,
        "oom": n_oom,
        "probe_miss": n_probe_miss,
        "killed": was_killed,
    }

    if sm and wm and tm:
        data.append((para, threshold, int(sm.group(1)),
                      float(wm.group(1)), float(tm.group(1)), errors))
    else:
        # Incomplete run — record with zeros for speed
        data.append((para, threshold, 0, 0.0, 0.0, errors))

if not data:
    print("No data found. Check results directory.")
    sys.exit(1)

# ── Group by threshold ────────────────────────────────────────────────
by_th = defaultdict(list)
for para, th, syms, wall, tp, errors in data:
    by_th[th].append((para, syms, wall, tp, errors))

thresholds = sorted(by_th.keys(), reverse=True)

# ── Per-paragraph symbol counts ───────────────────────────────────────
para_syms = {}
for para, th, syms, wall, tp, errors in data:
    if syms > 0:
        para_syms[para] = syms

total_syms = sum(para_syms.values())

print("=" * 72)
print("  PTB Benchmark Speed Analysis")
print("=" * 72)
print()
print(f"  Paragraphs with data: {sorted(para_syms.keys())}")
print(f"  Symbols per paragraph: " +
      ", ".join(f"p{k}={v}" for k, v in sorted(para_syms.items())))
print(f"  Total symbols (10 paras): {total_syms}")
print()

# ── Table 1: Speed per threshold ──────────────────────────────────────
print("  SPEED")
print("┌─────────────┬────────┬──────────────┬──────────────┬──────────────┐")
print("│  Threshold  │ N runs │  Mean sym/s  │  Mean wall/s │  Sym/s range │")
print("├─────────────┼────────┼──────────────┼──────────────┼──────────────┤")

th_mean_speed = {}
th_mean_wall = {}
th_mean_syms = {}

for th in thresholds:
    entries = by_th[th]
    completed = [(p, s, w, t, e) for p, s, w, t, e in entries if s > 0]
    if not completed:
        print(f"│ {th:>11.6f} │ {len(entries):>6d} │    (all failed)                       │")
        continue
    speeds = [t for _, _, _, t, _ in completed]
    walls = [w for _, _, w, _, _ in completed]
    syms_list = [s for _, s, _, _, _ in completed]
    mean_sp = np.mean(speeds)
    mean_wall = np.mean(walls)
    mean_syms = np.mean(syms_list)
    min_sp = np.min(speeds)
    max_sp = np.max(speeds)
    th_mean_speed[th] = mean_sp
    th_mean_wall[th] = mean_wall
    th_mean_syms[th] = mean_syms
    print(f"│ {th:>11.6f} │ {len(completed):>6d} │ {mean_sp:>10.1f}   │ {mean_wall:>10.1f}   │ {min_sp:.0f}-{max_sp:.0f}     │")

print("└─────────────┴────────┴──────────────┴──────────────┴──────────────┘")
print()

# ── Table 2: Errors/retries per threshold ─────────────────────────────
print("  ERRORS & RETRIES")
print("┌─────────────┬────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
print("│  Threshold  │ N runs │ Fallback │ Rebuilds │  Skipped │   Unrec. │   OOM    │")
print("├─────────────┼────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

for th in thresholds:
    entries = by_th[th]
    n = len(entries)
    tot_fb = sum(e["fallbacks"] for _, _, _, _, e in entries)
    tot_rb = sum(e["rebuilds"] for _, _, _, _, e in entries)
    tot_sk = sum(e["skipped"] for _, _, _, _, e in entries)
    tot_ur = sum(e["unrecoverable"] for _, _, _, _, e in entries)
    tot_oom = sum(e["oom"] for _, _, _, _, e in entries)
    tot_killed = sum(1 for _, _, _, _, e in entries if e["killed"])

    fb_str = f"{tot_fb}" if tot_fb else "-"
    rb_str = f"{tot_rb}" if tot_rb else "-"
    sk_str = f"{tot_sk}" if tot_sk else "-"
    ur_str = f"{tot_ur}" if tot_ur else "-"
    oom_str = f"{tot_oom}" if tot_oom else "-"
    if tot_killed:
        oom_str = f"{tot_oom}+{tot_killed}K"

    print(f"│ {th:>11.6f} │ {n:>6d} │ {fb_str:>8s} │ {rb_str:>8s} │ {sk_str:>8s} │ {ur_str:>8s} │ {oom_str:>8s} │")

print("└─────────────┴────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
print()

# ── Per-run detail for runs with errors ───────────────────────────────
error_runs = [(p, th, s, e) for p, th, s, _, _, e in data
              if e["fallbacks"] or e["skipped"] or e["unrecoverable"]
              or e["oom"] or e["killed"] or e["rebuilds"]]
if error_runs:
    print("  RUNS WITH ISSUES")
    print(f"  {'File':<35s} {'Syms':>5s}  Issues")
    print(f"  {'─'*35} {'─'*5}  {'─'*30}")
    for p, th, s, e in error_runs:
        fname = f"ptb_pg{p}_th{th}"
        issues = []
        if e["fallbacks"]:
            issues.append(f"{e['fallbacks']} fallback")
        if e["rebuilds"]:
            issues.append(f"{e['rebuilds']} rebuild")
        if e["skipped"]:
            issues.append(f"{e['skipped']} skipped")
        if e["unrecoverable"]:
            issues.append(f"{e['unrecoverable']} unrecov")
        if e["oom"]:
            issues.append(f"{e['oom']} OOM")
        if e["killed"]:
            issues.append("KILLED")
        print(f"  {fname:<35s} {s:>5d}  {', '.join(issues)}")
    print()

# ── Time projection ──────────────────────────────────────────────────
all_thresholds = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001,
                  0.00003, 0.00001, 0.000003, 0.000001]
n_paras = 10

if len(th_mean_speed) >= 2:
    log_th = np.log10(np.array(list(th_mean_speed.keys())))
    log_sp = np.log10(np.array(list(th_mean_speed.values())))
    coeffs = np.polyfit(log_th, log_sp, 1)
    slope, intercept = coeffs

    print("  TIME PROJECTION")
    print(f"  Speed model: log10(sym/s) = {slope:.3f} * log10(threshold) + {intercept:.3f}")
    print(f"  (speed {'increases' if slope > 0 else 'decreases'} ~{abs(slope):.2f}x per 10x threshold change)")
    print()

    print("┌─────────────┬──────────────┬──────────────┬──────────────┬──────────┐")
    print("│  Threshold  │ Pred. sym/s  │ Time/para(s) │ 10 paras (m) │  Status  │")
    print("├─────────────┼──────────────┼──────────────┼──────────────┼──────────┤")

    total_minutes = 0
    for th in all_thresholds:
        if th in th_mean_speed:
            pred_speed = th_mean_speed[th]
            status = "done"
            avg_syms = th_mean_syms[th]
        else:
            pred_speed = 10 ** (slope * np.log10(th) + intercept)
            status = "projected"
            avg_syms = np.mean(list(para_syms.values()))

        time_per_para = avg_syms / pred_speed
        time_10_paras = time_per_para * n_paras / 60
        total_minutes += time_10_paras

        print(f"│ {th:>11.7f} │ {pred_speed:>10.1f}   │ {time_per_para:>10.1f}   │ {time_10_paras:>10.1f}   │ {status:>8s} │")

    print("├─────────────┼──────────────┼──────────────┼──────────────┼──────────┤")
    print(f"│ {'TOTAL':>11s} │              │              │ {total_minutes:>10.1f}   │          │")
    print("└─────────────┴──────────────┴──────────────┴──────────────┴──────────┘")
    print()

    hours = total_minutes / 60
    done_minutes = sum(
        th_mean_syms[th] / th_mean_speed[th] * n_paras / 60
        for th in all_thresholds if th in th_mean_speed
    )
    remaining = total_minutes - done_minutes
    print(f"  Projected total wall time: {total_minutes:.0f} min ({hours:.1f} hours)")
    print(f"  Already completed: {done_minutes:.0f} min")
    print(f"  Remaining:         {remaining:.0f} min ({remaining/60:.1f} hours)")
else:
    print("  Need at least 2 thresholds to fit speed model.")
