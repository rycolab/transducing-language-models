"""
Pruning functions for beam search. Implements adaptive pruning threshold
and pruning by log weights.

Copied from transducedLM/pruning/pruning.py to make the module self-contained.
"""

import numpy as np


def adaptive_prune_threshold(
    n_beams: int,
    prune_threshold: float = 0.005,  # Threshold from config
    pivot: int = 100,  # Probably best at median num beams
    alpha: float = 0.5,  # Impact of beam size
    max_drop: float = 0.15,  # Maximum drop in probability mass
) -> float:
    if n_beams <= pivot:
        return prune_threshold
    mass = prune_threshold * (n_beams / pivot) ** alpha
    return min(mass, max_drop)


def _prune_by_logweights(
    logw: np.ndarray,
    thld: float,
    cand_thld: float,
    alpha: float,
    max_prune_mass: float,
    max_cand: int | None,
    must_keep: np.ndarray | None = None,
) -> np.ndarray:
    logw = np.asarray(logw, dtype=np.float32, order="C")
    n = logw.size
    if n == 0:
        return np.empty(0, dtype=np.int64)

    # All -inf: keep top max_cand if capped, else all
    if np.isneginf(logw).all():
        if (max_cand is not None) and (n > max_cand):
            return np.argpartition(logw, -max_cand)[-max_cand:].astype(
                np.int64, copy=False
            )
        return np.arange(n, dtype=np.int64)

    # If no pruning, just apply max candidates
    if thld <= 0:
        if (max_cand is not None) and (n > max_cand):
            return np.argpartition(logw, -max_cand)[-max_cand:].astype(
                np.int64, copy=False
            )
        return np.arange(n, dtype=np.int64)

    # Adaptive drop fraction in [0,1]
    thr = adaptive_prune_threshold(
        n,
        prune_threshold=thld,
        pivot=cand_thld,
        alpha=alpha,
        max_drop=max_prune_mass,
    )
    thr = float(min(max(thr, 0.0), 1.0))

    # Stable weights
    m = float(np.max(logw))
    w = np.exp(logw - m, dtype=np.float32)
    total = float(w.sum(dtype=np.float64))

    # Convert "drop smallest mass up to thr" to "keep largest mass at least target_keep"
    target_keep = total - min(thr * total, np.nextafter(total, 0.0))

    # Seed with must_keep
    if must_keep is not None and len(must_keep) > 0:
        mk = np.asarray(must_keep, dtype=np.int64)
        mk = mk[(mk >= 0) & (mk < n)]
        if mk.size:
            target_keep = max(0.0, target_keep - float(w[mk].sum(dtype=np.float64)))
        else:
            mk = None
    else:
        mk = None

    # If target is now <= 0, we only need must_keep
    if not (target_keep > 0.0):
        if mk is not None and mk.size:
            keep = np.unique(mk)
        else:
            keep = np.array([int(np.argmax(logw))], dtype=np.int64)
        # Respect max_cand if present
        if (max_cand is not None) and (keep.size > max_cand):
            kk = np.argpartition(logw[keep], -max_cand)[-max_cand:]
            keep = keep[kk]
        return keep.astype(np.int64, copy=False)

    # collect want minimal K sum of top K(w) >= target_keep
    cap = n if max_cand is None else min(max_cand, n)
    K = min(32, cap) if cap > 0 else 0
    if K == 0:
        if mk is not None and mk.size:
            return np.unique(mk).astype(np.int64, copy=False)
        return np.array([np.argmax(logw)], dtype=np.int64)

    # Grow top-K block until its total mass is enough
    while True:
        idx = np.argpartition(logw, -K)[-K:]
        if w[idx].sum(dtype=np.float64) >= target_keep or K >= cap:
            break
        K = min(K * 2, cap)
    order_desc = np.argsort(logw[idx])[::-1]
    idx_sorted = idx[order_desc]
    cum = np.cumsum(w[idx_sorted], dtype=np.float32)
    k_star = np.searchsorted(cum, target_keep, side="left") + 1
    if k_star > K:
        k_star = K
    keep = idx_sorted[:k_star]
    if mk is not None and mk.size:
        keep = np.unique(np.concatenate([keep, mk])).astype(np.int64, copy=False)
    if (max_cand is not None) and (keep.size > cap):
        sel = np.argpartition(logw[keep], -cap)[-cap:]
        keep = keep[sel]
    if keep.size == 0:
        keep = np.array([np.argmax(logw)], dtype=np.int64)
    return keep.astype(np.int64, copy=False)
