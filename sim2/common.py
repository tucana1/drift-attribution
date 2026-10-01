"""Shared helpers for the E-series scripts: seed blocks, intervals, expected metrics, JSON output.

Seed convention (task R2). Every row of a sweep draws from its own
``np.random.SeedSequence([root, experiment, row, replicate, ...])`` block, so
bias claims across rows never reuse patients. Estimators that are compared
within a row share that row's draws (paired comparisons).

Interval convention (task E7). Summaries report the across-replicate mean with
a 95% percentile bootstrap interval over replicates (resampling whole
replicates). This interval describes Monte Carlo uncertainty in the simulation
mean. Where an analyst-facing interval is reported instead, the scripts
resample patients within one dataset and say so.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRS = (ROOT / "figures", ROOT / "aaai" / "figures")
BOOT_REPS = 4000


def seed_block(*keys):
    """Independent random stream for one (experiment, row, replicate, ...) cell."""
    return np.random.SeedSequence([int(k) for k in keys])


def rng_for(*keys):
    return np.random.default_rng(seed_block(*keys))


def bootstrap_mean_ci(values, reps=BOOT_REPS, level=0.95, seed=0):
    """Percentile bootstrap interval for the mean of independent replicates."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    means = a[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def summarize(values, target=None, seed=0):
    """Mean, SD, Monte Carlo SE and 95% bootstrap interval over replicates.

    With a target, the bias and its interval are added. Targets are either
    fixed numbers or per-replicate arrays (paired truth).
    """
    a = np.asarray(values, dtype=float)
    out = {"n": int(np.isfinite(a).sum()), "mean": float(np.nanmean(a)),
           "sd": float(np.nanstd(a, ddof=1)) if len(a) > 1 else float("nan")}
    out["mcse"] = out["sd"] / np.sqrt(max(out["n"], 1))
    out["ci95"] = list(bootstrap_mean_ci(a, seed=seed))
    if target is not None:
        err = a - np.asarray(target, dtype=float)
        out["bias"] = float(np.nanmean(err))
        out["bias_ci95"] = list(bootstrap_mean_ci(err, seed=seed + 1))
    return out


def expected_auroc(score, q, weights=None):
    """Population AUROC of ``score`` when patient i has event probability q_i.

    Equals P(score_i > score_j | Y_i = 1, Y_j = 0) for a draw of the outcomes,
    computed without outcome noise (ties count one half).
    """
    score = np.asarray(score, float)
    q = np.asarray(q, float)
    w = np.ones_like(q) if weights is None else np.asarray(weights, float)
    order = np.argsort(score, kind="mergesort")
    s, pos, neg = score[order], (w * q)[order], (w * (1 - q))[order]
    # group ties
    uniq, start = np.unique(s, return_index=True)
    pos_g = np.add.reduceat(pos, start)
    neg_g = np.add.reduceat(neg, start)
    neg_below = np.cumsum(neg_g) - neg_g
    num = float(np.sum(pos_g * (neg_below + 0.5 * neg_g)))
    den = float(pos.sum() * neg.sum())
    return num / den if den > 0 else float("nan")


def expected_brier(score, q):
    """E[(score - Y)^2] when Y ~ Bernoulli(q)."""
    score = np.asarray(score, float)
    return float(np.mean(score ** 2 + (1 - 2 * score) * np.asarray(q, float)))


_NUMBER_LIST = re.compile(r"\[\s*(-?[\d.eE+-]+|true|false|null|NaN|-?Infinity|\"[^\"]*\")"
                          r"(\s*,\s*(-?[\d.eE+-]+|true|false|null|NaN|-?Infinity|\"[^\"]*\"))*\s*\]")


def _compact_lists(text):
    return _NUMBER_LIST.sub(lambda m: "[" + ", ".join(p.strip() for p in m.group(0)[1:-1].split(",")) + "]", text)


def write_json(name, obj, dirs=FIGURE_DIRS):
    """Write ``obj`` to every figure directory with byte-identical copies.

    Objects are indented; lists of scalars stay on one line so long replicate
    arrays remain diffable without inflating the file.
    """
    text = _compact_lists(json.dumps(obj, indent=1, allow_nan=True)) + "\n"
    for folder in dirs:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text)
    return text


def columns(rows, keys):
    """Row dicts to a column-oriented dict (smaller JSON, same information)."""
    return {k: [r[k] for r in rows] for k in keys}


def save_figure(fig, name, dirs=FIGURE_DIRS):
    """Save once without a creation date (reproducible bytes) and copy to every figure directory."""
    first = Path(dirs[0]) / name
    first.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(first, bbox_inches="tight", metadata={"CreationDate": None})
    for folder in dirs[1:]:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / name).write_bytes(first.read_bytes())
