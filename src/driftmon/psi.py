"""Population Stability Index (PSI).

PSI quantifies how much a distribution has shifted between a reference
("expected", baseline/training) sample and a target ("actual", current/serving)
sample.  For a set of bins it is defined as::

    PSI = sum_i (actual_i% - expected_i%) * ln(actual_i% / expected_i%)

where ``expected_i%`` / ``actual_i%`` are the proportions of the baseline /
current samples falling in bin ``i``.  PSI is symmetric-ish, always >= 0, and 0
only when the two binned distributions are identical.

Zero-frequency handling
------------------------
If a bin is empty in either sample the ratio / log is undefined.  We apply
**epsilon smoothing**: any proportion equal to 0 is replaced by ``epsilon``
(default ``1e-6``) before the ratio is taken.  This is the standard, documented
mitigation and keeps PSI finite while still producing a large per-bin
contribution when a category/bin appears in one sample but not the other.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .binning import bin_counts, clean_numeric, compute_bin_edges

__all__ = [
    "DEFAULT_EPSILON",
    "psi_from_counts",
    "numerical_psi",
    "categorical_psi",
]

DEFAULT_EPSILON = 1e-6


def psi_from_counts(expected_counts, actual_counts, epsilon: float = DEFAULT_EPSILON):
    """Compute PSI from raw per-bin counts.

    Returns ``(psi, per_bin_contributions)`` where ``per_bin_contributions`` is
    the vector summed to produce the scalar PSI (useful for reporting which bin
    drove the drift).
    """
    expected = np.asarray(expected_counts, dtype=float)
    actual = np.asarray(actual_counts, dtype=float)
    if expected.shape != actual.shape:
        raise ValueError("expected and actual count vectors must have the same length")

    exp_total = expected.sum()
    act_total = actual.sum()
    if exp_total <= 0 or act_total <= 0:
        raise ValueError("both samples must contain at least one observation")

    expected_pct = expected / exp_total
    actual_pct = actual / act_total

    # Epsilon smoothing for empty bins (documented above).
    expected_pct = np.where(expected_pct <= 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct <= 0, epsilon, actual_pct)

    contributions = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(contributions)), contributions


def numerical_psi(
    baseline,
    current,
    n_bins: int = 10,
    strategy: str = "quantile",
    epsilon: float = DEFAULT_EPSILON,
) -> Dict:
    """PSI for a numerical feature using baseline-derived bin edges.

    If the baseline has fewer than two distinct finite values it cannot define a
    meaningful continuous binning, so we transparently fall back to a
    categorical (exact-value) comparison and flag it via ``fallback`` in the
    returned details.
    """
    base = clean_numeric(baseline)
    curr = clean_numeric(current)
    if base.size == 0 or curr.size == 0:
        raise ValueError("both baseline and current must contain finite values")

    if np.unique(base).size < 2:
        result = categorical_psi(base.tolist(), curr.tolist(), epsilon=epsilon)
        result["fallback"] = "categorical (constant baseline)"
        result["strategy"] = strategy
        return result

    edges = compute_bin_edges(base, n_bins, strategy)
    expected_counts = bin_counts(base, edges)
    actual_counts = bin_counts(curr, edges)
    psi, contributions = psi_from_counts(expected_counts, actual_counts, epsilon)

    return {
        "psi": psi,
        "n_bins": int(edges.size - 1),
        "strategy": strategy,
        "edges": edges.tolist(),
        "expected_counts": expected_counts.astype(int).tolist(),
        "actual_counts": actual_counts.astype(int).tolist(),
        "contributions": contributions.tolist(),
        "fallback": None,
    }


def categorical_psi(baseline, current, epsilon: float = DEFAULT_EPSILON) -> Dict:
    """PSI for a categorical feature via exact category-frequency comparison.

    The bin set is the *union* of categories seen in either sample, so a new
    category appearing only in the current data (or one that vanished) is scored
    correctly rather than being ignored.
    """
    base = [str(v) for v in _iter_nonnull(baseline)]
    curr = [str(v) for v in _iter_nonnull(current)]
    if not base or not curr:
        raise ValueError("both baseline and current must contain values")

    categories: List[str] = sorted(set(base) | set(curr))
    base_counter = _counts(base, categories)
    curr_counter = _counts(curr, categories)
    psi, contributions = psi_from_counts(base_counter, curr_counter, epsilon)

    return {
        "psi": psi,
        "n_bins": len(categories),
        "strategy": "categorical",
        "categories": categories,
        "expected_counts": [int(c) for c in base_counter],
        "actual_counts": [int(c) for c in curr_counter],
        "contributions": contributions.tolist(),
        "fallback": None,
    }


def _iter_nonnull(values):
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        yield v


def _counts(values: List[str], categories: List[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(categories)}
    out = np.zeros(len(categories), dtype=float)
    for v in values:
        out[index[v]] += 1
    return out
