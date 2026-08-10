"""Binning strategies for numerical PSI.

The Population Stability Index for numerical features requires the continuous
value range to be discretised into bins.  The *reference* (baseline) sample is
always used to derive the bin edges so that the same edges are applied to both
the baseline and the current sample -- this is what makes PSI a stable,
comparable statistic across time.

Two strategies are supported:

``quantile`` (default, recommended)
    Bin edges are the empirical quantiles of the baseline sample so each bin
    holds (approximately) the same number of baseline observations.  This is the
    industry-standard approach because it is robust to skew and puts resolution
    where the data actually lives.

``fixed`` / ``uniform``
    Equal-width bins spanning the baseline ``[min, max]`` range.  Simpler, but
    fragile for skewed distributions (most mass can collapse into one bin).

Regardless of strategy the outer edges are extended to ``-inf`` / ``+inf`` so
that live values falling outside the baseline range are still counted (in the
first / last bin) rather than silently dropped.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "clean_numeric",
    "quantile_bin_edges",
    "fixed_width_bin_edges",
    "compute_bin_edges",
    "bin_counts",
]


def clean_numeric(values) -> np.ndarray:
    """Coerce ``values`` to a 1-D float array with NaN/inf removed."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def quantile_bin_edges(baseline: np.ndarray, n_bins: int) -> np.ndarray:
    """Return bin edges from the quantiles of ``baseline``.

    Duplicate edges (produced when a distribution has repeated values / low
    cardinality) are collapsed with :func:`numpy.unique`, which naturally yields
    *fewer* bins than requested when the baseline has fewer distinct values than
    ``n_bins``.  The extreme edges are pushed out to +/- infinity so that
    out-of-range live values are captured in the tail bins.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    return _extend_outer(edges)


def fixed_width_bin_edges(baseline: np.ndarray, n_bins: int) -> np.ndarray:
    """Return equal-width bin edges spanning the baseline ``[min, max]``."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    lo = float(np.min(baseline))
    hi = float(np.max(baseline))
    if lo == hi:
        # Degenerate (constant) range -- a single interior edge.
        edges = np.array([lo])
    else:
        edges = np.unique(np.linspace(lo, hi, n_bins + 1))
    return _extend_outer(edges)


def _extend_outer(edges: np.ndarray) -> np.ndarray:
    """Replace the outer edges with +/- inf, preserving interior boundaries.

    ``edges`` is assumed sorted and unique.  A baseline with a single unique
    value collapses to one interior edge; we then synthesise a symmetric pair of
    edges around it so callers still receive a valid two-edge (single-bin) array
    without losing the boundary information.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.size >= 2:
        out = edges.copy()
        out[0] = -np.inf
        out[-1] = np.inf
        return out
    # Single interior edge (constant baseline): build [-inf, edge, +inf] so the
    # value itself acts as a boundary and callers get two bins.
    v = float(edges[0])
    return np.array([-np.inf, v, np.inf])


def compute_bin_edges(baseline: np.ndarray, n_bins: int, strategy: str = "quantile") -> np.ndarray:
    """Dispatch to the requested binning ``strategy``."""
    strategy = (strategy or "quantile").lower()
    if strategy == "quantile":
        return quantile_bin_edges(baseline, n_bins)
    if strategy in ("fixed", "uniform", "fixed_width", "equal_width"):
        return fixed_width_bin_edges(baseline, n_bins)
    raise ValueError(f"Unknown binning strategy: {strategy!r}")


def bin_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Count ``values`` into the half-open bins defined by ``edges``.

    Bins are ``[edges[i], edges[i+1])`` with the final bin closed on the right.
    Uses :func:`numpy.searchsorted` (rather than :func:`numpy.histogram`) so that
    infinite outer edges are handled without arithmetic on ``inf`` bin widths.
    """
    n_edges = edges.size
    if n_edges < 2:
        raise ValueError("edges must contain at least two values")
    idx = np.searchsorted(edges, values, side="right") - 1
    # Values equal to the last edge (or +inf) land in the final bin.
    idx = np.clip(idx, 0, n_edges - 2)
    return np.bincount(idx, minlength=n_edges - 1).astype(float)
