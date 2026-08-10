"""Binning tests: quantile edge correctness and edge cases."""

import numpy as np
import pytest

from driftmon.binning import (
    bin_counts,
    compute_bin_edges,
    fixed_width_bin_edges,
    quantile_bin_edges,
)
from driftmon.psi import numerical_psi


def test_quantile_edges_from_baseline():
    """Edges are the baseline deciles, outer edges pushed to +/- inf.

    For 0..99, quantile at q = 0.1*k sits at index 0.1*k*99, i.e. 9.9, 19.8, ...
    """
    base = np.arange(100)
    edges = quantile_bin_edges(base, n_bins=10)
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    expected_interior = [9.9, 19.8, 29.7, 39.6, 49.5, 59.4, 69.3, 79.2, 89.1]
    np.testing.assert_allclose(edges[1:-1], expected_interior, rtol=1e-9)


def test_quantile_bins_equal_population():
    """Baseline binned by its own quantiles -> ~equal counts per bin."""
    base = np.arange(100)
    edges = compute_bin_edges(base, 10, "quantile")
    counts = bin_counts(base, edges)
    np.testing.assert_array_equal(counts, np.full(10, 10.0))


def test_bin_counts_capture_out_of_range():
    base = np.arange(10)
    edges = compute_bin_edges(base, 5, "quantile")
    # values well outside the baseline range must fall into the tail bins
    counts = bin_counts(np.array([-1000.0, 1000.0]), edges)
    assert counts[0] == 1
    assert counts[-1] == 1
    assert counts.sum() == 2


def test_fewer_distinct_values_than_bins():
    """Discrete low-cardinality data yields fewer bins, no crash, PSI finite."""
    base = np.array([0, 1, 2] * 100)  # only 3 distinct values, ask for 10 bins
    edges = compute_bin_edges(base, 10, "quantile")
    assert edges.size - 1 < 10  # collapsed to fewer bins
    detail_same = numerical_psi(base, base, n_bins=10)
    assert detail_same["psi"] == pytest.approx(0.0, abs=1e-12)
    detail_shift = numerical_psi(base, np.array([2] * 300), n_bins=10)
    assert detail_shift["psi"] > 0.0


def test_constant_baseline_falls_back_to_categorical():
    base = np.full(100, 5.0)
    # current has a value never seen in baseline -> should register drift
    curr = np.array([5.0] * 50 + [6.0] * 50)
    detail = numerical_psi(base, curr, n_bins=10)
    assert detail["fallback"] is not None
    assert detail["psi"] > 0.0
    # constant vs identical constant -> no drift
    detail_same = numerical_psi(base, base, n_bins=10)
    assert detail_same["psi"] == pytest.approx(0.0, abs=1e-12)


def test_fixed_width_edges():
    base = np.arange(101)  # 0..100
    edges = fixed_width_bin_edges(base, 10)
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    # interior edges are equally spaced multiples of 10 (10..90)
    np.testing.assert_allclose(edges[1:-1], [10, 20, 30, 40, 50, 60, 70, 80, 90])


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        compute_bin_edges(np.arange(10), 5, "nonsense")
