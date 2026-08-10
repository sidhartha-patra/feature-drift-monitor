"""KS tests: hand-computed statistic and optional SciPy cross-check."""

import numpy as np
import pytest

from driftmon.ks import kolmogorov_sf, ks_2samp


def test_ks_statistic_hand_example():
    """a = [1,2,3,4], b = [2,3,4,5].

    ECDF difference is 1/4 at x in {1,2,3,4} and 0 at x=5, so D = 0.25.
    """
    d, p = ks_2samp([1, 2, 3, 4], [2, 3, 4, 5])
    assert d == pytest.approx(0.25, abs=1e-12)
    assert 0.0 <= p <= 1.0


def test_ks_identical_samples():
    d, p = ks_2samp([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert d == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_ks_disjoint_samples():
    d, p = ks_2samp([1, 2, 3, 4, 5], [100, 101, 102, 103, 104])
    assert d == pytest.approx(1.0, abs=1e-12)
    assert p < 0.05


def test_kolmogorov_sf_bounds():
    assert kolmogorov_sf(0.0) == 1.0
    assert kolmogorov_sf(-1.0) == 1.0
    assert kolmogorov_sf(10.0) == pytest.approx(0.0, abs=1e-9)
    # monotonically non-increasing
    xs = np.linspace(0.2, 3.0, 30)
    vals = [kolmogorov_sf(x) for x in xs]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))


def test_ks_crosscheck_scipy():
    """Cross-check D exactly and the asymptotic p-value against SciPy.

    Skipped when SciPy is not installed; must pass when it is.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 1.0, 400)
    b = rng.normal(0.4, 1.2, 500)

    d_mine, p_mine = ks_2samp(a, b)
    res = scipy_stats.ks_2samp(a, b, method="asymp")

    assert d_mine == pytest.approx(res.statistic, abs=1e-12)
    assert p_mine == pytest.approx(res.pvalue, rel=1e-4, abs=1e-6)
