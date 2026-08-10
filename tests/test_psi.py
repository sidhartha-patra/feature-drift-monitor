"""PSI tests, including two hand-computed worked examples."""

import math

import numpy as np
import pytest

from driftmon.psi import categorical_psi, numerical_psi, psi_from_counts


def test_worked_example_1():
    """Hand computed.

    expected% = [0.5, 0.3, 0.2], actual% = [0.4, 0.4, 0.2]

    PSI = (0.4-0.5)ln(0.4/0.5) + (0.4-0.3)ln(0.4/0.3) + (0.2-0.2)ln(1)
        = (-0.1)(-0.22314355) + (0.1)(0.28768207) + 0
        = 0.022314355 + 0.028768207
        = 0.051082562
    """
    psi, contrib = psi_from_counts([50, 30, 20], [40, 40, 20])
    assert psi == pytest.approx(0.051082562, rel=1e-6)
    # third bin is unchanged -> zero contribution
    assert contrib[2] == pytest.approx(0.0, abs=1e-12)


def test_worked_example_2():
    """Hand computed.

    expected% = [0.25]*4, actual% = [0.10, 0.20, 0.30, 0.40]

    PSI = (-0.15)ln(0.4) + (-0.05)ln(0.8) + (0.05)ln(1.2) + (0.15)ln(1.6)
        = 0.137443610 + 0.011157178 + 0.009116078 + 0.070500544
        = 0.228217410
    """
    psi, _ = psi_from_counts([25, 25, 25, 25], [10, 20, 30, 40])
    assert psi == pytest.approx(0.228217410, rel=1e-6)


def test_psi_zero_when_identical():
    counts = [10, 20, 30, 40]
    psi, _ = psi_from_counts(counts, counts)
    assert psi == pytest.approx(0.0, abs=1e-12)


def test_psi_non_negative():
    rng = np.random.default_rng(0)
    for _ in range(20):
        e = rng.integers(1, 100, size=5)
        a = rng.integers(1, 100, size=5)
        psi, _ = psi_from_counts(e, a)
        assert psi >= -1e-12


def test_zero_frequency_smoothing():
    """A bin present in baseline but empty in current must stay finite.

    expected% = [0.5, 0.5], actual% = [1.0, 0.0] -> 0.0 smoothed to epsilon=1e-6.

    Bin1: 0.5 * ln(2)                       = 0.34657359
    Bin2: (1e-6 - 0.5) * ln(2e-6)           = (-0.499999)(-13.12236338) = 6.56116857
    PSI  ~= 6.90774216
    """
    psi, contrib = psi_from_counts([50, 50], [100, 0], epsilon=1e-6)
    assert math.isfinite(psi)
    assert psi == pytest.approx(6.90774216, rel=1e-4)
    # the empty bin dominates
    assert contrib[1] > contrib[0]


def test_categorical_new_category_in_current():
    baseline = ["a"] * 50 + ["b"] * 50
    current = ["a"] * 40 + ["b"] * 40 + ["c"] * 20  # 'c' is brand new
    detail = categorical_psi(baseline, current)
    assert "c" in detail["categories"]
    # baseline had zero 'c' -> smoothed, large contribution -> notable PSI
    assert detail["psi"] > 0.25


def test_categorical_identical_zero():
    data = ["x"] * 30 + ["y"] * 70
    detail = categorical_psi(data, data)
    assert detail["psi"] == pytest.approx(0.0, abs=1e-12)


def test_numerical_psi_identical_is_zero():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    detail = numerical_psi(x, x, n_bins=10)
    assert detail["psi"] == pytest.approx(0.0, abs=1e-12)
    assert detail["n_bins"] == 10
    assert sum(detail["expected_counts"]) == 1000


def test_numerical_psi_detects_shift():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 5000)
    curr = rng.normal(1.5, 1, 5000)  # clear mean shift
    detail = numerical_psi(base, curr, n_bins=10)
    assert detail["psi"] > 0.25
