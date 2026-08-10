"""Two-sample Kolmogorov-Smirnov test.

The KS statistic ``D`` is the maximum absolute difference between the empirical
cumulative distribution functions (ECDFs) of two samples::

    D = sup_x |F_baseline(x) - F_current(x)|

It is a distribution-free measure of how different two continuous samples are.

This module provides a small, dependency-light implementation (numpy only).
The statistic ``D`` is computed exactly.  The p-value uses the asymptotic
Kolmogorov distribution (a.k.a. ``kstwobign``)::

    P(D_n > d) ~= Q_KS(sqrt(n_e) * d),  n_e = n1*n2 / (n1 + n2)

    Q_KS(t) = 2 * sum_{k=1..inf} (-1)^{k-1} * exp(-2 k^2 t^2)

This matches :func:`scipy.stats.ks_2samp(..., method="asymp")`.  A test in the
suite cross-checks against SciPy when it is installed (skipped otherwise); the
statistic ``D`` also matches SciPy's exact statistic to numerical precision.

For small samples SciPy's default ``method="auto"`` uses an *exact* p-value that
differs slightly from the asymptotic value here -- the asymptotic form is a
well-understood, standard approximation and is more than adequate for drift
monitoring, where sample sizes are typically large.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .binning import clean_numeric

__all__ = ["ks_2samp", "kolmogorov_sf"]


def kolmogorov_sf(t: float, terms: int = 100) -> float:
    """Survival function of the Kolmogorov distribution, ``Q_KS(t)``.

    Returns a probability in ``[0, 1]``.  The alternating series converges very
    quickly for the ``t`` values seen in practice (``t`` roughly 1-4).
    """
    if t <= 0:
        return 1.0
    total = 0.0
    for k in range(1, terms + 1):
        total += ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * t * t)
    p = 2.0 * total
    # Clamp to guard against tiny negative/over-one values from truncation.
    return float(min(1.0, max(0.0, p)))


def ks_2samp(baseline, current) -> Tuple[float, float]:
    """Return ``(D, p_value)`` for the two-sample KS test.

    Parameters
    ----------
    baseline, current:
        1-D sequences of numeric values.  NaN/inf are dropped.
    """
    a = np.sort(clean_numeric(baseline))
    b = np.sort(clean_numeric(current))
    n1 = a.size
    n2 = b.size
    if n1 == 0 or n2 == 0:
        raise ValueError("both samples must contain at least one finite value")

    # Evaluate both ECDFs on the pooled set of observations.
    pooled = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, pooled, side="right") / n1
    cdf_b = np.searchsorted(b, pooled, side="right") / n2
    d = float(np.max(np.abs(cdf_a - cdf_b)))

    en = math.sqrt(n1 * n2 / (n1 + n2))
    p = kolmogorov_sf(en * d)
    return d, p
