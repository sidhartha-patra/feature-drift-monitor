"""Feature Drift Monitor -- univariate statistical drift/skew detection.

Public API::

    from driftmon import (
        compute_drift_report, DriftReport, FeatureDriftResult,
        numerical_psi, categorical_psi, psi_from_counts,
        ks_2samp, classify_severity,
    )

See the module docstrings and README for the statistical definitions,
worked examples, and severity thresholds.
"""

from __future__ import annotations

from .binning import compute_bin_edges, fixed_width_bin_edges, quantile_bin_edges
from .ks import ks_2samp, kolmogorov_sf
from .psi import DEFAULT_EPSILON, categorical_psi, numerical_psi, psi_from_counts
from .report import (
    DEFAULT_THRESHOLDS,
    SEVERITY_MODERATE,
    SEVERITY_NONE,
    SEVERITY_SIGNIFICANT,
    DriftReport,
    FeatureDriftResult,
    classify_severity,
    compute_drift_report,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # psi
    "numerical_psi",
    "categorical_psi",
    "psi_from_counts",
    "DEFAULT_EPSILON",
    # ks
    "ks_2samp",
    "kolmogorov_sf",
    # binning
    "compute_bin_edges",
    "quantile_bin_edges",
    "fixed_width_bin_edges",
    # report
    "compute_drift_report",
    "DriftReport",
    "FeatureDriftResult",
    "classify_severity",
    "DEFAULT_THRESHOLDS",
    "SEVERITY_NONE",
    "SEVERITY_MODERATE",
    "SEVERITY_SIGNIFICANT",
]
