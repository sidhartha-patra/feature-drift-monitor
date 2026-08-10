"""Multi-feature drift report: orchestration, severity, and rendering.

Given a *baseline* and a *current* dataset (each a mapping of feature-name ->
sequence of values; a pandas ``DataFrame`` is also accepted if pandas is
installed) this module computes per-feature PSI (and KS for numerical features),
classifies each feature by severity, and produces a queryable result object that
renders to text or JSON.

Severity thresholds (PSI)
-------------------------
The following thresholds are the widely used industry rule-of-thumb, originating
in credit-risk scorecard monitoring and adopted by modern ML monitoring tools:

===================  ==========================================================
PSI range            Interpretation
===================  ==========================================================
PSI < 0.10           No significant population change ("none")
0.10 <= PSI < 0.25   Moderate shift -- investigate ("moderate")
PSI >= 0.25          Significant shift -- action required ("significant")
===================  ==========================================================

They are configurable via the report config (``thresholds``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .ks import ks_2samp
from .psi import DEFAULT_EPSILON, categorical_psi, numerical_psi

__all__ = [
    "SEVERITY_NONE",
    "SEVERITY_MODERATE",
    "SEVERITY_SIGNIFICANT",
    "DEFAULT_THRESHOLDS",
    "FeatureDriftResult",
    "DriftReport",
    "classify_severity",
    "compute_drift_report",
]

SEVERITY_NONE = "none"
SEVERITY_MODERATE = "moderate"
SEVERITY_SIGNIFICANT = "significant"

# Lower bounds for each severity band (see module docstring for citation).
DEFAULT_THRESHOLDS = {"moderate": 0.10, "significant": 0.25}


def classify_severity(psi: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """Map a PSI value to a severity label using ``thresholds``."""
    thr = thresholds or DEFAULT_THRESHOLDS
    if psi >= thr["significant"]:
        return SEVERITY_SIGNIFICANT
    if psi >= thr["moderate"]:
        return SEVERITY_MODERATE
    return SEVERITY_NONE


@dataclass
class FeatureDriftResult:
    """Drift outcome for a single feature."""

    name: str
    feature_type: str  # "numerical" | "categorical"
    psi: float
    severity: str
    n_baseline: int
    n_current: int
    n_bins: Optional[int] = None
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    fallback: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    """Dataset-level drift report."""

    features: List[FeatureDriftResult]
    thresholds: Dict[str, float]
    skipped: Dict[str, str] = field(default_factory=dict)

    # -- queries -----------------------------------------------------------
    def get(self, name: str) -> Optional[FeatureDriftResult]:
        for f in self.features:
            if f.name == name:
                return f
        return None

    @property
    def significant_features(self) -> List[FeatureDriftResult]:
        return [f for f in self.features if f.severity == SEVERITY_SIGNIFICANT]

    @property
    def moderate_features(self) -> List[FeatureDriftResult]:
        return [f for f in self.features if f.severity == SEVERITY_MODERATE]

    @property
    def drifted_features(self) -> List[FeatureDriftResult]:
        return [f for f in self.features if f.severity != SEVERITY_NONE]

    def has_significant_drift(self) -> bool:
        return len(self.significant_features) > 0

    @property
    def max_psi(self) -> float:
        return max((f.psi for f in self.features), default=0.0)

    def summary(self) -> Dict[str, Any]:
        counts = {SEVERITY_NONE: 0, SEVERITY_MODERATE: 0, SEVERITY_SIGNIFICANT: 0}
        for f in self.features:
            counts[f.severity] += 1
        return {
            "n_features": len(self.features),
            "severity_counts": counts,
            "max_psi": self.max_psi,
            "has_significant_drift": self.has_significant_drift(),
            "significant_features": [f.name for f in self.significant_features],
        }

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "thresholds": self.thresholds,
            "features": [f.to_dict() for f in self.features],
            "skipped": self.skipped,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def render_text(self) -> str:
        return _render_text(self)


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

def _as_column_mapping(data) -> Dict[str, Any]:
    """Normalise a DataFrame / mapping into an ordinary dict of columns."""
    if data is None:
        raise ValueError("dataset is None")
    # Duck-type a pandas DataFrame without importing pandas.
    if hasattr(data, "columns") and hasattr(data, "to_dict"):
        return {str(col): np.asarray(data[col].values) for col in data.columns}
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}
    raise TypeError("dataset must be a dict of columns or a pandas DataFrame")


def _is_numeric_column(values) -> bool:
    arr = np.asarray(list(values), dtype=object)
    for v in arr:
        if v is None:
            continue
        if isinstance(v, bool):
            return False
        if isinstance(v, (int, float, np.integer, np.floating)):
            continue
        return False
    return True


def compute_drift_report(baseline, current, config: Optional[Dict[str, Any]] = None) -> DriftReport:
    """Compute a :class:`DriftReport` comparing ``current`` against ``baseline``.

    Config keys (all optional):

    ``n_bins`` (int, default 10)
        Number of bins for numerical PSI.
    ``strategy`` ("quantile" | "fixed", default "quantile")
        Numerical binning strategy.
    ``epsilon`` (float, default 1e-6)
        Zero-frequency smoothing constant.
    ``thresholds`` (dict, default industry standard)
        ``{"moderate": 0.10, "significant": 0.25}``.
    ``categorical_features`` / ``numerical_features`` (list[str])
        Force the type of specific features (otherwise inferred).
    ``features`` (list[str])
        Restrict the report to this subset (default: all shared features).
    """
    config = dict(config or {})
    n_bins = int(config.get("n_bins", 10))
    strategy = config.get("strategy", "quantile")
    epsilon = float(config.get("epsilon", DEFAULT_EPSILON))
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(config.get("thresholds", {}))
    forced_categorical = set(config.get("categorical_features", []) or [])
    forced_numerical = set(config.get("numerical_features", []) or [])

    base_cols = _as_column_mapping(baseline)
    curr_cols = _as_column_mapping(current)

    requested = config.get("features")
    if requested:
        feature_names = [f for f in requested]
    else:
        # Preserve baseline column order; a baseline feature missing from the
        # current dataset is a schema-drift signal and is recorded as skipped
        # (below) rather than silently dropped.
        feature_names = list(base_cols.keys())

    features: List[FeatureDriftResult] = []
    skipped: Dict[str, str] = {}

    for name in feature_names:
        if name not in base_cols or name not in curr_cols:
            skipped[name] = "feature missing from baseline or current dataset"
            continue

        base_vals = list(base_cols[name])
        curr_vals = list(curr_cols[name])

        if name in forced_numerical:
            is_numeric = True
        elif name in forced_categorical:
            is_numeric = False
        else:
            is_numeric = _is_numeric_column(base_vals) and _is_numeric_column(curr_vals)

        try:
            if is_numeric:
                result = _numeric_feature(name, base_vals, curr_vals, n_bins, strategy, epsilon, thresholds)
            else:
                result = _categorical_feature(name, base_vals, curr_vals, epsilon, thresholds)
        except ValueError as exc:
            skipped[name] = str(exc)
            continue
        features.append(result)

    return DriftReport(features=features, thresholds=thresholds, skipped=skipped)


def _numeric_feature(name, base_vals, curr_vals, n_bins, strategy, epsilon, thresholds) -> FeatureDriftResult:
    psi_detail = numerical_psi(base_vals, curr_vals, n_bins=n_bins, strategy=strategy, epsilon=epsilon)
    fallback = psi_detail.get("fallback")
    # KS only makes sense on a genuine continuous comparison.
    ks_stat = ks_p = None
    if fallback is None:
        ks_stat, ks_p = ks_2samp(base_vals, curr_vals)
    feature_type = "categorical" if fallback else "numerical"
    return FeatureDriftResult(
        name=name,
        feature_type=feature_type,
        psi=psi_detail["psi"],
        severity=classify_severity(psi_detail["psi"], thresholds),
        n_baseline=_count_finite(base_vals),
        n_current=_count_finite(curr_vals),
        n_bins=psi_detail.get("n_bins"),
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        fallback=fallback,
        details=psi_detail,
    )


def _categorical_feature(name, base_vals, curr_vals, epsilon, thresholds) -> FeatureDriftResult:
    psi_detail = categorical_psi(base_vals, curr_vals, epsilon=epsilon)
    return FeatureDriftResult(
        name=name,
        feature_type="categorical",
        psi=psi_detail["psi"],
        severity=classify_severity(psi_detail["psi"], thresholds),
        n_baseline=len(base_vals),
        n_current=len(curr_vals),
        n_bins=psi_detail.get("n_bins"),
        ks_statistic=None,
        ks_pvalue=None,
        fallback=None,
        details=psi_detail,
    )


def _count_finite(values) -> int:
    arr = np.asarray(values, dtype=float)
    return int(np.count_nonzero(np.isfinite(arr)))


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

_SEVERITY_MARK = {
    SEVERITY_NONE: "ok  ",
    SEVERITY_MODERATE: "WARN",
    SEVERITY_SIGNIFICANT: "DRIFT",
}


def _render_text(report: DriftReport) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("Feature Drift Report")
    lines.append("=" * 78)
    summary = report.summary()
    counts = summary["severity_counts"]
    lines.append(
        f"features: {summary['n_features']}   "
        f"none: {counts[SEVERITY_NONE]}   "
        f"moderate: {counts[SEVERITY_MODERATE]}   "
        f"significant: {counts[SEVERITY_SIGNIFICANT]}"
    )
    lines.append(
        f"max PSI: {summary['max_psi']:.4f}   "
        f"thresholds: moderate>={report.thresholds['moderate']:g}, "
        f"significant>={report.thresholds['significant']:g}"
    )
    lines.append("-" * 78)
    header = f"{'feature':<22}{'type':<12}{'PSI':>10}{'KS':>10}{'p-value':>12}  severity"
    lines.append(header)
    lines.append("-" * 78)
    for f in sorted(report.features, key=lambda x: x.psi, reverse=True):
        ks = "" if f.ks_statistic is None else f"{f.ks_statistic:.4f}"
        pv = "" if f.ks_pvalue is None else f"{f.ks_pvalue:.4g}"
        mark = _SEVERITY_MARK.get(f.severity, f.severity)
        lines.append(
            f"{f.name:<22}{f.feature_type:<12}{f.psi:>10.4f}{ks:>10}{pv:>12}  {mark}"
        )
    if report.skipped:
        lines.append("-" * 78)
        lines.append("skipped features:")
        for name, reason in report.skipped.items():
            lines.append(f"  - {name}: {reason}")
    lines.append("=" * 78)
    verdict = "SIGNIFICANT DRIFT DETECTED" if report.has_significant_drift() else "no significant drift"
    lines.append(f"verdict: {verdict}")
    lines.append("=" * 78)
    return "\n".join(lines)
