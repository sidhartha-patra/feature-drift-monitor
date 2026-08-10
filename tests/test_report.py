"""Report orchestration, severity classification, and serialisation tests."""

import json

import numpy as np
import pytest

from driftmon.report import (
    SEVERITY_MODERATE,
    SEVERITY_NONE,
    SEVERITY_SIGNIFICANT,
    classify_severity,
    compute_drift_report,
)


def test_classify_severity_thresholds():
    assert classify_severity(0.05) == SEVERITY_NONE
    assert classify_severity(0.15) == SEVERITY_MODERATE
    assert classify_severity(0.30) == SEVERITY_SIGNIFICANT
    # boundary values fall into the higher band (>=)
    assert classify_severity(0.10) == SEVERITY_MODERATE
    assert classify_severity(0.25) == SEVERITY_SIGNIFICANT


def test_classify_severity_custom_thresholds():
    thr = {"moderate": 0.2, "significant": 0.5}
    assert classify_severity(0.15, thr) == SEVERITY_NONE
    assert classify_severity(0.3, thr) == SEVERITY_MODERATE
    assert classify_severity(0.6, thr) == SEVERITY_SIGNIFICANT


def _datasets():
    rng = np.random.default_rng(11)
    baseline = {
        "age": rng.normal(40, 10, 2000),
        "income": rng.normal(60000, 15000, 2000),
        "region": rng.choice(["north", "south", "east", "west"], 2000).tolist(),
    }
    current = {
        "age": rng.normal(48, 11, 2000),   # significant numeric shift
        "income": rng.normal(60200, 15000, 2000),  # ~stable
        "region": rng.choice(["north", "south", "east", "west", "central"], 2000).tolist(),  # new cat
    }
    return baseline, current


def test_report_end_to_end():
    baseline, current = _datasets()
    report = compute_drift_report(baseline, current)

    assert {f.name for f in report.features} == {"age", "income", "region"}

    age = report.get("age")
    assert age.feature_type == "numerical"
    assert age.ks_statistic is not None
    assert age.ks_pvalue is not None
    assert age.severity == SEVERITY_SIGNIFICANT

    income = report.get("income")
    assert income.severity == SEVERITY_NONE

    region = report.get("region")
    assert region.feature_type == "categorical"
    assert region.ks_statistic is None  # no KS for categorical
    assert region.severity in (SEVERITY_MODERATE, SEVERITY_SIGNIFICANT)

    assert report.has_significant_drift()
    assert "age" in [f.name for f in report.significant_features]


def test_report_no_drift():
    rng = np.random.default_rng(3)
    data = {"x": rng.normal(0, 1, 1000), "g": rng.choice(["a", "b"], 1000).tolist()}
    # compare a dataset against itself -> zero drift
    report = compute_drift_report(data, data)
    assert not report.has_significant_drift()
    assert report.max_psi == pytest.approx(0.0, abs=1e-9)
    assert all(f.severity == SEVERITY_NONE for f in report.features)


def test_report_constant_feature_fallback():
    baseline = {"c": [5.0] * 200}
    current = {"c": [5.0] * 100 + [6.0] * 100}
    report = compute_drift_report(baseline, current)
    c = report.get("c")
    assert c.feature_type == "categorical"
    assert c.fallback is not None
    assert c.severity == SEVERITY_SIGNIFICANT


def test_report_forced_categorical():
    # integer-coded categorical should be treatable as categorical on request
    baseline = {"code": [1, 2, 3] * 100}
    current = {"code": [1, 2, 3, 4] * 75}
    report = compute_drift_report(baseline, current, {"categorical_features": ["code"]})
    assert report.get("code").feature_type == "categorical"


def test_report_json_roundtrip():
    baseline, current = _datasets()
    report = compute_drift_report(baseline, current)
    payload = json.loads(report.to_json())
    assert "summary" in payload
    assert "features" in payload
    assert payload["summary"]["has_significant_drift"] is True
    assert isinstance(payload["thresholds"]["significant"], float)


def test_report_text_render():
    baseline, current = _datasets()
    text = compute_drift_report(baseline, current).render_text()
    assert "Feature Drift Report" in text
    assert "age" in text
    assert "verdict" in text


def test_report_skips_missing_feature():
    baseline = {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}
    current = {"a": [1.0, 2.0, 3.0]}  # 'b' missing
    report = compute_drift_report(baseline, current)
    assert report.get("b") is None
    assert "b" in report.skipped
