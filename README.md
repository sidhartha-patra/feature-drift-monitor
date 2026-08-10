# Feature Drift Monitor

A small, dependency-light (numpy-only) library for **univariate statistical
feature drift / skew monitoring** between a training-time *baseline*
distribution and a production/serving-time *current* distribution. It is the
kind of check an ML platform team runs on a schedule (or in CI) to catch
**silent data drift** before it quietly degrades a deployed model.

It implements the two most widely used drift statistics — **PSI (Population
Stability Index)** and the **two-sample Kolmogorov–Smirnov test** — a
multi-feature report with configurable severity classification, JSON output for
pipeline integration, and a CLI that returns a **non-zero exit code** when
significant drift is detected (so it can act as a CI/pipeline gate).

---

## 1. Problem statement — why silent feature drift matters

A supervised model learns the relationship `P(y | x)` from a fixed training
distribution `P_train(x)`. Once deployed, nothing forces the serving inputs to
keep looking like the training inputs. When `P_serving(x)` drifts away from
`P_train(x)` — a marketing campaign changes the user mix, an upstream ETL job
starts emitting values in a new unit, a sensor is recalibrated, a new category
appears in a categorical field — the model is now extrapolating on inputs it was
never trained for. Accuracy silently decays.

The failure is *silent* because:

- The pipeline does not error. Rows still flow, predictions are still produced.
- Ground-truth labels usually arrive **late** (or never), so you cannot compute
  live accuracy to notice the regression in time.

Monitoring the **input feature distributions** is the earliest available signal.
It requires no labels and can run continuously. This library computes that
signal per feature and rolls it up into a dataset-level verdict.

---

## 2. Statistics implemented

### 2.1 Population Stability Index (PSI)

For a feature discretised into bins `i`, with `E_i%` = proportion of the
*expected* (baseline) sample in bin `i` and `A_i%` = proportion of the *actual*
(current) sample in bin `i`:

```
PSI = Σ_i (A_i% − E_i%) · ln(A_i% / E_i%)
```

- **Numerical features** are binned. The bin *edges are derived from the
  baseline* and then applied to both samples (see §2.3).
- **Categorical features** use exact category-frequency comparison, where the
  bins are the *union* of categories seen in either sample — so a brand-new
  category in the current data is scored rather than ignored.

**Zero-frequency handling.** If a bin/category is empty in one sample the ratio
and log are undefined. We apply **epsilon smoothing**: any proportion equal to
`0` is replaced by `epsilon` (default `1e-6`) before the ratio is computed. This
keeps PSI finite while still producing a large contribution when something
appears in one sample but not the other. The epsilon is configurable.

#### Worked example 1 (unit-tested)

| bin | expected % | actual % | (A−E)·ln(A/E) |
|-----|-----------:|---------:|--------------:|
| 1   | 0.50       | 0.40     | (−0.10)·ln(0.80) = 0.0223144 |
| 2   | 0.30       | 0.40     | (0.10)·ln(1.3333) = 0.0287682 |
| 3   | 0.20       | 0.20     | 0 |
| **PSI** | | | **0.0510826** |

#### Worked example 2 (unit-tested)

| bin | expected % | actual % | (A−E)·ln(A/E) |
|-----|-----------:|---------:|--------------:|
| 1 | 0.25 | 0.10 | (−0.15)·ln(0.4) = 0.1374436 |
| 2 | 0.25 | 0.20 | (−0.05)·ln(0.8) = 0.0111572 |
| 3 | 0.25 | 0.30 | (0.05)·ln(1.2) = 0.0091161 |
| 4 | 0.25 | 0.40 | (0.15)·ln(1.6) = 0.0705005 |
| **PSI** | | | **0.2282174** |

Both values are asserted in `tests/test_psi.py`.

### 2.2 Two-sample Kolmogorov–Smirnov test

The KS statistic is the maximum vertical distance between the two empirical CDFs:

```
D = sup_x |F_baseline(x) − F_current(x)|
```

`D` is computed **exactly**. The p-value uses the asymptotic Kolmogorov
distribution (`kstwobign`):

```
P(D_n > d) ≈ Q_KS(√n_e · d),   n_e = n1·n2 / (n1 + n2)
Q_KS(t)   = 2 · Σ_{k=1..∞} (−1)^(k−1) · exp(−2 k² t²)
```

This matches `scipy.stats.ks_2samp(..., method="asymp")`. The library has **no
scipy dependency** — the KS statistic and p-value are hand-implemented in
`ks.py`. A cross-check test (`tests/test_ks.py::test_ks_crosscheck_scipy`)
verifies `D` to numerical precision and the p-value to `rtol=1e-4` against SciPy
*when SciPy is installed*, and is skipped otherwise.

> KS applies only to numerical features. For small samples SciPy's default
> `method="auto"` computes an *exact* p-value that differs slightly from the
> asymptotic value used here; the asymptotic form is a standard, well-understood
> approximation and is appropriate for the large samples typical of monitoring.

### 2.3 Binning strategy for numerical PSI

- **Quantile binning (default, recommended).** Bin edges are the empirical
  quantiles of the *baseline* sample, so each bin holds ~equal baseline mass.
  This is the standard approach: robust to skew, and it places resolution where
  the data actually is. Duplicate edges from low-cardinality data are collapsed,
  naturally yielding fewer bins than requested when the baseline has fewer
  distinct values.
- **Fixed-width binning (alternative).** Equal-width bins over the baseline
  `[min, max]`. Selectable via config `strategy: "fixed"` or `--strategy fixed`.

The outer edges are extended to `−∞ / +∞` so live values outside the baseline
range fall into the tail bins instead of being dropped.

**Edge cases handled (unit-tested):**

- *Constant baseline* (zero variance): a continuous binning is undefined, so the
  feature transparently falls back to a categorical (exact-value) comparison,
  flagged via `fallback` in the result.
- *All-identical in one sample but not the other*: handled by the same fallback;
  drift is detected when the other sample introduces new values.
- *Fewer distinct values than requested bins*: edges collapse to the available
  distinct values; PSI stays finite and is `0` for identical inputs.

---

## 3. Severity thresholds

PSI is classified into three bands using the widely used industry rule of thumb,
originating in credit-risk scorecard monitoring and adopted by modern ML
monitoring tooling:

| PSI range            | Severity      | Interpretation                          |
|----------------------|---------------|-----------------------------------------|
| `PSI < 0.10`         | `none`        | No significant population change        |
| `0.10 ≤ PSI < 0.25`  | `moderate`    | Moderate shift — investigate            |
| `PSI ≥ 0.25`         | `significant` | Significant shift — action required     |

These are the conventional `0.1` / `0.25` cutoffs cited across the credit-risk
and ML-monitoring literature (e.g. Siddiqi, *Credit Risk Scorecards*; and
tooling such as Evidently / WhyLabs documentation). Thresholds are configurable
via the `thresholds` config key.

---

## 4. Installation

```bash
pip install -e .          # from a clone (numpy only)
# or
pip install -r requirements.txt
```

Requires Python 3.11+. Runtime dependency: **numpy**. `scipy` and `pytest` are
only needed to run the test suite.

---

## 5. Library usage

```python
from driftmon import compute_drift_report

baseline = {
    "age":    [41.0, 39.5, 44.2, ...],
    "region": ["north", "south", "east", ...],
}
current = {
    "age":    [48.1, 47.0, 52.3, ...],
    "region": ["north", "central", "south", ...],   # 'central' is new
}

report = compute_drift_report(baseline, current, config={"n_bins": 10})

report.has_significant_drift()          # -> True
report.get("age").psi                   # -> float
report.get("age").ks_statistic          # -> float
report.summary()                        # -> dict rollup
print(report.render_text())             # human-readable table
report.to_json()                        # JSON string for pipelines
```

Inputs may be plain dicts of lists / numpy arrays, or a pandas `DataFrame`
(pandas is *optional* and detected by duck-typing — it is not a dependency).

---

## 6. CLI usage

```bash
python -m driftmon --baseline baseline.csv --current current.csv [options]
```

| option        | description                                             |
|---------------|---------------------------------------------------------|
| `--baseline`  | baseline (reference) CSV file **(required)**            |
| `--current`   | current (serving) CSV file **(required)**               |
| `--config`    | optional JSON config file                               |
| `--bins N`    | number of bins for numerical PSI                        |
| `--strategy`  | `quantile` (default) or `fixed`                         |
| `--json`      | emit JSON instead of the text table                     |

**Exit codes:** `0` = no significant drift · `1` = significant drift detected
(pipeline gate) · `2` = usage/input error.

CSV columns whose every non-empty value parses as a float are treated as
numerical; all others as categorical. Override via the JSON config
(`categorical_features` / `numerical_features`).

### Example config (`config.json`)

```json
{
  "n_bins": 10,
  "strategy": "quantile",
  "epsilon": 1e-6,
  "thresholds": { "moderate": 0.10, "significant": 0.25 },
  "categorical_features": ["region", "device"]
}
```

### Sample output

Running against the bundled fixtures:

```
$ python -m driftmon --baseline examples/baseline.csv --current examples/current.csv
==============================================================================
Feature Drift Report
==============================================================================
features: 4   none: 2   moderate: 0   significant: 2
max PSI: 3.0034   thresholds: moderate>=0.1, significant>=0.25
------------------------------------------------------------------------------
feature               type               PSI        KS     p-value  severity
------------------------------------------------------------------------------
region                categorical     3.0034                        DRIFT
age                   numerical       0.6165    0.3383   2.971e-30  DRIFT
income                numerical       0.0426    0.0717     0.09176  ok
device                categorical     0.0046                        ok
==============================================================================
verdict: SIGNIFICANT DRIFT DETECTED
==============================================================================
$ echo $?
1
```

`region` drifts hard because the current data introduces a new `central`
category; `age` shows a real mean shift; `income` and `device` are stable.

---

## 7. Limitations & scope

This library is intentionally focused. It deliberately does **not** do:

- **Multivariate / joint-distribution drift.** It measures each feature
  *independently* (univariate). It will not detect drift that only manifests in
  the *correlation* between features (e.g. each marginal looks unchanged but
  their joint relationship shifted). Techniques like domain-classifier drift or
  multivariate distance tests are out of scope.
- **Concept drift** (`P(y | x)` change). It monitors inputs only; it does not
  observe labels or model outputs and cannot detect a changed input→output
  relationship on its own.
- **Automatic remediation.** It reports and gates; it does not trigger retraining
  or roll anything back. Wire the exit code / JSON into your own orchestration.
- **Streaming.** It is a **batch** comparison of two samples, not an online /
  windowed streaming detector.
- **Statistical multiplicity correction.** KS p-values are reported per feature
  with no family-wise / FDR correction across many features — treat PSI severity
  as the primary signal and p-values as supporting context.

Additional practical notes: PSI depends on binning (bin count/strategy affect the
value); very small samples make both PSI and KS noisy; and KS is only defined for
numerical features.

---

## 8. Development

```bash
pip install -r requirements.txt
pip install -e .
pytest -v
```

CI (`.github/workflows/ci.yml`) runs the suite on Python 3.11 and 3.12.

## License

MIT © 2026 Sidhartha Patra. See [LICENSE](LICENSE).
