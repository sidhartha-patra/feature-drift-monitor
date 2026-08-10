"""Command-line interface for the Feature Drift Monitor.

Usage::

    python -m driftmon --baseline baseline.csv --current current.csv \\
        --config config.json [--json] [--bins N] [--strategy quantile|fixed]

CSV files are read with the standard-library :mod:`csv` module (no pandas
dependency).  Columns whose every non-empty value parses as a float are treated
as numerical; everything else is treated as categorical.  Type inference can be
overridden through the JSON config (``categorical_features`` /
``numerical_features``).

Exit codes
----------
``0``  no feature crossed the "significant" PSI threshold.
``1``  at least one feature shows significant drift (CI/pipeline gate).
``2``  usage / input error.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any, Dict, List

from .report import compute_drift_report

__all__ = ["main", "read_csv_columns"]


def read_csv_columns(path: str) -> Dict[str, List[Any]]:
    """Read a CSV into an ordered dict of column-name -> list of values.

    Values that parse as ``float`` are stored as floats; empty cells are stored
    as ``None`` (and later ignored by the statistics). Columns with any
    non-numeric, non-empty cell stay as strings.
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: file is empty")
        raw: List[List[str]] = [list(row) for row in reader]

    columns: Dict[str, List[Any]] = {name: [] for name in header}
    numeric_flags = {name: True for name in header}

    for row in raw:
        for i, name in enumerate(header):
            cell = row[i].strip() if i < len(row) else ""
            columns[name].append(cell)
            if cell != "" and numeric_flags[name]:
                try:
                    float(cell)
                except ValueError:
                    numeric_flags[name] = False

    out: Dict[str, List[Any]] = {}
    for name in header:
        cells = columns[name]
        if numeric_flags[name]:
            out[name] = [float(c) if c != "" else None for c in cells]
        else:
            out[name] = [c if c != "" else None for c in cells]
    return out


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driftmon",
        description="Statistical feature drift monitor (PSI + KS).",
    )
    parser.add_argument("--baseline", required=True, help="Baseline (reference) CSV file.")
    parser.add_argument("--current", required=True, help="Current (serving) CSV file.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--bins", type=int, help="Number of bins for numerical PSI (overrides config).")
    parser.add_argument("--strategy", choices=["quantile", "fixed"], help="Numerical binning strategy.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON instead of text.")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config: Dict[str, Any] = {}
    try:
        if args.config:
            config = _load_config(args.config)
        if args.bins is not None:
            config["n_bins"] = args.bins
        if args.strategy is not None:
            config["strategy"] = args.strategy

        baseline = read_csv_columns(args.baseline)
        current = read_csv_columns(args.current)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = compute_drift_report(baseline, current, config)

    if args.json:
        print(report.to_json())
    else:
        print(report.render_text())

    return 1 if report.has_significant_drift() else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
