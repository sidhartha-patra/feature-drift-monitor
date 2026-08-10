"""CLI tests: CSV parsing, JSON output, and the exit-code drift gate."""

import csv
import json

import pytest

from driftmon.cli import main, read_csv_columns


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_read_csv_columns_type_inference(tmp_path):
    p = tmp_path / "d.csv"
    _write_csv(p, ["num", "cat"], [["1.5", "a"], ["2.5", "b"], ["3.5", "a"]])
    cols = read_csv_columns(str(p))
    assert cols["num"] == [1.5, 2.5, 3.5]  # parsed as floats
    assert cols["cat"] == ["a", "b", "a"]  # stays string


def test_read_csv_handles_missing_cells(tmp_path):
    p = tmp_path / "d.csv"
    _write_csv(p, ["num"], [["1.0"], [""], ["3.0"]])
    cols = read_csv_columns(str(p))
    assert cols["num"] == [1.0, None, 3.0]


def test_cli_exit_zero_when_no_drift(tmp_path, capsys):
    header = ["x"]
    rows = [[str(i)] for i in range(100)]
    base = tmp_path / "base.csv"
    curr = tmp_path / "curr.csv"
    _write_csv(base, header, rows)
    _write_csv(curr, header, rows)  # identical -> no drift

    code = main(["--baseline", str(base), "--current", str(curr)])
    out = capsys.readouterr().out
    assert code == 0
    assert "no significant drift" in out


def test_cli_exit_one_when_significant_drift(tmp_path, capsys):
    header = ["x"]
    base = tmp_path / "base.csv"
    curr = tmp_path / "curr.csv"
    _write_csv(base, header, [[str(i)] for i in range(100)])          # 0..99
    _write_csv(curr, header, [[str(i)] for i in range(200, 300)])     # 200..299 (disjoint)

    code = main(["--baseline", str(base), "--current", str(curr)])
    out = capsys.readouterr().out
    assert code == 1
    assert "SIGNIFICANT DRIFT DETECTED" in out


def test_cli_json_output(tmp_path, capsys):
    header = ["x"]
    base = tmp_path / "base.csv"
    curr = tmp_path / "curr.csv"
    _write_csv(base, header, [[str(i)] for i in range(100)])
    _write_csv(curr, header, [[str(i)] for i in range(100)])

    code = main(["--baseline", str(base), "--current", str(curr), "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)  # must be valid JSON
    assert code == 0
    assert payload["summary"]["n_features"] == 1


def test_cli_bad_path_returns_2(capsys):
    code = main(["--baseline", "does_not_exist.csv", "--current", "nope.csv"])
    err = capsys.readouterr().err
    assert code == 2
    assert "error" in err
