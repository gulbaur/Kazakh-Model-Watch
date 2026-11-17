"""Manually computed aggregates and local artifact integrity, with no network."""

import csv
import json

import pytest

from conftest import ScriptedAdapter
from kazakh_model_watch.config import load_config
from kazakh_model_watch.reporting import RunError, aggregate, atomic_write, read_json, report, write_json
from kazakh_model_watch.runner import run_benchmark


def test_metrics_include_errors_in_denominator_and_zero_latency(evaluation_inputs):
    config = load_config(evaluation_inputs[1])
    rows = []
    for model in config["models"]:
        for index, status in enumerate(["ok", "ok", "timeout", "api_error", "truncated", "not_run"]):
            is_first = model["key"] == "mock_1"
            rows.append({"model_key": model["key"], "category": ["numbers", "dates", "morphology"][index // 2],
                         "status": status if is_first else "not_run", "correct": int(is_first and index == 0),
                         "exact_match": int(is_first and index == 0), "latency_ms": [0, 10][index] if is_first and index < 2 else None,
                         "retries": int(is_first and index == 2)})
    a, b = aggregate(rows, config)
    assert (a["correct"], a["n"], a["accuracy"], a["exact_match_rate"], a["coverage"], a["errors"]) == (1, 6, 1 / 6, 1 / 6, 2 / 6, 4)
    assert a["latency_sample_count"] == 2 and a["latency_mean_ms"] == a["latency_median_ms"] == 5
    assert a["status_counts"] == {"ok": 2, "api_error": 1, "timeout": 1, "empty_response": 0, "truncated": 1, "not_run": 1}
    assert a["retry_count"] == 1 and a["cost"] is None
    assert a["categories"]["numbers"]["accuracy"] == 0.5
    assert a["categories"]["dates"]["coverage"] == 0
    assert b["n"] == 6 and b["latency_sample_count"] == 0 and b["latency_mean_ms"] is None and b["latency_median_ms"] is None


def test_report_is_identical_offline_and_needs_no_key(evaluation_inputs, tmp_path, monkeypatch):
    directory = run_benchmark(*evaluation_inputs, mock=True, output_root=tmp_path / "results")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("Offline report called a provider")
    monkeypatch.setattr("kazakh_model_watch.openrouter_adapter.OpenRouterAdapter.generate", forbidden)
    first = report(directory)
    assert report(directory) == first
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert all(m["n"] == 6 and m["latency_mean_ms"] is None for m in first["models"])


def test_csv_preserves_cyrillic_quotes_newlines_and_nested_values(evaluation_inputs, tmp_path):
    raw = 'Қала, "жауап"\n5'
    adapters = {"mock_1": ScriptedAdapter([("ok", raw, {})]), "mock_2": ScriptedAdapter()}
    directory = run_benchmark(*evaluation_inputs, mock=True, adapters=adapters, output_root=tmp_path / "results")
    rows = read_json(directory / "responses.json")
    with (directory / "responses.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(rows) == 12
    assert csv_rows[0]["raw_response"] == raw and csv_rows[0]["prompt"] == rows[0]["prompt"]
    assert json.loads(csv_rows[0]["attempts"]) == rows[0]["attempts"]
    assert json.loads(csv_rows[0]["usage"])["prompt_tokens"] == 0
    assert csv_rows[0]["latency_ms"] == csv_rows[0]["cost"] == ""
    assert rows[0]["correct"] == rows[0]["exact_match"] == 0
    with (directory / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 8


def test_report_recomputes_scores_instead_of_trusting_saved_values(evaluation_inputs, tmp_path):
    directory = run_benchmark(*evaluation_inputs, mock=True, output_root=tmp_path / "results")
    rows = read_json(directory / "responses.json")
    rows[0].update(correct=999, exact_match=999, normalized_response="invented")
    write_json(directory / "responses.json", rows)
    report(directory)
    row = read_json(directory / "responses.json")[0]
    assert row["correct"] == row["exact_match"] == 0
    assert row["normalized_response"] == "synthetic offline response"


def test_missing_pairs_in_incomplete_report_are_not_run(evaluation_inputs, tmp_path):
    directory = run_benchmark(*evaluation_inputs, mock=True, output_root=tmp_path / "results")
    manifest = read_json(directory / "manifest.json")
    manifest["run_status"] = "incomplete"
    write_json(directory / "manifest.json", manifest)
    rows = read_json(directory / "responses.json")
    write_json(directory / "responses.json", rows[1:])
    result = report(directory)
    first = read_json(directory / "responses.json")[0]
    assert result["pair_count"] == 12 and result["run_status"] == "incomplete"
    assert first["status"] == "not_run" and first["latency_ms"] is None and first["total_elapsed_ms"] is None
    assert first["usage"] is None and first["attempts"] == [] and first["cost"] is None
    assert "incomplete" in (directory / "leaderboard.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown_pair", "identity", "attempt_limit", "invalid_usage", "negative_time", "mock_timing", "early_success"])
def test_report_rejects_inconsistent_saved_data(evaluation_inputs, tmp_path, mutation):
    directory = run_benchmark(*evaluation_inputs, mock=True, output_root=tmp_path / "results")
    rows = read_json(directory / "responses.json")
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "unknown_pair":
        rows[0]["item_id"] = "unknown"
    elif mutation == "identity":
        rows[0]["requested_model_id"] = "different-model"
    elif mutation == "attempt_limit":
        rows[0]["attempts"] *= 3
    elif mutation == "invalid_usage":
        rows[0]["attempts"][0]["usage"] = {"prompt_tokens": True}
    elif mutation == "negative_time":
        rows[0]["attempts"][0]["elapsed_ms"] = -1
    elif mutation == "mock_timing":
        rows[0]["total_elapsed_ms"] = 0
    else:
        rows[0]["attempts"] *= 2
    write_json(directory / "responses.json", rows)
    with pytest.raises(RunError):
        report(directory)


@pytest.mark.parametrize("filename", ["dataset.jsonl", "config.json"])
def test_snapshot_hash_is_checked(evaluation_inputs, tmp_path, filename):
    directory = run_benchmark(*evaluation_inputs, mock=True, output_root=tmp_path / "results")
    path = directory / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RunError, match="input_hash_mismatch"):
        report(directory)


def test_atomic_replace_failure_preserves_previous_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "responses.json"
    path.write_bytes(b'{"previous":true}')
    def fail(*args):
        raise OSError("synthetic replace failure")
    monkeypatch.setattr("kazakh_model_watch.reporting.os.replace", fail)
    with pytest.raises(OSError):
        atomic_write(path, b'{"next":true}')
    assert path.read_bytes() == b'{"previous":true}'
    assert list(tmp_path.iterdir()) == [path]


def test_tied_accuracy_keeps_equal_rank_and_stable_model_order(evaluation_inputs, tmp_path):
    directory = run_benchmark(*evaluation_inputs, mock=True, output_root=tmp_path / "results")
    text = (directory / "leaderboard.md").read_text(encoding="utf-8")
    assert text.count("| 1 | mock/") == 2
    assert text.index("mock/mock-model-1") < text.index("mock/mock-model-2")
    assert "N/A" in text and "OFFLINE SYNTHETIC" in text


def test_report_rejects_a_retry_for_a_nonretryable_error(evaluation_inputs, tmp_path):
    adapters = {"mock_1": ScriptedAdapter([("timeout", None, {})]), "mock_2": ScriptedAdapter()}
    directory = run_benchmark(*evaluation_inputs, mock=True, output_root=tmp_path / "results", sleep=lambda _: None, adapters=adapters)
    rows = read_json(directory / "responses.json")
    first = rows[0]["attempts"][0]
    first.update(status="api_error", error_code="http_error", http_status=401)
    write_json(directory / "responses.json", rows)
    with pytest.raises(RunError, match="invalid_retry_chain"):
        report(directory)


def test_legacy_attempts_without_new_diagnostics_are_reported_as_unknown(evaluation_inputs, tmp_path):
    directory = run_benchmark(*evaluation_inputs, mock=True, output_root=tmp_path / "results")
    rows = read_json(directory / "responses.json")
    for row in rows:
        for attempt in row["attempts"]:
            del attempt["upstream_provider"], attempt["reasoning_evidence"]
    write_json(directory / "responses.json", rows)
    manifest = read_json(directory / "manifest.json")
    for model in manifest["models"]:
        del model["upstream_providers"]
    write_json(directory / "manifest.json", manifest)
    report(directory)
    rows = read_json(directory / "responses.json")
    assert all(a["upstream_provider"] is None and a["reasoning_evidence"] is None for row in rows for a in row["attempts"])
    assert "Upstream providers: unknown" in (directory / "leaderboard.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("evidence", [
    {"reasoning": "synthetic private thought"},
    "synthetic private thought",
    {"reasoning": {"present": True, "type": "string", "nonempty": True},
     "reasoning_content": {"present": False, "type": "null", "nonempty": False},
     "reasoning_details": {"present": False, "type": "null", "nonempty": False},
     "detail_types": ["synthetic private type"], "detail_text_nonempty": False, "detail_data_nonempty": False},
])
def test_report_rejects_saved_evidence_outside_fixed_labels(evaluation_inputs, tmp_path, evidence):
    directory = run_benchmark(*evaluation_inputs, mock=True, output_root=tmp_path / "results")
    rows = read_json(directory / "responses.json")
    rows[0]["attempts"][0]["reasoning_evidence"] = evidence
    write_json(directory / "responses.json", rows)
    with pytest.raises(RunError, match="invalid_reasoning_evidence"):
        report(directory)


def test_report_drops_unsafe_saved_upstream_provider(evaluation_inputs, tmp_path):
    directory = run_benchmark(*evaluation_inputs, mock=True, output_root=tmp_path / "results")
    rows = read_json(directory / "responses.json")
    rows[0]["attempts"][0]["upstream_provider"] = "synthetic\nmultiline"
    write_json(directory / "responses.json", rows)
    report(directory)
    assert read_json(directory / "responses.json")[0]["attempts"][0]["upstream_provider"] is None
