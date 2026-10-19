"""CLI tests opt into no real network or model API calls."""

import json

import pytest

from conftest import ROOT
from kazakh_model_watch.__main__ import main


def test_development_validate_is_offline_without_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("validate touched a provider")
    monkeypatch.setattr("kazakh_model_watch.runner.OpenRouterAdapter", forbidden)
    assert main(["validate", "--development", "--dataset", str(ROOT / "data/benchmark.jsonl"), "--config", str(ROOT / "config/models.json")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dataset_items"] == 24 and not result["final"]
    assert all(m["readiness"] == "not_ready" for m in result["model_readiness"])


@pytest.mark.parametrize("command", [["validate"], ["run"], ["run", "--live"]])
def test_repository_drafts_block_final_cli_before_network(command, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Draft CLI attempted provider construction")
    monkeypatch.setattr("kazakh_model_watch.runner.OpenRouterAdapter", forbidden)
    argv = command + ["--dataset", str(ROOT / "data/benchmark.jsonl"), "--config", str(ROOT / "config/models.json")]
    if command[0] == "run":
        argv += ["--output-root", str(tmp_path / "results")]
    assert main(argv) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "dataset_invalid"
    assert not (tmp_path / "results").exists()


def test_cli_mock_then_report_without_key_or_dotenv(evaluation_inputs, tmp_path, monkeypatch, capsys):
    import dotenv
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("Offline CLI attempted dotenv loading")
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    dataset, config = evaluation_inputs
    assert main(["run", "--mock", "--dataset", str(dataset), "--config", str(config), "--run-id", "cli", "--output-root", str(tmp_path / "results")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "mock"
    assert main(["report", result["run_dir"]]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "mock" and report["pair_count"] == 12
    assert not (tmp_path / "LEADERBOARD.md").exists()


def test_valid_real_fixture_still_requires_explicit_opt_in(input_factory, tmp_path, capsys):
    dataset, config = input_factory(full=True, real=True)
    assert main(["run", "--dataset", str(dataset), "--config", str(config), "--output-root", str(tmp_path / "results")]) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "live_opt_in_required"
    assert not (tmp_path / "results").exists()


def test_cli_report_missing_artifacts_is_diagnostic_and_safe(tmp_path, capsys):
    assert main(["report", str(tmp_path / "missing")]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result == {"status": "error", "error_code": "invalid_input_or_artifact"}


def test_not_ready_cli_has_explicit_error_before_adapter(input_factory, tmp_path, monkeypatch, capsys):
    dataset, config = input_factory(full=True, real=True, ready=False)
    def forbidden(*args, **kwargs):
        pytest.fail("Not-ready CLI constructed an adapter")
    monkeypatch.setattr("kazakh_model_watch.runner.OpenRouterAdapter", forbidden)
    assert main(["run", "--live", "--dataset", str(dataset), "--config", str(config), "--output-root", str(tmp_path / "results")]) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "model_not_ready"
    assert not (tmp_path / "results").exists()
