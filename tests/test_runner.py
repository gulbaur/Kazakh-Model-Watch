"""Synthetic orchestration and MockTransport only; socket access is disabled."""

import itertools
import json
import random
from copy import deepcopy

import httpx
import pytest

from conftest import ROOT, ScriptedAdapter
from kazakh_model_watch.config import ConfigValidationError, load_config
from kazakh_model_watch.dataset import DatasetValidationError, load_dataset
from kazakh_model_watch.model_adapter import ModelResponse, MockAdapter
from kazakh_model_watch.openrouter_adapter import OpenRouterAdapter
from kazakh_model_watch.reporting import RunError, digest, read_json, report
from kazakh_model_watch.runner import run_benchmark


def test_sequence_seed_rotation_identical_prompts_and_artifacts(evaluation_inputs, tmp_path):
    dataset, config_path = evaluation_inputs
    config = load_config(config_path)
    calls = []
    adapters = {m["key"]: ScriptedAdapter(calls=calls) for m in config["models"]}
    directory = run_benchmark(dataset, config_path, output_root=tmp_path / "results", run_id="synthetic-order", mock=True, adapters=adapters)
    items = load_dataset(dataset)
    random.Random(42).shuffle(items)
    assert len(calls) == 12
    for index, item in enumerate(items):
        pair = calls[index * 2:index * 2 + 2]
        expected_ids = [m["model_id"] for m in config["models"]]
        if index % 2:
            expected_ids.reverse()
        assert [r.model_id for r in pair] == expected_ids
        assert pair[0].prompt == pair[1].prompt == config["common_instruction"] + "\n\n" + item["prompt"]
        assert all(r.temperature == 0 and r.max_output_tokens == 128 and r.timeout_s == 60 for r in pair)
        assert all("reference_answer" not in r.prompt and "accepted_answers" not in r.prompt and "review" not in r.prompt for r in pair)
    names = {p.name for p in directory.iterdir()}
    assert names == {"manifest.json", "dataset.jsonl", "config.json", "responses.json", "responses.csv", "metrics.json", "metrics.csv", "leaderboard.md"}
    manifest = read_json(directory / "manifest.json")
    assert manifest["run_status"] == "complete" and manifest["mode"] == "mock"
    assert manifest["input_hashes"] == {"dataset": digest(dataset.read_bytes()), "config": digest(config_path.read_bytes()), "instruction": digest(config["common_instruction"].encode("utf-8"))}
    assert (directory / "dataset.jsonl").read_bytes() == dataset.read_bytes()
    assert (directory / "config.json").read_bytes() == config_path.read_bytes()
    assert manifest["unicode_version"] and manifest["dependencies"]["httpx"] and manifest["git_commit"]
    rows = read_json(directory / "responses.json")
    assert all(r["latency_ms"] is None and r["total_elapsed_ms"] is None and r["cost"] is None for r in rows)
    assert all(r["usage"]["prompt_tokens"] == 0 for r in rows)
    assert "OFFLINE SYNTHETIC" in (directory / "leaderboard.md").read_text(encoding="utf-8")
    assert not (tmp_path / "LEADERBOARD.md").exists()


@pytest.mark.parametrize(("status", "diagnostic", "retry"), [
    ("timeout", {}, True), ("api_error", {"error_code": "network_error"}, True),
    ("api_error", {"http_status": 429}, True), ("api_error", {"http_status": 500}, True),
    ("api_error", {"http_status": 503}, True), ("api_error", {"http_status": 401}, False),
    ("api_error", {"http_status": 402}, False), ("api_error", {"http_status": 403}, False),
    ("api_error", {"http_status": 200, "error_code": "provider_error"}, False),
    ("api_error", {}, False),
    ("api_error", {"http_status": 500, "error_code": "sensitive_response"}, False),
    ("api_error", {"http_status": 429, "finish_reason": "content_filter", "error_code": "content_filtered"}, False),
    ("empty_response", {}, False), ("truncated", {"finish_reason": "length"}, False),
    ("ok", {}, False),
])
def test_only_eligible_failures_retry(evaluation_inputs, tmp_path, status, diagnostic, retry):
    raw = "not the correct answer" if status == "ok" else None
    calls, waits = [], []
    adapters = {"mock_1": ScriptedAdapter([(status, raw, diagnostic)], calls), "mock_2": ScriptedAdapter()}
    directory = run_benchmark(*evaluation_inputs, mock=True, adapters=adapters, sleep=waits.append, output_root=tmp_path / "results")
    assert len(calls) == 6 + int(retry)
    assert waits == ([2.0] if retry else [])
    first = next(row for row in read_json(directory / "responses.json") if row["model_key"] == "mock_1")
    assert len(first["attempts"]) == 1 + int(retry)
    assert first["retries"] == int(retry)


@pytest.mark.parametrize(("header_delay", "wait"), [(0, 2), (1, 2), (30, 30), (60, 60), (61, None)])
def test_retry_after_minimum_and_cap(evaluation_inputs, tmp_path, header_delay, wait):
    adapter, waits = ScriptedAdapter([("api_error", None, {"http_status": 429, "retry_after_s": header_delay})]), []
    directory = run_benchmark(*evaluation_inputs, mock=True, adapters={"mock_1": adapter, "mock_2": ScriptedAdapter()},
                              sleep=waits.append, output_root=tmp_path / "results")
    assert waits == ([] if wait is None else [wait])
    first = read_json(directory / "responses.json")[0]
    assert first["status"] == ("api_error" if wait is None else "ok")
    assert first["attempts"][0]["retry_after_s"] == header_delay


def test_no_third_attempt_or_usage_guessing(evaluation_inputs, tmp_path):
    adapter = ScriptedAdapter([("timeout", None, {}), ("timeout", None, {})])
    directory = run_benchmark(*evaluation_inputs, mock=True, adapters={"mock_1": adapter, "mock_2": ScriptedAdapter()}, sleep=lambda _: None, output_root=tmp_path / "results")
    first = read_json(directory / "responses.json")[0]
    assert first["status"] == "timeout" and len(first["attempts"]) == 2
    assert first["cost"] is None and first["latency_ms"] is None
    assert len(adapter.calls) == 7


def test_draft_real_run_rejected_before_adapter_or_credentials(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Draft preflight touched adapters or credentials")
    monkeypatch.setattr("kazakh_model_watch.runner.OpenRouterAdapter", forbidden)
    from kazakh_model_watch import runner
    original_get = runner.os.environ.get
    monkeypatch.setattr(runner.os.environ, "get", lambda key, default=None: forbidden() if key == "OPENROUTER_API_KEY" else original_get(key, default))
    with pytest.raises(DatasetValidationError):
        run_benchmark(ROOT / "data/benchmark.jsonl", ROOT / "config/models.json", live=True, output_root=tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_readiness_and_opt_in_and_keys_precede_network(input_factory, tmp_path, monkeypatch):
    dataset, config = input_factory(full=True, real=True, ready=False)
    with pytest.raises(ConfigValidationError, match="model_not_ready"):
        run_benchmark(dataset, config, live=True, output_root=tmp_path / "results")
    dataset, config = input_factory(full=True, real=True)
    with pytest.raises(RunError, match="live_opt_in_required"):
        run_benchmark(dataset, config, output_root=tmp_path / "results")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RunError, match="missing_or_invalid_api_key"):
        run_benchmark(dataset, config, live=True, output_root=tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_interruption_preserves_attempt_and_not_run_pairs(evaluation_inputs, tmp_path):
    directory = tmp_path / "results/interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(*evaluation_inputs, mock=True, run_id="interrupted", output_root=directory.parent,
                      adapters={"mock_1": ScriptedAdapter([KeyboardInterrupt()]), "mock_2": ScriptedAdapter()})
    manifest = read_json(directory / "manifest.json")
    rows = read_json(directory / "responses.json")
    assert manifest["run_status"] == "incomplete" and manifest["finished_at"] is None and manifest["interrupted_at"]
    assert rows[0]["status"] == "api_error" and rows[0]["error"]["error_code"] == "interrupted"
    assert sum(row["status"] == "not_run" for row in rows) == 11
    assert all(row["latency_ms"] is None and row["total_elapsed_ms"] is None for row in rows)
    assert report(directory)["run_status"] == "incomplete"


def test_interrupt_in_retry_wait_keeps_only_actual_attempt(evaluation_inputs, tmp_path):
    def interrupt(_):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(*evaluation_inputs, mock=True, run_id="wait", output_root=tmp_path / "results", sleep=interrupt,
                      adapters={"mock_1": ScriptedAdapter([("timeout", None, {})]), "mock_2": ScriptedAdapter()})
    row = read_json(tmp_path / "results/wait/responses.json")[0]
    assert len(row["attempts"]) == 1 and row["status"] == "timeout"
    assert row["attempts"][0]["retry_wait_completed"] is False


def test_duplicate_run_id_never_overwrites(evaluation_inputs, tmp_path):
    directory = run_benchmark(*evaluation_inputs, mock=True, run_id="same", output_root=tmp_path / "results")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(RunError, match="run_id_already_exists"):
        run_benchmark(*evaluation_inputs, mock=True, run_id="same", output_root=tmp_path / "results")
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "", "a/b", "a\\b", ".hidden"])
def test_run_id_cannot_escape_output_root(evaluation_inputs, tmp_path, run_id):
    with pytest.raises(RunError, match="invalid_run_id"):
        run_benchmark(*evaluation_inputs, mock=True, run_id=run_id, output_root=tmp_path / "results")


def test_live_branch_uses_two_mock_transports_only(input_factory, tmp_path, monkeypatch):
    dataset, config_path = input_factory(full=True, real=True)
    config = load_config(config_path)
    key = "synthetic-unit-test-credential-only"
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["temperature"] == 0 and payload["max_tokens"] == 128 and payload["stream"] is False
        assert payload["provider"] == {"require_parameters": True, "allow_fallbacks": False}
        assert list(payload["messages"][0]) == ["role", "content"]
        return httpx.Response(200, json={"id": "synthetic-request", "model": payload["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": "5"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 2, "total_tokens": 2}})
    adapters = {m["key"]: OpenRouterAdapter(m["metadata"]["model"], transport=httpx.MockTransport(handler)) for m in config["models"]}
    directory = run_benchmark(dataset, config_path, live=True, adapters=adapters, clock=itertools.count(0, 0.01).__next__, output_root=tmp_path / "results")
    assert len(requests) == 48 and len({r["model"] for r in requests}) == 2
    rows = read_json(directory / "responses.json")
    assert all(row["latency_ms"] > 0 and row["total_elapsed_ms"] >= row["latency_ms"] for row in rows)
    assert all(row["usage"]["prompt_tokens"] == 0 and row["cost"] is None for row in rows)
    assert key not in "".join(path.read_text(encoding="utf-8") for path in directory.iterdir())


def test_adapter_exception_message_never_saved(evaluation_inputs, tmp_path):
    secret = "synthetic-secret-exception"
    directory = run_benchmark(*evaluation_inputs, mock=True, adapters={"mock_1": ScriptedAdapter([RuntimeError(secret)]), "mock_2": ScriptedAdapter()}, output_root=tmp_path / "results")
    assert secret not in "".join(p.read_text(encoding="utf-8") for p in directory.iterdir())
    assert read_json(directory / "responses.json")[0]["error"]["error_code"] == "adapter_exception"


def test_live_retry_keeps_unknown_usage_for_the_failed_attempt(input_factory, tmp_path, monkeypatch):
    dataset, config_path = input_factory(full=True, real=True)
    config = load_config(config_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-unit-test-credential-only")
    requests, waits = [], []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            raise httpx.ReadTimeout("synthetic timeout detail must not be saved", request=request)
        return httpx.Response(200, json={"id": "synthetic-request", "model": payload["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": "5"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 2, "total_tokens": 2}})
    adapters = {m["key"]: OpenRouterAdapter(m["metadata"]["model"], transport=httpx.MockTransport(handler)) for m in config["models"]}
    directory = run_benchmark(dataset, config_path, live=True, adapters=adapters, sleep=waits.append,
                              clock=itertools.count(0, 0.01).__next__, output_root=tmp_path / "results")
    assert len(requests) == 49 and waits == [2]
    row = read_json(directory / "responses.json")[0]
    assert row["status"] == "ok" and row["retries"] == 1
    assert row["attempts"][0]["usage"] is None and row["attempts"][0]["latency_ms"] is None
    assert row["attempts"][1]["usage"]["prompt_tokens"] == row["usage"]["prompt_tokens"] == 0
    assert row["cost"] is None and row["attempts"][0]["cost"] is None
    assert "synthetic timeout detail" not in "".join(p.read_text(encoding="utf-8") for p in directory.iterdir())


def test_completed_pairs_and_returned_ids_survive_later_interrupt(evaluation_inputs, tmp_path):
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(*evaluation_inputs, mock=True, run_id="after-one", output_root=tmp_path / "results",
                      adapters={"mock_1": ScriptedAdapter(), "mock_2": ScriptedAdapter([KeyboardInterrupt()])})
    directory = tmp_path / "results/after-one"
    rows = read_json(directory / "responses.json")
    manifest = read_json(directory / "manifest.json")
    assert rows[0]["status"] == "ok" and rows[0]["raw_response"] == "5"
    assert rows[1]["status"] == "api_error" and sum(row["status"] == "not_run" for row in rows) == 10
    assert manifest["models"][0]["returned_model_ids"] == ["mock-model-1"]
    assert manifest["models"][1]["returned_model_ids"] == []


def test_every_completed_pair_is_saved_as_a_whole_json_snapshot(evaluation_inputs, tmp_path, monkeypatch):
    from kazakh_model_watch import runner
    original = runner.write_json
    snapshots = []
    def record(path, value):
        original(path, value)
        if path.name == "responses.json":
            decoded = read_json(path)
            assert len(decoded) == 12
            snapshots.append(sum(row["status"] != "not_run" for row in decoded))
    monkeypatch.setattr(runner, "write_json", record)
    run_benchmark(*evaluation_inputs, mock=True, output_root=tmp_path / "results")
    assert snapshots == list(range(13))


def test_secrets_in_public_inputs_are_rejected_before_output(input_factory, tmp_path, monkeypatch):
    dataset, config_path = input_factory(full=True, real=True)
    key = "synthetic-credential-never-in-public-input"
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["models"][0]["metadata"]["model"]["description"] = key
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(RunError, match="secret_in_public_inputs"):
        run_benchmark(dataset, config_path, live=True, output_root=tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_missing_build_dependency_version_remains_unknown(monkeypatch):
    from kazakh_model_watch import runner
    original = runner.metadata.version
    def version(name):
        if name == "setuptools":
            raise runner.metadata.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(runner.metadata, "version", version)
    assert runner._dependency_versions()["setuptools"] is None


def test_adapter_cannot_return_a_not_run_attempt(evaluation_inputs, tmp_path):
    class InvalidAdapter(ScriptedAdapter):
        def generate(self, request):
            return ModelResponse(status="not_run", raw_response=None, requested_model_id=request.model_id)
    directory = run_benchmark(*evaluation_inputs, mock=True, output_root=tmp_path / "results",
                              adapters={"mock_1": InvalidAdapter(), "mock_2": ScriptedAdapter()})
    row = read_json(directory / "responses.json")[0]
    assert row["status"] == "api_error" and row["error"]["error_code"] == "invalid_adapter_response"


def test_optional_reasoning_survives_mock_requests_snapshots_manifest_and_report(evaluation_inputs, tmp_path, monkeypatch):
    dataset, config_path = evaluation_inputs
    config = load_config(config_path)
    config["profile"]["reasoning"] = {"enabled": False}
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls = []
    adapters = {m["key"]: ScriptedAdapter(calls=calls) for m in config["models"]}
    directory = run_benchmark(dataset, config_path, mock=True, adapters=adapters, output_root=tmp_path / "results")
    assert len(calls) == 12 and all(request.reasoning == {"enabled": False} for request in calls)
    assert read_json(directory / "manifest.json")["profile"]["reasoning"] == {"enabled": False}
    assert read_json(directory / "config.json")["profile"]["reasoning"] == {"enabled": False}
    assert any("not guaranteed" in limitation for limitation in read_json(directory / "manifest.json")["limitations"])
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    report(directory)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_non_thinking_mock_transport_roundtrip_preserves_failures_and_usage(input_factory, tmp_path, monkeypatch):
    # "live" exercises the gated branch on synthetic verified fixtures only.
    # Every HTTP call uses MockTransport; pytest-socket blocks actual sockets.
    dataset, config_path = input_factory(full=True, real=True)
    config = load_config(config_path)
    config["profile"]["reasoning"] = {"enabled": False}
    for model in config["models"]:
        model["metadata"]["model"]["supported_parameters"].append("reasoning")
        model["metadata"]["endpoints"][0]["supported_parameters"].append("reasoning")
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    key = "synthetic-non-thinking-test-credential"
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    calls = []
    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert payload["reasoning"] == {"enabled": False}
        assert payload["temperature"] == 0 and payload["max_tokens"] == 128 and payload["stream"] is False
        assert payload["provider"] == {"require_parameters": True, "allow_fallbacks": False}
        assert all(value == 60 for value in request.extensions["timeout"].values())
        if len(calls) == 3:
            return httpx.Response(400, json={"error": {"message": "Synthetic unsupported backend"}})
        return httpx.Response(200, json={"model": payload["model"], "id": "synthetic-reasoning-id",
            "choices": [{"finish_reason": "length" if len(calls) == 2 else "stop", "message": {"content": "5"}}],
            "usage": {"completion_tokens": 293, "completion_tokens_details": {"reasoning_tokens": 291}} if len(calls) <= 2 else None})
    adapters = {m["key"]: OpenRouterAdapter(m["metadata"]["model"], transport=httpx.MockTransport(handler)) for m in config["models"]}
    def no_wait(_):
        pytest.fail("Non-thinking profile or HTTP 400 failure was retried")
    directory = run_benchmark(dataset, config_path, live=True, adapters=adapters, sleep=no_wait, output_root=tmp_path / "results")
    rows = read_json(directory / "responses.json")
    assert len(calls) == 48 and len({payload["model"] for payload in calls}) == 2
    assert rows[0]["status"] == "api_error" and rows[0]["error"]["error_code"] == "reasoning_not_disabled"
    assert rows[1]["status"] == "truncated" and rows[1]["finish_reason"] == "length"
    assert rows[2]["status"] == "api_error" and rows[2]["error"]["http_status"] == 400
    for row in rows[:2]:
        assert row["raw_response"] == "5" and row["usage"] == {"completion_tokens": 293, "completion_tokens_details.reasoning_tokens": 291}
        assert row["exact_match"] == row["correct"] == 0 and len(row["attempts"]) == 1
    assert all(row["status"] == "ok" and row["usage"] is None for row in rows[3:])
    assert read_json(directory / "manifest.json")["profile"]["reasoning"] == {"enabled": False}
    assert key not in "".join(p.read_text(encoding="utf-8") for p in directory.iterdir())
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    monkeypatch.delenv("OPENROUTER_API_KEY")
    report(directory)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_unsupported_reasoning_preflight_blocks_adapter_construction(input_factory, tmp_path, monkeypatch):
    dataset, config_path = input_factory(full=True, real=True)
    config = load_config(config_path)
    config["profile"]["reasoning"] = {"enabled": False}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported reasoning constructed an adapter")
    monkeypatch.setattr("kazakh_model_watch.runner.OpenRouterAdapter", forbidden)
    with pytest.raises(ConfigValidationError):
        run_benchmark(dataset, config_path, live=True, output_root=tmp_path / "results")
    assert not (tmp_path / "results").exists()
