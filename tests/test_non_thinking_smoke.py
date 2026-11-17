"""Synthetic keys/metadata and MockTransport only, including --live branches."""

import itertools
import json
from types import SimpleNamespace

import httpx
import pytest

from conftest import ROOT
from kazakh_model_watch import non_thinking_smoke as smoke
from kazakh_model_watch.openrouter_adapter import COMMON_INSTRUCTION, OpenRouterAdapter


KEY = "synthetic-limited-smoke-test-credential"
LIVE = ["--live", "--approved-budget-usd", "0.10", "--run-id", "synthetic"]


@pytest.fixture
def local_io(tmp_path, monkeypatch):
    path = tmp_path / "models.json"
    path.write_bytes((ROOT / "config/models.json").read_bytes())
    output = tmp_path / "logs/non-thinking-smoke"
    monkeypatch.setattr(smoke, "CONFIG_PATH", path)
    monkeypatch.setattr(smoke, "OUTPUT_ROOT", output)
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(smoke, "time", SimpleNamespace(perf_counter=itertools.count(0, 0.125).__next__))
    return path, output


def _body(model_id, *, finish="stop", content="  ҚАЛАҒА\n", usage=True):
    return {"id": "synthetic-smoke-request", "model": model_id,
            "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}],
            "usage": {"completion_tokens": 2, "completion_tokens_details": {"reasoning_tokens": 0}} if usage else None}


def _transport(monkeypatch, handler=None):
    calls = []
    def handle(request):
        calls.append(request)
        model_id = json.loads(request.content)["model"]
        return handler(request, model_id, len(calls)) if handler else httpx.Response(200, json=_body(model_id))
    monkeypatch.setattr(smoke, "OpenRouterAdapter", lambda metadata: OpenRouterAdapter(metadata, transport=httpx.MockTransport(handle)))
    return calls


def _saved(output):
    return json.loads((output / "synthetic/diagnostics.json").read_text(encoding="utf-8"))


def test_default_is_offline_without_key_dotenv_adapter_or_files(local_io, monkeypatch, capsys):
    import dotenv
    def forbidden(*args, **kwargs):
        pytest.fail("Dry-run touched credentials, adapter or dotenv")
    monkeypatch.setattr(OpenRouterAdapter, "__init__", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(smoke, "os", SimpleNamespace(environ=SimpleNamespace(get=forbidden)))
    assert smoke.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "dry_run" and result["maximum_posts"] == 2
    assert [payload["model"] for payload in result["payloads"]] == list(smoke.MODEL_IDS)
    for payload in result["payloads"]:
        assert payload["reasoning"] == {"enabled": False}
        assert payload["messages"] == [{"role": "user", "content": COMMON_INSTRUCTION + "\n\n" + smoke.SMOKE_TASK}]
    assert not local_io[1].exists()


@pytest.mark.parametrize("reverse", [False, True])
def test_two_successful_probes_use_fixed_order_payload_and_safe_local_fields(local_io, monkeypatch, capsys, reverse):
    path, output = local_io
    config = json.loads(path.read_text(encoding="utf-8"))
    if reverse:
        config["models"].reverse()
        path.write_text(json.dumps(config), encoding="utf-8")
    before = path.read_bytes()
    calls = _transport(monkeypatch)
    assert smoke.main(LIVE) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "complete" and "raw_response" not in summary
    assert len(calls) == 2
    for call, model_id in zip(calls, smoke.MODEL_IDS):
        payload = json.loads(call.content)
        assert payload == {"model": model_id, "temperature": 0, "max_tokens": 128, "stream": False,
                           "reasoning": {"enabled": False}, "messages": [{"role": "user", "content": COMMON_INSTRUCTION + "\n\n" + smoke.SMOKE_TASK}],
                           "provider": {"require_parameters": True, "allow_fallbacks": False}}
        assert all(value == 60 for value in call.extensions["timeout"].values())
        assert KEY not in call.content.decode("utf-8")
    saved = _saved(output)
    assert saved["run_status"] == "complete" and saved["budget_is_enforced_by_cli"] is False
    assert saved["approved_budget_usd"] == "0.10"
    assert saved["profile"] == config["profile"]
    for row, model_id in zip(saved["responses"], smoke.MODEL_IDS):
        assert row["requested_model_id"] == row["returned_model_id"] == model_id
        assert row["status"] == "ok" and row["finish_reason"] == "stop" and row["http_status"] == 200
        assert row["raw_response"] == "  ҚАЛАҒА\n" and row["request_id"] == "synthetic-smoke-request"
        assert row["usage"] == {"completion_tokens": 2, "completion_tokens_details.reasoning_tokens": 0}
        assert row["latency_ms"] == row["elapsed_ms"] == 125 and row["checks"] == []
        assert row["reasoning_evidence"]["reasoning_details"] == {"present": False, "type": "null", "nonempty": False}
        assert row["reasoning_evidence"]["detail_types"] == []
    assert path.read_bytes() == before
    assert KEY not in (output / "synthetic/diagnostics.json").read_text(encoding="utf-8")
    assert set((output / "synthetic").iterdir()) == {output / "synthetic/diagnostics.json"}


@pytest.mark.parametrize("status", [400, 401, 402, 404, 429, 500, 503])
def test_first_http_error_does_not_retry_or_call_second(local_io, monkeypatch, capsys, status):
    calls = _transport(monkeypatch, lambda *_: httpx.Response(status, json={"error": {"message": "synthetic provider error"}}))
    assert smoke.main(LIVE) == 1 and len(calls) == 1
    saved = _saved(local_io[1])
    assert saved["run_status"] == "stopped" and saved["responses"][0]["http_status"] == status
    assert saved["responses"][0]["error_code"] == "http_error"
    assert saved["responses"][0]["latency_ms"] is None and saved["responses"][0]["elapsed_ms"] == 125
    second = saved["responses"][1]
    assert second["status"] == "not_run" and second["request_attempted"] is False
    assert saved["responses"][0]["reasoning_evidence"] is None and second["reasoning_evidence"] is None
    assert all(second[field] is None for field in ("raw_response", "usage", "latency_ms", "elapsed_ms", "http_status"))


def test_timeout_no_retry_or_second_model(local_io, monkeypatch):
    def handler(request, *_):
        raise httpx.ReadTimeout(KEY, request=request)
    calls = _transport(monkeypatch, handler)
    assert smoke.main(LIVE) == 1 and len(calls) == 1
    row = _saved(local_io[1])["responses"][0]
    assert row["status"] == "timeout" and row["error_code"] == "request_timeout"
    assert row["usage"] is None and row["elapsed_ms"] == 125
    assert KEY not in (local_io[1] / "synthetic/diagnostics.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("reasoning", [0, 291])
def test_excess_reported_usage_is_preserved_and_blocks_second(local_io, monkeypatch, reasoning):
    def handler(_, model_id, __):
        body = _body(model_id, content="5")
        body["usage"] = {"completion_tokens": 293, "completion_tokens_details": {"reasoning_tokens": reasoning}}
        return httpx.Response(200, json=body)
    calls = _transport(monkeypatch, handler)
    assert smoke.main(LIVE) == 1 and len(calls) == 1
    saved = _saved(local_io[1])
    row = saved["responses"][0]
    assert row["status"] == ("api_error" if reasoning else "ok") and row["finish_reason"] == "stop"
    assert row["raw_response"] == "5"
    assert row["usage"] == {"completion_tokens": 293, "completion_tokens_details.reasoning_tokens": reasoning}
    assert "reported_completion_exceeds_requested_max_tokens" in row["checks"]
    if reasoning:
        assert row["error_code"] == "reasoning_not_disabled" and "reasoning_reported" in row["checks"]
    assert saved["responses"][1]["status"] == "not_run"


@pytest.mark.parametrize("evidence", ["reasoning", "reasoning_details"])
def test_reasoning_content_without_counters_blocks_second(local_io, monkeypatch, evidence):
    def handler(_, model_id, __):
        body = _body(model_id)
        body["choices"][0]["message"][evidence] = "Synthetic thought" if evidence == "reasoning" else [{"text": "Synthetic thought"}]
        return httpx.Response(200, json=body)
    calls = _transport(monkeypatch, handler)
    assert smoke.main(LIVE) == 1 and len(calls) == 1
    row = _saved(local_io[1])["responses"][0]
    assert row["status"] == "api_error" and row["error_code"] == "reasoning_not_disabled"
    assert "Synthetic thought" not in (local_io[1] / "synthetic/diagnostics.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("usage", [None, {"completion_tokens": 2}, {"completion_tokens_details": {"reasoning_tokens": 0}}])
def test_unknown_usage_is_not_a_zero_or_permission_for_second(local_io, monkeypatch, usage):
    def handler(_, model_id, __):
        body = _body(model_id)
        body["usage"] = usage
        return httpx.Response(200, json=body)
    calls = _transport(monkeypatch, handler)
    assert smoke.main(LIVE) == 1 and len(calls) == 1
    row = _saved(local_io[1])["responses"][0]
    assert row["status"] == "ok" and any(check.endswith("usage_unknown") for check in row["checks"])
    assert row["usage"] == ({"completion_tokens_details.reasoning_tokens": 0} if usage and "completion_tokens_details" in usage else usage)


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls", None])
def test_finish_reason_anomaly_stops_before_second(local_io, monkeypatch, finish):
    calls = _transport(monkeypatch, lambda _, model_id, __: httpx.Response(200, json=_body(model_id, finish=finish)))
    assert smoke.main(LIVE) == 1 and len(calls) == 1
    row = _saved(local_io[1])["responses"][0]
    assert row["finish_reason"] == finish
    assert row["status"] == ("truncated" if finish == "length" else "ok" if finish is None else "api_error")


def test_first_checkpoint_exists_before_second_error(local_io, monkeypatch):
    def handler(_, model_id, number):
        if number == 2:
            first = _saved(local_io[1])["responses"][0]
            assert first["status"] == "ok" and first["raw_response"] == "  ҚАЛАҒА\n"
            assert first["reasoning_evidence"]["detail_types"] == []
            assert _saved(local_io[1])["responses"][1]["reasoning_evidence"] is None
            return httpx.Response(500, json={"error": {"message": "synthetic second failure"}})
        return httpx.Response(200, json=_body(model_id))
    calls = _transport(monkeypatch, handler)
    assert smoke.main(LIVE) == 1 and len(calls) == 2
    rows = _saved(local_io[1])["responses"]
    assert rows[0]["status"] == "ok" and rows[1]["http_status"] == 500


@pytest.mark.parametrize("mutation", ["model_parameter", "endpoint_parameter", "mandatory", "reasoning_absent", "temperature", "wrong_id"])
def test_preflight_checks_both_models_before_any_post(local_io, monkeypatch, mutation):
    path, output = local_io
    config = json.loads(path.read_text(encoding="utf-8"))
    meta = config["models"][1]["metadata"]
    if mutation == "model_parameter":
        meta["model"]["supported_parameters"].remove("reasoning")
    elif mutation == "endpoint_parameter":
        for endpoint in meta["endpoints"]:
            endpoint["supported_parameters"].remove("reasoning")
    elif mutation == "mandatory":
        meta["model"]["reasoning"]["mandatory"] = True
    elif mutation == "reasoning_absent":
        config["profile"].pop("reasoning")
    elif mutation == "temperature":
        config["profile"]["temperature"] = 1
    else:
        config["models"][1]["model_id"] = "synthetic/wrong-id"
    path.write_text(json.dumps(config), encoding="utf-8")
    calls = _transport(monkeypatch)
    assert smoke.main(LIVE) == 1 and calls == [] and not output.exists()


@pytest.mark.parametrize("budget", [None, "0", "-1", "NaN", "Infinity", "invalid"])
def test_live_requires_explicit_positive_budget(local_io, monkeypatch, budget):
    calls = _transport(monkeypatch)
    args = ["--live"] + (["--approved-budget-usd", budget] if budget is not None else [])
    assert smoke.main(args) == 1 and calls == [] and not local_io[1].exists()


def test_missing_key_is_offline_and_dotenv_never_loads(local_io, monkeypatch):
    import dotenv
    monkeypatch.delenv("OPENROUTER_API_KEY")
    def forbidden(*args, **kwargs):
        pytest.fail("Smoke attempted dotenv loading")
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    calls = _transport(monkeypatch)
    assert smoke.main(LIVE) == 1 and calls == [] and not local_io[1].exists()


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "a/b", "a\\b", ""])
def test_unsafe_run_id_cannot_write_or_call(local_io, monkeypatch, run_id):
    calls = _transport(monkeypatch)
    assert smoke.main(LIVE[:-1] + [run_id]) == 1 and calls == [] and not local_io[1].exists()


def test_reused_run_id_cannot_repeat_posts_or_overwrite(local_io, monkeypatch):
    calls = _transport(monkeypatch)
    assert smoke.main(LIVE) == 0 and len(calls) == 2
    output = local_io[1] / "synthetic/diagnostics.json"
    before = output.read_bytes()
    assert smoke.main(LIVE) == 1 and len(calls) == 2
    assert output.read_bytes() == before


def test_checkpoint_write_failure_blocks_second_without_error_text(local_io, monkeypatch, capsys):
    calls = _transport(monkeypatch)
    original = smoke.write_json
    count = 0
    def write(path, value):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError(KEY)
        original(path, value)
    monkeypatch.setattr(smoke, "write_json", write)
    assert smoke.main(LIVE) == 1 and len(calls) == 1
    assert KEY not in capsys.readouterr().out
    assert _saved(local_io[1])["responses"][0]["error_code"] == "unfinished_attempt"
    assert _saved(local_io[1])["responses"][0]["reasoning_evidence"] is None


def test_interrupted_call_is_not_retried_and_second_is_not_run(local_io, monkeypatch):
    def handler(*_):
        raise KeyboardInterrupt
    calls = _transport(monkeypatch, handler)
    assert smoke.main(LIVE) == 130 and len(calls) == 1
    saved = _saved(local_io[1])
    assert saved["run_status"] == "interrupted"
    assert saved["responses"][0]["error_code"] == "interrupted" and saved["responses"][1]["status"] == "not_run"
    assert all(row["reasoning_evidence"] is None for row in saved["responses"])


def test_reflected_key_is_never_saved_or_printed(local_io, monkeypatch, capsys, caplog):
    calls = _transport(monkeypatch, lambda _, model_id, __: httpx.Response(200, json=_body(model_id, content=KEY)))
    assert smoke.main(LIVE) == 1 and len(calls) == 1
    assert KEY not in capsys.readouterr().out + caplog.text + (local_io[1] / "synthetic/diagnostics.json").read_text(encoding="utf-8")
    row = _saved(local_io[1])["responses"][0]
    assert row["raw_response"] is None and row["error_code"] == "sensitive_response"
    assert row["reasoning_evidence"] is None


def test_zero_usage_and_zero_measured_time_stay_zero(local_io, monkeypatch):
    monkeypatch.setattr(smoke.time, "perf_counter", lambda: 0)
    def handler(_, model_id, __):
        body = _body(model_id)
        body["usage"]["completion_tokens"] = 0
        return httpx.Response(200, json=body)
    calls = _transport(monkeypatch, handler)
    assert smoke.main(LIVE) == 0 and len(calls) == 2
    assert all(row["latency_ms"] == 0 and row["usage"]["completion_tokens"] == 0 for row in _saved(local_io[1])["responses"])


@pytest.mark.parametrize("malformed", ["json", "shape", "model"])
def test_malformed_or_unexpected_response_never_calls_second(local_io, monkeypatch, malformed):
    def handler(_, model_id, __):
        if malformed == "json":
            return httpx.Response(200, content=b"not JSON")
        body = {} if malformed == "shape" else _body("synthetic/unexpected-model")
        return httpx.Response(200, json=body)
    calls = _transport(monkeypatch, handler)
    assert smoke.main(LIVE) == 1 and len(calls) == 1
    saved = _saved(local_io[1])
    assert saved["responses"][0]["error_code"] == ("unexpected_model_id" if malformed == "model" else "malformed_response")
    assert saved["responses"][1]["status"] == "not_run"


@pytest.mark.parametrize("location", ["path", "config"])
def test_key_in_output_path_or_public_metadata_fails_without_leak(local_io, monkeypatch, capsys, location):
    args = LIVE
    if location == "path":
        args = LIVE[:-1] + [KEY]
    else:
        config = json.loads(local_io[0].read_text(encoding="utf-8"))
        config["models"][0]["metadata"]["model"]["synthetic_extra"] = KEY
        local_io[0].write_text(json.dumps(config), encoding="utf-8")
    calls = _transport(monkeypatch)
    assert smoke.main(args) == 1 and calls == [] and not local_io[1].exists()
    assert KEY not in capsys.readouterr().out


@pytest.mark.parametrize("detail_type", ["reasoning.text", "reasoning.summary", "reasoning.encrypted", "synthetic private type"])
def test_gemini_evidence_is_saved_without_payloads_and_still_stops(local_io, monkeypatch, capsys, detail_type):
    def handler(_, model_id, number):
        body = _body(model_id, content="5")
        if number == 2:
            body["choices"][0]["message"]["reasoning_details"] = [{
                "type": detail_type,
                "text": "synthetic private thought" if detail_type != "reasoning.encrypted" else "",
                "data": "synthetic opaque signature" if detail_type == "reasoning.encrypted" else "",
            }]
        return httpx.Response(200, json=body)
    calls = _transport(monkeypatch, handler)
    assert smoke.main(LIVE) == 1 and len(calls) == 2
    saved = _saved(local_io[1])
    first, second = saved["responses"]
    assert first["status"] == "ok" and first["reasoning_evidence"]["detail_types"] == []
    assert second["status"] == "api_error" and second["error_code"] == "reasoning_not_disabled"
    assert second["reasoning_evidence"]["detail_types"] == [detail_type if detail_type.startswith("reasoning.") else "unknown"]
    assert second["reasoning_evidence"]["detail_text_nonempty"] is (detail_type != "reasoning.encrypted")
    assert second["reasoning_evidence"]["detail_data_nonempty"] is (detail_type == "reasoning.encrypted")
    assert second["usage"]["completion_tokens_details.reasoning_tokens"] == 0
    assert saved["run_status"] == "stopped" and saved["stop_reason"] == "response_not_ok"
    serialized = (local_io[1] / "synthetic/diagnostics.json").read_text(encoding="utf-8") + capsys.readouterr().out
    assert all(text not in serialized for text in (KEY, "synthetic private thought", "synthetic opaque signature", "synthetic private type"))


def test_upstream_provider_is_saved_for_each_probe(local_io, monkeypatch):
    def handler(_, model_id, number):
        return httpx.Response(200, json={**_body(model_id), **({"provider": "Synthetic Upstream"} if number == 1 else {})})
    calls = _transport(monkeypatch, handler)
    assert smoke.main(LIVE) == 0 and len(calls) == 2
    first, second = _saved(local_io[1])["responses"]
    assert first["upstream_provider"] == "Synthetic Upstream" and second["upstream_provider"] is None
    assert first["checks"] == second["checks"] == []


def test_not_run_and_unfinished_rows_have_unknown_upstream(local_io, monkeypatch):
    calls = _transport(monkeypatch, lambda *_: httpx.Response(500, json={"error": {"message": "synthetic"}}))
    assert smoke.main(LIVE) == 1 and len(calls) == 1
    first, second = _saved(local_io[1])["responses"]
    assert first["upstream_provider"] is None and second["upstream_provider"] is None
