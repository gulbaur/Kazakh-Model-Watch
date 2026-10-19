"""Only MockTransport is used; all credentials and responses are synthetic."""

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest

from kazakh_model_watch import openrouter_smoke
from kazakh_model_watch.model_adapter import ModelRequest
from kazakh_model_watch.openrouter_adapter import CHAT_URL, COMMON_INSTRUCTION, OpenRouterAdapter
from kazakh_model_watch.scoring import score_response


MODEL_ID = "qwen/qwen3.5-9b"
TEST_KEY = "synthetic-key-for-offline-unit-tests-only"


@pytest.fixture(autouse=True)
def synthetic_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", TEST_KEY)


@pytest.fixture
def model_request():
    return ModelRequest(
        provider="openrouter", model_id=MODEL_ID,
        prompt=f"{COMMON_INSTRUCTION}\n\n{openrouter_smoke.SMOKE_TASK}",
    )


@pytest.fixture
def metadata():
    return {
        "id": MODEL_ID, "canonical_slug": "qwen/qwen3.5-9b-20260310",
        "supported_parameters": ["temperature", "max_tokens"],
        "top_provider": {"max_completion_tokens": 32768},
    }


def _body(content="5", finish_reason="stop"):
    return {
        "id": "synthetic-request-id", "model": MODEL_ID,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
    }


def _adapter(metadata, response_body, *, status_code=200):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(status_code, json=response_body)

    return OpenRouterAdapter(metadata, transport=httpx.MockTransport(handle)), calls


def test_request_and_success_preserve_contract_and_cyrillic(metadata, model_request):
    body = _body("  ҚАЛАҒА\n")
    adapter, calls = _adapter(metadata, body)
    response = adapter.generate(model_request)
    assert len(calls) == 1
    sent = calls[0]
    assert sent.method == "POST" and str(sent.url) == CHAT_URL
    assert sent.headers["Authorization"] == f"Bearer {TEST_KEY}"
    assert sent.headers["Content-Type"] == "application/json"
    assert COMMON_INSTRUCTION in sent.content.decode("utf-8")
    assert json.loads(sent.content) == {
        "model": MODEL_ID, "temperature": 0.0, "max_tokens": 128, "stream": False,
        "messages": [{"role": "user", "content": model_request.prompt}],
        "provider": {"require_parameters": True, "allow_fallbacks": False},
    }
    assert all(value == 60.0 for value in sent.extensions["timeout"].values())
    assert asdict(response) == {
        "status": "ok", "raw_response": "  ҚАЛАҒА\n", "requested_model_id": MODEL_ID,
        "returned_model_id": MODEL_ID, "request_id": "synthetic-request-id",
        "finish_reason": "stop", "usage": body["usage"], "latency_ms": None,
    }
    assert adapter.last_diagnostic == {"error_code": None, "http_status": 200, "request_attempted": True}


@pytest.mark.parametrize("content", [None, "", " \t\n", "\u00a0\u2028"])
def test_empty_response_preserves_text(metadata, model_request, content):
    adapter, calls = _adapter(metadata, _body(content))
    response = adapter.generate(model_request)
    assert response.status == "empty_response" and response.raw_response == content
    assert response.latency_ms is None and len(calls) == 1


@pytest.mark.parametrize("content", ["5", "", None])
def test_length_finish_reason_is_truncated_even_without_text(metadata, model_request, content):
    adapter, calls = _adapter(metadata, _body(content, "length"))
    response = adapter.generate(model_request)
    assert response.status == "truncated" and response.raw_response == content
    assert response.finish_reason == "length" and len(calls) == 1


@pytest.mark.parametrize("content", ["5", "", " \t\n", None])
def test_content_filter_is_error_and_keeps_diagnostics(metadata, model_request, content):
    body = _body(content, "content_filter")
    adapter, calls = _adapter(metadata, body)
    response = adapter.generate(model_request)
    assert response.status == "api_error"
    assert response.raw_response == content and response.finish_reason == "content_filter"
    assert response.request_id == body["id"] and response.returned_model_id == MODEL_ID
    assert response.usage == body["usage"] and response.latency_ms is None
    assert adapter.last_diagnostic == {"error_code": "content_filtered", "http_status": 200, "request_attempted": True}
    assert len(calls) == 1
    assert score_response(content, "5", ["5"], status=response.status) == {"exact_match": 0, "correct": 0}


def test_complete_textual_refusal_remains_ok(metadata, model_request):
    text = "Кешіріңіз, бұл сұраққа жауап бере алмаймын."
    adapter, _ = _adapter(metadata, _body(text, "stop"))
    response = adapter.generate(model_request)
    assert response.status == "ok" and response.raw_response == text
    assert response.finish_reason == "stop"
    assert score_response(text, "5", ["5"], status=response.status) == {"exact_match": 0, "correct": 0}


@pytest.mark.parametrize("status_code", [401, 402, 429, 500, 502, 503, 504])
def test_http_errors_never_retry(metadata, model_request, status_code):
    adapter, calls = _adapter(metadata, {"error": {"message": "Synthetic failure"}}, status_code=status_code)
    response = adapter.generate(model_request)
    assert response.status == "api_error" and response.raw_response is None
    assert response.requested_model_id == MODEL_ID and response.usage is None
    assert adapter.last_diagnostic["http_status"] == status_code
    assert adapter.last_diagnostic["error_code"] == "http_error"
    assert len(calls) == 1


@pytest.mark.parametrize(("error_type", "expected_status"), [
    (httpx.ReadTimeout, "timeout"), (httpx.ConnectTimeout, "timeout"),
    (httpx.ConnectError, "api_error"), (httpx.ReadError, "api_error"),
])
def test_transport_errors_are_safe_and_not_retried(metadata, model_request, error_type, expected_status, caplog, capsys):
    calls = []

    def handle(request):
        calls.append(request)
        raise error_type(f"Authorization: Bearer {TEST_KEY}", request=request)

    adapter = OpenRouterAdapter(metadata, transport=httpx.MockTransport(handle))
    with caplog.at_level(logging.DEBUG):
        response = adapter.generate(model_request)
    assert response.status == expected_status and len(calls) == 1
    assert adapter.last_diagnostic["http_status"] is None
    assert TEST_KEY not in repr(response) + repr(adapter.last_diagnostic) + caplog.text
    assert "Authorization" not in caplog.text
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_redirect_is_not_followed(metadata, model_request):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(307, headers={"Location": "https://example.org/other"})

    adapter = OpenRouterAdapter(metadata, transport=httpx.MockTransport(handle))
    assert adapter.generate(model_request).status == "api_error"
    assert adapter.last_diagnostic["http_status"] == 307
    assert len(calls) == 1


def test_missing_key_is_reported_without_network(metadata, model_request, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    adapter, calls = _adapter(metadata, _body())
    response = adapter.generate(model_request)
    assert response.status == "api_error" and not calls
    assert adapter.last_diagnostic == {"error_code": "missing_api_key", "http_status": None, "request_attempted": False}


@pytest.mark.parametrize("key", [" \t ", "secret\nvalue", "не-ascii-key"])
def test_invalid_key_never_reaches_transport(metadata, model_request, monkeypatch, key, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    adapter, calls = _adapter(metadata, _body())
    assert adapter.generate(model_request).status == "api_error"
    assert not calls and adapter.last_diagnostic["request_attempted"] is False
    assert key not in caplog.text


@pytest.mark.parametrize(("field", "value", "code"), [
    ("provider", "mock", "invalid_provider"),
    ("temperature", 1.0, "unsupported_mvp_profile"),
    ("max_output_tokens", 256, "unsupported_mvp_profile"),
    ("timeout_s", 30.0, "unsupported_mvp_profile"),
    ("model_id", "different-model", "unsupported_model_parameters"),
])
def test_request_mismatch_does_not_change_parameters_or_model(metadata, model_request, field, value, code):
    adapter, calls = _adapter(metadata, _body())
    assert adapter.generate(replace(model_request, **{field: value})).status == "api_error"
    assert adapter.last_diagnostic["error_code"] == code and not calls


@pytest.mark.parametrize("parameters", [["temperature"], ["max_tokens"], None, "temperature,max_tokens"])
def test_missing_parameter_support_prevents_paid_call(metadata, model_request, parameters):
    metadata["supported_parameters"] = parameters
    adapter, calls = _adapter(metadata, _body())
    assert adapter.generate(model_request).status == "api_error"
    assert adapter.last_diagnostic["error_code"] == "unsupported_model_parameters" and not calls


def test_insufficient_output_limit_prevents_call(metadata, model_request):
    metadata["top_provider"]["max_completion_tokens"] = 64
    adapter, calls = _adapter(metadata, _body())
    assert adapter.generate(model_request).status == "api_error" and not calls


def test_capability_snapshot_is_not_mutated_by_caller(metadata, model_request):
    adapter, calls = _adapter(metadata, _body())
    metadata["supported_parameters"].clear()
    assert adapter.generate(model_request).status == "ok" and len(calls) == 1


def test_unknown_metadata_remains_none(metadata, model_request):
    body = {"choices": [{"message": {"content": "5"}}]}
    adapter, _ = _adapter(metadata, body)
    response = adapter.generate(model_request)
    assert response.status == "ok"
    for field in ["returned_model_id", "request_id", "finish_reason", "usage", "latency_ms"]:
        assert getattr(response, field) is None


def test_token_details_are_preserved_without_cost_conversion(metadata, model_request):
    body = _body()
    body["usage"] = {
        "prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": None},
        "completion_tokens_details": {"reasoning_tokens": 1},
        "cost": 0.001, "cost_details": {"upstream_inference_cost": 0.001}, "is_byok": False,
    }
    adapter, _ = _adapter(metadata, body)
    assert adapter.generate(model_request).usage == {
        "prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22,
        "prompt_tokens_details.cached_tokens": 0,
        "completion_tokens_details.reasoning_tokens": 1,
    }


def test_cost_only_response_does_not_imply_known_usage(metadata, model_request):
    body = _body()
    body["usage"] = {"cost": 0.001}
    adapter, _ = _adapter(metadata, body)
    assert adapter.generate(model_request).usage is None


@pytest.mark.parametrize("usage", [[], {"prompt_tokens": True}, {"completion_tokens": -1}, {"total_tokens": "22"}, {"completion_tokens_details": []}])
def test_invalid_usage_is_diagnosable_without_losing_valid_ids(metadata, model_request, usage):
    body = _body()
    body["usage"] = usage
    adapter, _ = _adapter(metadata, body)
    response = adapter.generate(model_request)
    assert response.status == "api_error"
    assert response.request_id == "synthetic-request-id" and response.returned_model_id == MODEL_ID
    assert adapter.last_diagnostic["error_code"] == "malformed_response"


@pytest.mark.parametrize("body", [
    [], {}, {"choices": []}, {"choices": [None]}, {"choices": [{}, {}]},
    {"choices": [{"message": {}}]}, {"choices": [{"message": {"content": 5}}]},
    {"choices": [{"message": {"content": ["5"]}}]},
    {"choices": [{"message": {"content": "5", "role": "user"}}]},
])
def test_unexpected_response_shape_is_diagnosable(metadata, model_request, body):
    adapter, calls = _adapter(metadata, body)
    assert adapter.generate(model_request).status == "api_error"
    assert adapter.last_diagnostic["error_code"] == "malformed_response" and len(calls) == 1


def test_non_json_response_does_not_leak_body(metadata, model_request, caplog):
    def handle(request):
        return httpx.Response(200, text=f"malformed Authorization: Bearer {TEST_KEY}")

    adapter = OpenRouterAdapter(metadata, transport=httpx.MockTransport(handle))
    response = adapter.generate(model_request)
    assert response.status == "api_error" and response.raw_response is None
    assert adapter.last_diagnostic["error_code"] == "malformed_response"
    assert TEST_KEY not in repr(response) + repr(adapter.last_diagnostic) + caplog.text


@pytest.mark.parametrize("status_code", [200, 401, 500])
def test_reflected_key_is_never_returned_or_logged(metadata, model_request, caplog, capsys, status_code):
    body = _body(TEST_KEY)
    body["error"] = {"message": f"Authorization: Bearer {TEST_KEY}"}
    adapter, calls = _adapter(metadata, body, status_code=status_code)
    with caplog.at_level(logging.DEBUG):
        response = adapter.generate(model_request)
    assert response.status == "api_error" and response.raw_response is None and len(calls) == 1
    assert adapter.last_diagnostic["error_code"] == "sensitive_response"
    assert TEST_KEY not in repr(response) + repr(adapter.last_diagnostic) + caplog.text
    assert "Authorization" not in caplog.text
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_provider_error_inside_http_200_is_not_success(metadata, model_request):
    adapter, _ = _adapter(metadata, {"id": "synthetic-error-id", "error": {"code": 502}})
    response = adapter.generate(model_request)
    assert response.status == "api_error" and response.request_id == "synthetic-error-id"
    assert adapter.last_diagnostic["error_code"] == "provider_error"


def test_null_error_field_does_not_hide_a_valid_response(metadata, model_request):
    body = _body()
    body["error"] = None
    adapter, calls = _adapter(metadata, body)
    assert adapter.generate(model_request).status == "ok" and len(calls) == 1


def test_unexpected_model_is_not_silently_accepted(metadata, model_request):
    body = _body()
    body["model"] = "different-model"
    adapter, calls = _adapter(metadata, body)
    response = adapter.generate(model_request)
    assert response.status == "api_error" and response.returned_model_id == "different-model"
    assert response.raw_response == "5" and adapter.last_diagnostic["error_code"] == "unexpected_model_id"
    assert len(calls) == 1


def test_known_canonical_model_id_is_preserved(metadata, model_request):
    body = _body()
    body["model"] = metadata["canonical_slug"]
    adapter, _ = _adapter(metadata, body)
    response = adapter.generate(model_request)
    assert response.status == "ok" and response.returned_model_id == metadata["canonical_slug"]


@pytest.mark.parametrize("finish_reason", ["tool_calls", "error", "future_finish_reason"])
def test_unsupported_finish_reason_is_error_with_original_text(metadata, model_request, finish_reason):
    adapter, _ = _adapter(metadata, _body("5", finish_reason))
    response = adapter.generate(model_request)
    assert response.status == "api_error" and response.raw_response == "5"
    assert response.finish_reason == finish_reason
    assert adapter.last_diagnostic["error_code"] == "unexpected_finish_reason"


def test_invalid_finish_reason_keeps_available_text_and_ids(metadata, model_request):
    adapter, _ = _adapter(metadata, _body("5", False))
    response = adapter.generate(model_request)
    assert response.status == "api_error" and response.raw_response == "5"
    assert response.request_id == "synthetic-request-id" and response.finish_reason is None


def test_diagnostics_reset_after_each_call(metadata, model_request, monkeypatch):
    adapter, calls = _adapter(metadata, _body())
    assert adapter.generate(model_request).status == "ok"
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert adapter.generate(model_request).status == "api_error"
    assert adapter.last_diagnostic["http_status"] is None and adapter.last_diagnostic["request_attempted"] is False
    assert len(calls) == 1


@pytest.mark.parametrize("argv", [[], ["--dotenv"]])
def test_smoke_default_is_offline_and_never_reads_dotenv(monkeypatch, capsys, argv):
    import dotenv

    def forbidden(*args, **kwargs):
        pytest.fail("Offline CLI attempted network or dotenv loading")

    monkeypatch.setattr(httpx, "HTTPTransport", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    assert openrouter_smoke.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "dry_run"
    assert result["payload"]["model"] == MODEL_ID
    assert result["payload"]["messages"] == [{"role": "user", "content": f"{COMMON_INSTRUCTION}\n\n{openrouter_smoke.SMOKE_TASK}"}]
    assert TEST_KEY not in repr(result)


def test_smoke_live_branch_is_mocked_and_sends_one_post(metadata, monkeypatch, capsys):
    # The --live branch is exercised only with an injected offline MockTransport.
    adapter, calls = _adapter(metadata, _body())
    monkeypatch.setattr(openrouter_smoke, "OpenRouterAdapter", lambda model_metadata: adapter)
    assert openrouter_smoke.main(["--live"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok" and result["requested_model_id"] == MODEL_ID
    assert result["elapsed_ms"] >= 0 and result["http_status"] == 200 and result["error_code"] is None
    assert len(calls) == 1 and TEST_KEY not in repr(result)


def test_smoke_dotenv_loading_is_explicit_and_mocked(metadata, monkeypatch, capsys):
    import dotenv

    adapter, calls = _adapter(metadata, _body())
    dotenv_calls = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda **kwargs: dotenv_calls.append(kwargs))
    monkeypatch.setattr(openrouter_smoke, "OpenRouterAdapter", lambda model_metadata: adapter)
    assert openrouter_smoke.main(["--live", "--dotenv"]) == 0
    assert dotenv_calls == [{"dotenv_path": Path.cwd() / ".env", "override": False}]
    assert len(calls) == 1 and TEST_KEY not in capsys.readouterr().out


def test_smoke_missing_key_has_unknown_duration_and_http_status(metadata, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    adapter, calls = _adapter(metadata, _body())
    monkeypatch.setattr(openrouter_smoke, "OpenRouterAdapter", lambda model_metadata: adapter)
    assert openrouter_smoke.main(["--live"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error_code"] == "missing_api_key" and not calls
    assert result["elapsed_ms"] is None and result["http_status"] is None


def test_smoke_metadata_file_error_is_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(openrouter_smoke, "METADATA_PATH", tmp_path / "missing.json")
    assert openrouter_smoke.main(["--live"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error_code"] == "invalid_model_metadata"
    assert result["requested_model_id"] == MODEL_ID and result["http_status"] is None


def test_saved_capabilities_are_a_public_models_api_snapshot():
    snapshot = json.loads(openrouter_smoke.METADATA_PATH.read_text(encoding="utf-8"))
    assert snapshot["source_url"] == "https://openrouter.ai/api/v1/models"
    assert snapshot["model"]["id"] == MODEL_ID
    assert {"temperature", "max_tokens"}.issubset(snapshot["model"]["supported_parameters"])


@pytest.mark.parametrize(("header", "expected"), [("0", 0), ("2", 2), ("60", 60), ("61", 61), ("bad", None), ("nan", None), ("-1", None)])
def test_retry_after_is_safe_optional_metadata_without_retry(metadata, model_request, header, expected):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": header}, json={"error": {"message": "Synthetic failure"}})
    adapter = OpenRouterAdapter(metadata, transport=httpx.MockTransport(handler))
    assert adapter.generate(model_request).status == "api_error"
    assert adapter.get_diagnostic()["retry_after_s"] == expected
    assert len(calls) == 1
    copy = adapter.get_diagnostic()
    copy["http_status"] = 200
    assert adapter.last_diagnostic["http_status"] == 429


def test_retry_after_http_date_and_reset(metadata, model_request):
    from kazakh_model_watch.openrouter_adapter import _retry_after
    assert _retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0
    assert _retry_after("Fri, 01 Jan 2100 00:00:00 GMT") > 60
    assert _retry_after("inf") is None
    adapter, calls = _adapter(metadata, _body())
    assert adapter.generate(model_request).status == "ok"
    assert adapter.get_diagnostic()["retry_after_s"] is None


def test_default_adapter_diagnostic_is_backwards_compatible(model_request):
    from kazakh_model_watch.model_adapter import MockAdapter, ModelResponse
    mock = MockAdapter(ModelResponse(status="ok", raw_response="synthetic", requested_model_id="mock-fixture"))
    assert mock.get_diagnostic() == {}
