"""Offline contract tests; all response IDs, usage and timing are synthetic."""

import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from kazakh_model_watch.model_adapter import (
    MockAdapter, ModelAdapter, ModelRequest, ModelResponse,
)
from kazakh_model_watch.scoring import score_response


@pytest.fixture
def model_request():
    return ModelRequest(provider="mock", model_id="mock-model", prompt="Тек жауапты жазыңыз.")


def _response(**changes):
    fields = {"status": "ok", "raw_response": "5", "requested_model_id": "mock-model"}
    fields.update(changes)
    return ModelResponse(**fields)


def test_request_defaults_follow_mvp_profile_and_preserve_prompt(model_request):
    assert asdict(model_request) == {
        "provider": "mock", "model_id": "mock-model", "prompt": "Тек жауапты жазыңыз.",
        "temperature": 0.0, "max_output_tokens": 128, "timeout_s": 60.0,
    }
    assert replace(model_request, prompt="  ҮИ\u0306\n").prompt == "  ҮИ\u0306\n"


def test_mock_returns_correct_text_and_all_response_fields(model_request):
    preset = _response(
        raw_response="  ҚАЛАҒА\n", returned_model_id="mock-snapshot",
        request_id="synthetic-request", finish_reason="stop",
        usage={"input_tokens": 10, "output_tokens": 2}, latency_ms=1.5,
    )
    response = MockAdapter(preset).generate(model_request)
    expected = {
        "status": "ok", "raw_response": "  ҚАЛАҒА\n",
        "requested_model_id": "mock-model", "returned_model_id": "mock-snapshot",
        "request_id": "synthetic-request", "finish_reason": "stop",
        "usage": {"input_tokens": 10, "output_tokens": 2}, "latency_ms": 1.5,
    }
    assert asdict(response) == expected
    assert json.loads(json.dumps(asdict(response), ensure_ascii=False)) == expected


def test_unknown_metadata_is_none_not_fabricated(model_request):
    response = MockAdapter(_response()).generate(model_request)
    for field in ("returned_model_id", "request_id", "finish_reason", "usage", "latency_ms"):
        assert getattr(response, field) is None


@pytest.mark.parametrize("text", [None, "", " \t\n", "\u00a0\u2028"])
def test_mock_empty_response_retains_original_text(model_request, text):
    response = MockAdapter(_response(status="empty_response", raw_response=text)).generate(model_request)
    assert response.status == "empty_response"
    assert response.raw_response == text
    assert response.latency_ms is None
    assert score_response(response.raw_response, "5", ["5"], status=response.status) == {
        "exact_match": 0, "correct": 0,
    }


@pytest.mark.parametrize("status", ["timeout", "api_error"])
def test_mock_returns_failure_status_without_raising_or_waiting(model_request, status):
    response = MockAdapter(_response(status=status, raw_response=None)).generate(model_request)
    assert response.status == status
    assert response.raw_response is None
    assert response.requested_model_id == model_request.model_id
    assert response.usage is None
    assert response.latency_ms is None


def test_truncated_response_keeps_text_but_cannot_score(model_request):
    response = MockAdapter(_response(status="truncated", finish_reason="length")).generate(model_request)
    assert response.raw_response == "5"
    assert score_response(response.raw_response, "5", ["5"], status=response.status) == {
        "exact_match": 0, "correct": 0,
    }


def test_refusal_with_visible_text_remains_ok(model_request):
    response = MockAdapter(_response(raw_response="Кешіріңіз, жауап бере алмаймын.")).generate(model_request)
    assert response.status == "ok"
    assert score_response(response.raw_response, "5", ["5"], status=response.status)["correct"] == 0


def test_not_run_has_no_text_usage_or_latency():
    response = _response(status="not_run", raw_response=None)
    assert response.raw_response is None
    assert response.usage is None
    assert response.latency_ms is None


@pytest.mark.parametrize("usage", [
    {"input_tokens": 0, "output_tokens": 0},
    {"input_tokens": 10, "output_tokens": 2},
])
def test_not_run_rejects_usage_even_with_zero_counters(usage):
    with pytest.raises(ValueError, match="usage must be None for status not_run"):
        _response(status="not_run", raw_response=None, usage=usage)


def test_mock_is_repeatable_and_isolates_mutable_usage(model_request):
    preset = _response(usage={"input_tokens": 10, "output_tokens": 2})
    adapter = MockAdapter(preset)
    preset.usage["input_tokens"] = 999
    first = adapter.generate(model_request)
    first.usage["output_tokens"] = 999
    second = adapter.generate(model_request)
    assert second.usage == {"input_tokens": 10, "output_tokens": 2}
    assert adapter.generate(model_request) == second


def test_request_and_response_attributes_cannot_be_reassigned(model_request):
    with pytest.raises(FrozenInstanceError):
        model_request.model_id = "different-model"
    with pytest.raises(FrozenInstanceError):
        _response().status = "api_error"


def test_base_interface_requires_generate():
    with pytest.raises(TypeError):
        ModelAdapter()


def test_mock_implements_shared_interface():
    assert isinstance(MockAdapter(_response()), ModelAdapter)


def test_mock_cannot_impersonate_a_real_provider(model_request):
    with pytest.raises(ValueError, match="provider"):
        MockAdapter(_response()).generate(replace(model_request, provider="real-provider"))


def test_mock_rejects_request_for_different_model(model_request):
    with pytest.raises(ValueError, match="model_id"):
        MockAdapter(_response()).generate(replace(model_request, model_id="different-model"))


def test_mock_rejects_untyped_inputs():
    with pytest.raises(TypeError, match="response"):
        MockAdapter({"status": "ok"})
    with pytest.raises(TypeError, match="request"):
        MockAdapter(_response()).generate({"provider": "mock"})


@pytest.mark.parametrize(("field", "value", "error"), [
    ("provider", None, TypeError), ("provider", " ", ValueError),
    ("model_id", 1, TypeError), ("model_id", "", ValueError),
    ("prompt", [], TypeError), ("prompt", "\t", ValueError),
    ("temperature", True, TypeError), ("temperature", -1, ValueError),
    ("temperature", float("nan"), ValueError), ("temperature", float("inf"), ValueError),
    ("max_output_tokens", True, TypeError), ("max_output_tokens", 128.0, TypeError),
    ("max_output_tokens", 0, ValueError), ("max_output_tokens", -1, ValueError),
    ("timeout_s", "60", TypeError), ("timeout_s", False, TypeError),
    ("timeout_s", 0, ValueError), ("timeout_s", -1, ValueError),
    ("timeout_s", float("inf"), ValueError),
])
def test_request_rejects_invalid_fields(model_request, field, value, error):
    with pytest.raises(error, match=field):
        replace(model_request, **{field: value})


@pytest.mark.parametrize(("field", "value", "error"), [
    ("status", "unknown", ValueError), ("status", None, TypeError),
    ("raw_response", 5, TypeError), ("raw_response", b"5", TypeError),
    ("raw_response", None, ValueError), ("raw_response", "", ValueError),
    ("raw_response", "\t\n", ValueError),
    ("requested_model_id", "", ValueError), ("requested_model_id", 1, TypeError),
    ("returned_model_id", " ", ValueError), ("returned_model_id", False, TypeError),
    ("request_id", [], TypeError), ("request_id", "", ValueError),
    ("finish_reason", 1, TypeError), ("finish_reason", "", ValueError),
    ("usage", [], TypeError), ("usage", {}, ValueError),
    ("usage", {"output_tokens": -1}, ValueError),
    ("usage", {"output_tokens": True}, TypeError),
    ("usage", {"output_tokens": 2.5}, TypeError),
    ("usage", {1: 2}, TypeError), ("usage", {"": 2}, ValueError),
    ("latency_ms", "1", TypeError), ("latency_ms", True, TypeError),
    ("latency_ms", -1, ValueError), ("latency_ms", float("nan"), ValueError),
    ("latency_ms", float("inf"), ValueError),
])
def test_response_rejects_invalid_fields(field, value, error):
    with pytest.raises(error, match=field.split("_")[0]):
        _response(**{field: value})


@pytest.mark.parametrize("status", ["timeout", "api_error", "empty_response", "truncated", "not_run"])
def test_unsuccessful_response_has_no_successful_attempt_latency(status):
    with pytest.raises(ValueError, match="latency_ms"):
        _response(status=status, raw_response=None, latency_ms=0)


def test_empty_response_cannot_contain_visible_text():
    with pytest.raises(ValueError, match="raw_response"):
        _response(status="empty_response")


def test_not_run_cannot_contain_response_text():
    with pytest.raises(ValueError, match="raw_response"):
        _response(status="not_run")


def test_zero_usage_and_zero_success_latency_are_measured_values():
    response = _response(usage={"output_tokens": 0}, latency_ms=0)
    assert response.usage == {"output_tokens": 0}
    assert response.latency_ms == 0
