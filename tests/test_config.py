"""Public metadata validation is offline and never establishes live behavior."""

import json
from copy import deepcopy

import pytest

from conftest import ROOT
from kazakh_model_watch.config import ConfigValidationError, load_config, validate_config


def test_repository_real_config_has_two_distinct_not_ready_models():
    config = load_config(ROOT / "config/models.json")
    assert config["protocol_version"] == "mvp-v2-non-thinking"
    assert config["profile"]["reasoning"] == {"enabled": False}
    assert [m["model_id"] for m in config["models"]] == ["qwen/qwen3.5-9b", "google/gemini-3.1-flash-lite"]
    assert all(m["readiness"] == "not_ready" and m["readiness_reason"] for m in config["models"])
    assert all(m["api_key_env"] == "OPENROUTER_API_KEY" and not m["metadata"]["profile_verified"] for m in config["models"])
    with pytest.raises(ConfigValidationError, match="model_not_ready"):
        load_config(ROOT / "config/models.json", require_ready=True)


@pytest.mark.parametrize(("field", "value"), [
    ("temperature", 1), ("temperature", True), ("max_output_tokens", 129), ("max_output_tokens", 128.0),
    ("timeout_s", 59), ("stream", True), ("message_role", "system"),
])
def test_reject_profile_changes(evaluation_inputs, field, value):
    config = load_config(evaluation_inputs[1])
    config["profile"][field] = value
    with pytest.raises(ConfigValidationError):
        validate_config(config)


def test_zero_and_timeout_accept_json_floats(evaluation_inputs):
    config = load_config(evaluation_inputs[1])
    config["profile"].update(temperature=0.0, timeout_s=60.0)
    assert validate_config(config)["profile"]["temperature"] == 0


@pytest.mark.parametrize("field", ["api_key", "Authorization", "reference_answer"])
def test_no_secret_or_gold_fields_in_public_config(evaluation_inputs, field):
    config = load_config(evaluation_inputs[1])
    config[field] = "synthetic-secret-never-a-real-key"
    with pytest.raises(ConfigValidationError) as exc:
        validate_config(config)
    assert "synthetic-secret" not in str(exc.value)


@pytest.mark.parametrize("mutation", ["duplicate_key", "duplicate_id", "mixed_provider", "missing_model", "instruction", "seed", "readiness_type"])
def test_invalid_model_identity_and_frozen_settings(evaluation_inputs, mutation):
    config = load_config(evaluation_inputs[1])
    if mutation == "duplicate_key":
        config["models"][1]["key"] = config["models"][0]["key"]
    elif mutation == "duplicate_id":
        config["models"][1]["model_id"] = config["models"][0]["model_id"]
    elif mutation == "mixed_provider":
        config["models"][0]["provider"] = "openrouter"
    elif mutation == "missing_model":
        config["models"].pop()
    elif mutation == "instruction":
        config["common_instruction"] += " changed"
    elif mutation == "seed":
        config["order_seed"] = True
    else:
        config["models"][0]["readiness"] = []
    with pytest.raises(ConfigValidationError):
        validate_config(config)


@pytest.mark.parametrize("mutation", ["parameters", "limit", "endpoints", "endpoint_parameters", "text", "source", "profile", "canonical", "env_value"])
def test_ready_requires_synthetic_capability_evidence(input_factory, mutation):
    _, path = input_factory(real=True, full=True)
    config = load_config(path, require_ready=True)
    model = config["models"][0]
    meta = model["metadata"]
    if mutation == "parameters":
        meta["model"]["supported_parameters"] = ["temperature"]
    elif mutation == "limit":
        meta["model"]["top_provider"]["max_completion_tokens"] = 64
    elif mutation == "endpoints":
        meta["endpoints"] = []
    elif mutation == "endpoint_parameters":
        meta["endpoints"][0]["supported_parameters"] = None
    elif mutation == "text":
        meta["model"]["architecture"]["output_modalities"] = ["image"]
    elif mutation == "source":
        meta["source_url"] = "https://example.invalid/"
    elif mutation == "profile":
        meta["profile_verified"] = False
    elif mutation == "canonical":
        config["models"][1]["metadata"]["model"]["canonical_slug"] = meta["model"]["canonical_slug"]
    else:
        model["api_key_env"] = "synthetic-key-value"
    with pytest.raises(ConfigValidationError):
        validate_config(config, require_ready=True)


def test_reject_duplicate_keys_and_nonfinite_json(tmp_path):
    path = tmp_path / "config.json"
    for content in ['{"models": [], "models": []}', '{"models": NaN}']:
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigValidationError):
            load_config(path)


def test_validation_returns_copy(evaluation_inputs):
    config = load_config(evaluation_inputs[1])
    original = deepcopy(config)
    copy = validate_config(config)
    copy["models"][0]["model_id"] = "mock-changed"
    assert config == original


@pytest.mark.parametrize("protocol", ["mvp-v1", "mvp-v2-non-thinking"])
@pytest.mark.parametrize("reasoning", [None, False, [], "false", {}, {"enabled": True}, {"enabled": 0},
                                      {"enabled": "false"}, {"exclude": True}, {"effort": "none"},
                                      {"enabled": False, "exclude": True}])
def test_profile_rejects_invalid_reasoning(evaluation_inputs, reasoning, protocol):
    config = load_config(evaluation_inputs[1])
    config["protocol_version"] = protocol
    config["profile"]["reasoning"] = reasoning
    with pytest.raises(ConfigValidationError, match="profile.reasoning"):
        validate_config(config)


def test_legacy_and_mock_config_keep_reasoning_optional(evaluation_inputs):
    config = load_config(evaluation_inputs[1], require_ready=True)
    assert "reasoning" not in config["profile"]
    config["profile"]["reasoning"] = {"enabled": False}
    copy = validate_config(config, require_ready=True)
    copy["profile"]["reasoning"]["enabled"] = True
    assert config["profile"]["reasoning"] == {"enabled": False}


@pytest.mark.parametrize("failure", [None, "model_parameter", "endpoint_parameter", "mandatory_model", "mandatory_endpoint"])
def test_non_thinking_requires_compatible_model_and_endpoint(input_factory, failure):
    _, path = input_factory(real=True, full=True)
    config = load_config(path)
    config["protocol_version"] = "mvp-v2-non-thinking"
    config["profile"]["reasoning"] = {"enabled": False}
    for model in config["models"]:
        meta = model["metadata"]
        meta["model"]["supported_parameters"].append("reasoning")
        meta["model"]["reasoning"] = {"mandatory": False}
        meta["endpoints"][0]["supported_parameters"].append("reasoning")
    meta = config["models"][0]["metadata"]
    if failure == "model_parameter":
        meta["model"]["supported_parameters"].remove("reasoning")
    elif failure == "endpoint_parameter":
        meta["endpoints"][0]["supported_parameters"].remove("reasoning")
    elif failure == "mandatory_model":
        meta["model"]["reasoning"]["mandatory"] = True
    elif failure == "mandatory_endpoint":
        meta["endpoints"][0]["reasoning"] = {"mandatory": True}
    if failure is None:
        assert validate_config(config, require_ready=True)["profile"]["reasoning"] == {"enabled": False}
    else:
        with pytest.raises(ConfigValidationError):
            validate_config(config, require_ready=True)


@pytest.mark.parametrize("protocol", ["mvp-v3", "", None, [], {}])
def test_unknown_protocol_is_rejected(evaluation_inputs, protocol):
    config = load_config(evaluation_inputs[1])
    config["protocol_version"] = protocol
    with pytest.raises(ConfigValidationError, match="protocol_version"):
        validate_config(config)


def test_v2_requires_explicit_disabled_reasoning(evaluation_inputs):
    config = load_config(evaluation_inputs[1])
    config["protocol_version"] = "mvp-v2-non-thinking"
    with pytest.raises(ConfigValidationError, match="profile.reasoning"):
        validate_config(config)
    config["profile"]["reasoning"] = {"enabled": False}
    assert validate_config(config, require_ready=True)["protocol_version"] == "mvp-v2-non-thinking"


def test_legacy_real_reasoning_config_is_offline_readable_but_cannot_start_a_run(input_factory):
    _, path = input_factory(real=True, full=True)
    config = load_config(path)
    config["profile"]["reasoning"] = {"enabled": False}
    for model in config["models"]:
        model["metadata"]["model"]["supported_parameters"].append("reasoning")
        model["metadata"]["endpoints"][0]["supported_parameters"].append("reasoning")
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_config(path)["protocol_version"] == "mvp-v1"
    with pytest.raises(ConfigValidationError, match="protocol_version"):
        load_config(path, require_ready=True)
