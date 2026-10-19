"""Public metadata validation is offline and never establishes live behavior."""

import json
from copy import deepcopy

import pytest

from conftest import ROOT
from kazakh_model_watch.config import ConfigValidationError, load_config, validate_config


def test_repository_real_config_has_two_distinct_not_ready_models():
    config = load_config(ROOT / "config/models.json")
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
