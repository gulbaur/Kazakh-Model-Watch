"""Small, strict public configuration; validation never loads credentials."""

import json
import re
from copy import deepcopy
from datetime import date
from pathlib import Path

from .model_adapter import validate_reasoning
from .openrouter_adapter import COMMON_INSTRUCTION


PROFILE = {"temperature": 0, "max_output_tokens": 128, "timeout_s": 60, "stream": False, "message_role": "user"}
VERSIONS = {"protocol_version": "mvp-v1", "scoring_version": "binary-v1", "normalization_version": "kk-short-v1"}


class ConfigValidationError(ValueError):
    """Messages contain field names, never input values or credentials."""


def read_json(path: str | Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ConfigValidationError("duplicate JSON field")
            result[key] = value
        return result

    def invalid(_):
        raise ConfigValidationError("non-finite JSON number")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ConfigValidationError(f"{label}: invalid fields")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{label}: non-empty string required")


def validate_config(value: dict, *, require_ready: bool = False) -> dict:
    """Validate one frozen profile and two distinct identities, without network."""
    _fields(value, [*VERSIONS, "dataset_version", "order_seed", "common_instruction", "profile", "models"], "config")
    for key, expected in VERSIONS.items():
        if value[key] != expected:
            raise ConfigValidationError(f"{key}: unsupported version")
    _text(value["dataset_version"], "dataset_version")
    if type(value["order_seed"]) is not int or value["order_seed"] != 42:
        raise ConfigValidationError("order_seed: expected 42")
    if value["common_instruction"] != COMMON_INSTRUCTION:
        raise ConfigValidationError("common_instruction: frozen instruction required")
    profile = value["profile"]
    if not isinstance(profile, dict) or set(profile) not in (set(PROFILE), set(PROFILE) | {"reasoning"}):
        raise ConfigValidationError("profile: invalid fields")
    if "reasoning" in profile:
        try:
            if profile["reasoning"] is None:
                raise ValueError
            validate_reasoning(profile["reasoning"])
        except (TypeError, ValueError):
            raise ConfigValidationError("profile.reasoning: only enabled=false is supported") from None
    required_parameters = {"temperature", "max_tokens"} | ({"reasoning"} if "reasoning" in profile else set())
    for key, expected in PROFILE.items():
        actual = value["profile"][key]
        numeric = key in {"temperature", "timeout_s"}
        valid_type = type(actual) in (int, float) if numeric else type(actual) is type(expected)
        if not valid_type or actual != expected:
            raise ConfigValidationError(f"profile.{key}: unsupported setting")
    models = value["models"]
    if not isinstance(models, list) or len(models) != 2:
        raise ConfigValidationError("models: exactly two models required")
    keys, identities = set(), set()
    for model in models:
        _fields(model, ["key", "provider", "model_id", "api_key_env", "readiness", "readiness_reason", "metadata"], "model")
        for key in ("key", "provider", "model_id"):
            _text(model[key], f"model.{key}")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", model["key"]):
            raise ConfigValidationError("model.key: invalid identifier")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", model["model_id"]):
            raise ConfigValidationError("model.model_id: invalid identifier")
        identity = (model["provider"], model["model_id"])
        if model["key"] in keys or identity in identities:
            raise ConfigValidationError("models: duplicate identity or key")
        keys.add(model["key"])
        identities.add(identity)
        if not isinstance(model["readiness"], str) or model["readiness"] not in {"ready", "not_ready"}:
            raise ConfigValidationError("model.readiness: unsupported state")
        if model["readiness"] == "not_ready":
            _text(model["readiness_reason"], "model.readiness_reason")
        elif model["readiness_reason"] is not None:
            raise ConfigValidationError("ready model must have null readiness_reason")
        if require_ready and model["readiness"] != "ready":
            raise ConfigValidationError("model_not_ready")
        if model["provider"] == "mock":
            if model["api_key_env"] is not None or model["metadata"] is not None or not model["model_id"].startswith("mock-"):
                raise ConfigValidationError("mock model must use a synthetic ID and no credentials/metadata")
        elif model["provider"] == "openrouter":
            if model["api_key_env"] != "OPENROUTER_API_KEY":
                raise ConfigValidationError("model.api_key_env: expected environment variable name")
            metadata = model["metadata"]
            if metadata is None and model["readiness"] == "not_ready":
                continue
            _fields(metadata, ["checked_on", "source_url", "endpoints_source_url", "model", "endpoints", "profile_verified"], "metadata")
            try:
                date.fromisoformat(metadata["checked_on"])
            except (TypeError, ValueError):
                raise ConfigValidationError("metadata.checked_on: ISO date required") from None
            if metadata["source_url"] != "https://openrouter.ai/api/v1/models" or metadata["endpoints_source_url"] != f"https://openrouter.ai/api/v1/models/{model['model_id']}/endpoints":
                raise ConfigValidationError("metadata: unexpected public source")
            if type(metadata["profile_verified"]) is not bool:
                raise ConfigValidationError("metadata.profile_verified: boolean required")
            details = metadata["model"]
            if not isinstance(details, dict) or details.get("id") != model["model_id"]:
                raise ConfigValidationError("metadata: model ID mismatch")
            parameters = details.get("supported_parameters")
            if not isinstance(parameters, list) or not all(isinstance(p, str) for p in parameters):
                raise ConfigValidationError("metadata: invalid supported_parameters")
            if model["readiness"] == "ready":
                endpoints = metadata["endpoints"]
                reasoning = details.get("reasoning")
                if "reasoning" in profile and isinstance(reasoning, dict) and reasoning.get("mandatory") is True:
                    raise ConfigValidationError("metadata: reasoning cannot be disabled")
                if not metadata["profile_verified"] or not required_parameters <= set(parameters):
                    raise ConfigValidationError("model profile compatibility is not verified")
                compatible = []
                if isinstance(endpoints, list):
                    for endpoint in endpoints:
                        if not isinstance(endpoint, dict):
                            continue
                        supported = endpoint.get("supported_parameters")
                        limit = endpoint.get("max_completion_tokens")
                        reasoning = endpoint.get("reasoning")
                        mandatory = isinstance(reasoning, dict) and reasoning.get("mandatory") is True
                        if (
                            isinstance(supported, list) and all(isinstance(p, str) for p in supported)
                            and required_parameters <= set(supported)
                            and not ("reasoning" in profile and mandatory)
                            and (limit is None or type(limit) is int and limit >= 128)
                        ):
                            compatible.append(endpoint)
                if not compatible:
                    raise ConfigValidationError("metadata: no compatible endpoint")
                architecture = details.get("architecture")
                if not isinstance(architecture, dict) or "text" not in (architecture.get("input_modalities") or []) or "text" not in (architecture.get("output_modalities") or []):
                    raise ConfigValidationError("metadata: text input/output required")
                top_provider = details.get("top_provider")
                if not isinstance(top_provider, dict):
                    raise ConfigValidationError("metadata: invalid top_provider")
                limit = top_provider.get("max_completion_tokens")
                if limit is not None and (type(limit) is not int or limit < 128):
                    raise ConfigValidationError("metadata: insufficient output limit")
        else:
            raise ConfigValidationError("model.provider: unsupported provider")
    if len({m["provider"] for m in models}) != 1:
        raise ConfigValidationError("models: real and mock providers cannot be mixed")
    canonical = [m["metadata"]["model"].get("canonical_slug") for m in models if m["metadata"] is not None]
    canonical = [slug for slug in canonical if isinstance(slug, str) and slug]
    if len(set(canonical)) != len(canonical):
        raise ConfigValidationError("models: aliases refer to the same canonical model")
    return deepcopy(value)


def load_config(path: str | Path, *, require_ready: bool = False) -> dict:
    return validate_config(read_json(path), require_ready=require_ready)
