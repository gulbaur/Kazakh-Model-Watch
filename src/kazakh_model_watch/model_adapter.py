"""Synchronous model contract and explicitly synthetic offline responses."""

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal


ModelStatus = Literal["ok", "api_error", "timeout", "empty_response", "truncated", "not_run"]
_STATUSES = {"ok", "api_error", "timeout", "empty_response", "truncated", "not_run"}


def _nonempty_string(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be empty")


def _number(value: object, field: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        limit = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be finite and {limit}")


def validate_reasoning(value: object) -> None:
    """Only an omitted setting or explicit disabling is supported in issue #9."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise TypeError("reasoning must be a dictionary or None")
    if set(value) != {"enabled"} or value["enabled"] is not False:
        raise ValueError("reasoning supports only enabled=false")


@dataclass(frozen=True)
class ModelRequest:
    """One complete prompt; defaults follow the MVP generation profile."""

    provider: str
    model_id: str
    prompt: str
    temperature: float = 0.0
    max_output_tokens: int = 128
    timeout_s: float = 60.0
    reasoning: dict[str, bool] | None = None

    def __post_init__(self) -> None:
        for field in ("provider", "model_id", "prompt"):
            _nonempty_string(getattr(self, field), field)
        _number(self.temperature, "temperature")
        _number(self.timeout_s, "timeout_s", positive=True)
        if isinstance(self.max_output_tokens, bool) or not isinstance(self.max_output_tokens, int):
            raise TypeError("max_output_tokens must be an integer")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        validate_reasoning(self.reasoning)
        object.__setattr__(self, "reasoning", deepcopy(self.reasoning))


@dataclass(frozen=True)
class ModelResponse:
    """Visible text and provider metadata, without guessed usage or timing.

    Unknown fields remain None. An ok response has non-whitespace visible text;
    an unsuccessful response has no successful-attempt latency. Real latency
    measurement belongs to the future runner, as specified by MVP_SPEC.md.
    """

    status: ModelStatus
    raw_response: str | None
    requested_model_id: str
    returned_model_id: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        _nonempty_string(self.status, "status")
        if self.status not in _STATUSES:
            raise ValueError("status is not supported by the MVP protocol")
        _nonempty_string(self.requested_model_id, "requested_model_id")
        for field in ("returned_model_id", "request_id", "finish_reason"):
            value = getattr(self, field)
            if value is not None:
                _nonempty_string(value, field)
        if self.raw_response is not None and not isinstance(self.raw_response, str):
            raise TypeError("raw_response must be a string or None")
        has_text = self.raw_response is not None and bool(self.raw_response.strip())
        if self.status == "ok" and not has_text:
            raise ValueError("raw_response must contain visible text for status ok")
        if self.status == "empty_response" and has_text:
            raise ValueError("raw_response must be empty for status empty_response")
        if self.status == "not_run" and self.raw_response is not None:
            raise ValueError("raw_response must be None for status not_run")
        if self.status == "not_run" and self.usage is not None:
            raise ValueError("usage must be None for status not_run")
        if self.latency_ms is not None:
            _number(self.latency_ms, "latency_ms")
            if self.status != "ok":
                raise ValueError("latency_ms must be None without a successful response")
        if self.usage is not None:
            if not isinstance(self.usage, dict):
                raise TypeError("usage must be a dictionary of token counters or None")
            if not self.usage:
                raise ValueError("usage must contain counters; use None if unknown")
            for key, value in self.usage.items():
                _nonempty_string(key, "usage key")
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError("usage values must be integer token counters")
                if value < 0:
                    raise ValueError("usage values must be non-negative")


class ModelAdapter(ABC):
    """An adapter performs one synchronous invocation, without runner retries."""

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Return text/metadata or a typed failure status for one model request."""
        raise NotImplementedError

    def get_diagnostic(self) -> dict[str, str | int | float | bool | None]:
        """Optional safe metadata for the last invocation; never raw errors.

        Existing adapters need not override this. A runner must not infer HTTP
        failures or retry delays from response text when metadata is absent.
        """
        return {}


class MockAdapter(ModelAdapter):
    """Return a preset synthetic response; never call a provider or measure time.

    Only provider="mock" is accepted. A future real benchmark must exclude this
    provider and its fixture values. Each call returns a separate response copy.
    """

    def __init__(self, response: ModelResponse) -> None:
        if not isinstance(response, ModelResponse):
            raise TypeError("response must be a ModelResponse")
        self._response = deepcopy(response)

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        if request.provider != "mock":
            raise ValueError("MockAdapter requires provider='mock'")
        if request.model_id != self._response.requested_model_id:
            raise ValueError("request model_id must match the preset requested_model_id")
        return deepcopy(self._response)
