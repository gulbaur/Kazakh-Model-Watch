"""One OpenRouter invocation with explicit capabilities and safe diagnostics."""

import os
from copy import deepcopy
from typing import Any

import httpx

from .model_adapter import ModelAdapter, ModelRequest, ModelResponse


CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
COMMON_INSTRUCTION = (
    "Тапсырманы орындаңыз. Жауапты сұралған форматта ғана жазыңыз. "
    "Түсіндірме қоспаңыз."
)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Invalid text metadata")
    return value


def _token_usage(value: Any) -> dict[str, int] | None:
    """Keep reported token counters, flattening nested token-detail keys.

    Cost and non-token metadata are not converted into token counts or prices.
    Missing counters remain unknown; totals are never derived.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Invalid usage metadata")
    result = {}
    for key, count in value.items():
        if not isinstance(key, str):
            raise ValueError("Invalid usage key")
        if key in {"prompt_tokens_details", "completion_tokens_details"}:
            if count is None:
                continue
            if not isinstance(count, dict):
                raise ValueError("Invalid usage details")
            for detail, detail_count in count.items():
                if not isinstance(detail, str):
                    raise ValueError("Invalid usage detail key")
                if detail.endswith("_tokens") and detail_count is not None:
                    if type(detail_count) is not int or detail_count < 0:
                        raise ValueError("Invalid token counter")
                    result[f"{key}.{detail}"] = detail_count
        elif key.endswith("_tokens") and count is not None:
            if type(count) is not int or count < 0:
                raise ValueError("Invalid token counter")
            result[key] = count
    return result or None


def _contains_secret(value: Any, secret: str) -> bool:
    if isinstance(value, str):
        return secret in value
    if isinstance(value, dict):
        return any(_contains_secret(key, secret) or _contains_secret(item, secret) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_secret(item, secret) for item in value)
    return False


class OpenRouterAdapter(ModelAdapter):
    """Send one POST, without retries, redirects, or model/provider fallback.

    model_metadata is a record from the public Models API. Supply a reviewed
    snapshot to avoid extra HTTP calls during generation. ModelRequest.prompt
    already contains the common instruction and task; it is sent unchanged.
    The last diagnostic contains only static error codes and HTTP status.
    """

    def __init__(self, model_metadata: dict[str, Any], *, transport: httpx.BaseTransport | None = None) -> None:
        if not isinstance(model_metadata, dict):
            raise TypeError("model_metadata must be a Models API object")
        self._model_metadata = deepcopy(model_metadata)
        self._transport = transport
        self.last_diagnostic: dict[str, str | int | bool | None] = {}

    def _failure(
        self, request: ModelRequest, code: str, *, status: str = "api_error",
        raw_response: str | None = None, **metadata: Any,
    ) -> ModelResponse:
        self.last_diagnostic["error_code"] = code
        return ModelResponse(status=status, raw_response=raw_response, requested_model_id=request.model_id, **metadata)

    def _supported(self, request: ModelRequest) -> bool:
        parameters = self._model_metadata.get("supported_parameters")
        if self._model_metadata.get("id") != request.model_id:
            return False
        if not isinstance(parameters, list) or not all(isinstance(item, str) for item in parameters):
            return False
        if not {"temperature", "max_tokens"}.issubset(parameters):
            return False
        top_provider = self._model_metadata.get("top_provider", {})
        if not isinstance(top_provider, dict):
            return False
        limit = top_provider.get("max_completion_tokens")
        return limit is None or (type(limit) is int and limit >= request.max_output_tokens)

    @staticmethod
    def payload(request: ModelRequest) -> dict[str, Any]:
        """Whitelist request fields; gold data and credentials cannot enter JSON."""
        return {
            "model": request.model_id,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "stream": False,
            "messages": [{"role": "user", "content": request.prompt}],
            "provider": {"require_parameters": True, "allow_fallbacks": False},
        }

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        self.last_diagnostic = {"error_code": None, "http_status": None, "request_attempted": False}
        if request.provider != "openrouter":
            return self._failure(request, "invalid_provider")
        if request.temperature != 0 or request.max_output_tokens != 128 or request.timeout_s != 60:
            return self._failure(request, "unsupported_mvp_profile")
        if not self._supported(request):
            return self._failure(request, "unsupported_model_parameters")

        # The adapter never reads .env. Only the CLI may load it after --live.
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key or not api_key.strip():
            return self._failure(request, "missing_api_key")
        if any(ord(character) < 33 or ord(character) > 126 for character in api_key):
            return self._failure(request, "invalid_api_key")

        try:
            transport = self._transport if self._transport is not None else httpx.HTTPTransport(retries=0)
            with httpx.Client(transport=transport, follow_redirects=False, trust_env=False) as client:
                self.last_diagnostic["request_attempted"] = True
                response = client.post(
                    CHAT_URL,
                    json=self.payload(request),
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=request.timeout_s,
                )
        except httpx.TimeoutException:
            return self._failure(request, "request_timeout", status="timeout")
        except httpx.HTTPError:
            return self._failure(request, "network_error")
        except (ValueError, UnicodeError):
            return self._failure(request, "invalid_http_request")

        self.last_diagnostic["http_status"] = response.status_code
        try:
            body = response.json()
        except ValueError:
            code = "http_error" if not response.is_success else "malformed_response"
            return self._failure(request, code)
        # Fail closed if a service reflects a credential into any response field.
        if _contains_secret(body, api_key):
            return self._failure(request, "sensitive_response")
        if not isinstance(body, dict):
            return self._failure(request, "http_error" if not response.is_success else "malformed_response")

        metadata: dict[str, Any] = {}
        raw_response = None
        try:
            metadata["returned_model_id"] = _optional_text(body.get("model"))
            metadata["request_id"] = _optional_text(body.get("id"))
            metadata["usage"] = _token_usage(body.get("usage"))
            if not response.is_success:
                return self._failure(request, "http_error", **metadata)
            if body.get("error") is not None:
                return self._failure(request, "provider_error", **metadata)
            choices = body.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise ValueError("Invalid choices")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict) or "content" not in message:
                raise ValueError("Invalid message")
            if message.get("role", "assistant") != "assistant":
                raise ValueError("Invalid message role")
            raw_response = message["content"]
            if raw_response is not None and not isinstance(raw_response, str):
                raise ValueError("Invalid content")
            finish_reason = _optional_text(choice.get("finish_reason"))
            returned_model = metadata["returned_model_id"]
            allowed_ids = {request.model_id}
            canonical = self._model_metadata.get("canonical_slug")
            if isinstance(canonical, str):
                allowed_ids.add(canonical)
            if returned_model is not None and returned_model not in allowed_ids:
                status = "api_error"
                self.last_diagnostic["error_code"] = "unexpected_model_id"
            elif finish_reason == "content_filter":
                status = "api_error"
                self.last_diagnostic["error_code"] = "content_filtered"
            elif finish_reason == "length":
                status = "truncated"
            elif finish_reason not in {None, "stop"}:
                status = "api_error"
                self.last_diagnostic["error_code"] = "unexpected_finish_reason"
            elif raw_response is None or not raw_response.strip():
                status = "empty_response"
            else:
                status = "ok"
            return ModelResponse(
                status=status, raw_response=raw_response, requested_model_id=request.model_id,
                finish_reason=finish_reason, **metadata,
            )
        except (ValueError, TypeError):
            # Never serialize bodies, headers, exception text, or chained errors.
            safe_text = raw_response if isinstance(raw_response, str) else None
            return self._failure(request, "malformed_response", raw_response=safe_text, **metadata)
