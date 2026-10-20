"""Two fixed profile probes, offline by default; no runner, retries or fallback."""

import argparse
import json
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .config import load_config
from .model_adapter import ModelRequest, ModelResponse
from .openrouter_adapter import OpenRouterAdapter, _contains_secret
from .openrouter_smoke import SMOKE_TASK
from .reporting import digest, safe_diagnostic, write_json


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/models.json"
OUTPUT_ROOT = ROOT / "logs/non-thinking-smoke"
MODEL_IDS = ("qwen/qwen3.5-9b", "google/gemini-3.1-flash-lite")


def _plan(path: Path) -> tuple[dict, bytes, list[tuple[dict, ModelRequest]]]:
    content = path.read_bytes()
    config = load_config(path)  # Smoke may probe not_ready; never changes readiness.
    if content != path.read_bytes():
        raise ValueError
    if config["profile"].get("reasoning") != {"enabled": False}:
        raise ValueError
    models = {model["model_id"]: model for model in config["models"]}
    if set(models) != set(MODEL_IDS) or any(model["provider"] != "openrouter" for model in models.values()):
        raise ValueError
    required = {"temperature", "max_tokens", "reasoning"}
    plan = []
    for model_id in MODEL_IDS:
        model = models[model_id]
        meta = model["metadata"]
        details = meta["model"]
        if not required <= set(details["supported_parameters"]) or details.get("reasoning", {}).get("mandatory") is True:
            raise ValueError
        limit = details["top_provider"]["max_completion_tokens"]
        if limit is not None and (type(limit) is not int or limit < 128):
            raise ValueError
        if not any(
            isinstance(endpoint, dict) and required <= set(endpoint.get("supported_parameters") or [])
            and endpoint.get("reasoning", {}).get("mandatory") is not True
            and (endpoint.get("max_completion_tokens") is None or type(endpoint["max_completion_tokens"]) is int and endpoint["max_completion_tokens"] >= 128)
            for endpoint in meta["endpoints"]
        ):
            raise ValueError
        profile = config["profile"]
        request = ModelRequest(provider="openrouter", model_id=model_id, prompt=config["common_instruction"] + "\n\n" + SMOKE_TASK,
                               **{key: profile[key] for key in ("temperature", "max_output_tokens", "timeout_s", "reasoning")})
        plan.append((model, request))
    return config, content, plan


def _checks(response: ModelResponse, diagnostic: dict) -> list[str]:
    """Conservative stop conditions, without changing response status or usage."""
    checks = []
    if response.status != "ok":
        checks.append("response_not_ok")
    if response.finish_reason != "stop":
        checks.append("finish_reason_unconfirmed")
    if response.returned_model_id is None:
        checks.append("returned_model_unknown")
    status = diagnostic["http_status"]
    if status is None or not 200 <= status <= 299:
        checks.append("http_not_successful")
    usage = response.usage or {}
    completion = usage.get("completion_tokens")
    if completion is None:
        checks.append("completion_usage_unknown")
    elif completion > 128:
        # This is an observation, not a claim about the visible-token budget.
        checks.append("reported_completion_exceeds_requested_max_tokens")
    reasoning = [usage[key] for key in ("reasoning_tokens", "completion_tokens_details.reasoning_tokens") if key in usage]
    if not reasoning:
        checks.append("reasoning_usage_unknown")
    elif any(count > 0 for count in reasoning):
        checks.append("reasoning_reported")
    if diagnostic["error_code"] is not None:
        checks.append("adapter_diagnostic_error")
    return checks


def _empty_result(model_id: str) -> dict:
    return {**asdict(ModelResponse(status="not_run", raw_response=None, requested_model_id=model_id)),
            "elapsed_ms": None, "http_status": None, "error_code": None, "request_attempted": False, "checks": []}


def _live(config: dict, content: bytes, plan: list, budget: str, run_id: str) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key or any(ord(character) < 33 or ord(character) > 126 for character in key):
        return {"status": "error", "error_code": "missing_or_invalid_api_key"}
    if key in content.decode("utf-8") or key in str(OUTPUT_ROOT / run_id):
        return {"status": "error", "error_code": "secret_in_public_input_or_path"}
    directory = OUTPUT_ROOT / run_id
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)  # No overwrite, resume or repeat in this run.
    result = {"purpose": "limited_profile_smoke", "run_status": "incomplete", "stop_reason": None,
              "config_sha256": digest(content), "profile": config["profile"],
              "approved_budget_usd": budget, "budget_is_enforced_by_cli": False,
              "responses": [_empty_result(model_id) for model_id in MODEL_IDS]}
    output = directory / "diagnostics.json"
    write_json(output, result)
    for index, (model, request) in enumerate(plan):
        adapter = OpenRouterAdapter(model["metadata"]["model"])
        # Persist an uncertain attempt before POST: even a killed process cannot
        # resume this run or misrepresent a potentially billable call as not_run.
        pending = {**asdict(ModelResponse(status="api_error", raw_response=None, requested_model_id=request.model_id)),
                   "elapsed_ms": None, "http_status": None, "error_code": "unfinished_attempt",
                   "request_attempted": None, "checks": ["unfinished_attempt"]}
        result["responses"][index] = pending
        write_json(output, result)
        started = time.perf_counter()
        interrupted = False
        try:
            response = adapter.generate(request)  # Exactly one invocation, never retried.
            ended = time.perf_counter()
            diagnostic = safe_diagnostic(adapter.get_diagnostic())
        except (KeyboardInterrupt, SystemExit):
            ended = time.perf_counter()
            interrupted = True
            response = ModelResponse(status="api_error", raw_response=None, requested_model_id=request.model_id)
            diagnostic = safe_diagnostic({"error_code": "interrupted"})
        except Exception:
            ended = time.perf_counter()
            response = ModelResponse(status="api_error", raw_response=None, requested_model_id=request.model_id)
            diagnostic = safe_diagnostic({"error_code": "adapter_exception"})
        if not isinstance(response, ModelResponse) or response.requested_model_id != request.model_id or _contains_secret(asdict(response), key):
            response = ModelResponse(status="api_error", raw_response=None, requested_model_id=request.model_id)
            diagnostic = safe_diagnostic({"error_code": "sensitive_response"})
        elapsed = (ended - started) * 1000 if diagnostic["request_attempted"] is True else None
        response = replace(response, latency_ms=elapsed if response.status == "ok" else None)
        checks = _checks(response, diagnostic)
        result["responses"][index] = {**asdict(response), "elapsed_ms": elapsed,
                                      **{key: diagnostic[key] for key in ("http_status", "error_code", "request_attempted")}, "checks": checks}
        if checks:
            result.update(run_status="interrupted" if interrupted else "stopped", stop_reason=checks[0])
        elif index == len(plan) - 1:
            result["run_status"] = "complete"
        write_json(output, result)  # Save first diagnostics before considering Gemini.
        if checks:
            break
    return {"status": result["run_status"], "diagnostics_path": str(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Two fixed non-thinking probes; dry-run unless --live. Never a benchmark.")
    parser.add_argument("--live", action="store_true", help="Only after owner approval: at most one POST per fixed model.")
    parser.add_argument("--approved-budget-usd", help="Record the separately approved positive USD ceiling; NOT a billing hard cap.")
    parser.add_argument("--run-id", default=uuid.uuid4().hex, help="Unique local diagnostic directory; cannot be reused.")
    args = parser.parse_args(argv)
    try:
        config, content, plan = _plan(CONFIG_PATH)
        if not args.live:
            result = {"mode": "dry_run", "maximum_posts": 2, "stop_before_second_on_error_or_unknown_usage": True,
                      "profile": config["profile"], "payloads": [OpenRouterAdapter.payload(request) for _, request in plan]}
        else:
            try:
                budget = Decimal(args.approved_budget_usd or "NaN")
                if not budget.is_finite() or budget <= 0:
                    raise ValueError
            except (InvalidOperation, ValueError):
                print(json.dumps({"status": "error", "error_code": "approved_budget_required"}))
                return 1
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
                raise ValueError
            result = _live(config, content, plan, str(budget), args.run_id)
    except (ValueError, TypeError, KeyError, OSError, AttributeError):
        result = {"status": "error", "error_code": "invalid_config_or_local_output"}
    except KeyboardInterrupt:
        result = {"status": "interrupted"}
    # Console gets only a path/summary or public dry-run payload, never raw text.
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("mode") == "dry_run" or result.get("status") == "complete" else (130 if result.get("status") == "interrupted" else 1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
