"""Review a fixed request offline; explicitly opt in to one paid invocation."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .model_adapter import ModelRequest, ModelResponse
from .openrouter_adapter import COMMON_INSTRUCTION, OpenRouterAdapter


SMOKE_MODEL_ID = "qwen/qwen3.5-9b"
SMOKE_TASK = "Екіге үшті қосқанда қанша болады? Жауапты тек бүтін санмен жазыңыз."
METADATA_PATH = Path(__file__).resolve().parents[2] / "config" / "openrouter_model.json"


def _preflight_failure(request: ModelRequest, code: str) -> int:
    result = asdict(ModelResponse(status="api_error", raw_response=None, requested_model_id=request.model_id))
    result.update(elapsed_ms=None, http_status=None, error_code=code)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One OpenRouter smoke request; offline unless --live.")
    parser.add_argument("--live", action="store_true", help="Explicitly allow one billable POST, without retries.")
    parser.add_argument("--dotenv", action="store_true", help="With --live only: load local .env without overriding environment.")
    args = parser.parse_args(argv)
    request = ModelRequest(
        provider="openrouter", model_id=SMOKE_MODEL_ID,
        prompt=f"{COMMON_INSTRUCTION}\n\n{SMOKE_TASK}",
    )
    if not args.live:
        print(json.dumps({"mode": "dry_run", "payload": OpenRouterAdapter.payload(request)}, ensure_ascii=False, indent=2))
        return 0

    try:
        snapshot = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        model_metadata = snapshot["model"]
        if snapshot.get("source_url") != "https://openrouter.ai/api/v1/models":
            raise ValueError("Invalid metadata source")
        adapter = OpenRouterAdapter(model_metadata)
    except (OSError, ValueError, KeyError, TypeError):
        return _preflight_failure(request, "invalid_model_metadata")
    if args.dotenv:
        from dotenv import load_dotenv

        try:
            load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        except (OSError, UnicodeError, ValueError):
            return _preflight_failure(request, "dotenv_load_failed")

    started = time.perf_counter()
    response = adapter.generate(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    result = asdict(response)
    result["elapsed_ms"] = elapsed_ms if adapter.last_diagnostic["request_attempted"] else None
    result["http_status"] = adapter.last_diagnostic["http_status"]
    result["error_code"] = adapter.last_diagnostic["error_code"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if response.status == "ok" else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
