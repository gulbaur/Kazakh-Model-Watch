"""Sequential evaluation with preflight gates, bounded retry and local snapshots."""

import os
import platform
import re
import subprocess
import time
import unicodedata
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .config import load_config
from .dataset import load_dataset
from .model_adapter import ModelAdapter, ModelRequest, ModelResponse, MockAdapter
from .openrouter_adapter import OpenRouterAdapter, _contains_secret
from .reporting import RunError, apply_attempt, atomic_write, digest, missing_response, report, retry_delay, safe_diagnostic, schedule, write_json


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_state() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL))
        return {"git_commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "working_tree_dirty": None}


def _returned_ids(manifest: dict, rows: list[dict]) -> None:
    for model in manifest["models"]:
        model["returned_model_ids"] = sorted({attempt["returned_model_id"] for row in rows if row["model_key"] == model["key"] for attempt in row["attempts"] if attempt["returned_model_id"] is not None})
    manifest["actual_adapter_attempt_order"] = [
        {"item_id": row["item_id"], "model_key": row["model_key"], "attempt_number": attempt["attempt_number"]}
        for row in rows for attempt in row["attempts"]
    ]


def _dependency_versions() -> dict:
    versions = {}
    for name in ("kazakh-model-watch", "httpx", "python-dotenv", "setuptools"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            # Build isolation can keep setuptools out of the runtime venv.
            versions[name] = None
    return versions


def prepare(dataset_path: str | Path, config_path: str | Path, *, mock: bool = False, live: bool = False):
    if type(mock) is not bool or type(live) is not bool or (mock and live):
        raise RunError("invalid_run_mode")
    dataset_path, config_path = Path(dataset_path), Path(config_path)
    dataset_bytes, config_bytes = dataset_path.read_bytes(), config_path.read_bytes()
    # Final validation precedes adapters, credential lookups, results and network.
    items = load_dataset(dataset_path, final=not mock)
    config = load_config(config_path, require_ready=True)
    if dataset_path.read_bytes() != dataset_bytes or config_path.read_bytes() != config_bytes:
        raise RunError("inputs_changed_during_preflight")
    if any((m["provider"] == "mock") != mock for m in config["models"]):
        raise RunError("provider_mode_mismatch")
    if not mock and not live:
        raise RunError("live_opt_in_required")
    secrets = []
    if live:
        for model in config["models"]:
            key = os.environ.get(model["api_key_env"])
            if not key or any(ord(c) < 33 or ord(c) > 126 for c in key):
                raise RunError("missing_or_invalid_api_key")
            secrets.append(key)
        if any(key in dataset_bytes.decode("utf-8") or key in config_bytes.decode("utf-8") for key in secrets):
            raise RunError("secret_in_public_inputs")
    return items, config, dataset_bytes, config_bytes, secrets


def run_benchmark(
    dataset_path: str | Path, config_path: str | Path, *, output_root: str | Path = "results",
    run_id: str | None = None, mock: bool = False, live: bool = False,
    adapters: dict[str, ModelAdapter] | None = None, clock=time.perf_counter, sleep=time.sleep,
) -> Path:
    items, config, dataset_bytes, config_bytes, secrets = prepare(dataset_path, config_path, mock=mock, live=live)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise RunError("invalid_run_id")
    if adapters is None:
        adapters = {
            model["key"]: MockAdapter(ModelResponse(status="ok", raw_response="synthetic offline response", requested_model_id=model["model_id"]))
            if mock else OpenRouterAdapter(model["metadata"]["model"])
            for model in config["models"]
        }
    if set(adapters) != {m["key"] for m in config["models"]} or not all(isinstance(a, ModelAdapter) for a in adapters.values()):
        raise RunError("adapter_mapping_mismatch")
    if live and any(isinstance(a, MockAdapter) for a in adapters.values()):
        raise RunError("mock_adapter_in_live_run")
    directory = Path(output_root) / run_id
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir(exist_ok=False)
    except FileExistsError:
        raise RunError("run_id_already_exists") from None
    plan = schedule(items, config["models"], config["order_seed"])
    rows = [missing_response(run_id, item, model, config["common_instruction"]) for item, model in plan]
    manifest = {
        "run_id": run_id, "run_status": "incomplete", "mode": "mock" if mock else "live",
        "started_at": _utc(), "finished_at": None, "interrupted_at": None,
        **{key: config[key] for key in ("protocol_version", "dataset_version", "scoring_version", "normalization_version", "order_seed", "profile")},
        "input_hashes": {"dataset": digest(dataset_bytes), "config": digest(config_bytes), "instruction": digest(config["common_instruction"].encode("utf-8"))},
        "pair_order": [{"item_id": item["id"], "model_key": model["key"]} for item, model in plan],
        "actual_adapter_attempt_order": [],
        "item_order": list(dict.fromkeys(item["id"] for item, _ in plan)),
        "models": [{"key": m["key"], "provider": m["provider"], "requested_model_id": m["model_id"], "returned_model_ids": []} for m in config["models"]],
        "python_version": platform.python_version(), "unicode_version": unicodedata.unidata_version,
        "dependencies": _dependency_versions(),
        "os": platform.platform(), "runner": "sequential-python", **_git_state(),
        "limitations": ["Mutable aliases may change; returned IDs are recorded.", "Provider caching and unspecified API defaults are not overridden.",
                        "Reasoning disable is requested, not guaranteed by upstream; this profile is an issue #9 proposal, not an approved mvp-v1 benchmark profile."
                        if "reasoning" in config["profile"] else "Reasoning defaults are not overridden.",
                        "HTTPX timeout is per network phase; latency includes network.", "Mock outputs and timings are not real model measurements." if mock else "Profile compatibility must be approved before this run."],
    }
    write_json(directory / "manifest.json", manifest)
    atomic_write(directory / "dataset.jsonl", dataset_bytes)
    atomic_write(directory / "config.json", config_bytes)
    write_json(directory / "responses.json", rows)
    row, pair_started = None, None
    try:
        for row, (item, model) in zip(rows, plan):
            adapter = adapters[model["key"]]
            request = ModelRequest(provider=model["provider"], model_id=model["model_id"], prompt=row["prompt"],
                                   **{key: config["profile"][key] for key in ("temperature", "max_output_tokens", "timeout_s")},
                                   reasoning=config["profile"].get("reasoning"))
            pair_started = clock() if live else None
            for number in (1, 2):
                started = clock() if live else None
                interrupted = False
                try:
                    response = adapter.generate(request)
                    ended = clock() if live else None
                    if not isinstance(response, ModelResponse):
                        raise TypeError("invalid response")
                    diagnostic = safe_diagnostic(adapter.get_diagnostic())
                except (KeyboardInterrupt, SystemExit):
                    ended = clock() if live else None
                    interrupted = True
                    response = ModelResponse(status="api_error", raw_response=None, requested_model_id=model["model_id"])
                    diagnostic = safe_diagnostic({"error_code": "interrupted"})
                except Exception:
                    ended = clock() if live else None
                    response = ModelResponse(status="api_error", raw_response=None, requested_model_id=model["model_id"])
                    diagnostic = safe_diagnostic({"error_code": "adapter_exception"})
                elapsed = max(0.0, (ended - started) * 1000) if live else None
                if response.status == "not_run":
                    response = ModelResponse(status="api_error", raw_response=None, requested_model_id=model["model_id"])
                    diagnostic = safe_diagnostic({"error_code": "invalid_adapter_response"})
                if response.requested_model_id != model["model_id"]:
                    response = ModelResponse(status="api_error", raw_response=None, requested_model_id=model["model_id"])
                    diagnostic = safe_diagnostic({"error_code": "unexpected_model_id"})
                if any(_contains_secret(asdict(response), secret) for secret in secrets):
                    response = ModelResponse(status="api_error", raw_response=None, requested_model_id=model["model_id"])
                    diagnostic = safe_diagnostic({"error_code": "sensitive_response"})
                response = replace(response, latency_ms=elapsed if live and response.status == "ok" else None)
                delay = retry_delay(response, diagnostic) if number == 1 and not interrupted else None
                attempt = {**asdict(response), **diagnostic, "attempt_number": number, "elapsed_ms": elapsed,
                           "retry_delay_s": delay, "retry_wait_completed": False if delay is not None else None,
                           "cost": None, "cost_unknown_reason": "billing_not_collected"}
                apply_attempt(row, attempt, item, max(0.0, (clock() - pair_started) * 1000) if live else None)
                write_json(directory / "responses.json", rows)
                _returned_ids(manifest, rows)
                write_json(directory / "manifest.json", manifest)
                if interrupted:
                    raise KeyboardInterrupt
                if delay is None:
                    break
                sleep(delay)
                attempt["retry_wait_completed"] = True
                write_json(directory / "responses.json", rows)
        _returned_ids(manifest, rows)
        manifest["run_status"] = "complete"
        manifest["finished_at"] = _utc()
        write_json(directory / "manifest.json", manifest)
        report(directory)
    except BaseException:
        manifest["run_status"] = "incomplete"
        manifest["finished_at"] = None
        manifest["interrupted_at"] = _utc()
        _returned_ids(manifest, rows)
        if live and row is not None and pair_started is not None:
            row["total_elapsed_ms"] = max(0.0, (clock() - pair_started) * 1000)
            write_json(directory / "responses.json", rows)
        write_json(directory / "manifest.json", manifest)
        # A failed report must not mask the original interruption or I/O failure.
        try:
            report(directory)
        except Exception:
            pass
        raise
    return directory
