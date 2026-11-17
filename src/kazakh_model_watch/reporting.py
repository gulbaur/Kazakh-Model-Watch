"""Atomic local artifacts and deterministic offline reports; no provider calls."""

import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

from .config import load_config, read_json
from .dataset import load_dataset
from .model_adapter import ModelResponse
from .normalization import normalize_answer
from .openrouter_adapter import REASONING_DETAIL_TYPES, upstream_provider_name
from .scoring import score_response


STATUSES = ("ok", "api_error", "timeout", "empty_response", "truncated", "not_run")
RESPONSE_FIELDS = tuple(field.name for field in fields(ModelResponse))
ERROR_CODES = {
    "invalid_provider", "unsupported_mvp_profile", "unsupported_model_parameters",
    "missing_api_key", "invalid_api_key", "request_timeout", "network_error",
    "invalid_http_request", "http_error", "malformed_response", "sensitive_response",
    "provider_error", "unexpected_model_id", "content_filtered", "unexpected_finish_reason",
    "reasoning_not_disabled",
    "adapter_exception", "invalid_adapter_response", "interrupted", "api_error", "timeout", "empty_response", "truncated", "not_run",
}


class RunError(ValueError):
    """Static diagnostic code safe to expose at the CLI boundary."""


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def atomic_write(path: str | Path, content: bytes) -> None:
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def schedule(items: list[dict], models: list[dict], seed: int = 42) -> list[tuple[dict, dict]]:
    ordered = list(items)
    random.Random(seed).shuffle(ordered)
    return [(item, model) for index, item in enumerate(ordered) for model in (models[index % len(models):] + models[:index % len(models)])]


def missing_response(run_id: str, item: dict, model: dict, instruction: str) -> dict:
    response = asdict(ModelResponse(status="not_run", raw_response=None, requested_model_id=model["model_id"]))
    return {
        "run_id": run_id, "item_id": item["id"], "category": item["category"],
        "model_key": model["key"], "provider": model["provider"],
        "prompt": instruction + "\n\n" + item["prompt"], **response,
        "normalized_response": None, "exact_match": 0, "correct": 0,
        "total_elapsed_ms": None, "retries": 0, "attempts": [],
        "error": {"error_code": "not_run", "http_status": None},
        "cost": None, "cost_method": None, "currency": None, "cost_unknown_reason": "not_run",
    }


def safe_diagnostic(value) -> dict:
    value = value if isinstance(value, dict) else {}
    status = value.get("http_status")
    code = value.get("error_code")
    delay = value.get("retry_after_s")
    attempted = value.get("request_attempted")
    return {
        "http_status": status if type(status) is int and 100 <= status <= 599 else None,
        "error_code": code if isinstance(code, str) and code in ERROR_CODES else None,
        "retry_after_s": delay if type(delay) in (int, float) and math.isfinite(delay) and delay >= 0 else None,
        "request_attempted": attempted if type(attempted) is bool else None,
        "upstream_provider": upstream_provider_name(value.get("upstream_provider")),
    }


_EVIDENCE_FIELDS = ("reasoning", "reasoning_content", "reasoning_details")
_EVIDENCE_KINDS = {"null", "string", "array", "object", "other"}
_DETAIL_LABELS = (*REASONING_DETAIL_TYPES, "unknown")


def safe_reasoning_evidence(value) -> dict | None:
    """Rebuild evidence from fixed labels and booleans; reject anything else.

    Evidence describes field shapes only. It never proves or disproves internal
    reasoning, and free text from a provider or a saved file cannot pass here.
    """
    if value is None:
        return None
    expected = {*_EVIDENCE_FIELDS, "detail_types", "detail_text_nonempty", "detail_data_nonempty"}
    if not isinstance(value, dict) or set(value) != expected:
        raise RunError("invalid_reasoning_evidence")
    result = {}
    for key in _EVIDENCE_FIELDS:
        field = value[key]
        if (not isinstance(field, dict) or set(field) != {"present", "type", "nonempty"}
                or type(field["present"]) is not bool or type(field["nonempty"]) is not bool
                or not isinstance(field["type"], str) or field["type"] not in _EVIDENCE_KINDS):
            raise RunError("invalid_reasoning_evidence")
        result[key] = {"present": field["present"], "type": field["type"], "nonempty": field["nonempty"]}
    types = value["detail_types"]
    if (not isinstance(types, list) or not all(isinstance(t, str) and t in _DETAIL_LABELS for t in types)
            or types != sorted(set(types))):
        raise RunError("invalid_reasoning_evidence")
    result["detail_types"] = list(types)
    for key in ("detail_text_nonempty", "detail_data_nonempty"):
        if type(value[key]) is not bool:
            raise RunError("invalid_reasoning_evidence")
        result[key] = value[key]
    return result


def upstream_providers(rows: list[dict], model_key: str) -> list[str]:
    return sorted({attempt["upstream_provider"] for row in rows if row["model_key"] == model_key
                   for attempt in row["attempts"] if attempt.get("upstream_provider") is not None})


def retry_delay(response: ModelResponse, diagnostic: dict) -> float | None:
    """Shared protocol rule for execution and offline attempt-chain validation."""
    diagnostic = safe_diagnostic(diagnostic)
    if response.finish_reason == "content_filter" or diagnostic["error_code"] in {"content_filtered", "sensitive_response", "unexpected_model_id"}:
        return None
    retry = response.status == "timeout" or (
        response.status == "api_error" and (
            diagnostic["error_code"] == "network_error" or diagnostic["http_status"] == 429
            or (diagnostic["http_status"] is not None and 500 <= diagnostic["http_status"] <= 599)
        )
    )
    if not retry:
        return None
    delay = diagnostic["retry_after_s"]
    return None if delay is not None and delay > 60 else max(2.0, delay or 0.0)


def apply_attempt(row: dict, attempt: dict, item: dict, total_elapsed_ms) -> None:
    row["attempts"].append(attempt)
    row.update({key: attempt[key] for key in RESPONSE_FIELDS})
    row["total_elapsed_ms"] = total_elapsed_ms
    row["retries"] = len(row["attempts"]) - 1
    row["normalized_response"] = normalize_answer(row["raw_response"]) if row["raw_response"] is not None else None
    row.update(score_response(row["raw_response"], item["reference_answer"], item["accepted_answers"], status=row["status"]))
    row["error"] = None if row["status"] == "ok" else {
        "error_code": attempt["error_code"] or row["status"], "http_status": attempt["http_status"],
    }
    row["cost_unknown_reason"] = "billing_not_collected"


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    latency = [r["latency_ms"] for r in rows if r["status"] == "ok" and r["latency_ms"] is not None]
    return {
        "n": n, "correct": sum(r["correct"] for r in rows), "exact_match": sum(r["exact_match"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / n if n else None,
        "exact_match_rate": sum(r["exact_match"] for r in rows) / n if n else None,
        "coverage": sum(r["status"] == "ok" for r in rows) / n if n else None,
        "errors": sum(r["status"] != "ok" for r in rows),
        "status_counts": {status: sum(r["status"] == status for r in rows) for status in STATUSES},
        "retry_count": sum(r["retries"] for r in rows), "latency_sample_count": len(latency),
        "latency_mean_ms": statistics.mean(latency) if latency else None,
        "latency_median_ms": statistics.median(latency) if latency else None,
        "cost": None, "cost_unknown_reason": "billing_not_collected",
    }


def aggregate(rows: list[dict], config: dict) -> list[dict]:
    result = []
    for model in config["models"]:
        subset = [r for r in rows if r["model_key"] == model["key"]]
        metrics = {"model_key": model["key"], "provider": model["provider"], "model_id": model["model_id"], **_metrics(subset)}
        metrics["categories"] = {
            category: _metrics([r for r in subset if r["category"] == category])
            for category in sorted({r["category"] for r in subset})
        }
        result.append(metrics)
    return result


def _csv(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def _upstream_line(providers: dict[str, list[str]]) -> str:
    if not any(providers.values()):
        return "Upstream providers: unknown (not reported by the API)."
    return "Upstream providers: " + "; ".join(
        f"{key}: {', '.join(names) if names else 'unknown'}" for key, names in providers.items()
    ) + ". Different upstreams may differ in serving setup; see attempts in responses.json."


def _leaderboard(manifest: dict, metrics: list[dict], providers: dict[str, list[str]]) -> str:
    label = "OFFLINE SYNTHETIC — NOT A MODEL BENCHMARK" if manifest["mode"] == "mock" else "Benchmark run"
    lines = [f"# {label}", "", f"Run `{manifest['run_id']}`; status **{manifest['run_status']}**; started {manifest['started_at']}.", "",
             "| Rank | Provider/model | Correct | Accuracy | Exact match | Coverage | Errors | Mean ms | Median ms | Latency N | Cost |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    ordered = sorted(metrics, key=lambda m: (-m["accuracy"], m["provider"], m["model_id"]))
    previous, rank = None, 0
    def number(value):
        return "N/A" if value is None else f"{value:.2f}"
    for index, model in enumerate(ordered, 1):
        if model["accuracy"] != previous:
            rank = index
        previous = model["accuracy"]
        lines.append(f"| {rank} | {model['provider']}/{model['model_id']} | {model['correct']}/{model['n']} | {model['accuracy']:.2%} | {model['exact_match_rate']:.2%} | {model['coverage']:.2%} | {model['errors']} | {number(model['latency_mean_ms'])} | {number(model['latency_median_ms'])} | {model['latency_sample_count']} | N/A |")
    lines += ["", "## Categories", "", "| Provider/model | Category | Correct | Accuracy | Exact match | Coverage |", "| --- | --- | --- | --- | --- | --- |"]
    for model in ordered:
        for category, values in model["categories"].items():
            lines.append(f"| {model['provider']}/{model['model_id']} | {category} | {values['correct']}/{values['n']} | {values['accuracy']:.2%} | {values['exact_match_rate']:.2%} | {values['coverage']:.2%} |")
    lines += ["", f"Versions: {manifest['protocol_version']}, {manifest['dataset_version']}, {manifest['scoring_version']}, {manifest['normalization_version']}.",
              "Profile: " + json.dumps(manifest["profile"], ensure_ascii=False, sort_keys=True) + ".",
              _upstream_line(providers),
              "Small fixed dataset; latency includes network and is not pure inference speed. Mutable aliases and provider defaults limit reproducibility.",
              "Cost is unknown; no prices or token counts have been estimated.",
              "", "Artifacts: [manifest](manifest.json), [dataset](dataset.jsonl), [config](config.json), [responses JSON](responses.json), [responses CSV](responses.csv), [metrics JSON](metrics.json), [metrics CSV](metrics.csv).", ""]
    return "\n".join(lines)


def _duration(value):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise RunError("invalid_saved_duration")


def report(run_dir: str | Path) -> dict:
    """Reconstruct missing pairs and scores solely from validated local artifacts."""
    directory = Path(run_dir)
    manifest = read_json(directory / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("mode") not in {"live", "mock"} or manifest.get("run_status") not in {"complete", "incomplete"}:
        raise RunError("invalid_manifest")
    config = load_config(directory / "config.json")
    items = load_dataset(directory / "dataset.jsonl", final=manifest["mode"] == "live")
    for key in ("protocol_version", "dataset_version", "scoring_version", "normalization_version", "order_seed", "profile"):
        if manifest.get(key) != config[key]:
            raise RunError("manifest_config_mismatch")
    if any((m["provider"] == "mock") != (manifest["mode"] == "mock") for m in config["models"]):
        raise RunError("saved_mode_mismatch")
    hashes = {"dataset": digest((directory / "dataset.jsonl").read_bytes()), "config": digest((directory / "config.json").read_bytes()), "instruction": digest(config["common_instruction"].encode("utf-8"))}
    if manifest.get("input_hashes") != hashes:
        raise RunError("input_hash_mismatch")
    plan = schedule(items, config["models"], config["order_seed"])
    expected_order = [{"item_id": item["id"], "model_key": model["key"]} for item, model in plan]
    if manifest.get("pair_order") != expected_order:
        raise RunError("saved_order_mismatch")
    raw_rows = read_json(directory / "responses.json")
    if not isinstance(raw_rows, list):
        raise RunError("invalid_saved_responses")
    saved = {}
    expected = {(item["id"], model["key"]) for item, model in plan}
    for row in raw_rows:
        if not isinstance(row, dict):
            raise RunError("invalid_saved_response")
        if not isinstance(row.get("item_id"), str) or not isinstance(row.get("model_key"), str):
            raise RunError("invalid_saved_response")
        key = (row.get("item_id"), row.get("model_key"))
        if key not in expected or key in saved:
            raise RunError("duplicate_or_unknown_pair")
        saved[key] = row
    rows = []
    for item, model in plan:
        row = missing_response(manifest["run_id"], item, model, config["common_instruction"])
        original = saved.get((item["id"], model["key"]), row)
        for key in ("run_id", "provider", "requested_model_id", "category", "prompt"):
            if original.get(key) != row[key]:
                raise RunError("saved_identity_mismatch")
        attempts = original.get("attempts")
        if not isinstance(attempts, list) or len(attempts) > 2:
            raise RunError("invalid_attempts")
        if not attempts and original.get("status") != "not_run":
            raise RunError("response_without_attempt")
        _duration(original.get("total_elapsed_ms"))
        if manifest["mode"] == "mock" and original.get("total_elapsed_ms") is not None:
            raise RunError("mock_timing_is_not_model_latency")
        for index, attempt in enumerate(attempts, 1):
            if not isinstance(attempt, dict):
                raise RunError("invalid_attempt")
            try:
                response = ModelResponse(**{key: attempt[key] for key in RESPONSE_FIELDS})
            except (KeyError, TypeError, ValueError):
                raise RunError("invalid_saved_model_response") from None
            if response.requested_model_id != model["model_id"] or response.status == "not_run" or (index == 2 and row["status"] == "ok"):
                raise RunError("invalid_attempt_identity_or_order")
            _duration(attempt.get("elapsed_ms"))
            _duration(attempt.get("retry_delay_s"))
            if attempt.get("retry_wait_completed") is not None and type(attempt["retry_wait_completed"]) is not bool:
                raise RunError("invalid_retry_wait_state")
            if index == 2:
                prior = row["attempts"][0]
                prior_response = ModelResponse(**{key: prior[key] for key in RESPONSE_FIELDS})
                allowed_delay = retry_delay(prior_response, prior)
                if allowed_delay is None or prior["retry_delay_s"] != allowed_delay or prior["retry_wait_completed"] is not True:
                    raise RunError("invalid_retry_chain")
            if manifest["mode"] == "mock" and any(attempt.get(key) is not None for key in ("latency_ms", "elapsed_ms")):
                raise RunError("mock_timing_is_not_model_latency")
            sanitized = {**asdict(response), **safe_diagnostic(attempt),
                         "reasoning_evidence": safe_reasoning_evidence(attempt.get("reasoning_evidence")), "attempt_number": index,
                         "elapsed_ms": attempt.get("elapsed_ms"), "retry_delay_s": attempt.get("retry_delay_s"),
                         "retry_wait_completed": attempt.get("retry_wait_completed"),
                         "cost": None, "cost_unknown_reason": "billing_not_collected"}
            apply_attempt(row, sanitized, item, original.get("total_elapsed_ms"))
        if manifest["mode"] == "mock":
            row["total_elapsed_ms"] = None
        rows.append(row)
    if manifest["run_status"] == "complete" and any(row["status"] == "not_run" for row in rows):
        raise RunError("complete_run_has_missing_pairs")
    metrics = aggregate(rows, config)
    write_json(directory / "responses.json", rows)
    atomic_write(directory / "responses.csv", _csv(rows))
    write_json(directory / "metrics.json", {"run_id": manifest["run_id"], "run_status": manifest["run_status"], "mode": manifest["mode"], "models": metrics})
    flat = []
    for model in metrics:
        base = {key: value for key, value in model.items() if key != "categories"}
        flat.append({**base, "category": "all"})
        flat.extend({**base, **values, "category": category} for category, values in model["categories"].items())
    atomic_write(directory / "metrics.csv", _csv(flat))
    providers = {model["key"]: upstream_providers(rows, model["key"]) for model in config["models"]}
    atomic_write(directory / "leaderboard.md", _leaderboard(manifest, metrics, providers).encode("utf-8"))
    return {"run_id": manifest["run_id"], "run_status": manifest["run_status"], "mode": manifest["mode"], "pair_count": len(rows), "models": metrics}
