"""Small CLI: offline validation/report and explicitly gated generation."""

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigValidationError, load_config
from .dataset import DatasetValidationError, load_dataset
from .reporting import RunError, report
from .runner import run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kazakh Model Watch; model API calls require run --live and successful preflight.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset", type=Path, default=Path("data/benchmark.jsonl"))
        sub.add_argument("--config", type=Path, default=Path("config/models.json"))
        if name == "validate":
            sub.add_argument("--development", action="store_true", help="Allow an incomplete draft dataset; does not authorize generation.")
        else:
            mode = sub.add_mutually_exclusive_group()
            mode.add_argument("--live", action="store_true", help="Explicit opt-in to billable model calls after final validation.")
            mode.add_argument("--mock", action="store_true", help="Synthetic offline development; requires a mock-only config.")
            sub.add_argument("--output-root", type=Path, default=Path("results"))
            sub.add_argument("--run-id")
    commands.add_parser("report").add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            items = load_dataset(args.dataset, final=not args.development)
            config = load_config(args.config, require_ready=not args.development)
            result = {"status": "valid", "final": not args.development, "dataset_items": len(items),
                      "model_readiness": [{"key": model["key"], "readiness": model["readiness"], "reason": model["readiness_reason"]} for model in config["models"]]}
        elif args.command == "run":
            directory = run_benchmark(args.dataset, args.config, output_root=args.output_root, run_id=args.run_id, mock=args.mock, live=args.live)
            result = {"status": "complete", "mode": "mock" if args.mock else "live", "run_dir": str(directory)}
        else:
            result = report(args.run_dir)
    except KeyboardInterrupt:
        result = {"status": "error", "error_code": "interrupted_run_is_incomplete"}
        print(json.dumps(result))
        return 130
    except (ValueError, OSError, TypeError, KeyError):
        # No exception bodies, secrets, raw responses or headers at this boundary.
        exception = sys.exc_info()[1]
        code = str(exception) if isinstance(exception, RunError) else (
            "dataset_invalid" if isinstance(exception, DatasetValidationError) else
            ("model_not_ready" if str(exception) == "model_not_ready" else "config_invalid")
            if isinstance(exception, ConfigValidationError) else "invalid_input_or_artifact"
        )
        print(json.dumps({"status": "error", "error_code": code}))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
