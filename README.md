# Kazakh Model Watch

A small, reproducible benchmark of large language models on short tasks in Kazakh, with deterministic scoring and every model response saved.

> **Status:** the evaluation pipeline (validation, gated model calls, scoring, reports) is implemented and tested. The first benchmark run is pending: the 24-item dataset is awaiting review by a Kazakh speaker, so no leaderboard has been published yet.

## Why

Kazakh is barely represented in common LLM leaderboards. This project asks a narrow question that can be checked: given the same short prompt in Kazakh Cyrillic, does a model return exactly the right number, date or word form? Scoring is plain string comparison, with no LLM judge, so every score can be traced back to a saved response.

## What is measured

| Category | Items | Task | Expected answer |
| --- | --- | --- | --- |
| `numbers` | 8 | Multi-step word problems | One integer, e.g. `-15` |
| `dates` | 8 | Calendar arithmetic with explicit dates | `YYYY-MM-DD` |
| `morphology` | 8 | Build one inflected word form | One word, e.g. `құлағыңа` |

Example (morphology): put «құлақ» into the 2nd-person singular possessive and dative case in «Ақсақалдың айтқан өсиетін [құлақ] құйып ал». Expected: `құлағыңа`.

Metrics per model, overall and per category:

- **Accuracy**: the whole response matches an accepted answer after `kk-short-v1` normalization (Unicode NFC, trimmed and collapsed whitespace, casefold). Nothing else is cleaned: `Жауап: 5`, `5.0` or `калага` instead of `қалаға` are wrong.
- **Exact match**: the response equals the reference character for character, which measures format compliance.
- **Coverage**: share of complete responses. Errors, timeouts and truncations stay in the denominator.
- **Latency**: mean and median over successful calls, with sample size; includes network time.
- Status counts and retries. Cost is not collected and is shown as `N/A`.

## Models

| Model | Access | Readiness |
| --- | --- | --- |
| `qwen/qwen3.5-9b` | OpenRouter | `not_ready` |
| `google/gemini-3.1-flash-lite` | OpenRouter | `not_ready` |

Both models get the same request (protocol `mvp-v2-non-thinking`): one user message with a fixed Kazakh instruction and the task, `temperature=0`, `max_tokens=128`, 60 s timeout, no streaming, `reasoning: {"enabled": false}`, and `provider.require_parameters=true`, `allow_fallbacks=false`. There is no system prompt, no few-shot and no per-model prompt tuning. Readiness is set by the owner after reviewing live evidence; see [Current results](#current-results).

## Architecture

```text
data/benchmark.jsonl ──┐
                       ├─> validate ──> run ──────────────> results/<run_id>/ ──> report ──> leaderboard.md
config/models.json ────┘   (offline)    (--mock or --live)   manifest, inputs,     (offline,
                                                              responses, attempts   recomputes scores)
```

| Module | Role |
| --- | --- |
| [`dataset.py`](src/kazakh_model_watch/dataset.py) | Loads and validates the JSONL dataset: schema, IDs, answer formats, review metadata |
| [`config.py`](src/kazakh_model_watch/config.py) | Validates the frozen profile, model identities and saved provider metadata |
| [`openrouter_adapter.py`](src/kazakh_model_watch/openrouter_adapter.py) | One HTTP call per request; no SDK, redirects, retries or fallback |
| [`runner.py`](src/kazakh_model_watch/runner.py) | Preflight gates, seeded order, at most one retry, atomic writes after every attempt |
| [`scoring.py`](src/kazakh_model_watch/scoring.py), [`normalization.py`](src/kazakh_model_watch/normalization.py) | Binary exact-match and normalized correctness |
| [`reporting.py`](src/kazakh_model_watch/reporting.py) | Metrics, JSON/CSV export, Markdown leaderboard, offline report |
| [`non_thinking_smoke.py`](src/kazakh_model_watch/non_thinking_smoke.py) | Limited two-call diagnostic of the shared profile; dry run by default |

Design choices: a single sequential process, plain files instead of a database, and no hidden retries. A retry happens only after a timeout, a network error or HTTP 429/5xx. An interrupted run is saved as `incomplete`, and pairs that never ran are recorded as `not_run`.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"     # Windows: .venv\Scripts\python.exe
.venv/bin/python -m pytest -q                   # offline; sockets are blocked
```

Validate the data and configuration (no keys, no network):

```bash
.venv/bin/python -m kazakh_model_watch validate --development
```

Run the full pipeline on two synthetic mock models, then rebuild the report from saved files:

```bash
.venv/bin/python -m kazakh_model_watch run --mock --config config/mock_models.json --run-id demo-001
.venv/bin/python -m kazakh_model_watch report results/demo-001
```

Mock answers are synthetic, and the generated `leaderboard.md` is labelled `OFFLINE SYNTHETIC — NOT A MODEL BENCHMARK`.

## Reproducibility

- `manifest.json` records SHA-256 hashes of the dataset, configuration and instruction, the Git commit and dirty flag, Python, Unicode and dependency versions, the seeded item order (seed 42), the actual call order, returned model IDs and observed upstream providers.
- `report` recomputes every score and metric from the saved responses only, without keys or network. Two mock runs give byte-identical responses and metrics apart from the run ID, and `report` rebuilds the derived files identically.
- Repeating live calls at temperature 0 can still return different text. Here, reproducible means re-scoring the saved run, not regenerating it.

## Live runs (paid)

Live calls are blocked unless all of these hold: an explicit `--live` flag, a final dataset of 24 verified items, both models marked `ready` after review, and `OPENROUTER_API_KEY` set in the environment. The runner never reads `.env`, and CI has no API keys. A full run is 48 calls, at most 96 HTTP requests with retries.

```bash
.venv/bin/python -m kazakh_model_watch run --live      # refuses to start until the gates above pass
```

## Current results

No benchmark results yet. Verified so far:

- The offline suite (660 tests) covers loading, scoring, adapters, retries, interrupted runs, exports and the report. The GitHub Actions workflow runs it on Python 3.11 and 3.13 with no keys and no network.
- Two live diagnostic checks were made on a single prompt (2 + 3); they are not benchmark results:
  - **2026-10-15, Qwen, earlier profile without the reasoning flag:** answered `5`. The API reported 293 completion tokens, 291 of them reasoning, even though `max_tokens=128` was requested. The cause is not established.
  - **2026-10-20, both models, reasoning disabled:** Qwen answered with zero reported reasoning tokens. Gemini also reported zero reasoning tokens but returned a non-empty reasoning field, so the check stopped by design (`reasoning_not_disabled`). The field's type was not logged at the time; the current code records it.

## Limitations

- **Draft dataset.** All 24 items are AI-generated candidates; the owner has skimmed the numbers and dates items. None has yet been verified by an independent Kazakh speaker. A final run requires every item to be marked `verified` with a real reviewer and date.
- **Small sample.** One answer moves overall accuracy by 4.2 points and a category by 12.5 points. Differences below roughly 20 points should not be read as a ranking.
- **Narrow scope.** `numbers` and `dates` test multi-step calculation in Kazakh without chain of thought, within 128 tokens. Only `morphology` is purely linguistic. Results do not measure general Kazakh proficiency.
- **Reasoning cannot be verified.** `reasoning.enabled=false` is a request. A provider may still reason internally, and zero counters do not prove otherwise. `max_tokens=128` is a requested limit, not a verified visible-token or billing cap.
- **Open rule for Gemini.** Any non-empty reasoning field, including opaque encrypted metadata, currently counts as `reasoning_not_disabled`. If Gemini always returns such metadata, it would score zero coverage. This rule needs an owner decision before a full run.
- **Routing.** OpenRouter can serve one model from different upstreams and quantizations. The upstream is recorded when the API reports it, but it is not pinned.
- **Contamination and drift.** Public tasks can end up in training data, and model aliases can change. Returned model IDs are saved.
- **No cost data.** Cost is shown as `N/A`, never estimated.

## Roadmap

1. Independent review of all 24 items by a Kazakh speaker.
2. Decide the rule for opaque reasoning metadata; re-check Gemini or choose another model.
3. Mark both models ready after review, then run the full benchmark under an approved budget.
4. Publish the leaderboard together with the run's artifacts.

## Repository layout

```text
config/       models.json (two OpenRouter models), mock_models.json, historical smoke snapshot
data/         benchmark.jsonl: 24 draft items, 8 per category
docs/         MVP_SPEC.md (protocol), DEVELOPMENT_NOTES.md and review notes (in Russian)
src/kazakh_model_watch/   package and CLI (python -m kazakh_model_watch)
tests/        offline pytest suite with synthetic data and mocked HTTP
.github/      offline CI, tracked-file safety check, issue and PR templates
AGENTS.md     working rules for coding agents and contributors (in Russian)
```

The full protocol, including scoring rules and acceptance criteria, is in [docs/MVP_SPEC.md](docs/MVP_SPEC.md) (Russian).

## Security

API keys are read only from environment variables and are never written to configuration, results or logs. `.env`, `results/` and `logs/` are ignored by Git. CI runs [a tracked-file check](.github/scripts/check_tracked_safety.py) for local-only paths and common credential patterns, and tests cannot open network sockets. [.env.example](.env.example) only names the variable: the runner and the limited smoke read the environment directly, and only the historical single-call smoke loads `.env`, with an explicit `--dotenv`.

## License

[MIT](LICENSE).
