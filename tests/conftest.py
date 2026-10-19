"""New evaluation fixtures are synthetic; verified flags are not human reviews."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from kazakh_model_watch.model_adapter import ModelAdapter, ModelResponse


ROOT = Path(__file__).resolve().parents[1]


class ScriptedAdapter(ModelAdapter):
    def __init__(self, actions=(), calls=None):
        self.actions = list(actions)
        self.calls = calls if calls is not None else []
        self.diagnostic = {}
        self.count = 0

    def generate(self, request):
        self.calls.append(request)
        action = self.actions[self.count] if self.count < len(self.actions) else ("ok", "5", {})
        self.count += 1
        if isinstance(action, BaseException):
            raise action
        status, raw, diagnostic = action
        self.diagnostic = diagnostic
        return ModelResponse(status=status, raw_response=raw, requested_model_id=request.model_id,
                             returned_model_id=request.model_id, finish_reason=diagnostic.get("finish_reason"),
                             usage={"prompt_tokens": 0, "completion_tokens": 2, "total_tokens": 2})

    def get_diagnostic(self):
        return deepcopy(self.diagnostic)


@pytest.fixture
def input_factory(tmp_path):
    count = 0

    def create(*, full=False, real=False, ready=True):
        nonlocal count
        count += 1
        directory = tmp_path / f"inputs-{count}"
        directory.mkdir()
        config = json.loads((ROOT / "config/mock_models.json").read_text(encoding="utf-8"))
        config["dataset_version"] = "synthetic-tests-v1"
        if real:
            for index, model in enumerate(config["models"]):
                model.update(provider="openrouter", model_id=f"synthetic/model-{index}", api_key_env="OPENROUTER_API_KEY",
                             readiness="ready" if ready else "not_ready", readiness_reason=None if ready else "Synthetic not-ready fixture.")
                config_model = {"id": model["model_id"], "canonical_slug": model["model_id"] + "-fixed",
                                "supported_parameters": ["temperature", "max_tokens"],
                                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                                "top_provider": {"max_completion_tokens": 128}}
                model["metadata"] = {"source_url": "https://openrouter.ai/api/v1/models", "checked_on": "2026-10-19",
                                     "endpoints_source_url": f"https://openrouter.ai/api/v1/models/{model['model_id']}/endpoints",
                                     "model": config_model, "profile_verified": ready,
                                     "endpoints": [{"supported_parameters": ["temperature", "max_tokens"], "max_completion_tokens": 128}]}
        records = []
        for category, answer_type, answer in [("numbers", "integer", "5"), ("dates", "iso_date", "2024-02-29"), ("morphology", "word", "қалаға")]:
            for index in range(8 if full else 2):
                records.append({"id": f"synthetic_{category}_{index}", "category": category, "language": "kk",
                                "prompt": f'Синтетикалық тапсырма: {category}, "{index}".\nТек жауап.',
                                "reference_answer": answer, "accepted_answers": [answer], "answer_type": answer_type,
                                "source": {"type": "author_created", "attribution": "Synthetic unit-test fixture."},
                                "review": {"status": "verified" if full else "draft", "reviewer_id": "synthetic-reviewer" if full else None,
                                           "reviewed_at": "2026-10-19" if full else None, "note": "Synthetic metadata; not a real human review."}})
        dataset, config_path = directory / "dataset.jsonl", directory / "models.json"
        dataset.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return dataset, config_path

    return create


@pytest.fixture
def evaluation_inputs(input_factory):
    return input_factory()
