"""Synthetic responses test scoring rules, not real model performance."""

from copy import deepcopy
from pathlib import Path

import pytest

from kazakh_model_watch.dataset import load_dataset
from kazakh_model_watch.normalization import normalize_answer
from kazakh_model_watch.scoring import score_response


@pytest.mark.parametrize("answer", [
    "5", "0", "-4", "123456789012345678901234567890",
    "2024-02-29", "қалаға", "Ата-ана",
])
def test_exact_reference_scores_one_for_both_metrics(answer):
    result = score_response(answer, answer, [answer], status="ok")
    assert result == {"exact_match": 1, "correct": 1}
    assert all(type(value) is int for value in result.values())


@pytest.mark.parametrize(("raw_response", "reference"), [
    (" 5\n", "5"),
    ("\t5\r\n", "5"),
    ("\u00a0\u2003ҚАЛАҒА\u202f\u2028", "қалаға"),
    ("ҮЙ", "үй"),
    ("үи\u0306", "үй"),
    ("үй", "үи\u0306"),
    ("АТА-АНА", "Ата-ана"),
])
def test_normalized_match_does_not_count_as_exact(raw_response, reference):
    assert score_response(raw_response, reference, [reference], status="ok") == {
        "exact_match": 0, "correct": 1,
    }


@pytest.mark.parametrize("raw_response", ["үй", " \tҮИ\u0306\n"])
def test_preapproved_alternative_is_correct_but_not_exact(raw_response):
    assert score_response(raw_response, "үйге", ["үйге", "үй"], status="ok") == {
        "exact_match": 0, "correct": 1,
    }


def test_unlisted_alternative_is_not_automatically_accepted():
    assert score_response("үй", "үйге", ["үйге"], status="ok") == {
        "exact_match": 0, "correct": 0,
    }


@pytest.mark.parametrize("raw_response", [
    "6", "Жауап: 5", "5 немесе 6", "2 + 3 = 5", "5\nТүсіндірме",
    "5.", '"5"', "'5'", "`5`", "**5**", "```\n5\n```", "- 5",
    "5 5", "Кешіріңіз, жауап бере алмаймын.",
])
def test_scores_entire_response_without_extracting_answer(raw_response):
    assert score_response(raw_response, "5", ["5"], status="ok") == {
        "exact_match": 0, "correct": 0,
    }


@pytest.mark.parametrize(("raw_response", "reference"), [
    ("5.0", "5"), ("05", "5"), ("+5", "5"), ("5e0", "5"),
    ("-0", "0"), ("−4", "-4"), ("٥", "5"),
    ("５", "5"), ("⁵", "5"),
    ("05/03/2026", "2026-03-05"), ("2026-3-5", "2026-03-05"),
    ("2026-05-03", "2026-03-05"), ("2026-03-05T00:00:00", "2026-03-05"),
])
def test_does_not_repair_numbers_dates_or_compatibility_characters(raw_response, reference):
    assert score_response(raw_response, reference, [reference], status="ok") == {
        "exact_match": 0, "correct": 0,
    }


@pytest.mark.parametrize(("raw_response", "reference"), [
    ("а", "ә"), ("г", "ғ"), ("к", "қ"), ("н", "ң"), ("о", "ө"),
    ("у", "ұ"), ("у", "ү"), ("ұ", "ү"), ("и", "і"),
    ("калага", "қалаға"), ("qalağa", "қалаға"), ("қалaға", "қалаға"),
    ("i", "і"), ("қала ға", "қалаға"),
    ("атаана", "ата-ана"), ("ата–ана", "ата-ана"), ("қалаға.", "қалаға"),
])
def test_preserves_kazakh_letters_confusables_punctuation_and_hyphens(raw_response, reference):
    assert score_response(raw_response, reference, [reference], status="ok") == {
        "exact_match": 0, "correct": 0,
    }


@pytest.mark.parametrize("character", ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"])
def test_does_not_remove_zero_width_characters(character):
    assert score_response(f"5{character}", "5", ["5"], status="ok") == {
        "exact_match": 0, "correct": 0,
    }


@pytest.mark.parametrize("status", [
    "api_error", "timeout", "empty_response", "truncated", "not_run",
    "OK", "unknown", "",
])
@pytest.mark.parametrize("raw_response", ["5", " 5\n", None])
def test_every_non_ok_status_scores_zero_even_for_matching_text(status, raw_response):
    assert score_response(raw_response, "5", ["5"], status=status) == {
        "exact_match": 0, "correct": 0,
    }


@pytest.mark.parametrize("raw_response", [None, "", " \t\r\n", "\u00a0\u2028"])
def test_missing_or_whitespace_only_text_scores_zero_even_with_ok_status(raw_response):
    assert score_response(raw_response, "5", ["5"], status="ok") == {
        "exact_match": 0, "correct": 0,
    }


@pytest.mark.parametrize("raw_response", [5, 5.0, True, b"5", ["5"]])
def test_does_not_coerce_invalid_response_types_into_strings(raw_response):
    with pytest.raises(TypeError, match="raw_response"):
        score_response(raw_response, "5", ["5"], status="ok")


@pytest.mark.parametrize(("text", "expected"), [
    ("  ҚАЛАҒА\n", "қалаға"),
    ("\u00a0Ата\t\n\u2003АНА\u202f", "ата ана"),
    ("ҮИ\u0306", "үй"),
    ("Straße", "strasse"),
    ("\u0390", "\u03b9\u0308\u0301"),
    ("５", "５"),
    ("қала\u200bға.", "қала\u200bға."),
])
def test_shared_normalization_contract(text, expected):
    # Non-Kazakh examples isolate Unicode behavior, not valid benchmark golds.
    assert normalize_answer(text) == expected


def test_scoring_normalizes_internal_whitespace_on_both_sides():
    # Multiword strings exercise the shared protocol independently of MVP types.
    assert score_response(
        "\nАТА\t\u2003АНА ", "Ата\u00a0ана", ["Ата\u00a0ана"], status="ok",
    ) == {"exact_match": 0, "correct": 1}


def test_scoring_uses_casefold_rather_than_lower():
    assert score_response("STRASSE", "Straße", ["Straße"], status="ok") == {
        "exact_match": 0, "correct": 1,
    }


def test_scoring_is_repeatable_and_preserves_input_values():
    response = {"status": "ok", "raw_response": " \tҮИ\u0306\n"}
    gold = {"reference_answer": "үйге", "accepted_answers": ["үйге", "үй"]}
    original = deepcopy((response, gold))

    first = score_response(**response, **gold)
    second = score_response(**response, **gold)

    assert first == second == {"exact_match": 0, "correct": 1}
    assert (response, gold) == original


def test_scoring_accepts_gold_answers_from_dataset_loader():
    path = Path(__file__).resolve().parents[1] / "data" / "benchmark.jsonl"
    records = load_dataset(path)
    original = deepcopy(records)

    for record in records:
        assert score_response(
            record["reference_answer"], record["reference_answer"],
            record["accepted_answers"], status="ok",
        ) == {"exact_match": 1, "correct": 1}
    assert records == original
