"""Synthetic validation fixtures do not represent human-reviewed benchmark tasks."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from kazakh_model_watch.dataset import DatasetValidationError, load_dataset


def _record(category="numbers", *, item_id="test_001", verified=False):
    answer_type, reference = {
        "numbers": ("integer", "5"),
        "dates": ("iso_date", "2024-02-29"),
        "morphology": ("word", "қалаға"),
    }[category]
    return {
        "id": item_id,
        "category": category,
        "language": "kk",
        "prompt": "Синтетикалық тест тапсырмасы.",
        "reference_answer": reference,
        "accepted_answers": [reference],
        "answer_type": answer_type,
        "source": {"type": "author_created", "attribution": "Synthetic test fixture."},
        "review": {
            "status": "verified" if verified else "draft",
            "reviewer_id": "synthetic-test-reviewer" if verified else None,
            "reviewed_at": "2026-10-14" if verified else None,
            "note": "Synthetic metadata for testing, not a real human review.",
        },
    }


def _write_dataset(tmp_path, records):
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _set_field(record, field, value):
    keys = field.split(".")
    parent = record
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = value


def _full_dataset(*, verified=True):
    return [
        _record(category, item_id=f"{category}_{index}", verified=verified)
        for category in ("numbers", "dates", "morphology")
        for index in range(8)
    ]


def test_development_loads_one_draft_and_preserves_data(tmp_path):
    record = _record()
    record["metadata"] = {"draft_version": 1}
    path = _write_dataset(tmp_path, [record])
    original_bytes = path.read_bytes()

    assert load_dataset(str(path)) == [record]
    assert path.read_bytes() == original_bytes


def test_repository_dataset_contains_24_unreviewed_drafts():
    path = Path(__file__).resolve().parents[1] / "data" / "benchmark.jsonl"
    records = load_dataset(path)

    assert len(records) == 24
    assert {record["category"] for record in records} == {"numbers", "dates", "morphology"}
    assert all(sum(r["category"] == category for r in records) == 8 for category in ("numbers", "dates", "morphology"))
    assert all(record["review"]["status"] == "draft" for record in records)
    assert all(record["review"]["reviewer_id"] is None for record in records)
    assert all(record["review"]["reviewed_at"] is None for record in records)


def test_development_accepts_full_draft_set(tmp_path):
    records = _full_dataset(verified=False)
    assert load_dataset(_write_dataset(tmp_path, records)) == records


def test_final_accepts_balanced_set_with_verified_fixture_metadata(tmp_path):
    records = _full_dataset()
    assert load_dataset(_write_dataset(tmp_path, records), final=True) == records


@pytest.mark.parametrize("count", [1, 23, 25])
def test_final_rejects_wrong_size(tmp_path, count):
    records = [_record(item_id=f"test_{index}", verified=True) for index in range(count)]
    with pytest.raises(DatasetValidationError, match="exactly 24"):
        load_dataset(_write_dataset(tmp_path, records), final=True)


def test_final_rejects_unbalanced_categories(tmp_path):
    records = _full_dataset()
    records[-1] = _record("numbers", item_id="extra_numbers", verified=True)
    with pytest.raises(DatasetValidationError, match="8 records per category"):
        load_dataset(_write_dataset(tmp_path, records), final=True)


def test_final_rejects_any_unreviewed_record(tmp_path):
    records = _full_dataset()
    records[-1] = _record("morphology", item_id="unreviewed")
    with pytest.raises(DatasetValidationError, match=r":24: review.status.*verified"):
        load_dataset(_write_dataset(tmp_path, records), final=True)


@pytest.mark.parametrize("field", [
    "id", "category", "language", "prompt", "reference_answer", "accepted_answers",
    "answer_type", "source", "review", "source.type", "source.attribution",
    "review.status", "review.reviewer_id", "review.reviewed_at", "review.note",
])
def test_rejects_missing_required_fields(tmp_path, field):
    record = _record()
    keys = field.split(".")
    parent = record if len(keys) == 1 else record[keys[0]]
    del parent[keys[-1]]
    with pytest.raises(DatasetValidationError, match="required field is missing"):
        load_dataset(_write_dataset(tmp_path, [record]))


@pytest.mark.parametrize(("field", "value"), [
    ("id", 1), ("id", " "), ("category", []), ("category", "reading"),
    ("language", None), ("language", "en"), ("prompt", False), ("prompt", "\t"),
    ("reference_answer", 5), ("reference_answer", ""),
    ("answer_type", {}), ("answer_type", "word"),
    ("accepted_answers", "5"), ("accepted_answers", []),
    ("accepted_answers", [5]), ("accepted_answers", ["5", None]),
    ("accepted_answers", ["5", " "]), ("accepted_answers", ["6"]),
    ("source", []), ("source.type", True), ("source.type", "private"),
    ("source.attribution", None), ("review", None),
    ("review.status", "pending"), ("review.status", 1), ("review.note", []),
    ("review.reviewer_id", "someone"), ("review.reviewed_at", "2026-10-14"),
])
def test_rejects_invalid_fields_and_types(tmp_path, field, value):
    record = _record()
    _set_field(record, field, value)
    with pytest.raises(DatasetValidationError, match=field.split(".")[0]):
        load_dataset(_write_dataset(tmp_path, [record]))


def test_duplicate_id_reports_second_physical_line(tmp_path):
    path = _write_dataset(tmp_path, [_record(), _record("dates")])
    with pytest.raises(DatasetValidationError, match=r":2: id: duplicate ID"):
        load_dataset(path)


@pytest.mark.parametrize("answers", [["5", "5"], ["5", " 5\t"]])
def test_rejects_duplicate_variants_after_whitespace_normalization(tmp_path, answers):
    record = _record()
    record["accepted_answers"] = answers
    with pytest.raises(DatasetValidationError, match="duplicate after"):
        load_dataset(_write_dataset(tmp_path, [record]))


@pytest.mark.parametrize("answers", [["үй", "ҮЙ"], ["үй", "үи\u0306"]])
def test_rejects_casefold_and_nfc_equivalent_variants(tmp_path, answers):
    record = _record("morphology")
    record["reference_answer"] = "үй"
    record["accepted_answers"] = answers
    with pytest.raises(DatasetValidationError, match="duplicate after"):
        load_dataset(_write_dataset(tmp_path, [record]))


def test_normalization_does_not_merge_distinct_kazakh_letters(tmp_path):
    record = _record("morphology")
    record["reference_answer"] = "үн"
    record["accepted_answers"] = ["үн", "ұн"]
    # Intentionally different words test string identity, not semantic correctness.
    assert load_dataset(_write_dataset(tmp_path, [record])) == [record]


@pytest.mark.parametrize("answer", ["0", "-4", "123456789012345678901234567890"])
def test_accepts_canonical_integers_without_float_conversion(tmp_path, answer):
    record = _record()
    record["reference_answer"] = answer
    record["accepted_answers"] = [answer]
    assert load_dataset(_write_dataset(tmp_path, [record]))[0]["reference_answer"] == answer


@pytest.mark.parametrize(("category", "answer"), [
    ("numbers", "5.0"), ("numbers", "+5"), ("numbers", "05"),
    ("numbers", "-0"), ("numbers", " 5"), ("numbers", "٥"),
    ("dates", "2023-02-29"), ("dates", "2024-04-31"),
    ("dates", "2024-13-01"), ("dates", "0000-01-01"),
    ("dates", "20240229"), ("dates", "2024-2-29"), ("dates", "29/02/2024"),
    ("morphology", "qalağa"), ("morphology", "қала ға"),
    ("morphology", "қалаға."), ("morphology", "қала2"),
    ("morphology", "-қалаға"), ("morphology", "қала--ға"),
    ("morphology", "қала\u200bға"),
])
def test_rejects_invalid_reference_answer_format(tmp_path, category, answer):
    record = _record(category)
    record["reference_answer"] = answer
    record["accepted_answers"] = [answer]
    with pytest.raises(DatasetValidationError, match="reference_answer"):
        load_dataset(_write_dataset(tmp_path, [record]))


@pytest.mark.parametrize(("category", "answer"), [
    ("numbers", "5.0"), ("dates", "2023-02-29"), ("morphology", "қалаға."),
])
def test_checks_format_of_every_accepted_answer(tmp_path, category, answer):
    record = _record(category)
    record["accepted_answers"].append(answer)
    with pytest.raises(DatasetValidationError, match=r"accepted_answers\[1\]"):
        load_dataset(_write_dataset(tmp_path, [record]))


def test_accepts_hyphenated_cyrillic_word_and_preserves_case(tmp_path):
    record = _record("morphology")
    record["reference_answer"] = "Ата-ана"
    record["accepted_answers"] = ["Ата-ана"]
    assert load_dataset(_write_dataset(tmp_path, [record])) == [record]


@pytest.mark.parametrize(("field", "value"), [
    ("review.reviewer_id", None), ("review.reviewer_id", ""),
    ("review.reviewed_at", None), ("review.reviewed_at", "20261014"),
    ("review.reviewed_at", "2026-02-30"),
])
def test_verified_review_always_requires_valid_metadata(tmp_path, field, value):
    record = _record(verified=True)
    _set_field(record, field, value)
    with pytest.raises(DatasetValidationError, match=field):
        load_dataset(_write_dataset(tmp_path, [record]))


def test_public_source_with_attribution_url_and_license(tmp_path):
    record = _record()
    record["source"] = {
        "type": "public", "attribution": "Synthetic public-source metadata.",
        "url": "https://example.org/dataset", "license": "CC0-1.0",
    }
    assert load_dataset(_write_dataset(tmp_path, [record])) == [record]


@pytest.mark.parametrize(("field", "value"), [
    ("url", None), ("url", "https://"), ("url", "file:///private/data"),
    ("url", "https://user:password@example.org/data"),
    ("url", "https://bad host.example/data"), ("license", ""), ("license", 1),
])
def test_public_source_rejects_invalid_required_metadata(tmp_path, field, value):
    record = _record()
    record["source"].update(type="public", url="https://example.org", license="CC0-1.0")
    record["source"][field] = value
    with pytest.raises(DatasetValidationError, match=f"source.{field}"):
        load_dataset(_write_dataset(tmp_path, [record]))


@pytest.mark.parametrize("field", ["url", "license"])
def test_public_source_requires_url_and_license(tmp_path, field):
    record = _record()
    record["source"].update(type="public", url="https://example.org", license="CC0-1.0")
    del record["source"][field]
    with pytest.raises(DatasetValidationError, match=f"source.{field}"):
        load_dataset(_write_dataset(tmp_path, [record]))


@pytest.mark.parametrize("text", [
    "not json", "# comment", "{", "[]", '"text"', "123", "null",
    '{"id":"first","id":"second"}',
    '{"review":{"status":"draft","status":"verified"}}',
    '{"extra":NaN}', '{"extra":Infinity}',
])
def test_rejects_malformed_or_non_object_jsonl(tmp_path, text):
    path = tmp_path / "broken.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=r"broken.jsonl:1:"):
        load_dataset(path)


@pytest.mark.parametrize("text", ["", "\n\n", " \t\n"])
def test_rejects_empty_dataset(tmp_path, text):
    path = tmp_path / "empty.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="at least one record"):
        load_dataset(path)


@pytest.mark.parametrize("contents", [b"\xff", b"\xef\xbb\xbf{}"])
def test_rejects_invalid_utf8_or_bom(tmp_path, contents):
    path = tmp_path / "encoding.jsonl"
    path.write_bytes(contents)
    with pytest.raises(DatasetValidationError, match="UTF-8"):
        load_dataset(path)


def test_blank_lines_crlf_and_unicode_line_separator_preserve_content(tmp_path):
    record = _record()
    record["prompt"] = "Бірінші\u2028екінші"
    path = tmp_path / "benchmark.jsonl"
    path.write_bytes(("\r\n" + json.dumps(record, ensure_ascii=False) + "\r\n\r\n").encode("utf-8"))
    assert load_dataset(path) == [record]


def test_error_reports_file_physical_line_and_field(tmp_path):
    valid = _record()
    invalid = deepcopy(valid)
    invalid["prompt"] = ""
    path = tmp_path / "benchmark.jsonl"
    path.write_text(json.dumps(valid) + "\n\n" + json.dumps(invalid), encoding="utf-8")
    with pytest.raises(DatasetValidationError) as caught:
        load_dataset(path)
    assert str(caught.value).startswith(f"{path}:3: prompt:")


def test_missing_file_preserves_io_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.jsonl")
