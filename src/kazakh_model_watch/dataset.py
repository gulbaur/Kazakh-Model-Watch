"""Load JSONL datasets and validate the MVP dataset contract without network calls."""

import json
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .normalization import normalize_answer


_CATEGORY_TYPES = {"numbers": "integer", "dates": "iso_date", "morphology": "word"}
_REQUIRED_FIELDS = (
    "id", "category", "language", "prompt", "reference_answer",
    "accepted_answers", "answer_type", "source", "review",
)
_INTEGER = re.compile(r"(?:0|-?[1-9][0-9]*)")
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_LETTERS = "А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі"
_WORD = re.compile(rf"[{_LETTERS}]+(?:-[{_LETTERS}]+)*")


class DatasetValidationError(ValueError):
    """Dataset data does not satisfy the required structure or validation mode."""


def _object(value: Any, fields: tuple[str, ...], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{label}: expected a JSON object")
    for field in fields:
        if field not in value:
            raise DatasetValidationError(f"{label}.{field}: required field is missing")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field}: expected a non-empty string")
    return value


def _validate_date(value: Any, field: str) -> None:
    text = _string(value, field)
    if not _ISO_DATE.fullmatch(text):
        raise DatasetValidationError(f"{field}: expected YYYY-MM-DD")
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise DatasetValidationError(f"{field}: invalid calendar date") from exc


def _validate_answer(value: str, answer_type: str, field: str) -> None:
    if answer_type == "integer":
        if not _INTEGER.fullmatch(value):
            raise DatasetValidationError(f"{field}: expected a canonical integer")
    elif answer_type == "iso_date":
        _validate_date(value, field)
    elif not _WORD.fullmatch(unicodedata.normalize("NFC", value)):
        raise DatasetValidationError(f"{field}: expected one word in Kazakh Cyrillic")


def _validate_source(value: Any) -> None:
    source = _object(value, ("type", "attribution"), "source")
    source_type = _string(source["type"], "source.type")
    if source_type not in {"author_created", "public"}:
        raise DatasetValidationError("source.type: expected author_created or public")
    _string(source["attribution"], "source.attribution")
    if source_type == "public":
        _object(source, ("url", "license"), "source")
        url = _string(source["url"], "source.url")
        _string(source["license"], "source.license")
        try:
            parsed = urlsplit(url)
            valid = (
                parsed.scheme in {"http", "https"}
                and parsed.hostname
                and parsed.username is None
                and not any(character.isspace() for character in url)
            )
        except ValueError:
            valid = False
        if not valid:
            raise DatasetValidationError("source.url: expected an HTTP(S) URL without credentials")


def _validate_review(value: Any) -> None:
    review = _object(value, ("status", "reviewer_id", "reviewed_at", "note"), "review")
    status = _string(review["status"], "review.status")
    _string(review["note"], "review.note")
    if status == "verified":
        _string(review["reviewer_id"], "review.reviewer_id")
        _validate_date(review["reviewed_at"], "review.reviewed_at")
    elif status == "draft":
        for field in ("reviewer_id", "reviewed_at"):
            if review[field] is not None:
                raise DatasetValidationError(f"review.{field}: must be null for a draft")
    else:
        raise DatasetValidationError("review.status: expected draft or verified")


def _validate_record(value: Any) -> dict[str, Any]:
    record = _object(value, _REQUIRED_FIELDS, "record")
    for field in ("id", "category", "language", "prompt", "reference_answer", "answer_type"):
        _string(record[field], field)
    category = record["category"]
    if category not in _CATEGORY_TYPES:
        raise DatasetValidationError("category: expected numbers, dates or morphology")
    if record["language"] != "kk":
        raise DatasetValidationError("language: expected kk")
    answer_type = record["answer_type"]
    if answer_type != _CATEGORY_TYPES[category]:
        raise DatasetValidationError(f"answer_type: {category} requires {_CATEGORY_TYPES[category]}")

    answers = record["accepted_answers"]
    if not isinstance(answers, list) or not answers:
        raise DatasetValidationError("accepted_answers: expected a non-empty array of strings")
    normalized = set()
    for index, answer in enumerate(answers):
        field = f"accepted_answers[{index}]"
        text = _string(answer, field)
        normalized_text = normalize_answer(text)
        if normalized_text in normalized:
            raise DatasetValidationError(f"{field}: duplicate after NFC/whitespace/casefold normalization")
        normalized.add(normalized_text)
    if record["reference_answer"] not in answers:
        raise DatasetValidationError("accepted_answers: must include reference_answer exactly")
    _validate_answer(record["reference_answer"], answer_type, "reference_answer")
    for index, answer in enumerate(answers):
        _validate_answer(answer, answer_type, f"accepted_answers[{index}]")
    _validate_source(record["source"])
    _validate_review(record["review"])
    return record


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise DatasetValidationError(f"invalid JSON constant: {value}")


def load_dataset(path: str | Path, *, final: bool = False) -> list[dict[str, Any]]:
    """Load a non-empty UTF-8 JSONL dataset, preserving order and original values.

    Development mode accepts any non-empty number of records, including drafts.
    A draft has null reviewer_id/reviewed_at; verified records require both.
    Final mode additionally requires 24 verified records, eight per category.
    Metadata validation cannot establish that a human review actually occurred.

    Raises DatasetValidationError for invalid data and OSError for file I/O errors.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(f"{path}: expected UTF-8 text") from exc
    if text.startswith("\ufeff"):
        raise DatasetValidationError(f"{path}: UTF-8 BOM is not allowed")

    records = []
    record_lines = []
    seen_ids = set()
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            record = _validate_record(json.loads(
                line, object_pairs_hook=_json_object, parse_constant=_reject_constant,
            ))
            if record["id"] in seen_ids:
                raise DatasetValidationError(f"id: duplicate ID {record['id']!r}")
        except ValueError as exc:
            raise DatasetValidationError(f"{path}:{line_number}: {exc}") from exc
        seen_ids.add(record["id"])
        records.append(record)
        record_lines.append(line_number)

    if not records:
        raise DatasetValidationError(f"{path}: dataset must contain at least one record")
    if final:
        if len(records) != 24:
            raise DatasetValidationError(f"{path}: final dataset requires exactly 24 records; got {len(records)}")
        counts = Counter(record["category"] for record in records)
        if any(counts[category] != 8 for category in _CATEGORY_TYPES):
            raise DatasetValidationError(f"{path}: final dataset requires 8 records per category; got {dict(counts)}")
        for record, line_number in zip(records, record_lines):
            if record["review"]["status"] != "verified":
                raise DatasetValidationError(f"{path}:{line_number}: review.status: final dataset requires verified records")
    return records
