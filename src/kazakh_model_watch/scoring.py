"""Deterministic, whole-response binary scoring defined by the MVP spec."""

from collections.abc import Sequence

from .normalization import normalize_answer


def score_response(
    raw_response: str | None,
    reference_answer: str,
    accepted_answers: Sequence[str],
    *,
    status: str,
) -> dict[str, int]:
    """Return exact_match and correct scores without changing the inputs.

    Gold answers must come from a validated dataset. Exact match compares Python
    strings directly with reference_answer; correct compares the normalized full
    response with each normalized accepted answer using kk-short-v1.

    Every status other than "ok" scores zero, even if text matches the reference.
    Missing or whitespace-only text also scores zero. A non-string, non-null
    response with status="ok" raises TypeError; values are never coerced to text.
    """
    if status != "ok" or raw_response is None:
        return {"exact_match": 0, "correct": 0}
    if not isinstance(raw_response, str):
        raise TypeError("raw_response must be a string or None")
    if not raw_response.strip():
        return {"exact_match": 0, "correct": 0}

    normalized_response = normalize_answer(raw_response)
    return {
        "exact_match": int(raw_response == reference_answer),
        "correct": int(any(
            normalized_response == normalize_answer(answer)
            for answer in accepted_answers
        )),
    }
