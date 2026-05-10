"""Auxiliary judge evaluation details and sanitization helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|authorization|bearer|token|secret|password|credential|database_url)",
    re.IGNORECASE,
)
SENSITIVE_VALUE_PATTERN = re.compile(
    r"(Bearer\s+[A-Za-z0-9._~+/=-]+|sk-[A-Za-z0-9_-]+|[A-Za-z0-9_-]{24,})",
    re.IGNORECASE,
)

CORE_JUDGE_OUTPUT_KEYS = {
    "score",
    "nota",
    "rationale",
    "justificativa",
    "explanation",
    "legal_accuracy",
    "hallucination_risk",
    "rubric_alignment",
    "requires_human_review",
    "criteria",
}


@dataclass(frozen=True)
class EvaluationDetails:
    """Auxiliary metadata parsed from judge output."""

    legal_accuracy: str | None = None
    hallucination_risk: str | None = None
    rubric_alignment: str | None = None
    requires_human_review: bool | None = None
    criteria: dict[str, Any] = field(default_factory=dict)
    raw_output_jsonb: dict[str, Any] | None = None
    source_log_path: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class EvaluationDetailsImportReport:
    """Summary of an idempotent details import."""

    processed: int
    imported: int
    skipped: int
    problems: tuple[str, ...] = ()


def build_criteria(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge explicit criteria with extra top-level judge output keys."""
    raw_criteria = payload.get("criteria")
    criteria = dict(raw_criteria) if isinstance(raw_criteria, dict) else {}
    for key, value in payload.items():
        if key not in CORE_JUDGE_OUTPUT_KEYS:
            criteria[key] = value
    sanitized = sanitize_json_value(criteria)
    return sanitized if isinstance(sanitized, dict) else {}


def sanitize_json_value(value: Any) -> Any:
    """Recursively redact secret-like keys and string values before JSONB persistence."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if SENSITIVE_KEY_PATTERN.search(key_text):
                sanitized[key_text] = "<redacted>"
            else:
                sanitized[key_text] = sanitize_json_value(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, str):
        return SENSITIVE_VALUE_PATTERN.sub("<redacted>", value).replace("\x00", "").strip()
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def details_from_payload(
    payload: dict[str, Any],
    *,
    source_log_path: str | None = None,
    run_id: str | None = None,
) -> EvaluationDetails:
    """Build sanitized auxiliary details from a parsed judge JSON payload."""
    raw_output = sanitize_json_value(payload)
    return EvaluationDetails(
        legal_accuracy=_optional_string(payload.get("legal_accuracy")),
        hallucination_risk=_optional_string(payload.get("hallucination_risk")),
        rubric_alignment=_optional_string(payload.get("rubric_alignment")),
        requires_human_review=_optional_bool(payload.get("requires_human_review")),
        criteria=build_criteria(payload),
        raw_output_jsonb=raw_output if isinstance(raw_output, dict) else None,
        source_log_path=source_log_path,
        run_id=run_id,
    )


def jsonb_dumps(value: dict[str, Any] | None) -> str | None:
    """Serialize a JSONB object deterministically for psycopg2 parameters."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_json_object(value: str) -> dict[str, Any] | None:
    """Load a JSON object from a string if one is present."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def iter_json_records(path: Path) -> list[dict[str, Any]]:
    """Read JSON or JSONL records from a historical metadata source."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    parsed = load_json_object(text)
    if parsed is not None:
        return [parsed]
    try:
        parsed_json = json.loads(text)
    except json.JSONDecodeError:
        parsed_json = None
    if isinstance(parsed_json, list):
        return [item for item in parsed_json if isinstance(item, dict)]

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        item = line.strip()
        if not item:
            continue
        parsed_line = load_json_object(item)
        if parsed_line is not None:
            records.append(parsed_line)
    return records


def _optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "yes", "sim"}:
            return True
        if lowered in {"false", "f", "0", "no", "nao", "não"}:
            return False
    return None
