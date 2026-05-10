"""Import auxiliary judge evaluation details from explicit historical sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .audit_log_parser import AUDIT_LINE_PATTERN, DEFAULT_PROD_LOGS_MANIFEST, _key_values, _parse_int
from .evaluation_details import (
    CORE_JUDGE_OUTPUT_KEYS,
    EvaluationDetailsImportReport,
    details_from_payload,
    iter_json_records,
    load_json_object,
)
from .repositories import JudgeRepository

RAW_OUTPUT_KEYS = ("raw_output", "raw_response", "output_json", "parsed_output")
EVALUATION_ID_KEYS = ("id_avaliacao", "evaluation_id", "matched_evaluation_id")


class EvaluationDetailsImporter:
    """Idempotently import auxiliary metadata without changing official scores."""

    def __init__(self, repository: JudgeRepository) -> None:
        self.repository = repository

    def import_sources(
        self,
        *,
        manifest_path: Path | str = DEFAULT_PROD_LOGS_MANIFEST,
        raw_output_dirs: tuple[Path | str, ...] = (),
    ) -> EvaluationDetailsImportReport:
        processed = 0
        imported = 0
        skipped = 0
        problems: list[str] = []

        for source_path, record in _iter_raw_output_records(raw_output_dirs):
            processed += 1
            outcome = self._import_record(record, source_path=str(source_path), run_id=_run_id(source_path))
            if outcome is None:
                imported += 1
            else:
                skipped += 1
                problems.append(outcome)

        for source_path, record in _iter_manifest_log_records(Path(manifest_path)):
            processed += 1
            outcome = self._import_record(record, source_path=str(source_path), run_id=_run_id(source_path))
            if outcome is None:
                imported += 1
            else:
                skipped += 1
                problems.append(outcome)

        return EvaluationDetailsImportReport(
            processed=processed,
            imported=imported,
            skipped=skipped,
            problems=tuple(problems),
        )

    def _import_record(self, record: dict[str, Any], *, source_path: str, run_id: str | None) -> str | None:
        payload = _payload_from_record(record)
        if payload is None:
            return f"{source_path}: no structured judge output found"

        evaluation_id = _evaluation_id_from_record(record)
        if evaluation_id is None:
            evaluation_id = self._match_evaluation_id(record)
        if evaluation_id is None:
            return f"{source_path}: could not resolve a unique evaluation"

        self.repository.persist_evaluation_details(
            evaluation_id=evaluation_id,
            details=details_from_payload(payload, source_log_path=source_path, run_id=run_id),
        )
        return None

    def _match_evaluation_id(self, record: dict[str, Any]) -> int | None:
        answer_id = _parse_int_value(
            record.get("answer_id")
            or record.get("id_resposta")
            or record.get("id_resposta_ativa1")
        )
        judge_model = _optional_string(record.get("judge_model") or record.get("model") or record.get("modelo_juiz"))
        if answer_id is None or judge_model is None:
            return None
        return self.repository.find_evaluation_id_for_details(
            answer_id=answer_id,
            judge_model=judge_model,
            role=_optional_string(record.get("role") or record.get("papel_juiz")),
            panel_mode=_optional_string(record.get("panel_mode") or record.get("mode")),
            trigger_reason=_optional_string(record.get("trigger") or record.get("trigger_reason")),
            score=_parse_int_value(record.get("score") or record.get("nota")),
        )


def _iter_raw_output_records(raw_output_dirs: tuple[Path | str, ...]) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for raw_dir in raw_output_dirs:
        path = Path(raw_dir)
        paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
        for item in paths:
            if item.suffix.lower() not in {".json", ".jsonl"}:
                continue
            for record in iter_json_records(item):
                records.append((item, record))
    return records


def _iter_manifest_log_records(manifest_path: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not manifest_path.exists():
        return []
    records: list[tuple[Path, dict[str, Any]]] = []
    for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        log_path = Path(item)
        if not log_path.is_absolute():
            log_path = manifest_path.parent.parent.parent / log_path
        if not log_path.exists() or not log_path.is_file():
            continue
        records.extend((log_path, record) for record in _records_from_log(log_path))
    return records


def _records_from_log(log_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed_line = AUDIT_LINE_PATTERN.match(line)
        if parsed_line is None:
            continue
        message = parsed_line.group(2).strip()
        if message not in {"evaluation_parsed", "evaluation_skipped"}:
            continue
        values = _key_values(parsed_line.group(3).strip() if parsed_line.group(3) else None)
        if any(key in values for key in RAW_OUTPUT_KEYS):
            records.append(values)
    return records


def _payload_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    for key in RAW_OUTPUT_KEYS:
        value = record.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = load_json_object(value)
            if parsed is not None:
                return parsed
    structured = {key: value for key, value in record.items() if key in CORE_JUDGE_OUTPUT_KEYS}
    if structured:
        return structured
    return None


def _evaluation_id_from_record(record: dict[str, Any]) -> int | None:
    for key in EVALUATION_ID_KEYS:
        parsed = _parse_int_value(record.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_int_value(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _parse_int(value.strip())
    return None


def _optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _run_id(path: Path) -> str | None:
    return path.stem if path.suffix == ".log" else None
