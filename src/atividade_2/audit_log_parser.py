"""Read-only parser for versioned production audit logs."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_PROD_LOGS_MANIFEST = Path("outputs") / "audit" / "prod_logs_manifest.txt"
AUDIT_LINE_PATTERN = re.compile(r"^([^|]+)\s+\|\s+([^|]+)(?:\s+\|\s+(.*))?$")
KEY_PATTERN = re.compile(r"(?<!\S)([A-Za-z_][A-Za-z0-9_]*)=")
SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|authorization|bearer|token|secret|password|credential|database_url)",
    re.IGNORECASE,
)
SENSITIVE_VALUE_PATTERN = re.compile(
    r"(Bearer\s+[A-Za-z0-9._~+/=-]+|sk-[A-Za-z0-9_-]+|[A-Za-z0-9_-]{24,})",
    re.IGNORECASE,
)
SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"\b(api[_-]?key|authorization|bearer|token|secret|password|credential|database_url)\b\s*[:=]\s*(?!<redacted>|<missing>|\s)(\S+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedAuditEvent:
    """Sanitized operational metadata extracted from one audit log line."""

    event_type: str
    timestamp: str
    run_id: str
    log_path: str
    dataset: str | None = None
    panel_mode: str | None = None
    execution_strategy: str | None = None
    judge_model: str | None = None
    role: str | None = None
    answer_id: int | None = None
    score: int | None = None
    latency_ms: int | None = None
    status_code: int | None = None
    retry_count: int | None = None
    error: str | None = None
    arbiter_reason: str | None = None
    matched_evaluation_id: int | None = None


@dataclass(frozen=True)
class ParsedAuditLog:
    """Parsed result for one log file."""

    run_id: str
    log_path: str
    started_at: str | None
    finished_at: str | None
    dataset: str | None
    panel_mode: str | None
    execution_strategy: str | None
    events: tuple[ParsedAuditEvent, ...]
    ignored_count: int
    unknown_count: int
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditParseReport:
    """Aggregate parser report for a manifest-driven parse."""

    manifest_path: str
    logs: tuple[ParsedAuditLog, ...]
    missing_logs: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()

    @property
    def parsed_event_count(self) -> int:
        return sum(len(log.events) for log in self.logs)

    @property
    def ignored_count(self) -> int:
        return sum(log.ignored_count for log in self.logs)

    @property
    def unknown_count(self) -> int:
        return sum(log.unknown_count for log in self.logs)


def parse_prod_logs_manifest(manifest_path: Path | str = DEFAULT_PROD_LOGS_MANIFEST) -> AuditParseReport:
    """Parse only logs explicitly listed in the production manifest."""
    manifest = Path(manifest_path)
    if not manifest.exists():
        return AuditParseReport(
            manifest_path=str(manifest),
            logs=(),
            problems=(f"manifest not found: {manifest}",),
        )

    log_paths = _read_manifest(manifest)
    logs: list[ParsedAuditLog] = []
    missing_logs: list[str] = []
    for raw_path in log_paths:
        log_path = raw_path if raw_path.is_absolute() else manifest.parent.parent.parent / raw_path
        if not log_path.exists() or not log_path.is_file():
            missing_logs.append(str(raw_path))
            continue
        logs.append(parse_audit_log(log_path))

    return AuditParseReport(
        manifest_path=str(manifest),
        logs=tuple(logs),
        missing_logs=tuple(missing_logs),
    )


def parse_audit_log(log_path: Path | str) -> ParsedAuditLog:
    """Parse one audit log without exposing raw sensitive fields."""
    path = Path(log_path)
    run_id = path.stem
    started_at: str | None = None
    finished_at: str | None = None
    dataset: str | None = None
    panel_mode: str | None = None
    execution_strategy: str | None = None
    events: list[ParsedAuditEvent] = []
    ignored_count = 0
    unknown_count = 0
    problems: list[str] = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        parsed_line = _parse_line(line)
        if parsed_line is None:
            ignored_count += 1
            continue

        timestamp, message, detail = parsed_line
        started_at = started_at or timestamp
        finished_at = timestamp

        if _contains_secret(line):
            problems.append(f"{path}:{line_number}: sensitive field redacted")

        values = _key_values(detail)
        if message == "execution_summary":
            panel_mode = _summary_value(detail, "Judge mode") or panel_mode
            execution_strategy = _summary_value(detail, "Judge execution strategy") or execution_strategy
            unknown_count += 1
            continue
        if message == "command_preview":
            dataset = _extract_cli_arg(detail, "--dataset") or dataset
            unknown_count += 1
            continue
        if message.startswith("START Counting eligible answers for ") or message.startswith(
            "START Selecting pending candidate answers for "
        ):
            dataset = values.get("dataset") or dataset
            unknown_count += 1
            continue
        if message == "audit_log_started":
            unknown_count += 1
            continue
        if message == "audit_log_finished":
            continue

        event = _event_from_message(
            message=message,
            timestamp=timestamp,
            run_id=run_id,
            log_path=str(path),
            values=values,
            dataset=dataset,
            panel_mode=panel_mode,
            execution_strategy=execution_strategy,
        )
        if event is None:
            unknown_count += 1
            continue
        events.append(event)

    return ParsedAuditLog(
        run_id=run_id,
        log_path=str(path),
        started_at=started_at,
        finished_at=finished_at,
        dataset=dataset,
        panel_mode=panel_mode,
        execution_strategy=execution_strategy,
        events=tuple(events),
        ignored_count=ignored_count,
        unknown_count=unknown_count,
        problems=tuple(problems),
    )


def format_audit_parse_report(report: AuditParseReport) -> str:
    """Format a secret-safe operational report."""
    sensitive_findings = sum(len(log.problems) for log in report.logs)
    problem_count = len(report.problems) + sensitive_findings + len(report.missing_logs)
    lines = [
        "Production audit log parse report",
        f"Manifest: {report.manifest_path}",
        f"Logs parsed: {len(report.logs)}",
        f"Events parsed: {report.parsed_event_count}",
        f"Ignored lines: {report.ignored_count}",
        f"Unknown events: {report.unknown_count}",
        f"Problems: {problem_count}",
        f"Sensitive findings redacted: {sensitive_findings}",
    ]
    if report.missing_logs:
        lines.append(f"Missing logs: {len(report.missing_logs)}")
    for problem in report.problems:
        lines.append(f"- {problem}")
    return "\n".join(lines)


def _read_manifest(manifest: Path) -> list[Path]:
    paths: list[Path] = []
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        paths.append(Path(item))
    return paths


def _parse_line(line: str) -> tuple[str, str, str | None] | None:
    match = AUDIT_LINE_PATTERN.match(line)
    if not match:
        return None
    timestamp = match.group(1).strip()
    try:
        datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    return timestamp, match.group(2).strip(), match.group(3).strip() if match.group(3) else None


def _event_from_message(
    *,
    message: str,
    timestamp: str,
    run_id: str,
    log_path: str,
    values: dict[str, str],
    dataset: str | None,
    panel_mode: str | None,
    execution_strategy: str | None,
) -> ParsedAuditEvent | None:
    if message in {"evaluation_parsed", "evaluation_skipped"}:
        return ParsedAuditEvent(
            event_type=message,
            timestamp=timestamp,
            run_id=run_id,
            log_path=log_path,
            dataset=dataset,
            panel_mode=values.get("mode") or panel_mode,
            execution_strategy=execution_strategy,
            judge_model=values.get("model"),
            role=values.get("role"),
            answer_id=_parse_int(values.get("answer_id")),
            score=_parse_int(values.get("score") or values.get("existing_score")),
            latency_ms=_parse_int(values.get("latency_ms")),
            status_code=_parse_int(values.get("status_code")),
            matched_evaluation_id=_parse_int(values.get("matched_evaluation_id")),
        )
    if message in {"adaptive_task_requeued", "adaptive_task_failed"}:
        return ParsedAuditEvent(
            event_type=message,
            timestamp=timestamp,
            run_id=run_id,
            log_path=log_path,
            dataset=dataset,
            panel_mode=panel_mode,
            execution_strategy=execution_strategy,
            judge_model=values.get("model"),
            role=values.get("role"),
            answer_id=_parse_int(values.get("answer_id")),
            status_code=_status_code_from_error(values.get("error")),
            retry_count=_parse_int(values.get("attempt")),
            error=_sanitize_text(values.get("error")),
        )
    if message.startswith("FAIL Running answer "):
        return ParsedAuditEvent(
            event_type="judge_call_failed",
            timestamp=timestamp,
            run_id=run_id,
            log_path=log_path,
            dataset=dataset,
            panel_mode=panel_mode,
            execution_strategy=execution_strategy,
            judge_model=values.get("model"),
            role=values.get("role"),
            answer_id=_parse_int(values.get("answer_id")),
            latency_ms=_parse_int(values.get("elapsed_ms")),
            status_code=_status_code_from_error(values.get("error")),
            error=_sanitize_text(values.get("error")),
        )
    if message == "arbiter_skipped":
        return ParsedAuditEvent(
            event_type=message,
            timestamp=timestamp,
            run_id=run_id,
            log_path=log_path,
            dataset=dataset,
            panel_mode=panel_mode,
            execution_strategy=execution_strategy,
            answer_id=_parse_int(values.get("answer_id")),
            arbiter_reason=_sanitize_text(values.get("reason") or "score_delta_below_threshold"),
        )
    return None


def _key_values(detail: str | None) -> dict[str, str]:
    if not detail:
        return {}
    matches = list(KEY_PATTERN.finditer(detail))
    values: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(detail)
        raw_value = detail[start:end].strip()
        if SENSITIVE_KEY_PATTERN.search(key):
            values[key] = "<redacted>"
        else:
            values[key] = _sanitize_text(raw_value)
    return values


def _sanitize_text(value: str | None) -> str | None:
    if value is None:
        return None
    sanitized = SENSITIVE_VALUE_PATTERN.sub("<redacted>", value)
    return sanitized.replace("\x00", "").strip()


def _contains_secret(text: str) -> bool:
    return bool(SENSITIVE_ASSIGNMENT_PATTERN.search(text) or re.search(r"\bBearer\s+\S+", text, re.IGNORECASE))


def _summary_value(summary: str | None, label: str) -> str | None:
    if not summary:
        return None
    prefix = f"{label}: "
    for part in (item.strip() for item in summary.split("|")):
        if part.startswith(prefix):
            return part.removeprefix(prefix).strip()
    return None


def _extract_cli_arg(command: str | None, option: str) -> str | None:
    if not command:
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    for index, part in enumerate(parts):
        if part == option and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _status_code_from_error(error: str | None) -> int | None:
    if not error:
        return None
    match = re.search(r"HTTP\s+(\d{3})", error)
    return int(match.group(1)) if match else None
