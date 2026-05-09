"""Read-only operational summaries for production audit logs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .audit_log_parser import DEFAULT_PROD_LOGS_MANIFEST, ParsedAuditLog, parse_prod_logs_manifest


@dataclass(frozen=True)
class AuditLogSummaryService:
    """Load manifest-confirmed production logs without affecting persisted metrics."""

    manifest_path: Path | str = DEFAULT_PROD_LOGS_MANIFEST

    def load(self) -> dict:
        try:
            report = parse_prod_logs_manifest(self.manifest_path)
        except OSError as error:
            return _empty_response(str(self.manifest_path), problems=(f"audit log read failed: {error}",))

        return {
            "source": "prod_logs_manifest",
            "manifest_path": report.manifest_path,
            "available": bool(report.logs),
            "totals": {
                "logs": len(report.logs),
                "events": report.parsed_event_count,
                "retries": sum(_retry_count(log) for log in report.logs),
                "failures": sum(_failure_count(log) for log in report.logs),
                "ignored": report.ignored_count,
                "unknown": report.unknown_count,
                "missing_logs": len(report.missing_logs),
                "problems": len(report.problems) + sum(len(log.problems) for log in report.logs),
            },
            "logs": [_serialize_log(log) for log in report.logs],
            "missing_logs": list(report.missing_logs),
            "problems": list(report.problems),
        }


def _empty_response(manifest_path: str, *, problems: tuple[str, ...] = ()) -> dict:
    return {
        "source": "prod_logs_manifest",
        "manifest_path": manifest_path,
        "available": False,
        "totals": {
            "logs": 0,
            "events": 0,
            "retries": 0,
            "failures": 0,
            "ignored": 0,
            "unknown": 0,
            "missing_logs": 0,
            "problems": len(problems),
        },
        "logs": [],
        "missing_logs": [],
        "problems": list(problems),
    }


def _serialize_log(log: ParsedAuditLog) -> dict:
    events = [asdict(event) for event in log.events]
    latencies = [event.latency_ms for event in log.events if event.latency_ms is not None]
    return {
        "run_id": log.run_id,
        "log_path": log.log_path,
        "started_at": log.started_at,
        "finished_at": log.finished_at,
        "dataset": log.dataset,
        "panel_mode": log.panel_mode,
        "execution_strategy": log.execution_strategy,
        "models": sorted({event.judge_model for event in log.events if event.judge_model}),
        "roles": sorted({event.role for event in log.events if event.role}),
        "total_events": len(log.events),
        "total_retries": _retry_count(log),
        "failures": _failure_count(log),
        "average_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "ignored_events": log.ignored_count,
        "unknown_events": log.unknown_count,
        "problems": list(log.problems),
        "events": events,
    }


def _retry_count(log: ParsedAuditLog) -> int:
    return sum(event.retry_count or 0 for event in log.events)


def _failure_count(log: ParsedAuditLog) -> int:
    return sum(1 for event in log.events if event.event_type in {"adaptive_task_failed", "judge_call_failed"})
