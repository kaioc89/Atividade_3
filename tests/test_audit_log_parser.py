from __future__ import annotations

from pathlib import Path

from atividade_2.audit_log_parser import (
    format_audit_parse_report,
    parse_audit_log,
    parse_prod_logs_manifest,
)


def test_parse_valid_evaluation_event_extracts_operational_metadata(tmp_path: Path) -> None:
    log = tmp_path / "judge_run_20260503_224237.log"
    log.write_text(
        "\n".join(
            [
                "2026-05-03T22:42:37+00:00 | audit_log_started | path=outputs/audit/judge_run_20260503_224237.log",
                "2026-05-03T22:42:37+00:00 | execution_summary | Judge provider: remote_http | Judge mode: 2plus1 | Judge execution strategy: adaptive",
                "2026-05-03T22:42:37+00:00 | command_preview | .venv/bin/python -m atividade_2.cli run-judge --panel-mode 2plus1 --dataset J1 --batch-size 1",
                "2026-05-03T22:43:29+00:00 | evaluation_parsed | answer_id=1499 question_id=77 model=Unbabel/M-Prometheus-14B role=principal trigger=primary_panel score=1 latency_ms=31547 status_code=200 matched_evaluation_id=55",
                "2026-05-03T22:43:29+00:00 | audit_log_finished",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_audit_log(log)

    assert parsed.run_id == "judge_run_20260503_224237"
    assert parsed.started_at == "2026-05-03T22:42:37+00:00"
    assert parsed.finished_at == "2026-05-03T22:43:29+00:00"
    assert parsed.dataset == "J1"
    assert parsed.panel_mode == "2plus1"
    assert parsed.execution_strategy == "adaptive"
    assert parsed.ignored_count == 0
    event = parsed.events[0]
    assert event.answer_id == 1499
    assert event.judge_model == "Unbabel/M-Prometheus-14B"
    assert event.role == "principal"
    assert event.score == 1
    assert event.latency_ms == 31547
    assert event.status_code == 200
    assert event.matched_evaluation_id == 55


def test_missing_log_invalid_line_and_missing_optional_field_are_reported(tmp_path: Path) -> None:
    manifest = tmp_path / "outputs" / "audit" / "prod_logs_manifest.txt"
    manifest.parent.mkdir(parents=True)
    log = manifest.parent / "sample.log"
    log.write_text(
        "\n".join(
            [
                "invalid line",
                "2026-05-03T22:43:29+00:00 | evaluation_parsed | answer_id=1499 model=test-model role=principal score=1",
            ]
        ),
        encoding="utf-8",
    )
    manifest.write_text("outputs/audit/sample.log\noutputs/audit/missing.log\n", encoding="utf-8")

    report = parse_prod_logs_manifest(manifest)

    assert len(report.logs) == 1
    assert report.missing_logs == ("outputs/audit/missing.log",)
    assert report.ignored_count == 1
    event = report.logs[0].events[0]
    assert event.latency_ms is None
    assert event.status_code is None


def test_manifest_parser_reads_only_listed_logs(tmp_path: Path) -> None:
    manifest = tmp_path / "outputs" / "audit" / "prod_logs_manifest.txt"
    manifest.parent.mkdir(parents=True)
    listed = manifest.parent / "listed.log"
    unlisted = manifest.parent / "unlisted.log"
    listed.write_text(
        "2026-05-03T22:43:29+00:00 | evaluation_parsed | answer_id=1 model=listed role=principal score=4\n",
        encoding="utf-8",
    )
    unlisted.write_text(
        "2026-05-03T22:43:29+00:00 | evaluation_parsed | answer_id=2 model=unlisted role=principal score=0\n",
        encoding="utf-8",
    )
    manifest.write_text("outputs/audit/listed.log\n", encoding="utf-8")

    report = parse_prod_logs_manifest(manifest)

    assert report.parsed_event_count == 1
    assert report.logs[0].events[0].judge_model == "listed"


def test_sanitizes_secret_material_from_events_and_report(tmp_path: Path) -> None:
    log = tmp_path / "secret.log"
    log.write_text(
        "2026-05-03T22:42:55+00:00 | adaptive_task_requeued | provider=remote_http api_key=sk-test-secret-token model=model-a answer_id=10 role=principal attempt=1 error=Remote judge returned HTTP 503 Authorization=Bearer abcdefghijklmnopqrstuvwxyz\n",
        encoding="utf-8",
    )

    parsed = parse_audit_log(log)
    report_text = format_audit_parse_report(
        parse_prod_logs_manifest(_manifest_for(tmp_path, "outputs/audit/secret.log", log))
    )

    assert parsed.events[0].error == "Remote judge returned HTTP 503"
    assert "sk-test-secret-token" not in str(parsed)
    assert "abcdefghijklmnopqrstuvwxyz" not in str(parsed)
    assert "sk-test-secret-token" not in report_text
    assert "abcdefghijklmnopqrstuvwxyz" not in report_text
    assert parsed.problems


def _manifest_for(tmp_path: Path, entry: str, log: Path) -> Path:
    manifest = tmp_path / "outputs" / "audit" / "prod_logs_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    target = manifest.parent / log.name
    target.write_text(log.read_text(encoding="utf-8"), encoding="utf-8")
    manifest.write_text(f"{entry}\n", encoding="utf-8")
    return manifest
