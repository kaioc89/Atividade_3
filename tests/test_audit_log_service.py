from __future__ import annotations

from pathlib import Path

from atividade_2.audit_log_service import AuditLogSummaryService


def test_service_returns_operational_summary_from_valid_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    log = manifest.parent / "sample.log"
    log.write_text(
        "\n".join(
            [
                "2026-05-03T22:42:37+00:00 | execution_summary | Judge mode: 2plus1 | Judge execution strategy: adaptive",
                "2026-05-03T22:42:37+00:00 | command_preview | .venv/bin/python -m atividade_2.cli run-judge --dataset J2",
                "2026-05-03T22:43:29+00:00 | evaluation_parsed | answer_id=1 model=model-a role=principal score=4 latency_ms=100 status_code=200",
                "2026-05-03T22:43:30+00:00 | adaptive_task_requeued | model=model-b role=secondary answer_id=2 attempt=1 error=Remote judge returned HTTP 503",
                "2026-05-03T22:43:31+00:00 | FAIL Running answer 3 | model=model-b role=secondary answer_id=3 elapsed_ms=200 error=Remote judge returned HTTP 500",
            ]
        ),
        encoding="utf-8",
    )
    manifest.write_text("outputs/audit/sample.log\n", encoding="utf-8")

    summary = AuditLogSummaryService(manifest).load()

    assert summary["available"] is True
    assert summary["totals"]["logs"] == 1
    assert summary["totals"]["events"] == 3
    assert summary["totals"]["retries"] == 1
    assert summary["totals"]["failures"] == 1
    log_summary = summary["logs"][0]
    assert log_summary["run_id"] == "sample"
    assert log_summary["dataset"] == "J2"
    assert log_summary["panel_mode"] == "2plus1"
    assert log_summary["execution_strategy"] == "adaptive"
    assert log_summary["models"] == ["model-a", "model-b"]
    assert log_summary["roles"] == ["principal", "secondary"]
    assert log_summary["average_latency_ms"] == 150


def test_service_returns_controlled_empty_response_when_manifest_is_missing(tmp_path: Path) -> None:
    summary = AuditLogSummaryService(tmp_path / "outputs" / "audit" / "prod_logs_manifest.txt").load()

    assert summary["available"] is False
    assert summary["logs"] == []
    assert summary["totals"]["events"] == 0
    assert "manifest not found" in summary["problems"][0]


def test_service_does_not_use_broad_log_glob_by_default(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)
    listed = manifest.parent / "listed.log"
    listed.write_text("2026-05-03T22:43:29+00:00 | evaluation_parsed | model=listed role=principal\n", encoding="utf-8")
    (manifest.parent / "unlisted.log").write_text(
        "2026-05-03T22:43:29+00:00 | evaluation_parsed | model=unlisted role=principal\n",
        encoding="utf-8",
    )
    manifest.write_text("outputs/audit/listed.log\n", encoding="utf-8")

    def fail_glob(self: Path, pattern: str):
        raise AssertionError(f"unexpected glob: {self} {pattern}")

    monkeypatch.setattr(Path, "glob", fail_glob)

    summary = AuditLogSummaryService(manifest).load()

    assert summary["totals"]["events"] == 1
    assert summary["logs"][0]["models"] == ["listed"]


def test_service_exposes_only_sanitized_secret_metadata(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    log = manifest.parent / "secret.log"
    log.write_text(
        "2026-05-03T22:42:55+00:00 | adaptive_task_requeued | api_key=sk-test-secret-token model=model-a role=principal answer_id=10 attempt=1 error=Remote judge returned HTTP 503 Authorization=Bearer abcdefghijklmnopqrstuvwxyz\n",
        encoding="utf-8",
    )
    manifest.write_text("outputs/audit/secret.log\n", encoding="utf-8")

    summary_text = str(AuditLogSummaryService(manifest).load())

    assert "sk-test-secret-token" not in summary_text
    assert "abcdefghijklmnopqrstuvwxyz" not in summary_text
    assert "sensitive field redacted" in summary_text


def test_service_returns_partial_summary_when_one_manifest_log_fails(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)
    good = manifest.parent / "good.log"
    bad = manifest.parent / "bad.log"
    good.write_text("2026-05-03T22:43:29+00:00 | evaluation_parsed | model=good role=principal\n", encoding="utf-8")
    bad.write_text("2026-05-03T22:43:29+00:00 | evaluation_parsed | model=bad role=principal\n", encoding="utf-8")
    manifest.write_text("outputs/audit/good.log\noutputs/audit/bad.log\n", encoding="utf-8")
    original_read_text = Path.read_text

    def read_text(self: Path, *args, **kwargs):
        if self == bad:
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    summary = AuditLogSummaryService(manifest).load()

    assert summary["available"] is True
    assert summary["totals"]["logs"] == 1
    assert summary["totals"]["events"] == 1
    assert summary["missing_logs"] == ["outputs/audit/bad.log"]
    assert "read failed" in summary["problems"][0]


def _manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "outputs" / "audit" / "prod_logs_manifest.txt"
    manifest.parent.mkdir(parents=True)
    return manifest
