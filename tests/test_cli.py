"""Tests for the command-line interface baseline."""

from __future__ import annotations

import pytest

from atividade_2 import cli
from atividade_2.config import load_settings, resolve_runtime_config


def test_cli_help_exits_successfully() -> None:
    """The CLI help command should be available through argparse."""
    with pytest.raises(SystemExit) as exit_error:
        cli.main(["--help"])

    assert exit_error.value.code == 0


def test_run_judge_help_exits_successfully() -> None:
    """The judge command should expose runtime options."""
    with pytest.raises(SystemExit) as exit_error:
        cli.main(["run-judge", "--help"])

    assert exit_error.value.code == 0


def test_save_default_prompt_help_exits_successfully() -> None:
    with pytest.raises(SystemExit) as exit_error:
        cli.main(["save-default-prompt", "--help"])

    assert exit_error.value.code == 0


def test_import_evaluation_details_help_exits_successfully() -> None:
    with pytest.raises(SystemExit) as exit_error:
        cli.main(["import-evaluation-details", "--help"])

    assert exit_error.value.code == 0


def test_run_judge_help_exposes_batch_size(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_error:
        cli.main(["run-judge", "--help"])

    output = capsys.readouterr().out
    assert exit_error.value.code == 0
    assert "--batch-size" in output
    assert "--preflight-report" in output


def test_run_judge_dry_run_prints_single_summary(capsys: pytest.CaptureFixture[str], tmp_path) -> None:
    """Dry-run should resolve config without DB or HTTP calls."""
    audit_path = tmp_path / "audit.log"

    exit_code = cli.main(
        [
            "run-judge",
            "--panel-mode",
            "single",
            "--judge-model",
            "m-prometheus-14b",
            "--dataset",
            "J2",
            "--limit",
            "1",
            "--batch-size",
            "7",
            "--dry-run",
            "--audit-log",
            str(audit_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Judge mode: single" in output
    assert "Batch size: 7" in output
    assert "m-prometheus-14b -> Unbabel/M-Prometheus-14B | endpoint=" in output
    assert f"Audit log: {audit_path}" in output
    assert "test-key" not in output
    audit_text = audit_path.read_text(encoding="utf-8")
    assert "START Loading configuration" in audit_text
    assert "execution_summary" in audit_text
    assert "dry_run_finished" in audit_text


def test_execution_summary_includes_endpoint_hosts_without_api_keys() -> None:
    settings = load_settings(
        dotenv_path=None,
        env={
            "REMOTE_JUDGE_MODEL": "openai/gpt-oss-120b:free",
            "REMOTE_SECONDARY_JUDGE_MODEL": "llama-3.3-70b-versatile",
            "REMOTE_JUDGE_BASE_URL": "https://openrouter.ai/api/v1",
            "REMOTE_JUDGE_API_KEY": "openrouter-secret",
            "REMOTE_SECONDARY_JUDGE_BASE_URL": "https://api.groq.com/openai/v1",
            "REMOTE_SECONDARY_JUDGE_API_KEY": "groq-secret",
            "REMOTE_ARBITER_JUDGE_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai",
            "REMOTE_ARBITER_JUDGE_API_KEY": "gemini-secret",
            "REMOTE_ARBITER_JUDGE_MODEL": "gemini-2.5-flash",
        },
    )
    config = resolve_runtime_config(settings, panel_mode="2plus1")

    summary = cli.format_execution_summary(config)

    assert "- openai/gpt-oss-120b:free | endpoint=openrouter.ai" in summary
    assert "- llama-3.3-70b-versatile | endpoint=api.groq.com" in summary
    assert "- gemini-2.5-flash | endpoint=generativelanguage.googleapis.com" in summary
    assert "openrouter-secret" not in summary
    assert "groq-secret" not in summary
    assert "gemini-secret" not in summary
