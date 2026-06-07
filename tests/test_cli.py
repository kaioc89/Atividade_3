"""Tests for the command-line interface baseline."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from atividade_2 import cli
from atividade_2.config import load_settings, resolve_runtime_config
from atividade_2.contracts import CandidateModelAssignment
from atividade_2.provider_catalogs import (
    DEFAULT_FEATHERLESS_CATALOG_MAX_RESPONSE_BYTES,
    DEFAULT_OPENROUTER_CATALOG_MAX_RESPONSE_BYTES,
    FakeProviderCatalogClient,
    FeatherlessCatalogClient,
    OpenRouterCatalogClient,
    ProviderCatalogError,
)
from atividade_2.provider_validation_contracts import ProviderModelCatalogEntry
from atividade_2.repositories import _default_candidate_model_assignments
from atividade_2.run_candidates_rag_service import CandidateRunSummary, RunCandidatesRagResult


class FakeRunCandidatesRagService:
    def __init__(
        self,
        *,
        result: RunCandidatesRagResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or RunCandidatesRagResult(
            dry_run=True,
            audit_log="logs/candidate-rag-test.log",
            execution_summary=(
                "Dataset: J1\n"
                "Candidate model: openai/gpt-5.4\n"
                "Provider: remote_http\n"
                "Batch size: 2\n"
                "Question id: -\n"
                "Question range: -"
            ),
            batch_size=2,
            dataset="J1",
            model_name="openai/gpt-5.4",
            provider="remote_http",
        )
        self.error = error
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


class FakeAssignmentRepository:
    def __init__(self, assignments: tuple[CandidateModelAssignment, ...]) -> None:
        self._assignments = assignments

    def list_candidate_model_assignments(self) -> tuple[CandidateModelAssignment, ...]:
        return self._assignments


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


def test_run_candidates_rag_help_exits_successfully() -> None:
    with pytest.raises(SystemExit) as exit_error:
        cli.main(["run-candidates-rag", "--help"])

    assert exit_error.value.code == 0


def test_run_candidates_rag_help_exposes_candidate_execution_strategy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_error:
        cli.main(["run-candidates-rag", "--help"])

    output = capsys.readouterr().out
    assert exit_error.value.code == 0
    assert "--candidate-execution-strategy" in output
    assert "--candidate-parallel-max-workers" in output


def test_save_default_prompt_help_exits_successfully() -> None:
    with pytest.raises(SystemExit) as exit_error:
        cli.main(["save-default-prompt", "--help"])

    assert exit_error.value.code == 0


def test_validate_provider_models_help_exits_successfully() -> None:
    with pytest.raises(SystemExit) as exit_error:
        cli.main(["validate-provider-models", "--help"])

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
    assert "av3_j2_com_rag" in output


def test_build_parser_accepts_av3_j2_judge_input_source() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "run-judge",
            "--judge-input-source",
            "av3_j2_com_rag",
            "--dataset",
            "J2",
        ]
    )

    assert args.judge_input_source == "av3_j2_com_rag"
    assert args.dataset == "J2"


def test_build_parser_exposes_run_candidates_rag() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "run-candidates-rag",
            "--dataset",
            "J1",
            "--candidate-model",
            "openai/gpt-5.4",
            "--provider",
            "remote_http",
            "--batch-size",
            "2",
        ]
    )

    assert args.command == "run-candidates-rag"
    assert args.handler is cli.run_candidates_rag_command


def test_build_parser_exposes_validate_provider_models() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["validate-provider-models", "--provider", "openrouter", "--json"])

    assert args.command == "validate-provider-models"
    assert args.handler is cli.validate_provider_models_command
    assert args.provider == ["openrouter"]
    assert args.json is True


def test_validate_provider_models_parser_supports_repeated_provider() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "validate-provider-models",
            "--provider",
            "openrouter",
            "--provider",
            "featherless",
        ]
    )

    assert args.provider == ["openrouter", "featherless"]


def test_build_provider_catalog_clients_uses_env_overrides_for_catalog_limits() -> None:
    clients = cli._build_provider_catalog_clients(
        {
            "OPENROUTER_CATALOG_MAX_RESPONSE_BYTES": "7000000",
            "FEATHERLESS_CATALOG_MAX_RESPONSE_BYTES": "42000000",
        }
    )

    openrouter_client = clients["openrouter"]
    featherless_client = clients["featherless"]

    assert isinstance(openrouter_client, OpenRouterCatalogClient)
    assert isinstance(featherless_client, FeatherlessCatalogClient)
    assert openrouter_client.max_response_bytes == 7_000_000
    assert featherless_client.max_response_bytes == 42_000_000


def test_build_provider_catalog_clients_uses_provider_default_catalog_limits() -> None:
    clients = cli._build_provider_catalog_clients({})

    openrouter_client = clients["openrouter"]
    featherless_client = clients["featherless"]

    assert isinstance(openrouter_client, OpenRouterCatalogClient)
    assert isinstance(featherless_client, FeatherlessCatalogClient)
    assert openrouter_client.max_response_bytes == DEFAULT_OPENROUTER_CATALOG_MAX_RESPONSE_BYTES
    assert featherless_client.max_response_bytes == DEFAULT_FEATHERLESS_CATALOG_MAX_RESPONSE_BYTES


def test_run_candidates_rag_parser_accepts_j1() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "run-candidates-rag",
            "--dataset",
            "J1",
            "--candidate-model",
            "openai/gpt-5.4",
            "--provider",
            "remote_http",
            "--batch-size",
            "2",
        ]
    )

    assert args.dataset == "J1"


def test_run_candidates_rag_parser_accepts_j2() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "run-candidates-rag",
            "--dataset",
            "J2",
            "--candidate-model",
            "openai/gpt-5.4",
            "--provider",
            "remote_http",
            "--batch-size",
            "2",
        ]
    )

    assert args.dataset == "J2"


def test_run_candidates_rag_parser_accepts_candidate_execution_strategy() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "run-candidates-rag",
            "--dataset",
            "J1",
            "--candidate-model",
            "openai/gpt-5.4",
            "--provider",
            "remote_http",
            "--batch-size",
            "2",
            "--candidate-execution-strategy",
            "adaptive",
            "--candidate-parallel-max-workers",
            "3",
        ]
    )

    assert args.candidate_execution_strategy == "adaptive"
    assert args.candidate_parallel_max_workers == 3


def test_run_candidates_rag_parser_requires_candidate_model() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exit_error:
        parser.parse_args(
            [
                "run-candidates-rag",
                "--dataset",
                "J1",
                "--provider",
                "remote_http",
                "--batch-size",
                "2",
            ]
        )

    assert exit_error.value.code == 2


def test_run_candidates_rag_parser_validates_positive_batch_size() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exit_error:
        parser.parse_args(
            [
                "run-candidates-rag",
                "--dataset",
                "J1",
                "--candidate-model",
                "openai/gpt-5.4",
                "--provider",
                "remote_http",
                "--batch-size",
                "0",
            ]
        )

    assert exit_error.value.code == 2


def test_validate_provider_models_returns_nonzero_when_checked_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from atividade_2 import repositories

    assignments = tuple(
        replace(assignment, av3_provider_model_id="openai/gpt-5-missing")
        if assignment.id_modelo_av2 == 14
        else assignment
        for assignment in _default_candidate_model_assignments()
    )
    monkeypatch.setattr(repositories, "JudgeRepository", lambda connection: FakeAssignmentRepository(assignments))
    monkeypatch.setattr(cli, "load_settings", lambda: SimpleNamespace(database_url="postgresql://example.invalid/app"))
    monkeypatch.setattr(cli, "load_env", lambda: {})
    monkeypatch.setattr(cli, "connect", lambda _database_url: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        cli,
        "_build_provider_catalog_clients",
        lambda _env: {
            "openrouter": FakeProviderCatalogClient(
                entries=(
                    ProviderModelCatalogEntry(
                        provider="openrouter",
                        model_id="meta-llama/llama-3.2-3b-instruct",
                    ),
                    ProviderModelCatalogEntry(provider="openrouter", model_id="openai/gpt-5-chat"),
                    ProviderModelCatalogEntry(
                        provider="openrouter",
                        model_id="google/gemini-3.5-flash",
                    ),
                    ProviderModelCatalogEntry(
                        provider="openrouter",
                        model_id="x-ai/grok-4.3",
                    ),
                )
            ),
            "featherless": FakeProviderCatalogClient(entries=()),
        },
    )

    exit_code = cli.main(
        [
            "validate-provider-models",
            "--provider",
            "openrouter",
            "--include-pending-confirmation",
            "--json",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 1
    assert '"missing": 1' in output


def test_validate_provider_models_reports_jose_grok_under_openrouter_when_pending_enabled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from atividade_2 import repositories

    monkeypatch.setattr(
        repositories,
        "JudgeRepository",
        lambda connection: FakeAssignmentRepository(_default_candidate_model_assignments()),
    )
    monkeypatch.setattr(cli, "load_settings", lambda: SimpleNamespace(database_url="postgresql://example.invalid/app"))
    monkeypatch.setattr(cli, "load_env", lambda: {})
    monkeypatch.setattr(cli, "connect", lambda _database_url: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        cli,
        "_build_provider_catalog_clients",
        lambda _env: {
            "openrouter": FakeProviderCatalogClient(
                entries=(
                    ProviderModelCatalogEntry(
                        provider="openrouter",
                        model_id="meta-llama/llama-3.2-3b-instruct",
                    ),
                    ProviderModelCatalogEntry(provider="openrouter", model_id="openai/gpt-5-chat"),
                    ProviderModelCatalogEntry(
                        provider="openrouter",
                        model_id="google/gemini-3.5-flash",
                    ),
                    ProviderModelCatalogEntry(
                        provider="openrouter",
                        model_id="x-ai/grok-4.3",
                    ),
                )
            ),
            "featherless": FakeProviderCatalogClient(entries=()),
        },
    )

    exit_code = cli.main(
        [
            "validate-provider-models",
            "--provider",
            "openrouter",
            "--include-pending-confirmation",
            "--json",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"total_assignments": 5' in output
    assert '"id_modelo_av2": 5' in output
    assert '"id_modelo_av2": 9' in output
    assert '"id_modelo_av2": 15' in output
    assert '"av3_provider": "openrouter"' in output
    assert '"av3_provider_model_id": "x-ai/grok-4.3"' in output
    assert '"status": "found"' in output


def test_validate_provider_models_returns_two_when_provider_catalog_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from atividade_2 import repositories

    monkeypatch.setattr(
        repositories,
        "JudgeRepository",
        lambda connection: FakeAssignmentRepository(_default_candidate_model_assignments()),
    )
    monkeypatch.setattr(cli, "load_settings", lambda: SimpleNamespace(database_url="postgresql://example.invalid/app"))
    monkeypatch.setattr(cli, "load_env", lambda: {})
    monkeypatch.setattr(cli, "connect", lambda _database_url: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        cli,
        "_build_provider_catalog_clients",
        lambda _env: {
            "openrouter": FakeProviderCatalogClient(error=ProviderCatalogError("OpenRouter down")),
            "featherless": FakeProviderCatalogClient(entries=()),
        },
    )

    exit_code = cli.main(["validate-provider-models", "--provider", "openrouter", "--json"])

    output = capsys.readouterr().out

    assert exit_code == 2
    assert '"total_assignments": 4' in output
    assert '"provider_errors": 4' in output


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


def test_run_candidates_rag_dry_run_calls_service_with_dry_run_true(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = FakeRunCandidatesRagService()
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: SimpleNamespace(
            remote_candidate_temperature=0.2,
            remote_candidate_max_tokens=1024,
            remote_candidate_top_p=0.9,
            remote_candidate_context_safety_margin_tokens=512,
            remote_candidate_context_window_tokens=None,
            remote_candidate_retry_on_context_window=False,
        ),
    )
    monkeypatch.setattr(cli, "RunCandidatesRagService", lambda: service)

    exit_code = cli.main(
        [
            "run-candidates-rag",
            "--dataset",
            "J1",
            "--candidate-model",
            "openai/gpt-5.4",
            "--provider",
            "remote_http",
            "--batch-size",
            "2",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert service.requests[0].dry_run is True
    assert service.requests[0].remote_candidate_temperature == 0.2
    assert service.requests[0].remote_candidate_max_tokens == 1024
    assert service.requests[0].remote_candidate_top_p == 0.9
    assert service.requests[0].remote_candidate_context_safety_margin_tokens == 512
    assert service.requests[0].remote_candidate_context_window_tokens is None
    assert "Dataset: J1" in output
    assert "Batch size: 2" in output


def test_run_candidates_rag_passes_audit_log_and_question_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    service = FakeRunCandidatesRagService()
    audit_path = tmp_path / "candidate-audit.log"
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: SimpleNamespace(
            remote_candidate_temperature=0.2,
            remote_candidate_max_tokens=1024,
            remote_candidate_top_p=0.9,
            remote_candidate_context_safety_margin_tokens=512,
            remote_candidate_context_window_tokens=None,
            remote_candidate_retry_on_context_window=False,
        ),
    )
    monkeypatch.setattr(cli, "RunCandidatesRagService", lambda: service)

    exit_code = cli.main(
        [
            "run-candidates-rag",
            "--dataset",
            "J2",
            "--candidate-model",
            "candidate-j2",
            "--provider",
            "remote_http",
            "--batch-size",
            "3",
            "--audit-log",
            str(audit_path),
            "--question-id",
            "202",
            "--question-sequence-start",
            "10",
            "--question-sequence-end",
            "20",
            "--prompt-id",
            "7",
            "--retrieval-run-id",
            "21",
            "--no-skip-existing",
            "--no-audit-animation",
        ]
    )

    request = service.requests[0]
    assert exit_code == 0
    assert request.audit_log == str(audit_path)
    assert request.question_id == 202
    assert request.question_sequence_start == 10
    assert request.question_sequence_end == 20
    assert request.prompt_id == 7
    assert request.retrieval_run_id == 21
    assert request.skip_existing_successful is False
    assert request.no_audit_animation is True


def test_run_candidates_rag_passes_retry_on_context_window_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeRunCandidatesRagService()
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: SimpleNamespace(
            remote_candidate_temperature=0.2,
            remote_candidate_max_tokens=1024,
            remote_candidate_top_p=0.9,
            remote_candidate_context_safety_margin_tokens=512,
            remote_candidate_context_window_tokens=None,
            remote_candidate_retry_on_context_window=False,
        ),
    )
    monkeypatch.setattr(cli, "RunCandidatesRagService", lambda: service)

    exit_code = cli.main(
        [
            "run-candidates-rag",
            "--dataset",
            "J1",
            "--candidate-model",
            "candidate-j1",
            "--provider",
            "remote_http",
            "--batch-size",
            "1",
            "--retry-on-context-window",
        ]
    )

    assert exit_code == 0
    assert service.requests[0].remote_candidate_retry_on_context_window is True


def test_run_candidates_rag_cli_strategy_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeRunCandidatesRagService()
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: SimpleNamespace(
            remote_candidate_temperature=0.2,
            remote_candidate_max_tokens=1024,
            remote_candidate_top_p=0.9,
            remote_candidate_context_safety_margin_tokens=512,
            remote_candidate_context_window_tokens=None,
            remote_candidate_retry_on_context_window=False,
            candidate_execution_strategy="sequential",
            candidate_parallel_max_workers=2,
        ),
    )
    monkeypatch.setattr(cli, "RunCandidatesRagService", lambda: service)

    exit_code = cli.main(
        [
            "run-candidates-rag",
            "--dataset",
            "J1",
            "--candidate-model",
            "candidate-j1",
            "--provider",
            "remote_http",
            "--batch-size",
            "1",
            "--candidate-execution-strategy",
            "parallel",
            "--candidate-parallel-max-workers",
            "4",
        ]
    )

    assert exit_code == 0
    assert service.requests[0].candidate_execution_strategy == "parallel"
    assert service.requests[0].candidate_parallel_max_workers == 4


def test_run_candidates_rag_command_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = FakeRunCandidatesRagService(
        result=RunCandidatesRagResult(
            dry_run=False,
            audit_log="logs/candidate-rag-real.log",
            execution_summary=(
                "Dataset: J2\n"
                "Candidate model: candidate-j2\n"
                "Provider: remote_http\n"
                "Batch size: 3\n"
                "Question id: 202\n"
                "Question range: 10-20"
            ),
            batch_size=3,
            dataset="J2",
            model_name="candidate-j2",
            provider="remote_http",
            runtime_config_summary=(
                "Candidate runtime preflight:\n"
                "  model: candidate-j2\n"
                "  technical provider: remote_http\n"
                "  av3 provider: featherless\n"
                "  base_url: https://api.featherless.ai/v1\n"
                "  api_key: <set>\n"
                "  temperature: 0.2\n"
                "  top_p: 0.9\n"
                "  requested_max_tokens: 1024\n"
                "  final_max_tokens: 1024\n"
                "  context_window_tokens: unknown\n"
                "  safety_margin_tokens: 512\n"
                "  save_raw_response: false"
            ),
            candidate_run_id=501,
            retrieval_run_id=21,
            prompt_id=7,
            summary=CandidateRunSummary(
                selected_questions=3,
                processed_questions=2,
                successful_answers=1,
                failed_answers=1,
                skipped_questions=1,
            ),
        )
    )
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: SimpleNamespace(
            remote_candidate_temperature=0.2,
            remote_candidate_max_tokens=1024,
            remote_candidate_top_p=0.9,
            remote_candidate_context_safety_margin_tokens=512,
            remote_candidate_context_window_tokens=None,
            remote_candidate_retry_on_context_window=False,
        ),
    )
    monkeypatch.setattr(cli, "RunCandidatesRagService", lambda: service)

    exit_code = cli.main(
        [
            "run-candidates-rag",
            "--dataset",
            "J2",
            "--candidate-model",
            "candidate-j2",
            "--provider",
            "remote_http",
            "--batch-size",
            "3",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Candidate runtime preflight:" in output
    assert "api_key: <set>" in output
    assert "Execution result:" in output
    assert "Candidate run id: 501" in output
    assert "Retrieval run id: 21" in output
    assert "Prompt id: 7" in output
    assert "Selected questions: 3" in output
    assert "Processed questions: 2" in output
    assert "Successful answers: 1" in output
    assert "Failed answers: 1" in output
    assert "Skipped questions: 1" in output


def test_run_candidates_rag_returns_exit_code_2_on_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeRunCandidatesRagService(error=ValueError("invalid candidate configuration"))
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: SimpleNamespace(
            remote_candidate_temperature=0.2,
            remote_candidate_max_tokens=1024,
            remote_candidate_top_p=0.9,
            remote_candidate_context_safety_margin_tokens=512,
            remote_candidate_context_window_tokens=None,
            remote_candidate_retry_on_context_window=False,
        ),
    )
    monkeypatch.setattr(cli, "RunCandidatesRagService", lambda: service)

    with pytest.raises(SystemExit) as exit_error:
        cli.main(
            [
                "run-candidates-rag",
                "--dataset",
                "J1",
                "--candidate-model",
                "openai/gpt-5.4",
                "--provider",
                "remote_http",
                "--batch-size",
                "2",
            ]
        )

    assert exit_error.value.code == 2


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
