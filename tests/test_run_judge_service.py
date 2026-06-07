from __future__ import annotations

from atividade_2.config import load_settings
from atividade_2.contracts import CandidateAnswerContext, EligibilitySummary, JudgeRawResponse
from atividade_2.repositories import InMemoryJudgeRepository
from atividade_2.run_judge_service import RunJudgeRequest, RunJudgeService


BASE_ENV = {
    "REMOTE_JUDGE_BASE_URL": "https://example.invalid/v1",
    "REMOTE_JUDGE_API_KEY": "test-key",
    "REMOTE_JUDGE_MODEL": "gpt-oss-120b",
    "REMOTE_SECONDARY_JUDGE_MODEL": "llama-3.3-70b-instruct",
    "REMOTE_ARBITER_JUDGE_MODEL": "m-prometheus-14b",
}


class FakeConnection:
    def close(self) -> None:
        return None


class EligibilityRepository(InMemoryJudgeRepository):
    def __init__(self) -> None:
        super().__init__()
        self.eligibility_calls = 0
        self.select_calls: list[tuple[str, str, int]] = []
        self.eligibility_sources: list[tuple[str, str, int]] = []

    def ensure_schema(self) -> None:
        return None

    def select_pending_candidate_answers(self, *, dataset, batch_size, required_evaluations, judge_input_source="av2"):
        self.select_calls.append((judge_input_source, dataset, batch_size))
        return [
            CandidateAnswerContext(
                av1_answer_id=1,
                candidate_answer_id=None,
                question_id=1,
                dataset_name="OAB_Exames",
                question_text="Enunciado",
                reference_answer="Resposta ouro",
                candidate_answer="Resposta candidata",
                candidate_model="modelo",
            )
        ]

    def summarize_eligibility(self, *, dataset, batch_size, required_evaluations, judge_input_source="av2"):
        self.eligibility_calls += 1
        self.eligibility_sources.append((judge_input_source, dataset, batch_size))
        if self.eligibility_calls == 1:
            return EligibilitySummary(missing=1, failed=0, successful=8, batch_size=batch_size, will_process=1)
        return EligibilitySummary(missing=0, failed=0, successful=9, batch_size=batch_size, will_process=0)


class FakeClient:
    def __init__(self, settings) -> None:
        return None

    def judge(self, prompt: str, model: str, *, requested_model: str | None = None, endpoint_key: str | None = None):
        return JudgeRawResponse(text='{"score": 5, "rationale": "ok"}', provider="fake", model=model, latency_ms=1)


def test_dry_run_does_not_connect_to_database(tmp_path) -> None:
    def fail_connect(database_url: str):
        raise AssertionError("dry-run must not connect to PostgreSQL")

    service = RunJudgeService(
        settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV),
        connect_func=fail_connect,
    )

    result = service.run(
        RunJudgeRequest(
            panel_mode="single",
            judge_model="m-prometheus-14b",
            batch_size=1,
            dry_run=True,
            audit_log=str(tmp_path / "dry-run.log"),
            no_audit_animation=True,
        )
    )

    assert result.dry_run is True
    assert result.summary is None
    assert "Judge mode: single" in result.execution_summary
    assert "test-key" not in result.execution_summary
    assert "--dry-run" in result.command_preview


def test_describe_config_exposes_av3_j2_source_and_preset() -> None:
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV))

    description = service.describe_config()

    assert "av3_j2_com_rag" in description["supported"]["judge_input_sources"]
    assert {
        "name": "AV3 J2 Com_RAG",
        "panel_mode": "single",
        "dataset": "J2",
        "judge_input_source": "av3_j2_com_rag",
        "batch_size": 1,
    } in description["presets"]


def test_preflight_report_does_not_connect_to_database(tmp_path) -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"

    def fail_connect(database_url: str):
        raise AssertionError("preflight report must not connect to PostgreSQL")

    service = RunJudgeService(
        settings_loader=lambda: load_settings(dotenv_path=None, env=env),
        connect_func=fail_connect,
    )

    result = service.run(
        RunJudgeRequest(
            panel_mode="2plus1",
            batch_size=2,
            preflight_report=True,
            audit_log=str(tmp_path / "preflight.log"),
            no_audit_animation=True,
        )
    )

    assert result.summary is None
    assert result.preflight_report is not None
    assert "Preflight report:" in result.preflight_report
    assert "Execution strategy: adaptive" in result.preflight_report
    assert "Priority order: judge_1 -> judge_2 -> arbiter" in result.preflight_report
    assert "test-key" not in result.preflight_report
    assert "--preflight-report" in result.command_preview


def test_describe_config_is_secret_safe() -> None:
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV))

    config = service.describe_config()

    assert config["endpoints"]["JUDGE"] == {"host": "example.invalid", "has_api_key": True}
    assert "test-key" not in str(config)


def test_invalid_configuration_is_reported_in_config_description() -> None:
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env={}))

    config = service.describe_config()

    assert "configuration_error" in config
    assert "REMOTE_JUDGE_BASE_URL is required" in config["configuration_error"]


def test_settings_loader_error_is_reported_in_config_description() -> None:
    env = dict(BASE_ENV)
    env["APP_ENV"] = "local"
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env=env))

    config = service.describe_config()

    assert "configuration_error" in config
    assert "APP_ENV must be one of: dev, prod, test" in config["configuration_error"]


def test_resolve_applies_web_endpoint_and_advanced_overrides() -> None:
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV))

    resolved = service.resolve(
        RunJudgeRequest(
            panel_mode="2plus1",
            remote_judge_base_url="https://judge1.example.invalid/v1",
            remote_judge_api_key="key-1",
            remote_secondary_judge_base_url="https://judge2.example.invalid/v1",
            remote_secondary_judge_api_key="key-2",
            remote_arbiter_judge_base_url="https://arbiter.example.invalid/v1",
            remote_arbiter_judge_api_key="key-3",
            judge_arbitration_min_delta=1,
            remote_judge_timeout_seconds=240,
            remote_judge_temperature=0.0,
            remote_judge_max_tokens=4000,
            remote_judge_top_p=1.0,
            remote_judge_openai_compatible=True,
            judge_save_raw_response=False,
        )
    )

    settings = resolved.runtime_config.settings
    assert settings.remote_judge_base_url == "https://judge1.example.invalid/v1"
    assert settings.remote_judge_api_key == "key-1"
    assert settings.remote_judge_endpoints["SECONDARY_JUDGE"].base_url == "https://judge2.example.invalid/v1"
    assert settings.remote_judge_endpoints["ARBITER"].api_key == "key-3"
    assert resolved.runtime_config.arbitration_min_delta == 1
    assert settings.remote_judge_timeout_seconds == 240
    assert settings.remote_judge_max_tokens == 4000
    assert settings.judge_save_raw_response is False
    assert "key-" not in resolved.command_preview


def test_resolve_copies_primary_env_endpoint_to_secondary() -> None:
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV))

    resolved = service.resolve(
        RunJudgeRequest(
            panel_mode="2plus1",
            endpoint_source_secondary="judge",
        )
    )

    secondary = resolved.runtime_config.settings.remote_judge_endpoints["SECONDARY_JUDGE"]
    assert secondary.base_url == "https://example.invalid/v1"
    assert secondary.api_key == "test-key"
    assert "test-key" not in resolved.command_preview


def test_resolve_env_secondary_endpoint_falls_back_to_primary_when_env_missing() -> None:
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV))

    resolved = service.resolve(
        RunJudgeRequest(
            panel_mode="2plus1",
            endpoint_source_secondary="env",
        )
    )

    secondary = resolved.runtime_config.settings.remote_judge_endpoints["SECONDARY_JUDGE"]
    assert secondary.base_url == "https://example.invalid/v1"
    assert secondary.api_key == "test-key"


def test_resolve_env_arbiter_endpoint_falls_back_to_primary_when_env_empty() -> None:
    env = dict(BASE_ENV)
    env["REMOTE_ARBITER_JUDGE_BASE_URL"] = ""
    env["REMOTE_ARBITER_JUDGE_API_KEY"] = ""
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env=env))

    resolved = service.resolve(
        RunJudgeRequest(
            panel_mode="2plus1",
            endpoint_source_arbiter="env",
        )
    )

    arbiter = resolved.runtime_config.settings.remote_judge_endpoints["ARBITER"]
    assert arbiter.base_url == "https://example.invalid/v1"
    assert arbiter.api_key == "test-key"


def test_resolve_copies_secondary_endpoint_to_arbiter() -> None:
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV))

    resolved = service.resolve(
        RunJudgeRequest(
            panel_mode="2plus1",
            remote_secondary_judge_base_url="https://judge2.example.invalid/v1",
            remote_secondary_judge_api_key="key-2",
            endpoint_source_arbiter="secondary",
        )
    )

    settings = resolved.runtime_config.settings
    assert settings.remote_judge_endpoints["ARBITER"].base_url == "https://judge2.example.invalid/v1"
    assert settings.remote_judge_endpoints["ARBITER"].api_key == "key-2"
    assert "key-2" not in resolved.command_preview


def test_web_false_always_run_arbiter_overrides_env_true() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_ALWAYS_RUN_ARBITER"] = "true"
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env=env))

    resolved = service.resolve(
        RunJudgeRequest(
            panel_mode="2plus1",
            always_run_arbiter=False,
        )
    )

    assert resolved.runtime_config.always_run_arbiter is False


def test_endpoint_override_requires_url_and_key_together() -> None:
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV))

    try:
        service.resolve(
            RunJudgeRequest(
                panel_mode="single",
                remote_secondary_judge_base_url="https://judge2.example.invalid/v1",
            )
        )
    except ValueError as error:
        assert "Both URL and token/key are required" in str(error)
    else:
        raise AssertionError("incomplete endpoint override should fail")


def test_real_run_emits_initial_and_final_eligibility_counts(tmp_path) -> None:
    repository = EligibilityRepository()
    eligibility_events: list[EligibilitySummary] = []
    service = RunJudgeService(
        settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV),
        connect_func=lambda database_url: FakeConnection(),
        repository_factory=lambda connection: repository,
        client_factory=FakeClient,
    )

    result = service.run(
        RunJudgeRequest(
            panel_mode="single",
            batch_size=1,
            audit_log=str(tmp_path / "run.log"),
            no_audit_animation=True,
        ),
        eligibility_callback=eligibility_events.append,
    )

    assert result.summary is not None
    assert result.summary.executed_evaluations == 1
    assert eligibility_events[0] == EligibilitySummary(
        missing=1,
        failed=0,
        successful=8,
        batch_size=1,
        will_process=1,
    )
    assert eligibility_events[-1] == EligibilitySummary(
        missing=0,
        failed=0,
        successful=9,
        batch_size=1,
        will_process=0,
    )
    assert result.eligibility == eligibility_events[-1]


def test_resolve_rejects_av3_source_for_non_j1_dataset() -> None:
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV))

    try:
        service.resolve(
            RunJudgeRequest(
                judge_input_source="av3_j1_com_rag",
                dataset="J2",
            )
        )
    except ValueError as error:
        assert "requires dataset J1/OAB_Bench" in str(error)
    else:
        raise AssertionError("expected invalid AV3 source/dataset combination to fail")


def test_resolve_rejects_av3_j2_source_for_non_j2_dataset() -> None:
    service = RunJudgeService(settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV))

    try:
        service.resolve(
            RunJudgeRequest(
                judge_input_source="av3_j2_com_rag",
                dataset="J1",
            )
        )
    except ValueError as error:
        assert "requires dataset J2/OAB_Exames" in str(error)
    else:
        raise AssertionError("expected invalid AV3 J2 source/dataset combination to fail")


def test_real_run_passes_av3_input_source_to_repository(tmp_path) -> None:
    repository = EligibilityRepository()
    service = RunJudgeService(
        settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV),
        connect_func=lambda database_url: FakeConnection(),
        repository_factory=lambda connection: repository,
        client_factory=FakeClient,
    )

    result = service.run(
        RunJudgeRequest(
            judge_input_source="av3_j1_com_rag",
            dataset="J1",
            panel_mode="single",
            batch_size=1,
            audit_log=str(tmp_path / "av3-run.log"),
            no_audit_animation=True,
        )
    )

    assert result.summary is not None
    assert repository.select_calls == [("av3_j1_com_rag", "J1", 1)]
    assert repository.eligibility_sources == [
        ("av3_j1_com_rag", "J1", 1),
        ("av3_j1_com_rag", "J1", 1),
        ("av3_j1_com_rag", "J1", 1),
    ]


def test_real_run_passes_av3_j2_input_source_to_repository(tmp_path) -> None:
    repository = EligibilityRepository()
    service = RunJudgeService(
        settings_loader=lambda: load_settings(dotenv_path=None, env=BASE_ENV),
        connect_func=lambda database_url: FakeConnection(),
        repository_factory=lambda connection: repository,
        client_factory=FakeClient,
    )

    result = service.run(
        RunJudgeRequest(
            judge_input_source="av3_j2_com_rag",
            dataset="J2",
            panel_mode="single",
            batch_size=1,
            audit_log=str(tmp_path / "av3-j2-run.log"),
            no_audit_animation=True,
        )
    )

    assert result.summary is not None
    assert repository.select_calls == [("av3_j2_com_rag", "J2", 1)]
    assert repository.eligibility_sources == [
        ("av3_j2_com_rag", "J2", 1),
        ("av3_j2_com_rag", "J2", 1),
        ("av3_j2_com_rag", "J2", 1),
    ]
