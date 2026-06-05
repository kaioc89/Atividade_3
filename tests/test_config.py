from __future__ import annotations

import pytest

from atividade_2.config import ConfigurationError, load_settings, resolve_runtime_config


BASE_ENV = {
    "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/app_dev",
    "JUDGE_PROVIDER": "remote_http",
    "REMOTE_JUDGE_BASE_URL": "https://example.invalid/v1",
    "REMOTE_JUDGE_API_KEY": "test-key",
    "JUDGE_PANEL_MODE": "2plus1",
    "REMOTE_JUDGE_MODEL": "gpt-oss-120b",
    "REMOTE_SECONDARY_JUDGE_MODEL": "llama-3.3-70b-instruct",
    "REMOTE_ARBITER_JUDGE_MODEL": "m-prometheus-14b",
    "JUDGE_EXECUTION_STRATEGY": "sequential",
}


def test_settings_load_default_models_from_env() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)

    assert settings.app_env == "dev"
    assert settings.backup_root_file == "backup_atividade_2.sql"
    assert settings.featherless_url == "https://api.featherless.ai/v1"
    assert settings.featherless_api_key is None
    assert settings.openrouter_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_api_key is None
    assert settings.remote_judge_base_url == "https://example.invalid/v1"
    assert settings.remote_judge_api_key == "test-key"
    assert settings.remote_judge_endpoints == {}
    assert settings.remote_judge_default_model == "gpt-oss-120b"
    assert settings.remote_secondary_judge_model == "llama-3.3-70b-instruct"
    assert settings.remote_arbiter_judge_model == "m-prometheus-14b"
    assert settings.judge_execution_strategy == "sequential"
    assert settings.judge_batch_size == 10
    assert settings.judge_adaptive_initial_concurrency == 1
    assert settings.judge_adaptive_max_concurrency == 4
    assert settings.judge_adaptive_success_threshold == 5
    assert settings.judge_adaptive_max_retries == 3
    assert settings.judge_adaptive_base_backoff_seconds == 2.0
    assert settings.judge_adaptive_max_backoff_seconds == 60.0
    assert settings.remote_candidate_retry_on_context_window is False
    assert settings.candidate_execution_strategy == "sequential"
    assert settings.candidate_parallel_max_workers == 2


def test_app_env_can_be_loaded_from_env() -> None:
    env = dict(BASE_ENV)
    env["APP_ENV"] = "prod"

    settings = load_settings(dotenv_path=None, env=env)

    assert settings.app_env == "prod"


def test_backup_root_file_can_be_loaded_from_env() -> None:
    env = dict(BASE_ENV)
    env["BACKUP_ROOT_FILE"] = "/workspace/backup_atividade_2.sql"

    settings = load_settings(dotenv_path=None, env=env)

    assert settings.backup_root_file == "/workspace/backup_atividade_2.sql"


def test_invalid_app_env_fails() -> None:
    env = dict(BASE_ENV)
    env["APP_ENV"] = "staging"

    with pytest.raises(ConfigurationError, match="APP_ENV"):
        load_settings(dotenv_path=None, env=env)


def test_batch_size_can_be_loaded_from_env() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_BATCH_SIZE"] = "25"

    settings = load_settings(dotenv_path=None, env=env)

    assert settings.judge_batch_size == 25


def test_invalid_batch_size_fails() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_BATCH_SIZE"] = "0"

    with pytest.raises(ConfigurationError, match="JUDGE_BATCH_SIZE"):
        load_settings(dotenv_path=None, env=env)


def test_dotenv_values_override_process_environment(tmp_path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "REMOTE_JUDGE_BASE_URL=https://dotenv.example.invalid/v1",
                "REMOTE_JUDGE_API_KEY=dotenv-key",
                "REMOTE_JUDGE_MODEL=openai/gpt-oss-120b",
                "REMOTE_SECONDARY_JUDGE_MODEL=llama-3.3-70b-versatile",
                "REMOTE_ARBITER_JUDGE_MODEL=gemini-2.5-flash",
                "JUDGE_EXECUTION_STRATEGY=parallel",
            ]
        ),
        encoding="utf-8",
    )
    env = {
        "REMOTE_JUDGE_BASE_URL": "https://env.example.invalid/v1",
        "REMOTE_JUDGE_API_KEY": "env-key",
        "REMOTE_JUDGE_MODEL": "gpt-oss-120b",
        "REMOTE_SECONDARY_JUDGE_MODEL": "llama-3.3-70b-instruct",
        "REMOTE_ARBITER_JUDGE_MODEL": "m-prometheus-14b",
        "JUDGE_EXECUTION_STRATEGY": "sequential",
    }

    settings = load_settings(dotenv_path=dotenv_path, env=env)

    assert settings.remote_judge_base_url == "https://dotenv.example.invalid/v1"
    assert settings.remote_judge_api_key == "dotenv-key"
    assert settings.remote_judge_default_model == "openai/gpt-oss-120b"
    assert settings.remote_secondary_judge_model == "llama-3.3-70b-versatile"
    assert settings.remote_arbiter_judge_model == "gemini-2.5-flash"
    assert settings.judge_execution_strategy == "parallel"


def test_candidate_provider_env_values_can_be_loaded_from_env() -> None:
    env = dict(BASE_ENV)
    env["FEATHERLESS_URL"] = "https://featherless.example.invalid/v1"
    env["FEATHERLESS_API"] = "featherless-secret"
    env["OPENROUTER_URL"] = "https://openrouter.example.invalid/api/v1"
    env["OPENROUTER_KEY"] = "openrouter-secret"
    env["REMOTE_CANDIDATE_MAX_TOKENS"] = "1024"
    env["REMOTE_CANDIDATE_TEMPERATURE"] = "0.2"
    env["REMOTE_CANDIDATE_TOP_P"] = "0.9"
    env["REMOTE_CANDIDATE_CONTEXT_SAFETY_MARGIN_TOKENS"] = "384"
    env["REMOTE_CANDIDATE_CONTEXT_WINDOW_TOKENS"] = "8192"
    env["REMOTE_CANDIDATE_RETRY_ON_CONTEXT_WINDOW"] = "true"
    env["EMBEDDING_API_KEY"] = "embedding-secret"

    settings = load_settings(dotenv_path=None, env=env)

    assert settings.featherless_url == "https://featherless.example.invalid/v1"
    assert settings.featherless_api_key == "featherless-secret"
    assert settings.openrouter_url == "https://openrouter.example.invalid/api/v1"
    assert settings.openrouter_api_key == "openrouter-secret"
    assert settings.remote_candidate_max_tokens == 1024
    assert settings.remote_candidate_temperature == 0.2
    assert settings.remote_candidate_top_p == 0.9
    assert settings.remote_candidate_context_safety_margin_tokens == 384
    assert settings.remote_candidate_context_window_tokens == 8192
    assert settings.remote_candidate_retry_on_context_window is True
    assert settings.remote_candidate_temperature == 0.2
    assert settings.remote_candidate_top_p == 0.9
    assert settings.embedding_api_key == "embedding-secret"


def test_candidate_execution_strategy_parses_sequential() -> None:
    env = dict(BASE_ENV)
    env["CANDIDATE_EXECUTION_STRATEGY"] = "sequential"

    settings = load_settings(dotenv_path=None, env=env)

    assert settings.candidate_execution_strategy == "sequential"


def test_candidate_execution_strategy_parses_parallel() -> None:
    env = dict(BASE_ENV)
    env["CANDIDATE_EXECUTION_STRATEGY"] = "parallel"

    settings = load_settings(dotenv_path=None, env=env)

    assert settings.candidate_execution_strategy == "parallel"


def test_invalid_candidate_execution_strategy_fails() -> None:
    env = dict(BASE_ENV)
    env["CANDIDATE_EXECUTION_STRATEGY"] = "adaptive"

    with pytest.raises(ConfigurationError, match="CANDIDATE_EXECUTION_STRATEGY"):
        load_settings(dotenv_path=None, env=env)


def test_candidate_parallel_max_workers_parses_positive_integer() -> None:
    env = dict(BASE_ENV)
    env["CANDIDATE_PARALLEL_MAX_WORKERS"] = "4"

    settings = load_settings(dotenv_path=None, env=env)

    assert settings.candidate_parallel_max_workers == 4


def test_invalid_candidate_parallel_max_workers_fails() -> None:
    env = dict(BASE_ENV)
    env["CANDIDATE_PARALLEL_MAX_WORKERS"] = "0"

    with pytest.raises(ConfigurationError, match="CANDIDATE_PARALLEL_MAX_WORKERS"):
        load_settings(dotenv_path=None, env=env)


def test_judge_model_cli_override_forces_single_mode() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, judge_model="custom/provider")

    assert config.panel_mode == "single"
    assert config.single_judge is not None
    assert config.single_judge.provider_model == "custom/provider"


def test_primary_judges_can_be_overridden_by_cli() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(
        settings,
        panel_mode="primary_only",
        judge_model="model-a",
        secondary_judge_model="model-b",
    )

    assert [model.provider_model for model in config.primary_panel] == ["model-a", "model-b"]


def test_primary_panel_can_be_resolved_from_first_and_second_judge_models() -> None:
    env = dict(BASE_ENV)
    env["REMOTE_JUDGE_MODEL"] = "gpt-oss-120b"
    env["REMOTE_SECONDARY_JUDGE_MODEL"] = "llama-3.3-70b-instruct"
    settings = load_settings(dotenv_path=None, env=env)

    config = resolve_runtime_config(settings, panel_mode="primary_only")

    assert settings.remote_secondary_judge_model == "llama-3.3-70b-instruct"
    assert [model.provider_model for model in config.primary_panel] == [
        "openai/gpt-oss-120b",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]


def test_arbiter_cli_override_wins() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(
        settings,
        panel_mode="2plus1",
        arbiter_judge_model="custom-arbiter",
    )

    assert config.arbiter is not None
    assert config.arbiter.provider_model == "custom-arbiter"


def test_remote_http_requires_base_url() -> None:
    env = dict(BASE_ENV)
    env.pop("REMOTE_JUDGE_BASE_URL")
    settings = load_settings(dotenv_path=None, env=env)

    with pytest.raises(ConfigurationError, match="REMOTE_JUDGE_BASE_URL"):
        resolve_runtime_config(settings, panel_mode="single")


def test_settings_load_per_judge_endpoint_overrides() -> None:
    env = dict(BASE_ENV)
    env["REMOTE_JUDGE_GPT_OSS_120B_BASE_URL"] = "https://gpt.example.invalid/v1"
    env["REMOTE_JUDGE_GPT_OSS_120B_API_KEY"] = "gpt-key"

    settings = load_settings(dotenv_path=None, env=env)

    endpoint = settings.remote_judge_endpoints["GPT_OSS_120B"]
    assert endpoint.base_url == "https://gpt.example.invalid/v1"
    assert endpoint.api_key == "gpt-key"


def test_per_judge_endpoint_requires_url_and_key_together() -> None:
    env = dict(BASE_ENV)
    env["REMOTE_JUDGE_GPT_OSS_120B_BASE_URL"] = "https://gpt.example.invalid/v1"

    with pytest.raises(ConfigurationError, match="REMOTE_JUDGE_GPT_OSS_120B"):
        load_settings(dotenv_path=None, env=env)


def test_execution_strategy_cli_override_wins() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(
        settings,
        panel_mode="primary_only",
        execution_strategy="parallel",
    )

    assert config.execution_strategy == "parallel"


def test_adaptive_execution_strategy_is_supported() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"

    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="single")

    assert settings.judge_execution_strategy == "adaptive"
    assert config.execution_strategy == "adaptive"


def test_invalid_adaptive_initial_concurrency_fails() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_ADAPTIVE_INITIAL_CONCURRENCY"] = "0"

    with pytest.raises(ConfigurationError, match="JUDGE_ADAPTIVE_INITIAL_CONCURRENCY"):
        load_settings(dotenv_path=None, env=env)


def test_adaptive_initial_concurrency_cannot_exceed_max() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_ADAPTIVE_INITIAL_CONCURRENCY"] = "5"
    env["JUDGE_ADAPTIVE_MAX_CONCURRENCY"] = "4"

    with pytest.raises(ConfigurationError, match="INITIAL_CONCURRENCY"):
        load_settings(dotenv_path=None, env=env)


def test_invalid_execution_strategy_fails() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "batch"

    with pytest.raises(ConfigurationError, match="JUDGE_EXECUTION_STRATEGY"):
        load_settings(dotenv_path=None, env=env)
