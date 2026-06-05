"""Configuration loading and runtime judge-mode resolution."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

from .contracts import (
    JudgeExecutionStrategy,
    JudgeSettings,
    ModelSpec,
    PanelMode,
    RemoteJudgeEndpoint,
    RuntimeJudgeConfig,
)
from .model_aliases import resolve_judge_model

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/app_dev"
DEFAULT_FEATHERLESS_URL = "https://api.featherless.ai/v1"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
SUPPORTED_PANEL_MODES: set[str] = {"single", "primary_only", "2plus1"}
SUPPORTED_PROVIDERS: set[str] = {"remote_http"}
SUPPORTED_EXECUTION_STRATEGIES: set[str] = {"sequential", "parallel", "adaptive"}
SUPPORTED_CANDIDATE_EXECUTION_STRATEGIES: set[str] = {"sequential", "parallel", "adaptive"}
SUPPORTED_APP_ENVS: set[str] = {"dev", "test", "prod"}


class ConfigurationError(ValueError):
    """Raised when runtime configuration is missing or invalid."""


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse a small dotenv file without expanding variables or commands."""
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigurationError(f"Invalid .env line {line_number}: expected KEY=value.")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigurationError(f"Invalid .env line {line_number}: key cannot be empty.")
        values[key] = _strip_quotes(value.strip())
    return values


def load_env(dotenv_path: str | Path | None = ".env", env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Load environment values, then overlay dotenv values."""
    values: dict[str, str] = dict(os.environ if env is None else env)
    if dotenv_path is not None:
        values.update(parse_env_file(dotenv_path))
    return values


def load_settings(dotenv_path: str | Path | None = ".env", env: Mapping[str, str] | None = None) -> JudgeSettings:
    """Load and validate static settings before CLI overrides."""
    values = load_env(dotenv_path=dotenv_path, env=env)
    provider = _get_choice(values, "JUDGE_PROVIDER", "remote_http", SUPPORTED_PROVIDERS)
    panel_mode = _get_choice(values, "JUDGE_PANEL_MODE", "2plus1", SUPPORTED_PANEL_MODES)
    execution_strategy = _get_choice(
        values,
        "JUDGE_EXECUTION_STRATEGY",
        "sequential",
        SUPPORTED_EXECUTION_STRATEGIES,
    )

    adaptive_initial_concurrency = _parse_int(
        values,
        "JUDGE_ADAPTIVE_INITIAL_CONCURRENCY",
        1,
        minimum=1,
    )
    adaptive_max_concurrency = _parse_int(
        values,
        "JUDGE_ADAPTIVE_MAX_CONCURRENCY",
        4,
        minimum=1,
    )
    if adaptive_initial_concurrency > adaptive_max_concurrency:
        raise ConfigurationError(
            "JUDGE_ADAPTIVE_INITIAL_CONCURRENCY must be <= JUDGE_ADAPTIVE_MAX_CONCURRENCY."
        )

    candidate_adaptive_initial_concurrency = _parse_int(
        values,
        "CANDIDATE_ADAPTIVE_INITIAL_CONCURRENCY",
        1,
        minimum=1,
    )
    candidate_adaptive_max_concurrency = _parse_int(
        values,
        "CANDIDATE_ADAPTIVE_MAX_CONCURRENCY",
        2,
        minimum=1,
    )
    if candidate_adaptive_initial_concurrency > candidate_adaptive_max_concurrency:
        raise ConfigurationError(
            "CANDIDATE_ADAPTIVE_INITIAL_CONCURRENCY must be <= CANDIDATE_ADAPTIVE_MAX_CONCURRENCY."
        )
    candidate_adaptive_base_backoff_seconds = _parse_float(
        values,
        "CANDIDATE_ADAPTIVE_BASE_BACKOFF_SECONDS",
        2.0,
        minimum=0.0,
    )
    candidate_adaptive_max_backoff_seconds = _parse_float(
        values,
        "CANDIDATE_ADAPTIVE_MAX_BACKOFF_SECONDS",
        60.0,
        minimum=0.0,
    )
    if candidate_adaptive_max_backoff_seconds < candidate_adaptive_base_backoff_seconds:
        raise ConfigurationError(
            "CANDIDATE_ADAPTIVE_MAX_BACKOFF_SECONDS must be >= CANDIDATE_ADAPTIVE_BASE_BACKOFF_SECONDS."
        )

    return JudgeSettings(
        app_env=_get_choice(values, "APP_ENV", "dev", SUPPORTED_APP_ENVS),  # type: ignore[arg-type]
        database_url=values.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        backup_root_file=values.get("BACKUP_ROOT_FILE", "backup_atividade_2.sql"),
        judge_provider=provider,  # type: ignore[arg-type]
        embedding_api_key=_empty_to_none(values.get("EMBEDDING_API_KEY")),
        featherless_url=_empty_to_none(values.get("FEATHERLESS_URL", DEFAULT_FEATHERLESS_URL)),
        featherless_api_key=_empty_to_none(values.get("FEATHERLESS_API")),
        openrouter_url=_empty_to_none(values.get("OPENROUTER_URL", DEFAULT_OPENROUTER_URL)),
        openrouter_api_key=_empty_to_none(values.get("OPENROUTER_KEY")),
        remote_judge_base_url=_empty_to_none(values.get("REMOTE_JUDGE_BASE_URL")),
        remote_judge_api_key=_empty_to_none(values.get("REMOTE_JUDGE_API_KEY")),
        remote_judge_endpoints=_parse_remote_judge_endpoints(values),
        judge_panel_mode=panel_mode,  # type: ignore[arg-type]
        remote_judge_default_model=_empty_to_none(values.get("REMOTE_JUDGE_MODEL")),
        remote_secondary_judge_model=_empty_to_none(values.get("REMOTE_SECONDARY_JUDGE_MODEL")),
        remote_arbiter_judge_model=_empty_to_none(values.get("REMOTE_ARBITER_JUDGE_MODEL")),
        judge_arbitration_min_delta=_parse_int(values, "JUDGE_ARBITRATION_MIN_DELTA", 2, minimum=0),
        judge_always_run_arbiter=_parse_bool(values, "JUDGE_ALWAYS_RUN_ARBITER", False),
        remote_judge_timeout_seconds=_parse_int(values, "REMOTE_JUDGE_TIMEOUT_SECONDS", 180, minimum=1),
        remote_judge_temperature=_parse_float(values, "REMOTE_JUDGE_TEMPERATURE", 0.0, minimum=0.0),
        remote_judge_max_tokens=_parse_int(values, "REMOTE_JUDGE_MAX_TOKENS", 1200, minimum=1),
        remote_judge_top_p=_parse_float(values, "REMOTE_JUDGE_TOP_P", 1.0, minimum=0.0),
        remote_judge_openai_compatible=_parse_bool(values, "REMOTE_JUDGE_OPENAI_COMPATIBLE", True),
        remote_candidate_temperature=_parse_float(values, "REMOTE_CANDIDATE_TEMPERATURE", 0.2, minimum=0.0),
        remote_candidate_max_tokens=_parse_optional_int(values, "REMOTE_CANDIDATE_MAX_TOKENS", minimum=1),
        remote_candidate_top_p=_parse_float(values, "REMOTE_CANDIDATE_TOP_P", 0.9, minimum=0.0),
        remote_candidate_context_safety_margin_tokens=_parse_int(
            values,
            "REMOTE_CANDIDATE_CONTEXT_SAFETY_MARGIN_TOKENS",
            512,
            minimum=0,
        ),
        remote_candidate_context_window_tokens=_parse_optional_int(
            values,
            "REMOTE_CANDIDATE_CONTEXT_WINDOW_TOKENS",
            minimum=1,
        ),
        remote_candidate_retry_on_context_window=_parse_bool(
            values,
            "REMOTE_CANDIDATE_RETRY_ON_CONTEXT_WINDOW",
            False,
        ),
        candidate_execution_strategy=_get_choice(
            values,
            "CANDIDATE_EXECUTION_STRATEGY",
            "sequential",
            SUPPORTED_CANDIDATE_EXECUTION_STRATEGIES,
        ),  # type: ignore[arg-type]
        candidate_parallel_max_workers=_parse_int(
            values,
            "CANDIDATE_PARALLEL_MAX_WORKERS",
            2,
            minimum=1,
        ),
        candidate_adaptive_initial_concurrency=candidate_adaptive_initial_concurrency,
        candidate_adaptive_max_concurrency=candidate_adaptive_max_concurrency,
        candidate_adaptive_success_threshold=_parse_int(
            values,
            "CANDIDATE_ADAPTIVE_SUCCESS_THRESHOLD",
            3,
            minimum=1,
        ),
        candidate_adaptive_max_retries=_parse_int(
            values,
            "CANDIDATE_ADAPTIVE_MAX_RETRIES",
            2,
            minimum=0,
        ),
        candidate_adaptive_base_backoff_seconds=candidate_adaptive_base_backoff_seconds,
        candidate_adaptive_max_backoff_seconds=candidate_adaptive_max_backoff_seconds,
        judge_save_raw_response=_parse_bool(values, "JUDGE_SAVE_RAW_RESPONSE", True),
        judge_execution_strategy=execution_strategy,  # type: ignore[arg-type]
        judge_batch_size=_parse_int(values, "JUDGE_BATCH_SIZE", 10, minimum=1),
        judge_adaptive_initial_concurrency=adaptive_initial_concurrency,
        judge_adaptive_max_concurrency=adaptive_max_concurrency,
        judge_adaptive_success_threshold=_parse_int(
            values,
            "JUDGE_ADAPTIVE_SUCCESS_THRESHOLD",
            5,
            minimum=1,
        ),
        judge_adaptive_max_retries=_parse_int(values, "JUDGE_ADAPTIVE_MAX_RETRIES", 3, minimum=0),
        judge_adaptive_base_backoff_seconds=_parse_float(
            values,
            "JUDGE_ADAPTIVE_BASE_BACKOFF_SECONDS",
            2.0,
            minimum=0.0,
        ),
        judge_adaptive_max_backoff_seconds=_parse_float(
            values,
            "JUDGE_ADAPTIVE_MAX_BACKOFF_SECONDS",
            60.0,
            minimum=0.0,
        ),
    )


def resolve_runtime_config(
    settings: JudgeSettings,
    *,
    judge_provider: str | None = None,
    panel_mode: str | None = None,
    judge_model: str | None = None,
    secondary_judge_model: str | None = None,
    arbiter_judge_model: str | None = None,
    always_run_arbiter: bool | None = None,
    execution_strategy: str | None = None,
) -> RuntimeJudgeConfig:
    """Resolve effective judge mode/model/panel using CLI-over-env precedence."""
    provider = judge_provider or settings.judge_provider
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigurationError(f"Unsupported judge provider: {provider}. Supported values: remote_http.")

    effective_mode = _resolve_panel_mode(settings, panel_mode, judge_model)
    effective_strategy = _resolve_execution_strategy(settings, execution_strategy)
    _validate_remote_settings(settings)

    if effective_mode == "single":
        model_value = judge_model or settings.remote_judge_default_model
        if not model_value:
            raise ConfigurationError(
                "Single judge mode requires --judge-model or REMOTE_JUDGE_MODEL."
            )
        single_judge = resolve_judge_model(model_value)
        return RuntimeJudgeConfig(
            provider=provider,  # type: ignore[arg-type]
            panel_mode="single",
            single_judge=single_judge,
            primary_panel=(),
            arbiter=None,
            arbitration_min_delta=settings.judge_arbitration_min_delta,
            always_run_arbiter=False,
            execution_strategy=effective_strategy,
            settings=settings,
            model_source="CLI argument --judge-model" if judge_model else ".env REMOTE_JUDGE_MODEL",
        )

    panel_values = _resolve_primary_panel_values(
        settings,
        judge_model=judge_model,
        secondary_judge_model=secondary_judge_model,
    )
    if not panel_values:
        raise ConfigurationError(
            f"{effective_mode} mode requires REMOTE_JUDGE_MODEL and REMOTE_SECONDARY_JUDGE_MODEL."
        )
    if effective_mode == "2plus1" and len(panel_values) != 2:
        raise ConfigurationError("2plus1 mode requires exactly two primary judge models.")

    primary_panel = tuple(resolve_judge_model(value) for value in panel_values)
    arbiter = None
    if effective_mode == "2plus1":
        arbiter_value = arbiter_judge_model or settings.remote_arbiter_judge_model
        if not arbiter_value:
            raise ConfigurationError(
                "2plus1 mode requires --arbiter-judge-model or REMOTE_ARBITER_JUDGE_MODEL."
            )
        arbiter = resolve_judge_model(arbiter_value)

    return RuntimeJudgeConfig(
        provider=provider,  # type: ignore[arg-type]
        panel_mode=effective_mode,
        single_judge=None,
        primary_panel=primary_panel,
        arbiter=arbiter,
        arbitration_min_delta=settings.judge_arbitration_min_delta,
        always_run_arbiter=(
            settings.judge_always_run_arbiter
            if always_run_arbiter is None
            else always_run_arbiter
        ),
        execution_strategy=effective_strategy,
        settings=settings,
        model_source=_panel_model_source(judge_model, secondary_judge_model, arbiter_judge_model),
    )


def _resolve_panel_mode(settings: JudgeSettings, panel_mode: str | None, judge_model: str | None) -> PanelMode:
    if judge_model and panel_mode in {None, "single"}:
        return "single"
    effective_mode = panel_mode or settings.judge_panel_mode
    if effective_mode not in SUPPORTED_PANEL_MODES:
        raise ConfigurationError(
            f"Unsupported panel mode: {effective_mode}. Supported values: single, primary_only, 2plus1."
        )
    return effective_mode  # type: ignore[return-value]


def _resolve_execution_strategy(
    settings: JudgeSettings,
    execution_strategy: str | None,
) -> JudgeExecutionStrategy:
    effective_strategy = execution_strategy or settings.judge_execution_strategy
    if effective_strategy not in SUPPORTED_EXECUTION_STRATEGIES:
        raise ConfigurationError(
            "Unsupported judge execution strategy: "
            f"{effective_strategy}. Supported values: adaptive, sequential, parallel."
        )
    return effective_strategy  # type: ignore[return-value]


def _parse_optional_int(
    values: Mapping[str, str],
    key: str,
    *,
    minimum: int | None = None,
) -> int | None:
    raw_value = values.get(key)
    if raw_value is None or not raw_value.strip():
        return None
    return _parse_int(values, key, 0, minimum=minimum)


def _validate_remote_settings(settings: JudgeSettings) -> None:
    if settings.judge_provider != "remote_http":
        raise ConfigurationError(f"Unsupported judge provider: {settings.judge_provider}.")
    if not settings.remote_judge_base_url:
        raise ConfigurationError("REMOTE_JUDGE_BASE_URL is required for JUDGE_PROVIDER=remote_http.")
    if not settings.remote_judge_base_url.startswith(("http://", "https://")):
        raise ConfigurationError("REMOTE_JUDGE_BASE_URL must start with http:// or https://.")
    if not settings.remote_judge_api_key:
        raise ConfigurationError("REMOTE_JUDGE_API_KEY is required for JUDGE_PROVIDER=remote_http.")
    for key, endpoint in settings.remote_judge_endpoints.items():
        if not endpoint.base_url.startswith(("http://", "https://")):
            raise ConfigurationError(f"REMOTE_JUDGE_{key}_BASE_URL must start with http:// or https://.")


def _panel_model_source(
    judge_model: str | None,
    secondary_judge_model: str | None,
    arbiter_judge_model: str | None,
) -> str:
    sources: list[str] = []
    sources.append("CLI --judge-model" if judge_model else ".env REMOTE_JUDGE_MODEL")
    sources.append(
        "CLI --secondary-judge-model"
        if secondary_judge_model
        else ".env REMOTE_SECONDARY_JUDGE_MODEL"
    )
    sources.append("CLI --arbiter-judge-model" if arbiter_judge_model else ".env REMOTE_ARBITER_JUDGE_MODEL")
    return " / ".join(sources)


def _resolve_primary_panel_values(
    settings: JudgeSettings,
    *,
    judge_model: str | None,
    secondary_judge_model: str | None,
) -> tuple[str, ...]:
    first = judge_model or settings.remote_judge_default_model
    second = secondary_judge_model or settings.remote_secondary_judge_model
    if first and second:
        return (first, second)
    return ()


def _parse_panel(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    panel = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(set(panel)) != len(panel):
        raise ConfigurationError("Judge panel cannot contain duplicate models.")
    return panel


def _parse_remote_judge_endpoints(values: Mapping[str, str]) -> dict[str, RemoteJudgeEndpoint]:
    prefix = "REMOTE_JUDGE_"
    base_suffix = "_BASE_URL"
    key_suffix = "_API_KEY"
    endpoint_keys: set[str] = set()

    endpoints: dict[str, RemoteJudgeEndpoint] = {}
    for endpoint_key, base_env_key, api_key_env_key in (
        ("SECONDARY_JUDGE", "REMOTE_SECONDARY_JUDGE_BASE_URL", "REMOTE_SECONDARY_JUDGE_API_KEY"),
        ("ARBITER", "REMOTE_ARBITER_JUDGE_BASE_URL", "REMOTE_ARBITER_JUDGE_API_KEY"),
    ):
        base_url = _empty_to_none(values.get(base_env_key))
        api_key = _empty_to_none(values.get(api_key_env_key))
        if bool(base_url) != bool(api_key):
            raise ConfigurationError(f"{base_env_key} and {api_key_env_key} must be configured together.")
        if base_url and api_key:
            endpoints[endpoint_key] = RemoteJudgeEndpoint(base_url=base_url, api_key=api_key)

    for env_key in values:
        if not env_key.startswith(prefix):
            continue
        if env_key in {"REMOTE_JUDGE_BASE_URL", "REMOTE_JUDGE_API_KEY"}:
            continue
        if env_key.endswith(base_suffix):
            endpoint_keys.add(_remote_endpoint_key(env_key.removeprefix(prefix).removesuffix(base_suffix)))
        elif env_key.endswith(key_suffix):
            endpoint_keys.add(_remote_endpoint_key(env_key.removeprefix(prefix).removesuffix(key_suffix)))

    for endpoint_key in sorted(endpoint_keys):
        base_url = _empty_to_none(values.get(f"{prefix}{endpoint_key}{base_suffix}"))
        api_key = _empty_to_none(values.get(f"{prefix}{endpoint_key}{key_suffix}"))
        if bool(base_url) != bool(api_key):
            raise ConfigurationError(
                f"REMOTE_JUDGE_{endpoint_key}_BASE_URL and REMOTE_JUDGE_{endpoint_key}_API_KEY "
                "must be configured together."
            )
        if base_url and api_key:
            endpoints[endpoint_key] = RemoteJudgeEndpoint(base_url=base_url, api_key=api_key)
    return endpoints


def _remote_endpoint_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _parse_bool(values: Mapping[str, str], key: str, default: bool) -> bool:
    raw_value = values.get(key)
    if raw_value is None or raw_value == "":
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{key} must be a boolean value.")


def _parse_int(values: Mapping[str, str], key: str, default: int, *, minimum: int) -> int:
    raw_value = values.get(key)
    if raw_value is None or raw_value == "":
        return default
    try:
        parsed = int(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{key} must be an integer.") from error
    if parsed < minimum:
        raise ConfigurationError(f"{key} must be >= {minimum}.")
    return parsed


def _parse_float(values: Mapping[str, str], key: str, default: float, *, minimum: float) -> float:
    raw_value = values.get(key)
    if raw_value is None or raw_value == "":
        return default
    try:
        parsed = float(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{key} must be a number.") from error
    if parsed < minimum:
        raise ConfigurationError(f"{key} must be >= {minimum}.")
    return parsed


def _get_choice(values: Mapping[str, str], key: str, default: str, choices: set[str]) -> str:
    value = values.get(key, default)
    if value not in choices:
        raise ConfigurationError(f"{key} must be one of: {', '.join(sorted(choices))}.")
    return value


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
