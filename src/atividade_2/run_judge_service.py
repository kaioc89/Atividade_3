"""Shared application service for CLI and local Web judge runs."""

from __future__ import annotations

import re
import shlex
import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .audit import AuditLogger
from .config import ConfigurationError, load_settings, resolve_runtime_config
from .contracts import (
    BatchProgress,
    EvaluationProgress,
    EligibilitySummary,
    JUDGE_INPUT_SOURCE_AV2,
    JUDGE_INPUT_SOURCE_AV3_J1_COM_RAG,
    JUDGE_INPUT_SOURCE_AV3_J2_COM_RAG,
    JudgeSettings,
    ModelSpec,
    PipelineSummary,
    RemoteJudgeEndpoint,
    RuntimeJudgeConfig,
    SUPPORTED_JUDGE_INPUT_SOURCES,
    StoredJudgeRole,
    get_judge_input_source_descriptor,
)
from .db import connect
from .judge_clients.remote_http import RemoteHttpJudgeClient
from .model_aliases import format_model_mapping
from .pipeline import JudgePipeline
from .repositories import JudgeRepository


@dataclass(frozen=True)
class RunJudgeRequest:
    """User-provided run options before env/default resolution."""

    judge_input_source: str = JUDGE_INPUT_SOURCE_AV2
    judge_provider: str | None = None
    panel_mode: str | None = None
    judge_model: str | None = None
    secondary_judge_model: str | None = None
    arbiter_judge_model: str | None = None
    always_run_arbiter: bool | None = None
    judge_execution_strategy: str | None = None
    dataset: str = "J2"
    batch_size: int | None = None
    remote_judge_base_url: str | None = None
    remote_judge_api_key: str | None = None
    remote_secondary_judge_base_url: str | None = None
    remote_secondary_judge_api_key: str | None = None
    remote_arbiter_judge_base_url: str | None = None
    remote_arbiter_judge_api_key: str | None = None
    endpoint_source_judge: str | None = None
    endpoint_source_secondary: str | None = None
    endpoint_source_arbiter: str | None = None
    judge_arbitration_min_delta: int | None = None
    remote_judge_timeout_seconds: int | None = None
    remote_judge_temperature: float | None = None
    remote_judge_max_tokens: int | None = None
    remote_judge_top_p: float | None = None
    remote_judge_openai_compatible: bool | None = None
    judge_save_raw_response: bool | None = None
    dry_run: bool = False
    preflight_report: bool = False
    audit_log: str | None = None
    no_audit_animation: bool = False


@dataclass(frozen=True)
class ResolvedRun:
    """Effective run config after settings and overrides are resolved."""

    runtime_config: RuntimeJudgeConfig
    batch_size: int
    audit_path: Path
    execution_summary: str
    command_preview: str
    preflight_report: str | None = None


@dataclass(frozen=True)
class RunJudgeResult:
    """Structured result for CLI and Web adapters."""

    dry_run: bool
    audit_log: str
    execution_summary: str
    command_preview: str
    batch_size: int
    eligibility: EligibilitySummary | None = None
    summary: PipelineSummary | None = None
    preflight_report: str | None = None


class RunJudgeService:
    """Application boundary for running the judge pipeline without subprocesses."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], JudgeSettings] = load_settings,
        connect_func: Callable[[str], Any] = connect,
        repository_factory: Callable[[Any], JudgeRepository] = JudgeRepository,
        client_factory: Callable[[JudgeSettings], RemoteHttpJudgeClient] = RemoteHttpJudgeClient,
    ) -> None:
        self._settings_loader = settings_loader
        self._connect = connect_func
        self._repository_factory = repository_factory
        self._client_factory = client_factory

    def describe_config(self) -> dict[str, Any]:
        """Return secret-safe defaults and effective configuration for the Web UI."""
        try:
            settings = self._settings_loader()
        except ConfigurationError as error:
            return _config_error_description(str(error))
        base = {
            "defaults": {
                "judge_input_source": JUDGE_INPUT_SOURCE_AV2,
                "panel_mode": settings.judge_panel_mode,
                "dataset": "J2",
                "batch_size": settings.judge_batch_size,
                "judge_execution_strategy": settings.judge_execution_strategy,
                "judge_model": settings.remote_judge_default_model,
                "secondary_judge_model": settings.remote_secondary_judge_model,
                "arbiter_judge_model": settings.remote_arbiter_judge_model,
                "always_run_arbiter": settings.judge_always_run_arbiter,
                "judge_arbitration_min_delta": settings.judge_arbitration_min_delta,
                "remote_judge_timeout_seconds": settings.remote_judge_timeout_seconds,
                "remote_judge_temperature": settings.remote_judge_temperature,
                "remote_judge_max_tokens": settings.remote_judge_max_tokens,
                "remote_judge_top_p": settings.remote_judge_top_p,
                "remote_judge_openai_compatible": settings.remote_judge_openai_compatible,
                "judge_save_raw_response": settings.judge_save_raw_response,
            },
            "supported": {
                "panel_modes": ["single", "primary_only", "2plus1"],
                "datasets": ["J1", "J2"],
                "judge_input_sources": list(SUPPORTED_JUDGE_INPUT_SOURCES),
                "judge_execution_strategies": ["sequential", "parallel", "adaptive"],
            },
            "endpoints": _endpoint_overview(settings),
            "presets": [
                {"name": "Smoke J2", "panel_mode": "single", "dataset": "J2", "batch_size": 1},
                {"name": "Smoke J1", "panel_mode": "single", "dataset": "J1", "batch_size": 1},
                {
                    "name": "AV3 J1 Com_RAG",
                    "panel_mode": "single",
                    "dataset": "J1",
                    "judge_input_source": JUDGE_INPUT_SOURCE_AV3_J1_COM_RAG,
                    "batch_size": 1,
                },
                {
                    "name": "AV3 J2 Com_RAG",
                    "panel_mode": "single",
                    "dataset": "J2",
                    "judge_input_source": JUDGE_INPUT_SOURCE_AV3_J2_COM_RAG,
                    "batch_size": 1,
                },
                {"name": "Comparacao primaria", "panel_mode": "primary_only"},
                {"name": "AV2 principal", "panel_mode": "2plus1"},
                {"name": "Auditoria completa", "panel_mode": "2plus1", "always_run_arbiter": True},
            ],
        }
        try:
            resolved = self.resolve(RunJudgeRequest())
        except ConfigurationError as error:
            base["configuration_error"] = str(error)
            return base
        base["execution_summary"] = resolved.execution_summary
        base["command_preview"] = resolved.command_preview
        return base

    def resolve(self, request: RunJudgeRequest) -> ResolvedRun:
        """Resolve settings and CLI/Web overrides without touching DB or remote HTTP."""
        _validate_judge_input_source(request)
        settings = _apply_request_overrides(self._settings_loader(), request)
        runtime_config = resolve_runtime_config(
            settings,
            judge_provider=request.judge_provider,
            panel_mode=request.panel_mode,
            judge_model=request.judge_model,
            secondary_judge_model=request.secondary_judge_model,
            arbiter_judge_model=request.arbiter_judge_model,
            always_run_arbiter=request.always_run_arbiter,
            execution_strategy=request.judge_execution_strategy,
        )
        batch_size = request.batch_size or settings.judge_batch_size
        execution_summary = format_execution_summary(runtime_config)
        return ResolvedRun(
            runtime_config=runtime_config,
            batch_size=batch_size,
            audit_path=_resolve_audit_path(request.audit_log),
            execution_summary=execution_summary,
            command_preview=build_command_preview(request, runtime_config, batch_size),
        )

    def run(
        self,
        request: RunJudgeRequest,
        *,
        on_resolved: Callable[[ResolvedRun], None] | None = None,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        eligibility_callback: Callable[[EligibilitySummary], None] | None = None,
        evaluation_callback: Callable[[EvaluationProgress], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> RunJudgeResult:
        """Run or dry-run the judge pipeline."""
        _validate_judge_input_source(request)
        audit_path = _resolve_audit_path(request.audit_log)
        animate = False if request.no_audit_animation else None
        with AuditLogger(file_path=audit_path, animate=animate) as audit:
            with audit.step("Loading configuration"):
                settings = _apply_request_overrides(self._settings_loader(), request)
            with audit.step(
                "Resolving judge mode and models",
                detail=(
                    f"panel_mode_cli={_present(request.panel_mode)} judge_model_cli={_present(request.judge_model)} "
                    f"secondary_judge_cli={_present(request.secondary_judge_model)} "
                    f"arbiter_cli={_present(request.arbiter_judge_model)} "
                    f"execution_strategy_cli={_present(request.judge_execution_strategy)}"
                ),
            ):
                runtime_config = resolve_runtime_config(
                    settings,
                    judge_provider=request.judge_provider,
                    panel_mode=request.panel_mode,
                    judge_model=request.judge_model,
                    secondary_judge_model=request.secondary_judge_model,
                    arbiter_judge_model=request.arbiter_judge_model,
                    always_run_arbiter=request.always_run_arbiter,
                    execution_strategy=request.judge_execution_strategy,
                )
            resolved = ResolvedRun(
                runtime_config=runtime_config,
                batch_size=request.batch_size or settings.judge_batch_size,
                audit_path=audit_path,
                execution_summary=format_execution_summary(runtime_config),
                command_preview=build_command_preview(request, runtime_config, request.batch_size or settings.judge_batch_size),
                preflight_report=build_preflight_report(runtime_config, request.batch_size or settings.judge_batch_size)
                if request.preflight_report
                else None,
            )
            if on_resolved is not None:
                on_resolved(resolved)
            audit.file_event("execution_summary", resolved.execution_summary.replace("\n", " | "))
            audit.file_event("command_preview", resolved.command_preview)
            if request.preflight_report:
                assert resolved.preflight_report is not None
                audit.terminal_event(resolved.preflight_report)
                audit.file_event("preflight_report", resolved.preflight_report.replace("\n", " | "))
                audit.file_event("preflight_finished", "no database rows selected and no remote judge calls made")
                return _result(request, resolved, None)
            if request.dry_run:
                audit.terminal_event("Dry run: no database rows selected and no remote judge calls made.")
                audit.file_event("dry_run_finished", "no database rows selected and no remote judge calls made")
                return _result(request, resolved, None)

            with audit.step("Connecting to local PostgreSQL", detail="DATABASE_URL=<redacted>"):
                connection = self._connect(settings.database_url)
            try:
                repository = self._repository_factory(connection)
                with audit.step("Ensuring judge metadata schema"):
                    repository.ensure_schema()
                required_evaluations = _required_evaluations(runtime_config)
                with audit.step(
                    f"Counting eligible answers for {request.dataset}",
                    detail=(
                        f"dataset={request.dataset} source={request.judge_input_source} "
                        f"batch_size={resolved.batch_size}"
                    ),
                ):
                    eligibility = repository.summarize_eligibility(
                        dataset=request.dataset,
                        batch_size=resolved.batch_size,
                        required_evaluations=required_evaluations,
                        judge_input_source=request.judge_input_source,
                    )
                audit.terminal_event(
                    "Respostas elegiveis: "
                    f"missing={eligibility.missing} failed={eligibility.failed} "
                    f"success={eligibility.successful} batch={eligibility.will_process}"
                )
                if eligibility_callback is not None:
                    _safe_emit_eligibility(audit, eligibility_callback, eligibility)
                audit.file_event(
                    "eligibility_summary",
                    (
                        f"missing={eligibility.missing} failed={eligibility.failed} "
                        f"successful={eligibility.successful} will_process={eligibility.will_process}"
                    ),
                )
                with audit.step(
                    f"Selecting pending candidate answers for {request.dataset}",
                    detail=(
                        f"dataset={request.dataset} source={request.judge_input_source} "
                        f"batch_size={resolved.batch_size}"
                    ),
                ):
                    answers = repository.select_pending_candidate_answers(
                        dataset=request.dataset,
                        batch_size=resolved.batch_size,
                        required_evaluations=required_evaluations,
                        judge_input_source=request.judge_input_source,
                    )
                audit.file_event("answers_selected", f"count={len(answers)}")
                client = self._client_factory(settings)
                reported_eligibility = eligibility

                def report_progress(progress: BatchProgress) -> None:
                    nonlocal reported_eligibility
                    if progress_callback is not None:
                        try:
                            progress_callback(progress)
                        except Exception as error:
                            audit.file_event("batch_progress_callback_failed", f"error={error}")
                    refreshed = repository.summarize_eligibility(
                        dataset=request.dataset,
                        batch_size=resolved.batch_size,
                        required_evaluations=required_evaluations,
                        judge_input_source=request.judge_input_source,
                    )
                    reported_eligibility = refreshed
                    if eligibility_callback is not None:
                        _safe_emit_eligibility(audit, eligibility_callback, refreshed)
                    audit.file_event(
                        "eligibility_progress",
                        (
                            f"missing={refreshed.missing} failed={refreshed.failed} "
                            f"successful={refreshed.successful} will_process={refreshed.will_process}"
                        ),
                    )

                with audit.step(
                    "Running judge pipeline",
                    detail=f"answers={len(answers)} mode={runtime_config.panel_mode}",
                ):
                    stop_requested = should_stop or (lambda: False)
                    summary = JudgePipeline(
                        repository,
                        client,
                        audit=audit,
                        progress_callback=report_progress,
                        evaluation_callback=evaluation_callback,
                        should_stop=stop_requested,
                    ).run(answers, runtime_config)
                eligibility = repository.summarize_eligibility(
                    dataset=request.dataset,
                    batch_size=resolved.batch_size,
                    required_evaluations=required_evaluations,
                    judge_input_source=request.judge_input_source,
                )
                if eligibility != reported_eligibility and eligibility_callback is not None:
                    _safe_emit_eligibility(audit, eligibility_callback, eligibility)
                audit.file_event(
                    "eligibility_final",
                    (
                        f"missing={eligibility.missing} failed={eligibility.failed} "
                        f"successful={eligibility.successful} will_process={eligibility.will_process}"
                    ),
                )
            finally:
                with audit.step("Closing PostgreSQL connection"):
                    connection.close()

            audit.file_event(
                "execution_result",
                (
                    f"selected={summary.selected_answers} executed={summary.executed_evaluations} "
                    f"skipped={summary.skipped_evaluations} arbiters={summary.arbiter_evaluations}"
                ),
            )
            return _result(request, resolved, summary, eligibility)


def format_execution_summary(config: RuntimeJudgeConfig) -> str:
    """Build a secret-safe execution summary."""
    lines = [
        f"Judge provider: {config.provider}",
        f"Judge mode: {config.panel_mode}",
        f"Judge execution strategy: {config.execution_strategy}",
    ]
    if config.panel_mode == "single":
        assert config.single_judge is not None
        lines.extend(
            [
                "Judge model:",
                _format_model_with_endpoint(config, config.single_judge, "SINGLE"),
                f"Model source: {config.model_source}",
            ]
        )
        return "\n".join(lines)

    lines.append("Primary judges:")
    lines.extend(
        _format_model_with_endpoint(config, model, endpoint_key)
        for model, endpoint_key in zip(config.primary_panel, ("JUDGE", "SECONDARY_JUDGE"), strict=True)
    )
    if config.panel_mode == "primary_only":
        lines.extend(
            [
                "Arbiter: disabled for primary_only mode",
                f"Model source: {config.model_source}",
            ]
        )
        return "\n".join(lines)

    assert config.arbiter is not None
    lines.extend(
        [
            "Arbiter:",
            _format_model_with_endpoint(config, config.arbiter, "ARBITER"),
            f"Arbitration min delta: {config.arbitration_min_delta}",
            f"Always run arbiter: {str(config.always_run_arbiter).lower()}",
            f"Model source: {config.model_source}",
        ]
    )
    return "\n".join(lines)


def build_preflight_report(config: RuntimeJudgeConfig, batch_size: int) -> str:
    """Build a secret-safe adaptive execution plan without DB or judge calls."""
    lines = [
        "Preflight report:",
        f"Mode: {config.panel_mode}",
        f"Execution strategy: {config.execution_strategy}",
        f"Batch size: {batch_size}",
    ]
    if config.execution_strategy != "adaptive":
        lines.append("Adaptive scheduler: disabled for this execution strategy.")
        return "\n".join(lines)

    lines.extend(
        [
            f"Adaptive initial concurrency: {config.settings.judge_adaptive_initial_concurrency}",
            f"Adaptive max concurrency: {config.settings.judge_adaptive_max_concurrency}",
            "Priority order: judge_1 -> judge_2 -> arbiter",
            "Groups:",
        ]
    )
    groups = _preflight_groups(config)
    for group in groups.values():
        lines.append(
            "- "
            f"role={group['roles']} provider={config.provider} endpoint={_endpoint_host(group['base_url'])} "
            f"api_key={group['api_key_fingerprint']} model={group['model_id']} "
            f"initial={config.settings.judge_adaptive_initial_concurrency} "
            f"max={config.settings.judge_adaptive_max_concurrency}"
        )

    snapshots = _fetch_featherless_snapshots(groups.values())
    if snapshots:
        lines.append("Provider concurrency snapshots:")
        lines.extend(f"- {snapshot}" for snapshot in snapshots)
    else:
        lines.append("Provider concurrency snapshots: unavailable; fallback is sequential-safe execution.")
    return "\n".join(lines)


def _preflight_groups(config: RuntimeJudgeConfig) -> dict[tuple[str, str, str], dict[str, str]]:
    groups: dict[tuple[str, str, str], dict[str, str]] = {}
    for role_label, endpoint_key, model in _preflight_model_slots(config):
        endpoint = _resolve_endpoint(config, model, endpoint_key)
        key = (endpoint.base_url, _fingerprint(endpoint.api_key), model.provider_model)
        group = groups.setdefault(
            key,
            {
                "roles": "",
                "base_url": endpoint.base_url,
                "api_key": endpoint.api_key,
                "api_key_fingerprint": _fingerprint(endpoint.api_key),
                "model_id": model.provider_model,
            },
        )
        group["roles"] = ",".join(filter(None, [group["roles"], role_label]))
    return groups


def _preflight_model_slots(config: RuntimeJudgeConfig) -> list[tuple[str, str, ModelSpec]]:
    if config.panel_mode == "single":
        assert config.single_judge is not None
        return [("judge_1", "JUDGE", config.single_judge)]
    slots = [
        ("judge_1", "JUDGE", config.primary_panel[0]),
        ("judge_2", "SECONDARY_JUDGE", config.primary_panel[1]),
    ]
    if config.panel_mode == "2plus1":
        assert config.arbiter is not None
        slots.append(("arbiter", "ARBITER", config.arbiter))
    return slots


def _resolve_endpoint(config: RuntimeJudgeConfig, model: ModelSpec, endpoint_key: str) -> RemoteJudgeEndpoint:
    normalized_endpoint_key = _endpoint_key(endpoint_key)
    if normalized_endpoint_key == "JUDGE":
        return RemoteJudgeEndpoint(
            base_url=config.settings.remote_judge_base_url or "",
            api_key=config.settings.remote_judge_api_key or "",
        )
    endpoint = config.settings.remote_judge_endpoints.get(normalized_endpoint_key)
    if endpoint is not None:
        return endpoint
    for candidate in (model.requested, model.provider_model):
        for candidate_key in _endpoint_keys(candidate):
            endpoint = config.settings.remote_judge_endpoints.get(candidate_key)
            if endpoint is not None:
                return endpoint
    return RemoteJudgeEndpoint(
        base_url=config.settings.remote_judge_base_url or "",
        api_key=config.settings.remote_judge_api_key or "",
    )


def _fetch_featherless_snapshots(groups: Any) -> list[str]:
    snapshots: list[str] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        base_url = group["base_url"]
        api_key = group["api_key"]
        host = _endpoint_host(base_url)
        if "featherless.ai" not in host or not api_key:
            continue
        snapshot_key = (base_url, group["api_key_fingerprint"])
        if snapshot_key in seen:
            continue
        seen.add(snapshot_key)
        snapshot = _fetch_featherless_snapshot(base_url, api_key)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


def _fetch_featherless_snapshot(base_url: str, api_key: str) -> str | None:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    url = f"{parsed.scheme}://{parsed.netloc}/account/concurrency"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "atividade-2-judge/0.1",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read(100_000).decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None
    if not isinstance(payload, dict):
        return None
    return (
        f"endpoint={_endpoint_host(base_url)} limit={payload.get('limit')} "
        f"used_cost={payload.get('used_cost')} request_count={payload.get('request_count')}"
    )


def build_command_preview(request: RunJudgeRequest, config: RuntimeJudgeConfig, batch_size: int) -> str:
    """Build the equivalent CLI command without secrets."""
    args = [
        ".venv/bin/python",
        "-m",
        "atividade_2.cli",
        "run-judge",
        "--panel-mode",
        config.panel_mode,
        "--dataset",
        request.dataset,
        "--batch-size",
        str(batch_size),
        "--judge-execution-strategy",
        config.execution_strategy,
    ]
    if request.judge_input_source != JUDGE_INPUT_SOURCE_AV2:
        args.extend(["--judge-input-source", request.judge_input_source])
    if config.panel_mode == "single":
        assert config.single_judge is not None
        args.extend(["--judge-model", config.single_judge.requested])
    else:
        if config.primary_panel:
            args.extend(["--judge-model", config.primary_panel[0].requested])
        if len(config.primary_panel) > 1:
            args.extend(["--secondary-judge-model", config.primary_panel[1].requested])
        if config.arbiter is not None:
            args.extend(["--arbiter-judge-model", config.arbiter.requested])
    if config.always_run_arbiter:
        args.append("--always-run-arbiter")
    if request.dry_run:
        args.append("--dry-run")
    if request.preflight_report:
        args.append("--preflight-report")
    return shlex.join(args)


def _result(
    request: RunJudgeRequest,
    resolved: ResolvedRun,
    summary: PipelineSummary | None,
    eligibility: EligibilitySummary | None = None,
) -> RunJudgeResult:
    return RunJudgeResult(
        dry_run=request.dry_run,
        audit_log=str(resolved.audit_path),
        execution_summary=resolved.execution_summary,
        command_preview=resolved.command_preview,
        batch_size=resolved.batch_size,
        eligibility=eligibility,
        summary=summary,
        preflight_report=resolved.preflight_report,
    )


def _safe_emit_eligibility(
    audit: AuditLogger,
    eligibility_callback: Callable[[EligibilitySummary], None],
    eligibility: EligibilitySummary,
) -> None:
    try:
        eligibility_callback(eligibility)
    except Exception as error:
        audit.file_event("eligibility_callback_failed", f"error={error}")


def _validate_judge_input_source(request: RunJudgeRequest) -> None:
    descriptor = get_judge_input_source_descriptor(request.judge_input_source)
    if not descriptor.is_av3:
        return
    assert descriptor.dataset_code is not None
    assert descriptor.dataset_name is not None
    if request.dataset.upper() not in {descriptor.dataset_code, descriptor.dataset_name.upper()}:
        raise ValueError(
            f"judge_input_source={request.judge_input_source} requires dataset "
            f"{descriptor.dataset_code}/{descriptor.dataset_name}."
        )


def _required_evaluations(config: RuntimeJudgeConfig) -> tuple[tuple[ModelSpec, StoredJudgeRole, str], ...]:
    if config.panel_mode == "single":
        assert config.single_judge is not None
        return ((config.single_judge, "principal", config.panel_mode),)
    return tuple(
        (model, role, config.panel_mode)
        for model, role in zip(config.primary_panel, ("principal", "controle"), strict=False)
    )


def _apply_request_overrides(settings: JudgeSettings, request: RunJudgeRequest) -> JudgeSettings:
    """Apply Web-only per-run settings while preserving .env fallbacks."""
    primary_base_url = settings.remote_judge_base_url
    primary_api_key = settings.remote_judge_api_key
    if request.remote_judge_base_url or request.remote_judge_api_key:
        primary_endpoint = _complete_endpoint_override(
            base_url=request.remote_judge_base_url,
            api_key=request.remote_judge_api_key,
            label="primary judge",
        )
        primary_base_url = primary_endpoint.base_url
        primary_api_key = primary_endpoint.api_key

    endpoint_overrides = dict(settings.remote_judge_endpoints)
    if request.remote_secondary_judge_base_url or request.remote_secondary_judge_api_key:
        endpoint_overrides["SECONDARY_JUDGE"] = _complete_endpoint_override(
            base_url=request.remote_secondary_judge_base_url,
            api_key=request.remote_secondary_judge_api_key,
            label="secondary judge",
        )
    elif request.endpoint_source_secondary == "env" and "SECONDARY_JUDGE" not in endpoint_overrides:
        endpoint_overrides["SECONDARY_JUDGE"] = _complete_endpoint_override(
            base_url=primary_base_url,
            api_key=primary_api_key,
            label="secondary judge",
        )
    elif request.endpoint_source_secondary == "judge":
        endpoint_overrides["SECONDARY_JUDGE"] = _complete_endpoint_override(
            base_url=primary_base_url,
            api_key=primary_api_key,
            label="secondary judge",
        )
    if request.remote_arbiter_judge_base_url or request.remote_arbiter_judge_api_key:
        endpoint_overrides["ARBITER"] = _complete_endpoint_override(
            base_url=request.remote_arbiter_judge_base_url,
            api_key=request.remote_arbiter_judge_api_key,
            label="arbiter judge",
        )
    elif request.endpoint_source_arbiter == "judge":
        endpoint_overrides["ARBITER"] = _complete_endpoint_override(
            base_url=primary_base_url,
            api_key=primary_api_key,
            label="arbiter judge",
        )
    elif request.endpoint_source_arbiter == "env" and "ARBITER" not in endpoint_overrides:
        endpoint_overrides["ARBITER"] = _complete_endpoint_override(
            base_url=primary_base_url,
            api_key=primary_api_key,
            label="arbiter judge",
        )
    elif request.endpoint_source_arbiter == "secondary":
        secondary_endpoint = endpoint_overrides.get("SECONDARY_JUDGE")
        endpoint_overrides["ARBITER"] = _complete_endpoint_override(
            base_url=secondary_endpoint.base_url if secondary_endpoint else primary_base_url,
            api_key=secondary_endpoint.api_key if secondary_endpoint else primary_api_key,
            label="arbiter judge",
        )

    return replace(
        settings,
        remote_judge_base_url=primary_base_url,
        remote_judge_api_key=primary_api_key,
        remote_judge_endpoints=endpoint_overrides,
        judge_arbitration_min_delta=(
            request.judge_arbitration_min_delta
            if request.judge_arbitration_min_delta is not None
            else settings.judge_arbitration_min_delta
        ),
        remote_judge_timeout_seconds=(
            request.remote_judge_timeout_seconds
            if request.remote_judge_timeout_seconds is not None
            else settings.remote_judge_timeout_seconds
        ),
        remote_judge_temperature=(
            request.remote_judge_temperature
            if request.remote_judge_temperature is not None
            else settings.remote_judge_temperature
        ),
        remote_judge_max_tokens=(
            request.remote_judge_max_tokens
            if request.remote_judge_max_tokens is not None
            else settings.remote_judge_max_tokens
        ),
        remote_judge_top_p=(
            request.remote_judge_top_p if request.remote_judge_top_p is not None else settings.remote_judge_top_p
        ),
        remote_judge_openai_compatible=(
            request.remote_judge_openai_compatible
            if request.remote_judge_openai_compatible is not None
            else settings.remote_judge_openai_compatible
        ),
        judge_save_raw_response=(
            request.judge_save_raw_response
            if request.judge_save_raw_response is not None
            else settings.judge_save_raw_response
        ),
    )


def _complete_endpoint_override(
    *,
    base_url: str | None,
    api_key: str | None,
    label: str,
) -> Any:
    from .contracts import RemoteJudgeEndpoint

    if not base_url or not api_key:
        raise ConfigurationError(f"Both URL and token/key are required for {label} endpoint overrides.")
    return RemoteJudgeEndpoint(base_url=base_url, api_key=api_key)


def _format_model_with_endpoint(config: RuntimeJudgeConfig, model: ModelSpec, endpoint_key: str) -> str:
    mapping = format_model_mapping(model)
    endpoint = _resolve_endpoint_base_url(config, model, endpoint_key)
    host = _endpoint_host(endpoint)
    return f"{mapping} | endpoint={host}"


def _resolve_endpoint_base_url(config: RuntimeJudgeConfig, model: ModelSpec, endpoint_key: str) -> str | None:
    normalized_endpoint_key = _endpoint_key(endpoint_key)
    if normalized_endpoint_key == "JUDGE":
        return config.settings.remote_judge_base_url
    endpoint = config.settings.remote_judge_endpoints.get(normalized_endpoint_key)
    if endpoint is not None:
        return endpoint.base_url
    for candidate in (model.requested, model.provider_model):
        for key in _endpoint_keys(candidate):
            endpoint = config.settings.remote_judge_endpoints.get(key)
            if endpoint is not None:
                return endpoint.base_url
    return config.settings.remote_judge_base_url


def _endpoint_overview(settings: JudgeSettings) -> dict[str, dict[str, Any]]:
    return {
        "JUDGE": {
            "host": _endpoint_host(settings.remote_judge_base_url),
            "has_api_key": bool(settings.remote_judge_api_key),
        },
        "SECONDARY_JUDGE": {
            "host": _endpoint_host(
                settings.remote_judge_endpoints.get("SECONDARY_JUDGE").base_url
                if settings.remote_judge_endpoints.get("SECONDARY_JUDGE")
                else settings.remote_judge_base_url
            ),
            "has_api_key": bool(
                settings.remote_judge_endpoints.get("SECONDARY_JUDGE")
                or settings.remote_judge_api_key
            ),
        },
        "ARBITER": {
            "host": _endpoint_host(
                settings.remote_judge_endpoints.get("ARBITER").base_url
                if settings.remote_judge_endpoints.get("ARBITER")
                else settings.remote_judge_base_url
            ),
            "has_api_key": bool(settings.remote_judge_endpoints.get("ARBITER") or settings.remote_judge_api_key),
        },
    }


def _config_error_description(message: str) -> dict[str, Any]:
    return {
        "defaults": {},
        "supported": {
            "panel_modes": ["single", "primary_only", "2plus1"],
            "datasets": ["J1", "J2"],
            "judge_execution_strategies": ["sequential", "parallel", "adaptive"],
        },
        "endpoints": {},
        "presets": [],
        "configuration_error": message,
    }


def _endpoint_keys(model: str) -> tuple[str, ...]:
    keys = [_endpoint_key(model)]
    if "/" in model:
        keys.append(_endpoint_key(model.rsplit("/", 1)[-1]))
    return tuple(dict.fromkeys(key for key in keys if key))


def _endpoint_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _endpoint_host(base_url: str | None) -> str:
    if not base_url:
        return "<missing>"
    host = urlparse(base_url).hostname
    return host or "<invalid>"


def _fingerprint(value: str | None) -> str:
    if not value:
        return "<missing>"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _resolve_audit_path(raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "audit" / f"judge_run_{timestamp}.log"


def _present(value: str | None) -> str:
    return "provided" if value else "not_provided"
