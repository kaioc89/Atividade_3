"""Local FastAPI console for running the AV2 judge pipeline."""

from __future__ import annotations

import csv
import os
import re
import secrets
import shlex
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from .assistant import AssistantService
from .audit_log_service import AuditLogSummaryService
from .config import ConfigurationError
from .contracts import BatchProgress, EligibilitySummary, EvaluationProgress, PipelineSummary
from .dashboard import DashboardService, parse_dashboard_filters
from .database_dump import DatabaseDumpService, DatabaseResetService, resolve_backup_dir, resolve_dump_path
from .db import connect
from .judge_prompt_configs import JudgePromptConfigService
from .judge_clients.remote_http import RemoteJudgeError
from .meta_evaluations import MetaEvaluationService
from .parser import JudgeParseError
from .rag_curation import RagCurationService
from .rag_embedding_configs import RagEmbeddingConfigService
from .rag_embeddings import RagEmbeddingGenerationService
from .rag_embedding_smoke import RagEmbeddingSmokeTestService
from .rag_vector_queries import RagVectorQueryService
from .rag_vector_runs import RagVectorRunService
from .repositories import JudgeRepository
from .run_judge_service import RunJudgeRequest, RunJudgeResult, RunJudgeService
from .config import load_settings


RunStatus = Literal["queued", "running", "cancelling", "completed", "failed", "cancelled"]
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
AUDIT_TIMESTAMP_PATTERN = re.compile(r"^([^|]+)\s+\|\s+([^|]+)(?:\s+\|\s+(.*))?$")
AUDIT_KEY_VALUE_PATTERN = re.compile(r"([A-Za-z_]+)=([^ ]+)")
DEFAULT_AUDIT_DIR = Path("outputs") / "audit"
DEFAULT_BACKUP_DIR = Path("outputs") / "backup"


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


class RunPayload(BaseModel):
    judge_input_source: Literal["av2", "av3_j1_com_rag"] = "av2"
    panel_mode: Literal["single", "primary_only", "2plus1"] | None = None
    dataset: Literal["J1", "J2", "OAB_Bench", "OAB_Exames"] = "J2"
    batch_size: int | None = Field(default=None, ge=1)
    judge_execution_strategy: Literal["sequential", "parallel", "adaptive"] | None = None
    judge_model: str | None = None
    secondary_judge_model: str | None = None
    arbiter_judge_model: str | None = None
    always_run_arbiter: bool = False
    remote_judge_base_url: str | None = None
    remote_judge_api_key: str | None = None
    remote_secondary_judge_base_url: str | None = None
    remote_secondary_judge_api_key: str | None = None
    remote_arbiter_judge_base_url: str | None = None
    remote_arbiter_judge_api_key: str | None = None
    endpoint_source_judge: Literal["env", "custom"] | None = None
    endpoint_source_secondary: Literal["env", "judge", "custom"] | None = None
    endpoint_source_arbiter: Literal["env", "judge", "secondary", "custom"] | None = None
    judge_arbitration_min_delta: int | None = Field(default=None, ge=0)
    remote_judge_timeout_seconds: int | None = Field(default=None, ge=1)
    remote_judge_temperature: float | None = Field(default=None, ge=0)
    remote_judge_max_tokens: int | None = Field(default=None, ge=1)
    remote_judge_top_p: float | None = Field(default=None, ge=0)
    remote_judge_openai_compatible: bool | None = None
    judge_save_raw_response: bool | None = None

    def to_request(self, *, dry_run: bool) -> RunJudgeRequest:
        return RunJudgeRequest(
            judge_input_source=self.judge_input_source,
            panel_mode=self.panel_mode,
            judge_model=self.judge_model or None,
            secondary_judge_model=self.secondary_judge_model or None,
            arbiter_judge_model=self.arbiter_judge_model or None,
            always_run_arbiter=self.always_run_arbiter,
            judge_execution_strategy=self.judge_execution_strategy,
            dataset=self.dataset,
            batch_size=self.batch_size,
            remote_judge_base_url=self.remote_judge_base_url or None,
            remote_judge_api_key=self.remote_judge_api_key or None,
            remote_secondary_judge_base_url=self.remote_secondary_judge_base_url or None,
            remote_secondary_judge_api_key=self.remote_secondary_judge_api_key or None,
            remote_arbiter_judge_base_url=self.remote_arbiter_judge_base_url or None,
            remote_arbiter_judge_api_key=self.remote_arbiter_judge_api_key or None,
            endpoint_source_judge=self.endpoint_source_judge,
            endpoint_source_secondary=self.endpoint_source_secondary,
            endpoint_source_arbiter=self.endpoint_source_arbiter,
            judge_arbitration_min_delta=self.judge_arbitration_min_delta,
            remote_judge_timeout_seconds=self.remote_judge_timeout_seconds,
            remote_judge_temperature=self.remote_judge_temperature,
            remote_judge_max_tokens=self.remote_judge_max_tokens,
            remote_judge_top_p=self.remote_judge_top_p,
            remote_judge_openai_compatible=self.remote_judge_openai_compatible,
            judge_save_raw_response=self.judge_save_raw_response,
            dry_run=dry_run,
            no_audit_animation=True,
        )


class PromptConfigPayload(BaseModel):
    dataset: str
    prompt: str
    persona: str
    context: str
    rubric: str
    output: str
    changed_by: str


class MetaEvaluationPayload(BaseModel):
    meta_evaluation_id: int | None = None
    evaluation_id: int
    evaluator_name: str
    score: int = Field(ge=1, le=5)
    rationale: str


class RagEmbeddingConfigPayload(BaseModel):
    dataset: str
    provider: str
    model_name: str
    dimensions: int | None = Field(default=None, ge=1)
    api_base_url: str | None = None
    notes: str | None = None
    updated_by: str


class RagEmbeddingSmokeTestPayload(BaseModel):
    dataset: str
    sample_text: str | None = None


class RagEmbeddingGenerationPayload(BaseModel):
    dataset: str
    batch_size: int | None = Field(default=None, ge=1)
    question_sequence_start: int | None = Field(default=None, ge=1)
    question_sequence_end: int | None = Field(default=None, ge=1)


class RagVectorSearchPayload(BaseModel):
    dataset: str
    query_text: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class AssistantPayload(BaseModel):
    message: str = Field(min_length=1)


@dataclass
class JobState:
    run_id: str
    status: RunStatus
    request: RunJudgeRequest
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: BatchProgress = field(default_factory=lambda: _initial_progress())
    result: RunJudgeResult | None = None
    error: str | None = None
    audit_log: str | None = None
    command_preview: str | None = None
    eligibility: EligibilitySummary | None = None
    evaluation_events: list[EvaluationProgress] = field(default_factory=list)
    cancel_requested: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class RunHistoryEntry:
    run_id: str
    timestamp: str | None
    finished_at: str | None
    mode: str | None
    dataset: str | None
    batch_size: int | None
    successes: int
    failures: int
    duration_seconds: int | None
    duration: str | None
    log_path: str
    log_url: str
    summary: str | None


class JobRegistry:
    """In-memory job registry for a single local operator process."""

    def __init__(self, service: RunJudgeService) -> None:
        self.service = service
        self._jobs: dict[str, JobState] = {}
        self._active_run_id: str | None = None
        self._lock = threading.Lock()

    def create(self, request: RunJudgeRequest) -> JobState:
        resolved = self.service.resolve(request)
        with self._lock:
            if self._active_run_id is not None:
                active = self._jobs.get(self._active_run_id)
                if active is not None and active.status in {"queued", "running"}:
                    raise HTTPException(status_code=409, detail="Another judge run is already active.")
            run_id = uuid.uuid4().hex
            job = JobState(
                run_id=run_id,
                status="queued",
                request=request,
                audit_log=str(resolved.audit_path),
                command_preview=resolved.command_preview,
            )
            self._jobs[run_id] = job
            self._active_run_id = run_id
        threading.Thread(target=self._run, args=(run_id,), daemon=True).start()
        return job

    def get(self, run_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(run_id)

    def cancel(self, run_id: str) -> JobState | None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return None
            if job.status in {"queued", "running", "cancelling"}:
                job.cancel_requested = True
                job.cancel_event.set()
                job.status = "cancelling"
                return job
            return job

    def _run(self, run_id: str) -> None:
        with self._lock:
            job = self._jobs[run_id]
            if job.cancel_requested:
                job.status = "cancelled"
                job.finished_at = datetime.now()
                if self._active_run_id == run_id:
                    self._active_run_id = None
                return
            job.status = "running"
            job.started_at = datetime.now()

        def update_progress(progress: BatchProgress) -> None:
            with self._lock:
                job = self._jobs[run_id]
                job.progress = progress
                _sync_progress_with_execution_table(job)

        def update_eligibility(eligibility: EligibilitySummary) -> None:
            with self._lock:
                self._jobs[run_id].eligibility = eligibility

        def update_evaluation(evaluation: EvaluationProgress) -> None:
            if evaluation.status == "skipped":
                return
            with self._lock:
                job = self._jobs[run_id]
                _upsert_evaluation_event(job.evaluation_events, evaluation)
                _sync_progress_with_execution_table(job)

        try:
            result = self.service.run(
                job.request,
                progress_callback=update_progress,
                eligibility_callback=update_eligibility,
                evaluation_callback=update_evaluation,
                should_stop=job.cancel_event.is_set,
            )
        except (ConfigurationError, RemoteJudgeError, JudgeParseError, RuntimeError, ValueError) as error:
            with self._lock:
                job = self._jobs[run_id]
                job.status = "failed"
                job.finished_at = datetime.now()
                job.error = str(error)
        else:
            with self._lock:
                job = self._jobs[run_id]
                job.status = "cancelled" if job.cancel_requested else "completed"
                job.finished_at = datetime.now()
                job.result = result
                job.audit_log = result.audit_log
                job.command_preview = result.command_preview
                job.eligibility = result.eligibility
        finally:
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None


@dataclass
class RagEmbeddingJobState:
    job_id: str
    status: RunStatus
    payload: RagEmbeddingGenerationPayload
    started_at: datetime | None = None
    finished_at: datetime | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None


class RagEmbeddingJobRegistry:
    """In-memory progress registry for long-running RAG embedding jobs."""

    def __init__(self, service: RagEmbeddingGenerationService) -> None:
        self.service = service
        self._jobs: dict[str, RagEmbeddingJobState] = {}
        self._active_job_id: str | None = None
        self._lock = threading.Lock()

    def create(self, payload: RagEmbeddingGenerationPayload) -> RagEmbeddingJobState:
        with self._lock:
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active is not None and active.status in {"queued", "running"}:
                    raise HTTPException(status_code=409, detail="Another RAG embedding generation is already active.")
            job_id = uuid.uuid4().hex
            job = RagEmbeddingJobState(job_id=job_id, status="queued", payload=payload)
            self._append_event_locked(job, "Geracao enfileirada.", state="running")
            self._jobs[job_id] = job
            self._active_job_id = job_id
        threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
        return job

    def get(self, job_id: str) -> RagEmbeddingJobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = datetime.now()
            self._append_event_locked(job, "Execucao iniciada.", state="running")

        def update_progress(event: dict[str, Any]) -> None:
            with self._lock:
                job = self._jobs[job_id]
                self._append_event_locked(
                    job,
                    str(event.get("message") or "Progresso atualizado."),
                    state=str(event.get("state") or "running"),
                    metadata={key: value for key, value in event.items() if key not in {"message", "state"}},
                )

        try:
            result = self.service.run(
                dataset=job.payload.dataset,
                batch_size=job.payload.batch_size,
                question_sequence_start=job.payload.question_sequence_start,
                question_sequence_end=job.payload.question_sequence_end,
                progress_callback=update_progress,
            )
        except (RuntimeError, ValueError) as error:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.finished_at = datetime.now()
                job.error = str(error)
                self._append_event_locked(job, str(error), state="error")
        else:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "completed"
                job.finished_at = datetime.now()
                job.result = result
                self._append_event_locked(job, "Execucao concluida.", state="done")
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _append_event_locked(
        self,
        job: RagEmbeddingJobState,
        message: str,
        *,
        state: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        job.events.append(
            {
                "index": len(job.events),
                "timestamp": datetime.now().isoformat(),
                "state": state,
                "message": message,
                "metadata": metadata or {},
            }
        )


def create_app(
    service: RunJudgeService | None = None,
    *,
    audit_dir: Path | str = DEFAULT_AUDIT_DIR,
    backup_dir: Path | str = DEFAULT_BACKUP_DIR,
    dashboard_service: DashboardService | None = None,
    dump_service: DatabaseDumpService | None = None,
    database_reset_service: DatabaseResetService | None = None,
    judge_prompt_service: JudgePromptConfigService | None = None,
    meta_evaluation_service: MetaEvaluationService | None = None,
    rag_curation_service: RagCurationService | None = None,
    rag_embedding_config_service: RagEmbeddingConfigService | None = None,
    rag_embedding_smoke_test_service: RagEmbeddingSmokeTestService | None = None,
    rag_embedding_generation_service: RagEmbeddingGenerationService | None = None,
    rag_vector_query_service: RagVectorQueryService | None = None,
    rag_vector_run_service: RagVectorRunService | None = None,
    audit_log_summary_service: AuditLogSummaryService | None = None,
    assistant_service: AssistantService | None = None,
) -> FastAPI:
    app = FastAPI(title="Atividade 2 Judge Console")
    startup_schema_mode = os.environ.get("ENSURE_SCHEMA_ON_STARTUP", "").strip().lower()
    ensure_schema_on_startup = startup_schema_mode in {"1", "true", "yes", "on"}
    assistant_enabled = _env_flag("ENABLE_AI_ASSISTANT", default=False)
    app.state.csrf_token = secrets.token_urlsafe(32)
    app.state.jobs = JobRegistry(service or RunJudgeService())
    resolved_backup_dir = resolve_backup_dir(getattr(dump_service, "output_dir", backup_dir))
    app.state.audit_dir = Path(audit_dir)
    app.state.backup_dir = resolved_backup_dir
    app.state.dashboard = dashboard_service or DashboardService()
    app.state.dump_service = dump_service or DatabaseDumpService(output_dir=resolved_backup_dir)
    app.state.database_reset_service = database_reset_service or DatabaseResetService()
    app.state.judge_prompt_service = judge_prompt_service or JudgePromptConfigService()
    app.state.meta_evaluation_service = meta_evaluation_service or MetaEvaluationService()
    app.state.rag_curation_service = rag_curation_service or RagCurationService()
    app.state.rag_embedding_config_service = rag_embedding_config_service or RagEmbeddingConfigService()
    app.state.rag_embedding_smoke_test_service = (
        rag_embedding_smoke_test_service or RagEmbeddingSmokeTestService()
    )
    app.state.rag_embedding_generation_service = (
        rag_embedding_generation_service or RagEmbeddingGenerationService()
    )
    app.state.rag_embedding_jobs = RagEmbeddingJobRegistry(app.state.rag_embedding_generation_service)
    app.state.rag_vector_query_service = rag_vector_query_service or RagVectorQueryService()
    app.state.rag_vector_run_service = rag_vector_run_service or RagVectorRunService()
    app.state.audit_log_summary_service = audit_log_summary_service or AuditLogSummaryService()
    app.state.assistant_enabled = assistant_enabled
    app.state.assistant_service = assistant_service or AssistantService(
        dashboard_service=app.state.dashboard,
        audit_log_summary_service=app.state.audit_log_summary_service,
    )

    @app.on_event("startup")
    def ensure_runtime_schema() -> None:
        if not ensure_schema_on_startup:
            return
        settings = load_settings()
        connection = connect(settings.database_url)
        try:
            JudgeRepository(connection).ensure_schema()
        finally:
            connection.close()

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        assistant_hidden = " hidden" if not request.app.state.assistant_enabled else ""
        return HTMLResponse(
            _INDEX_HTML.replace("__ASSISTANT_WIDGET_HIDDEN__", assistant_hidden),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/config")
    def get_config(request: Request) -> dict:
        config = request.app.state.jobs.service.describe_config()
        defaults = config.get("defaults") or {}
        configured_models = [
            defaults.get("judge_model"),
            defaults.get("secondary_judge_model"),
            defaults.get("arbiter_judge_model"),
        ]
        config["judge_model_options"] = list(dict.fromkeys(model for model in configured_models if model))
        config["csrf_token"] = request.app.state.csrf_token
        config["feature_flags"] = {"ai_assistant": request.app.state.assistant_enabled}
        return config

    @app.get("/api/dashboard")
    def get_dashboard(request: Request) -> dict:
        try:
            filters = parse_dashboard_filters(
                {
                    "dataset": request.query_params.get("dataset"),
                    "candidate_model": request.query_params.get("candidate_model"),
                    "judge_model": request.query_params.get("judge_model"),
                    "status": request.query_params.get("status"),
                    "date_from": request.query_params.get("date_from"),
                    "date_to": request.query_params.get("date_to"),
                    "group_by": request.query_params.get("group_by"),
                }
            )
            return request.app.state.dashboard.load(filters)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/api/judge-prompts/options")
    def get_judge_prompt_options(request: Request) -> dict:
        try:
            return request.app.state.judge_prompt_service.options()
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/api/judge-prompts")
    def get_judge_prompt_config(dataset: str, request: Request) -> dict:
        try:
            return request.app.state.judge_prompt_service.get(dataset=dataset)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.put("/api/judge-prompts", dependencies=[Depends(_require_csrf)])
    def save_judge_prompt_config(payload: PromptConfigPayload, request: Request) -> dict:
        try:
            return request.app.state.judge_prompt_service.save(
                dataset=payload.dataset,
                prompt=payload.prompt,
                persona=payload.persona,
                context=payload.context,
                rubric=payload.rubric,
                output=payload.output,
                changed_by=payload.changed_by,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/meta-evaluations/options")
    def get_meta_evaluation_options(request: Request) -> dict:
        try:
            return request.app.state.meta_evaluation_service.options()
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/api/meta-evaluations")
    def get_meta_evaluation(evaluation_id: int, request: Request) -> dict:
        try:
            return request.app.state.meta_evaluation_service.get(evaluation_id=evaluation_id)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/meta-evaluations/history")
    def get_meta_evaluation_history(request: Request) -> dict:
        try:
            return request.app.state.meta_evaluation_service.history()
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/api/rag-curation/options")
    def get_rag_curation_options(request: Request) -> dict:
        try:
            return request.app.state.rag_curation_service.options()
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/api/rag-curation")
    def get_rag_curation(dataset: str, request: Request) -> dict:
        try:
            return request.app.state.rag_curation_service.get(dataset=dataset)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/rag-curation/items/{curation_id}")
    def get_rag_curation_detail(curation_id: int, dataset: str, request: Request) -> dict:
        try:
            return request.app.state.rag_curation_service.detail(curation_id=curation_id, dataset=dataset)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/rag-curation/import", dependencies=[Depends(_require_csrf)])
    async def import_rag_curation(request: Request) -> dict:
        filename = request.headers.get("x-curation-filename", "")
        imported_by = request.headers.get("x-curation-imported-by", "")
        if not filename.lower().endswith(".json"):
            raise HTTPException(status_code=400, detail="Selecione um arquivo .json de curadoria.")
        try:
            body = await request.body()
            if not body:
                raise HTTPException(status_code=400, detail="Arquivo de curadoria vazio.")
            text = body.decode("utf-8")
            return request.app.state.rag_curation_service.import_json(
                filename=filename,
                imported_by=imported_by,
                raw_text=text,
            )
        except UnicodeDecodeError as error:
            raise HTTPException(status_code=400, detail="O arquivo de curadoria deve estar em UTF-8.") from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/rag-curation/runs/{run_id}/activate", dependencies=[Depends(_require_csrf)])
    def activate_rag_curation_run(run_id: int, dataset: str, request: Request) -> dict:
        try:
            return request.app.state.rag_curation_service.activate_run(run_id=run_id, dataset=dataset)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/rag-embedding-config")
    def get_rag_embedding_config(dataset: str, request: Request) -> dict:
        try:
            return request.app.state.rag_embedding_config_service.get(dataset=dataset)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.put("/api/rag-embedding-config", dependencies=[Depends(_require_csrf)])
    def save_rag_embedding_config(payload: RagEmbeddingConfigPayload, request: Request) -> dict:
        try:
            return request.app.state.rag_embedding_config_service.save(
                dataset=payload.dataset,
                provider=payload.provider,
                model_name=payload.model_name,
                dimensions=payload.dimensions,
                api_base_url=payload.api_base_url,
                notes=payload.notes,
                updated_by=payload.updated_by,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/rag-embedding-config/smoke-test", dependencies=[Depends(_require_csrf)])
    def run_rag_embedding_smoke_test(payload: RagEmbeddingSmokeTestPayload, request: Request) -> dict:
        try:
            return request.app.state.rag_embedding_smoke_test_service.run(
                dataset=payload.dataset,
                sample_text=payload.sample_text,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/rag-vector/generate-embeddings", dependencies=[Depends(_require_csrf)])
    def generate_rag_vector_embeddings(payload: RagEmbeddingGenerationPayload, request: Request) -> dict:
        try:
            result = request.app.state.rag_embedding_generation_service.run(
                dataset=payload.dataset,
                batch_size=payload.batch_size,
                question_sequence_start=payload.question_sequence_start,
                question_sequence_end=payload.question_sequence_end,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _serialize_rag_embedding_result(result)

    @app.post("/api/rag-vector/generate-embeddings/jobs", dependencies=[Depends(_require_csrf)])
    def start_rag_vector_embedding_job(payload: RagEmbeddingGenerationPayload, request: Request) -> dict:
        try:
            job = request.app.state.rag_embedding_jobs.create(payload)
        except HTTPException:
            raise
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _serialize_rag_embedding_job(job)

    @app.get("/api/rag-vector/generate-embeddings/jobs/{job_id}")
    def get_rag_vector_embedding_job(job_id: str, request: Request) -> dict:
        if not RUN_ID_PATTERN.match(job_id):
            raise HTTPException(status_code=400, detail="Invalid job id.")
        job = request.app.state.rag_embedding_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="RAG embedding job not found.")
        return _serialize_rag_embedding_job(job)

    @app.post("/api/rag-vector/runs/{run_id}/activate", dependencies=[Depends(_require_csrf)])
    def activate_rag_vector_run(run_id: int, dataset: str, request: Request) -> dict:
        try:
            result = request.app.state.rag_vector_run_service.activate(run_id=run_id, dataset=dataset)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            **result,
            "vector_base": asdict(result["vector_base"])
            if result.get("vector_base") is not None and hasattr(result["vector_base"], "__dataclass_fields__")
            else result.get("vector_base"),
            "runs": [
                asdict(item) if hasattr(item, "__dataclass_fields__") else item
                for item in (result.get("runs") or [])
            ],
        }

    @app.delete("/api/rag-vector/runs/{run_id}", dependencies=[Depends(_require_csrf)])
    def delete_rag_vector_run(run_id: int, dataset: str, request: Request) -> dict:
        try:
            result = request.app.state.rag_vector_run_service.delete(run_id=run_id, dataset=dataset)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            **result,
            "vector_base": asdict(result["vector_base"])
            if result.get("vector_base") is not None and hasattr(result["vector_base"], "__dataclass_fields__")
            else result.get("vector_base"),
            "runs": [
                asdict(item) if hasattr(item, "__dataclass_fields__") else item
                for item in (result.get("runs") or [])
            ],
        }

    @app.get("/api/rag-vector/preview")
    def get_rag_vector_preview(request: Request, dataset: str, limit: int = 8) -> dict:
        try:
            data = request.app.state.rag_vector_query_service.preview(dataset=dataset, limit=limit)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        vector_base = data.get("vector_base")
        return {
            **data,
            "vector_base": asdict(vector_base)
            if vector_base is not None and hasattr(vector_base, "__dataclass_fields__")
            else vector_base,
        }

    @app.post("/api/rag-vector/search", dependencies=[Depends(_require_csrf)])
    def search_rag_vector(payload: RagVectorSearchPayload, request: Request) -> dict:
        try:
            return request.app.state.rag_vector_query_service.search(
                dataset=payload.dataset,
                query_text=payload.query_text,
                top_k=payload.top_k,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/operational-log-summary")
    def get_operational_log_summary(request: Request) -> dict:
        return request.app.state.audit_log_summary_service.load()

    @app.post("/api/assistant/chat", dependencies=[Depends(_require_csrf)])
    def assistant_chat(payload: AssistantPayload, request: Request) -> dict:
        try:
            return request.app.state.assistant_service.answer(payload.message)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.put("/api/meta-evaluations", dependencies=[Depends(_require_csrf)])
    def save_meta_evaluation(payload: MetaEvaluationPayload, request: Request) -> dict:
        try:
            return request.app.state.meta_evaluation_service.save(
                meta_evaluation_id=payload.meta_evaluation_id,
                evaluation_id=payload.evaluation_id,
                evaluator_name=payload.evaluator_name,
                score=payload.score,
                rationale=payload.rationale,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.delete("/api/meta-evaluations/{meta_evaluation_id}", dependencies=[Depends(_require_csrf)])
    def delete_meta_evaluation(meta_evaluation_id: int, evaluation_id: int, request: Request) -> dict:
        try:
            return request.app.state.meta_evaluation_service.delete(
                meta_evaluation_id=meta_evaluation_id,
                evaluation_id=evaluation_id,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/runs/dry-run", dependencies=[Depends(_require_csrf)])
    def dry_run(payload: RunPayload, request: Request) -> dict:
        service: RunJudgeService = request.app.state.jobs.service
        try:
            result = service.run(payload.to_request(dry_run=True))
        except (ConfigurationError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _serialize_result(result)

    @app.post("/api/runs", dependencies=[Depends(_require_csrf)])
    def create_run(payload: RunPayload, request: Request) -> dict:
        registry: JobRegistry = request.app.state.jobs
        try:
            job = registry.create(payload.to_request(dry_run=False))
        except ConfigurationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _serialize_job(job)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, request: Request) -> dict:
        registry: JobRegistry = request.app.state.jobs
        job = registry.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return _serialize_job(job)

    @app.post("/api/runs/{run_id}/cancel", dependencies=[Depends(_require_csrf)])
    def cancel_run(run_id: str, request: Request) -> dict:
        registry: JobRegistry = request.app.state.jobs
        job = registry.cancel(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return _serialize_job(job)

    @app.get("/api/runs/{run_id}/audit-log", response_class=PlainTextResponse)
    def get_audit_log(run_id: str, request: Request) -> str:
        registry: JobRegistry = request.app.state.jobs
        job = registry.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        if not job.audit_log:
            raise HTTPException(status_code=404, detail="Audit log not available.")
        audit_path = Path(job.audit_log)
        if not audit_path.exists() or not audit_path.is_file():
            raise HTTPException(status_code=404, detail="Audit log file not found.")
        return audit_path.read_text(encoding="utf-8")

    @app.get("/api/run-history")
    def get_run_history(request: Request) -> list[dict]:
        return _list_run_history(request.app.state.audit_dir)

    @app.get("/api/run-history/export.json")
    def export_run_history_json(request: Request) -> list[dict]:
        return _list_run_history(request.app.state.audit_dir)

    @app.get("/api/run-history/export.csv")
    def export_run_history_csv(request: Request) -> Response:
        rows = _list_run_history(request.app.state.audit_dir)
        output = StringIO()
        fieldnames = [
            "run_id",
            "timestamp",
            "mode",
            "dataset",
            "batch_size",
            "successes",
            "failures",
            "duration",
            "log_path",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
        return Response(content=output.getvalue(), media_type="text/csv; charset=utf-8")

    @app.get("/api/run-history/{run_id}/audit-log", response_class=PlainTextResponse)
    def get_run_history_audit_log(run_id: str, request: Request) -> str:
        audit_path = _resolve_history_log_path(request.app.state.audit_dir, run_id)
        if not audit_path.exists() or not audit_path.is_file():
            raise HTTPException(status_code=404, detail="Audit log file not found.")
        return audit_path.read_text(encoding="utf-8")

    @app.post("/api/database-dumps", dependencies=[Depends(_require_csrf)])
    def create_database_dump(request: Request) -> dict:
        try:
            result = request.app.state.dump_service.create_dump()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return asdict(result)

    @app.post("/api/database-reset", dependencies=[Depends(_require_csrf)])
    def reset_database(request: Request) -> dict:
        try:
            return request.app.state.database_reset_service.reset_to_initial_state()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/api/database-restore", dependencies=[Depends(_require_csrf)])
    async def restore_database_backup(request: Request) -> dict:
        filename = request.headers.get("x-backup-filename", "")
        if not filename.endswith(".sql"):
            raise HTTPException(status_code=400, detail="Selecione um arquivo .sql.")
        restore_dir = request.app.state.backup_dir / ".restore_uploads"
        restore_dir.mkdir(parents=True, exist_ok=True)
        restore_path = restore_dir / f"{uuid.uuid4()}_{Path(filename).name}"
        try:
            body = await request.body()
            if not body:
                raise HTTPException(status_code=400, detail="Arquivo de backup vazio.")
            restore_path.write_bytes(body)
            return request.app.state.database_reset_service.restore_backup(restore_path)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        finally:
            restore_path.unlink(missing_ok=True)

    @app.get("/api/database-dumps/{filename}")
    def download_database_dump(filename: str, request: Request) -> FileResponse:
        try:
            dump_path = resolve_dump_path(request.app.state.backup_dir, filename)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if not dump_path.exists() or not dump_path.is_file():
            raise HTTPException(status_code=404, detail="Database dump file not found.")
        return FileResponse(
            dump_path,
            media_type="application/sql; charset=utf-8",
            filename=filename,
        )

    return app


def _require_csrf(request: Request) -> None:
    token = request.headers.get("x-csrf-token")
    if not token or token != request.app.state.csrf_token:
        raise HTTPException(status_code=403, detail="Invalid CSRF token.")


def _serialize_job(job: JobState) -> dict:
    eligibility = job.eligibility
    if eligibility is None and job.result is not None:
        eligibility = job.result.eligibility
    return {
        "run_id": job.run_id,
        "status": job.status,
        "started_at": job.started_at.isoformat() if job.started_at is not None else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at is not None else None,
        "duration_seconds": _duration_seconds(job.started_at, job.finished_at, 0),
        "duration": _format_duration(_duration_seconds(job.started_at, job.finished_at, 0)),
        "progress": asdict(_effective_progress(job, eligibility)),
        "audit_log": job.audit_log,
        "audit_log_url": f"/api/runs/{job.run_id}/audit-log" if job.audit_log else None,
        "command_preview": job.command_preview,
        "eligibility": asdict(eligibility) if eligibility is not None else None,
        "evaluation_events": [asdict(event) for event in _sorted_evaluation_events(job.evaluation_events)],
        "error": job.error,
        "result": _serialize_result(job.result) if job.result is not None else None,
    }


def _serialize_rag_embedding_job(job: RagEmbeddingJobState) -> dict:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "started_at": job.started_at.isoformat() if job.started_at is not None else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at is not None else None,
        "duration_seconds": _duration_seconds(job.started_at, job.finished_at, 0),
        "duration": _format_duration(_duration_seconds(job.started_at, job.finished_at, 0)),
        "payload": job.payload.dict(),
        "events": job.events,
        "error": job.error,
        "result": _serialize_rag_embedding_result(job.result) if job.result is not None else None,
    }


def _serialize_rag_embedding_result(result: dict[str, Any] | None) -> dict | None:
    if result is None:
        return None
    summary = result.get("summary")
    return {
        "materialized_base": bool(result.get("materialized_base")),
        "question_sequence_range": result.get("question_sequence_range") or {},
        "source_url_summary": result.get("source_url_summary") or {},
        "chunk_summary": result.get("chunk_summary") or {},
        "summary": asdict(summary)
        if summary is not None and hasattr(summary, "__dataclass_fields__")
        else summary,
    }


def _effective_progress(job: JobState, eligibility: EligibilitySummary | None) -> BatchProgress:
    if job.result is not None and job.result.summary is not None:
        summary = job.result.summary
        total = summary.selected_answers
        return BatchProgress(
            current=total,
            total=total,
            percent=100,
            executed_evaluations=summary.executed_evaluations,
            skipped_evaluations=summary.skipped_evaluations,
            arbiter_evaluations=summary.arbiter_evaluations,
        )
    if job.status in {"completed", "failed"}:
        total = job.progress.total
        if total == 0 and eligibility is not None:
            total = eligibility.will_process
        return BatchProgress(
            current=total,
            total=total,
            percent=100,
            executed_evaluations=job.progress.executed_evaluations,
            skipped_evaluations=job.progress.skipped_evaluations,
            arbiter_evaluations=job.progress.arbiter_evaluations,
        )
    if eligibility is not None and job.progress.total == 0:
        return BatchProgress(
            current=0,
            total=eligibility.will_process,
            percent=0,
            executed_evaluations=job.progress.executed_evaluations,
            skipped_evaluations=job.progress.skipped_evaluations,
            arbiter_evaluations=job.progress.arbiter_evaluations,
        )
    return job.progress


def _serialize_result(result: RunJudgeResult) -> dict:
    return {
        "dry_run": result.dry_run,
        "audit_log": result.audit_log,
        "execution_summary": result.execution_summary,
        "command_preview": result.command_preview,
        "batch_size": result.batch_size,
        "eligibility": asdict(result.eligibility) if result.eligibility is not None else None,
        "summary": _serialize_summary(result.summary),
    }


def _serialize_summary(summary: PipelineSummary | None) -> dict | None:
    if summary is None:
        return None
    return asdict(summary)


def _list_run_history(audit_dir: Path) -> list[dict]:
    if not audit_dir.exists() or not audit_dir.is_dir():
        return []
    entries = [_parse_audit_log(path) for path in audit_dir.glob("*.log") if path.is_file()]
    entries.sort(key=lambda entry: entry.timestamp or "", reverse=True)
    return [asdict(entry) for entry in entries]


def _parse_audit_log(path: Path) -> RunHistoryEntry:
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    mode: str | None = None
    dataset: str | None = None
    batch_size: int | None = None
    successes = 0
    failures = 0
    summary: str | None = None
    elapsed_ms_total = 0

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = _parse_audit_line(line)
        if parsed is None:
            continue
        timestamp, message, detail = parsed
        first_timestamp = first_timestamp or timestamp
        last_timestamp = timestamp
        elapsed_ms_total += _extract_elapsed_ms(detail)
        if message == "execution_summary":
            summary = detail
            mode = _extract_summary_value(detail, "Judge mode")
        elif message == "command_preview":
            dataset = _extract_cli_arg(detail, "--dataset") or dataset
            batch_size = _parse_int(_extract_cli_arg(detail, "--batch-size")) or batch_size
        elif message.startswith("START Counting eligible answers for ") or message.startswith(
            "START Selecting pending candidate answers for "
        ):
            values = _key_values(detail)
            dataset = values.get("dataset") or dataset
            batch_size = _parse_int(values.get("batch_size")) or batch_size
        elif message == "execution_result":
            values = _key_values(detail)
            successes = _parse_int(values.get("executed")) or successes
        if _is_failure_event(message, detail):
            failures += 1

    duration_seconds = _duration_seconds(first_timestamp, last_timestamp, elapsed_ms_total)
    run_id = path.stem
    return RunHistoryEntry(
        run_id=run_id,
        timestamp=first_timestamp.isoformat() if first_timestamp is not None else None,
        finished_at=last_timestamp.isoformat() if last_timestamp is not None else None,
        mode=mode,
        dataset=dataset,
        batch_size=batch_size,
        successes=successes,
        failures=failures,
        duration_seconds=duration_seconds,
        duration=_format_duration(duration_seconds),
        log_path=str(path),
        log_url=f"/api/run-history/{run_id}/audit-log",
        summary=summary,
    )


def _parse_audit_line(line: str) -> tuple[datetime, str, str | None] | None:
    match = AUDIT_TIMESTAMP_PATTERN.match(line)
    if not match:
        return None
    try:
        timestamp = datetime.fromisoformat(match.group(1).strip())
    except ValueError:
        return None
    return timestamp, match.group(2).strip(), match.group(3).strip() if match.group(3) else None


def _extract_summary_value(summary: str | None, label: str) -> str | None:
    if not summary:
        return None
    prefix = f"{label}: "
    for part in (value.strip() for value in summary.split("|")):
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
    for index, part in enumerate(parts[:-1]):
        if part == option:
            return parts[index + 1]
    return None


def _key_values(detail: str | None) -> dict[str, str]:
    if not detail:
        return {}
    return {match.group(1): match.group(2) for match in AUDIT_KEY_VALUE_PATTERN.finditer(detail)}


def _extract_elapsed_ms(detail: str | None) -> int:
    return _parse_int(_key_values(detail).get("elapsed_ms")) or 0


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _is_failure_event(message: str, detail: str | None) -> bool:
    if message.startswith("FAIL ") or message == "audit_log_failed":
        return True
    return "status=failed" in (detail or "")


def _duration_seconds(
    first_timestamp: datetime | None,
    last_timestamp: datetime | None,
    elapsed_ms_total: int,
) -> int | None:
    if first_timestamp is not None and last_timestamp is not None:
        return max(0, round((last_timestamp - first_timestamp).total_seconds()))
    if elapsed_ms_total:
        return round(elapsed_ms_total / 1000)
    return None


def _format_duration(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{remaining_minutes:02d}min{remaining_seconds:02d}s"
    if minutes:
        return f"{minutes}min{remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def _resolve_history_log_path(audit_dir: Path, run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="Invalid run id.")
    root = audit_dir.resolve()
    path = (root / f"{run_id}.log").resolve()
    if path.parent != root:
        raise HTTPException(status_code=400, detail="Invalid run id.")
    return path


def _upsert_evaluation_event(events: list[EvaluationProgress], event: EvaluationProgress) -> None:
    event_key = _evaluation_event_key(event)
    for index, existing in enumerate(events):
        if _evaluation_event_key(existing) == event_key:
            events[index] = event
            return
    events.append(event)


def _sync_progress_with_execution_table(job: JobState) -> None:
    total = job.progress.total
    if total == 0 and job.eligibility is not None:
        total = job.eligibility.will_process
    if total <= 0:
        return
    successful_answer_ids = {event.answer_id for event in job.evaluation_events if event.status == "success"}
    current = min(total, len(successful_answer_ids))
    if current == job.progress.current and total == job.progress.total:
        return
    job.progress = BatchProgress(
        current=current,
        total=total,
        percent=int(current / total * 100),
        executed_evaluations=job.progress.executed_evaluations,
        skipped_evaluations=job.progress.skipped_evaluations,
        arbiter_evaluations=job.progress.arbiter_evaluations,
    )


def _sorted_evaluation_events(events: list[EvaluationProgress]) -> list[EvaluationProgress]:
    status_priority = {"running": 0, "failed": 1, "success": 2}
    return sorted(
        events,
        key=lambda event: (
            status_priority.get(event.status, 3),
            event.dataset,
            -event.question_id,
            event.answer_id,
            event.candidate_model,
            event.judge_model,
            event.role,
            event.panel_mode,
            event.trigger_reason or "",
        ),
    )


def _evaluation_event_key(event: EvaluationProgress) -> tuple:
    return (
        event.dataset,
        event.question_id,
        event.answer_id,
        event.candidate_model,
        event.judge_model,
        event.role,
        event.panel_mode,
        event.trigger_reason,
    )


def _initial_progress() -> BatchProgress:
    return BatchProgress(
        current=0,
        total=0,
        percent=0,
        executed_evaluations=0,
        skipped_evaluations=0,
        arbiter_evaluations=0,
    )


_INDEX_HTML = """
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Atividade 2 Judge Console</title>
  <style>
    :root { color-scheme: light; --ink:#18212f; --muted:#5b6472; --line:#d8dde6; --bg:#f6f7f9; --accent:#1769aa; --ok:#1d7f4e; --bad:#b42318; --warn:#9a5b00; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
    header { padding:20px 28px 12px; border-bottom:1px solid var(--line); background:#fff; }
    h1 { margin:0 0 6px; font-size:22px; letter-spacing:0; }
    .tabs { display:flex; gap:8px; padding:12px 28px 0; background:#fff; border-bottom:1px solid var(--line); }
    .tab-button { min-height:34px; color:var(--muted); background:#fff; border-color:transparent; border-bottom:2px solid transparent; border-radius:0; }
    .tab-button.active { color:var(--accent); border-bottom-color:var(--accent); }
    .rag-subtabs { display:flex; gap:8px; margin:0 0 16px; padding-bottom:10px; border-bottom:1px solid var(--line); overflow:auto; }
    .rag-subtab-button { min-height:34px; color:var(--muted); background:#fff; border-color:var(--line); }
    .rag-subtab-button.active { color:#fff; background:var(--accent); border-color:var(--accent); }
    .rag-subpanel[hidden] { display:none; }
    .tab-panel[hidden] { display:none; }
    main { width:min(100%, 1440px); margin:0 auto; padding:20px; display:grid; grid-template-columns: minmax(320px,380px) minmax(0,1fr); gap:18px; }
    section, aside { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px; }
    section, aside { min-width:0; }
    aside { padding-bottom:82px; }
    h2 { font-size:15px; margin:0 0 12px; }
    label { display:grid; gap:5px; margin:10px 0; color:var(--muted); font-size:12px; }
    input, select, textarea { width:100%; min-height:36px; border:1px solid var(--line); border-radius:6px; padding:7px 9px; font:inherit; color:var(--ink); background:#fff; }
    textarea { min-height:112px; resize:vertical; }
    button { border:1px solid var(--accent); background:var(--accent); color:#fff; border-radius:6px; min-height:36px; padding:0 12px; font-weight:650; cursor:pointer; }
    .button-link { display:inline-flex; align-items:center; min-height:32px; padding:0 10px; border:1px solid var(--accent); border-radius:6px; color:var(--accent); background:#fff; font-size:12px; font-weight:650; text-decoration:none; }
    button.secondary { color:var(--accent); background:#fff; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .judge-block { border-top:1px solid var(--line); padding-top:10px; margin-top:10px; }
    .endpoint-fields[hidden] { display:none; }
    .secret-row { display:grid; grid-template-columns:1fr 38px; gap:8px; align-items:center; }
    .icon-button { min-height:36px; padding:0; border-color:var(--line); background:#fff; color:var(--ink); font-size:16px; font-weight:500; }
    .inline { display:flex; align-items:center; gap:8px; margin:8px 0; color:var(--muted); font-size:12px; }
    .inline input { width:auto; min-height:auto; }
    .hint { color:var(--muted); font-size:12px; line-height:1.35; margin:-4px 0 8px; }
    .warn { color:var(--warn); }
    details { border-top:1px solid var(--line); margin-top:12px; padding-top:10px; }
    summary { cursor:pointer; color:var(--ink); font-size:13px; font-weight:650; }
    .status-icon { display:inline-grid; place-items:center; width:18px; height:18px; margin-right:6px; vertical-align:-3px; }
    .spinner { border:2px solid #c9d2de; border-top-color:var(--accent); border-radius:50%; animation:spin .8s linear infinite; }
    @keyframes spin { to { transform:rotate(360deg); } }
    .actions { position:sticky; bottom:0; display:flex; gap:10px; margin:14px -16px -16px; padding:12px 16px; border-top:1px solid var(--line); background:#fff; border-radius:0 0 8px 8px; }
    .actions button { flex:1; }
    .presets { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px; }
    .presets button { min-height:32px; font-size:12px; }
    .status { font-size:13px; color:var(--muted); }
    pre { max-width:100%; overflow:auto; background:#101828; color:#f9fafb; border-radius:6px; padding:12px; min-height:76px; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; }
    progress { width:100%; height:22px; accent-color:var(--accent); }
    table { width:100%; border-collapse:collapse; margin-top:12px; font-size:13px; }
    th, td { text-align:left; border-bottom:1px solid var(--line); padding:8px; }
    .table-wrap { width:100%; max-width:100%; overflow:auto; border:1px solid var(--line); border-radius:8px; margin-top:12px; }
    .table-wrap table { min-width:1180px; margin-top:0; }
    .history-layout { width:min(100%, 1440px); margin:0 auto; padding:20px; display:grid; grid-template-columns:minmax(0,1fr) minmax(320px,480px); gap:18px; }
    .history-actions { display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap; }
    .history-actions div { display:flex; gap:8px; }
    .history-row { cursor:pointer; }
    .history-row:hover { background:#f7fbff; }
    .history-row.selected { background:#eef7ff; box-shadow:inset 3px 0 0 var(--accent); }
    .history-log { min-height:520px; max-height:calc(100vh - 260px); }
    .history-export-links { display:flex; gap:8px; white-space:nowrap; }
    .audit-log-row { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:8px; align-items:center; }
    .audit-log-row a { overflow-wrap:anywhere; }
    .audit-log-button { display:inline-flex; align-items:center; justify-content:center; gap:6px; min-height:30px; padding:0 9px; border-color:var(--line); background:#fff; color:var(--accent); font-size:12px; white-space:nowrap; }
    .audit-log-button-icon { font-size:15px; line-height:1; }
    .audit-log-button:disabled { color:var(--muted); }
    .audit-log-content { min-height:420px; max-height:calc(100vh - 210px); margin:0; overflow:auto; }
    .dashboard-layout { width:min(100%, 1440px); margin:0 auto; padding:20px; display:grid; grid-template-columns:minmax(280px,340px) minmax(0,1fr); gap:18px; }
    .dashboard-head { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:14px; }
    .dashboard-head p { margin:4px 0 0; color:var(--muted); font-size:13px; line-height:1.4; }
    .dashboard-actions { position:relative; display:flex; flex-direction:column; align-items:flex-end; gap:6px; min-width:230px; }
    .database-actions-toggle { display:inline-flex; align-items:center; justify-content:center; gap:8px; min-width:174px; min-height:38px; padding:0 12px; border-color:var(--accent); background:var(--accent); color:#fff; font-size:13px; font-weight:750; box-shadow:0 6px 14px rgba(23,105,170,.22); }
    .database-actions-toggle-icon { font-size:17px; line-height:1; }
    .database-actions-toggle-caret { font-size:11px; line-height:1; opacity:.9; }
    .database-actions-menu { position:absolute; top:42px; right:0; z-index:20; display:grid; gap:4px; width:240px; padding:6px; border:1px solid var(--line); border-radius:8px; background:#fff; box-shadow:0 12px 28px rgba(16,24,40,.14); }
    .database-actions-menu[hidden] { display:none; }
    .database-actions-menu button { width:100%; min-height:34px; border-color:transparent; background:#fff; color:var(--ink); text-align:left; font-size:13px; }
    .database-actions-menu button:hover { background:#f2f8fd; color:var(--accent); }
    .database-actions-menu button.danger { color:var(--bad); }
    .database-actions-menu button.danger:hover { background:#fff5f5; }
    .prompt-layout { width:min(100%, 1440px); margin:0 auto; padding:20px; display:grid; grid-template-columns:minmax(320px,420px) minmax(0,1fr); gap:18px; }
    .prompt-layout > section, .prompt-layout > aside { min-width:0; }
    .prompt-status { margin-top:10px; }
    .rag-operation-panel { margin-top:18px; border:1px solid var(--line); border-radius:8px; background:#fff; padding:12px; }
    .rag-operation-panel h3 { margin:0 0 8px; font-size:13px; }
    .rag-progress-log { display:grid; gap:6px; min-height:120px; max-height:360px; overflow:auto; padding:10px; border:1px solid var(--line); border-radius:8px; background:#f8fbfe; font-size:12px; color:var(--muted); }
    .rag-progress-log[hidden] { display:none; }
    .rag-progress-placeholder { color:var(--muted); font-size:12px; }
    .rag-progress-item { display:grid; grid-template-columns:64px 24px minmax(0,1fr); gap:7px; align-items:start; }
    .rag-progress-time { color:#64748b; font-variant-numeric:tabular-nums; }
    .rag-progress-marker { display:grid; place-items:center; width:24px; min-height:18px; border-radius:999px; background:#dbeafe; color:var(--accent); font-size:10px; font-weight:800; line-height:1; }
    .rag-progress-item.done .rag-progress-marker { background:#dcfce7; color:#166534; }
    .rag-progress-item.error .rag-progress-marker { background:#fee2e2; color:#991b1b; }
    .rag-progress-text { min-width:0; overflow-wrap:anywhere; line-height:1.35; }
    .prompt-log-table table { min-width:1480px; }
    .prompt-preview { margin-top:18px; display:grid; grid-template-columns:repeat(auto-fit, minmax(280px,1fr)); gap:10px; align-items:start; }
    .prompt-preview-card { border:1px solid var(--line); border-radius:8px; background:#fff; padding:12px; min-width:0; overflow:hidden; }
    .prompt-preview-card h3 { margin:0 0 8px; font-size:13px; }
    .prompt-preview-card pre { margin:0; white-space:pre-wrap; word-break:break-word; max-height:320px; overflow:auto; }
    .rag-curation-table table { min-width:980px; }
    .rag-curation-articles-table table { min-width:720px; }
    .rag-curation-detail-grid { grid-template-columns:repeat(auto-fit, minmax(320px,1fr)); }
    .meta-mode-switch { width:min(100%, 1440px); margin:0 auto; padding:20px 20px 0; display:flex; gap:8px; }
    .meta-mode-button { min-height:34px; border-color:var(--line); background:#fff; color:var(--muted); }
    .meta-mode-button.active { border-color:var(--accent); color:#fff; background:var(--accent); }
    .meta-mode-panel[hidden] { display:none; }
    .meta-history-table table { min-width:980px; }
    .meta-history-table th button { width:100%; min-height:auto; padding:0; border:0; background:transparent; color:var(--ink); text-align:left; font:inherit; cursor:pointer; }
    .meta-history-table tr.selected { background:#eef7ff; }
    .meta-history-table tbody tr { cursor:pointer; }
    .meta-detail-actions { display:flex; gap:8px; margin-top:10px; }
    .meta-detail-actions button { flex:1; }
    .meta-evaluation-field { position:relative; }
    .meta-evaluation-input-wrap { position:relative; }
    .meta-evaluation-input-wrap input { padding-right:40px; }
    .meta-clear-button { position:absolute; top:50%; right:8px; transform:translateY(-50%); display:grid; place-items:center; width:26px; height:26px; min-height:26px; padding:0; border-color:transparent; border-radius:50%; background:#eef2f7; color:var(--muted); font-size:18px; font-weight:800; line-height:1; }
    .meta-clear-button:hover, .meta-clear-button:focus { background:#dbeafe; color:var(--accent); outline:none; }
    .meta-options-list { position:absolute; top:100%; left:0; z-index:30; display:grid; gap:2px; width:min(760px, calc(100vw - 56px)); max-height:420px; overflow:auto; margin-top:4px; padding:6px; border:1px solid var(--line); border-radius:8px; background:#fff; box-shadow:0 14px 32px rgba(16,24,40,.18); }
    .meta-options-list[hidden] { display:none; }
    .meta-option { width:100%; min-height:34px; padding:8px 10px; border-color:transparent; background:#fff; color:var(--ink); text-align:left; font-size:13px; font-weight:600; line-height:1.35; overflow-wrap:anywhere; }
    .meta-option:hover, .meta-option:focus { background:#f2f8fd; color:var(--accent); outline:none; }
    .meta-option-empty { padding:8px 10px; color:var(--muted); font-size:13px; }
    .dashboard-filters select[multiple] { min-height:92px; }
    .dashboard-filter-actions { display:flex; gap:8px; margin-top:12px; }
    .dashboard-filter-actions button { flex:1; }
    .dashboard-note { color:var(--muted); font-size:12px; line-height:1.45; margin-top:10px; }
    .dashboard-table table { min-width:960px; }
    .judge-agreement-table table { min-width:1160px; }
    .post-run-panel { margin-top:18px; border-top:1px solid var(--line); padding-top:16px; }
    .metric-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px,1fr)); gap:10px; margin:10px 0 16px; }
    .metric-card { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcfe; min-width:0; }
    .metric-value { display:block; font-size:22px; font-weight:750; line-height:1.15; overflow-wrap:anywhere; }
    .metric-label { display:block; color:var(--muted); font-size:12px; margin-top:3px; }
    .operational-card { border-color:#c6d8e7; background:#f8fbfd; }
    .operational-section { margin:14px 0 0; padding-top:14px; border-top:1px solid var(--line); }
    .operational-section[hidden] { display:none; }
    .operational-section h3 { margin:0 0 6px; font-size:13px; }
    .operational-section table { margin-top:8px; }
    .operational-blocks { display:grid; gap:14px; }
    .operational-category-table table { min-width:420px; }
    .operational-category-table th:nth-child(n+2), .operational-category-table td:nth-child(n+2) { width:118px; text-align:right; }
    .chart-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px,1fr)); gap:14px; }
    .chart { border:1px solid var(--line); border-radius:8px; padding:12px; min-width:0; }
    .chart h3 { margin:0 0 10px; font-size:13px; }
    .carousel-card { border:1px solid #b9d5eb; border-radius:8px; padding:12px; margin:0 0 14px; background:#fff; overflow:hidden; box-shadow:0 8px 22px rgba(23,105,170,.08); }
    .carousel-head { display:flex; justify-content:space-between; align-items:center; gap:12px; min-width:0; margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid var(--line); }
    .carousel-controls { flex:0 0 auto; display:flex; align-items:center; gap:8px; }
    .carousel-button { width:38px; min-height:34px; padding:0; border-color:#b9d5eb; background:#f2f8fd; color:var(--accent); font-size:20px; line-height:1; }
    .carousel-tabs { flex:1 1 auto; display:flex; align-items:center; gap:6px; min-width:0; max-width:100%; overflow-x:auto; overflow-y:hidden; scroll-behavior:smooth; scroll-padding-left:8px; scroll-padding-right:8px; scrollbar-width:thin; }
    .carousel-tab { flex:0 0 auto; min-height:36px; padding:0 12px; border-color:var(--line); background:#fff; color:var(--muted); font-size:13px; font-weight:750; white-space:nowrap; }
    .carousel-tab.active { border-color:var(--accent); background:var(--accent); color:#fff; box-shadow:0 6px 14px rgba(23,105,170,.24); }
    .carousel-viewport { width:100%; overflow:hidden; touch-action:pan-y; }
    .carousel-track { display:flex; gap:0; width:100%; transform:translateX(0); transition:transform .24s ease; }
    .dashboard-carousel-slide { flex:0 0 100%; width:100%; min-width:100%; padding:2px; }
    .dashboard-carousel-slide h3 { margin:0 0 10px; font-size:13px; }
    .dashboard-carousel-slide .metric-grid { margin-bottom:0; }
    .model-distribution-list { display:grid; gap:10px; }
    .model-distribution-card { border:1px solid var(--line); border-radius:8px; padding:12px; background:#fbfcfe; }
    .model-card-head { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:10px; }
    .model-card-title { min-width:0; font-weight:750; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .model-card-meta { color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; white-space:nowrap; }
    .stacked-bar { display:flex; height:18px; border-radius:999px; overflow:hidden; background:#e5e9f0; box-shadow:inset 0 0 0 1px rgba(24,33,47,.04); }
    .stacked-segment { min-width:0; transition:width .24s ease; }
    .stacked-segment.score-1 { background:#b42318; }
    .stacked-segment.score-2 { background:#d97706; }
    .stacked-segment.score-3 { background:#1769aa; }
    .stacked-segment.score-4 { background:#1d7f4e; }
    .stacked-segment.score-5 { background:#0f766e; }
    .score-legend { display:grid; grid-template-columns:repeat(5, minmax(72px,1fr)); gap:6px; margin-top:10px; overflow:auto; padding-bottom:2px; }
    .score-chip { display:flex; justify-content:space-between; align-items:center; gap:4px; min-width:0; border:1px solid var(--line); border-radius:6px; padding:5px 6px; color:var(--muted); font-size:12px; }
    .score-swatch { width:8px; height:8px; border-radius:999px; flex:0 0 auto; }
    .score-chip strong { color:var(--ink); font-variant-numeric:tabular-nums; }
    .carousel-empty { width:100%; }
    .heatmap-wrap { width:100%; overflow:auto; border:1px solid var(--line); border-radius:8px; }
    .dashboard-compact-viz { max-height:280px; overflow:auto; }
    .heatmap-grid { display:grid; min-width:620px; font-size:12px; }
    .heatmap-cell { min-height:42px; display:flex; align-items:center; justify-content:center; padding:8px; border-right:1px solid rgba(255,255,255,.42); border-bottom:1px solid rgba(255,255,255,.42); font-variant-numeric:tabular-nums; }
    .heatmap-head { background:#eef3f8; color:var(--ink); font-weight:750; border-color:var(--line); }
    .heatmap-model { justify-content:flex-start; background:#f8fafc; color:var(--ink); font-weight:750; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .heatmap-value { color:#111827; font-weight:750; }
    .judge-agreement-layout { display:grid; grid-template-columns:minmax(0,1fr); gap:14px; }
    .boxplot-wrap { width:100%; max-height:280px; overflow:auto; border:1px solid var(--line); border-radius:8px; background:#fbfcfe; padding:10px; }
    .boxplot-list { display:grid; gap:8px; min-width:560px; }
    .boxplot-scale { display:grid; grid-template-columns:96px minmax(240px,1fr) 116px; gap:10px; align-items:center; color:var(--muted); font-size:11px; }
    .boxplot-ticks { position:relative; height:16px; border-bottom:1px solid #d9e1ea; }
    .boxplot-tick { position:absolute; bottom:-15px; transform:translateX(-50%); font-size:10px; color:var(--muted); }
    .boxplot-row { display:grid; grid-template-columns:96px minmax(240px,1fr) 116px; gap:10px; align-items:center; min-height:30px; }
    .boxplot-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#475569; font-size:12px; text-align:right; }
    .boxplot-track { position:relative; height:24px; border-radius:6px; background:linear-gradient(to right, transparent calc(50% - .5px), #cfd8e3 calc(50% - .5px), #cfd8e3 calc(50% + .5px), transparent calc(50% + .5px)), #eef3f8; }
    .boxplot-whisker { position:absolute; top:11px; height:2px; background:#1769aa; border-radius:2px; }
    .boxplot-box { position:absolute; top:5px; height:14px; border:1px solid #1769aa; border-radius:4px; background:#d7ecfb; }
    .boxplot-median { position:absolute; top:2px; width:2px; height:20px; background:#b42318; border-radius:2px; }
    .boxplot-threshold { position:absolute; top:0; bottom:0; width:2px; background:#b42318; opacity:.8; border-radius:2px; }
    .boxplot-meta { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--ink); font-size:11px; font-weight:700; }
    .judge-agreement-copy { margin:0 0 8px; color:var(--muted); font-size:12px; line-height:1.45; max-width:920px; }
    .confusion-layout { display:grid; grid-template-columns:minmax(0,1.45fr) minmax(240px,.8fr); gap:14px; align-items:start; }
    .confusion-summary { display:grid; gap:8px; }
    .confusion-card { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcfe; }
    .confusion-card strong { display:block; font-size:13px; margin-bottom:4px; }
    .confusion-card span { color:var(--muted); font-size:12px; }
    .critical-error-table table { min-width:1120px; }
    .critical-error-table a { color:var(--accent); font-weight:650; text-decoration:none; }
    .scatter-wrap { width:100%; overflow:auto; border:1px solid var(--line); border-radius:8px; background:#fbfcfe; }
    .scatter-svg { display:block; width:100%; min-width:680px; height:auto; }
    .scatter-axis { stroke:#9aa7b7; stroke-width:1; }
    .scatter-grid { stroke:#e5e9f0; stroke-width:1; }
    .scatter-point { fill:#1769aa; fill-opacity:.72; stroke:#fff; stroke-width:1.5; }
    .scatter-label { fill:var(--muted); font-size:12px; }
    .scatter-stat { fill:var(--ink); font-size:13px; font-weight:750; }
    .line-chart-wrap { width:100%; overflow:auto; border:1px solid var(--line); border-radius:8px; background:#fbfcfe; }
    .line-chart-svg { display:block; width:100%; min-width:720px; height:auto; }
    .line-series { fill:none; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; }
    .line-point { stroke:#fff; stroke-width:1.5; }
    .line-legend { display:flex; flex-wrap:wrap; gap:8px 14px; margin-top:10px; color:var(--muted); font-size:12px; }
    .line-legend-item { display:flex; align-items:center; gap:6px; min-width:0; }
    .line-legend-swatch { width:10px; height:10px; border-radius:999px; flex:0 0 auto; }
    .bar-row { display:grid; grid-template-columns:minmax(82px,132px) minmax(88px,1fr) 104px; gap:8px; align-items:center; margin:7px 0; font-size:12px; }
    .bar-label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--muted); }
    .bar-track { height:14px; border-radius:999px; background:#e5e9f0; overflow:hidden; box-shadow:inset 0 0 0 1px rgba(24,33,47,.03); }
    .bar-fill { height:100%; min-width:4px; border-radius:999px; background:linear-gradient(90deg, #1769aa, #1d7f4e); }
    .bar-fill.score-1 { background:#b42318; }
    .bar-fill.score-2 { background:#d97706; }
    .bar-fill.score-3 { background:#1769aa; }
    .bar-fill.score-4 { background:#1d7f4e; }
    .bar-fill.score-5 { background:#0f766e; }
    .bar-fill.failed { background:#b42318; }
    .bar-fill.arbiter { background:#7c3aed; }
    .bar-fill.none { background:#64748b; }
    .bar-fill.zero { min-width:0; }
    .bar-value { display:grid; grid-template-columns:42px 54px; justify-content:end; align-items:center; gap:6px; font-variant-numeric:tabular-nums; color:var(--ink); font-weight:800; white-space:nowrap; font-size:13px; }
    .bar-count { --pill-fill:#1769aa; --pill-bg:#eaf3fb; --pill-pct:0%; width:42px; text-align:center; border-radius:999px; padding:2px 0; color:var(--accent); background:linear-gradient(90deg, color-mix(in srgb, var(--pill-fill) 26%, white) 0 var(--pill-pct), var(--pill-bg) var(--pill-pct) 100%); }
    .bar-count.positive { --pill-fill:#1d7f4e; --pill-bg:#e7f7ee; color:var(--ok); }
    .bar-count.warning { --pill-fill:#9a5b00; --pill-bg:#fff4df; color:var(--warn); }
    .bar-count.bad { --pill-fill:#b42318; --pill-bg:#fff1f0; color:var(--bad); }
    .bar-percent { color:var(--muted); font-weight:600; text-align:right; }
    .badge { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 7px; font-size:12px; white-space:nowrap; }
    .badge.success { color:var(--ok); border-color:#b7dfc8; background:#f0fbf4; }
    .badge.failed { color:var(--bad); border-color:#f0b8b2; background:#fff5f5; }
    .badge.running { color:var(--accent); border-color:#b9d5eb; background:#f2f8fd; }
    .badge.skipped { color:var(--warn); border-color:#ead0a6; background:#fff8eb; }
    #execution-table-body tr { background:#fff; transition:transform .42s ease, background-color .9s ease; }
    #execution-table-body tr.execution-row-enter { animation:execution-row-running 1.4s ease-out; }
    #execution-table-body tr.execution-row-success { animation:execution-row-success 1.5s ease-out; }
    #execution-table-body tr.execution-row-failed { animation:execution-row-failed 1.5s ease-out; }
    @keyframes execution-row-running { 0% { background:#dff0ff; } 100% { background:#fff; } }
    @keyframes execution-row-success { 0% { background:#dff0ff; } 18% { background:#dff0ff; } 45% { background:#dff7e8; } 100% { background:#fff; } }
    @keyframes execution-row-failed { 0% { background:#dff0ff; } 18% { background:#dff0ff; } 45% { background:#ffe4e0; } 100% { background:#fff; } }
    @media (prefers-reduced-motion: reduce) {
      #execution-table-body tr { transition:none; }
      #execution-table-body tr.execution-row-enter,
      #execution-table-body tr.execution-row-success,
      #execution-table-body tr.execution-row-failed { animation:none; }
    }
    .detail-button { min-height:30px; padding:0 9px; border-color:var(--line); background:#fff; color:var(--accent); font-size:12px; }
    dialog { width:min(900px, calc(100vw - 28px)); border:1px solid var(--line); border-radius:8px; padding:0; }
    dialog::backdrop { background:rgba(16,24,40,.42); }
    .dialog-head { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 14px; border-bottom:1px solid var(--line); }
    .dialog-body { padding:14px; }
    .dialog-body h3 { margin:12px 0 6px; font-size:13px; }
    .confirm-dialog { width:min(520px, calc(100vw - 28px)); }
    .confirm-dialog .dialog-body { display:grid; gap:10px; }
    .confirm-dialog p { margin:0; color:var(--muted); font-size:13px; line-height:1.45; }
    .confirm-actions { display:flex; justify-content:flex-end; align-items:center; gap:8px; padding:12px 14px; border-top:1px solid var(--line); }
    .confirm-actions button { min-width:96px; white-space:nowrap; }
    .danger-button { border-color:var(--bad); background:var(--bad); color:#fff; }
    .assistant-widget { position:fixed; right:22px; bottom:22px; z-index:60; display:grid; justify-items:end; gap:10px; pointer-events:none; }
    .assistant-widget[hidden] { display:none; }
    .assistant-toggle { pointer-events:auto; display:inline-flex; align-items:center; justify-content:center; gap:8px; min-height:44px; padding:0 16px; border-radius:999px; box-shadow:0 12px 28px rgba(16,24,40,.22); }
    .assistant-toggle-icon { font-size:17px; line-height:1; }
    .assistant-panel { pointer-events:auto; width:min(390px, calc(100vw - 28px)); max-height:min(620px, calc(100vh - 92px)); display:grid; grid-template-rows:auto minmax(180px, 1fr) auto; border:1px solid var(--line); border-radius:8px; background:#fff; box-shadow:0 18px 42px rgba(16,24,40,.2); overflow:hidden; }
    .assistant-panel.has-table { width:min(720px, calc(100vw - 28px)); }
    .assistant-panel[hidden] { display:none; }
    .assistant-head { display:flex; align-items:center; justify-content:space-between; gap:10px; padding:12px 14px; border-bottom:1px solid var(--line); }
    .assistant-head strong { font-size:14px; }
    .assistant-close { width:32px; min-height:32px; padding:0; border-color:var(--line); background:#fff; color:var(--muted); font-size:18px; line-height:1; }
    .assistant-messages { display:grid; align-content:start; gap:10px; min-height:220px; overflow:auto; padding:14px; background:#fbfcfe; }
    .assistant-empty { align-self:center; color:var(--muted); font-size:13px; line-height:1.4; text-align:center; }
    .assistant-message { max-width:86%; border:1px solid var(--line); border-radius:8px; padding:9px 10px; font-size:13px; line-height:1.4; overflow-wrap:anywhere; }
    .assistant-message.user { justify-self:end; border-color:#b9d5eb; background:#eef7ff; }
    .assistant-message.assistant { justify-self:start; max-width:96%; background:#fff; }
    .assistant-message.error { justify-self:start; border-color:#f0b8b2; background:#fff5f5; color:var(--bad); }
    .assistant-message-content { display:grid; gap:8px; }
    .assistant-message-content p { margin:0; white-space:pre-wrap; }
    .assistant-table-wrap { max-width:100%; overflow:auto; border:1px solid var(--line); border-radius:6px; }
    .assistant-message table { width:max-content; min-width:100%; border-collapse:collapse; font-size:12px; background:#fff; }
    .assistant-message th, .assistant-message td { padding:6px 8px; border-bottom:1px solid var(--line); border-right:1px solid var(--line); text-align:left; vertical-align:top; white-space:nowrap; }
    .assistant-message th:last-child, .assistant-message td:last-child { border-right:0; }
    .assistant-message tr:last-child td { border-bottom:0; }
    .assistant-message th { background:#f5f7fb; color:var(--text); font-weight:600; }
    .assistant-message tbody tr:nth-child(even) { background:#f8fafc; }
    .assistant-loading { display:flex; align-items:center; gap:8px; min-height:22px; color:var(--muted); font-size:12px; }
    .assistant-loading[hidden], .assistant-error[hidden] { display:none; }
    .assistant-loading .spinner { width:16px; height:16px; flex:0 0 auto; }
    .assistant-error { color:var(--bad); font-size:12px; line-height:1.35; }
    .assistant-form { display:grid; gap:8px; padding:12px; border-top:1px solid var(--line); background:#fff; }
    .assistant-input-row { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:8px; align-items:end; }
    .assistant-input-row textarea { min-height:44px; max-height:120px; resize:vertical; }
    .assistant-input-row button { min-width:76px; }
    .ok { color:var(--ok); }
    .bad { color:var(--bad); }
    .muted { color:var(--muted); }
    @media (max-width: 860px) { main, .history-layout, .dashboard-layout { grid-template-columns:minmax(0,1fr); padding:12px; } .dashboard-head { flex-direction:column; } .dashboard-actions { align-items:stretch; width:100%; } .assistant-widget { right:14px; bottom:14px; } .assistant-toggle { min-height:42px; padding:0 14px; } }
  </style>
</head>
<body>
  <header>
    <h1>Atividade 2 Judge Console</h1>
    <div id="config-status" class="status">Carregando configuracao local...</div>
  </header>
  <nav class="tabs" aria-label="Navegacao principal">
    <button class="tab-button active" type="button" data-tab="dashboard-panel">Dashboard</button>
    <button class="tab-button" type="button" data-tab="execution-panel">Execucao</button>
    <button class="tab-button" type="button" data-tab="history-panel">Execucoes anteriores</button>
      <button class="tab-button" type="button" data-tab="prompt-panel">Prompt Juizes</button>
      <button class="tab-button" type="button" data-tab="meta-panel">Meta-Avaliacao</button>
      <button class="tab-button" type="button" data-tab="rag-panel">RAG</button>
  </nav>
  <main id="dashboard-panel" class="dashboard-layout tab-panel">
    <aside class="dashboard-filters">
      <h2>Filtros globais</h2>
      <label>Dataset
        <select id="dashboard_dataset"><option value="J1">J1</option><option value="J2">J2</option><option value="all">Todos</option></select>
      </label>
      <label>Modelo candidato
        <select id="dashboard_candidate_model" multiple></select>
      </label>
      <label>Modelo juiz
        <select id="dashboard_judge_model" multiple></select>
      </label>
      <label>Status
        <select id="dashboard_status"><option value="all">todos</option><option value="sucesso">sucesso</option><option value="erro">erro</option></select>
      </label>
      <label>Agrupamento
        <select id="dashboard_group_by"><option value="modelo">por modelo</option><option value="juiz">por juiz</option><option value="dataset">por dataset</option><option value="disciplina">por disciplina</option><option value="dificuldade">por dificuldade</option></select>
      </label>
      <div class="dashboard-filter-actions">
        <button id="dashboard-refresh" type="button">Atualizar</button>
        <button id="dashboard-clear" class="secondary" type="button">Limpar</button>
      </div>
      <p class="dashboard-note">J1 e o dataset padrao. Spearman principal em J1 so aparece quando houver nota de referencia ordinal persistida; juiz x arbitro e exibido separadamente como consistencia.</p>
    </aside>
    <section>
      <div class="dashboard-head">
        <div>
          <h2>Resultados e Auditoria da Avaliacao</h2>
          <p>Visao consolidada das avaliacoes LLM-as-a-Judge, correlacao, distribuicao de notas e analise de erros.</p>
        </div>
        <div class="dashboard-actions">
          <button id="database-actions-toggle" class="database-actions-toggle" type="button" aria-haspopup="menu" aria-expanded="false" title="Acoes do banco">
            <span class="database-actions-toggle-icon" aria-hidden="true">&#9881;</span>
            <span>Acoes do Banco</span>
            <span class="database-actions-toggle-caret" aria-hidden="true">▼</span>
          </button>
          <div id="database-actions-menu" class="database-actions-menu" role="menu" hidden>
            <button id="database-clean" class="danger" type="button" role="menuitem">Clean DB (Initial State)</button>
            <button id="database-restore" type="button" role="menuitem">Restaurar Backup</button>
            <button id="database-dump" type="button" role="menuitem">Exportar Dump do Banco</button>
          </div>
          <input id="database-restore-file" type="file" accept=".sql,application/sql,text/plain" hidden>
          <span id="database-dump-status" class="status"></span>
        </div>
      </div>
      <div class="carousel-card" aria-label="Resumo do dashboard">
        <div class="carousel-head">
          <div id="dashboard-model-carousel-dots" class="carousel-tabs" role="tablist" aria-label="Paginas do dashboard">
            <button class="carousel-tab active" type="button" data-carousel-index="0" role="tab" aria-selected="true">Indicadores gerais</button>
            <button class="carousel-tab" type="button" data-carousel-index="1" role="tab" aria-selected="false">Distribuicao das notas por modelo</button>
            <button class="carousel-tab" type="button" data-carousel-index="2" role="tab" aria-selected="false">Especialidades juridicas</button>
            <button class="carousel-tab" type="button" data-carousel-index="3" role="tab" aria-selected="false">Erros criticos</button>
            <button class="carousel-tab" type="button" data-carousel-index="4" role="tab" aria-selected="false">Concordancia entre Juizes</button>
            <button class="carousel-tab" type="button" data-carousel-index="5" role="tab" aria-selected="false">Logs operacionais</button>
          </div>
          <div class="carousel-controls" aria-label="Navegacao do carousel">
            <button id="dashboard-model-carousel-prev" class="carousel-button" type="button" aria-label="Pagina anterior">&lsaquo;</button>
            <button id="dashboard-model-carousel-next" class="carousel-button" type="button" aria-label="Proxima pagina">&rsaquo;</button>
          </div>
        </div>
        <div class="carousel-viewport">
          <div id="dashboard-model-distribution-carousel" class="carousel-track">
            <div class="dashboard-carousel-slide">
              <h3>Indicadores gerais</h3>
              <div id="dashboard-cards" class="metric-grid"></div>
              <div class="chart-grid">
                <div class="chart">
                  <h3>Ranking geral dos modelos candidatos</h3>
                  <div id="dashboard-candidate-ranking"></div>
                </div>
                <div class="chart">
                  <h3>Distribuicao de notas 1-5</h3>
                  <div id="dashboard-score-distribution"></div>
                </div>
                <div class="chart">
                  <h3>Media por juiz</h3>
                  <div id="dashboard-judge-average"></div>
                </div>
                <div class="chart">
                  <h3>Divergencias para auditoria</h3>
                  <div id="dashboard-divergences"></div>
                </div>
                <div class="chart">
                  <h3>Casos criticos</h3>
                  <div id="dashboard-critical-chart"></div>
                </div>
              </div>
              <p id="dashboard-methodology" class="dashboard-note"></p>
              <h3 style="margin-top:18px">Casos criticos e divergencias</h3>
              <div class="table-wrap dashboard-table">
                <table aria-label="Casos criticos do dashboard">
                  <thead>
                    <tr>
                      <th>motivo</th>
                      <th>dataset</th>
                      <th>id_resposta</th>
                      <th>id_pergunta</th>
                      <th>modelo_candidato</th>
                      <th>juiz</th>
                      <th>papel</th>
                      <th>nota</th>
                      <th>status</th>
                      <th>Auditar</th>
                    </tr>
                  </thead>
                  <tbody id="dashboard-cases-body">
                    <tr><td colspan="10" class="muted">Carregando dashboard.</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div class="dashboard-carousel-slide">
              <h3>Distribuicao das notas por modelo</h3>
              <div id="dashboard-model-distribution-chart" class="model-distribution-list"></div>
            </div>
            <div class="dashboard-carousel-slide">
              <h3>Desempenho por especialidade juridica</h3>
              <div id="dashboard-legal-specialty-performance" class="heatmap-wrap"></div>
            </div>
            <div class="dashboard-carousel-slide">
              <h3>Analise de erros criticos</h3>
              <div class="chart">
                <h3>Categorias de erro</h3>
                <div id="dashboard-critical-error-chart"></div>
              </div>
              <div class="table-wrap critical-error-table">
                <table aria-label="Analise de erros criticos">
                  <thead>
                    <tr>
                      <th>Pergunta</th>
                      <th>Modelo candidato</th>
                      <th>Juiz</th>
                      <th>Nota</th>
                      <th>Tipo de erro</th>
                      <th>Justificativa curta</th>
                      <th>Auditar</th>
                    </tr>
                  </thead>
                  <tbody id="dashboard-critical-error-body">
                    <tr><td colspan="7" class="muted">Carregando analise.</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div class="dashboard-carousel-slide">
              <h3>Concordancia entre Juizes</h3>
              <div class="judge-agreement-layout">
                <div id="dashboard-judge-agreement-cards" class="metric-grid">
                  <div class="metric-card"><span class="metric-value">-</span><span class="metric-label">Carregando</span></div>
                </div>
                <div>
                  <h3>Media da nota por juiz e modelo candidato</h3>
                  <div id="dashboard-judge-candidate-heatmap" class="heatmap-wrap dashboard-compact-viz"></div>
                </div>
                <div>
                  <h3>Diferenca maxima entre juizes por modelo</h3>
                  <p class="judge-agreement-copy">Cada linha resume, para um modelo candidato, o quanto os juizes discordaram nas mesmas respostas. A escala vai de 0 a 4 pontos: 0 significa notas iguais; valores a partir de 2 indicam casos que devem ir para auditoria.</p>
                  <div id="dashboard-judge-disagreement-boxplot" class="boxplot-wrap"></div>
                </div>
              </div>
              <div class="table-wrap judge-agreement-table">
                <table aria-label="Concordancia entre juizes com arbitragem">
                  <thead>
                    <tr>
                      <th>ID da resposta</th>
                      <th>ID da pergunta</th>
                      <th>Modelo candidato</th>
                      <th>Nota Juiz 1</th>
                      <th>Nota Juiz 2</th>
                      <th>Delta</th>
                      <th>Nota arbitro</th>
                      <th>Motivo de acionamento</th>
                    </tr>
                  </thead>
                  <tbody id="dashboard-judge-agreement-body">
                    <tr><td colspan="8" class="muted">Carregando concordancia.</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div class="dashboard-carousel-slide">
              <h3>Logs operacionais</h3>
              <p class="dashboard-note">Enriquecimento read-only dos logs confirmados. Nao recalcula metricas oficiais do banco.</p>
              <div id="dashboard-operational-section" class="operational-section" hidden>
                <div id="dashboard-operational-cards" class="metric-grid"></div>
                <div class="operational-blocks">
                  <div class="chart">
                    <h3>Tempo medio por juiz/arbitro</h3>
                    <div class="table-wrap dashboard-table">
                      <table aria-label="Tempo medio operacional por juiz e arbitro">
                        <thead>
                          <tr>
                            <th>papel</th>
                            <th>modelo</th>
                            <th>tempo medio</th>
                            <th>eventos</th>
                          </tr>
                        </thead>
                        <tbody id="dashboard-operational-latency-body">
                          <tr><td colspan="4" class="muted">Metadados operacionais indisponiveis.</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <div class="chart">
                    <h3>Falhas por categoria</h3>
                    <div class="table-wrap dashboard-table operational-category-table">
                      <table aria-label="Falhas operacionais por categoria">
                        <thead>
                          <tr>
                            <th>categoria</th>
                            <th>incidentes</th>
                          </tr>
                        </thead>
                        <tbody id="dashboard-operational-categories-body">
                          <tr><td colspan="2" class="muted">Metadados operacionais indisponiveis.</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
                <h3>Logs com sucesso parcial</h3>
                <p class="dashboard-note">Conta runs que tiveram ao menos uma falha operacional e tambem ao menos um evento parseado com sucesso no mesmo log.</p>
                <div class="table-wrap dashboard-table">
                  <table aria-label="Detalhe de logs com sucesso parcial">
                    <thead>
                      <tr>
                        <th>run_id</th>
                        <th>log</th>
                        <th>eventos</th>
                        <th>sucessos</th>
                        <th>falhas</th>
                        <th>retries</th>
                      </tr>
                    </thead>
                    <tbody id="dashboard-operational-partial-body">
                      <tr><td colspan="6" class="muted">Sem logs com sucesso parcial.</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
              <p id="dashboard-operational-empty" class="muted">Metadados operacionais indisponiveis.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <main id="execution-panel" class="tab-panel" hidden>
    <aside>
      <h2>Presets</h2>
      <div id="presets" class="presets"></div>
      <h2>Configuracao</h2>
      <label>Fonte das respostas
        <select id="judge_input_source">
          <option value="av2">AV2 baseline</option>
          <option value="av3_j1_com_rag">AV3 J1 Com_RAG</option>
        </select>
      </label>
      <label>Modo
        <select id="panel_mode"><option>single</option><option>primary_only</option><option>2plus1</option></select>
      </label>
      <div class="row">
        <label>Dataset
          <select id="dataset"><option>J2</option><option>J1</option></select>
        </label>
        <label>Batch size
          <input id="batch_size" type="number" min="1" value="10">
        </label>
      </div>
      <label>Estrategia
        <select id="judge_execution_strategy"><option>sequential</option><option>parallel</option><option>adaptive</option></select>
      </label>
      <div class="hint">Sequential e melhor para endpoint local ou fragil. Parallel usa concorrencia fixa por resposta. Adaptive ajusta concorrencia por endpoint/modelo.</div>
      <div id="judge_block" class="judge-block" data-judge-block="judge">
      <label>Juiz 1 - modelo
        <select id="judge_model"></select>
      </label>
      <label>Endpoint do juiz 1
        <select id="endpoint_source_judge"><option value="env">Usar .env</option><option value="custom">Config propria</option></select>
      </label>
      <div id="endpoint_fields_judge" class="endpoint-fields" hidden>
      <label>Juiz 1 - URL
        <input id="remote_judge_base_url" autocomplete="off" placeholder="usa REMOTE_JUDGE_BASE_URL se vazio">
      </label>
      <label>Juiz 1 - token/key
        <span class="secret-row">
          <input id="remote_judge_api_key" type="password" autocomplete="off" placeholder="usa REMOTE_JUDGE_API_KEY se vazio">
          <button class="icon-button" type="button" data-toggle-secret="remote_judge_api_key" aria-label="Exibir token/key do juiz 1" aria-pressed="false">◉</button>
        </span>
      </label>
      </div>
      </div>
      <div id="secondary_block" class="judge-block" data-judge-block="secondary">
      <label>Juiz 2 - modelo
        <select id="secondary_judge_model"></select>
      </label>
      <label>Endpoint do juiz 2
        <select id="endpoint_source_secondary"><option value="env">Usar .env</option><option value="judge">Copiar do juiz 1</option><option value="custom">Config propria</option></select>
      </label>
      <div id="endpoint_fields_secondary" class="endpoint-fields" hidden>
      <label>Juiz 2 - URL
        <input id="remote_secondary_judge_base_url" autocomplete="off" placeholder="usa endpoint global se vazio">
      </label>
      <label>Juiz 2 - token/key
        <span class="secret-row">
          <input id="remote_secondary_judge_api_key" type="password" autocomplete="off" placeholder="usa token global se vazio">
          <button class="icon-button" type="button" data-toggle-secret="remote_secondary_judge_api_key" aria-label="Exibir token/key do juiz 2" aria-pressed="false">◉</button>
        </span>
      </label>
      </div>
      </div>
      <div id="arbiter_block" class="judge-block" data-judge-block="arbiter">
      <label>Arbitro - modelo
        <select id="arbiter_judge_model"></select>
      </label>
      <label>Endpoint do arbitro
        <select id="endpoint_source_arbiter"><option value="env">Usar .env</option><option value="judge">Copiar do juiz 1</option><option value="secondary">Copiar do juiz 2</option><option value="custom">Config propria</option></select>
      </label>
      <div id="endpoint_fields_arbiter" class="endpoint-fields" hidden>
      <label>Arbitro - URL
        <input id="remote_arbiter_judge_base_url" autocomplete="off" placeholder="usa endpoint global se vazio">
      </label>
      <label>Arbitro - token/key
        <span class="secret-row">
          <input id="remote_arbiter_judge_api_key" type="password" autocomplete="off" placeholder="usa token global se vazio">
          <button class="icon-button" type="button" data-toggle-secret="remote_arbiter_judge_api_key" aria-label="Exibir token/key do arbitro" aria-pressed="false">◉</button>
        </span>
      </label>
      </div>
      </div>
      <label class="inline"><input id="always_run_arbiter" type="checkbox"> Rodar arbitro sempre <span class="warn">aumenta custo e chamadas remotas</span></label>
      <details>
        <summary>Campos avancados</summary>
        <label class="inline"><input id="judge_save_raw_response" type="checkbox"> Salvar resposta bruta do juiz</label>
        <div class="row">
          <label>Timeout (s)
            <input id="remote_judge_timeout_seconds" type="number" min="1">
          </label>
          <label>Arbitration min delta
            <input id="judge_arbitration_min_delta" type="number" min="0">
          </label>
        </div>
        <div class="row">
          <label>Temperature
            <input id="remote_judge_temperature" type="number" min="0" step="0.1">
          </label>
          <label>Max tokens
            <input id="remote_judge_max_tokens" type="number" min="1">
          </label>
        </div>
        <div class="row">
          <label>Top P
            <input id="remote_judge_top_p" type="number" min="0" step="0.1">
          </label>
          <label>OpenAI compatible
            <select id="remote_judge_openai_compatible"><option value="true">true</option><option value="false">false</option></select>
          </label>
        </div>
      </details>
      <div class="actions">
        <button class="secondary" id="dry-run" disabled>Validar configuracao</button>
        <button class="secondary" id="stop-run" type="button" disabled>Parar</button>
        <button id="run" disabled>Executar</button>
      </div>
    </aside>
    <section>
      <h2>Execucao</h2>
      <progress id="batch-progress" max="100" value="0"></progress>
      <div id="progress-label" class="status">0% - aguardando execucao</div>
      <table>
        <tbody>
          <tr><th>Status</th><td><span id="run-status-icon" class="status-icon">-</span><span id="run-status">idle</span></td></tr>
          <tr><th>Audit log</th><td id="audit-log" class="muted">-</td></tr>
          <tr><th>Missing</th><td id="eligible-missing">-</td></tr>
          <tr><th>Failed</th><td id="eligible-failed">-</td></tr>
          <tr><th>Ja avaliadas com sucesso</th><td id="eligible-successful">-</td></tr>
          <tr><th>Serao processadas neste batch</th><td id="eligible-will-process">-</td></tr>
          <tr><th>Selecionadas</th><td id="selected">-</td></tr>
          <tr><th>Executadas</th><td id="executed">-</td></tr>
          <tr><th>Puladas</th><td id="skipped">-</td></tr>
          <tr><th>Arbitragens</th><td id="arbiters">-</td></tr>
        </tbody>
      </table>
      <h2 style="margin-top:18px">Comando equivalente</h2>
      <pre id="command-preview"></pre>
      <h2>Resumo / erro</h2>
      <pre id="output"></pre>
      <div id="post-run-panel" class="post-run-panel" hidden>
        <h2>Batch finalizado</h2>
        <div id="post-run-cards" class="metric-grid"></div>
        <div class="chart-grid">
          <div class="chart">
            <h3>Distribuicao de notas 1-5</h3>
            <div id="score-distribution-chart"></div>
          </div>
          <div class="chart">
            <h3>Falhas por juiz</h3>
            <div id="judge-failures-chart"></div>
          </div>
          <div class="chart">
            <h3>Arbitragens</h3>
            <div id="arbitration-chart"></div>
          </div>
          <div class="chart">
            <h3>Media por modelo candidato</h3>
            <div id="candidate-average-chart"></div>
          </div>
          <div class="chart">
            <h3>Media por juiz</h3>
            <div id="judge-average-chart"></div>
          </div>
        </div>
      </div>
      <h2 style="margin-top:18px">Tabela dinamica de execucao</h2>
      <div class="table-wrap">
        <table aria-label="Tabela dinamica de execucao">
          <thead>
            <tr>
              <th>status</th>
              <th>dataset</th>
              <th>id_pergunta</th>
              <th>modelo_candidato</th>
              <th>juiz</th>
              <th>papel</th>
              <th>nota</th>
              <th>delta</th>
              <th>arbitro acionado?</th>
              <th>motivo_acionamento</th>
              <th>latencia</th>
              <th>erro</th>
              <th>ver detalhes</th>
            </tr>
          </thead>
          <tbody id="execution-table-body">
            <tr><td colspan="13" class="muted">Aguardando execucao.</td></tr>
          </tbody>
        </table>
      </div>
    </section>
  </main>
  <div id="history-panel" class="history-layout tab-panel" hidden>
    <section>
      <div class="history-actions">
        <h2>Execucoes anteriores</h2>
        <div>
          <a class="button-link" href="/api/run-history/export.csv" download="run-history.csv">CSV</a>
          <a class="button-link" href="/api/run-history/export.json" download="run-history.json">JSON</a>
        </div>
      </div>
      <div class="table-wrap">
        <table aria-label="Tabela de execucoes anteriores">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Data/hora</th>
              <th>Modo</th>
              <th>Dataset</th>
              <th>Batch size</th>
              <th>Sucessos</th>
              <th>Falhas</th>
              <th>Duracao</th>
              <th>Log</th>
              <th>Exportar</th>
            </tr>
          </thead>
          <tbody id="history-table-body">
            <tr><td colspan="10" class="muted">Carregando historico.</td></tr>
          </tbody>
        </table>
      </div>
    </section>
    <aside>
      <h2>Log</h2>
      <table>
        <tbody>
          <tr><th>Run ID</th><td id="history-log-run-id" class="muted">Selecione uma execucao.</td></tr>
          <tr><th>Arquivo</th><td id="history-log-path" class="muted">-</td></tr>
        </tbody>
      </table>
      <pre id="history-log-content" class="history-log">Selecione uma execucao.</pre>
    </aside>
  </div>
    <div id="prompt-panel" class="prompt-layout tab-panel" hidden>
      <aside>
        <h2>Configuracao do prompt</h2>
        <label>Dataset
          <select id="prompt_dataset"></select>
        </label>
        <label>Prompt
          <textarea id="prompt_body" placeholder="[PERSONA]\n\n[CONTEXTO]\n\n[RUBRICA]\n\n[SAIDA]"></textarea>
        </label>
        <label>Persona
          <textarea id="prompt_persona" placeholder="Descreva a persona do juiz"></textarea>
        </label>
        <label>Contexto
          <textarea id="prompt_context" placeholder="Pergunta: {pergunta_oab}\nGabarito: {resposta_ouro}\nResposta: {resposta_modelo_edge}"></textarea>
        </label>
        <label>Rubrica
          <textarea id="prompt_rubric" placeholder="Defina a rubrica de avaliacao"></textarea>
        </label>
        <label>Saida
          <textarea id="prompt_output" placeholder="json com o seguinte formato"></textarea>
        </label>
        <label>Alterado por
          <input id="prompt_changed_by" autocomplete="off" placeholder="Nome do responsavel pela alteracao">
        </label>
        <div class="actions">
          <button class="secondary" id="prompt_reload" type="button">Recarregar</button>
          <button id="prompt_save" type="button">Salvar</button>
        </div>
        <div id="prompt_status" class="status prompt-status">Selecione um dataset para carregar.</div>
      </aside>
      <section>
        <div class="prompt-preview">
          <div class="prompt-preview-card">
            <h3>Preview com questao de exemplo</h3>
            <div class="muted" id="prompt_preview_meta">Nenhum preview carregado.</div>
            <pre id="prompt_preview_content">Selecione um dataset para montar o prompt.</pre>
          </div>
        </div>
        <div style="height:16px"></div>
        <h2>Versoes do prompt</h2>
        <div class="table-wrap prompt-log-table">
          <table aria-label="Versoes de prompt de juizes">
            <thead>
              <tr>
                <th>versao</th>
                <th>status</th>
                <th>quando</th>
                <th>quem</th>
                <th>prompt</th>
                <th>persona</th>
                <th>contexto</th>
                <th>rubrica</th>
                <th>saida</th>
              </tr>
            </thead>
            <tbody id="prompt_logs_body">
              <tr><td colspan="9" class="muted">Nenhuma configuracao carregada.</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
    <div id="meta-panel" class="tab-panel" hidden>
      <div class="meta-mode-switch" role="tablist" aria-label="Modos da meta-avaliacao">
        <button class="meta-mode-button active" id="meta_mode_new" type="button" data-meta-mode="new" role="tab" aria-selected="true">Nova auditoria</button>
        <button class="meta-mode-button" id="meta_mode_history" type="button" data-meta-mode="history" role="tab" aria-selected="false">Auditorias realizadas</button>
      </div>
      <div id="meta_new_panel" class="prompt-layout meta-mode-panel">
        <aside>
        <h2>Meta-Avaliacao</h2>
        <label class="meta-evaluation-field">Avaliacao J1
          <span class="meta-evaluation-input-wrap">
            <input id="meta_evaluation_select" autocomplete="off" placeholder="Digite o id ou filtre as avaliacoes" aria-controls="meta_evaluation_options">
            <button id="meta_evaluation_clear" class="meta-clear-button" type="button" title="Limpar avaliacao" aria-label="Limpar avaliacao">x</button>
          </span>
          <div id="meta_evaluation_options" class="meta-options-list" role="listbox" hidden></div>
        </label>
        <label>Avaliador
          <input id="meta_evaluator_name" autocomplete="off" placeholder="Nome de quem esta auditando">
        </label>
        <input id="meta_editing_id" type="hidden">
        <label>Nota da Meta-Avaliacao (1 a 5)
          <select id="meta_score">
            <option value="1">1 - Totalmente equivocada</option>
            <option value="2">2 - Excessivamente subestimada/superestimada</option>
            <option value="3">3 - Muito subestimada/superestimada</option>
            <option value="4">4 - Pouco subestimada/superestimada</option>
            <option value="5">5 - Justo</option>
          </select>
        </label>
        <label>Justificativa
          <textarea id="meta_rationale" placeholder="Explique se o juiz foi justo, aderente ao gabarito e consistente na nota atribuida."></textarea>
        </label>
        <div class="actions">
          <button class="secondary" id="meta_reload" type="button">Recarregar</button>
          <button class="secondary" id="meta_cancel_edit" type="button" style="display:none;">Cancelar edicao</button>
          <button id="meta_save" type="button">Salvar</button>
        </div>
        <div id="meta_status" class="status prompt-status">Selecione uma avaliacao J1 para iniciar a meta-avaliacao.</div>
      </aside>
      <section>
        <h2>Meta-avaliacoes registradas</h2>
        <div class="table-wrap prompt-log-table">
          <table aria-label="Historico de meta-avaliacoes">
            <thead>
              <tr>
                <th>quando</th>
                <th>avaliador</th>
                <th>nota</th>
                <th>justificativa</th>
                <th>acoes</th>
              </tr>
            </thead>
            <tbody id="meta_records_body">
              <tr><td colspan="5" class="muted">Nenhuma meta-avaliacao carregada.</td></tr>
            </tbody>
          </table>
        </div>
        <div style="height:16px"></div>
        <div class="prompt-preview">
          <div class="prompt-preview-card">
            <h3>Avaliacao selecionada</h3>
            <div class="muted" id="meta_subject_meta">Nenhuma avaliacao carregada.</div>
            <table aria-label="Resumo da avaliacao selecionada">
              <tbody>
                <tr><th>Modelo candidato</th><td id="meta_subject_candidate_model" class="muted">-</td></tr>
                <tr><th>Modelo juiz</th><td id="meta_subject_judge_model" class="muted">-</td></tr>
                <tr><th>Nota do juiz</th><td id="meta_subject_judge_score" class="muted">-</td></tr>
                <tr><th>Prompt</th><td id="meta_subject_prompt_version" class="muted">-</td></tr>
                <tr><th>Avaliada em</th><td id="meta_subject_evaluated_at" class="muted">-</td></tr>
              </tbody>
            </table>
          </div>
          <div id="meta_operational_card" class="prompt-preview-card operational-section" hidden>
            <h3>Metadados operacionais dos logs</h3>
            <p class="dashboard-note">Enriquecimento read-only; a avaliacao final continua vindo do PostgreSQL.</p>
            <table aria-label="Metadados operacionais da avaliacao selecionada">
              <tbody>
                <tr><th>Run ID / log</th><td id="meta_operational_run" class="muted">-</td></tr>
                <tr><th>Latencia</th><td id="meta_operational_latency" class="muted">-</td></tr>
              </tbody>
            </table>
          </div>
          <div class="prompt-preview-card">
            <h3>Questao</h3>
            <pre id="meta_subject_question">Selecione uma avaliacao para ver o enunciado.</pre>
          </div>
          <div class="prompt-preview-card">
            <h3>Gabarito</h3>
            <pre id="meta_subject_reference">Selecione uma avaliacao para ver o gabarito.</pre>
          </div>
          <div class="prompt-preview-card">
            <h3>Resposta do candidato</h3>
            <pre id="meta_subject_candidate_answer">Selecione uma avaliacao para ver a resposta candidata.</pre>
          </div>
          <div class="prompt-preview-card">
            <h3>Chain of Thoughts</h3>
            <pre id="meta_subject_chain_of_thought">Selecione uma avaliacao para ver o Chain of Thoughts.</pre>
          </div>
        </div>
        </section>
      </div>
      <div id="meta_history_panel" class="prompt-layout meta-mode-panel" hidden>
        <aside>
          <h2>Auditorias realizadas</h2>
          <label>Avaliador
            <select id="meta_history_evaluator"></select>
          </label>
          <div id="meta_history_count" class="status">0 auditorias encontradas</div>
          <div class="actions">
            <button class="secondary" id="meta_history_reload" type="button">Recarregar</button>
          </div>
        </aside>
        <section>
          <h2>Historico de auditorias</h2>
          <div class="table-wrap meta-history-table">
            <table aria-label="Auditorias realizadas">
              <thead>
                <tr>
                  <th><button type="button" data-meta-history-sort="created_at">Quando</button></th>
                  <th><button type="button" data-meta-history-sort="evaluator_name">Avaliador</button></th>
                  <th><button type="button" data-meta-history-sort="score">Nota meta</button></th>
                  <th><button type="button" data-meta-history-sort="judge_model">Modelo juiz</button></th>
                  <th><button type="button" data-meta-history-sort="judge_score">Nota juiz</button></th>
                  <th><button type="button" data-meta-history-sort="candidate_model">Modelo candidato</button></th>
                  <th><button type="button" data-meta-history-sort="question_id">Questao</button></th>
                </tr>
              </thead>
              <tbody id="meta_history_body">
                <tr><td colspan="7" class="muted">Nenhuma auditoria carregada.</td></tr>
              </tbody>
            </table>
          </div>
          <div class="prompt-preview">
            <div class="prompt-preview-card">
              <h3>Auditoria selecionada</h3>
              <div class="muted" id="meta_history_detail_meta">Selecione uma auditoria para ver os detalhes.</div>
              <table aria-label="Resumo da auditoria selecionada">
                <tbody>
                  <tr><th>Avaliador</th><td id="meta_history_detail_evaluator" class="muted">-</td></tr>
                  <tr><th>Nota meta</th><td id="meta_history_detail_score" class="muted">-</td></tr>
                  <tr><th>Justificativa</th><td id="meta_history_detail_rationale" class="muted">-</td></tr>
                  <tr><th>Modelo candidato</th><td id="meta_history_detail_candidate_model" class="muted">-</td></tr>
                  <tr><th>Modelo juiz</th><td id="meta_history_detail_judge_model" class="muted">-</td></tr>
                  <tr><th>Nota juiz</th><td id="meta_history_detail_judge_score" class="muted">-</td></tr>
                </tbody>
              </table>
              <div class="meta-detail-actions">
                <button class="secondary" id="meta_history_previous" type="button">Anterior</button>
                <button class="secondary" id="meta_history_next" type="button">Proxima</button>
              </div>
            </div>
            <div class="prompt-preview-card">
              <h3>Questao</h3>
              <pre id="meta_history_detail_question">Selecione uma auditoria para ver o enunciado.</pre>
            </div>
            <div class="prompt-preview-card">
              <h3>Gabarito</h3>
              <pre id="meta_history_detail_reference">Selecione uma auditoria para ver o gabarito.</pre>
            </div>
            <div class="prompt-preview-card">
              <h3>Resposta do candidato</h3>
              <pre id="meta_history_detail_candidate_answer">Selecione uma auditoria para ver a resposta candidata.</pre>
            </div>
            <div class="prompt-preview-card">
              <h3>Chain of Thoughts</h3>
              <pre id="meta_history_detail_chain_of_thought">Selecione uma auditoria para ver o Chain of Thoughts.</pre>
            </div>
          </div>
        </section>
      </div>
    </div>
    <div id="rag-panel" class="tab-panel" hidden>
      <main class="prompt-layout" style="display:block; width:min(100%, 1440px);">
        <section>
          <div class="rag-subtabs" aria-label="Submenus de RAG">
            <button class="rag-subtab-button active" type="button" data-rag-subtab="rag-curation-subpanel">Curadoria</button>
            <button class="rag-subtab-button" type="button" data-rag-subtab="rag-embedding-subpanel">Modelo Embedding</button>
            <button class="rag-subtab-button" type="button" data-rag-subtab="rag-vector-subpanel">Base Vetorial</button>
            <button class="rag-subtab-button" type="button" data-rag-subtab="rag-query-subpanel">Consulta</button>
          </div>

          <div id="rag-curation-subpanel" class="rag-subpanel">
            <div class="prompt-layout" style="padding:0; width:100%;">
              <aside>
                <h2>Curadoria RAG</h2>
                <label>Dataset
                  <select id="rag_curation_dataset"></select>
                </label>
                <label>Importado por
                  <input id="rag_curation_imported_by" autocomplete="off" placeholder="Nome de quem esta importando o JSON">
                </label>
                <div class="actions">
                  <button class="secondary" id="rag_curation_reload" type="button">Recarregar</button>
                  <button class="secondary" id="rag_curation_pick_file" type="button">Selecionar JSON</button>
                  <button id="rag_curation_import" type="button">Importar JSON</button>
                </div>
                <input id="rag_curation_file" type="file" accept=".json,application/json,text/plain" hidden>
                <div id="rag_curation_status" class="status prompt-status">Selecione um dataset para visualizar a curadoria importada.</div>
              </aside>
              <section>
                <div class="prompt-preview">
                  <div class="prompt-preview-card">
                    <h3>Versao ativa</h3>
                    <div class="muted" id="rag_curation_active_meta">Nenhuma versao ativa carregada.</div>
                    <table aria-label="Resumo da versao ativa da curadoria">
                      <tbody>
                        <tr><th>Arquivo</th><td id="rag_curation_active_filename" class="muted">-</td></tr>
                        <tr><th>Importado por</th><td id="rag_curation_active_imported_by" class="muted">-</td></tr>
                        <tr><th>Importado em</th><td id="rag_curation_active_imported_at" class="muted">-</td></tr>
                        <tr><th>Questoes curadas</th><td id="rag_curation_active_coverage" class="muted">-</td></tr>
                        <tr><th>Itens importados</th><td id="rag_curation_active_item_count" class="muted">-</td></tr>
                        <tr><th>Artigos curados</th><td id="rag_curation_active_article_count" class="muted">-</td></tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                <div style="height:16px"></div>
                <h2>Importacoes recentes</h2>
                <div class="table-wrap prompt-log-table rag-curation-table">
                  <table aria-label="Historico de importacoes da curadoria RAG">
                    <thead>
                      <tr>
                        <th>run</th>
                        <th>quando</th>
                        <th>quem</th>
                        <th>arquivo</th>
                        <th>itens</th>
                        <th>artigos</th>
                        <th>status</th>
                        <th>acao</th>
                      </tr>
                    </thead>
                    <tbody id="rag_curation_runs_body">
                      <tr><td colspan="8" class="muted">Nenhuma importacao carregada.</td></tr>
                    </tbody>
                  </table>
                </div>
                <div style="height:16px"></div>
                <h2>Questoes curadas</h2>
                <div class="table-wrap prompt-log-table rag-curation-table">
                  <table aria-label="Lista de questoes curadas para RAG">
                    <thead>
                      <tr>
                        <th>detalhe</th>
                        <th>seq</th>
                        <th>id_pergunta</th>
                        <th>tipo</th>
                        <th>disciplina</th>
                        <th>tema</th>
                        <th>curador</th>
                        <th>norma</th>
                        <th>artigos</th>
                      </tr>
                    </thead>
                    <tbody id="rag_curation_items_body">
                      <tr><td colspan="9" class="muted">Nenhuma curadoria carregada.</td></tr>
                    </tbody>
                  </table>
                </div>
                <div style="height:16px"></div>
                <div class="prompt-preview rag-curation-detail-grid">
                  <div class="prompt-preview-card">
                    <h3>Curadoria selecionada</h3>
                    <div class="muted" id="rag_curation_detail_meta">Selecione uma questao curada para ver os detalhes.</div>
                    <table aria-label="Resumo da curadoria selecionada">
                      <tbody>
                        <tr><th>Tipo</th><td id="rag_curation_detail_type" class="muted">-</td></tr>
                        <tr><th>Disciplina</th><td id="rag_curation_detail_discipline" class="muted">-</td></tr>
                        <tr><th>Assunto</th><td id="rag_curation_detail_subject" class="muted">-</td></tr>
                        <tr><th>Tema</th><td id="rag_curation_detail_theme" class="muted">-</td></tr>
                        <tr><th>Curador</th><td id="rag_curation_detail_curator" class="muted">-</td></tr>
                        <tr><th>Classificada em</th><td id="rag_curation_detail_classified_at" class="muted">-</td></tr>
                        <tr><th>Lei / norma</th><td id="rag_curation_detail_lei_norma" class="muted">-</td></tr>
                        <tr><th>Fonte</th><td id="rag_curation_detail_url" class="muted">-</td></tr>
                        <tr><th>URN</th><td id="rag_curation_detail_urn" class="muted">-</td></tr>
                      </tbody>
                    </table>
                  </div>
                  <div class="prompt-preview-card">
                    <h3>Artigos curados</h3>
                    <div class="table-wrap prompt-log-table rag-curation-articles-table">
                      <table aria-label="Artigos curados da questao selecionada">
                        <thead>
                          <tr>
                            <th>ordem</th>
                            <th>artigo</th>
                            <th>topico</th>
                            <th>relevancia</th>
                            <th>tipo</th>
                          </tr>
                        </thead>
                        <tbody id="rag_curation_articles_body">
                          <tr><td colspan="5" class="muted">Nenhum artigo carregado.</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <div class="prompt-preview-card">
                    <h3>Questao</h3>
                    <pre id="rag_curation_detail_question">Selecione uma questao curada para ver o enunciado.</pre>
                  </div>
                  <div class="prompt-preview-card">
                    <h3>Gabarito</h3>
                    <pre id="rag_curation_detail_answer_key">Selecione uma questao curada para ver o gabarito.</pre>
                  </div>
                  <div class="prompt-preview-card">
                    <h3>Metadados</h3>
                    <pre id="rag_curation_detail_metadata">Selecione uma questao curada para ver os metadados.</pre>
                  </div>
                  <div class="prompt-preview-card">
                    <h3>Payload bruto</h3>
                    <pre id="rag_curation_detail_raw_payload">Selecione uma questao curada para ver o payload original.</pre>
                  </div>
                </div>
              </section>
            </div>
          </div>

          <div id="rag-vector-subpanel" class="rag-subpanel" hidden>
            <div class="prompt-layout" style="padding:0; width:100%;">
              <aside>
                <h2>Base Vetorial</h2>
                <label>Dataset
                  <select id="rag_vector_dataset"></select>
                </label>
                <label>Questao inicial
                  <input id="rag_vector_question_start" type="number" min="1" step="1" placeholder="Todas">
                </label>
                <label>Questao final
                  <input id="rag_vector_question_end" type="number" min="1" step="1" placeholder="Todas">
                </label>
                <div class="actions">
                  <button class="secondary" id="rag_vector_reload" type="button">Recarregar</button>
                  <button id="rag_vector_generate" type="button">Gerar embeddings</button>
                </div>
                <div id="rag_vector_status" class="status prompt-status">Selecione um dataset para visualizar a base vetorial.</div>
                <div class="prompt-preview">
                  <div class="prompt-preview-card">
                    <h3>Cobertura por dataset</h3>
                    <div class="table-wrap prompt-log-table">
                      <table aria-label="Cobertura de curadoria por dataset">
                        <thead>
                          <tr>
                            <th>dataset</th>
                            <th>base</th>
                            <th>curadas</th>
                            <th>run ativa</th>
                            <th>retrieval</th>
                            <th>docs</th>
                            <th>chunks</th>
                            <th>vetorial</th>
                            <th>ultima carga</th>
                          </tr>
                        </thead>
                        <tbody id="rag_curation_datasets_body">
                          <tr><td colspan="9" class="muted">Nenhuma cobertura carregada.</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </aside>
              <section>
                <div class="prompt-preview">
                  <div class="prompt-preview-card">
                    <h3>Base vetorial ativa</h3>
                    <div class="muted" id="rag_vector_active_meta">Nenhuma base vetorial ativa para esse dataset.</div>
                    <table aria-label="Resumo da base vetorial ativa">
                      <tbody>
                        <tr><th>Run</th><td id="rag_vector_active_run" class="muted">-</td></tr>
                        <tr><th>Status</th><td id="rag_vector_active_status" class="muted">-</td></tr>
                        <tr><th>Estrategia</th><td id="rag_vector_active_strategy" class="muted">-</td></tr>
                        <tr><th>Top-K</th><td id="rag_vector_active_top_k" class="muted">-</td></tr>
                        <tr><th>Documentos</th><td id="rag_vector_active_document_count" class="muted">-</td></tr>
                        <tr><th>Chunks</th><td id="rag_vector_active_chunk_count" class="muted">-</td></tr>
                        <tr><th>Embeddings</th><td id="rag_vector_active_embedding_count" class="muted">-</td></tr>
                        <tr><th>Modelo embedding</th><td id="rag_vector_active_embedding_model" class="muted">-</td></tr>
                        <tr><th>Criada em</th><td id="rag_vector_active_created_at" class="muted">-</td></tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                <div style="height:16px"></div>
                <div class="prompt-preview">
                  <div class="prompt-preview-card">
                    <h3>Runs disponiveis</h3>
                    <div class="table-wrap prompt-log-table">
                      <table aria-label="Runs da base vetorial">
                        <thead>
                          <tr>
                            <th>run</th>
                            <th>estrategia</th>
                            <th>docs</th>
                            <th>chunks</th>
                            <th>embeddings</th>
                            <th>criada em</th>
                            <th>acoes</th>
                          </tr>
                        </thead>
                        <tbody id="rag_vector_runs_body">
                          <tr><td colspan="7" class="muted">Nenhuma run vetorial carregada.</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </section>
            </div>
            <div class="rag-operation-panel">
              <h3>Logs da operacao</h3>
              <div id="rag_vector_progress" class="rag-progress-log" aria-live="polite">
                <div class="rag-progress-placeholder">Nenhuma geracao de embeddings iniciada nesta sessao.</div>
              </div>
            </div>
          </div>

          <div id="rag-query-subpanel" class="rag-subpanel" hidden>
            <div class="prompt-layout" style="padding:0; width:100%;">
              <aside>
                <h2>Consulta</h2>
                <label>Dataset
                  <select id="rag_query_dataset"></select>
                </label>
                <label>Top-K
                  <input id="rag_query_top_k" type="number" min="1" max="50" step="1" value="5">
                </label>
                <label>Texto da consulta
                  <textarea id="rag_query_text" placeholder="Ex.: improbidade administrativa, ato de agente publico e independencia das instancias"></textarea>
                </label>
                <div class="actions">
                  <button class="secondary" id="rag_query_reload" type="button">Recarregar</button>
                  <button id="rag_query_search" type="button">Buscar</button>
                </div>
                <div id="rag_query_status" class="status prompt-status">Selecione um dataset para inspecionar a base vetorial.</div>
              </aside>
              <section>
                <div class="prompt-preview">
                  <div class="prompt-preview-card">
                    <h3>Resumo da consulta</h3>
                    <div class="muted" id="rag_query_meta">Nenhuma consulta executada.</div>
                    <table aria-label="Resumo da consulta vetorial">
                      <tbody>
                        <tr><th>Dataset</th><td id="rag_query_summary_dataset" class="muted">-</td></tr>
                        <tr><th>Top-K</th><td id="rag_query_summary_top_k" class="muted">-</td></tr>
                        <tr><th>Latencia</th><td id="rag_query_summary_latency" class="muted">-</td></tr>
                        <tr><th>Dimensoes</th><td id="rag_query_summary_dimensions" class="muted">-</td></tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                <div style="height:16px"></div>
                <div class="prompt-preview">
                  <div class="prompt-preview-card">
                    <h3>Resultados da busca</h3>
                    <div class="table-wrap prompt-log-table">
                      <table aria-label="Resultados da busca vetorial">
                        <thead>
                          <tr>
                            <th>detalhe</th>
                            <th>rank</th>
                            <th>score</th>
                            <th>lei</th>
                            <th>norma</th>
                            <th>texto</th>
                          </tr>
                        </thead>
                        <tbody id="rag_query_results_body">
                          <tr><td colspan="6" class="muted">Nenhuma busca executada.</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
                <div style="height:16px"></div>
                <div class="prompt-preview">
                  <div class="prompt-preview-card">
                    <h3>Resultado selecionado</h3>
                    <div class="muted" id="rag_query_selected_meta">Nenhum resultado selecionado.</div>
                    <table aria-label="Resumo do resultado selecionado">
                      <tbody>
                        <tr><th>Rank</th><td id="rag_query_selected_rank" class="muted">-</td></tr>
                        <tr><th>Score</th><td id="rag_query_selected_score" class="muted">-</td></tr>
                        <tr><th>Chunk</th><td id="rag_query_selected_chunk_id" class="muted">-</td></tr>
                        <tr><th>Documento</th><td id="rag_query_selected_document_key" class="muted">-</td></tr>
                        <tr><th>Lei</th><td id="rag_query_selected_lei" class="muted">-</td></tr>
                        <tr><th>Norma</th><td id="rag_query_selected_norma" class="muted">-</td></tr>
                        <tr><th>Origem</th><td id="rag_query_selected_source_kind" class="muted">-</td></tr>
                        <tr><th>URL</th><td id="rag_query_selected_url" class="muted">-</td></tr>
                        <tr><th>URN</th><td id="rag_query_selected_urn" class="muted">-</td></tr>
                      </tbody>
                    </table>
                    <div style="height:12px"></div>
                    <h4 style="margin:0 0 8px;">Chunk recuperado</h4>
                    <pre id="rag_query_selected_chunk_text">Selecione um resultado para ver o texto completo.</pre>
                  </div>
                </div>
                <div style="height:16px"></div>
                <div class="prompt-preview">
                  <div class="prompt-preview-card">
                    <h3>Amostra de documentos</h3>
                    <div class="table-wrap prompt-log-table">
                      <table aria-label="Amostra de documentos da base vetorial">
                        <thead>
                          <tr>
                            <th>documento</th>
                            <th>lei</th>
                            <th>norma</th>
                          </tr>
                        </thead>
                        <tbody id="rag_query_documents_body">
                          <tr><td colspan="3" class="muted">Nenhum documento carregado.</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
                <div style="height:16px"></div>
                <div class="prompt-preview">
                  <div class="prompt-preview-card">
                    <h3>Amostra de chunks</h3>
                    <div class="table-wrap prompt-log-table">
                      <table aria-label="Amostra de chunks da base vetorial">
                        <thead>
                          <tr>
                            <th>chunk</th>
                            <th>lei</th>
                            <th>norma</th>
                            <th>texto</th>
                          </tr>
                        </thead>
                        <tbody id="rag_query_chunks_body">
                          <tr><td colspan="4" class="muted">Nenhum chunk carregado.</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          </div>

          <div id="rag-embedding-subpanel" class="rag-subpanel" hidden>
            <div class="prompt-layout" style="padding:0; width:100%;">
              <aside>
                <h2>Modelo Embedding</h2>
                <label>Dataset
                  <select id="rag_embedding_dataset"></select>
                </label>
                <label>Alterado por
                  <input id="rag_embedding_updated_by" autocomplete="off" placeholder="Nome de quem alterou a configuracao">
                </label>
                <div class="actions">
                  <button class="secondary" id="rag_embedding_reload" type="button">Recarregar</button>
                  <button class="secondary" id="rag_embedding_test" type="button">Testar API</button>
                  <button id="rag_embedding_save" type="button">Salvar configuracao</button>
                </div>
                <div id="rag_embedding_status" class="status prompt-status">Selecione um dataset para configurar o modelo de embedding.</div>
              </aside>
              <section>
                <div class="prompt-preview">
                  <div class="prompt-preview-card">
                    <h3>Configuracao ativa</h3>
                    <div class="muted" id="rag_embedding_meta">Nenhuma configuracao carregada.</div>
                    <table aria-label="Resumo da configuracao de embedding">
                      <tbody>
                        <tr><th>Provider</th><td id="rag_embedding_provider_summary" class="muted">-</td></tr>
                        <tr><th>Modelo</th><td id="rag_embedding_model_summary" class="muted">-</td></tr>
                        <tr><th>Dimensoes</th><td id="rag_embedding_dimensions_summary" class="muted">-</td></tr>
                        <tr><th>API base</th><td id="rag_embedding_api_base_summary" class="muted">-</td></tr>
                        <tr><th>Atualizada por</th><td id="rag_embedding_updated_by_summary" class="muted">-</td></tr>
                        <tr><th>Atualizada em</th><td id="rag_embedding_updated_at_summary" class="muted">-</td></tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                <div style="height:16px"></div>
                <div class="prompt-preview">
                  <div class="prompt-preview-card">
                    <h3>Smoke test da API</h3>
                    <div class="muted" id="rag_embedding_test_meta">Nenhum teste executado.</div>
                    <table aria-label="Resumo do smoke test de embedding">
                      <tbody>
                        <tr><th>Endpoint</th><td id="rag_embedding_test_endpoint" class="muted">-</td></tr>
                        <tr><th>Latencia</th><td id="rag_embedding_test_latency" class="muted">-</td></tr>
                        <tr><th>Dimensoes retornadas</th><td id="rag_embedding_test_dimensions" class="muted">-</td></tr>
                        <tr><th>Testado em</th><td id="rag_embedding_test_time" class="muted">-</td></tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                <div style="height:16px"></div>
                <div class="prompt-preview">
                  <div class="prompt-preview-card">
                    <h3>Formulario</h3>
                    <label>Provider
                      <input id="rag_embedding_provider" autocomplete="off" placeholder="openai">
                    </label>
                    <label>Modelo
                      <input id="rag_embedding_model_name" autocomplete="off" placeholder="text-embedding-3-small">
                    </label>
                    <label>Dimensoes
                      <input id="rag_embedding_dimensions" type="number" min="1" step="1" placeholder="1536">
                    </label>
                    <label>API base URL
                      <input id="rag_embedding_api_base_url" autocomplete="off" placeholder="https://api.openai.com/v1">
                    </label>
                    <label>Notas
                      <textarea id="rag_embedding_notes" placeholder="Observacoes sobre a escolha do modelo de embedding"></textarea>
                    </label>
                  </div>
                </div>
              </section>
            </div>
          </div>
        </section>
      </main>
    </div>
  <div class="assistant-widget" aria-live="polite"__ASSISTANT_WIDGET_HIDDEN__>
    <section id="assistant-chat-panel" class="assistant-panel" aria-label="Chat do assistente" hidden>
      <div class="assistant-head">
        <strong>Assistente AV2</strong>
        <button id="assistant-chat-close" class="assistant-close" type="button" aria-label="Fechar assistente">×</button>
      </div>
      <div id="assistant-chat-messages" class="assistant-messages" aria-live="polite">
        <div class="assistant-empty">Envie uma pergunta sobre os resultados, execucoes ou configuracoes disponiveis.</div>
      </div>
      <form id="assistant-chat-form" class="assistant-form">
        <div id="assistant-chat-loading" class="assistant-loading" hidden>
          <span class="spinner" aria-hidden="true"></span>
          <span>Assistente esta respondendo...</span>
        </div>
        <div id="assistant-chat-error" class="assistant-error" hidden></div>
        <div class="assistant-input-row">
          <textarea id="assistant-chat-input" rows="2" placeholder="Pergunte ao assistente" aria-label="Mensagem para o assistente"></textarea>
          <button id="assistant-chat-send" type="submit">Enviar</button>
        </div>
      </form>
    </section>
    <button id="assistant-chat-toggle" class="assistant-toggle" type="button" aria-controls="assistant-chat-panel" aria-expanded="false">
      <span class="assistant-toggle-icon" aria-hidden="true">?</span>
      <span>Assistente</span>
    </button>
  </div>
  <dialog id="details-dialog">
    <div class="dialog-head">
      <strong id="details-title">Detalhes da avaliacao</strong>
      <button class="secondary" id="details-close" type="button">Fechar</button>
    </div>
    <div class="dialog-body">
      <h3>Prompt</h3>
      <pre id="details-prompt"></pre>
      <h3>Resposta do juiz</h3>
      <pre id="details-response"></pre>
      <h3>Justificativa</h3>
      <pre id="details-rationale"></pre>
    </div>
  </dialog>
  <dialog id="audit-log-dialog">
    <div class="dialog-head">
      <strong>Live audit log</strong>
      <button class="secondary" id="audit-log-close" type="button">Fechar</button>
    </div>
    <div class="dialog-body">
      <pre id="audit-log-content" class="audit-log-content">Audit log nao selecionado.</pre>
    </div>
  </dialog>
  <dialog id="database-clean-dialog" class="confirm-dialog">
    <div class="dialog-head">
      <strong>Clean DB (Initial State)</strong>
    </div>
    <div class="dialog-body">
      <p>Resetar o banco para o estado inicial?</p>
      <p>Ao continuar, o sistema gera um backup automaticamente antes de limpar os schemas public e av3, restaurar backup_atividade_2_reset.sql e validar o restore.</p>
    </div>
    <div class="confirm-actions">
      <button id="database-clean-cancel" class="secondary" type="button">Cancelar</button>
      <button id="database-clean-confirm" class="danger-button" type="button">Continuar</button>
    </div>
  </dialog>
  <dialog id="database-dump-dialog" class="confirm-dialog">
    <div class="dialog-head">
      <strong>Backup salvo</strong>
      <button id="database-dump-dialog-close" class="secondary" type="button">Fechar</button>
    </div>
    <div class="dialog-body">
      <p>O dump do banco foi criado com sucesso.</p>
      <p><a id="database-dump-download" class="button-link" href="#" download hidden>Baixar arquivo</a></p>
      <table>
        <tbody>
          <tr><th>Arquivo</th><td id="database-dump-filename"></td></tr>
          <tr><th>Caminho</th><td id="database-dump-path"></td></tr>
          <tr><th>Tamanho</th><td id="database-dump-size"></td></tr>
        </tbody>
      </table>
    </div>
  </dialog>
  <dialog id="meta-delete-dialog" class="confirm-dialog">
    <div class="dialog-head">
      <strong>Excluir meta-avaliacao</strong>
    </div>
    <div class="dialog-body">
      <p>Excluir esta meta-avaliacao?</p>
    </div>
    <div class="confirm-actions">
      <button id="meta-delete-cancel" class="secondary" type="button">Cancelar</button>
      <button id="meta-delete-confirm" class="danger-button" type="button">Excluir</button>
    </div>
  </dialog>
  <script>
    let csrfToken = "";
    let pollTimer = null;
    let historyLoaded = false;
    let runHistoryLoading = false;
    let runHistoryPromise = null;
    let promptOptionsLoaded = false;
    let metaOptionsLoaded = false;
    let ragCurationOptionsLoaded = false;
    let ragEmbeddingConfigLoaded = false;
    let metaHistoryLoaded = false;
    let metaHistoryRecords = [];
    let metaHistoryFilteredRecords = [];
    let ragCurationDatasetOptions = [];
    let selectedMetaHistoryId = null;
    let metaHistorySort = {key: "created_at", direction: "desc"};
    let metaEvaluationOptions = [];
    let selectedMetaEvaluationId = "";
    let selectedHistoryRunId = null;
    let dashboardLoaded = false;
    let operationalLogSummary = null;
    let operationalLogSummaryLoading = false;
    let operationalLogSummaryPromise = null;
    let currentMetaSubject = null;
    let currentAuditLogUrl = null;
    let activeRunId = null;
    let judgeModelOptions = [];
    let assistantMessages = [];
    let assistantLoading = false;
    const executionTableState = new Map();

    function value(id) { return document.getElementById(id).value; }
    function setText(id, text) { document.getElementById(id).textContent = text ?? "-"; }

    function truncateText(text, maxLength = 180) {
      const normalized = String(text ?? "");
      if (normalized.length <= maxLength) return normalized;
      return `${normalized.slice(0, maxLength - 3)}...`;
    }
    function display(value) { return value === null || value === undefined || value === "" ? "-" : value; }
    function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }
    function friendlyErrorMessage(message) {
      const raw = String(message || "");
      const normalized = raw.toLowerCase();
      const mappings = [
        ["REMOTE_JUDGE_BASE_URL is required", "Configure a URL do endpoint do juiz"],
        ["REMOTE_JUDGE_API_KEY is required", "Configure a key local; não commitar"],
        ["invalid JSON", "O modelo não respeitou o contrato de saída"],
        ["gated model", "Modelo sem acesso neste provedor"],
        ["don't have access", "Modelo sem acesso neste provedor"],
        ["model does not exist", "Modelo inválido ou sem acesso nesse provedor"],
        ["HTTP 401", "key inválida ou sem permissão"],
        ["HTTP 403", "acesso negado pelo provedor"],
        ["HTTP 404", "base URL/modelo incorreto"],
        ["unexpected end of json input", "resposta interrompida pelo servidor; confira o log da operacao e tente novamente"],
        ["broken pipe", "conexao interrompida com o provedor de embeddings; tente novamente"],
        ["embedding request failed after", "conexao interrompida com o provedor de embeddings; confira o log do lote e tente novamente"]
      ];
      for (const [needle, friendly] of mappings) {
        if (normalized.includes(needle.toLowerCase())) return friendly;
      }
      if (normalized.includes("timeout")) return "aumentar timeout ou reduzir batch";
      return raw || "Erro desconhecido.";
    }

    function formatDateTime(value) {
      if (!value) return "-";
      return new Date(value).toLocaleString();
    }

    function selectedValues(id) {
      return Array.from(document.getElementById(id).selectedOptions).map((option) => option.value).filter(Boolean);
    }

    function syncRagDatasetSelects(dataset) {
      for (const id of ["rag_curation_dataset", "rag_vector_dataset", "rag_query_dataset", "rag_embedding_dataset"]) {
        const select = document.getElementById(id);
        if (select && Array.from(select.options).some((option) => option.value === dataset)) {
          select.value = dataset;
        }
      }
    }

    function dashboardQuery() {
      const params = new URLSearchParams();
      params.set("dataset", value("dashboard_dataset"));
      params.set("status", value("dashboard_status"));
      params.set("group_by", value("dashboard_group_by"));
      const candidates = selectedValues("dashboard_candidate_model");
      const judges = selectedValues("dashboard_judge_model");
      if (candidates.length) params.set("candidate_model", candidates.join(","));
      if (judges.length) params.set("judge_model", judges.join(","));
      return params.toString();
    }

    async function loadDashboard() {
      const body = document.getElementById("dashboard-cases-body");
      body.innerHTML = '<tr><td colspan="10" class="muted">Carregando dashboard.</td></tr>';
      document.getElementById("dashboard-judge-agreement-body").innerHTML = '<tr><td colspan="8" class="muted">Carregando concordancia.</td></tr>';
      try {
        const response = await fetch(`/api/dashboard?${dashboardQuery()}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Dashboard indisponivel.");
        dashboardLoaded = true;
        renderDashboard(data);
        renderDashboardOperationalLogState();
      } catch (error) {
        body.innerHTML = "";
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 10;
        cell.className = "muted";
        cell.textContent = friendlyErrorMessage(error.message);
        row.appendChild(cell);
        body.appendChild(row);
        renderJudgeAgreement({}, [], friendlyErrorMessage(error.message));
        renderDashboardOperationalLogState();
      }
    }

    function renderDashboardOperationalLogState() {
      if (operationalLogSummaryLoading && !operationalLogSummary) {
        renderDashboardOperationalLoading();
      }
      getOperationalLogSummary().then(renderDashboardOperationalSummary);
    }

    function getOperationalLogSummary() {
      if (!operationalLogSummaryPromise) {
        operationalLogSummaryPromise = loadOperationalLogSummary();
      }
      return operationalLogSummaryPromise;
    }

    function prefetchOperationalLogSummary() {
      getOperationalLogSummary();
    }

    async function refreshOperationalLogSummary() {
      operationalLogSummaryPromise = loadOperationalLogSummary();
      return operationalLogSummaryPromise;
    }

    async function loadOperationalLogSummary() {
      operationalLogSummaryLoading = true;
      renderOperationalLogLoading();
      try {
        const response = await fetch("/api/operational-log-summary", {cache: "no-store"});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Metadados operacionais indisponiveis.");
        operationalLogSummary = data;
        operationalLogSummaryLoading = false;
        refreshOperationalLogConsumers(data);
        return data;
      } catch (error) {
        operationalLogSummary = null;
        operationalLogSummaryLoading = false;
        refreshOperationalLogConsumers(null);
        return null;
      }
    }

    function refreshOperationalLogConsumers(summary) {
      if (dashboardLoaded) renderDashboardOperationalSummary(summary);
      if (currentMetaSubject) renderMetaOperationalMetadata(currentMetaSubject, summary);
      if (historyLoaded) getRunHistory().then(renderHistory);
    }

    function renderOperationalLogLoading() {
      if (dashboardLoaded) renderDashboardOperationalLoading();
      if (currentMetaSubject) renderMetaOperationalLoading(currentMetaSubject);
    }

    function renderDashboard(data) {
      populateSelect("dashboard_candidate_model", data.options?.candidate_models || [], selectedValues("dashboard_candidate_model"));
      populateSelect("dashboard_judge_model", data.options?.judge_models || [], selectedValues("dashboard_judge_model"));
      renderDashboardCards(data.cards || {});
      renderModelDistributionChart(data.charts?.score_distribution_by_model || []);
      renderJudgeCandidateHeatmap(data.charts?.judge_candidate_heatmap || {});
      renderJudgeDisagreementBoxplot(data.charts?.judge_disagreement_boxplot || {});
      renderLegalSpecialtyPerformance(data.charts?.legal_specialty_performance || {});
      renderCriticalErrorAnalysis(data.charts?.critical_error_categories || [], data.tables?.critical_error_analysis || []);
      renderJudgeAgreement(data.cards?.judge_agreement || {}, data.tables?.judge_agreement_arbitrations || []);
      renderBarChart("dashboard-candidate-ranking", data.charts?.candidate_ranking || [], {scaleMax: 5});
      renderBarChart("dashboard-score-distribution", data.charts?.score_distribution || [], {scaleMax: 1, showPercent: true, colorByLabel: true});
      renderBarChart("dashboard-judge-average", data.charts?.judge_average || [], {scaleMax: 5});
      renderBarChart("dashboard-divergences", data.charts?.divergences || [], {scaleMax: 1, tone: "bad"});
      renderBarChart("dashboard-critical-chart", data.charts?.critical_cases || [], {scaleMax: 1, tone: "bad"});
      setText("dashboard-methodology", `${data.methodology?.primary_spearman || ""} ${data.methodology?.judge_arbiter || ""}`.trim());
      renderDashboardCases([
        ...(data.tables?.critical_cases || []),
        ...(data.tables?.minor_disagreement_cases || []),
        ...(data.tables?.divergence_cases || []),
      ]);
    }

    function visibleModelSelectIds() {
      const mode = value("panel_mode");
      if (mode === "single") return ["judge_model"];
      if (mode === "primary_only") return ["judge_model", "secondary_judge_model"];
      return ["judge_model", "secondary_judge_model", "arbiter_judge_model"];
    }

    function renderJudgeModelSelects() {
      const visibleIds = new Set(visibleModelSelectIds());
      const chosenBySelect = {};
      for (const id of ["judge_model", "secondary_judge_model", "arbiter_judge_model"]) {
        chosenBySelect[id] = value(id);
      }
      for (const id of ["judge_model", "secondary_judge_model", "arbiter_judge_model"]) {
        const select = document.getElementById(id);
        const selected = chosenBySelect[id];
        select.textContent = "";
        for (const model of judgeModelOptions) {
          const selectedElsewhere = Object.entries(chosenBySelect).some(([otherId, otherValue]) => (
            otherId !== id && visibleIds.has(otherId) && otherValue === model
          ));
          if (selectedElsewhere) continue;
          const option = document.createElement("option");
          option.value = model;
          option.textContent = model;
          option.selected = model === selected;
          select.appendChild(option);
        }
        if (selected && Array.from(select.options).some((option) => option.value === selected)) {
          select.value = selected;
        } else if (select.options.length) {
          select.selectedIndex = 0;
        }
      }
    }

    function renderJudgeBlocks() {
      const mode = value("panel_mode");
      document.getElementById("judge_block").hidden = false;
      document.getElementById("secondary_block").hidden = mode === "single";
      document.getElementById("arbiter_block").hidden = mode !== "2plus1";
      renderJudgeModelSelects();
    }

    function renderJudgeInputSource() {
      const source = value("judge_input_source");
      const datasetSelect = document.getElementById("dataset");
      const isAv3ComRag = source === "av3_j1_com_rag";
      if (isAv3ComRag) {
        datasetSelect.value = "J1";
      }
      datasetSelect.disabled = isAv3ComRag;
    }

    function renderDashboardCards(cards) {
      const root = document.getElementById("dashboard-cards");
      root.textContent = "";
      const coverage = cards.coverage || {};
      const metrics = [
        ["Avaliacoes realizadas", cards.evaluations],
        ["Cobertura do dataset", `${display(coverage.evaluated)}/${display(coverage.expected)} (${displayPercent(coverage.percent)})`],
        ["Taxa de sucesso", displayPercent(cards.success_rate)],
        ["Nota media geral", formatAverage(cards.average_score)],
        ["Spearman juiz x referencia", formatSpearman(cards.spearman_reference)],
        ["Consistencia juiz x arbitro", formatSpearman(cards.judge_arbiter_consistency)],
        ["Falhas criticas detectadas", cards.critical_failures],
        ["Divergencias leves (delta=1)", cards.minor_disagreements],
        ["Divergencias para auditoria", cards.audit_divergences]
      ];
      for (const metric of metrics) {
        const card = document.createElement("div");
        card.className = "metric-card";
        const value = document.createElement("span");
        value.className = "metric-value";
        value.textContent = display(metric[1]);
        const label = document.createElement("span");
        label.className = "metric-label";
        label.textContent = metric[0];
        card.appendChild(value);
        card.appendChild(label);
        const source = metric[0].startsWith("Spearman") ? cards.spearman_reference : metric[0].startsWith("Consistencia") ? cards.judge_arbiter_consistency : null;
        if (source?.note) {
          const note = document.createElement("span");
          note.className = "metric-label";
          note.textContent = source.note;
          card.appendChild(note);
        }
        root.appendChild(card);
      }
    }

    function renderDashboardOperationalSummary(summary) {
      if (operationalLogSummaryLoading && !summary) {
        renderDashboardOperationalLoading();
        return;
      }
      const section = document.getElementById("dashboard-operational-section");
      const empty = document.getElementById("dashboard-operational-empty");
      const cardsRoot = document.getElementById("dashboard-operational-cards");
      const latencyBody = document.getElementById("dashboard-operational-latency-body");
      const categoriesBody = document.getElementById("dashboard-operational-categories-body");
      const partialBody = document.getElementById("dashboard-operational-partial-body");
      cardsRoot.textContent = "";
      latencyBody.textContent = "";
      categoriesBody.textContent = "";
      partialBody.textContent = "";
      if (!summary || !summary.available) {
        section.hidden = true;
        empty.hidden = false;
        return;
      }
      section.hidden = false;
      empty.hidden = true;
      const logs = safeOperationalLogs(summary);
      const events = logs.flatMap((log) => safeOperationalEvents(log));
      const latencies = events.map((event) => Number(event.latency_ms)).filter(Number.isFinite);
      const averageLatency = latencies.length ? Math.round(latencies.reduce((sum, value) => sum + value, 0) / latencies.length) : null;
      const failedEvents = events.filter((event) => isOperationalFailure(event));
      const incidentEvents = events.filter((event) => isOperationalIncident(event));
      const partialRuns = operationalPartialRuns(logs);
      const metrics = [
        ["Tempo medio de avaliacao", formatLatency(averageLatency)],
        ["Total de retries", summary.totals?.retries ?? 0],
        ["Falhas operacionais", summary.totals?.failures ?? failedEvents.length],
        ["Logs com sucesso parcial", partialRuns.length || "-"],
      ];
      for (const [labelText, metricValue] of metrics) {
        const card = document.createElement("div");
        card.className = "metric-card operational-card";
        const value = document.createElement("span");
        value.className = "metric-value";
        value.textContent = display(metricValue);
        const label = document.createElement("span");
        label.className = "metric-label";
        label.textContent = labelText;
        card.appendChild(value);
        card.appendChild(label);
        cardsRoot.appendChild(card);
      }

      renderOperationalLatencyRows(latencyBody, events);
      renderOperationalCategoryRows(categoriesBody, incidentEvents);
      renderOperationalPartialRows(partialBody, partialRuns);
    }

    function renderDashboardOperationalLoading() {
      const section = document.getElementById("dashboard-operational-section");
      const empty = document.getElementById("dashboard-operational-empty");
      const cardsRoot = document.getElementById("dashboard-operational-cards");
      const latencyBody = document.getElementById("dashboard-operational-latency-body");
      const categoriesBody = document.getElementById("dashboard-operational-categories-body");
      const partialBody = document.getElementById("dashboard-operational-partial-body");
      section.hidden = false;
      empty.hidden = true;
      cardsRoot.textContent = "";
      latencyBody.textContent = "";
      categoriesBody.textContent = "";
      partialBody.textContent = "";
      for (const labelText of ["Logs operacionais", "Retries", "Falhas", "Sucesso parcial"]) {
        const card = document.createElement("div");
        card.className = "metric-card operational-card";
        const value = document.createElement("span");
        value.className = "metric-value";
        value.textContent = "-";
        const label = document.createElement("span");
        label.className = "metric-label";
        label.textContent = `Carregando ${labelText.toLowerCase()}`;
        card.appendChild(value);
        card.appendChild(label);
        cardsRoot.appendChild(card);
      }
      appendTableMessage(latencyBody, 4, "Carregando metadados operacionais.");
      appendTableMessage(categoriesBody, 2, "Carregando falhas por categoria.");
      appendTableMessage(partialBody, 6, "Carregando logs com sucesso parcial.");
    }

    function safeOperationalLogs(summary) {
      return Array.isArray(summary?.logs) ? summary.logs : [];
    }

    function safeOperationalEvents(log) {
      return Array.isArray(log?.events) ? log.events : [];
    }

    function isOperationalFailure(event) {
      return ["adaptive_task_failed", "judge_call_failed"].includes(event?.event_type);
    }

    function isOperationalIncident(event) {
      return isOperationalFailure(event) || event?.event_type === "adaptive_task_requeued";
    }

    function renderOperationalLatencyRows(body, events) {
      const groups = new Map();
      for (const event of events) {
        const latency = Number(event.latency_ms);
        if (!Number.isFinite(latency)) continue;
        if (!event.role || !event.judge_model) continue;
        const role = normalizeRole(event.role);
        const model = event.judge_model;
        const key = `${role}\u001f${model}`;
        const current = groups.get(key) || {role, model, sum: 0, count: 0};
        current.sum += latency;
        current.count += 1;
        groups.set(key, current);
      }
      const rows = Array.from(groups.values()).sort((a, b) => a.role.localeCompare(b.role) || a.model.localeCompare(b.model));
      if (!rows.length) {
        appendTableMessage(body, 4, "Sem latencia operacional nos logs disponiveis.");
        return;
      }
      rows.forEach((item) => {
        const row = document.createElement("tr");
        appendCell(row, item.role);
        appendCell(row, item.model);
        appendCell(row, formatLatency(Math.round(item.sum / item.count)));
        appendCell(row, item.count);
        body.appendChild(row);
      });
    }

    function renderOperationalCategoryRows(body, events) {
      const categories = new Map();
      for (const event of events) {
        const category = operationalFailureCategory(event);
        const current = categories.get(category) || {label: category, incidents: 0};
        current.incidents += 1;
        categories.set(category, current);
      }
      const rows = Array.from(categories.values()).sort((a, b) => b.incidents - a.incidents || a.label.localeCompare(b.label));
      if (!rows.length) {
        appendTableMessage(body, 2, "Sem incidentes operacionais categorizaveis.");
        return;
      }
      rows.forEach((item) => {
        const row = document.createElement("tr");
        appendCell(row, item.label);
        appendCell(row, item.incidents);
        body.appendChild(row);
      });
    }

    function operationalFailureCategory(event) {
      const error = String(event.error || "").toLowerCase();
      const status = Number(event.status_code);
      if (status === 429 || error.includes("rate limit") || error.includes("too many") || error.includes("concurrency")) {
        return "Concorrencia/rate limit do provedor";
      }
      if (status === 404 || error.includes("model") && (error.includes("unavailable") || error.includes("not found") || error.includes("does not exist")) || error.includes("gated") || error.includes("no access")) {
        return "Modelo indisponivel ou sem acesso";
      }
      if (status === 401 || status === 403 || error.includes("unauthorized") || error.includes("forbidden")) {
        return "Autenticacao/permissao do provedor";
      }
      if (error.includes("timeout") || error.includes("timed out")) {
        return "Timeout";
      }
      if (status >= 500) {
        return "Erro 5xx do provedor";
      }
      if (status >= 400) {
        return "Erro HTTP do provedor";
      }
      return "Falha operacional nao classificada";
    }

    function operationalPartialRuns(logs) {
      return logs.map((log) => {
        const events = safeOperationalEvents(log);
        const failures = events.filter((event) => isOperationalFailure(event)).length || Number(log.failures) || 0;
        const successes = events.filter((event) => event.event_type === "evaluation_parsed").length;
        return {
          runId: log.run_id,
          logPath: log.log_path,
          events: Number(log.total_events) || events.length,
          successes,
          failures,
          retries: Number(log.total_retries) || events.reduce((sum, event) => sum + (Number(event.retry_count) || 0), 0),
        };
      }).filter((run) => run.failures > 0 && run.successes > 0);
    }

    function renderOperationalPartialRows(body, rows) {
      if (!rows.length) {
        appendTableMessage(body, 6, "Sem logs com sucesso parcial.");
        return;
      }
      rows.forEach((item) => {
        const row = document.createElement("tr");
        appendCell(row, display(item.runId));
        const logCell = document.createElement("td");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "detail-button";
        button.textContent = display(item.logPath);
        button.title = "Abrir em Execucoes anteriores";
        button.onclick = () => openHistoryLogFromMeta(item.runId, item.logPath);
        logCell.appendChild(button);
        row.appendChild(logCell);
        appendCell(row, item.events);
        appendCell(row, item.successes);
        appendCell(row, item.failures);
        appendCell(row, item.retries);
        body.appendChild(row);
      });
    }

    function appendTableMessage(body, colSpan, message) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = colSpan;
      cell.className = "muted";
      cell.textContent = message;
      row.appendChild(cell);
      body.appendChild(row);
    }

    function renderModelDistributionChart(rows) {
      const root = document.getElementById("dashboard-model-distribution-chart");
      root.textContent = "";
      if (!rows.length) {
        const empty = document.createElement("div");
        empty.className = "muted carousel-empty";
        empty.textContent = "Sem dados por modelo.";
        root.appendChild(empty);
        return;
      }
      rows.forEach((row) => {
        const card = document.createElement("div");
        card.className = "model-distribution-card";
        const head = document.createElement("div");
        head.className = "model-card-head";
        const title = document.createElement("div");
        title.className = "model-card-title";
        title.title = row.label;
        title.textContent = row.label;
        const meta = document.createElement("div");
        meta.className = "model-card-meta";
        meta.textContent = `media ${formatAverage(row.average)} - n=${display(row.total)}`;
        head.appendChild(title);
        head.appendChild(meta);
        card.appendChild(head);
        card.appendChild(stackedScoreBar(row));
        card.appendChild(scoreLegend(row));
        root.appendChild(card);
      });
    }

    function stackedScoreBar(row) {
      const total = Number(row.total) || 0;
      const bar = document.createElement("div");
      bar.className = "stacked-bar";
      for (const score of ["1", "2", "3", "4", "5"]) {
        const value = Number(row.scores?.[score]) || 0;
        const segment = document.createElement("span");
        segment.className = `stacked-segment score-${score}`;
        segment.title = `Nota ${score}: ${value}`;
        segment.style.width = `${total ? (value / total) * 100 : 0}%`;
        bar.appendChild(segment);
      }
      return bar;
    }

    function scoreLegend(row) {
      const legend = document.createElement("div");
      legend.className = "score-legend";
      for (const score of ["1", "2", "3", "4", "5"]) {
        const chip = document.createElement("span");
        chip.className = "score-chip";
        const label = document.createElement("span");
        const swatch = document.createElement("span");
        swatch.className = `score-swatch stacked-segment score-${score}`;
        label.appendChild(swatch);
        label.appendChild(document.createTextNode(` ${score}`));
        const value = document.createElement("strong");
        value.textContent = String(Number(row.scores?.[score]) || 0);
        chip.appendChild(label);
        chip.appendChild(value);
        legend.appendChild(chip);
      }
      return legend;
    }

    function renderRubricHeatmap(heatmap) {
      const root = document.getElementById("dashboard-rubric-heatmap");
      root.textContent = "";
      const columns = heatmap.columns || [];
      const rows = heatmap.rows || [];
      if (!columns.length || !rows.length) {
        const empty = document.createElement("div");
        empty.className = "muted carousel-empty";
        empty.textContent = "Sem scores dimensionais para o filtro atual.";
        root.appendChild(empty);
        return;
      }
      const grid = document.createElement("div");
      grid.className = "heatmap-grid";
      grid.style.gridTemplateColumns = `minmax(160px, 1.4fr) repeat(${columns.length}, minmax(112px, 1fr))`;
      grid.appendChild(heatmapCell("Modelo", "heatmap-head"));
      columns.forEach((column) => grid.appendChild(heatmapCell(column, "heatmap-head")));
      rows.forEach((row) => {
        grid.appendChild(heatmapCell(row.label, "heatmap-model"));
        (row.values || []).forEach((value) => {
          const cell = heatmapCell(value == null ? "-" : formatAverage(value), "heatmap-value");
          cell.style.background = heatmapColor(value);
          cell.title = `${row.label} - n=${display(row.count)}`;
          grid.appendChild(cell);
        });
      });
      root.appendChild(grid);
    }

    function renderJudgeCandidateHeatmap(heatmap) {
      const root = document.getElementById("dashboard-judge-candidate-heatmap");
      root.textContent = "";
      const columns = heatmap.columns || [];
      const rows = heatmap.rows || [];
      if (!columns.length || !rows.length) {
        const empty = document.createElement("div");
        empty.className = "muted carousel-empty";
        empty.textContent = "Sem notas por juiz e modelo candidato para o filtro atual.";
        root.appendChild(empty);
        return;
      }
      const grid = document.createElement("div");
      grid.className = "heatmap-grid";
      grid.style.gridTemplateColumns = `minmax(180px, 1.3fr) repeat(${columns.length}, minmax(112px, 1fr)) minmax(70px, .55fr)`;
      grid.appendChild(heatmapCell("Juiz", "heatmap-head"));
      columns.forEach((column) => grid.appendChild(heatmapCell(column, "heatmap-head")));
      grid.appendChild(heatmapCell("n", "heatmap-head"));
      rows.forEach((row) => {
        grid.appendChild(heatmapCell(row.label, "heatmap-model"));
        (row.values || []).forEach((value, index) => {
          const candidate = columns[index] || "modelo";
          const cell = heatmapCell(value == null ? "-" : formatAverage(value), "heatmap-value");
          cell.style.background = heatmapColor(value);
          cell.title = `${row.label} | ${candidate} | media ${display(value)} | n=${display(row.count)}`;
          grid.appendChild(cell);
        });
        grid.appendChild(heatmapCell(display(row.count), "heatmap-head"));
      });
      root.appendChild(grid);
    }

    function renderJudgeDisagreementBoxplot(boxplot) {
      const root = document.getElementById("dashboard-judge-disagreement-boxplot");
      root.textContent = "";
      const rows = boxplot.rows || [];
      if (!rows.length) {
        const empty = document.createElement("div");
        empty.className = "muted carousel-empty";
        empty.textContent = "Sem respostas com dois ou mais juizes para o filtro atual.";
        root.appendChild(empty);
        return;
      }
      const maxDelta = Math.max(4, ...rows.map((row) => Number(row.max) || 0));
      const threshold = Number(boxplot.audit_threshold ?? 2);
      const list = document.createElement("div");
      list.className = "boxplot-list";
      list.setAttribute("role", "img");
      list.setAttribute("aria-label", "Distribuicao da diferenca maxima entre juizes por modelo candidato");
      const scale = document.createElement("div");
      scale.className = "boxplot-scale";
      scale.appendChild(document.createElement("span"));
      const ticks = document.createElement("div");
      ticks.className = "boxplot-ticks";
      for (let delta = 0; delta <= maxDelta; delta += 1) {
        const tick = document.createElement("span");
        tick.className = "boxplot-tick";
        tick.style.left = `${boxplotPercent(delta, maxDelta)}%`;
        tick.textContent = String(delta);
        ticks.appendChild(tick);
      }
      scale.appendChild(ticks);
      const scaleMeta = document.createElement("span");
      scaleMeta.textContent = "escala 0-4";
      scale.appendChild(scaleMeta);
      list.appendChild(scale);

      rows.forEach((row) => {
        const item = document.createElement("div");
        item.className = "boxplot-row";
        const name = document.createElement("div");
        name.className = "boxplot-name";
        name.textContent = row.label;
        name.title = row.label;
        const track = document.createElement("div");
        track.className = "boxplot-track";
        if (Number(row.max) >= threshold) {
          appendBoxplotPart(track, "boxplot-threshold", threshold, threshold, maxDelta);
        }
        appendBoxplotPart(track, "boxplot-whisker", row.min, row.max, maxDelta);
        appendBoxplotPart(track, "boxplot-box", row.q1, row.q3, maxDelta);
        appendBoxplotPart(track, "boxplot-median", row.median, row.median, maxDelta);
        track.title = `min ${display(row.min)} | q1 ${display(row.q1)} | mediana ${display(row.median)} | q3 ${display(row.q3)} | max ${display(row.max)}`;
        const meta = document.createElement("div");
        meta.className = "boxplot-meta";
        meta.textContent = `${display(row.count)} resp. | ${display(row.audit_count)} aud.`;
        meta.title = `${display(row.count)} respostas comparadas; ${display(row.audit_count)} com diferenca >= ${display(threshold)}`;
        item.appendChild(name);
        item.appendChild(track);
        item.appendChild(meta);
        list.appendChild(item);
      });
      root.appendChild(list);
    }

    function appendBoxplotPart(track, className, start, end, maxDelta) {
      const part = document.createElement("span");
      part.className = className;
      const left = boxplotPercent(start, maxDelta);
      const right = boxplotPercent(end, maxDelta);
      part.style.left = `${Math.min(left, right)}%`;
      part.style.width = className === "boxplot-median" || className === "boxplot-threshold"
        ? "2px"
        : `${Math.max(1, Math.abs(right - left))}%`;
      track.appendChild(part);
    }

    function boxplotPercent(value, maxDelta) {
      const number = Math.max(0, Math.min(maxDelta, Number(value) || 0));
      return (number / maxDelta) * 100;
    }

    function renderLegalSpecialtyPerformance(heatmap) {
      const root = document.getElementById("dashboard-legal-specialty-performance");
      root.textContent = "";
      const columns = heatmap.columns || [];
      const rows = heatmap.rows || [];
      if (!columns.length || !rows.length) {
        const empty = document.createElement("div");
        empty.className = "muted carousel-empty";
        empty.textContent = "Sem especialidades juridicas para o filtro atual.";
        root.appendChild(empty);
        return;
      }
      const grid = document.createElement("div");
      grid.className = "heatmap-grid";
      grid.style.gridTemplateColumns = `minmax(180px, 1.3fr) repeat(${columns.length}, minmax(112px, 1fr)) minmax(70px, .55fr)`;
      grid.appendChild(heatmapCell("Especialidade", "heatmap-head"));
      columns.forEach((column) => grid.appendChild(heatmapCell(column, "heatmap-head")));
      grid.appendChild(heatmapCell("n", "heatmap-head"));
      rows.forEach((row) => {
        grid.appendChild(heatmapCell(row.label, "heatmap-model"));
        (row.values || []).forEach((value, index) => {
          const model = columns[index] || "modelo";
          const cell = heatmapCell(value == null ? "-" : formatAverage(value), "heatmap-value");
          cell.style.background = heatmapColor(value);
          cell.title = `${row.label} | ${model} | media ${display(value)} | n=${display(row.count)}`;
          grid.appendChild(cell);
        });
        grid.appendChild(heatmapCell(display(row.count), "heatmap-head"));
      });
      root.appendChild(grid);
    }

    function renderDifficultyPerformance(chart) {
      const root = document.getElementById("dashboard-difficulty-performance");
      root.textContent = "";
      const difficulties = chart.difficulties || [];
      const series = chart.series || [];
      const hasValues = series.some((row) => (row.values || []).some((value) => value != null));
      if (!difficulties.length || !series.length || !hasValues) {
        const empty = document.createElement("div");
        empty.className = "muted carousel-empty";
        empty.textContent = "Sem dificuldade registrada para o filtro atual.";
        root.appendChild(empty);
        return;
      }

      const width = 760;
      const height = 360;
      const margin = {top: 24, right: 34, bottom: 64, left: 56};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const xStep = difficulties.length > 1 ? plotWidth / (difficulties.length - 1) : 0;
      const colors = ["#1769aa", "#1d7f4e", "#d97706", "#7c3aed", "#b42318", "#0f766e", "#475569"];
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.setAttribute("class", "line-chart-svg");
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", "Desempenho por dificuldade");

      for (let score = 1; score <= 5; score += 1) {
        const y = margin.top + plotHeight - ((score - 1) / 4) * plotHeight;
        svg.appendChild(svgLine(margin.left, y, width - margin.right, y, "scatter-grid"));
        svg.appendChild(svgText(margin.left - 10, y + 4, String(score), "scatter-label", "end"));
      }
      svg.appendChild(svgLine(margin.left, margin.top, margin.left, height - margin.bottom, "scatter-axis"));
      svg.appendChild(svgLine(margin.left, height - margin.bottom, width - margin.right, height - margin.bottom, "scatter-axis"));
      difficulties.forEach((difficulty, index) => {
        const x = margin.left + index * xStep;
        svg.appendChild(svgText(x, height - margin.bottom + 24, difficulty, "scatter-label", "middle"));
      });
      svg.appendChild(svgText(width / 2, height - 16, chart.x_label || "dificuldade", "scatter-label", "middle"));
      const yLabel = svgText(18, margin.top + plotHeight / 2, chart.y_label || "média da nota", "scatter-label", "middle");
      yLabel.setAttribute("transform", `rotate(-90 18 ${margin.top + plotHeight / 2})`);
      svg.appendChild(yLabel);

      series.forEach((row, rowIndex) => {
        const color = colors[rowIndex % colors.length];
        const points = (row.values || []).map((value, index) => {
          if (value == null) return null;
          const number = Number(value);
          const x = margin.left + index * xStep;
          const y = margin.top + plotHeight - ((number - 1) / 4) * plotHeight;
          return {x, y, value: number, difficulty: difficulties[index]};
        });
        const pathPoints = points.filter(Boolean);
        if (pathPoints.length > 1) {
          const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
          path.setAttribute("d", pathPoints.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" "));
          path.setAttribute("class", "line-series");
          path.setAttribute("stroke", color);
          svg.appendChild(path);
        }
        pathPoints.forEach((point) => {
          const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          circle.setAttribute("cx", String(point.x));
          circle.setAttribute("cy", String(point.y));
          circle.setAttribute("r", "4.5");
          circle.setAttribute("class", "line-point");
          circle.setAttribute("fill", color);
          const title = document.createElementNS(svg.namespaceURI, "title");
          title.textContent = `${row.label} | ${point.difficulty} | media ${formatAverage(point.value)}`;
          circle.appendChild(title);
          svg.appendChild(circle);
        });
      });

      root.appendChild(svg);
      const legend = document.createElement("div");
      legend.className = "line-legend";
      series.forEach((row, rowIndex) => {
        const item = document.createElement("span");
        item.className = "line-legend-item";
        const swatch = document.createElement("span");
        swatch.className = "line-legend-swatch";
        swatch.style.background = colors[rowIndex % colors.length];
        item.appendChild(swatch);
        item.appendChild(document.createTextNode(row.label));
        legend.appendChild(item);
      });
      root.appendChild(legend);
    }

    function renderOrdinalConfusion(confusion) {
      const root = document.getElementById("dashboard-ordinal-confusion");
      root.textContent = "";
      const rows = confusion.rows || [];
      const columns = confusion.columns || [];
      const matrix = confusion.matrix || [];
      if (!rows.length || !columns.length || !Number(confusion.total)) {
        const empty = document.createElement("div");
        empty.className = "muted carousel-empty";
        empty.textContent = "Sem pares humano x juiz para o filtro atual.";
        root.appendChild(empty);
        return;
      }
      const heatmap = document.createElement("div");
      heatmap.className = "heatmap-wrap";
      const grid = document.createElement("div");
      grid.className = "heatmap-grid";
      grid.style.gridTemplateColumns = `minmax(118px, 1.1fr) repeat(${columns.length}, minmax(86px, 1fr))`;
      grid.appendChild(heatmapCell("Humano \\ Juiz", "heatmap-head"));
      columns.forEach((column) => grid.appendChild(heatmapCell(column, "heatmap-head")));
      const maxValue = Math.max(1, ...matrix.flat().map((value) => Number(value) || 0));
      rows.forEach((label, rowIndex) => {
        grid.appendChild(heatmapCell(label, "heatmap-model"));
        columns.forEach((column, columnIndex) => {
          const value = Number(matrix[rowIndex]?.[columnIndex]) || 0;
          const cell = heatmapCell(String(value), "heatmap-value");
          cell.style.background = confusionHeatmapColor(value, maxValue, rowIndex, columnIndex);
          cell.title = `${label}, ${column}: ${value}`;
          grid.appendChild(cell);
        });
      });
      heatmap.appendChild(grid);
      const summary = document.createElement("div");
      summary.className = "confusion-summary";
      (confusion.highlights || []).forEach((item) => summary.appendChild(confusionSummaryCard(item)));
      root.appendChild(heatmap);
      root.appendChild(summary);
    }

    function confusionHeatmapColor(value, maxValue, rowIndex, columnIndex) {
      if (!value) return "#f4f7fb";
      const ratio = Math.max(0.14, Math.min(1, value / maxValue));
      if (columnIndex - rowIndex >= 3) return `rgba(180, 35, 24, ${0.2 + ratio * 0.58})`;
      if (rowIndex - columnIndex >= 3) return `rgba(154, 91, 0, ${0.2 + ratio * 0.58})`;
      if (rowIndex === columnIndex) return `rgba(29, 127, 78, ${0.18 + ratio * 0.54})`;
      return `rgba(23, 105, 170, ${0.14 + ratio * 0.42})`;
    }

    function confusionSummaryCard(item) {
      const card = document.createElement("div");
      card.className = "confusion-card";
      const title = document.createElement("strong");
      title.textContent = `${item.count ?? 0} - ${item.interpretation || item.label}`;
      const detail = document.createElement("span");
      detail.textContent = `${item.label || "-"} (${displayPercent(item.share)})`;
      card.appendChild(title);
      card.appendChild(detail);
      return card;
    }

    function heatmapCell(text, className) {
      const cell = document.createElement("div");
      cell.className = `heatmap-cell ${className}`;
      cell.textContent = text;
      return cell;
    }

    function heatmapColor(value) {
      if (value == null) return "#eef2f7";
      const score = Math.max(1, Math.min(5, Number(value)));
      const ratio = (score - 1) / 4;
      const hue = 8 + ratio * 136;
      const lightness = 88 - ratio * 25;
      return `hsl(${hue}, 62%, ${lightness}%)`;
    }

    function renderReferenceScatter(series, spearman) {
      const root = document.getElementById("dashboard-reference-scatter");
      root.textContent = "";
      const points = series.points || [];
      if (!points.length) {
        const empty = document.createElement("div");
        empty.className = "muted carousel-empty";
        empty.textContent = "Sem pares referencia x juiz para o filtro atual.";
        root.appendChild(empty);
        return;
      }
      const width = 760;
      const height = 430;
      const margin = {top: 46, right: 24, bottom: 58, left: 68};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const minScore = 1;
      const maxScore = 5;
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.setAttribute("role", "img");
      svg.setAttribute("class", "scatter-svg");
      svg.setAttribute("aria-label", "Scatter plot de nota de referencia no eixo X e nota do juiz no eixo Y");

      const stat = document.createElementNS(svg.namespaceURI, "text");
      stat.setAttribute("x", String(margin.left));
      stat.setAttribute("y", "25");
      stat.setAttribute("class", "scatter-stat");
      stat.textContent = `rho Spearman = ${formatSpearmanValue(spearman.value)} | p-value = ${formatPValue(spearman.p_value)} | n = ${display(spearman.sample_size ?? points.length)}`;
      svg.appendChild(stat);

      for (let score = minScore; score <= maxScore; score += 1) {
        const x = scaleScore(score, minScore, maxScore, margin.left, plotWidth);
        const y = scaleScore(score, minScore, maxScore, margin.top + plotHeight, -plotHeight);
        svg.appendChild(svgLine(x, margin.top, x, margin.top + plotHeight, "scatter-grid"));
        svg.appendChild(svgLine(margin.left, y, margin.left + plotWidth, y, "scatter-grid"));
        svg.appendChild(svgText(x, margin.top + plotHeight + 24, String(score), "scatter-label", "middle"));
        svg.appendChild(svgText(margin.left - 18, y + 4, String(score), "scatter-label", "end"));
      }
      svg.appendChild(svgLine(margin.left, margin.top + plotHeight, margin.left + plotWidth, margin.top + plotHeight, "scatter-axis"));
      svg.appendChild(svgLine(margin.left, margin.top, margin.left, margin.top + plotHeight, "scatter-axis"));

      for (const point of points) {
        const seed = Number(point.evaluation_id || point.answer_id || 0);
        const jitterX = deterministicJitter(seed, 13);
        const jitterY = deterministicJitter(seed * 7, 13);
        const circle = document.createElementNS(svg.namespaceURI, "circle");
        circle.setAttribute("cx", String(scaleScore(point.reference_score, minScore, maxScore, margin.left, plotWidth) + jitterX));
        circle.setAttribute("cy", String(scaleScore(point.judge_score, minScore, maxScore, margin.top + plotHeight, -plotHeight) + jitterY));
        circle.setAttribute("r", "5.5");
        circle.setAttribute("class", "scatter-point");
        const title = document.createElementNS(svg.namespaceURI, "title");
        title.textContent = `resposta ${point.answer_id}: referencia ${point.reference_score}, juiz ${point.judge_score}`;
        circle.appendChild(title);
        svg.appendChild(circle);
      }
      svg.appendChild(svgText(margin.left + plotWidth / 2, height - 16, series.x_label || "referencia", "scatter-label", "middle"));
      const yLabel = svgText(18, margin.top + plotHeight / 2, series.y_label || "nota do juiz", "scatter-label", "middle");
      yLabel.setAttribute("transform", `rotate(-90 18 ${margin.top + plotHeight / 2})`);
      svg.appendChild(yLabel);
      root.appendChild(svg);
    }

    function renderCriticalErrorAnalysis(categories, cases) {
      renderBarChart("dashboard-critical-error-chart", categories || [], {scaleMax: 1, showPercent: true, tone: "bad"});
      const body = document.getElementById("dashboard-critical-error-body");
      body.textContent = "";
      if (!cases.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 7;
        cell.className = "muted";
        cell.textContent = "Sem erros criticos detectados no filtro atual.";
        row.appendChild(cell);
        body.appendChild(row);
        return;
      }
      cases.slice(0, 40).forEach((item) => {
        const row = document.createElement("tr");
        for (const value of [
          item.question_id,
          item.candidate_model,
          item.judge_model,
          item.score,
          item.error_type,
          item.short_justification
        ]) appendCell(row, display(value));
        appendAuditCell(row, item.evaluation_id);
        body.appendChild(row);
      });
    }

    function renderJudgeAgreement(cards, cases, errorMessage) {
      const cardRoot = document.getElementById("dashboard-judge-agreement-cards");
      cardRoot.textContent = "";
      const metrics = [
        ["Avaliacoes comparadas", cards.total_compared],
        ["Notas iguais", cards.delta_0],
        ["Divergencia de 1 ponto", cards.delta_1],
        ["Divergencia de 2 pontos", cards.delta_2],
        ["Divergencia de 3 pontos", cards.delta_3],
        ["Divergencia de 4 pontos", cards.delta_4],
        ["Arbitro acionado", cards.arbiter_triggered],
      ];
      for (const metric of metrics) {
        const card = document.createElement("div");
        card.className = "metric-card";
        const value = document.createElement("span");
        value.className = "metric-value";
        value.textContent = display(metric[1]);
        const label = document.createElement("span");
        label.className = "metric-label";
        label.textContent = metric[0];
        card.appendChild(value);
        card.appendChild(label);
        cardRoot.appendChild(card);
      }

      const body = document.getElementById("dashboard-judge-agreement-body");
      body.textContent = "";
      if (errorMessage) {
        appendJudgeAgreementMessage(body, errorMessage);
        return;
      }
      if (!cases.length) {
        appendJudgeAgreementMessage(body, "Sem arbitragens acionadas para o filtro atual.");
        return;
      }
      cases.slice(0, 40).forEach((item) => {
        const row = document.createElement("tr");
        for (const value of [
          item.answer_id,
          item.question_id,
          item.candidate_model,
          item.judge_1_score,
          item.judge_2_score,
          item.delta,
          item.arbiter_score,
          item.arbitration_reason,
        ]) appendCell(row, display(value));
        body.appendChild(row);
      });
    }

    function appendJudgeAgreementMessage(body, message) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 8;
      cell.className = "muted";
      cell.textContent = message;
      row.appendChild(cell);
      body.appendChild(row);
    }

    function scaleScore(value, minScore, maxScore, origin, span) {
      return origin + ((Number(value) - minScore) / (maxScore - minScore)) * span;
    }

    function deterministicJitter(seed, amplitude) {
      const value = Math.sin(seed * 12.9898) * 43758.5453;
      return (value - Math.floor(value) - 0.5) * amplitude;
    }

    function svgLine(x1, y1, x2, y2, className) {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", String(x1));
      line.setAttribute("y1", String(y1));
      line.setAttribute("x2", String(x2));
      line.setAttribute("y2", String(y2));
      line.setAttribute("class", className);
      return line;
    }

    function svgText(x, y, text, className, anchor) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", "text");
      node.setAttribute("x", String(x));
      node.setAttribute("y", String(y));
      node.setAttribute("class", className);
      node.setAttribute("text-anchor", anchor);
      node.textContent = text;
      return node;
    }

    function formatSpearmanValue(value) {
      return value == null ? "-" : Number(value).toFixed(2);
    }

    function formatPValue(value) {
      if (value == null) return "-";
      const number = Number(value);
      return number < 0.001 ? "<0.001" : number.toFixed(3);
    }

    let dashboardCarouselIndex = 0;

    function moveCarousel(delta) {
      const track = document.getElementById("dashboard-model-distribution-carousel");
      const cards = Array.from(track.querySelectorAll(".dashboard-carousel-slide"));
      if (!cards.length) return;
      const nextIndex = (dashboardCarouselIndex + delta + cards.length) % cards.length;
      goToCarouselPage(nextIndex);
    }

    function goToCarouselPage(index) {
      const track = document.getElementById("dashboard-model-distribution-carousel");
      const cards = Array.from(track.querySelectorAll(".dashboard-carousel-slide"));
      if (!cards.length) return;
      const nextIndex = (index + cards.length) % cards.length;
      dashboardCarouselIndex = nextIndex;
      track.style.transform = `translateX(-${nextIndex * 100}%)`;
      updateCarouselState();
    }

    function updateCarouselState() {
      const track = document.getElementById("dashboard-model-distribution-carousel");
      const cards = Array.from(track.querySelectorAll(".dashboard-carousel-slide"));
      const index = dashboardCarouselIndex;
      cards.forEach((card, cardIndex) => card.classList.toggle("active", cardIndex === index));
      for (const [tabIndex, tab] of Array.from(document.getElementById("dashboard-model-carousel-dots").children).entries()) {
        tab.classList.toggle("active", tabIndex === index);
        tab.setAttribute("aria-selected", String(tabIndex === index));
      }
      scrollCarouselTabsIntoView(index);
      updateCarouselControls(index, cards.length);
    }

    function scrollCarouselTabsIntoView(index) {
      const root = document.getElementById("dashboard-model-carousel-dots");
      const tabs = Array.from(root.children);
      if (!tabs.length || root.scrollWidth <= root.clientWidth) return;
      if (index <= 1) {
        root.scrollTo({left: 0, behavior: "auto"});
        return;
      }
      const firstVisibleIndex = Math.max(0, index - 1);
      const lastVisibleIndex = Math.min(tabs.length - 1, index + 1);
      const activeTab = tabs[index];
      const firstTab = tabs[firstVisibleIndex];
      const lastTab = tabs[lastVisibleIndex];
      const padding = 8;
      const targetLeft = firstTab.offsetLeft - padding;
      const targetRight = lastTab.offsetLeft + lastTab.offsetWidth + padding;
      const targetSpan = targetRight - targetLeft;
      const maxScrollLeft = Math.max(0, root.scrollWidth - root.clientWidth);
      let desiredLeft = targetLeft - Math.max(0, root.clientWidth - targetSpan) / 2;
      if (targetSpan > root.clientWidth) {
        activeTab.scrollIntoView({behavior: "smooth", block: "nearest", inline: "start"});
        return;
      }
      root.scrollTo({left: Math.min(maxScrollLeft, Math.max(0, desiredLeft)), behavior: "smooth"});
    }

    function resetCarouselTabsScroll() {
      const root = document.getElementById("dashboard-model-carousel-dots");
      root.scrollLeft = 0;
      scrollCarouselTabsIntoView(dashboardCarouselIndex);
    }

    function updateCarouselControls(index, total) {
      document.getElementById("dashboard-model-carousel-prev").disabled = total <= 1;
      document.getElementById("dashboard-model-carousel-next").disabled = total <= 1;
    }

    function renderDashboardCases(cases) {
      const body = document.getElementById("dashboard-cases-body");
      body.textContent = "";
      if (!cases.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 10;
        cell.className = "muted";
        cell.textContent = "Sem casos criticos ou divergencias no filtro atual.";
        row.appendChild(cell);
        body.appendChild(row);
        return;
      }
      cases.slice(0, 40).forEach((item) => {
        const row = document.createElement("tr");
        for (const value of [
          item.reason,
          item.dataset,
          item.answer_id,
          item.question_id,
          item.candidate_model,
          item.judge_model,
          normalizeRole(item.role),
          item.score,
          item.status
        ]) appendCell(row, display(value));
        appendAuditCell(row, item.evaluation_id);
        body.appendChild(row);
      });
    }

    function appendAuditCell(row, evaluationId) {
      const cell = document.createElement("td");
      if (!evaluationId) {
        cell.textContent = "-";
        row.appendChild(cell);
        return;
      }
      const button = document.createElement("button");
      button.type = "button";
      button.className = "secondary";
      button.textContent = "Auditar";
      button.onclick = () => openMetaEvaluation(evaluationId);
      cell.appendChild(button);
      row.appendChild(cell);
    }

    async function openMetaEvaluation(evaluationId) {
      activateTab("meta-panel");
      if (!metaOptionsLoaded) {
        await loadMetaOptions(evaluationId);
        return;
      }
      selectMetaEvaluationById(evaluationId);
      await loadMetaEvaluation();
    }

    function displayPercent(value) {
      return value === null || value === undefined ? "-" : `${value}%`;
    }

    function formatSpearman(value) {
      if (!value || !value.available) return "N/A";
      return `${Number(value.value).toFixed(3)} (n=${value.sample_size})`;
    }

    function renderAuditLog(data) {
      const cell = document.getElementById("audit-log");
      const path = data.audit_log || data.result?.audit_log || "-";
      const auditLogUrl = data.audit_log_url;
      currentAuditLogUrl = auditLogUrl || null;
      cell.textContent = "";
      if (auditLogUrl) {
        const row = document.createElement("span");
        row.className = "audit-log-row";
        const link = document.createElement("a");
        link.href = auditLogUrl;
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = path;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "audit-log-button";
        button.title = "Abrir live audit log";
        button.setAttribute("aria-label", "Abrir live audit log");
        const icon = document.createElement("span");
        icon.className = "audit-log-button-icon";
        icon.setAttribute("aria-hidden", "true");
        icon.textContent = "▤";
        const label = document.createElement("span");
        label.textContent = "Live log";
        button.appendChild(icon);
        button.appendChild(label);
        button.onclick = openAuditLogDialog;
        row.appendChild(link);
        row.appendChild(button);
        cell.appendChild(row);
        if (document.getElementById("audit-log-dialog").open) loadCurrentAuditLog();
        return;
      }
      cell.textContent = path;
    }

    function openAuditLogDialog() {
      document.getElementById("audit-log-dialog").showModal();
      loadCurrentAuditLog();
    }

    async function loadCurrentAuditLog() {
      const liveLog = document.getElementById("audit-log-content");
      if (!liveLog || !currentAuditLogUrl) return;
      try {
        const response = await fetch(currentAuditLogUrl);
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.detail || "Audit log ainda nao disponivel.");
        }
        liveLog.textContent = await response.text();
        liveLog.scrollTop = liveLog.scrollHeight;
      } catch (error) {
        liveLog.textContent = friendlyErrorMessage(error.message);
      }
    }

    function payload() {
      const judgeEndpointSource = value("endpoint_source_judge");
      const secondaryEndpointSource = value("endpoint_source_secondary");
      const arbiterEndpointSource = value("endpoint_source_arbiter");
      return {
        judge_input_source: value("judge_input_source"),
        panel_mode: value("panel_mode"),
        dataset: value("dataset"),
        batch_size: Number(value("batch_size")),
        judge_execution_strategy: value("judge_execution_strategy"),
        judge_model: value("judge_model"),
        secondary_judge_model: value("secondary_judge_model"),
        arbiter_judge_model: value("arbiter_judge_model"),
        always_run_arbiter: document.getElementById("always_run_arbiter").checked,
        remote_judge_base_url: judgeEndpointSource === "custom" ? value("remote_judge_base_url") : "",
        remote_judge_api_key: judgeEndpointSource === "custom" ? value("remote_judge_api_key") : "",
        remote_secondary_judge_base_url: secondaryEndpointSource === "custom" ? value("remote_secondary_judge_base_url") : "",
        remote_secondary_judge_api_key: secondaryEndpointSource === "custom" ? value("remote_secondary_judge_api_key") : "",
        remote_arbiter_judge_base_url: arbiterEndpointSource === "custom" ? value("remote_arbiter_judge_base_url") : "",
        remote_arbiter_judge_api_key: arbiterEndpointSource === "custom" ? value("remote_arbiter_judge_api_key") : "",
        endpoint_source_judge: judgeEndpointSource,
        endpoint_source_secondary: secondaryEndpointSource,
        endpoint_source_arbiter: arbiterEndpointSource,
        judge_arbitration_min_delta: optionalNumber("judge_arbitration_min_delta"),
        remote_judge_timeout_seconds: optionalNumber("remote_judge_timeout_seconds"),
        remote_judge_temperature: optionalNumber("remote_judge_temperature"),
        remote_judge_max_tokens: optionalNumber("remote_judge_max_tokens"),
        remote_judge_top_p: optionalNumber("remote_judge_top_p"),
        remote_judge_openai_compatible: value("remote_judge_openai_compatible") === "true",
        judge_save_raw_response: document.getElementById("judge_save_raw_response").checked
      };
    }

    function optionalNumber(id) {
      const raw = value(id);
      return raw === "" ? null : Number(raw);
    }

    async function postJson(url, body, retryOnCsrf = true) {
      if (!csrfToken) await loadConfig();
      const response = await fetch(url, {
        method: "POST",
        headers: {"content-type": "application/json", "x-csrf-token": csrfToken},
        body: JSON.stringify(body)
      });
      const {data, responseText} = await readJsonResponse(response, "Request failed");
      const parsedData = data || {};
      if (response.status === 403 && parsedData.detail === "Invalid CSRF token." && retryOnCsrf) {
        await loadConfig();
        return postJson(url, body, false);
      }
      if (!response.ok) throw new Error(parsedData.detail || responseText || "Request failed");
      return parsedData;
    }

    async function readJsonResponse(response, fallbackMessage = "Request failed") {
      const responseText = await response.text();
      if (!responseText) {
        if (response.ok) {
          throw new Error("Resposta vazia do servidor ao consultar o progresso da operacao.");
        }
        return {data: {detail: fallbackMessage}, responseText};
      }
      try {
        return {data: JSON.parse(responseText), responseText};
      } catch (error) {
        if (response.ok) {
          throw new Error("Resposta JSON incompleta do servidor ao consultar o progresso da operacao.");
        }
        return {data: {detail: responseText || fallbackMessage}, responseText};
      }
    }

    function toggleAssistantChat(forceOpen = null) {
      const panel = document.getElementById("assistant-chat-panel");
      const toggle = document.getElementById("assistant-chat-toggle");
      const shouldOpen = forceOpen === null ? panel.hidden : Boolean(forceOpen);
      panel.hidden = !shouldOpen;
      toggle.setAttribute("aria-expanded", String(shouldOpen));
      if (shouldOpen) document.getElementById("assistant-chat-input").focus();
    }

    function renderAssistantMessages() {
      const messages = document.getElementById("assistant-chat-messages");
      const panel = document.getElementById("assistant-chat-panel");
      const hasTable = assistantMessages.some((message) => message.role === "assistant" && containsMarkdownTable(message.text));
      panel.classList.toggle("has-table", hasTable);
      messages.textContent = "";
      if (!assistantMessages.length) {
        const empty = document.createElement("div");
        empty.className = "assistant-empty";
        empty.textContent = "Envie uma pergunta sobre os resultados, execucoes ou configuracoes disponiveis.";
        messages.appendChild(empty);
        return;
      }
      for (const message of assistantMessages) {
        const bubble = document.createElement("div");
        bubble.className = `assistant-message ${message.role}`;
        if (message.role === "assistant") {
          bubble.appendChild(renderAssistantMarkdown(message.text));
        } else {
          bubble.textContent = message.text;
        }
        messages.appendChild(bubble);
      }
      messages.scrollTop = messages.scrollHeight;
    }

    function renderAssistantMarkdown(text) {
      const root = document.createElement("div");
      root.className = "assistant-message-content";
      const lines = String(text || "").replaceAll(String.fromCharCode(13), "").split(String.fromCharCode(10));
      let paragraph = [];
      let index = 0;

      function flushParagraph() {
        if (!paragraph.length) return;
        const node = document.createElement("p");
        node.textContent = paragraph.join(String.fromCharCode(10)).trim();
        root.appendChild(node);
        paragraph = [];
      }

      while (index < lines.length) {
        if (isMarkdownTableAt(lines, index)) {
          flushParagraph();
          const tableLines = [];
          while (index < lines.length && isPipeTableRow(lines[index])) {
            tableLines.push(lines[index]);
            index += 1;
          }
          root.appendChild(renderMarkdownTable(tableLines));
          continue;
        }
        paragraph.push(lines[index]);
        index += 1;
      }
      flushParagraph();
      return root;
    }

    function isMarkdownTableAt(lines, index) {
      return (
        index + 1 < lines.length &&
        isPipeTableRow(lines[index]) &&
        isMarkdownSeparatorRow(lines[index + 1])
      );
    }

    function containsMarkdownTable(text) {
      const lines = String(text || "").replaceAll(String.fromCharCode(13), "").split(String.fromCharCode(10));
      return lines.some((line, index) => isMarkdownTableAt(lines, index));
    }

    function isPipeTableRow(line) {
      const trimmed = String(line || "").trim();
      return trimmed.includes("|") && splitMarkdownTableRow(trimmed).length >= 2;
    }

    function isMarkdownSeparatorRow(line) {
      const cells = splitMarkdownTableRow(line);
      return cells.length >= 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.trim()));
    }

    function splitMarkdownTableRow(line) {
      return String(line || "")
        .trim()
        .replace(/^[|]/, "")
        .replace(/[|]$/, "")
        .split("|")
        .map((cell) => cell.trim());
    }

    function renderMarkdownTable(tableLines) {
      const wrapper = document.createElement("div");
      wrapper.className = "assistant-table-wrap";
      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const tbody = document.createElement("tbody");
      const headers = splitMarkdownTableRow(tableLines[0]);
      const columnCount = headers.length;
      const headerRow = document.createElement("tr");

      for (const header of headers) {
        const cell = document.createElement("th");
        cell.textContent = header || "-";
        headerRow.appendChild(cell);
      }
      thead.appendChild(headerRow);

      for (const line of tableLines.slice(2)) {
        const row = document.createElement("tr");
        const cells = splitMarkdownTableRow(line);
        for (let cellIndex = 0; cellIndex < columnCount; cellIndex += 1) {
          const cell = document.createElement("td");
          cell.textContent = cells[cellIndex] || "-";
          row.appendChild(cell);
        }
        tbody.appendChild(row);
      }

      table.appendChild(thead);
      table.appendChild(tbody);
      wrapper.appendChild(table);
      return wrapper;
    }

    function setAssistantLoading(loading) {
      assistantLoading = loading;
      document.getElementById("assistant-chat-loading").hidden = !loading;
      document.getElementById("assistant-chat-input").disabled = loading;
      document.getElementById("assistant-chat-send").disabled = loading;
    }

    async function submitAssistantMessage(event) {
      event.preventDefault();
      if (assistantLoading) return;
      const input = document.getElementById("assistant-chat-input");
      const error = document.getElementById("assistant-chat-error");
      const message = input.value.trim();
      if (!message) return;
      assistantMessages.push({role: "user", text: message});
      input.value = "";
      error.hidden = true;
      error.textContent = "";
      renderAssistantMessages();
      setAssistantLoading(true);
      try {
        const data = await postJson("/api/assistant/chat", {message: message});
        assistantMessages.push({role: "assistant", text: data.answer || "O assistente nao retornou uma resposta."});
      } catch (requestError) {
        const friendlyMessage = friendlyErrorMessage(requestError.message);
        error.textContent = friendlyMessage;
        error.hidden = false;
        assistantMessages.push({role: "error", text: friendlyMessage});
      } finally {
        setAssistantLoading(false);
        renderAssistantMessages();
      }
    }

    async function postBackupFile(file, retryOnCsrf = true) {
      if (!csrfToken) await loadConfig();
      const response = await fetch("/api/database-restore", {
        method: "POST",
        headers: {"content-type": "application/sql", "x-csrf-token": csrfToken, "x-backup-filename": file.name},
        body: await file.arrayBuffer()
      });
      const data = await response.json();
      if (response.status === 403 && data.detail === "Invalid CSRF token." && retryOnCsrf) {
        await loadConfig();
        return postBackupFile(file, false);
      }
      if (!response.ok) throw new Error(data.detail || "Request failed");
      return data;
    }

    function renderProgress(progress) {
      const pct = progress?.percent ?? 0;
      document.getElementById("batch-progress").value = pct;
      setText("progress-label", `${pct}% - ${progress?.current ?? 0}/${progress?.total ?? 0} respostas`);
    }

    function renderRun(data) {
      const status = data.status || "dry-run";
      setText("run-status", status);
      renderStatusIcon(status);
      updateStopButton(data.run_id, status);
      renderAuditLog(data);
      setText("command-preview", data.command_preview || data.result?.command_preview || "");
      renderProgress(data.progress);
      const eligibility = data.eligibility || data.result?.eligibility;
      setText("eligible-missing", eligibility?.missing);
      setText("eligible-failed", eligibility?.failed);
      setText("eligible-successful", eligibility?.successful);
      setText("eligible-will-process", eligibility?.will_process);
      const summary = data.result?.summary;
      setText("selected", summary?.selected_answers ?? eligibility?.will_process ?? data.progress?.total);
      setText("executed", summary?.executed_evaluations ?? data.progress?.executed_evaluations);
      setText("skipped", summary?.skipped_evaluations ?? data.progress?.skipped_evaluations);
      setText("arbiters", summary?.arbiter_evaluations ?? data.progress?.arbiter_evaluations);
      if (data.error) setText("output", friendlyErrorMessage(data.error));
      else if (data.result) setText("output", data.result.execution_summary);
      renderPostRunPanel(data);
      renderExecutionTable(data.evaluation_events || []);
    }

    function renderPostRunPanel(data) {
      const panel = document.getElementById("post-run-panel");
      const status = data.status || "dry-run";
      const shouldShow = ["completed", "failed", "cancelled"].includes(status) && Boolean(data.result);
      panel.hidden = !shouldShow;
      if (!shouldShow) return;
      const stats = buildPostRunStats(data);
      renderMetricCards(stats);
      renderBarChart("score-distribution-chart", stats.scoreDistribution, {scaleMax: 1, showPercent: true, colorByLabel: true});
      renderBarChart("judge-failures-chart", stats.failuresByJudge, {scaleMax: 1, showPercent: true, tone: "bad"});
      renderBarChart("arbitration-chart", stats.arbitrations, {scaleMax: 1, showPercent: true, tone: "arbiter"});
      renderBarChart("candidate-average-chart", stats.averageByCandidate, {scaleMax: 5});
      renderBarChart("judge-average-chart", stats.averageByJudge, {scaleMax: 5});
    }

    function buildPostRunStats(data) {
      const events = data.evaluation_events || [];
      const summary = data.result?.summary || {};
      const scoredEvents = events.filter((event) => event.status === "success" && Number.isFinite(Number(event.score)));
      const failedEvents = events.filter((event) => event.status === "failed");
      const successCount = events.filter((event) => event.status === "success").length;
      const scoreDistribution = [1, 2, 3, 4, 5].map((score) => ({
        label: String(score),
        value: scoredEvents.filter((event) => Number(event.score) === score).length
      }));
      const avgScore = average(scoredEvents.map((event) => Number(event.score)));
      return {
        selectedAnswers: summary.selected_answers ?? data.eligibility?.will_process ?? data.progress?.total ?? 0,
        judgeCalls: events.filter((event) => event.status !== "skipped").length || summary.executed_evaluations || 0,
        successCount,
        failedCount: failedEvents.length,
        arbiterCount: summary.arbiter_evaluations ?? data.progress?.arbiter_evaluations ?? 0,
        averageScore: avgScore,
        duration: data.duration || "-",
        scoreDistribution,
        failuresByJudge: countBy(failedEvents, (event) => event.judge_model || "sem juiz"),
        arbitrations: [
          {label: "acionadas", value: summary.arbiter_evaluations ?? data.progress?.arbiter_evaluations ?? 0},
          {label: "sem arbitro", value: Math.max(0, (summary.selected_answers ?? data.progress?.total ?? 0) - (summary.arbiter_evaluations ?? data.progress?.arbiter_evaluations ?? 0))}
        ],
        averageByCandidate: averageBy(scoredEvents, (event) => event.candidate_model || "sem modelo"),
        averageByJudge: averageBy(scoredEvents, (event) => event.judge_model || "sem juiz")
      };
    }

    function renderMetricCards(stats) {
      const root = document.getElementById("post-run-cards");
      root.textContent = "";
      for (const metric of [
        ["Respostas selecionadas", stats.selectedAnswers],
        ["Chamadas de juiz realizadas", stats.judgeCalls],
        ["Success", stats.successCount],
        ["Failed", stats.failedCount],
        ["Arbitragens acionadas", stats.arbiterCount],
        ["Nota media", formatAverage(stats.averageScore)],
        ["Tempo total", stats.duration]
      ]) {
        const card = document.createElement("div");
        card.className = "metric-card";
        const value = document.createElement("span");
        value.className = "metric-value";
        value.textContent = display(metric[1]);
        const label = document.createElement("span");
        label.className = "metric-label";
        label.textContent = metric[0];
        card.appendChild(value);
        card.appendChild(label);
        root.appendChild(card);
      }
    }

    function renderBarChart(id, rows, options = {}) {
      const root = document.getElementById(id);
      root.textContent = "";
      const values = rows || [];
      const total = values.reduce((sum, row) => sum + (Number(row.value) || 0), 0);
      const max = Math.max(options.scaleMax || 0, ...values.map((row) => Number(row.value) || 0));
      if (!values.length) {
        const empty = document.createElement("div");
        empty.className = "muted";
        empty.textContent = "Sem dados.";
        root.appendChild(empty);
        return;
      }
      for (const row of values) {
        const value = Number(row.value) || 0;
        const line = document.createElement("div");
        line.className = "bar-row";
        const label = document.createElement("span");
        label.className = "bar-label";
        label.title = row.label;
        label.textContent = row.label;
        const track = document.createElement("span");
        track.className = "bar-track";
        const fill = document.createElement("span");
        fill.className = "bar-fill";
        if (value === 0) fill.classList.add("zero");
        applyBarTone(fill, row, options);
        const basis = options.showPercent ? total : max;
        fill.style.width = `${basis ? Math.round((value / basis) * 100) : 0}%`;
        track.appendChild(fill);
        const number = document.createElement("span");
        number.className = "bar-value";
        const count = document.createElement("span");
        count.className = `bar-count ${valueTone(value, row, options)}`;
        count.style.setProperty("--pill-pct", `${max ? Math.round((value / max) * 100) : 0}%`);
        count.textContent = Number.isInteger(value) ? String(value) : value.toFixed(1);
        number.appendChild(count);
        if (options.showPercent) {
          const percent = document.createElement("span");
          percent.className = "bar-percent";
          percent.textContent = `(${total ? Math.round((value / total) * 100) : 0}%)`;
          number.appendChild(percent);
        }
        line.appendChild(label);
        line.appendChild(track);
        line.appendChild(number);
        root.appendChild(line);
      }
    }

    function applyBarTone(fill, row, options) {
      if (options.colorByLabel && ["1", "2", "3", "4", "5"].includes(String(row.label))) {
        fill.classList.add(`score-${row.label}`);
        return;
      }
      if (options.tone === "bad") fill.classList.add("failed");
      else if (options.tone === "arbiter" && row.label === "acionadas") fill.classList.add("arbiter");
      else if (options.tone === "arbiter") fill.classList.add("none");
    }

    function valueTone(value, row, options) {
      if (!value) return "";
      if (options.tone === "bad") return "bad";
      if (options.tone === "arbiter" && row.label === "acionadas") return "warning";
      if (options.colorByLabel && Number(row.label) <= 2) return "bad";
      if (options.colorByLabel && Number(row.label) >= 4) return "positive";
      return "";
    }

    function countBy(events, keyFn) {
      const counts = new Map();
      for (const event of events) counts.set(keyFn(event), (counts.get(keyFn(event)) || 0) + 1);
      return Array.from(counts, ([label, value]) => ({label, value})).sort((a, b) => b.value - a.value || a.label.localeCompare(b.label));
    }

    function averageBy(events, keyFn) {
      const groups = new Map();
      for (const event of events) {
        const key = keyFn(event);
        const current = groups.get(key) || {sum: 0, count: 0};
        current.sum += Number(event.score);
        current.count += 1;
        groups.set(key, current);
      }
      return Array.from(groups, ([label, value]) => ({label, value: value.sum / value.count})).sort((a, b) => b.value - a.value || a.label.localeCompare(b.label));
    }

    function average(values) {
      if (!values.length) return null;
      return values.reduce((sum, value) => sum + value, 0) / values.length;
    }

    function formatAverage(value) {
      return value === null || value === undefined ? "-" : value.toFixed(1);
    }

    function renderExecutionTable(events) {
      const body = document.getElementById("execution-table-body");
      const previousRects = new Map(Array.from(body.querySelectorAll("tr[data-event-key]")).map((row) => [row.dataset.eventKey, row.getBoundingClientRect()]));
      body.textContent = "";
      if (!events.length) {
        executionTableState.clear();
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 13;
        cell.className = "muted";
        cell.textContent = "Aguardando execucao.";
        row.appendChild(cell);
        body.appendChild(row);
        return;
      }
      events.forEach((event, index) => {
        const row = document.createElement("tr");
        const eventKey = executionEventKey(event);
        const previous = executionTableState.get(eventKey);
        row.dataset.eventKey = eventKey;
        appendStatusCell(row, event.status);
        for (const value of [
          event.dataset,
          event.question_id,
          event.candidate_model,
          event.judge_model,
          normalizeRole(event.role),
          event.score,
          event.delta,
          formatBoolean(event.arbiter_triggered),
          event.trigger_reason,
          formatLatency(event.latency_ms),
          event.error ? friendlyErrorMessage(event.error) : null
        ]) appendCell(row, display(value));
        const detailsCell = document.createElement("td");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "detail-button";
        button.textContent = "Detalhes";
        button.onclick = () => openDetails(event, index);
        detailsCell.appendChild(button);
        row.appendChild(detailsCell);
        body.appendChild(row);
        animateExecutionRow(row, previousRects.get(eventKey), previous?.status, event.status);
        executionTableState.set(eventKey, {status: event.status});
      });
      const currentKeys = new Set(events.map(executionEventKey));
      for (const key of executionTableState.keys()) {
        if (!currentKeys.has(key)) executionTableState.delete(key);
      }
    }

    function executionEventKey(event) {
      return [
        event.dataset,
        event.question_id,
        event.answer_id,
        event.candidate_model,
        event.judge_model,
        event.role,
        event.panel_mode,
        event.trigger_reason || ""
      ].map((value) => String(value ?? "")).join("\\u001f");
    }

    function animateExecutionRow(row, previousRect, previousStatus, nextStatus) {
      const isNew = previousStatus === undefined;
      const statusChanged = previousStatus !== undefined && previousStatus !== nextStatus;
      if (isNew && nextStatus === "running") row.classList.add("execution-row-enter");
      if (statusChanged && nextStatus === "success") row.classList.add("execution-row-success");
      if (statusChanged && nextStatus === "failed") row.classList.add("execution-row-failed");
      if (!statusChanged || !previousRect || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      const nextRect = row.getBoundingClientRect();
      const deltaY = previousRect.top - nextRect.top;
      if (Math.abs(deltaY) < 1) return;
      row.style.transform = `translateY(${deltaY}px)`;
      row.style.transition = "transform 0s";
      requestAnimationFrame(() => {
        row.style.transition = "";
        row.style.transform = "";
      });
    }

    function appendCell(row, value) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    }

    function confirmAction(message) {
      return window["confirm"](message);
    }

    function appendStatusCell(row, status) {
      const cell = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = `badge ${status || ""}`;
      badge.textContent = status || "-";
      cell.appendChild(badge);
      row.appendChild(cell);
    }

    function normalizeRole(role) {
      if (role === "principal") return "primary";
      if (role === "controle") return "secondary";
      if (role === "arbitro") return "arbiter";
      return role;
    }

    function formatBoolean(value) {
      if (value === true) return "sim";
      if (value === false) return "nao";
      return "-";
    }

    function formatLatency(value) {
      return value === null || value === undefined ? "-" : `${value} ms`;
    }

    function openDetails(event, index) {
      setText("details-title", `Detalhes da avaliacao #${index + 1} - resposta ${event.answer_id || "-"}`);
      setText("details-prompt", event.prompt || "-");
      setText("details-response", event.raw_response || "-");
      setText("details-rationale", event.rationale || friendlyErrorMessage(event.error) || "-");
      document.getElementById("details-dialog").showModal();
    }

    function renderStatusIcon(status) {
      const icon = document.getElementById("run-status-icon");
      icon.className = "status-icon";
      if (["queued", "running", "cancelling"].includes(status)) {
        icon.textContent = "";
        icon.classList.add("spinner");
      } else if (status === "completed" || status === "dry-run") {
        icon.textContent = "✓";
        icon.classList.add("ok");
      } else if (status === "failed") {
        icon.textContent = "!";
        icon.classList.add("bad");
      } else {
        icon.textContent = "-";
      }
    }

    function updateStopButton(runId, status) {
      const button = document.getElementById("stop-run");
      const canStop = Boolean(runId) && ["queued", "running", "cancelling"].includes(status);
      button.disabled = !canStop || status === "cancelling";
      button.textContent = status === "cancelling" ? "Parando..." : "Parar";
      activeRunId = canStop ? runId : null;
    }

    async function poll(runId) {
      let response;
      try {
        response = await fetch(`/api/runs/${runId}`, {cache: "no-store"});
      } catch (error) {
        setText("output", `Falha ao atualizar execucao: ${friendlyErrorMessage(error.message)}`);
        return;
      }
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setText("output", data.detail || "Falha ao atualizar execucao.");
        if (response.status === 404 && pollTimer) {
          clearInterval(pollTimer);
          pollTimer = null;
          activeRunId = null;
          updateStopButton(null, "idle");
        }
        return;
      }
      renderRun(data);
      if (["completed", "failed", "cancelled"].includes(data.status) && pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
        activeRunId = null;
        refreshRunHistory();
        refreshOperationalLogSummary();
      }
    }

    async function loadHistory() {
      await getRunHistory();
    }

    async function loadPromptOptions() {
      const response = await fetch("/api/judge-prompts/options");
      const data = await response.json();
      populateSelect("prompt_dataset", data.datasets || [], "value", "label");
      promptOptionsLoaded = true;
      if (value("prompt_dataset")) await loadPromptConfig();
    }

    async function loadPromptConfig() {
      const dataset = value("prompt_dataset");
      if (!dataset) return;
      setText("prompt_status", "Carregando configuracao...");
      try {
        const response = await fetch(`/api/judge-prompts?dataset=${encodeURIComponent(dataset)}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Falha ao carregar prompt.");
        const record = data.record;
        document.getElementById("prompt_body").value = record?.prompt || "";
        document.getElementById("prompt_persona").value = record?.persona || "";
        document.getElementById("prompt_context").value = record?.context || "";
        document.getElementById("prompt_rubric").value = record?.rubric || "";
        document.getElementById("prompt_output").value = record?.output || "";
        renderPromptLogs(data.versions || []);
        renderPromptPreview(data.preview);
        setText("prompt_status", record ? `Versao ativa v${display(record.version)} carregada. Criada em: ${formatDateTime(record.created_at)}` : "Nenhuma configuracao salva para esse dataset.");
      } catch (error) {
        setText("prompt_status", friendlyErrorMessage(error.message));
      }
    }

    async function loadMetaOptions(selectedEvaluationId = null) {
      const response = await fetch("/api/meta-evaluations/options");
      const data = await response.json();
      metaEvaluationOptions = data.evaluations || [];
      if (selectedEvaluationId) selectMetaEvaluationById(selectedEvaluationId);
      renderMetaEvaluationOptions(false);
      metaOptionsLoaded = true;
      if (getMetaEvaluationId()) await loadMetaEvaluation();
      else renderMetaEvaluationState(null, []);
    }

    async function loadRagCurationOptions(preferredDataset = null) {
      setText("rag_curation_status", "Carregando cobertura da curadoria...");
      setText("rag_vector_status", "Carregando base vetorial...");
      try {
        const response = await fetch("/api/rag-curation/options");
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Falha ao carregar datasets de curadoria.");
        ragCurationDatasetOptions = data.datasets || [];
        const options = ragCurationDatasetOptions.map((entry) => ({
          value: entry.dataset,
          label: `${display(entry.dataset)} | ${display(entry.dataset_name)} (${display(entry.curated_questions)}/${display(entry.total_questions)})`
        }));
        populateSelect("rag_curation_dataset", options, "value", "label");
        populateSelect("rag_vector_dataset", options, "value", "label");
        populateSelect("rag_query_dataset", options, "value", "label");
        populateSelect("rag_embedding_dataset", options, "value", "label");
        renderRagCurationDatasetCoverage(ragCurationDatasetOptions);
        ragCurationOptionsLoaded = true;
        const selectedDataset = preferredDataset || value("rag_curation_dataset");
        if (selectedDataset) {
          syncRagDatasetSelects(selectedDataset);
          await loadRagCurationDataset(selectedDataset);
          await loadRagVectorPreview(selectedDataset);
          await loadRagEmbeddingConfig(selectedDataset);
        } else {
          setText("rag_curation_status", "Nenhum dataset disponivel para curadoria.");
          setText("rag_vector_status", "Nenhum dataset disponivel para base vetorial.");
          setText("rag_query_status", "Nenhum dataset disponivel para consulta vetorial.");
          setText("rag_embedding_status", "Nenhum dataset disponivel para configuracao de embedding.");
          renderRagCurationDatasetState(null, [], [], null, []);
          renderRagVectorPreview(null);
          renderRagEmbeddingConfig(null);
        }
      } catch (error) {
        setText("rag_curation_status", friendlyErrorMessage(error.message));
        setText("rag_vector_status", friendlyErrorMessage(error.message));
        setText("rag_query_status", friendlyErrorMessage(error.message));
        setText("rag_embedding_status", friendlyErrorMessage(error.message));
      }
    }

    async function loadRagCurationDataset(dataset = value("rag_curation_dataset")) {
      if (!dataset) {
        renderRagCurationDatasetState(null, [], [], null);
        setText("rag_curation_status", "Selecione um dataset para visualizar a curadoria.");
        setText("rag_vector_status", "Selecione um dataset para visualizar a base vetorial.");
        return;
      }
      syncRagDatasetSelects(dataset);
      setText("rag_curation_status", "Carregando curadoria...");
      setText("rag_vector_status", "Carregando base vetorial...");
      try {
        const response = await fetch(`/api/rag-curation?dataset=${encodeURIComponent(dataset)}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Falha ao carregar curadoria.");
        renderRagCurationDatasetState(
          data.active,
          data.runs || [],
          data.items || [],
          data.vector_base || null,
          data.vector_runs || []
        );
        if (data.active) {
          setText(
            "rag_curation_status",
            `Curadoria ativa de ${display(dataset)} carregada com ${display(data.active.curated_questions)} questoes curadas.`
          );
        } else {
          setText("rag_curation_status", `Nenhuma versao ativa encontrada para ${display(dataset)}.`);
        }
        if (data.vector_base) {
          setText("rag_vector_status", `Base vetorial de ${display(dataset)} carregada com ${display(data.vector_base.document_count)} documentos e ${display(data.vector_base.chunk_count)} chunks.`);
        } else {
          setText("rag_vector_status", `Nenhuma base vetorial ativa encontrada para ${display(dataset)}.`);
        }
      } catch (error) {
        renderRagCurationDatasetState(null, [], [], null, []);
        setText("rag_curation_status", friendlyErrorMessage(error.message));
        setText("rag_vector_status", friendlyErrorMessage(error.message));
      }
    }

    function renderRagCurationDatasetCoverage(rows) {
      const body = document.getElementById("rag_curation_datasets_body");
      body.textContent = "";
      if (!rows.length) {
        appendTableMessage(body, 9, "Nenhum dataset de curadoria encontrado.");
        return;
      }
      rows.forEach((entry) => {
        const row = document.createElement("tr");
        appendCell(row, display(entry.dataset));
        appendCell(row, display(entry.dataset_name));
        appendCell(row, `${display(entry.curated_questions)} / ${display(entry.total_questions)}`);
        appendCell(row, entry.active_run_id ? `run ${display(entry.active_run_id)}` : "-");
        appendCell(
          row,
          entry.vector_retrieval_run_id
            ? `${display(entry.vector_retrieval_name)} (#${display(entry.vector_retrieval_run_id)})`
            : "-"
        );
        appendCell(row, display(entry.vector_document_count));
        appendCell(row, display(entry.vector_chunk_count));
        appendCell(row, formatRagVectorStatus(entry.vector_status));
        appendCell(row, formatDateTime(entry.active_imported_at));
        body.appendChild(row);
      });
    }

    function renderRagCurationDatasetState(active, runs, items, vectorBase, vectorRuns = []) {
      renderRagCurationActiveSummary(active);
      renderRagVectorBase(vectorBase);
      renderRagVectorRuns(vectorRuns || []);
      renderRagCurationRuns(runs || []);
      renderRagCurationItems(items || []);
      renderRagCurationDetail(null);
    }

    function clearRagVectorProgress() {
      const progress = document.getElementById("rag_vector_progress");
      progress.textContent = "";
    }

    function appendRagVectorProgress(text, state = "running") {
      const progress = document.getElementById("rag_vector_progress");
      const item = document.createElement("div");
      item.className = `rag-progress-item ${state}`;
      const time = document.createElement("span");
      time.className = "rag-progress-time";
      time.textContent = new Date().toLocaleTimeString("pt-BR", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
      });
      const marker = document.createElement("span");
      marker.className = "rag-progress-marker";
      marker.textContent = state === "error" ? "erro" : state === "done" ? "ok" : "...";
      const label = document.createElement("span");
      label.className = "rag-progress-text";
      label.textContent = text;
      item.appendChild(time);
      item.appendChild(marker);
      item.appendChild(label);
      progress.appendChild(item);
      progress.scrollTop = progress.scrollHeight;
    }

    function ragVectorQuestionRangePayload() {
      const startRaw = value("rag_vector_question_start").trim();
      const endRaw = value("rag_vector_question_end").trim();
      const start = startRaw ? Number(startRaw) : null;
      const end = endRaw ? Number(endRaw) : null;
      if (start !== null && (!Number.isInteger(start) || start < 1)) {
        throw new Error("Questao inicial deve ser um numero inteiro maior ou igual a 1.");
      }
      if (end !== null && (!Number.isInteger(end) || end < 1)) {
        throw new Error("Questao final deve ser um numero inteiro maior ou igual a 1.");
      }
      if (start !== null && end !== null && start > end) {
        throw new Error("Questao inicial nao pode ser maior que a questao final.");
      }
      return {
        question_sequence_start: start,
        question_sequence_end: end,
      };
    }

    function formatRagChunkOriginSummary(chunkSummary) {
      if (!chunkSummary) return "";
      const curation = Number(chunkSummary.curation_chunks || 0);
      const sourceUrls = Number(chunkSummary.source_url_chunks || 0);
      const parts = [];
      if (curation) parts.push(`${display(curation)} chunks da curadoria`);
      if (sourceUrls) parts.push(`${display(sourceUrls)} chunks de fonte/URL`);
      return parts.length ? parts.join(" + ") : "";
    }

    function formatRagVectorStatus(status) {
      switch (status) {
        case "pronta_com_embeddings":
          return "pronta com embeddings";
        case "materializada_sem_embeddings":
          return "materializada sem embeddings";
        case "desatualizada":
          return "desatualizada";
        case "nao_materializada":
          return "nao materializada";
        default:
          return display(status);
      }
    }

    function renderRagCurationActiveSummary(active) {
      if (!active) {
        setText("rag_curation_active_meta", "Nenhuma versao ativa carregada.");
        setText("rag_curation_active_filename", "-");
        setText("rag_curation_active_imported_by", "-");
        setText("rag_curation_active_imported_at", "-");
        setText("rag_curation_active_coverage", "-");
        setText("rag_curation_active_item_count", "-");
        setText("rag_curation_active_article_count", "-");
        return;
      }
      setText(
        "rag_curation_active_meta",
        `${display(active.dataset)} | ${display(active.dataset_name)} | run ${display(active.active_run_id)}`
      );
      setText("rag_curation_active_filename", display(active.active_filename));
      setText("rag_curation_active_imported_by", display(active.active_imported_by));
      setText("rag_curation_active_imported_at", formatDateTime(active.active_imported_at));
      setText(
        "rag_curation_active_coverage",
        `${display(active.curated_questions)} / ${display(active.total_questions)} questoes`
      );
      setText("rag_curation_active_item_count", display(active.active_item_count));
      setText("rag_curation_active_article_count", display(active.active_article_count));
    }

    function renderRagVectorBase(vectorBase) {
      if (!vectorBase) {
        setText("rag_vector_active_meta", "Nenhuma base vetorial ativa para esse dataset.");
        setText("rag_vector_active_run", "-");
        setText("rag_vector_active_status", "-");
        setText("rag_vector_active_strategy", "-");
        setText("rag_vector_active_top_k", "-");
        setText("rag_vector_active_document_count", "-");
        setText("rag_vector_active_chunk_count", "-");
        setText("rag_vector_active_embedding_count", "-");
        setText("rag_vector_active_embedding_model", "-");
        setText("rag_vector_active_created_at", "-");
        return;
      }
      setText(
        "rag_vector_active_meta",
        `${display(vectorBase.dataset)} | retrieval #${display(vectorBase.retrieval_run_id)} | import run ${display(vectorBase.import_run_id)}`
      );
      setText("rag_vector_active_run", display(vectorBase.retrieval_name));
      setText("rag_vector_active_status", formatRagVectorStatus(vectorBase.status));
      setText("rag_vector_active_strategy", display(vectorBase.retrieval_strategy));
      setText("rag_vector_active_top_k", display(vectorBase.top_k));
      setText("rag_vector_active_document_count", display(vectorBase.document_count));
      setText("rag_vector_active_chunk_count", display(vectorBase.chunk_count));
      setText("rag_vector_active_embedding_count", display(vectorBase.embedding_count));
      setText("rag_vector_active_embedding_model", display(vectorBase.embedding_model));
      setText("rag_vector_active_created_at", formatDateTime(vectorBase.created_at));
    }

    function renderRagVectorRuns(rows) {
      const body = document.getElementById("rag_vector_runs_body");
      body.textContent = "";
      if (!rows.length) {
        appendTableMessage(body, 7, "Nenhuma run vetorial carregada.");
        return;
      }
      rows.forEach((entry) => {
        const row = document.createElement("tr");
        appendCell(row, `${display(entry.retrieval_name)} (#${display(entry.run_id)})`);
        appendCell(row, display(entry.retrieval_strategy));
        appendCell(row, display(entry.document_count));
        appendCell(row, display(entry.chunk_count));
        appendCell(row, display(entry.embedding_count));
        appendCell(row, formatDateTime(entry.created_at));
        const actionsCell = document.createElement("td");
        if (entry.active) {
          actionsCell.textContent = "Ativa";
        } else {
          const activateButton = document.createElement("button");
          activateButton.type = "button";
          activateButton.className = "secondary";
          activateButton.textContent = "Ativar";
          activateButton.onclick = () => activateRagVectorRun(entry.run_id);
          actionsCell.appendChild(activateButton);

          const deleteButton = document.createElement("button");
          deleteButton.type = "button";
          deleteButton.className = "danger-button";
          deleteButton.style.marginLeft = "8px";
          deleteButton.textContent = "Excluir";
          deleteButton.onclick = () => deleteRagVectorRun(entry.run_id, entry.retrieval_name);
          actionsCell.appendChild(deleteButton);
        }
        row.appendChild(actionsCell);
        body.appendChild(row);
      });
    }

    function renderRagVectorPreview(data) {
      const documentsBody = document.getElementById("rag_query_documents_body");
      const chunksBody = document.getElementById("rag_query_chunks_body");
      documentsBody.textContent = "";
      chunksBody.textContent = "";
      if (!data || !Array.isArray(data.documents) || !data.documents.length) {
        appendTableMessage(documentsBody, 3, "Nenhum documento carregado.");
      } else {
        for (const entry of data.documents) {
          const row = document.createElement("tr");
          appendCell(row, display(entry.document_key));
          appendCell(row, display(entry.lei));
          appendCell(row, display(entry.norma));
          documentsBody.appendChild(row);
        }
      }
      if (!data || !Array.isArray(data.chunks) || !data.chunks.length) {
        appendTableMessage(chunksBody, 4, "Nenhum chunk carregado.");
      } else {
        for (const entry of data.chunks) {
          const row = document.createElement("tr");
          appendCell(row, display(entry.chunk_id));
          appendCell(row, display(entry.lei));
          appendCell(row, display(entry.norma));
          appendCell(row, truncateText(entry.chunk_text, 180));
          chunksBody.appendChild(row);
        }
      }
    }

    function renderRagQueryResults(data) {
      const body = document.getElementById("rag_query_results_body");
      body.textContent = "";
      if (!data || !Array.isArray(data.results) || !data.results.length) {
        appendTableMessage(body, 6, "Nenhuma busca executada.");
        renderRagQuerySelectedResult(null, data);
      } else {
        for (const entry of data.results) {
          const row = document.createElement("tr");
          const detailCell = document.createElement("td");
          const detailButton = document.createElement("button");
          detailButton.type = "button";
          detailButton.className = "detail-button";
          detailButton.textContent = "Detalhe";
          detailButton.onclick = () => renderRagQuerySelectedResult(entry, data);
          detailCell.appendChild(detailButton);
          row.appendChild(detailCell);
          appendCell(row, display(entry.rank));
          appendCell(row, entry.similarity !== null && entry.similarity !== undefined ? entry.similarity.toFixed(4) : "-");
          appendCell(row, display(entry.lei));
          appendCell(row, display(entry.norma));
          appendCell(row, truncateText(entry.chunk_text, 220));
          body.appendChild(row);
        }
        renderRagQuerySelectedResult(data.results[0], data);
      }
      if (!data) {
        setText("rag_query_meta", "Nenhuma consulta executada.");
        setText("rag_query_summary_dataset", "-");
        setText("rag_query_summary_top_k", "-");
        setText("rag_query_summary_latency", "-");
        setText("rag_query_summary_dimensions", "-");
        return;
      }
      setText("rag_query_meta", `${display(data.dataset)} | ${display(data.query)}`);
      setText("rag_query_summary_dataset", display(data.dataset));
      setText("rag_query_summary_top_k", display(data.top_k));
      setText("rag_query_summary_latency", data.latency_ms !== null && data.latency_ms !== undefined ? `${data.latency_ms} ms` : "-");
      setText("rag_query_summary_dimensions", display(data.returned_dimensions));
    }

    function renderRagQuerySelectedResult(entry, data = null) {
      const urlCell = document.getElementById("rag_query_selected_url");
      urlCell.textContent = "";
      if (!entry) {
        setText("rag_query_selected_meta", data ? "Nenhum resultado selecionado." : "Nenhum resultado selecionado.");
        setText("rag_query_selected_rank", "-");
        setText("rag_query_selected_score", "-");
        setText("rag_query_selected_chunk_id", "-");
        setText("rag_query_selected_document_key", "-");
        setText("rag_query_selected_lei", "-");
        setText("rag_query_selected_norma", "-");
        setText("rag_query_selected_source_kind", "-");
        setText("rag_query_selected_urn", "-");
        setText("rag_query_selected_chunk_text", "Selecione um resultado para ver o texto completo.");
        urlCell.textContent = "-";
        return;
      }
      const sourceKindLabels = {
        curated_article: "curadoria/artigo curado",
        curation_summary: "curadoria/resumo",
        source_url_content: "fonte/URL recuperada",
      };
      const score = entry.similarity !== null && entry.similarity !== undefined ? entry.similarity.toFixed(4) : "-";
      const metaParts = [
        display(data?.dataset),
        `rank ${display(entry.rank)}`,
        display(entry.document_key),
      ].filter(Boolean);
      setText("rag_query_selected_meta", metaParts.join(" | "));
      setText("rag_query_selected_rank", display(entry.rank));
      setText("rag_query_selected_score", score);
      setText("rag_query_selected_chunk_id", display(entry.chunk_id));
      setText("rag_query_selected_document_key", display(entry.document_key));
      setText("rag_query_selected_lei", display(entry.lei));
      setText("rag_query_selected_norma", display(entry.norma));
      setText("rag_query_selected_source_kind", display(sourceKindLabels[entry.chunk_kind] || entry.chunk_kind));
      setText("rag_query_selected_urn", display(entry.urn));
      setText("rag_query_selected_chunk_text", entry.chunk_text || "-");
      if (entry.url) {
        const link = document.createElement("a");
        link.href = entry.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = entry.url;
        urlCell.appendChild(link);
      } else {
        urlCell.textContent = "-";
      }
    }

    async function loadRagVectorPreview(dataset = value("rag_query_dataset")) {
      if (!dataset) {
        renderRagVectorPreview(null);
        renderRagQueryResults(null);
        setText("rag_query_status", "Selecione um dataset para inspecionar a base vetorial.");
        return;
      }
      syncRagDatasetSelects(dataset);
      setText("rag_query_status", "Carregando amostras da base vetorial...");
      try {
        const response = await fetch(`/api/rag-vector/preview?dataset=${encodeURIComponent(dataset)}&limit=8`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Falha ao carregar a base vetorial.");
        renderRagVectorPreview(data);
        renderRagQueryResults(null);
        const docs = Array.isArray(data.documents) ? data.documents.length : 0;
        const chunks = Array.isArray(data.chunks) ? data.chunks.length : 0;
        setText("rag_query_status", `Amostra carregada para ${display(dataset)} (${docs} documentos, ${chunks} chunks).`);
      } catch (error) {
        renderRagVectorPreview(null);
        renderRagQueryResults(null);
        setText("rag_query_status", friendlyErrorMessage(error.message));
      }
    }

    async function generateRagEmbeddings() {
      const dataset = value("rag_vector_dataset");
      if (!dataset) {
        setText("rag_vector_status", "Selecione um dataset para gerar embeddings.");
        return;
      }
      const generateButton = document.getElementById("rag_vector_generate");
      try {
        clearRagVectorProgress();
        generateButton.disabled = true;
        setText("rag_vector_status", `Gerando embeddings para ${display(dataset)}...`);
        const range = ragVectorQuestionRangePayload();
        const rangeText = range.question_sequence_start || range.question_sequence_end
          ? ` Intervalo: ${display(range.question_sequence_start || 1)}-${display(range.question_sequence_end || "fim")}.`
          : "";
        appendRagVectorProgress(`Iniciando job para ${display(dataset)}.${rangeText}`);
        const started = await postJson("/api/rag-vector/generate-embeddings/jobs", {
          dataset,
          batch_size: 16,
          ...range,
        });
        let lastEventIndex = -1;
        let data = started;
        while (true) {
          const events = Array.isArray(data.events) ? data.events : [];
          for (const event of events) {
            const index = Number(event.index ?? -1);
            if (index <= lastEventIndex) continue;
            lastEventIndex = index;
            appendRagVectorProgress(display(event.message), event.state || "running");
          }
          if (!["queued", "running"].includes(data.status)) break;
          const eventCount = events.length;
          setText("rag_vector_status", `Gerando embeddings para ${display(dataset)}... ${display(eventCount)} evento(s) registrados.`);
          await sleep(1000);
          const response = await fetch(`/api/rag-vector/generate-embeddings/jobs/${encodeURIComponent(data.job_id)}`);
          const parsed = await readJsonResponse(response, "Falha ao consultar progresso da geracao.");
          data = parsed.data || {};
          if (!response.ok) throw new Error(data.detail || parsed.responseText || "Falha ao consultar progresso da geracao.");
        }
        if (data.status === "failed") {
          throw new Error(data.error || "Falha ao gerar embeddings.");
        }
        const result = data.result || {};
        const summary = result.summary || null;
        const sourceSummary = result.source_url_summary || {};
        const chunkOriginSummary = formatRagChunkOriginSummary(result.chunk_summary || {});
        const sourceFailures = sourceSummary.failures || [];
        const sourceDedupText = sourceSummary.deduplicated
          ? `, ${display(sourceSummary.deduplicated)} URL(s) duplicada(s) por normalizacao`
          : "";
        const sourceFailureText = sourceFailures.length
          ? ` Fontes indisponiveis: ${sourceFailures.slice(0, 3).map((item) => `${display(item.url)} (${display(item.reason)})`).join("; ")}${sourceFailures.length > 3 ? "..." : ""}`
          : "";
        appendRagVectorProgress(
          result.materialized_base
            ? "Base vetorial materializada automaticamente."
            : "Base vetorial ativa encontrada e reutilizada.",
          "done"
        );
        appendRagVectorProgress(
          sourceSummary.attempted
            ? `Resultado das fontes: ${display(sourceSummary.references || sourceSummary.attempted)} referencias de URL, ${display(sourceSummary.attempted)} URLs consultadas${sourceDedupText}, ${display(sourceSummary.succeeded)} recuperadas, ${display(sourceSummary.failed)} falhas, ${display(sourceSummary.inserted_chunks)} chunks de fonte inseridos.`
            : "Nenhuma URL de fonte encontrada para consulta.",
          sourceSummary.failed ? "error" : "done"
        );
        if (sourceFailures.length) {
          for (const failure of sourceFailures.slice(0, 5)) {
            appendRagVectorProgress(`Fonte indisponivel: ${display(failure.url)} (${display(failure.reason)}).`, "error");
          }
          if (sourceFailures.length > 5) {
            appendRagVectorProgress(`${display(sourceFailures.length - 5)} falhas de fonte adicionais omitidas na tela.`, "error");
          }
        }
        if (summary) {
          appendRagVectorProgress(
            chunkOriginSummary
              ? `Embeddings gravados: ${display(summary.generated_embeddings)} chunks no total (${chunkOriginSummary}) com ${display(summary.embedding_model)}.`
              : `Embeddings gravados: ${display(summary.generated_embeddings)} chunks com ${display(summary.embedding_model)}.`,
            "done"
          );
        }
        await loadRagCurationDataset(dataset);
        await loadRagVectorPreview(dataset);
        setText(
          "rag_vector_status",
          summary ? [
            result.materialized_base ? "Base vetorial criada automaticamente." : "",
            sourceSummary.attempted ? `Fontes consultadas: ${display(sourceSummary.attempted)} URLs unicas${sourceDedupText}, ${display(sourceSummary.succeeded)} recuperadas, ${display(sourceSummary.failed)} falhas, ${display(sourceSummary.inserted_chunks)} chunks de fonte/URL inseridos.` : "",
            chunkOriginSummary
              ? `Embeddings gerados para ${display(dataset)}: ${display(summary.generated_embeddings)} chunks no total (${chunkOriginSummary}) com ${display(summary.embedding_model)}.`
              : `Embeddings gerados para ${display(dataset)}: ${display(summary.generated_embeddings)} chunks com ${display(summary.embedding_model)}.`
          ].filter(Boolean).join(" ") + sourceFailureText : `Embeddings gerados para ${display(dataset)}.`
        );
      } catch (error) {
        appendRagVectorProgress(friendlyErrorMessage(error.message), "error");
        setText("rag_vector_status", friendlyErrorMessage(error.message));
      } finally {
        generateButton.disabled = false;
      }
    }

    async function activateRagVectorRun(runId) {
      const dataset = value("rag_vector_dataset");
      if (!dataset) {
        setText("rag_vector_status", "Selecione um dataset para ativar uma run vetorial.");
        return;
      }
      try {
        setText("rag_vector_status", `Ativando run vetorial #${display(runId)}...`);
        await postJson(`/api/rag-vector/runs/${encodeURIComponent(runId)}/activate?dataset=${encodeURIComponent(dataset)}`, {});
        await loadRagCurationDataset(dataset);
        await loadRagVectorPreview(dataset);
        setText("rag_vector_status", `Run vetorial #${display(runId)} ativada para ${display(dataset)}.`);
      } catch (error) {
        setText("rag_vector_status", friendlyErrorMessage(error.message));
      }
    }

    async function deleteRagVectorRun(runId, retrievalName) {
      const dataset = value("rag_vector_dataset");
      if (!dataset) {
        setText("rag_vector_status", "Selecione um dataset para excluir uma run vetorial.");
        return;
      }
      if (!confirmAction(`Excluir a run vetorial ${display(retrievalName)} (#${display(runId)})?`)) return;
      try {
        setText("rag_vector_status", `Excluindo run vetorial #${display(runId)}...`);
        const response = await fetch(`/api/rag-vector/runs/${encodeURIComponent(runId)}?dataset=${encodeURIComponent(dataset)}`, {
          method: "DELETE",
          headers: {"x-csrf-token": csrfToken},
        });
        const parsed = await readJsonResponse(response, "Falha ao excluir a run vetorial.");
        if (!response.ok) throw new Error(parsed.data?.detail || parsed.responseText || "Falha ao excluir a run vetorial.");
        await loadRagCurationDataset(dataset);
        await loadRagVectorPreview(dataset);
        setText("rag_vector_status", `Run vetorial #${display(runId)} excluida de ${display(dataset)}.`);
      } catch (error) {
        setText("rag_vector_status", friendlyErrorMessage(error.message));
      }
    }

    async function searchRagVector() {
      const dataset = value("rag_query_dataset");
      const queryText = value("rag_query_text").trim();
      const topK = Number(value("rag_query_top_k") || 5);
      if (!dataset) {
        setText("rag_query_status", "Selecione um dataset para consultar.");
        return;
      }
      if (!queryText) {
        setText("rag_query_status", "Informe um texto para buscar na base vetorial.");
        return;
      }
      try {
        setText("rag_query_status", `Consultando a base vetorial de ${display(dataset)}...`);
        const data = await postJson("/api/rag-vector/search", {
          dataset,
          query_text: queryText,
          top_k: topK,
        });
        renderRagQueryResults(data);
        setText("rag_query_status", `Consulta concluida para ${display(dataset)} com top-k ${display(topK)}.`);
      } catch (error) {
        renderRagQueryResults(null);
        setText("rag_query_status", friendlyErrorMessage(error.message));
      }
    }

    async function loadRagEmbeddingConfig(dataset = value("rag_embedding_dataset")) {
      if (!dataset) {
        renderRagEmbeddingSmokeResult(null);
        renderRagEmbeddingConfig(null);
        setText("rag_embedding_status", "Selecione um dataset para configurar o embedding.");
        return;
      }
      syncRagDatasetSelects(dataset);
      setText("rag_embedding_status", "Carregando configuracao de embedding...");
      renderRagEmbeddingSmokeResult(null);
      try {
        const response = await fetch(`/api/rag-embedding-config?dataset=${encodeURIComponent(dataset)}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Falha ao carregar configuracao de embedding.");
        ragEmbeddingConfigLoaded = true;
        renderRagEmbeddingConfig(data.record || null);
        setText(
          "rag_embedding_status",
          data.record
            ? `Configuracao de embedding de ${display(dataset)} carregada.`
            : `Nenhuma configuracao de embedding encontrada para ${display(dataset)}.`
        );
      } catch (error) {
        renderRagEmbeddingConfig(null);
        setText("rag_embedding_status", friendlyErrorMessage(error.message));
      }
    }

    function renderRagEmbeddingConfig(record) {
      if (!record) {
        setText("rag_embedding_meta", "Nenhuma configuracao carregada.");
        setText("rag_embedding_provider_summary", "-");
        setText("rag_embedding_model_summary", "-");
        setText("rag_embedding_dimensions_summary", "-");
        setText("rag_embedding_api_base_summary", "-");
        setText("rag_embedding_updated_by_summary", "-");
        setText("rag_embedding_updated_at_summary", "-");
        document.getElementById("rag_embedding_provider").value = "";
        document.getElementById("rag_embedding_model_name").value = "";
        document.getElementById("rag_embedding_dimensions").value = "";
        document.getElementById("rag_embedding_api_base_url").value = "";
        document.getElementById("rag_embedding_notes").value = "";
        return;
      }
      setText("rag_embedding_meta", `${display(record.dataset)} | ${display(record.dataset_name)} | config ${display(record.config_id || "default")}`);
      setText("rag_embedding_provider_summary", display(record.provider));
      setText("rag_embedding_model_summary", display(record.model_name));
      setText("rag_embedding_dimensions_summary", display(record.dimensions));
      setText("rag_embedding_api_base_summary", display(record.api_base_url));
      setText("rag_embedding_updated_by_summary", display(record.updated_by));
      setText("rag_embedding_updated_at_summary", formatDateTime(record.updated_at));
      document.getElementById("rag_embedding_provider").value = record.provider || "";
      document.getElementById("rag_embedding_model_name").value = record.model_name || "";
      document.getElementById("rag_embedding_dimensions").value = record.dimensions || "";
      document.getElementById("rag_embedding_api_base_url").value = record.api_base_url || "";
      document.getElementById("rag_embedding_notes").value = record.notes || "";
    }

    function renderRagEmbeddingSmokeResult(result) {
      if (!result) {
        setText("rag_embedding_test_meta", "Nenhum teste executado.");
        setText("rag_embedding_test_endpoint", "-");
        setText("rag_embedding_test_latency", "-");
        setText("rag_embedding_test_dimensions", "-");
        setText("rag_embedding_test_time", "-");
        return;
      }
      setText(
        "rag_embedding_test_meta",
        `${display(result.dataset)} | ${display(result.provider)} | ${display(result.model_name)}`
      );
      setText("rag_embedding_test_endpoint", display(result.endpoint_host || result.endpoint_url));
      setText(
        "rag_embedding_test_latency",
        result.latency_ms !== null && result.latency_ms !== undefined ? `${result.latency_ms} ms` : "-"
      );
      setText("rag_embedding_test_dimensions", display(result.returned_dimensions));
      setText("rag_embedding_test_time", formatDateTime(result.tested_at));
    }

    async function saveRagEmbeddingConfig() {
      const dataset = value("rag_embedding_dataset");
      const updatedBy = value("rag_embedding_updated_by").trim();
      if (!dataset) {
        setText("rag_embedding_status", "Selecione um dataset para configurar o embedding.");
        return;
      }
      if (!updatedBy) {
        setText("rag_embedding_status", "Informe quem alterou a configuracao do embedding.");
        return;
      }
      const rawDimensions = value("rag_embedding_dimensions").trim();
      const dimensions = rawDimensions ? Number(rawDimensions) : null;
      try {
        setText("rag_embedding_status", `Salvando configuracao de embedding para ${display(dataset)}...`);
        const data = await putJson("/api/rag-embedding-config", {
          dataset,
          provider: value("rag_embedding_provider").trim(),
          model_name: value("rag_embedding_model_name").trim(),
          dimensions: dimensions,
          api_base_url: value("rag_embedding_api_base_url").trim() || null,
          notes: value("rag_embedding_notes").trim() || null,
          updated_by: updatedBy,
        });
        renderRagEmbeddingConfig(data.record || null);
        setText("rag_embedding_status", `Configuracao de embedding salva para ${display(dataset)}.`);
      } catch (error) {
        setText("rag_embedding_status", friendlyErrorMessage(error.message));
      }
    }

    async function testRagEmbeddingConfig() {
      const dataset = value("rag_embedding_dataset");
      if (!dataset) {
        setText("rag_embedding_status", "Selecione um dataset para testar o embedding.");
        return;
      }
      try {
        setText("rag_embedding_status", `Testando API de embedding para ${display(dataset)}...`);
        renderRagEmbeddingSmokeResult(null);
        const data = await postJson("/api/rag-embedding-config/smoke-test", {
          dataset,
        });
        renderRagEmbeddingSmokeResult(data.result || null);
        const dims = data.result?.returned_dimensions;
        setText(
          "rag_embedding_status",
          `Smoke test concluido para ${display(dataset)}${dims ? ` (${dims} dimensoes).` : "."}`
        );
      } catch (error) {
        renderRagEmbeddingSmokeResult(null);
        setText("rag_embedding_status", friendlyErrorMessage(error.message));
      }
    }

    function renderRagCurationRuns(rows) {
      const body = document.getElementById("rag_curation_runs_body");
      body.textContent = "";
      if (!rows.length) {
        appendTableMessage(body, 8, "Nenhuma importacao registrada para esse dataset.");
        return;
      }
      rows.forEach((entry) => {
        const row = document.createElement("tr");
        appendCell(row, display(entry.run_id));
        appendCell(row, formatDateTime(entry.imported_at));
        appendCell(row, display(entry.imported_by));
        appendCell(row, display(entry.filename));
        appendCell(row, display(entry.item_count));
        appendCell(row, display(entry.article_count));
        appendCell(row, entry.active ? "ativa" : "historica");
        const actionCell = document.createElement("td");
        if (entry.active) {
          actionCell.textContent = "Ativa";
        } else {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "secondary";
          button.textContent = "Ativar";
          button.onclick = () => activateRagCurationRun(entry.run_id);
          actionCell.appendChild(button);
        }
        row.appendChild(actionCell);
        body.appendChild(row);
      });
    }

    function renderRagCurationItems(rows) {
      const body = document.getElementById("rag_curation_items_body");
      body.textContent = "";
      if (!rows.length) {
        appendTableMessage(body, 9, "Nenhuma questao curada disponivel na versao ativa.");
        return;
      }
      rows.forEach((entry) => {
        const row = document.createElement("tr");
        const detailCell = document.createElement("td");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "detail-button";
        button.textContent = "Ver";
        button.onclick = () => loadRagCurationDetail(entry.curation_id);
        detailCell.appendChild(button);
        row.appendChild(detailCell);
        appendCell(row, display(entry.question_sequence));
        appendCell(row, display(entry.question_id));
        appendCell(row, display(entry.question_type));
        appendCell(row, display(entry.discipline));
        appendCell(row, display(entry.theme));
        appendCell(row, display(entry.curator));
        appendCell(row, display(entry.primary_norma));
        appendCell(row, display(entry.article_count));
        body.appendChild(row);
      });
    }

    async function loadRagCurationDetail(curationId) {
      const dataset = value("rag_curation_dataset");
      if (!dataset || !curationId) {
        renderRagCurationDetail(null);
        return;
      }
      try {
        const response = await fetch(`/api/rag-curation/items/${encodeURIComponent(curationId)}?dataset=${encodeURIComponent(dataset)}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Falha ao carregar detalhe da curadoria.");
        renderRagCurationDetail(data.detail || null);
      } catch (error) {
        renderRagCurationDetail(null);
        setText("rag_curation_status", friendlyErrorMessage(error.message));
      }
    }

    function renderRagCurationDetail(detail) {
      if (!detail) {
        setText("rag_curation_detail_meta", "Selecione uma questao curada para ver os detalhes.");
        setText("rag_curation_detail_type", "-");
        setText("rag_curation_detail_discipline", "-");
        setText("rag_curation_detail_subject", "-");
        setText("rag_curation_detail_theme", "-");
        setText("rag_curation_detail_curator", "-");
        setText("rag_curation_detail_classified_at", "-");
        setText("rag_curation_detail_lei_norma", "-");
        setText("rag_curation_detail_url", "-");
        setText("rag_curation_detail_urn", "-");
        setText("rag_curation_detail_question", "Selecione uma questao curada para ver o enunciado.");
        setText("rag_curation_detail_answer_key", "Selecione uma questao curada para ver o gabarito.");
        setText("rag_curation_detail_metadata", "Selecione uma questao curada para ver os metadados.");
        setText("rag_curation_detail_raw_payload", "Selecione uma questao curada para ver o payload original.");
        renderRagCurationArticles([]);
        return;
      }
      setText(
        "rag_curation_detail_meta",
        `Dataset ${display(detail.dataset)} | curadoria ${display(detail.curation_id)} | questao ${display(detail.question_sequence)} | id_pergunta ${display(detail.question_id)}`
      );
      setText("rag_curation_detail_type", display(detail.question_type));
      setText("rag_curation_detail_discipline", display(detail.discipline));
      setText("rag_curation_detail_subject", display(detail.subject));
      setText("rag_curation_detail_theme", display(detail.theme));
      setText("rag_curation_detail_curator", display(detail.curator));
      setText("rag_curation_detail_classified_at", formatDateTime(detail.classified_at));
      setText(
        "rag_curation_detail_lei_norma",
        `${detail.lei ? `${detail.lei} | ` : ""}${display(detail.norma)}`
      );
      setText("rag_curation_detail_url", display(detail.url));
      setText("rag_curation_detail_urn", display(detail.urn));
      setText("rag_curation_detail_question", detail.question_text || "-");
      setText("rag_curation_detail_answer_key", formatStructuredValue(detail.answer_key));
      setText("rag_curation_detail_metadata", formatStructuredValue(detail.metadata));
      setText("rag_curation_detail_raw_payload", formatStructuredValue(detail.raw_payload));
      renderRagCurationArticles(detail.articles || []);
    }

    function renderRagCurationArticles(rows) {
      const body = document.getElementById("rag_curation_articles_body");
      body.textContent = "";
      if (!rows.length) {
        appendTableMessage(body, 5, "Nenhum artigo curado disponivel.");
        return;
      }
      rows.forEach((entry) => {
        const row = document.createElement("tr");
        appendCell(row, display(entry.ordem));
        appendCell(row, display(entry.artigo));
        appendCell(row, display(entry.topico));
        appendCell(row, display(entry.relevancia));
        appendCell(row, display(entry.tipo));
        body.appendChild(row);
      });
    }

    function formatStructuredValue(value) {
      if (value === null || value === undefined || value === "") return "-";
      if (typeof value === "string") return value;
      try {
        return JSON.stringify(value, null, 2);
      } catch (error) {
        return String(value);
      }
    }

    async function postRagCurationFile(file, importedBy, retryOnCsrf = true) {
      if (!csrfToken) await loadConfig();
      const response = await fetch("/api/rag-curation/import", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-csrf-token": csrfToken,
          "x-curation-filename": file.name,
          "x-curation-imported-by": importedBy,
        },
        body: await file.arrayBuffer(),
      });
      const data = await response.json();
      if (response.status === 403 && data.detail === "Invalid CSRF token." && retryOnCsrf) {
        await loadConfig();
        return postRagCurationFile(file, importedBy, false);
      }
      if (!response.ok) throw new Error(data.detail || "Falha ao importar curadoria.");
      return data;
    }

    function clearRagCurationFileSelection() {
      const input = document.getElementById("rag_curation_file");
      input.value = "";
    }

    async function importRagCurationFile() {
      const file = document.getElementById("rag_curation_file").files?.[0];
      const importedBy = value("rag_curation_imported_by").trim();
      if (!importedBy) {
        setText("rag_curation_status", "Informe quem esta importando a curadoria.");
        return;
      }
      if (!file) {
        setText("rag_curation_status", "Selecione um arquivo JSON de curadoria.");
        return;
      }
      try {
        setText("rag_curation_status", `Importando ${file.name}...`);
        const data = await postRagCurationFile(file, importedBy);
        await loadRagCurationOptions(data.dataset || value("rag_curation_dataset"));
        setText(
          "rag_curation_status",
          data.action === "activated_existing"
            ? `JSON ja conhecido. Run existente reativada para ${display(data.dataset)}.`
            : `Curadoria importada com sucesso para ${display(data.dataset)} a partir de ${display(data.run?.filename)}.`
        );
        clearRagCurationFileSelection();
      } catch (error) {
        setText("rag_curation_status", friendlyErrorMessage(error.message));
      }
    }

    async function activateRagCurationRun(runId) {
      const dataset = value("rag_curation_dataset");
      if (!dataset || !runId) return;
      try {
        setText("rag_curation_status", `Ativando run ${display(runId)}...`);
        const response = await fetch(`/api/rag-curation/runs/${encodeURIComponent(runId)}/activate?dataset=${encodeURIComponent(dataset)}`, {
          method: "POST",
          headers: {"x-csrf-token": csrfToken},
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Falha ao ativar a run de curadoria.");
        await loadRagCurationOptions(dataset);
        setText("rag_curation_status", `Run ${display(runId)} ativada para ${display(dataset)}.`);
      } catch (error) {
        setText("rag_curation_status", friendlyErrorMessage(error.message));
      }
    }

    async function loadMetaEvaluation() {
      const evaluationId = getMetaEvaluationId();
      if (!evaluationId) {
        renderMetaEvaluationState(null, []);
        setText("meta_status", "Nenhuma avaliacao J1 disponivel para meta-avaliacao.");
        return;
      }
      if (!isValidMetaEvaluationId(evaluationId)) {
        renderMetaEvaluationState(null, []);
        setText("meta_status", "Digite um id numerico de avaliacao J1 valido.");
        return;
      }
      setText("meta_status", "Carregando avaliacao...");
      try {
        const response = await fetch(`/api/meta-evaluations?evaluation_id=${encodeURIComponent(evaluationId)}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Falha ao carregar meta-avaliacao.");
        renderMetaEvaluationState(data.subject, data.records || []);
        renderMetaOperationalLoading(data.subject);
        getOperationalLogSummary().then((summary) => renderMetaOperationalMetadata(data.subject, summary));
        if (data.subject) {
          setText(
            "meta_status",
            `Avaliacao ${display(data.subject.evaluation_id)} carregada para meta-avaliacao.`
          );
        } else {
          setText("meta_status", "Avaliacao nao encontrada.");
        }
      } catch (error) {
        setText("meta_status", friendlyErrorMessage(error.message));
        renderMetaOperationalMetadata(null, null);
      }
    }

    async function saveMetaEvaluation() {
      const evaluatorName = value("meta_evaluator_name").trim();
      const rationale = value("meta_rationale").trim();
      const evaluationIdValue = getMetaEvaluationId();
      const evaluationId = Number(evaluationIdValue);
      const metaEvaluationId = value("meta_editing_id") ? Number(value("meta_editing_id")) : null;
      if (!evaluationIdValue) {
        setText("meta_status", "Selecione uma avaliacao J1.");
        return;
      }
      if (!isValidMetaEvaluationId(evaluationIdValue)) {
        setText("meta_status", "Digite um id numerico de avaliacao J1 valido.");
        return;
      }
      if (!evaluatorName) {
        setText("meta_status", "Informe o nome do avaliador.");
        return;
      }
      if (!rationale) {
        setText("meta_status", "Informe a justificativa da meta-avaliacao.");
        return;
      }
      try {
        const data = await putJson("/api/meta-evaluations", {
          meta_evaluation_id: metaEvaluationId,
          evaluation_id: evaluationId,
          evaluator_name: evaluatorName,
          score: Number(value("meta_score")),
          rationale: rationale
        });
        await loadMetaOptions();
        renderMetaEvaluationState(data.subject, data.records || []);
        resetMetaForm();
        if (metaHistoryLoaded) await loadMetaHistory();
        setText(
          "meta_status",
          data.action === "updated"
            ? `Meta-avaliacao atualizada em ${formatDateTime(data.record?.created_at)}.`
            : `Meta-avaliacao registrada com nota ${display(data.record?.score)} em ${formatDateTime(data.record?.created_at)}.`
        );
      } catch (error) {
        setText("meta_status", friendlyErrorMessage(error.message));
      }
    }

    function renderMetaEvaluationState(subject, records) {
      renderMetaSubject(subject);
      renderMetaRecords(records || []);
    }

    function renderMetaSubject(subject) {
      currentMetaSubject = subject || null;
      if (!subject) {
        setText("meta_subject_meta", "Nenhuma avaliacao carregada.");
        setText("meta_subject_candidate_model", "-");
        setText("meta_subject_judge_model", "-");
        setText("meta_subject_judge_score", "-");
        setText("meta_subject_prompt_version", "-");
        setText("meta_subject_evaluated_at", "-");
        setText("meta_subject_question", "Selecione uma avaliacao para ver o enunciado.");
        setText("meta_subject_reference", "Selecione uma avaliacao para ver o gabarito.");
        setText("meta_subject_candidate_answer", "Selecione uma avaliacao para ver a resposta candidata.");
        setText("meta_subject_chain_of_thought", "Selecione uma avaliacao para ver o Chain of Thoughts.");
        renderMetaOperationalMetadata(null, null);
        return;
      }
      const promptLabel = subject.prompt_version
        ? `v${display(subject.prompt_version)}${subject.prompt_created_by ? ` | ${subject.prompt_created_by}` : ""}`
        : "-";
      setText(
        "meta_subject_meta",
        `Dataset ${display(subject.dataset)} | avaliacao ${display(subject.evaluation_id)} | questao ${display(subject.question_id)} | resposta ${display(subject.answer_id)}`
      );
      setText("meta_subject_candidate_model", display(subject.candidate_model));
      setText("meta_subject_judge_model", display(subject.judge_model));
      setText("meta_subject_judge_score", display(subject.judge_score));
      setText("meta_subject_prompt_version", promptLabel);
      setText("meta_subject_evaluated_at", formatDateTime(subject.evaluated_at));
      setText("meta_subject_question", subject.question_text || "-");
      setText("meta_subject_reference", subject.reference_answer || "-");
      setText("meta_subject_candidate_answer", subject.candidate_answer || "-");
      setText("meta_subject_chain_of_thought", subject.judge_chain_of_thought || "-");
    }

    function renderMetaOperationalMetadata(subject, summary) {
      if (operationalLogSummaryLoading && subject && !summary) {
        renderMetaOperationalLoading(subject);
        return;
      }
      const card = document.getElementById("meta_operational_card");
      if (!subject || !summary || !summary.available) {
        card.hidden = true;
        return;
      }
      const match = findOperationalEventForSubject(subject, summary);
      if (!match) {
        card.hidden = true;
        return;
      }
      card.hidden = false;
      renderMetaOperationalLogLink(match.log);
      setText("meta_operational_latency", formatLatency(match.event.latency_ms));
    }

    function renderMetaOperationalLoading(subject) {
      const card = document.getElementById("meta_operational_card");
      if (!subject) {
        card.hidden = true;
        return;
      }
      card.hidden = false;
      setText("meta_operational_run", "Carregando logs operacionais...");
      setText("meta_operational_latency", "Carregando...");
    }

    function renderMetaOperationalLogLink(log) {
      const cell = document.getElementById("meta_operational_run");
      cell.textContent = "";
      const runId = log.run_id;
      const logPath = log.log_path;
      if (!runId) {
        cell.textContent = display(logPath);
        return;
      }
      const button = document.createElement("button");
      button.type = "button";
      button.className = "detail-button";
      button.textContent = `${display(runId)} / ${display(logPath)}`;
      button.title = "Abrir em Execucoes anteriores";
      button.onclick = () => openHistoryLogFromMeta(runId, logPath);
      cell.appendChild(button);
    }

    function findOperationalEventForSubject(subject, summary) {
      for (const log of safeOperationalLogs(summary)) {
        for (const event of safeOperationalEvents(log)) {
          if (event.matched_evaluation_id && String(event.matched_evaluation_id) === String(subject.evaluation_id)) {
            return {log, event};
          }
        }
      }
      for (const log of safeOperationalLogs(summary)) {
        for (const event of safeOperationalEvents(log)) {
          const sameAnswer = event.answer_id && String(event.answer_id) === String(subject.answer_id);
          const sameJudge = !event.judge_model || !subject.judge_model || String(event.judge_model) === String(subject.judge_model);
          if (sameAnswer && sameJudge) return {log, event};
        }
      }
      return null;
    }

    function renderMetaRecords(records) {
      const body = document.getElementById("meta_records_body");
      body.textContent = "";
      if (!records.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 5;
        cell.className = "muted";
        cell.textContent = "Nenhuma meta-avaliacao registrada para essa avaliacao.";
        row.appendChild(cell);
        body.appendChild(row);
        return;
      }
      records.forEach((entry) => {
        const row = document.createElement("tr");
        appendCell(row, formatDateTime(entry.created_at));
        appendCell(row, display(entry.evaluator_name));
        appendCell(row, display(entry.score));
        appendCell(row, display(entry.rationale));
        const actionsCell = document.createElement("td");
        const editButton = document.createElement("button");
        editButton.type = "button";
        editButton.className = "secondary";
        editButton.textContent = "Editar";
        editButton.onclick = () => beginMetaEdit(entry);
        const deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "secondary";
        deleteButton.textContent = "Excluir";
        deleteButton.onclick = () => deleteMetaEvaluation(entry.meta_evaluation_id);
        actionsCell.appendChild(editButton);
        actionsCell.appendChild(document.createTextNode(" "));
        actionsCell.appendChild(deleteButton);
        row.appendChild(actionsCell);
        body.appendChild(row);
      });
    }

    function beginMetaEdit(entry) {
      document.getElementById("meta_editing_id").value = String(entry.meta_evaluation_id);
      document.getElementById("meta_evaluator_name").value = entry.evaluator_name || "";
      document.getElementById("meta_score").value = String(entry.score || 1);
      document.getElementById("meta_rationale").value = entry.rationale || "";
      document.getElementById("meta_save").textContent = "Atualizar";
      document.getElementById("meta_cancel_edit").style.display = "";
      setText("meta_status", `Editando meta-avaliacao ${display(entry.meta_evaluation_id)}.`);
    }

    function resetMetaForm() {
      document.getElementById("meta_editing_id").value = "";
      document.getElementById("meta_rationale").value = "";
      document.getElementById("meta_score").value = "1";
      document.getElementById("meta_save").textContent = "Salvar";
      document.getElementById("meta_cancel_edit").style.display = "none";
    }

    async function deleteMetaEvaluation(metaEvaluationId) {
      const evaluationId = Number(getMetaEvaluationId());
      if (!evaluationId || !metaEvaluationId) return;
      if (!(await requestMetaDelete())) return;
      try {
        const response = await fetch(`/api/meta-evaluations/${metaEvaluationId}?evaluation_id=${encodeURIComponent(evaluationId)}`, {
          method: "DELETE",
          headers: {"x-csrf-token": csrfToken},
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Falha ao excluir meta-avaliacao.");
        await loadMetaOptions();
        renderMetaEvaluationState(data.subject, data.records || []);
        resetMetaForm();
        if (metaHistoryLoaded) await loadMetaHistory();
        setText("meta_status", "Meta-avaliacao excluida.");
      } catch (error) {
        setText("meta_status", friendlyErrorMessage(error.message));
      }
    }

    function switchMetaMode(mode) {
      const isHistory = mode === "history";
      document.getElementById("meta_new_panel").hidden = isHistory;
      document.getElementById("meta_history_panel").hidden = !isHistory;
      for (const button of document.querySelectorAll("[data-meta-mode]")) {
        const active = button.dataset.metaMode === mode;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
      }
      if (isHistory && !metaHistoryLoaded) loadMetaHistory();
    }

    async function loadMetaHistory() {
      try {
        const response = await fetch("/api/meta-evaluations/history");
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Falha ao carregar auditorias.");
        metaHistoryRecords = data.records || [];
        metaHistoryLoaded = true;
        renderMetaHistoryEvaluatorOptions();
        renderMetaHistory();
      } catch (error) {
        metaHistoryLoaded = false;
        setText("meta_history_count", friendlyErrorMessage(error.message));
      }
    }

    function renderMetaHistoryEvaluatorOptions() {
      const select = document.getElementById("meta_history_evaluator");
      const current = select.value;
      const evaluators = Array.from(new Set(metaHistoryRecords.map((entry) => entry.evaluator_name).filter(Boolean)))
        .sort((a, b) => a.localeCompare(b));
      select.textContent = "";
      const allOption = document.createElement("option");
      allOption.value = "";
      allOption.textContent = "Todos";
      select.appendChild(allOption);
      for (const evaluator of evaluators) {
        const option = document.createElement("option");
        option.value = evaluator;
        option.textContent = evaluator;
        select.appendChild(option);
      }
      if (current && evaluators.includes(current)) select.value = current;
    }

    function sortedMetaHistoryRecords(records) {
      const direction = metaHistorySort.direction === "asc" ? 1 : -1;
      const key = metaHistorySort.key;
      return [...records].sort((left, right) => {
        const leftValue = left[key];
        const rightValue = right[key];
        const leftNumber = Number(leftValue);
        const rightNumber = Number(rightValue);
        if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) {
          return (leftNumber - rightNumber) * direction;
        }
        return String(leftValue || "").localeCompare(String(rightValue || "")) * direction;
      });
    }

    function currentMetaHistoryRows() {
      const evaluator = value("meta_history_evaluator");
      const filtered = evaluator
        ? metaHistoryRecords.filter((entry) => entry.evaluator_name === evaluator)
        : metaHistoryRecords;
      return sortedMetaHistoryRecords(filtered);
    }

    function renderMetaHistory() {
      metaHistoryFilteredRecords = currentMetaHistoryRows();
      if (
        selectedMetaHistoryId !== null
        && !metaHistoryFilteredRecords.some((entry) => entry.meta_evaluation_id === selectedMetaHistoryId)
      ) {
        selectedMetaHistoryId = null;
      }
      renderMetaHistoryTable();
      renderMetaHistoryDetail();
    }

    function renderMetaHistoryTable() {
      const body = document.getElementById("meta_history_body");
      body.textContent = "";
      const total = metaHistoryFilteredRecords.length;
      setText("meta_history_count", `${total} ${total === 1 ? "auditoria encontrada" : "auditorias encontradas"}`);
      if (!total) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 7;
        cell.className = "muted";
        cell.textContent = "Nenhuma auditoria encontrada para o filtro atual.";
        row.appendChild(cell);
        body.appendChild(row);
        return;
      }
      for (const entry of metaHistoryFilteredRecords) {
        const row = document.createElement("tr");
        row.className = "meta-history-row";
        if (entry.meta_evaluation_id === selectedMetaHistoryId) row.classList.add("selected");
        row.onclick = () => {
          selectedMetaHistoryId = entry.meta_evaluation_id;
          renderMetaHistory();
        };
        appendCell(row, formatDateTime(entry.created_at));
        appendCell(row, display(entry.evaluator_name));
        appendCell(row, display(entry.score));
        appendCell(row, display(entry.judge_model));
        appendCell(row, display(entry.judge_score));
        appendCell(row, display(entry.candidate_model));
        appendCell(row, display(entry.question_id));
        body.appendChild(row);
      }
    }

    function renderMetaHistoryDetail() {
      const entry = metaHistoryFilteredRecords.find((record) => record.meta_evaluation_id === selectedMetaHistoryId);
      const index = metaHistoryFilteredRecords.findIndex((record) => record.meta_evaluation_id === selectedMetaHistoryId);
      const previous = document.getElementById("meta_history_previous");
      const next = document.getElementById("meta_history_next");
      previous.disabled = index <= 0;
      next.disabled = index < 0 || index >= metaHistoryFilteredRecords.length - 1;
      if (!entry) {
        setText("meta_history_detail_meta", "Selecione uma auditoria para ver os detalhes.");
        setText("meta_history_detail_evaluator", "-");
        setText("meta_history_detail_score", "-");
        setText("meta_history_detail_rationale", "-");
        setText("meta_history_detail_candidate_model", "-");
        setText("meta_history_detail_judge_model", "-");
        setText("meta_history_detail_judge_score", "-");
        setText("meta_history_detail_question", "Selecione uma auditoria para ver o enunciado.");
        setText("meta_history_detail_reference", "Selecione uma auditoria para ver o gabarito.");
        setText("meta_history_detail_candidate_answer", "Selecione uma auditoria para ver a resposta candidata.");
        setText("meta_history_detail_chain_of_thought", "Selecione uma auditoria para ver o Chain of Thoughts.");
        return;
      }
      setText(
        "meta_history_detail_meta",
        `Dataset ${display(entry.dataset)} | auditoria ${display(entry.meta_evaluation_id)} | avaliacao ${display(entry.evaluation_id)} | questao ${display(entry.question_id)} | resposta ${display(entry.answer_id)}`
      );
      setText("meta_history_detail_evaluator", display(entry.evaluator_name));
      setText("meta_history_detail_score", display(entry.score));
      setText("meta_history_detail_rationale", display(entry.rationale));
      setText("meta_history_detail_candidate_model", display(entry.candidate_model));
      setText("meta_history_detail_judge_model", display(entry.judge_model));
      setText("meta_history_detail_judge_score", display(entry.judge_score));
      setText("meta_history_detail_question", entry.question_text || "-");
      setText("meta_history_detail_reference", entry.reference_answer || "-");
      setText("meta_history_detail_candidate_answer", entry.candidate_answer || "-");
      setText("meta_history_detail_chain_of_thought", entry.judge_chain_of_thought || "-");
    }

    function sortMetaHistory(key) {
      if (metaHistorySort.key === key) {
        metaHistorySort.direction = metaHistorySort.direction === "asc" ? "desc" : "asc";
      } else {
        metaHistorySort = {key, direction: "asc"};
      }
      renderMetaHistory();
    }

    function navigateMetaHistory(offset) {
      const currentIndex = metaHistoryFilteredRecords.findIndex((entry) => entry.meta_evaluation_id === selectedMetaHistoryId);
      const nextIndex = currentIndex + offset;
      if (nextIndex < 0 || nextIndex >= metaHistoryFilteredRecords.length) return;
      selectedMetaHistoryId = metaHistoryFilteredRecords[nextIndex].meta_evaluation_id;
      renderMetaHistory();
    }

    async function savePromptConfig() {
      const changedBy = value("prompt_changed_by").trim();
      if (!changedBy) {
        setText("prompt_status", "Informe quem esta alterando o prompt.");
        return;
      }
      try {
        const data = await putJson("/api/judge-prompts", {
          dataset: value("prompt_dataset"),
          prompt: value("prompt_body"),
          persona: value("prompt_persona"),
          context: value("prompt_context"),
          rubric: value("prompt_rubric"),
          output: value("prompt_output"),
          changed_by: changedBy
        });
        renderPromptLogs(data.versions || []);
        renderPromptPreview(data.preview);
        setText("prompt_status", `Nova versao salva: v${display(data.record?.version)} em ${formatDateTime(data.record?.created_at)}.`);
      } catch (error) {
        setText("prompt_status", friendlyErrorMessage(error.message));
      }
    }

    function renderPromptLogs(rows) {
      const body = document.getElementById("prompt_logs_body");
      body.textContent = "";
        if (!rows.length) {
          const row = document.createElement("tr");
          const cell = document.createElement("td");
          cell.colSpan = 9;
          cell.className = "muted";
          cell.textContent = "Nenhuma versao registrada.";
          row.appendChild(cell);
        body.appendChild(row);
        return;
      }
        rows.forEach((entry) => {
          const row = document.createElement("tr");
          for (const field of [
            `v${display(entry.version)}`,
            entry.active ? "ativa" : "historica",
            formatDateTime(entry.created_at),
            entry.created_by,
            `${entry.prompt_chars || 0} chars`,
            `${entry.persona_chars || 0} chars`,
            `${entry.context_chars || 0} chars`,
            `${entry.rubric_chars || 0} chars`,
            `${entry.output_chars || 0} chars`
          ]) appendCell(row, display(field));
          body.appendChild(row);
        });
      }

    function renderPromptPreview(preview) {
      if (!preview) {
        setText("prompt_preview_meta", "Nenhum preview disponivel.");
        setText("prompt_preview_content", "Nao foi possivel montar o prompt de exemplo.");
        return;
      }
      const versionLabel = preview.version ? ` | versao v${display(preview.version)}` : "";
      const meta = `Dataset ${display(preview.dataset)} | questao ${display(preview.question_id)} | resposta ${display(preview.answer_id)} | modelo ${display(preview.candidate_model)}${versionLabel}`;
      setText("prompt_preview_meta", meta);
      setText("prompt_preview_content", preview.rendered_prompt || "Salve a configuracao para visualizar o prompt montado.");
    }

    function renderHistory(rows) {
      const body = document.getElementById("history-table-body");
      body.textContent = "";
      window.latestRunHistoryRows = rows;
      if (!rows.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 10;
        cell.className = "muted";
        cell.textContent = "Nenhuma execucao encontrada.";
        row.appendChild(cell);
        body.appendChild(row);
        return;
      }
      rows.forEach((entry) => {
        const row = document.createElement("tr");
        row.className = "history-row";
        row.dataset.runId = entry.run_id || "";
        if (String(entry.run_id) === String(selectedHistoryRunId)) row.classList.add("selected");
        row.onclick = () => openHistoryLog(entry);
        for (const value of [
          entry.run_id,
          formatDateTime(entry.timestamp),
          entry.mode,
          entry.dataset,
          entry.batch_size,
          entry.successes,
          entry.failures,
          entry.duration
        ]) appendCell(row, display(value));
        const logCell = document.createElement("td");
        const openButton = document.createElement("button");
        openButton.type = "button";
        openButton.className = "detail-button";
        openButton.textContent = "Abrir";
        openButton.onclick = (event) => {
          event.stopPropagation();
          openHistoryLog(entry);
        };
        logCell.appendChild(openButton);
        row.appendChild(logCell);
        const exportCell = document.createElement("td");
        const links = document.createElement("span");
        links.className = "history-export-links";
        links.appendChild(historyExportLink("CSV", "/api/run-history/export.csv", "run-history.csv"));
        links.appendChild(historyExportLink("JSON", "/api/run-history/export.json", "run-history.json"));
        exportCell.appendChild(links);
        row.appendChild(exportCell);
        body.appendChild(row);
      });
    }

    function historyExportLink(label, href, filename) {
      const link = document.createElement("a");
      link.href = href;
      link.download = filename;
      link.textContent = label;
      return link;
    }

    async function openHistoryLog(entry) {
      selectedHistoryRunId = entry.run_id || null;
      highlightSelectedHistoryRow();
      setText("history-log-run-id", entry.run_id);
      setText("history-log-path", entry.log_path);
      setText("history-log-content", "Carregando log...");
      try {
        const response = await fetch(entry.log_url);
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.detail || "Log nao encontrado.");
        }
        setText("history-log-content", await response.text());
      } catch (error) {
        setText("history-log-content", friendlyErrorMessage(error.message));
      }
    }

    function highlightSelectedHistoryRow() {
      for (const row of document.querySelectorAll(".history-row")) {
        row.classList.toggle("selected", String(row.dataset.runId) === String(selectedHistoryRunId));
      }
    }

    async function openHistoryLogFromMeta(runId, logPath) {
      activateTab("history-panel");
      const rows = await getRunHistory();
      const entry = rows.find((item) => String(item.run_id) === String(runId))
        || {
          run_id: runId,
          log_path: logPath,
          log_url: `/api/run-history/${encodeURIComponent(runId)}/audit-log`,
        };
      await openHistoryLog(entry);
    }

    function getRunHistory() {
      if (!runHistoryPromise) {
        runHistoryPromise = loadHistoryRows();
      }
      return runHistoryPromise;
    }

    function prefetchRunHistory() {
      getRunHistory();
    }

    async function refreshRunHistory() {
      runHistoryPromise = loadHistoryRows();
      return runHistoryPromise;
    }

    async function loadHistoryRows() {
      runHistoryLoading = true;
      renderHistoryLoading();
      try {
        const response = await fetch("/api/run-history");
        const rows = await response.json();
        if (!response.ok) throw new Error(rows.detail || "Historico indisponivel.");
        renderHistory(rows);
        historyLoaded = true;
        return rows;
      } catch (error) {
        runHistoryPromise = null;
        renderHistoryError(friendlyErrorMessage(error.message));
        return [];
      } finally {
        runHistoryLoading = false;
      }
    }

    function renderHistoryLoading() {
      const body = document.getElementById("history-table-body");
      body.textContent = "";
      appendTableMessage(body, 10, "Carregando historico de execucoes.");
    }

    function renderHistoryError(message) {
      const body = document.getElementById("history-table-body");
      body.textContent = "";
      appendTableMessage(body, 10, message);
    }

    function activateTab(targetId) {
      for (const button of document.querySelectorAll(".tab-button")) {
        const active = button.dataset.tab === targetId;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
      }
      for (const panel of document.querySelectorAll(".tab-panel")) {
        panel.hidden = panel.id !== targetId;
      }
      if (targetId === "meta-panel" || targetId === "history-panel") {
        window.scrollTo({top: 0, left: 0, behavior: "auto"});
      }
    }

    function switchRagSubtab(targetId) {
      for (const button of document.querySelectorAll(".rag-subtab-button")) {
        const active = button.dataset.ragSubtab === targetId;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
      }
      for (const panel of document.querySelectorAll(".rag-subpanel")) {
        panel.hidden = panel.id !== targetId;
      }
      if (targetId === "rag-vector-subpanel") {
        loadRagCurationDataset(value("rag_vector_dataset") || value("rag_curation_dataset"));
      }
      if (targetId === "rag-query-subpanel") {
        loadRagVectorPreview(value("rag_query_dataset") || value("rag_curation_dataset"));
      }
      if (targetId === "rag-embedding-subpanel") {
        loadRagEmbeddingConfig(value("rag_embedding_dataset") || value("rag_curation_dataset"));
      }
    }

    function switchTab(targetId) {
      activateTab(targetId);
      if (targetId === "dashboard-panel") loadDashboard();
      if (targetId === "history-panel") loadHistory();
      if (targetId === "prompt-panel" && !promptOptionsLoaded) loadPromptOptions();
      if (targetId === "meta-panel" && !metaOptionsLoaded) loadMetaOptions();
      if (targetId === "rag-panel" && !ragCurationOptionsLoaded) loadRagCurationOptions();
    }

    async function loadConfig() {
      const config = await (await fetch("/api/config")).json();
      csrfToken = config.csrf_token;
      const defaults = config.defaults || {};
      judgeModelOptions = config.judge_model_options || [defaults.judge_model, defaults.secondary_judge_model, defaults.arbiter_judge_model].filter(Boolean);
      for (const key of ["judge_input_source", "panel_mode", "dataset", "batch_size", "judge_execution_strategy", "judge_arbitration_min_delta", "remote_judge_timeout_seconds", "remote_judge_temperature", "remote_judge_max_tokens", "remote_judge_top_p"]) {
        if (defaults[key] !== null && defaults[key] !== undefined) document.getElementById(key).value = defaults[key];
      }
      for (const key of ["judge_model", "secondary_judge_model", "arbiter_judge_model"]) {
        const model = defaults[key];
        if (model && !judgeModelOptions.includes(model)) judgeModelOptions.push(model);
        if (model) document.getElementById(key).dataset.defaultValue = model;
      }
      renderJudgeModelSelects();
      for (const key of ["judge_model", "secondary_judge_model", "arbiter_judge_model"]) {
        const model = document.getElementById(key).dataset.defaultValue;
        if (model) document.getElementById(key).value = model;
      }
      renderJudgeModelSelects();
      document.getElementById("always_run_arbiter").checked = false;
      document.getElementById("judge_save_raw_response").checked = Boolean(defaults.judge_save_raw_response);
      document.getElementById("remote_judge_openai_compatible").value = String(Boolean(defaults.remote_judge_openai_compatible));
      setText("config-status", config.configuration_error || `Endpoints: juiz 1 ${config.endpoints?.JUDGE?.host || "-"} / juiz 2 ${config.endpoints?.SECONDARY_JUDGE?.host || "-"}`);
      setText("command-preview", config.command_preview || "");
      const presetRoot = document.getElementById("presets");
      for (const preset of config.presets || []) {
        const btn = document.createElement("button");
        btn.className = "secondary";
        btn.textContent = preset.name;
        btn.onclick = () => {
          for (const [key, val] of Object.entries(preset)) {
            if (key === "name") continue;
            if (key === "always_run_arbiter") document.getElementById(key).checked = Boolean(val);
            else document.getElementById(key).value = val;
          }
          renderJudgeInputSource();
          renderJudgeBlocks();
        };
        presetRoot.appendChild(btn);
      }
      renderJudgeInputSource();
      renderJudgeBlocks();
      renderEndpointFields();
      document.getElementById("dry-run").disabled = false;
      document.getElementById("run").disabled = false;
      await loadDashboard();
    }

    function populateSelect(id, options, valueKey, labelKey) {
      const select = document.getElementById(id);
      const current = select.value;
      const selectedSet = new Set(Array.isArray(valueKey) ? valueKey : []);
      select.textContent = "";
      for (const optionData of options) {
        const option = document.createElement("option");
        if (optionData && typeof optionData === "object" && !Array.isArray(optionData)) {
          option.value = optionData[valueKey];
          option.textContent = optionData[labelKey];
        } else {
          option.value = optionData;
          option.textContent = optionData;
          option.selected = selectedSet.has(optionData);
        }
        select.appendChild(option);
      }
      if (current && Array.from(select.options).some((option) => option.value === current)) {
        select.value = current;
      }
    }

    function isValidMetaEvaluationId(value) {
      return /^[1-9]\\d*$/.test(String(value).trim());
    }

    function getMetaEvaluationId() {
      const typedValue = value("meta_evaluation_select").trim();
      if (selectedMetaEvaluationId) {
        const selectedOption = metaEvaluationOptions.find((option) => String(option.value) === String(selectedMetaEvaluationId));
        const selectedLabel = selectedOption ? String(selectedOption.label || selectedOption.value) : "";
        if (typedValue === String(selectedMetaEvaluationId) || typedValue === selectedLabel) return String(selectedMetaEvaluationId);
      }
      return typedValue;
    }

    function selectMetaEvaluationById(evaluationId) {
      const option = metaEvaluationOptions.find((item) => String(item.value) === String(evaluationId));
      selectedMetaEvaluationId = String(evaluationId);
      document.getElementById("meta_evaluation_select").value = option ? String(option.label || option.value) : String(evaluationId);
    }

    function clearMetaEvaluationSelection() {
      selectedMetaEvaluationId = "";
      document.getElementById("meta_evaluation_select").value = "";
      document.getElementById("meta_evaluation_options").hidden = true;
      resetMetaForm();
      renderMetaEvaluationState(null, []);
      setText("meta_status", "Selecione uma avaliacao J1 para iniciar a meta-avaliacao.");
    }

    function renderMetaEvaluationOptions(show = true) {
      const list = document.getElementById("meta_evaluation_options");
      const query = value("meta_evaluation_select").trim().toLowerCase();
      const matches = metaEvaluationOptions.filter((option) => {
        const optionValue = String(option.value || "");
        const optionLabel = String(option.label || "");
        return !query || optionValue.includes(query) || optionLabel.toLowerCase().includes(query);
      }).slice(0, 40);
      list.textContent = "";
      if (!matches.length) {
        const empty = document.createElement("div");
        empty.className = "meta-option-empty";
        empty.textContent = "Nenhuma avaliacao encontrada.";
        list.appendChild(empty);
        list.hidden = !show;
        return;
      }
      for (const option of matches) {
        const button = document.createElement("button");
        button.className = "meta-option";
        button.type = "button";
        button.setAttribute("role", "option");
        button.dataset.value = option.value;
        button.textContent = option.label || option.value;
        button.onclick = async () => {
          selectMetaEvaluationById(option.value);
          list.hidden = true;
          await loadMetaEvaluation();
        };
        list.appendChild(button);
      }
      list.hidden = !show;
    }

    async function putJson(url, body) {
      const response = await fetch(url, {
        method: "PUT",
        headers: {"content-type": "application/json", "x-csrf-token": csrfToken},
        body: JSON.stringify(body)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Request failed");
      return data;
    }

    function renderEndpointFields() {
      for (const name of ["judge", "secondary", "arbiter"]) {
        document.getElementById(`endpoint_fields_${name}`).hidden = value(`endpoint_source_${name}`) !== "custom";
      }
    }

    for (const id of ["endpoint_source_judge", "endpoint_source_secondary", "endpoint_source_arbiter"]) {
      document.getElementById(id).onchange = () => {
        renderEndpointFields();
      };
    }
    document.getElementById("panel_mode").onchange = renderJudgeBlocks;
    document.getElementById("judge_input_source").onchange = renderJudgeInputSource;
    for (const id of ["judge_model", "secondary_judge_model", "arbiter_judge_model"]) {
      document.getElementById(id).onchange = renderJudgeModelSelects;
    }

    for (const button of document.querySelectorAll("[data-toggle-secret]")) {
      button.onclick = () => {
        const input = document.getElementById(button.dataset.toggleSecret);
        const showing = input.type === "text";
        input.type = showing ? "password" : "text";
        button.textContent = showing ? "◉" : "◎";
        button.setAttribute("aria-pressed", String(!showing));
      };
    }
    document.getElementById("details-close").onclick = () => document.getElementById("details-dialog").close();
    document.getElementById("audit-log-close").onclick = () => document.getElementById("audit-log-dialog").close();
    document.getElementById("database-dump-dialog-close").onclick = () => document.getElementById("database-dump-dialog").close();
    function showDatabaseDumpDialog(data) {
      const downloadLink = document.getElementById("database-dump-download");
      setText("database-dump-filename", data.filename || "-");
      setText("database-dump-path", data.path || "-");
      setText("database-dump-size", `${Math.round((data.size_bytes || 0) / 1024)} KB`);
      if (data.download_url) {
        downloadLink.href = data.download_url;
        downloadLink.download = data.filename || "atividade_2_backup.sql";
        downloadLink.hidden = false;
      } else {
        downloadLink.hidden = true;
        downloadLink.removeAttribute("href");
        downloadLink.removeAttribute("download");
      }
      document.getElementById("database-dump-dialog").showModal();
    }
    function extractDownloadFilename(response, fallbackFilename) {
      const disposition = response.headers.get("content-disposition") || "";
      const match = disposition.match(/filename=\"?([^\";]+)\"?/i);
      return match?.[1] || fallbackFilename || "atividade_2_backup.sql";
    }
    async function downloadDatabaseDump(data) {
      const response = await fetch(data.download_url, {method: "GET", credentials: "same-origin"});
      if (!response.ok) {
        const detail = await response.text();
        let message = detail || "Falha ao baixar o backup.";
        try {
          const parsed = JSON.parse(detail);
          message = parsed.detail || message;
        } catch (error) {
          // Keep the raw response body when the error is not JSON.
        }
        throw new Error(message);
      }
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = extractDownloadFilename(response, data.filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
    }
    async function handleDatabaseDumpResult(data) {
      showDatabaseDumpDialog(data);
      if (data.delivery === "browser_download" && data.download_url) {
        await downloadDatabaseDump(data);
      }
    }
    function confirmDatabaseClean() {
      const dialog = document.getElementById("database-clean-dialog");
      return new Promise((resolve) => {
        const cancel = document.getElementById("database-clean-cancel");
        const confirm = document.getElementById("database-clean-confirm");
        let settled = false;
        const cleanup = (action) => {
          if (settled) return;
          settled = true;
          cancel.onclick = null;
          confirm.onclick = null;
          dialog.oncancel = null;
          dialog.onclose = null;
          if (dialog.open) dialog.close();
          resolve(action);
        };
        cancel.onclick = () => cleanup("cancel");
        confirm.onclick = () => cleanup("clean");
        dialog.oncancel = (event) => {
          event.preventDefault();
          cleanup("cancel");
        };
        dialog.onclose = () => cleanup("cancel");
        dialog.showModal();
      });
    }
    function requestMetaDelete() {
      const dialog = document.getElementById("meta-delete-dialog");
      return new Promise((resolve) => {
        const cancel = document.getElementById("meta-delete-cancel");
        const deleteButton = document.getElementById("meta-delete-confirm");
        let settled = false;
        const cleanup = (accepted) => {
          if (settled) return;
          settled = true;
          cancel.onclick = null;
          deleteButton.onclick = null;
          dialog.oncancel = null;
          dialog.onclose = null;
          if (dialog.open) dialog.close();
          resolve(accepted);
        };
        cancel.onclick = () => cleanup(false);
        deleteButton.onclick = () => cleanup(true);
        dialog.oncancel = (event) => {
          event.preventDefault();
          cleanup(false);
        };
        dialog.onclose = () => cleanup(false);
        dialog.showModal();
      });
    }
    for (const button of document.querySelectorAll(".tab-button")) {
      button.onclick = () => switchTab(button.dataset.tab);
    }
    for (const button of document.querySelectorAll(".rag-subtab-button")) {
      button.onclick = () => switchRagSubtab(button.dataset.ragSubtab);
    }
    document.getElementById("assistant-chat-toggle").onclick = () => toggleAssistantChat();
    document.getElementById("assistant-chat-close").onclick = () => toggleAssistantChat(false);
    document.getElementById("assistant-chat-form").onsubmit = submitAssistantMessage;
    document.getElementById("prompt_dataset").onchange = () => loadPromptConfig();
    document.getElementById("prompt_reload").onclick = () => loadPromptConfig();
    document.getElementById("prompt_save").onclick = () => savePromptConfig();
    document.getElementById("rag_curation_dataset").onchange = () => loadRagCurationDataset(value("rag_curation_dataset"));
    document.getElementById("rag_curation_reload").onclick = () => loadRagCurationDataset(value("rag_curation_dataset"));
    document.getElementById("rag_vector_dataset").onchange = () => loadRagCurationDataset(value("rag_vector_dataset"));
    document.getElementById("rag_vector_reload").onclick = () => loadRagCurationDataset(value("rag_vector_dataset"));
    document.getElementById("rag_vector_generate").onclick = () => generateRagEmbeddings();
    document.getElementById("rag_query_dataset").onchange = () => loadRagVectorPreview(value("rag_query_dataset"));
    document.getElementById("rag_query_reload").onclick = () => loadRagVectorPreview(value("rag_query_dataset"));
    document.getElementById("rag_query_search").onclick = () => searchRagVector();
    document.getElementById("rag_embedding_dataset").onchange = () => loadRagEmbeddingConfig(value("rag_embedding_dataset"));
    document.getElementById("rag_embedding_reload").onclick = () => loadRagEmbeddingConfig(value("rag_embedding_dataset"));
    document.getElementById("rag_embedding_test").onclick = () => testRagEmbeddingConfig();
    document.getElementById("rag_embedding_save").onclick = () => saveRagEmbeddingConfig();
    document.getElementById("rag_curation_pick_file").onclick = () => {
      clearRagCurationFileSelection();
      document.getElementById("rag_curation_file").click();
    };
    document.getElementById("rag_curation_import").onclick = () => importRagCurationFile();
    document.getElementById("rag_curation_file").onchange = () => {
      const file = document.getElementById("rag_curation_file").files?.[0];
      if (!file) return;
      setText("rag_curation_status", `Arquivo selecionado: ${file.name}. Clique em Importar JSON para concluir.`);
    };
    document.getElementById("meta_evaluation_select").onfocus = () => renderMetaEvaluationOptions();
    document.getElementById("meta_evaluation_select").oninput = () => {
      selectedMetaEvaluationId = "";
      renderMetaEvaluationOptions();
    };
    document.getElementById("meta_evaluation_select").onchange = () => loadMetaEvaluation();
    document.getElementById("meta_evaluation_clear").onclick = () => clearMetaEvaluationSelection();
    document.getElementById("meta_reload").onclick = () => loadMetaEvaluation();
    document.getElementById("meta_cancel_edit").onclick = () => {
      resetMetaForm();
      setText("meta_status", `Avaliacao ${display(getMetaEvaluationId())} carregada para meta-avaliacao.`);
    };
    document.getElementById("meta_save").onclick = () => saveMetaEvaluation();
    document.getElementById("meta_mode_new").onclick = () => switchMetaMode("new");
    document.getElementById("meta_mode_history").onclick = () => switchMetaMode("history");
    document.getElementById("meta_history_reload").onclick = () => loadMetaHistory();
    document.getElementById("meta_history_evaluator").onchange = () => renderMetaHistory();
    document.getElementById("meta_history_previous").onclick = () => navigateMetaHistory(-1);
    document.getElementById("meta_history_next").onclick = () => navigateMetaHistory(1);
    for (const button of document.querySelectorAll("[data-meta-history-sort]")) {
      button.onclick = () => sortMetaHistory(button.dataset.metaHistorySort);
    }
    document.getElementById("dashboard-refresh").onclick = loadDashboard;
    document.getElementById("dashboard-model-carousel-prev").onclick = () => moveCarousel(-1);
    document.getElementById("dashboard-model-carousel-next").onclick = () => moveCarousel(1);
    document.addEventListener("click", (event) => {
      if (!event.target.closest(".meta-evaluation-field")) {
        document.getElementById("meta_evaluation_options").hidden = true;
      }
    });
    for (const tab of document.querySelectorAll("[data-carousel-index]")) {
      tab.onclick = () => goToCarouselPage(Number(tab.dataset.carouselIndex));
    }
    updateCarouselState();
    resetCarouselTabsScroll();
    requestAnimationFrame(resetCarouselTabsScroll);
    document.getElementById("dashboard-clear").onclick = () => {
      document.getElementById("dashboard_dataset").value = "J1";
      document.getElementById("dashboard_status").value = "all";
      document.getElementById("dashboard_group_by").value = "modelo";
      for (const id of ["dashboard_candidate_model", "dashboard_judge_model"]) {
        for (const option of document.getElementById(id).options) option.selected = false;
      }
      loadDashboard();
    };

    const databaseActionsToggle = document.getElementById("database-actions-toggle");
    const databaseActionsMenu = document.getElementById("database-actions-menu");
    function setDatabaseActionsMenu(open) {
      databaseActionsMenu.hidden = !open;
      databaseActionsToggle.setAttribute("aria-expanded", String(open));
    }
    databaseActionsToggle.onclick = (event) => {
      event.stopPropagation();
      setDatabaseActionsMenu(databaseActionsMenu.hidden);
    };
    databaseActionsMenu.onclick = (event) => event.stopPropagation();
    document.addEventListener("click", () => setDatabaseActionsMenu(false));

    document.getElementById("database-clean").onclick = async () => {
      setDatabaseActionsMenu(false);
      const cleanAction = await confirmDatabaseClean();
      if (cleanAction === "cancel") return;
      const button = document.getElementById("database-clean");
      const status = document.getElementById("database-dump-status");
      button.disabled = true;
      try {
        status.textContent = "Gerando dump antes de limpar...";
        const dumpData = await postJson("/api/database-dumps", {});
        await handleDatabaseDumpResult(dumpData);
        status.textContent = "Restaurando banco para o estado inicial...";
        const data = await postJson("/api/database-reset", {});
        status.textContent = data.message || "Banco restaurado para o estado inicial.";
        await loadDashboard();
      } catch (error) {
        status.textContent = friendlyErrorMessage(error.message);
      } finally {
        button.disabled = false;
      }
    };

    const restoreFileInput = document.getElementById("database-restore-file");
    document.getElementById("database-restore").onclick = () => {
      setDatabaseActionsMenu(false);
      restoreFileInput.value = "";
      restoreFileInput.click();
    };
    restoreFileInput.onchange = async () => {
      const file = restoreFileInput.files?.[0];
      if (!file) return;
      const button = document.getElementById("database-restore");
      const status = document.getElementById("database-dump-status");
      button.disabled = true;
      status.textContent = `Restaurando backup ${file.name}...`;
      try {
        const data = await postBackupFile(file);
        status.textContent = data.message || "Backup restaurado.";
        await loadDashboard();
      } catch (error) {
        status.textContent = friendlyErrorMessage(error.message);
      } finally {
        button.disabled = false;
      }
    };

    document.getElementById("database-dump").onclick = async () => {
      setDatabaseActionsMenu(false);
      const button = document.getElementById("database-dump");
      const status = document.getElementById("database-dump-status");
      button.disabled = true;
      status.textContent = "Gerando dump completo...";
      try {
        const data = await postJson("/api/database-dumps", {});
        status.textContent = "";
        await handleDatabaseDumpResult(data);
      } catch (error) {
        status.textContent = friendlyErrorMessage(error.message);
      } finally {
        button.disabled = false;
      }
    };

    document.getElementById("dry-run").onclick = async () => {
      try {
        const data = await postJson("/api/runs/dry-run", payload());
        renderRun({status:"dry-run", result:data, progress:{percent:100,current:0,total:0}});
      } catch (error) {
        setText("run-status", "failed");
        setText("output", friendlyErrorMessage(error.message));
      }
    };
    document.getElementById("stop-run").onclick = async () => {
      if (!activeRunId) return;
      try {
        const data = await postJson(`/api/runs/${activeRunId}/cancel`, {});
        renderRun(data);
      } catch (error) {
        setText("output", friendlyErrorMessage(error.message));
      }
    };
    document.getElementById("run").onclick = async () => {
      try {
        const data = await postJson("/api/runs", payload());
        activeRunId = data.run_id;
        renderRun(data);
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(() => poll(data.run_id), 1000);
        await poll(data.run_id);
      } catch (error) {
        setText("run-status", "failed");
        setText("output", friendlyErrorMessage(error.message));
      }
    };
    prefetchOperationalLogSummary();
    prefetchRunHistory();
    loadConfig();
  </script>
</body>
</html>
"""


app = create_app()
