"""Service-layer orchestration for AV3 candidate RAG generation runs."""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from .audit import AuditEvent, AuditLogger
from .candidate_context_budget import (
    BudgetedCandidateRetrievalContext,
    CandidatePromptBudget,
    CandidateModelRuntimeProfile,
    aggregate_budget_metadata,
    budget_candidate_retrieval_context,
    budget_to_metadata,
    resolve_candidate_max_output_tokens,
    resolve_candidate_model_runtime_profile,
)
from .candidate_clients.base import CandidateClient
from .candidate_clients.remote_http import RemoteHttpCandidateClient, RemoteHttpCandidateClientConfig
from .candidate_runtime_learning import parse_candidate_runtime_observation
from .candidate_prompts import build_candidate_prompt
from .config import load_settings
from .contracts import (
    CandidateAnswerContextChunkRecord,
    CandidateAnswerRecord,
    CandidateModelAssignment,
    CandidateProgressCallback,
    CandidateProgressEvent,
    CandidateModelRuntimeObservationRecord,
    CandidateModelRuntimeProfileRecord,
    CandidatePromptContext,
    CandidatePromptRecord,
    CandidateQuestionRecord,
    CandidateQuestionSelectionResult,
    CandidateQuestionSelectionSummary,
    CandidateRunRecord,
    RagRetrievalResult,
)
from .db import connect
from .rag_context_snapshots import RagContextSnapshotService
from .rag_curation import resolve_rag_curation_dataset
from .rag_embedding_client import request_openai_compatible_embeddings
from .rag_retriever import RagRetrieverService
from .repositories import JudgeRepository


CandidateShouldStop = Callable[[], bool]


@dataclass(frozen=True)
class RunCandidatesRagRequest:
    """Request contract for one candidate RAG generation batch."""

    model_name: str
    provider: str
    dataset: str = "J1"
    batch_size: int | None = None
    question_sequence_start: int | None = None
    question_sequence_end: int | None = None
    question_id: int | None = None
    prompt_id: int | None = None
    retrieval_run_id: int | None = None
    skip_existing_successful: bool = True
    save_raw_response: bool = False
    dry_run: bool = False
    audit_log: str | None = None
    no_audit_animation: bool = False
    created_by: str = "system"
    remote_candidate_base_url: str | None = None
    remote_candidate_api_key: str | None = None
    remote_candidate_timeout_seconds: int = 120
    remote_candidate_temperature: float | None = None
    remote_candidate_max_tokens: int | None = None
    remote_candidate_top_p: float | None = None
    remote_candidate_context_safety_margin_tokens: int | None = None
    remote_candidate_context_window_tokens: int | None = None
    remote_candidate_retry_on_context_window: bool | None = None
    remote_candidate_openai_compatible: bool = True
    candidate_execution_strategy: str | None = None
    candidate_parallel_max_workers: int | None = None
    progress_callback: CandidateProgressCallback | None = None
    should_stop: CandidateShouldStop | None = None


@dataclass(frozen=True)
class ResolvedRunCandidatesRag:
    """Normalized request values before any DB or remote side effects."""

    dataset: str
    batch_size: int
    model_name: str
    provider: str
    question_sequence_start: int | None
    question_sequence_end: int | None
    question_id: int | None
    prompt_id: int | None
    retrieval_run_id: int | None
    skip_existing_successful: bool
    audit_path: Path
    execution_summary: str


@dataclass(frozen=True)
class CandidateQuestionRunResult:
    """Per-question execution result for one candidate run."""

    question_id: int
    question_sequence: int
    status: str
    retrieval_status: str | None = None
    candidate_answer_id: int | None = None
    final_choice: str | None = None
    error_message: str | None = None
    latency_ms: int | None = None


@dataclass(frozen=True)
class CandidateRunSummary:
    """Structured run summary returned by the candidate runner service."""

    selected_questions: int
    processed_questions: int
    successful_answers: int
    failed_answers: int
    skipped_questions: int
    question_results: list[CandidateQuestionRunResult] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedCandidateProviderConfig:
    """Provider-specific remote HTTP settings resolved from the AV3 registry."""

    av3_provider: str
    base_url: str
    api_key: str | None


@dataclass(frozen=True)
class ResolvedCandidateRuntimeConfig:
    """Resolved candidate runtime config after assignment and env handling."""

    model_name: str
    technical_provider: str
    av3_provider: str
    base_url: str
    api_key: str | None
    temperature: float
    top_p: float
    default_max_output_tokens: int
    max_output_tokens_cap: int | None
    requested_max_tokens: int
    max_tokens: int
    context_window_tokens: int | None
    safety_margin_tokens: int
    chars_per_token_estimate: float
    prompt_budget_utilization: float
    model_profile_source: str
    model_profile_confidence: str
    save_raw_response: bool
    retry_on_context_window: bool


@dataclass(frozen=True)
class ResolvedCandidateExecutionConfig:
    """Resolved candidate execution strategy and bounded parallelism."""

    strategy: str
    parallel_max_workers: int
    adaptive_initial_concurrency: int
    adaptive_max_concurrency: int
    adaptive_success_threshold: int
    adaptive_max_retries: int
    adaptive_base_backoff_seconds: float
    adaptive_max_backoff_seconds: float


@dataclass(frozen=True)
class CandidateGenerationOutcome:
    """Final per-question execution outcome after optional retry handling."""

    answer: CandidateAnswerRecord
    retrieval_result_for_prompt: RagRetrievalResult
    budget: CandidatePromptBudget | None
    runtime_config: ResolvedCandidateRuntimeConfig


@dataclass(frozen=True)
class CandidateQuestionTaskResult:
    """Internal per-question task result plus metadata needed for run aggregation."""

    question_result: CandidateQuestionRunResult
    budget: CandidatePromptBudget | None
    runtime_config: ResolvedCandidateRuntimeConfig


@dataclass(frozen=True)
class CandidateProgressCounts:
    """Terminal per-run counters exposed in batch progress events."""

    selected_questions: int
    processed_questions: int = 0
    successful_answers: int = 0
    failed_answers: int = 0
    skipped_questions: int = 0


class CandidateProgressReporter:
    """Emit typed progress events without affecting candidate execution."""

    def __init__(
        self,
        *,
        audit: AuditLogger,
        request: RunCandidatesRagRequest,
        resolved: ResolvedRunCandidatesRag,
        candidate_run_id: int | None = None,
        selected_questions: int = 0,
    ) -> None:
        self._audit = audit
        self._callback = request.progress_callback
        self._resolved = resolved
        self._candidate_run_id = candidate_run_id
        self._lock = threading.Lock()
        self._counts = CandidateProgressCounts(selected_questions=selected_questions)

    def set_run(self, *, candidate_run_id: int, selected_questions: int) -> None:
        with self._lock:
            self._candidate_run_id = candidate_run_id
            self._counts = CandidateProgressCounts(selected_questions=selected_questions)

    def emit(
        self,
        event_type: str,
        *,
        question: CandidateQuestionRecord | None = None,
        status: str | None = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = CandidateProgressEvent(
            event_type=event_type,
            candidate_run_id=self._candidate_run_id,
            dataset=self._resolved.dataset,
            model_name=self._resolved.model_name,
            provider=self._resolved.provider,
            question_id=None if question is None else question.question_id,
            question_sequence=None if question is None else question.question_sequence,
            status=status,
            message=message,
            metadata=dict(metadata or {}),
        )
        self._dispatch(event)

    def emit_question_terminal(self, result: CandidateQuestionTaskResult) -> None:
        with self._lock:
            counts = self._counts
            if result.question_result.status == "success":
                counts = replace(
                    counts,
                    processed_questions=counts.processed_questions + 1,
                    successful_answers=counts.successful_answers + 1,
                )
            elif result.question_result.status == "failed":
                counts = replace(
                    counts,
                    processed_questions=counts.processed_questions + 1,
                    failed_answers=counts.failed_answers + 1,
                )
            elif result.question_result.status == "skipped":
                counts = replace(
                    counts,
                    skipped_questions=counts.skipped_questions + 1,
                )
            self._counts = counts
            payload = {
                "selected_questions": counts.selected_questions,
                "processed_questions": counts.processed_questions,
                "successful_answers": counts.successful_answers,
                "failed_answers": counts.failed_answers,
                "skipped_questions": counts.skipped_questions,
            }
        self.emit("candidate_batch_progress", status="running", metadata=payload)

    def counts(self) -> CandidateProgressCounts:
        with self._lock:
            return self._counts

    def _dispatch(self, event: CandidateProgressEvent) -> None:
        if self._callback is None:
            return
        try:
            self._callback(event)
        except Exception as error:
            self._audit.event(
                AuditEvent(
                    "candidate_progress_callback_failed",
                    f"event_type={event.event_type} error_type={type(error).__name__}",
                )
            )


class CandidateStopState:
    """Track cooperative stop state and emit secret-safe stop progress."""

    def __init__(
        self,
        *,
        should_stop: CandidateShouldStop | None,
        audit: AuditLogger,
        progress: CandidateProgressReporter,
        execution_strategy: str,
    ) -> None:
        self._should_stop = should_stop
        self._audit = audit
        self._progress = progress
        self._execution_strategy = execution_strategy
        self._requested = False
        self._not_started_question_ids: set[int] = set()
        self._lock = threading.Lock()

    @property
    def requested(self) -> bool:
        with self._lock:
            return self._requested

    @property
    def not_started_due_to_stop(self) -> int:
        with self._lock:
            return len(self._not_started_question_ids)

    def check(self) -> bool:
        with self._lock:
            if self._requested:
                return True
            should_stop = self._should_stop
        if should_stop is None:
            return False
        try:
            requested = bool(should_stop())
        except Exception as error:
            self._audit.event(
                AuditEvent(
                    "candidate_should_stop_failed",
                    f"error_type={type(error).__name__}",
                )
            )
            return False
        if not requested:
            return False
        self._mark_requested()
        return True

    def mark_not_started(self, questions: list[CandidateQuestionRecord]) -> None:
        for question in questions:
            should_emit = False
            with self._lock:
                if question.question_id not in self._not_started_question_ids:
                    self._not_started_question_ids.add(question.question_id)
                    should_emit = True
            if should_emit:
                counts = self._progress.counts()
                self._progress.emit(
                    "candidate_task_not_started_due_to_stop",
                    question=question,
                    status="cancelled",
                    metadata={
                        "selected_questions": counts.selected_questions,
                        "processed_questions": counts.processed_questions,
                        "successful_answers": counts.successful_answers,
                        "failed_answers": counts.failed_answers,
                        "skipped_questions": counts.skipped_questions,
                        "not_started_due_to_stop": self.not_started_due_to_stop,
                        "stop_requested": True,
                        "stop_reason": "cooperative_stop",
                        "execution_strategy": self._execution_strategy,
                    },
                )

    def metadata(self) -> dict[str, Any]:
        return {
            "stop_requested": self.requested,
            "stop_reason": "cooperative_stop" if self.requested else None,
            "not_started_due_to_stop": self.not_started_due_to_stop,
        }

    def _mark_requested(self) -> None:
        with self._lock:
            if self._requested:
                return
            self._requested = True
        counts = self._progress.counts()
        self._audit.event(
            AuditEvent(
                "candidate_stop_requested",
                (
                    f"selected={counts.selected_questions} processed={counts.processed_questions} "
                    f"skipped={counts.skipped_questions} strategy={self._execution_strategy}"
                ),
            )
        )
        self._progress.emit(
            "candidate_stop_requested",
            status="cancelling",
            metadata={
                "selected_questions": counts.selected_questions,
                "processed_questions": counts.processed_questions,
                "successful_answers": counts.successful_answers,
                "failed_answers": counts.failed_answers,
                "skipped_questions": counts.skipped_questions,
                "not_started_due_to_stop": self.not_started_due_to_stop,
                "stop_requested": True,
                "stop_reason": "cooperative_stop",
                "execution_strategy": self._execution_strategy,
            },
        )


@dataclass(frozen=True)
class CandidateAdaptiveGroupKey:
    """Secret-safe grouping key for adaptive candidate scheduling."""

    av3_provider: str
    base_url: str
    api_key_fingerprint: str
    model_name: str

    @property
    def label(self) -> str:
        return (
            f"av3_provider={self.av3_provider} base_url={self.base_url} "
            f"api_key={self.api_key_fingerprint} model={self.model_name}"
        )


@dataclass
class CandidateAdaptiveGroupState:
    """Mutable per-provider/model scheduler metrics."""

    key: CandidateAdaptiveGroupKey
    current_concurrency: int
    max_concurrency: int
    in_flight: int = 0
    cooldown_until: float = 0.0
    consecutive_successes: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    rate_limits: int = 0
    transient_failures: int = 0
    non_retryable_failures: int = 0
    retries: int = 0
    requeued: int = 0
    disabled: bool = False

    @property
    def final_concurrency(self) -> int:
        return self.current_concurrency


@dataclass(frozen=True)
class CandidateAdaptiveQueuedTask:
    question: CandidateQuestionRecord
    group_key: CandidateAdaptiveGroupKey
    attempt: int = 0
    ready_at: float = 0.0


@dataclass(frozen=True)
class CandidateAdaptiveCompletedTask:
    queued: CandidateAdaptiveQueuedTask
    result: CandidateQuestionTaskResult


@dataclass(frozen=True)
class CandidateErrorClassification:
    reason: str
    retryable: bool
    fatal_group: bool = False
    timeout: bool = False
    rate_limit: bool = False


@dataclass(frozen=True)
class RunCandidatesRagResult:
    """Service result contract shared by future CLI/Web adapters."""

    dry_run: bool
    audit_log: str
    execution_summary: str
    batch_size: int
    dataset: str
    model_name: str
    provider: str
    runtime_config_summary: str | None = None
    candidate_run_id: int | None = None
    retrieval_run_id: int | None = None
    prompt_id: int | None = None
    summary: CandidateRunSummary | None = None


class CandidateRunRepositoryProtocol(Protocol):
    """Repository operations required by the candidate runner service."""

    def ensure_schema(self) -> None:
        """Ensure required AV3 tables exist."""

    def list_candidate_model_assignments(self) -> tuple[CandidateModelAssignment, ...]:
        """Return the centralized AV3 candidate-model registry."""

    def get_rag_vector_base_summary(self, *, dataset: str) -> Any | None:
        """Return the active vector base summary for the requested dataset."""

    def get_or_create_candidate_prompt(
        self,
        *,
        dataset: str,
        prompt_id: int | None = None,
    ) -> CandidatePromptRecord:
        """Return an explicit or active candidate prompt, seeding defaults when needed."""

    def create_candidate_run(self, *, run: CandidateRunRecord) -> CandidateRunRecord:
        """Persist one candidate run."""

    def get_candidate_model_runtime_profile(
        self,
        *,
        av3_provider: str,
        provider_model_id: str,
    ) -> CandidateModelRuntimeProfileRecord | None:
        """Return one persisted runtime profile, if available."""

    def upsert_candidate_model_runtime_profile(
        self,
        *,
        av3_provider: str,
        provider_model_id: str,
        context_window_tokens: int | None,
        default_max_output_tokens: int | None,
        safety_margin_tokens: int,
        source: str,
        confidence: str,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateModelRuntimeProfileRecord:
        """Insert or update one provider/model runtime profile."""

    def record_candidate_model_runtime_observation(
        self,
        *,
        av3_provider: str,
        provider_model_id: str,
        observed_context_window_tokens: int | None,
        observed_prompt_tokens: int | None,
        observed_requested_max_tokens: int | None,
        observed_total_tokens: int | None,
        error_class: str,
        error_message: str,
        candidate_run_id: int | None = None,
        candidate_answer_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateModelRuntimeObservationRecord:
        """Persist one immutable runtime observation."""

    def update_candidate_run_status(
        self,
        *,
        candidate_run_id: int,
        run_status: str,
        finished_at: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist the terminal status for one candidate run."""

    def select_candidate_questions(
        self,
        *,
        dataset: str,
        batch_size: int,
        question_sequence_start: int | None,
        question_sequence_end: int | None,
        question_id: int | None,
    ) -> list[CandidateQuestionRecord]:
        """Select candidate-safe questions for one execution batch."""

    def select_pending_candidate_questions(
        self,
        *,
        dataset: str,
        model_name: str,
        batch_size: int,
        question_sequence_start: int | None,
        question_sequence_end: int | None,
        question_id: int | None,
        skip_existing_successful: bool,
    ) -> CandidateQuestionSelectionResult:
        """Select model-aware pending candidate-safe questions for one execution batch."""

    def successful_candidate_answer_exists(
        self,
        *,
        dataset: str,
        model_name: str,
        question_id: int,
        exclude_candidate_run_id: int | None = None,
    ) -> bool:
        """Return whether a successful answer already exists for this dataset/model/question."""

    def persist_candidate_answer(self, *, answer: CandidateAnswerRecord) -> CandidateAnswerRecord:
        """Insert or update one candidate answer."""

    def persist_successful_candidate_answer_with_context_snapshot(
        self,
        *,
        answer: CandidateAnswerRecord,
        retrieval_result: RagRetrievalResult,
    ) -> tuple[CandidateAnswerRecord, list[CandidateAnswerContextChunkRecord]]:
        """Persist one successful candidate answer and its context snapshot atomically."""

    def get_rag_embedding_model_config(self, *, dataset: str) -> Any | None:
        """Return the dataset embedding config used to call the retriever's embedder."""


class RunCandidatesRagService:
    """Application boundary for AV3 candidate RAG execution without CLI wiring."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], Any] = load_settings,
        connect_func: Callable[[str], Any] = connect,
        repository_factory: Callable[[Any], CandidateRunRepositoryProtocol] = JudgeRepository,
        retriever_factory: Callable[[CandidateRunRepositoryProtocol, Any, str], RagRetrieverService] | None = None,
        client_factory: Callable[[RunCandidatesRagRequest, Any], CandidateClient] | None = None,
        snapshot_service_factory: Callable[[CandidateRunRepositoryProtocol], RagContextSnapshotService] | None = None,
        sleep_func: Callable[[float], None] = time.sleep,
        monotonic_func: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings_loader = settings_loader
        self._connect = connect_func
        self._repository_factory = repository_factory
        self._retriever_factory = retriever_factory or _default_retriever_factory
        self._client_factory = client_factory or _default_client_factory
        self._snapshot_service_factory = snapshot_service_factory or _default_snapshot_service_factory
        self._sleep = sleep_func
        self._monotonic = monotonic_func

    def resolve(self, request: RunCandidatesRagRequest) -> ResolvedRunCandidatesRag:
        """Normalize request values without opening the database or calling providers."""
        dataset = resolve_rag_curation_dataset(request.dataset)
        batch_size = max(1, int(request.batch_size or 10))
        start = int(request.question_sequence_start) if request.question_sequence_start is not None else None
        end = int(request.question_sequence_end) if request.question_sequence_end is not None else None
        if start is not None and start <= 0:
            raise ValueError("question_sequence_start must be greater than zero.")
        if end is not None and end <= 0:
            raise ValueError("question_sequence_end must be greater than zero.")
        if start is not None and end is not None and start > end:
            raise ValueError("question_sequence_start must be less than or equal to question_sequence_end.")
        if request.question_id is not None and int(request.question_id) <= 0:
            raise ValueError("question_id must be greater than zero.")
        if request.prompt_id is not None and int(request.prompt_id) <= 0:
            raise ValueError("prompt_id must be greater than zero.")
        if request.retrieval_run_id is not None and int(request.retrieval_run_id) <= 0:
            raise ValueError("retrieval_run_id must be greater than zero.")

        model_name = request.model_name.strip()
        provider = request.provider.strip()
        if not model_name:
            raise ValueError("model_name is required.")
        if not provider:
            raise ValueError("provider is required.")
        return ResolvedRunCandidatesRag(
            dataset=dataset,
            batch_size=batch_size,
            model_name=model_name,
            provider=provider,
            question_sequence_start=start,
            question_sequence_end=end,
            question_id=int(request.question_id) if request.question_id is not None else None,
            prompt_id=int(request.prompt_id) if request.prompt_id is not None else None,
            retrieval_run_id=int(request.retrieval_run_id) if request.retrieval_run_id is not None else None,
            skip_existing_successful=bool(request.skip_existing_successful),
            audit_path=_resolve_audit_path(request.audit_log),
            execution_summary=_build_execution_summary(
                dataset=dataset,
                batch_size=batch_size,
                model_name=model_name,
                provider=provider,
                question_sequence_start=start,
                question_sequence_end=end,
                question_id=request.question_id,
                candidate_execution_strategy=request.candidate_execution_strategy or "sequential",
                candidate_parallel_max_workers=request.candidate_parallel_max_workers or 2,
            ),
        )

    def run(self, request: RunCandidatesRagRequest) -> RunCandidatesRagResult:
        """Run or dry-run the candidate RAG execution flow."""
        resolved = self.resolve(request)
        animate = False if request.no_audit_animation else None
        with AuditLogger(file_path=resolved.audit_path, animate=animate) as audit:
            progress = CandidateProgressReporter(audit=audit, request=request, resolved=resolved)
            with audit.step("Loading configuration"):
                settings = self._settings_loader()
            execution_config = _resolve_candidate_execution_config(settings=settings, request=request)
            execution_summary = _build_execution_summary(
                dataset=resolved.dataset,
                batch_size=resolved.batch_size,
                model_name=resolved.model_name,
                provider=resolved.provider,
                question_sequence_start=resolved.question_sequence_start,
                question_sequence_end=resolved.question_sequence_end,
                question_id=resolved.question_id,
                candidate_execution_strategy=execution_config.strategy,
                candidate_parallel_max_workers=execution_config.parallel_max_workers,
                candidate_adaptive_initial_concurrency=execution_config.adaptive_initial_concurrency,
                candidate_adaptive_max_concurrency=execution_config.adaptive_max_concurrency,
                candidate_adaptive_success_threshold=execution_config.adaptive_success_threshold,
                candidate_adaptive_max_retries=execution_config.adaptive_max_retries,
                candidate_adaptive_base_backoff_seconds=execution_config.adaptive_base_backoff_seconds,
                candidate_adaptive_max_backoff_seconds=execution_config.adaptive_max_backoff_seconds,
            )
            audit.file_event("execution_summary", execution_summary.replace("\n", " | "))
            with audit.step("Connecting to local PostgreSQL", detail="DATABASE_URL=<redacted>"):
                connection = self._connect(settings.database_url)
            repository: CandidateRunRepositoryProtocol | None = None
            run: CandidateRunRecord | None = None
            try:
                repository = self._repository_factory(connection)
                snapshot_service = self._snapshot_service_factory(repository)
                with audit.step("Ensuring candidate metadata schema"):
                    repository.ensure_schema()
                vector_base = repository.get_rag_vector_base_summary(dataset=resolved.dataset)
                if vector_base is None:
                    raise ValueError(f"No active RAG vector base found for {resolved.dataset}.")
                if resolved.retrieval_run_id is not None and resolved.retrieval_run_id != int(vector_base.retrieval_run_id):
                    raise ValueError(
                        f"retrieval_run_id={resolved.retrieval_run_id} is not supported because only the active "
                        f"retrieval run is executable today (active={vector_base.retrieval_run_id})."
                    )
                prompt = repository.get_or_create_candidate_prompt(
                    dataset=resolved.dataset,
                    prompt_id=resolved.prompt_id,
                )
                with audit.step(
                    f"Selecting candidate questions for {resolved.dataset}",
                    detail=(
                        f"dataset={resolved.dataset} batch_size={resolved.batch_size} "
                        f"model={resolved.model_name} skip_existing_successful={resolved.skip_existing_successful} "
                        f"question_id={resolved.question_id} start={resolved.question_sequence_start} "
                        f"end={resolved.question_sequence_end}"
                    ),
                ):
                    selection_result = repository.select_pending_candidate_questions(
                        dataset=resolved.dataset,
                        model_name=resolved.model_name,
                        batch_size=resolved.batch_size,
                        question_sequence_start=resolved.question_sequence_start,
                        question_sequence_end=resolved.question_sequence_end,
                        question_id=resolved.question_id,
                        skip_existing_successful=resolved.skip_existing_successful,
                    )
                    questions = selection_result.questions
                selection_summary = _format_candidate_question_selection(selection_result.summary)
                audit.file_event("candidate_question_selection", selection_summary.replace("\n", " | "))
                audit.terminal_event(selection_summary)
                progress.emit(
                    "candidate_question_selected",
                    status="selected",
                    metadata={
                        "selected_count": selection_result.summary.selected,
                        "failed_retry_count": selection_result.summary.failed_retry_candidates,
                        "unanswered_count": selection_result.summary.unanswered_candidates,
                        "skipped_success_excluded_count": selection_result.summary.successful_excluded,
                        "selection_policy": selection_result.summary.policy,
                        "skip_existing_successful": selection_result.summary.skip_existing_successful,
                    },
                )
                runtime_config = _resolve_candidate_runtime_config(
                    repository=repository,
                    settings=settings,
                    request=request,
                    resolved=resolved,
                    questions=questions,
                    require_api_key=not request.dry_run,
                )
                _validate_candidate_provider_execution_strategy(
                    runtime_config=runtime_config,
                    execution_config=execution_config,
                )
                runtime_summary = _format_candidate_runtime_config(
                    runtime_config,
                    execution_config=execution_config,
                    api_key_state="<not required in dry-run>"
                    if request.dry_run
                    else ("<set>" if runtime_config.api_key else "<missing>"),
                )
                audit.file_event("candidate_runtime_config", runtime_summary.replace("\n", " | "))
                audit.terminal_event(runtime_summary)
                retriever = self._retriever_factory(repository, settings, resolved.dataset)
                if request.dry_run:
                    for question in questions:
                        retrieval_result = retriever.retrieve_for_question(
                            question_id=question.question_id,
                            dataset=resolved.dataset,
                        )
                        audit.event(
                            AuditEvent(
                                "dry_run_retrieval_finished",
                                (
                                    f"question_id={question.question_id} status={retrieval_result.status} "
                                    f"chunks={len(retrieval_result.chunks)}"
                                ),
                            )
                        )
                        if retrieval_result.status == "success":
                            budgeted_context = _budget_retrieval_for_question(
                                question=question,
                                retrieval_result=retrieval_result,
                                prompt=prompt,
                                runtime_config=runtime_config,
                            )
                            _log_candidate_prompt_budget(
                                audit=audit,
                                question=question,
                                budget=budgeted_context.budget,
                            )
                    audit.file_event("dry_run_finished", "no candidate_run rows created and no remote candidate calls made")
                    audit.terminal_event("Dry run: no candidate_run rows created and no remote candidate calls made.")
                    return RunCandidatesRagResult(
                        dry_run=True,
                        audit_log=str(resolved.audit_path),
                        execution_summary=execution_summary,
                        runtime_config_summary=runtime_summary,
                        batch_size=resolved.batch_size,
                        dataset=resolved.dataset,
                        model_name=resolved.model_name,
                        provider=resolved.provider,
                    )

                started_at = _utcnow_iso()
                run = repository.create_candidate_run(
                    run=CandidateRunRecord(
                        candidate_run_id=None,
                        dataset=resolved.dataset,
                        retrieval_run_id=int(vector_base.retrieval_run_id),
                        prompt_id=int(prompt.prompt_id),
                        model_name=resolved.model_name,
                        provider=resolved.provider,
                        batch_size=resolved.batch_size,
                        run_status="running",
                        temperature=runtime_config.temperature,
                        max_tokens=runtime_config.max_tokens,
                        top_p=runtime_config.top_p,
                        started_at=started_at,
                        created_by=request.created_by,
                        metadata=_run_metadata(resolved, vector_base, prompt, request, runtime_config, execution_config),
                    )
                )
                audit.event(
                    AuditEvent(
                        "run_started",
                        (
                            f"candidate_run_id={run.candidate_run_id} dataset={resolved.dataset} "
                            f"model={resolved.model_name} provider={resolved.provider} "
                            f"retrieval_run_id={vector_base.retrieval_run_id} prompt_id={prompt.prompt_id}"
                        ),
                    )
                )
                progress.set_run(candidate_run_id=int(run.candidate_run_id), selected_questions=len(questions))
                progress.emit(
                    "candidate_run_started",
                    status="running",
                    metadata={
                        "execution_strategy": execution_config.strategy,
                        "selected_count": len(questions),
                    },
                )
                client_request = request
                if self._client_factory is _default_client_factory:
                    client_request = _with_remote_candidate_config(request, runtime_config)
                stop_state = CandidateStopState(
                    should_stop=request.should_stop,
                    audit=audit,
                    progress=progress,
                    execution_strategy=execution_config.strategy,
                )
                adaptive_metrics: dict[str, Any] | None = None
                if execution_config.strategy == "adaptive":
                    scheduler = CandidateAdaptiveScheduler(
                        service=self,
                        execution_config=execution_config,
                        audit=audit,
                        progress=progress,
                        stop_state=stop_state,
                    )
                    task_results = scheduler.run(
                        settings=settings,
                        request=request,
                        client_request=client_request,
                        resolved=resolved,
                        prompt=prompt,
                        questions=questions,
                        runtime_config=runtime_config,
                        candidate_run_id=int(run.candidate_run_id),
                    )
                    adaptive_metrics = scheduler.summary()
                elif execution_config.strategy == "parallel":
                    task_results = self._run_candidate_questions_parallel(
                        settings=settings,
                        request=request,
                        client_request=client_request,
                        resolved=resolved,
                        prompt=prompt,
                        questions=questions,
                        runtime_config=runtime_config,
                        candidate_run_id=int(run.candidate_run_id),
                        execution_config=execution_config,
                        audit=audit,
                        progress=progress,
                        stop_state=stop_state,
                    )
                else:
                    client = None
                    task_results = []
                    for index, question in enumerate(questions):
                        if stop_state.check():
                            stop_state.mark_not_started(questions[index:])
                            break
                        if client is None:
                            client = self._client_factory(client_request, settings)
                        assert client is not None
                        task_results.append(
                            _execute_candidate_question_task(
                                repository=repository,
                                retriever=retriever,
                                client=client,
                                snapshot_service=snapshot_service,
                                audit=audit,
                                question=question,
                                prompt=prompt,
                                resolved=resolved,
                                request=request,
                                runtime_config=runtime_config,
                                candidate_run_id=int(run.candidate_run_id),
                                isolate_unhandled_errors=False,
                                parallel_audit=False,
                                progress=progress,
                            )
                        )
                        progress.emit_question_terminal(task_results[-1])
                        runtime_config = task_results[-1].runtime_config

                task_results = sorted(
                    task_results,
                    key=lambda item: (
                        item.question_result.question_sequence,
                        item.question_result.question_id,
                    ),
                )
                question_results = [item.question_result for item in task_results]
                successful_answers = sum(1 for item in question_results if item.status == "success")
                failed_answers = sum(1 for item in question_results if item.status == "failed")
                skipped_questions = sum(1 for item in question_results if item.status == "skipped")
                budget_summaries = [item.budget for item in task_results if item.budget is not None]
                if task_results:
                    runtime_config = task_results[-1].runtime_config

                summary = CandidateRunSummary(
                    selected_questions=len(questions),
                    processed_questions=successful_answers + failed_answers,
                    successful_answers=successful_answers,
                    failed_answers=failed_answers,
                    skipped_questions=skipped_questions,
                    question_results=question_results,
                )
                completion_metadata: dict[str, Any] = {
                    "selected_questions": summary.selected_questions,
                    "processed_questions": summary.processed_questions,
                    "successful_answers": summary.successful_answers,
                    "failed_answers": summary.failed_answers,
                    "skipped_questions": summary.skipped_questions,
                    "candidate_execution": {
                        "strategy": execution_config.strategy,
                        "parallel_max_workers": execution_config.parallel_max_workers,
                        "adaptive_initial_concurrency": execution_config.adaptive_initial_concurrency,
                        "adaptive_max_concurrency": execution_config.adaptive_max_concurrency,
                        "adaptive_success_threshold": execution_config.adaptive_success_threshold,
                        "adaptive_max_retries": execution_config.adaptive_max_retries,
                        "adaptive_base_backoff_seconds": execution_config.adaptive_base_backoff_seconds,
                        "adaptive_max_backoff_seconds": execution_config.adaptive_max_backoff_seconds,
                    },
                    "candidate_runtime_profile": _candidate_runtime_profile_metadata(runtime_config),
                }
                if adaptive_metrics is not None:
                    completion_metadata["candidate_adaptive"] = adaptive_metrics
                if stop_state.requested:
                    completion_metadata.update(stop_state.metadata())
                candidate_budget_metadata = aggregate_budget_metadata(
                    budgets=budget_summaries,
                    requested_max_tokens=runtime_config.requested_max_tokens,
                )
                if candidate_budget_metadata:
                    completion_metadata["candidate_budget"] = candidate_budget_metadata
                terminal_run_status = "cancelled" if stop_state.requested else "completed"
                repository.update_candidate_run_status(
                    candidate_run_id=int(run.candidate_run_id),
                    run_status=terminal_run_status,
                    finished_at=_utcnow_iso(),
                    metadata=completion_metadata,
                )
                counts = progress.counts()
                progress.emit(
                    "candidate_run_finished",
                    status=terminal_run_status,
                    metadata={
                        "run_status": terminal_run_status,
                        "selected_questions": counts.selected_questions,
                        "processed_questions": counts.processed_questions,
                        "successful_answers": counts.successful_answers,
                        "failed_answers": counts.failed_answers,
                        "skipped_questions": counts.skipped_questions,
                        "not_started_due_to_stop": stop_state.not_started_due_to_stop,
                        "stop_requested": stop_state.requested,
                        "execution_strategy": execution_config.strategy,
                    },
                )
                audit.event(
                    AuditEvent(
                        "run_finished",
                        (
                            f"candidate_run_id={run.candidate_run_id} selected={summary.selected_questions} "
                            f"processed={summary.processed_questions} success={summary.successful_answers} "
                            f"failed={summary.failed_answers} skipped={summary.skipped_questions}"
                        ),
                    )
                )
                return RunCandidatesRagResult(
                    dry_run=False,
                    audit_log=str(resolved.audit_path),
                    execution_summary=execution_summary,
                    runtime_config_summary=runtime_summary,
                    batch_size=resolved.batch_size,
                    dataset=resolved.dataset,
                    model_name=resolved.model_name,
                    provider=resolved.provider,
                    candidate_run_id=run.candidate_run_id,
                    retrieval_run_id=int(vector_base.retrieval_run_id),
                    prompt_id=int(prompt.prompt_id),
                    summary=summary,
                )
            except Exception as error:
                if repository is not None and run is not None and run.candidate_run_id is not None:
                    repository.update_candidate_run_status(
                        candidate_run_id=int(run.candidate_run_id),
                        run_status="failed",
                        finished_at=_utcnow_iso(),
                        metadata={
                            "error_message": str(error),
                            "error_type": type(error).__name__,
                        },
                    )
                    counts = progress.counts()
                    progress.emit(
                        "candidate_run_finished",
                        status="failed",
                        message=type(error).__name__,
                        metadata={
                            "run_status": "failed",
                            "selected_questions": counts.selected_questions,
                            "processed_questions": counts.processed_questions,
                            "successful_answers": counts.successful_answers,
                            "failed_answers": counts.failed_answers,
                            "skipped_questions": counts.skipped_questions,
                            "execution_strategy": execution_config.strategy,
                        },
                    )
                    audit.event(
                        AuditEvent(
                            "run_failed",
                            (
                                f"candidate_run_id={run.candidate_run_id} "
                                f"error_type={type(error).__name__} error={error}"
                            ),
                        )
                    )
                raise
            finally:
                with audit.step("Closing PostgreSQL connection"):
                    connection.close()

    def _run_candidate_questions_parallel(
        self,
        *,
        settings: Any,
        request: RunCandidatesRagRequest,
        client_request: RunCandidatesRagRequest,
        resolved: ResolvedRunCandidatesRag,
        prompt: CandidatePromptRecord,
        questions: list[CandidateQuestionRecord],
        runtime_config: ResolvedCandidateRuntimeConfig,
        candidate_run_id: int,
        execution_config: ResolvedCandidateExecutionConfig,
        audit: AuditLogger,
        progress: CandidateProgressReporter,
        stop_state: CandidateStopState,
    ) -> list[CandidateQuestionTaskResult]:
        if not questions:
            return []

        worker_count = min(execution_config.parallel_max_workers, len(questions))
        audit.event(
            AuditEvent(
                "candidate_parallel_started",
                (
                    f"candidate_run_id={candidate_run_id} model={resolved.model_name} "
                    f"workers={worker_count} selected={len(questions)}"
                ),
            )
        )
        task_results: list[CandidateQuestionTaskResult] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending = list(questions)
            futures: dict[Future[CandidateQuestionTaskResult], CandidateQuestionRecord] = {}

            def submit_next() -> bool:
                if not pending:
                    return False
                if stop_state.check():
                    stop_state.mark_not_started(pending)
                    pending.clear()
                    return False
                question = pending.pop(0)
                futures[
                    executor.submit(
                        self._run_candidate_question_worker,
                        settings=settings,
                        request=request,
                        client_request=client_request,
                        resolved=resolved,
                        prompt=prompt,
                        question=question,
                        runtime_config=runtime_config,
                        candidate_run_id=candidate_run_id,
                        audit=audit,
                        propagate_retryable_generation_errors=False,
                        progress=progress,
                    )
                ] = question
                return True

            while pending and len(futures) < worker_count:
                if not submit_next():
                    break

            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    result = future.result()
                    task_results.append(result)
                    progress.emit_question_terminal(result)
                while pending and len(futures) < worker_count:
                    if not submit_next():
                        break

        successful_answers = sum(1 for item in task_results if item.question_result.status == "success")
        failed_answers = sum(1 for item in task_results if item.question_result.status == "failed")
        skipped_questions = sum(1 for item in task_results if item.question_result.status == "skipped")
        audit.event(
            AuditEvent(
                "candidate_parallel_finished",
                (
                    f"candidate_run_id={candidate_run_id} workers={worker_count} "
                    f"success={successful_answers} failed={failed_answers} skipped={skipped_questions}"
                ),
            )
        )
        return task_results

    def _run_candidate_question_worker(
        self,
        *,
        settings: Any,
        request: RunCandidatesRagRequest,
        client_request: RunCandidatesRagRequest,
        resolved: ResolvedRunCandidatesRag,
        prompt: CandidatePromptRecord,
        question: CandidateQuestionRecord,
        runtime_config: ResolvedCandidateRuntimeConfig,
        candidate_run_id: int,
        audit: AuditLogger,
        propagate_retryable_generation_errors: bool = False,
        progress: CandidateProgressReporter | None = None,
    ) -> CandidateQuestionTaskResult:
        connection = self._connect(settings.database_url)
        try:
            repository = self._repository_factory(connection)
            retriever = self._retriever_factory(repository, settings, resolved.dataset)
            snapshot_service = self._snapshot_service_factory(repository)
            client = self._client_factory(client_request, settings)
            return _execute_candidate_question_task(
                repository=repository,
                retriever=retriever,
                client=client,
                snapshot_service=snapshot_service,
                audit=audit,
                question=question,
                prompt=prompt,
                resolved=resolved,
                request=request,
                runtime_config=runtime_config,
                candidate_run_id=candidate_run_id,
                isolate_unhandled_errors=not propagate_retryable_generation_errors,
                parallel_audit=True,
                propagate_retryable_generation_errors=propagate_retryable_generation_errors,
                progress=progress,
            )
        finally:
            connection.close()

    def _persist_adaptive_terminal_failure(
        self,
        *,
        settings: Any,
        resolved: ResolvedRunCandidatesRag,
        prompt: CandidatePromptRecord,
        question: CandidateQuestionRecord,
        runtime_config: ResolvedCandidateRuntimeConfig,
        candidate_run_id: int,
        error: Exception,
        audit: AuditLogger,
        progress: CandidateProgressReporter | None = None,
    ) -> CandidateQuestionTaskResult:
        connection = self._connect(settings.database_url)
        try:
            repository = self._repository_factory(connection)
            return _persist_unhandled_candidate_question_failure(
                repository=repository,
                audit=audit,
                question=question,
                prompt=prompt,
                resolved=resolved,
                runtime_config=runtime_config,
                candidate_run_id=candidate_run_id,
                error=error,
                progress=progress,
            )
        finally:
            connection.close()


class CandidateAdaptiveScheduler:
    """Conservative per-group scheduler for AV3 candidate execution."""

    def __init__(
        self,
        *,
        service: RunCandidatesRagService,
        execution_config: ResolvedCandidateExecutionConfig,
        audit: AuditLogger,
        progress: CandidateProgressReporter,
        stop_state: CandidateStopState,
    ) -> None:
        self.service = service
        self.execution_config = execution_config
        self.audit = audit
        self.progress = progress
        self.stop_state = stop_state
        self.groups: dict[CandidateAdaptiveGroupKey, CandidateAdaptiveGroupState] = {}

    def run(
        self,
        *,
        settings: Any,
        request: RunCandidatesRagRequest,
        client_request: RunCandidatesRagRequest,
        resolved: ResolvedRunCandidatesRag,
        prompt: CandidatePromptRecord,
        questions: list[CandidateQuestionRecord],
        runtime_config: ResolvedCandidateRuntimeConfig,
        candidate_run_id: int,
    ) -> list[CandidateQuestionTaskResult]:
        if not questions:
            return []

        group_key = candidate_adaptive_group_key(runtime_config)
        self._ensure_group(group_key, candidate_run_id=candidate_run_id)
        pending = [
            CandidateAdaptiveQueuedTask(
                question=question,
                group_key=group_key,
                ready_at=self.service._monotonic(),
            )
            for question in sorted(questions, key=lambda item: (item.question_sequence, item.question_id))
        ]
        max_workers = max(1, sum(group.max_concurrency for group in self.groups.values()))
        results: list[CandidateQuestionTaskResult] = []
        futures: dict[Future[CandidateAdaptiveCompletedTask], CandidateAdaptiveQueuedTask] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pending or futures:
                if self.stop_state.check():
                    self.stop_state.mark_not_started([queued.question for queued in pending])
                    pending.clear()
                if self._submit_ready(
                    pending=pending,
                    futures=futures,
                    executor=executor,
                    settings=settings,
                    request=request,
                    client_request=client_request,
                    resolved=resolved,
                    prompt=prompt,
                    runtime_config=runtime_config,
                    candidate_run_id=candidate_run_id,
                ):
                    continue

                if not futures:
                    delay = self._next_delay(pending)
                    if delay > 0:
                        self.service._sleep(delay)
                    continue

                done, _ = wait(
                    futures.keys(),
                    timeout=self._next_wait_timeout(pending),
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    continue
                for future in done:
                    queued = futures.pop(future)
                    state = self.groups[queued.group_key]
                    state.in_flight -= 1
                    try:
                        completed = future.result()
                    except Exception as error:
                        terminal_results = self._handle_exception(
                            error=error,
                            queued=queued,
                            pending=pending,
                            state=state,
                            settings=settings,
                            resolved=resolved,
                            prompt=prompt,
                            runtime_config=runtime_config,
                            candidate_run_id=candidate_run_id,
                        )
                        results.extend(terminal_results)
                        for item in terminal_results:
                            self.progress.emit_question_terminal(item)
                    else:
                        terminal_results = self._handle_result(
                            completed.result,
                            queued=queued,
                            pending=pending,
                            state=state,
                            settings=settings,
                            resolved=resolved,
                            prompt=prompt,
                            runtime_config=runtime_config,
                            candidate_run_id=candidate_run_id,
                        )
                        results.extend(terminal_results)
                        for item in terminal_results:
                            self.progress.emit_question_terminal(item)
        self._audit_final(candidate_run_id)
        return results

    def summary(self) -> dict[str, Any]:
        states = list(self.groups.values())
        if not states:
            return {
                "successes": 0,
                "failures": 0,
                "timeouts": 0,
                "rate_limits": 0,
                "transient_failures": 0,
                "non_retryable_failures": 0,
                "retries": 0,
                "requeued": 0,
                "final_concurrency": 1,
            }
        return {
            "successes": sum(state.successes for state in states),
            "failures": sum(state.failures for state in states),
            "timeouts": sum(state.timeouts for state in states),
            "rate_limits": sum(state.rate_limits for state in states),
            "transient_failures": sum(state.transient_failures for state in states),
            "non_retryable_failures": sum(state.non_retryable_failures for state in states),
            "retries": sum(state.retries for state in states),
            "requeued": sum(state.requeued for state in states),
            "final_concurrency": max(state.final_concurrency for state in states),
            "groups": [
                {
                    "av3_provider": state.key.av3_provider,
                    "base_url": state.key.base_url,
                    "api_key": state.key.api_key_fingerprint,
                    "model": state.key.model_name,
                    "current_concurrency": state.current_concurrency,
                    "max_concurrency": state.max_concurrency,
                    "consecutive_successes": state.consecutive_successes,
                    "successes": state.successes,
                    "failures": state.failures,
                    "timeouts": state.timeouts,
                    "rate_limits": state.rate_limits,
                    "transient_failures": state.transient_failures,
                    "non_retryable_failures": state.non_retryable_failures,
                    "retries": state.retries,
                    "requeued": state.requeued,
                    "disabled": state.disabled,
                    "final_concurrency": state.final_concurrency,
                }
                for state in states
            ],
        }

    def _submit_ready(
        self,
        *,
        pending: list[CandidateAdaptiveQueuedTask],
        futures: dict[Future[CandidateAdaptiveCompletedTask], CandidateAdaptiveQueuedTask],
        executor: ThreadPoolExecutor,
        settings: Any,
        request: RunCandidatesRagRequest,
        client_request: RunCandidatesRagRequest,
        resolved: ResolvedRunCandidatesRag,
        prompt: CandidatePromptRecord,
        runtime_config: ResolvedCandidateRuntimeConfig,
        candidate_run_id: int,
    ) -> bool:
        now = self.service._monotonic()
        submitted = False
        index = 0
        while index < len(pending):
            if self.stop_state.check():
                self.stop_state.mark_not_started([queued.question for queued in pending[index:]])
                del pending[index:]
                return submitted
            queued = pending[index]
            state = self.groups[queued.group_key]
            if state.disabled:
                pending.pop(index)
                continue
            if state.in_flight >= state.current_concurrency:
                index += 1
                continue
            if now < queued.ready_at or now < state.cooldown_until:
                index += 1
                continue
            pending.pop(index)
            state.in_flight += 1
            futures[
                executor.submit(
                    self._execute_queued_task,
                    queued=queued,
                    settings=settings,
                    request=request,
                    client_request=client_request,
                    resolved=resolved,
                    prompt=prompt,
                    runtime_config=runtime_config,
                    candidate_run_id=candidate_run_id,
                )
            ] = queued
            submitted = True
        return submitted

    def _execute_queued_task(
        self,
        *,
        queued: CandidateAdaptiveQueuedTask,
        settings: Any,
        request: RunCandidatesRagRequest,
        client_request: RunCandidatesRagRequest,
        resolved: ResolvedRunCandidatesRag,
        prompt: CandidatePromptRecord,
        runtime_config: ResolvedCandidateRuntimeConfig,
        candidate_run_id: int,
    ) -> CandidateAdaptiveCompletedTask:
        result = self.service._run_candidate_question_worker(
            settings=settings,
            request=request,
            client_request=client_request,
            resolved=resolved,
            prompt=prompt,
            question=queued.question,
            runtime_config=runtime_config,
            candidate_run_id=candidate_run_id,
            audit=self.audit,
            propagate_retryable_generation_errors=True,
            progress=self.progress,
        )
        return CandidateAdaptiveCompletedTask(queued=queued, result=result)

    def _handle_result(
        self,
        result: CandidateQuestionTaskResult,
        *,
        queued: CandidateAdaptiveQueuedTask,
        pending: list[CandidateAdaptiveQueuedTask],
        state: CandidateAdaptiveGroupState,
        settings: Any,
        resolved: ResolvedRunCandidatesRag,
        prompt: CandidatePromptRecord,
        runtime_config: ResolvedCandidateRuntimeConfig,
        candidate_run_id: int,
    ) -> list[CandidateQuestionTaskResult]:
        question_result = result.question_result
        if question_result.status == "success":
            self._handle_success(state, queued=queued, candidate_run_id=candidate_run_id)
            return [result]
        if question_result.status == "skipped":
            return [result]

        classification = classify_candidate_error_message(question_result.error_message or "")
        state.failures += 1
        state.consecutive_successes = 0
        if classification.fatal_group:
            state.non_retryable_failures += 1
            discarded = self._disable_group(
                state=state,
                pending=pending,
                reason=classification.reason,
                candidate_run_id=candidate_run_id,
                question=queued.question,
                attempt=queued.attempt,
                error_message=question_result.error_message or classification.reason,
            )
            terminal_results = [result]
            for discarded_task in discarded:
                terminal_results.append(
                    self.service._persist_adaptive_terminal_failure(
                        settings=settings,
                        resolved=resolved,
                        prompt=prompt,
                        question=discarded_task.question,
                        runtime_config=runtime_config,
                        candidate_run_id=candidate_run_id,
                        error=RuntimeError(question_result.error_message or classification.reason),
                        audit=self.audit,
                        progress=self.progress,
                    )
                )
            return terminal_results

        state.non_retryable_failures += 1
        self._audit_task_failed(
            state=state,
            candidate_run_id=candidate_run_id,
            question=queued.question,
            attempt=queued.attempt,
            reason=classification.reason,
            error_message=question_result.error_message or "",
        )
        return [result]

    def _handle_exception(
        self,
        *,
        error: Exception,
        queued: CandidateAdaptiveQueuedTask,
        pending: list[CandidateAdaptiveQueuedTask],
        state: CandidateAdaptiveGroupState,
        settings: Any,
        resolved: ResolvedRunCandidatesRag,
        prompt: CandidatePromptRecord,
        runtime_config: ResolvedCandidateRuntimeConfig,
        candidate_run_id: int,
    ) -> list[CandidateQuestionTaskResult]:
        classification = classify_candidate_error(error)
        state.failures += 1
        state.consecutive_successes = 0
        if classification.timeout:
            state.timeouts += 1
        if classification.rate_limit:
            state.rate_limits += 1
        if classification.retryable:
            state.transient_failures += 1
        else:
            state.non_retryable_failures += 1

        if not classification.retryable or queued.attempt >= self.execution_config.adaptive_max_retries:
            if classification.fatal_group:
                discarded = self._disable_group(
                    state=state,
                    pending=pending,
                    reason=classification.reason,
                    candidate_run_id=candidate_run_id,
                    question=queued.question,
                    attempt=queued.attempt,
                    error_message=str(error),
                )
            else:
                discarded = []
                self._audit_task_failed(
                    state=state,
                    candidate_run_id=candidate_run_id,
                    question=queued.question,
                    attempt=queued.attempt,
                    reason=classification.reason,
                    error_message=str(error),
                )
            terminal_results = [
                self.service._persist_adaptive_terminal_failure(
                    settings=settings,
                    resolved=resolved,
                    prompt=prompt,
                    question=queued.question,
                    runtime_config=runtime_config,
                    candidate_run_id=candidate_run_id,
                    error=error,
                    audit=self.audit,
                    progress=self.progress,
                )
            ]
            for discarded_task in discarded:
                terminal_results.append(
                    self.service._persist_adaptive_terminal_failure(
                        settings=settings,
                        resolved=resolved,
                        prompt=prompt,
                        question=discarded_task.question,
                        runtime_config=runtime_config,
                        candidate_run_id=candidate_run_id,
                        error=error,
                        audit=self.audit,
                        progress=self.progress,
                    )
                )
            return terminal_results

        if self.stop_state.check():
            self.stop_state.mark_not_started([queued.question])
            self.audit.event(
                AuditEvent(
                    "candidate_adaptive_requeue_suppressed_due_to_stop",
                    self._event_detail(
                        state=state,
                        candidate_run_id=candidate_run_id,
                        question=queued.question,
                        attempt=queued.attempt + 1,
                        reason=classification.reason,
                        backoff_seconds=0,
                    ),
                )
            )
            return []

        self._reduce_concurrency(
            state=state,
            reason=classification.reason,
            candidate_run_id=candidate_run_id,
            question=queued.question,
            attempt=queued.attempt,
        )
        backoff = self._backoff_seconds(queued.attempt)
        ready_at = self.service._monotonic() + backoff
        state.cooldown_until = max(state.cooldown_until, ready_at)
        state.retries += 1
        state.requeued += 1
        retried = CandidateAdaptiveQueuedTask(
            question=queued.question,
            group_key=queued.group_key,
            attempt=queued.attempt + 1,
            ready_at=ready_at,
        )
        pending.append(retried)
        pending.sort(key=lambda item: (item.ready_at, item.question.question_sequence, item.question.question_id))
        self.audit.event(
            AuditEvent(
                "candidate_adaptive_task_requeued",
                self._event_detail(
                    state=state,
                    candidate_run_id=candidate_run_id,
                    question=queued.question,
                    attempt=retried.attempt,
                    reason=classification.reason,
                    backoff_seconds=backoff,
                    extra=f"error_type={type(error).__name__}",
                ),
            )
        )
        return []

    def _handle_success(
        self,
        state: CandidateAdaptiveGroupState,
        *,
        queued: CandidateAdaptiveQueuedTask,
        candidate_run_id: int,
    ) -> None:
        state.successes += 1
        state.consecutive_successes += 1
        if (
            state.consecutive_successes >= self.execution_config.adaptive_success_threshold
            and state.current_concurrency < state.max_concurrency
        ):
            old = state.current_concurrency
            state.current_concurrency += 1
            state.consecutive_successes = 0
            self.audit.event(
                AuditEvent(
                    "candidate_adaptive_increased",
                    self._event_detail(
                        state=state,
                        candidate_run_id=candidate_run_id,
                        question=queued.question,
                        attempt=queued.attempt,
                        reason=f"success_threshold from={old} to={state.current_concurrency}",
                        backoff_seconds=0,
                    ),
                )
            )

    def _reduce_concurrency(
        self,
        *,
        state: CandidateAdaptiveGroupState,
        reason: str,
        candidate_run_id: int,
        question: CandidateQuestionRecord,
        attempt: int,
    ) -> None:
        old = state.current_concurrency
        state.current_concurrency = max(1, state.current_concurrency - 1)
        self.audit.event(
            AuditEvent(
                "candidate_adaptive_reduced",
                self._event_detail(
                    state=state,
                    candidate_run_id=candidate_run_id,
                    question=question,
                    attempt=attempt,
                    reason=f"{reason} from={old} to={state.current_concurrency}",
                    backoff_seconds=0,
                ),
            )
        )

    def _disable_group(
        self,
        *,
        state: CandidateAdaptiveGroupState,
        pending: list[CandidateAdaptiveQueuedTask],
        reason: str,
        candidate_run_id: int,
        question: CandidateQuestionRecord,
        attempt: int,
        error_message: str,
    ) -> list[CandidateAdaptiveQueuedTask]:
        state.disabled = True
        discarded: list[CandidateAdaptiveQueuedTask] = []
        index = 0
        while index < len(pending):
            if pending[index].group_key == state.key:
                discarded.append(pending.pop(index))
                continue
            index += 1
        state.failures += len(discarded)
        state.non_retryable_failures += len(discarded)
        self._audit_task_failed(
            state=state,
            candidate_run_id=candidate_run_id,
            question=question,
            attempt=attempt,
            reason=reason,
            error_message=error_message,
        )
        self.audit.event(
            AuditEvent(
                "candidate_adaptive_group_disabled",
                self._event_detail(
                    state=state,
                    candidate_run_id=candidate_run_id,
                    question=question,
                    attempt=attempt,
                    reason=f"{reason} discarded_pending={len(discarded)}",
                    backoff_seconds=0,
                ),
            )
        )
        return discarded

    def _audit_task_failed(
        self,
        *,
        state: CandidateAdaptiveGroupState,
        candidate_run_id: int,
        question: CandidateQuestionRecord,
        attempt: int,
        reason: str,
        error_message: str,
    ) -> None:
        self.audit.event(
            AuditEvent(
                "candidate_adaptive_task_failed",
                self._event_detail(
                    state=state,
                    candidate_run_id=candidate_run_id,
                    question=question,
                    attempt=attempt,
                    reason=reason,
                    backoff_seconds=0,
                    extra=f"error={_safe_error_message(error_message)}",
                ),
            )
        )

    def _ensure_group(self, key: CandidateAdaptiveGroupKey, *, candidate_run_id: int) -> None:
        if key in self.groups:
            return
        initial = min(
            self.execution_config.adaptive_initial_concurrency,
            self.execution_config.adaptive_max_concurrency,
        )
        state = CandidateAdaptiveGroupState(
            key=key,
            current_concurrency=initial,
            max_concurrency=self.execution_config.adaptive_max_concurrency,
        )
        self.groups[key] = state
        self.audit.event(
            AuditEvent(
                "candidate_adaptive_initial",
                (
                    f"candidate_run_id={candidate_run_id} {key.label} "
                    f"question_id=- question_sequence=- attempt=0 "
                    f"current_concurrency={state.current_concurrency} max_concurrency={state.max_concurrency} "
                    "reason=initial backoff_seconds=0"
                ),
            )
        )

    def _audit_final(self, candidate_run_id: int) -> None:
        for state in self.groups.values():
            self.audit.event(
                AuditEvent(
                    "candidate_adaptive_final",
                    (
                        f"candidate_run_id={candidate_run_id} {state.key.label} "
                        f"question_id=- question_sequence=- attempt=- "
                        f"current_concurrency={state.current_concurrency} max_concurrency={state.max_concurrency} "
                        "reason=final backoff_seconds=0 "
                        f"successes={state.successes} failures={state.failures} timeouts={state.timeouts} "
                        f"rate_limits={state.rate_limits} transient_failures={state.transient_failures} "
                        f"non_retryable_failures={state.non_retryable_failures} retries={state.retries} "
                        f"requeued={state.requeued} disabled={state.disabled} "
                        f"final_concurrency={state.final_concurrency}"
                    ),
                )
            )

    def _backoff_seconds(self, attempt: int) -> float:
        base = self.execution_config.adaptive_base_backoff_seconds
        maximum = self.execution_config.adaptive_max_backoff_seconds
        return min(maximum, base * (2 ** attempt))

    def _next_delay(self, pending: list[CandidateAdaptiveQueuedTask]) -> float:
        if not pending:
            return 0.0
        now = self.service._monotonic()
        return max(0.0, min(max(item.ready_at, self.groups[item.group_key].cooldown_until) for item in pending) - now)

    def _next_wait_timeout(self, pending: list[CandidateAdaptiveQueuedTask]) -> float:
        delay = self._next_delay(pending)
        if delay <= 0:
            return 0.05
        return min(delay, 0.05)

    def _event_detail(
        self,
        *,
        state: CandidateAdaptiveGroupState,
        candidate_run_id: int,
        question: CandidateQuestionRecord,
        attempt: int,
        reason: str,
        backoff_seconds: float,
        extra: str | None = None,
    ) -> str:
        detail = (
            f"candidate_run_id={candidate_run_id} {state.key.label} "
            f"question_id={question.question_id} question_sequence={question.question_sequence} "
            f"attempt={attempt} current_concurrency={state.current_concurrency} "
            f"max_concurrency={state.max_concurrency} reason={reason} "
            f"backoff_seconds={backoff_seconds:g}"
        )
        if extra:
            detail = f"{detail} {extra}"
        return detail


def candidate_adaptive_group_key(runtime_config: ResolvedCandidateRuntimeConfig) -> CandidateAdaptiveGroupKey:
    return CandidateAdaptiveGroupKey(
        av3_provider=runtime_config.av3_provider,
        base_url=_safe_base_url(runtime_config.base_url),
        api_key_fingerprint=_api_key_fingerprint(runtime_config.api_key),
        model_name=runtime_config.model_name,
    )


def _safe_base_url(value: str | None) -> str:
    if not value:
        return "<unset>"
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value.split("?", 1)[0] or "<unset>"
    hostname = parsed.hostname or parsed.netloc
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _api_key_fingerprint(value: str | None) -> str:
    if not value:
        return "<unset>"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"<set:{digest}>"


def classify_candidate_error(error: Exception) -> CandidateErrorClassification:
    status_code = getattr(error, "status_code", None)
    message = str(error)
    return classify_candidate_error_message(message, status_code=status_code, error=error)


def classify_candidate_error_message(
    message: str,
    *,
    status_code: int | None = None,
    error: Exception | None = None,
) -> CandidateErrorClassification:
    normalized = message.casefold()
    if parse_candidate_runtime_observation(message) is not None:
        return CandidateErrorClassification(reason="context_window", retryable=False)
    if status_code == 429 or "http 429" in normalized:
        return CandidateErrorClassification(reason="http_429", retryable=True, rate_limit=True)
    if status_code in {502, 503, 504} or any(f"http {code}" in normalized for code in (502, 503, 504)):
        return CandidateErrorClassification(reason=f"http_{status_code}" if status_code else "http_5xx", retryable=True)
    if isinstance(error, TimeoutError) or _message_contains_any(normalized, ("timed out", "timeout", "socket timeout")):
        return CandidateErrorClassification(reason="timeout", retryable=True, timeout=True)
    if _message_contains_any(
        normalized,
        (
            "connection reset",
            "temporarily unavailable",
            "temporary failure",
            "network is unreachable",
            "connection aborted",
            "connection refused",
            "remote candidate request failed",
        ),
    ):
        return CandidateErrorClassification(reason="temporary_network_failure", retryable=True)
    if _message_contains_any(
        normalized,
        (
            "api_key is required",
            "api key is required",
            "missing api key",
            "invalid api key",
            "invalid provider credential",
            "unauthorized",
            "forbidden",
            "access denied",
            "gated",
            "model not found",
            "unsupported candidate provider",
            "unsupported av3_provider",
            "no av3 candidate assignment",
            "no runnable av3 assignment",
            "invalid request",
            "schema",
            "prompt",
            "config",
        ),
    ):
        return CandidateErrorClassification(reason="non_retryable_config_or_access", retryable=False, fatal_group=True)
    if status_code is not None and 500 <= status_code <= 599:
        return CandidateErrorClassification(reason=f"http_{status_code}", retryable=True)
    return CandidateErrorClassification(reason="non_retryable_error", retryable=False)


def _message_contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def _safe_error_message(value: str) -> str:
    if not value:
        return "<empty>"
    return "<redacted>"


def _resolve_audit_path(value: str | None) -> Path:
    if value:
        return Path(value)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("logs") / f"candidate-rag-{timestamp}.log"


def _build_execution_summary(
    *,
    dataset: str,
    batch_size: int,
    model_name: str,
    provider: str,
    question_sequence_start: int | None,
    question_sequence_end: int | None,
    question_id: int | None,
    candidate_execution_strategy: str = "sequential",
    candidate_parallel_max_workers: int = 2,
    candidate_adaptive_initial_concurrency: int = 1,
    candidate_adaptive_max_concurrency: int = 2,
    candidate_adaptive_success_threshold: int = 3,
    candidate_adaptive_max_retries: int = 2,
    candidate_adaptive_base_backoff_seconds: float = 2.0,
    candidate_adaptive_max_backoff_seconds: float = 60.0,
) -> str:
    lines = [
        f"Dataset: {dataset}\n"
        f"Candidate model: {model_name}\n"
        f"Provider: {provider}\n"
        f"Batch size: {batch_size}\n"
        f"Question id: {question_id or '-'}\n"
        f"Question range: {_format_question_range(question_sequence_start, question_sequence_end)}\n"
        f"Candidate execution strategy: {candidate_execution_strategy}\n"
        f"Candidate parallel max workers: {candidate_parallel_max_workers}"
    ]
    if candidate_execution_strategy == "adaptive":
        lines.extend(
            [
                f"Candidate adaptive initial/max concurrency: {candidate_adaptive_initial_concurrency}/{candidate_adaptive_max_concurrency}",
                f"Candidate adaptive success threshold: {candidate_adaptive_success_threshold}",
                f"Candidate adaptive max retries: {candidate_adaptive_max_retries}",
                (
                    "Candidate adaptive backoff seconds: "
                    f"{candidate_adaptive_base_backoff_seconds:g}..{candidate_adaptive_max_backoff_seconds:g}"
                ),
            ]
        )
    return "\n".join(lines)


def _format_question_range(start: int | None, end: int | None) -> str:
    if start is None and end is None:
        return "-"
    if start is not None and end is not None:
        return f"{start}-{end}"
    if start is not None:
        return f"{start}-fim"
    return f"1-{end}"


def _run_metadata(
    resolved: ResolvedRunCandidatesRag,
    vector_base: Any,
    prompt: CandidatePromptRecord,
    request: RunCandidatesRagRequest,
    runtime_config: ResolvedCandidateRuntimeConfig | None = None,
    execution_config: ResolvedCandidateExecutionConfig | None = None,
) -> dict[str, Any]:
    return {
        "retrieval_name": getattr(vector_base, "retrieval_name", None),
        "embedding_model": getattr(vector_base, "embedding_model", None),
        "top_k": getattr(vector_base, "top_k", None),
        "prompt_version": prompt.version,
        "question_selection": {
            "question_id": resolved.question_id,
            "question_sequence_start": resolved.question_sequence_start,
            "question_sequence_end": resolved.question_sequence_end,
        },
        "skip_existing_successful": resolved.skip_existing_successful,
        "save_raw_response": request.save_raw_response,
        "candidate_execution": None
        if execution_config is None
        else {
            "strategy": execution_config.strategy,
            "parallel_max_workers": execution_config.parallel_max_workers,
            "adaptive_initial_concurrency": execution_config.adaptive_initial_concurrency,
            "adaptive_max_concurrency": execution_config.adaptive_max_concurrency,
            "adaptive_success_threshold": execution_config.adaptive_success_threshold,
            "adaptive_max_retries": execution_config.adaptive_max_retries,
            "adaptive_base_backoff_seconds": execution_config.adaptive_base_backoff_seconds,
            "adaptive_max_backoff_seconds": execution_config.adaptive_max_backoff_seconds,
        },
        "candidate_runtime": None
        if runtime_config is None
        else {
            "technical_provider": runtime_config.technical_provider,
            "av3_provider": runtime_config.av3_provider,
            "base_url": runtime_config.base_url,
            "api_key": "<set>" if runtime_config.api_key else "<missing>",
            "temperature": runtime_config.temperature,
            "top_p": runtime_config.top_p,
            "default_max_output_tokens": runtime_config.default_max_output_tokens,
            "max_output_tokens_cap": runtime_config.max_output_tokens_cap,
            "requested_max_tokens": runtime_config.requested_max_tokens,
            "max_tokens": runtime_config.max_tokens,
            "context_window_tokens": runtime_config.context_window_tokens,
            "safety_margin_tokens": runtime_config.safety_margin_tokens,
            "chars_per_token_estimate": runtime_config.chars_per_token_estimate,
            "prompt_budget_utilization": runtime_config.prompt_budget_utilization,
            "model_profile_source": runtime_config.model_profile_source,
        },
        "candidate_runtime_profile": None
        if runtime_config is None
        else _candidate_runtime_profile_metadata(runtime_config),
    }


def _render_prompt(
    *,
    question: CandidateQuestionRecord,
    retrieval_result: RagRetrievalResult,
    prompt: CandidatePromptRecord,
) -> str:
    return build_candidate_prompt(
        CandidatePromptContext(
            question_id=question.question_id,
            dataset_name=question.dataset,
            question_text=question.question_text,
            retrieved_chunks=list(retrieval_result.chunks),
            alternatives=question.alternatives,
            retrieval_run_id=retrieval_result.retrieval_run_id,
            retrieval_name=retrieval_result.retrieval_name,
            top_k=retrieval_result.top_k,
        ),
        template=prompt,
    )


def _execute_candidate_question_task(
    *,
    repository: CandidateRunRepositoryProtocol,
    retriever: Any,
    client: CandidateClient,
    snapshot_service: RagContextSnapshotService,
    audit: AuditLogger,
    question: CandidateQuestionRecord,
    prompt: CandidatePromptRecord,
    resolved: ResolvedRunCandidatesRag,
    request: RunCandidatesRagRequest,
    runtime_config: ResolvedCandidateRuntimeConfig,
    candidate_run_id: int,
    isolate_unhandled_errors: bool,
    parallel_audit: bool,
    propagate_retryable_generation_errors: bool = False,
    progress: CandidateProgressReporter | None = None,
) -> CandidateQuestionTaskResult:
    if parallel_audit:
        audit.event(
            AuditEvent(
                "candidate_parallel_task_started",
                (
                    f"candidate_run_id={candidate_run_id} question_id={question.question_id} "
                    f"question_sequence={question.question_sequence} model={resolved.model_name}"
                ),
            )
        )
    try:
        result = _execute_candidate_question_task_inner(
            repository=repository,
            retriever=retriever,
            client=client,
            snapshot_service=snapshot_service,
            audit=audit,
            question=question,
            prompt=prompt,
            resolved=resolved,
            request=request,
            runtime_config=runtime_config,
            candidate_run_id=candidate_run_id,
            propagate_retryable_generation_errors=propagate_retryable_generation_errors,
            progress=progress,
        )
    except Exception as error:
        if not isolate_unhandled_errors:
            raise
        result = _persist_unhandled_candidate_question_failure(
            repository=repository,
            audit=audit,
            question=question,
            prompt=prompt,
            resolved=resolved,
            runtime_config=runtime_config,
            candidate_run_id=candidate_run_id,
            error=error,
            progress=progress,
        )

    if parallel_audit:
        question_result = result.question_result
        if question_result.status == "failed":
            audit.event(
                AuditEvent(
                    "candidate_parallel_task_failed",
                    (
                        f"candidate_run_id={candidate_run_id} question_id={question.question_id} "
                        f"question_sequence={question.question_sequence} model={resolved.model_name} "
                        "status=failed"
                    ),
                )
            )
        audit.event(
            AuditEvent(
                "candidate_parallel_task_finished",
                (
                    f"candidate_run_id={candidate_run_id} question_id={question.question_id} "
                    f"question_sequence={question.question_sequence} model={resolved.model_name} "
                    f"status={question_result.status} latency_ms={question_result.latency_ms}"
                ),
            )
        )
    return result


def _execute_candidate_question_task_inner(
    *,
    repository: CandidateRunRepositoryProtocol,
    retriever: Any,
    client: CandidateClient,
    snapshot_service: RagContextSnapshotService,
    audit: AuditLogger,
    question: CandidateQuestionRecord,
    prompt: CandidatePromptRecord,
    resolved: ResolvedRunCandidatesRag,
    request: RunCandidatesRagRequest,
    runtime_config: ResolvedCandidateRuntimeConfig,
    candidate_run_id: int,
    propagate_retryable_generation_errors: bool = False,
    progress: CandidateProgressReporter | None = None,
) -> CandidateQuestionTaskResult:
    audit.event(
        AuditEvent(
            "question_started",
            (
                f"candidate_run_id={candidate_run_id} question_id={question.question_id} "
                f"sequence={question.question_sequence} dataset={question.dataset}"
            ),
        )
    )
    if progress is not None:
        progress.emit("candidate_question_started", question=question, status="running")
    if resolved.skip_existing_successful and repository.successful_candidate_answer_exists(
        dataset=resolved.dataset,
        model_name=resolved.model_name,
        question_id=question.question_id,
        exclude_candidate_run_id=candidate_run_id,
    ):
        audit.event(
            AuditEvent(
                "question_skipped",
                (
                    f"candidate_run_id={candidate_run_id} question_id={question.question_id} "
                    "reason=existing_successful_answer"
                ),
            )
        )
        if progress is not None:
            progress.emit(
                "candidate_question_skipped",
                question=question,
                status="skipped",
                metadata={"reason": "existing_successful_answer"},
            )
        return CandidateQuestionTaskResult(
            question_result=CandidateQuestionRunResult(
                question_id=question.question_id,
                question_sequence=question.question_sequence,
                status="skipped",
                error_message="existing_successful_answer",
            ),
            budget=None,
            runtime_config=runtime_config,
        )

    retrieval_result = retriever.retrieve_for_question(
        question_id=question.question_id,
        dataset=resolved.dataset,
    )
    audit.event(
        AuditEvent(
            "retrieval_finished",
            (
                f"candidate_run_id={candidate_run_id} question_id={question.question_id} "
                f"status={retrieval_result.status} chunks={len(retrieval_result.chunks)}"
            ),
        )
    )
    if progress is not None:
        progress.emit(
            "candidate_retrieval_finished",
            question=question,
            status=retrieval_result.status,
            metadata={
                "retrieval_status": retrieval_result.status,
                "retrieved_chunk_count": len(retrieval_result.chunks),
                "retrieval_run_id": retrieval_result.retrieval_run_id,
            },
        )
    if retrieval_result.status != "success":
        rendered_prompt = _render_prompt(
            question=question,
            retrieval_result=retrieval_result,
            prompt=prompt,
        )
        stored_answer = repository.persist_candidate_answer(
            answer=CandidateAnswerRecord(
                candidate_answer_id=None,
                candidate_run_id=candidate_run_id,
                question_id=question.question_id,
                model_name=resolved.model_name,
                rendered_prompt=rendered_prompt,
                status="failed",
                error_message=f"Retrieval failed: {retrieval_result.status}",
            )
        )
        audit.event(
            AuditEvent(
                "answer_persisted",
                (
                    f"candidate_answer_id={stored_answer.candidate_answer_id} "
                    f"question_id={question.question_id} status=failed"
                ),
            )
        )
        if progress is not None:
            progress.emit(
                "candidate_answer_persisted",
                question=question,
                status="failed",
                metadata={
                    "candidate_answer_id": stored_answer.candidate_answer_id,
                    "status": "failed",
                },
            )
            progress.emit(
                "candidate_question_failed",
                question=question,
                status="failed",
                message="Retrieval failed",
                metadata={
                    "error_class": "RetrievalError",
                    "error_message": f"Retrieval failed: {retrieval_result.status}",
                },
            )
        return CandidateQuestionTaskResult(
            question_result=CandidateQuestionRunResult(
                question_id=question.question_id,
                question_sequence=question.question_sequence,
                status="failed",
                retrieval_status=retrieval_result.status,
                candidate_answer_id=stored_answer.candidate_answer_id,
                error_message=stored_answer.error_message,
            ),
            budget=None,
            runtime_config=runtime_config,
        )

    outcome = _execute_candidate_generation(
        repository=repository,
        client=client,
        audit=audit,
        question=question,
        retrieval_result=retrieval_result,
        prompt=prompt,
        resolved=resolved,
        request=request,
        runtime_config=runtime_config,
        candidate_run_id=candidate_run_id,
        propagate_retryable_generation_errors=propagate_retryable_generation_errors,
        progress=progress,
    )
    if outcome.budget is not None:
        _log_candidate_prompt_budget(
            audit=audit,
            question=question,
            budget=outcome.budget,
        )
    stored_answer = outcome.answer
    snapshot_rows: list[CandidateAnswerContextChunkRecord] = []
    if stored_answer.status == "success":
        try:
            stored_answer, snapshot_rows = repository.persist_successful_candidate_answer_with_context_snapshot(
                answer=stored_answer,
                retrieval_result=outcome.retrieval_result_for_prompt,
            )
        except Exception as error:
            audit.event(
                AuditEvent(
                    "snapshot_persistence_failed",
                    (
                        f"candidate_run_id={candidate_run_id} question_id={question.question_id} "
                        f"error_type={type(error).__name__}"
                    ),
                )
            )
            if progress is not None:
                progress.emit(
                    "candidate_question_failed",
                    question=question,
                    status="failed",
                    message="context_snapshot_persistence_failed",
                    metadata={
                        "error_class": type(error).__name__,
                        "error_message": "context_snapshot_persistence_failed",
                    },
                )
            return CandidateQuestionTaskResult(
                question_result=CandidateQuestionRunResult(
                    question_id=question.question_id,
                    question_sequence=question.question_sequence,
                    status="failed",
                    retrieval_status=retrieval_result.status,
                    candidate_answer_id=None,
                    error_message="context_snapshot_persistence_failed",
                ),
                budget=outcome.budget,
                runtime_config=outcome.runtime_config,
            )
        audit.event(
            AuditEvent(
                "answer_persisted",
                (
                    f"candidate_answer_id={stored_answer.candidate_answer_id} "
                    f"question_id={question.question_id} status=success"
                ),
            )
        )
    else:
        assert stored_answer.candidate_answer_id is not None
    if progress is not None:
        progress.emit(
            "candidate_answer_persisted",
            question=question,
            status=stored_answer.status,
            metadata={
                "candidate_answer_id": stored_answer.candidate_answer_id,
                "status": stored_answer.status,
            },
        )
    if stored_answer.status != "success":
        snapshot_rows = snapshot_service.persist_retrieval_snapshot(
            candidate_answer_id=stored_answer.candidate_answer_id,
            retrieval_result=outcome.retrieval_result_for_prompt,
        )
    audit.event(
        AuditEvent(
            "snapshot_persisted",
            (
                f"candidate_answer_id={stored_answer.candidate_answer_id} "
                f"question_id={question.question_id} count={len(snapshot_rows)}"
            ),
        )
    )
    if progress is not None:
        if stored_answer.status == "failed":
            progress.emit(
                "candidate_question_failed",
                question=question,
                status="failed",
                message="Generation failed",
                metadata={
                    "error_class": "GenerationError",
                    "error_message": _safe_error_message(stored_answer.error_message or ""),
                },
            )
    if stored_answer.status == "success":
        question_result = CandidateQuestionRunResult(
            question_id=question.question_id,
            question_sequence=question.question_sequence,
            status="success",
            retrieval_status=retrieval_result.status,
            candidate_answer_id=stored_answer.candidate_answer_id,
            final_choice=stored_answer.final_choice,
            latency_ms=stored_answer.latency_ms,
        )
    else:
        question_result = CandidateQuestionRunResult(
            question_id=question.question_id,
            question_sequence=question.question_sequence,
            status="failed",
            retrieval_status=retrieval_result.status,
            candidate_answer_id=stored_answer.candidate_answer_id,
            error_message=stored_answer.error_message,
        )
    return CandidateQuestionTaskResult(
        question_result=question_result,
        budget=outcome.budget,
        runtime_config=outcome.runtime_config,
    )


def _persist_unhandled_candidate_question_failure(
    *,
    repository: CandidateRunRepositoryProtocol,
    audit: AuditLogger,
    question: CandidateQuestionRecord,
    prompt: CandidatePromptRecord,
    resolved: ResolvedRunCandidatesRag,
    runtime_config: ResolvedCandidateRuntimeConfig,
    candidate_run_id: int,
    error: Exception,
    progress: CandidateProgressReporter | None = None,
) -> CandidateQuestionTaskResult:
    retrieval_result = RagRetrievalResult(
        question_id=question.question_id,
        dataset=resolved.dataset,
        retrieval_run_id=None,
        retrieval_name=None,
        embedding_model=None,
        top_k=0,
        status="no_chunks_found",
        chunks=[],
    )
    stored_answer = repository.persist_candidate_answer(
        answer=CandidateAnswerRecord(
            candidate_answer_id=None,
            candidate_run_id=candidate_run_id,
            question_id=question.question_id,
            model_name=resolved.model_name,
            rendered_prompt=_render_prompt(
                question=question,
                retrieval_result=retrieval_result,
                prompt=prompt,
            ),
            status="failed",
            error_message=str(error),
        )
    )
    audit.event(
        AuditEvent(
            "answer_persisted",
            (
                f"candidate_answer_id={stored_answer.candidate_answer_id} "
                f"question_id={question.question_id} status=failed"
            ),
        )
    )
    if progress is not None:
        progress.emit(
            "candidate_answer_persisted",
            question=question,
            status="failed",
            metadata={
                "candidate_answer_id": stored_answer.candidate_answer_id,
                "status": "failed",
            },
        )
        progress.emit(
            "candidate_question_failed",
            question=question,
            status="failed",
            message=type(error).__name__,
            metadata={
                "error_class": type(error).__name__,
                "error_message": _safe_error_message(str(error)),
            },
        )
    return CandidateQuestionTaskResult(
        question_result=CandidateQuestionRunResult(
            question_id=question.question_id,
            question_sequence=question.question_sequence,
            status="failed",
            retrieval_status=None,
            candidate_answer_id=stored_answer.candidate_answer_id,
            error_message=stored_answer.error_message,
        ),
        budget=None,
        runtime_config=runtime_config,
    )


def _extract_final_choice(answer_text: str, *, dataset: str) -> str | None:
    if dataset != "J2":
        return None
    marker = "ALTERNATIVA FINAL:"
    upper = answer_text.upper()
    index = upper.rfind(marker)
    if index >= 0:
        snippet = upper[index + len(marker):]
        for char in snippet:
            if char in {"A", "B", "C", "D", "E"}:
                return char
    for char in upper:
        if char in {"A", "B", "C", "D", "E"}:
            return char
    return None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_snapshot_service_factory(repository: CandidateRunRepositoryProtocol) -> RagContextSnapshotService:
    return RagContextSnapshotService(repository=repository)


def _default_client_factory(request: RunCandidatesRagRequest, _settings: Any) -> CandidateClient:
    if request.provider != "remote_http":
        raise ValueError(f"Unsupported candidate provider: {request.provider}.")
    return RemoteHttpCandidateClient(
        config=RemoteHttpCandidateClientConfig(
            base_url=(request.remote_candidate_base_url or "").strip(),
            api_key=(request.remote_candidate_api_key or "").strip(),
            provider=request.provider,
            timeout_seconds=request.remote_candidate_timeout_seconds,
            temperature=float(request.remote_candidate_temperature if request.remote_candidate_temperature is not None else 0.2),
            max_tokens=int(request.remote_candidate_max_tokens or 1024),
            top_p=float(request.remote_candidate_top_p if request.remote_candidate_top_p is not None else 0.9),
            openai_compatible=request.remote_candidate_openai_compatible,
            save_raw_response=request.save_raw_response,
        )
    )


def _with_remote_candidate_config(
    request: RunCandidatesRagRequest,
    runtime_config: ResolvedCandidateRuntimeConfig,
) -> RunCandidatesRagRequest:
    return RunCandidatesRagRequest(
        model_name=request.model_name,
        provider=request.provider,
        dataset=request.dataset,
        batch_size=request.batch_size,
        question_sequence_start=request.question_sequence_start,
        question_sequence_end=request.question_sequence_end,
        question_id=request.question_id,
        prompt_id=request.prompt_id,
        retrieval_run_id=request.retrieval_run_id,
        skip_existing_successful=request.skip_existing_successful,
        save_raw_response=request.save_raw_response,
        dry_run=request.dry_run,
        audit_log=request.audit_log,
        no_audit_animation=request.no_audit_animation,
        created_by=request.created_by,
        remote_candidate_base_url=runtime_config.base_url,
        remote_candidate_api_key=runtime_config.api_key,
        remote_candidate_timeout_seconds=request.remote_candidate_timeout_seconds,
        remote_candidate_temperature=runtime_config.temperature,
        remote_candidate_max_tokens=runtime_config.max_tokens,
        remote_candidate_top_p=runtime_config.top_p,
        remote_candidate_context_safety_margin_tokens=runtime_config.safety_margin_tokens,
        remote_candidate_context_window_tokens=runtime_config.context_window_tokens,
        remote_candidate_retry_on_context_window=runtime_config.retry_on_context_window,
        remote_candidate_openai_compatible=request.remote_candidate_openai_compatible,
        candidate_execution_strategy=request.candidate_execution_strategy,
        candidate_parallel_max_workers=request.candidate_parallel_max_workers,
        progress_callback=request.progress_callback,
    )


def _resolve_candidate_provider_config(
    *,
    settings: Any,
    assignment: CandidateModelAssignment,
    require_api_key: bool,
) -> ResolvedCandidateProviderConfig:
    provider = assignment.av3_provider.casefold()
    if provider == "openrouter":
        api_key = (getattr(settings, "openrouter_api_key", None) or "").strip() or None
        if require_api_key and not api_key:
            raise ValueError("OPENROUTER_KEY is required for openrouter candidate execution.")
        base_url = (getattr(settings, "openrouter_url", None) or "").strip()
        if not base_url:
            raise ValueError("OPENROUTER_URL is required for openrouter candidate execution.")
        return ResolvedCandidateProviderConfig(
            av3_provider=provider,
            base_url=base_url,
            api_key=api_key,
        )
    if provider == "featherless":
        api_key = (getattr(settings, "featherless_api_key", None) or "").strip() or None
        if require_api_key and not api_key:
            raise ValueError("FEATHERLESS_API is required for featherless candidate execution.")
        base_url = (getattr(settings, "featherless_url", None) or "").strip()
        if not base_url:
            raise ValueError("FEATHERLESS_URL is required for featherless candidate execution.")
        return ResolvedCandidateProviderConfig(
            av3_provider=provider,
            base_url=base_url,
            api_key=api_key,
        )
    if provider == "llama_cpp":
        raw_api_key = getattr(settings, "llama_cpp_api_key", None)
        api_key = None if raw_api_key is None else str(raw_api_key).strip()
        base_url = (getattr(settings, "llama_cpp_url", None) or "").strip()
        if not base_url:
            raise ValueError("LLAMA_CPP_URL is required for llama.cpp candidate execution.")
        return ResolvedCandidateProviderConfig(
            av3_provider=provider,
            base_url=base_url,
            api_key=api_key,
        )
    raise ValueError(
        f"Candidate model {assignment.av3_provider_model_id} is mapped to unsupported av3_provider={assignment.av3_provider}."
    )


def _resolve_candidate_runtime_config(
    *,
    repository: CandidateRunRepositoryProtocol,
    settings: Any,
    request: RunCandidatesRagRequest,
    resolved: ResolvedRunCandidatesRag,
    questions: list[CandidateQuestionRecord],
    require_api_key: bool,
) -> ResolvedCandidateRuntimeConfig:
    assignment = _resolve_candidate_assignment(
        repository=repository,
        dataset=resolved.dataset,
        model_name=resolved.model_name,
        questions=questions,
    )
    provider_config = _resolve_candidate_provider_config(
        settings=settings,
        assignment=assignment,
        require_api_key=require_api_key,
    )
    safety_margin_tokens = _resolve_candidate_context_safety_margin_tokens(settings=settings, request=request)
    persisted_profile = repository.get_candidate_model_runtime_profile(
        av3_provider=provider_config.av3_provider,
        provider_model_id=resolved.model_name,
    )
    profile = resolve_candidate_model_runtime_profile(
        provider=provider_config.av3_provider,
        model_name=resolved.model_name,
        safety_margin_tokens=safety_margin_tokens,
        context_window_tokens_override=_resolve_candidate_context_window_tokens(settings=settings, request=request),
        persisted_profile=persisted_profile,
    )
    raw_requested_max_tokens = _resolve_requested_candidate_max_tokens(settings=settings, request=request)
    requested_max_tokens = (
        int(raw_requested_max_tokens)
        if raw_requested_max_tokens is not None
        else int(profile.default_max_output_tokens)
    )
    max_tokens = resolve_candidate_max_output_tokens(
        profile=profile,
        requested_max_tokens=raw_requested_max_tokens,
    )
    temperature = _resolve_candidate_temperature(settings=settings, request=request)
    top_p = _resolve_candidate_top_p(settings=settings, request=request)
    if questions:
        prompt = repository.get_or_create_candidate_prompt(dataset=resolved.dataset, prompt_id=resolved.prompt_id)
        for question in questions:
            budget_candidate_retrieval_context(
                question=question,
                retrieval_result=RagRetrievalResult(
                    question_id=question.question_id,
                    dataset=resolved.dataset,
                    retrieval_run_id=None,
                    retrieval_name=None,
                    embedding_model=None,
                    top_k=0,
                    status="success",
                    chunks=[],
                ),
                prompt=prompt,
                model_name=resolved.model_name,
                av3_provider=provider_config.av3_provider,
                max_tokens=max_tokens,
                safety_margin_tokens=profile.safety_margin_tokens,
                context_window_tokens=profile.context_window_tokens,
                chars_per_token_estimate=profile.chars_per_token_estimate,
                prompt_budget_utilization=profile.prompt_budget_utilization,
            )

    return ResolvedCandidateRuntimeConfig(
        model_name=resolved.model_name,
        technical_provider=resolved.provider,
        av3_provider=provider_config.av3_provider,
        base_url=provider_config.base_url,
        api_key=provider_config.api_key,
        temperature=temperature,
        top_p=top_p,
        default_max_output_tokens=profile.default_max_output_tokens,
        max_output_tokens_cap=profile.max_output_tokens_cap,
        requested_max_tokens=requested_max_tokens,
        max_tokens=max_tokens,
        context_window_tokens=profile.context_window_tokens,
        safety_margin_tokens=profile.safety_margin_tokens,
        chars_per_token_estimate=profile.chars_per_token_estimate,
        prompt_budget_utilization=profile.prompt_budget_utilization,
        model_profile_source=profile.source,
        model_profile_confidence=profile.confidence,
        save_raw_response=request.save_raw_response,
        retry_on_context_window=_resolve_candidate_retry_on_context_window(settings=settings, request=request),
    )


def _resolve_candidate_assignment(
    *,
    repository: CandidateRunRepositoryProtocol,
    dataset: str,
    model_name: str,
    questions: list[CandidateQuestionRecord],
) -> CandidateModelAssignment:
    normalized_model_name = model_name.strip().casefold()
    assignments = tuple(
        assignment
        for assignment in repository.list_candidate_model_assignments()
        if assignment.active
        and (assignment.av3_provider_model_id or "").strip().casefold() == normalized_model_name
        and any(assignment_range.dataset_code == dataset for assignment_range in assignment.ranges)
    )
    if not assignments:
        raise ValueError(
            f"No AV3 candidate assignment found for dataset={dataset} and candidate_model={model_name}."
        )

    if questions:
        question_sequences = {question.question_sequence for question in questions}
        assignments = tuple(
            assignment
            for assignment in assignments
            if any(assignment.covers(dataset=dataset, question_sequence=sequence) for sequence in question_sequences)
        )
        if not assignments:
            raise ValueError(
                f"Candidate model {model_name} does not cover the selected {dataset} question range."
            )

    runnable_assignments = tuple(assignment for assignment in assignments if assignment.is_runnable())
    if not runnable_assignments:
        raise ValueError(f"Candidate model {model_name} has no runnable AV3 assignment for dataset={dataset}.")

    providers = {assignment.av3_provider for assignment in runnable_assignments}
    if len(providers) != 1:
        raise ValueError(
            f"Candidate model {model_name} resolved to multiple AV3 providers for dataset={dataset}: "
            f"{', '.join(sorted(providers))}."
        )
    return runnable_assignments[0]


def resolve_candidate_max_tokens(
    *,
    model_name: str,
    av3_provider: str,
    requested_max_tokens: int | None,
) -> int:
    profile = resolve_candidate_model_runtime_profile(
        provider=av3_provider,
        model_name=model_name,
        safety_margin_tokens=512,
    )
    return resolve_candidate_max_output_tokens(
        profile=profile,
        requested_max_tokens=requested_max_tokens,
    )


def _resolve_candidate_temperature(*, settings: Any, request: RunCandidatesRagRequest) -> float:
    if request.remote_candidate_temperature is not None:
        return float(request.remote_candidate_temperature)
    return float(getattr(settings, "remote_candidate_temperature", 0.2))


def _resolve_candidate_top_p(*, settings: Any, request: RunCandidatesRagRequest) -> float:
    if request.remote_candidate_top_p is not None:
        return float(request.remote_candidate_top_p)
    return float(getattr(settings, "remote_candidate_top_p", 0.9))


def _resolve_requested_candidate_max_tokens(*, settings: Any, request: RunCandidatesRagRequest) -> int | None:
    if request.remote_candidate_max_tokens is not None:
        return int(request.remote_candidate_max_tokens)
    value = getattr(settings, "remote_candidate_max_tokens", None)
    return None if value is None else int(value)


def _resolve_candidate_context_safety_margin_tokens(*, settings: Any, request: RunCandidatesRagRequest) -> int:
    if request.remote_candidate_context_safety_margin_tokens is not None:
        return int(request.remote_candidate_context_safety_margin_tokens)
    return int(getattr(settings, "remote_candidate_context_safety_margin_tokens", 512))


def _resolve_candidate_context_window_tokens(*, settings: Any, request: RunCandidatesRagRequest) -> int | None:
    if request.remote_candidate_context_window_tokens is not None:
        return int(request.remote_candidate_context_window_tokens)
    value = getattr(settings, "remote_candidate_context_window_tokens", None)
    return None if value is None else int(value)


def _resolve_candidate_retry_on_context_window(*, settings: Any, request: RunCandidatesRagRequest) -> bool:
    if request.remote_candidate_retry_on_context_window is not None:
        return bool(request.remote_candidate_retry_on_context_window)
    return bool(getattr(settings, "remote_candidate_retry_on_context_window", False))


def _resolve_candidate_execution_config(
    *,
    settings: Any,
    request: RunCandidatesRagRequest,
) -> ResolvedCandidateExecutionConfig:
    strategy = (
        request.candidate_execution_strategy
        if request.candidate_execution_strategy is not None
        else getattr(settings, "candidate_execution_strategy", "sequential")
    )
    normalized_strategy = str(strategy).strip().casefold()
    if normalized_strategy not in {"sequential", "parallel", "adaptive"}:
        raise ValueError(
            "candidate_execution_strategy must be one of: adaptive, parallel, sequential."
        )

    raw_max_workers = (
        request.candidate_parallel_max_workers
        if request.candidate_parallel_max_workers is not None
        else getattr(settings, "candidate_parallel_max_workers", 2)
    )
    parallel_max_workers = int(raw_max_workers)
    if parallel_max_workers < 1:
        raise ValueError("candidate_parallel_max_workers must be >= 1.")

    adaptive_initial_concurrency = int(getattr(settings, "candidate_adaptive_initial_concurrency", 1))
    adaptive_max_concurrency = int(getattr(settings, "candidate_adaptive_max_concurrency", 2))
    adaptive_success_threshold = int(getattr(settings, "candidate_adaptive_success_threshold", 3))
    adaptive_max_retries = int(getattr(settings, "candidate_adaptive_max_retries", 2))
    adaptive_base_backoff_seconds = float(getattr(settings, "candidate_adaptive_base_backoff_seconds", 2.0))
    adaptive_max_backoff_seconds = float(getattr(settings, "candidate_adaptive_max_backoff_seconds", 60.0))
    if adaptive_initial_concurrency < 1:
        raise ValueError("candidate_adaptive_initial_concurrency must be >= 1.")
    if adaptive_max_concurrency < 1:
        raise ValueError("candidate_adaptive_max_concurrency must be >= 1.")
    if adaptive_initial_concurrency > adaptive_max_concurrency:
        raise ValueError("candidate_adaptive_initial_concurrency must be <= candidate_adaptive_max_concurrency.")
    if adaptive_success_threshold < 1:
        raise ValueError("candidate_adaptive_success_threshold must be >= 1.")
    if adaptive_max_retries < 0:
        raise ValueError("candidate_adaptive_max_retries must be >= 0.")
    if adaptive_base_backoff_seconds < 0:
        raise ValueError("candidate_adaptive_base_backoff_seconds must be >= 0.")
    if adaptive_max_backoff_seconds < adaptive_base_backoff_seconds:
        raise ValueError("candidate_adaptive_max_backoff_seconds must be >= candidate_adaptive_base_backoff_seconds.")

    return ResolvedCandidateExecutionConfig(
        strategy=normalized_strategy,
        parallel_max_workers=parallel_max_workers,
        adaptive_initial_concurrency=adaptive_initial_concurrency,
        adaptive_max_concurrency=adaptive_max_concurrency,
        adaptive_success_threshold=adaptive_success_threshold,
        adaptive_max_retries=adaptive_max_retries,
        adaptive_base_backoff_seconds=adaptive_base_backoff_seconds,
        adaptive_max_backoff_seconds=adaptive_max_backoff_seconds,
    )


def _validate_candidate_provider_execution_strategy(
    *,
    runtime_config: ResolvedCandidateRuntimeConfig,
    execution_config: ResolvedCandidateExecutionConfig,
) -> None:
    if runtime_config.av3_provider != "llama_cpp":
        return
    if execution_config.strategy == "sequential":
        return
    raise ValueError(
        "llama.cpp candidate assignments must use sequential execution. "
        f"Requested strategy: {execution_config.strategy}."
    )


def _format_candidate_question_selection(summary: CandidateQuestionSelectionSummary) -> str:
    lines = [
        "Candidate question selection:",
        f"  policy: {summary.policy}",
        f"  skip_existing_successful: {str(summary.skip_existing_successful).lower()}",
        f"  selected: {summary.selected}",
    ]
    if summary.skip_existing_successful:
        lines.extend(
            [
                f"  failed_retry_candidates: {summary.failed_retry_candidates}",
                f"  unanswered_candidates: {summary.unanswered_candidates}",
                f"  successful_excluded: {summary.successful_excluded if summary.successful_excluded is not None else 'unknown'}",
            ]
        )
    return "\n".join(lines)


def _format_candidate_runtime_config(
    runtime_config: ResolvedCandidateRuntimeConfig,
    *,
    execution_config: ResolvedCandidateExecutionConfig | None = None,
    api_key_state: str,
) -> str:
    context_window = (
        str(runtime_config.context_window_tokens)
        if runtime_config.context_window_tokens is not None
        else "unknown"
    )
    return (
        "Candidate runtime preflight:\n"
        f"  model: {runtime_config.model_name}\n"
        f"  technical provider: {runtime_config.technical_provider}\n"
        f"  av3 provider: {runtime_config.av3_provider}\n"
        f"  base_url: {runtime_config.base_url}\n"
        f"  api_key: {api_key_state}\n"
        f"  temperature: {runtime_config.temperature}\n"
        f"  top_p: {runtime_config.top_p}\n"
        f"  default_max_output_tokens: {runtime_config.default_max_output_tokens}\n"
        f"  max_output_tokens_cap: {runtime_config.max_output_tokens_cap or 'none'}\n"
        f"  requested_max_tokens: {runtime_config.requested_max_tokens}\n"
        f"  final_max_tokens: {runtime_config.max_tokens}\n"
        f"  context_window_tokens: {context_window}\n"
        f"  safety_margin_tokens: {runtime_config.safety_margin_tokens}\n"
        f"  chars_per_token_estimate: {runtime_config.chars_per_token_estimate:g}\n"
        f"  prompt_budget_utilization: {runtime_config.prompt_budget_utilization:g}\n"
        f"  model_profile_source: {runtime_config.model_profile_source}\n"
        f"  save_raw_response: {str(runtime_config.save_raw_response).lower()}"
        f"{_format_candidate_execution_preflight(execution_config)}"
    )


def _format_candidate_execution_preflight(
    execution_config: ResolvedCandidateExecutionConfig | None,
) -> str:
    if execution_config is None:
        return ""
    summary = (
        f"\n  Candidate execution strategy: {execution_config.strategy}"
        f"\n  Candidate parallel max workers: {execution_config.parallel_max_workers}"
    )
    if execution_config.strategy == "adaptive":
        summary = (
            f"{summary}"
            "\n  Candidate adaptive initial/max concurrency: "
            f"{execution_config.adaptive_initial_concurrency}/{execution_config.adaptive_max_concurrency}"
            f"\n  Candidate adaptive success threshold: {execution_config.adaptive_success_threshold}"
            f"\n  Candidate adaptive max retries: {execution_config.adaptive_max_retries}"
            "\n  Candidate adaptive backoff seconds: "
            f"{execution_config.adaptive_base_backoff_seconds:g}..{execution_config.adaptive_max_backoff_seconds:g}"
        )
    return summary


def _candidate_runtime_profile_metadata(
    runtime_config: ResolvedCandidateRuntimeConfig,
) -> dict[str, Any]:
    return {
        "av3_provider": runtime_config.av3_provider,
        "provider_model_id": runtime_config.model_name,
        "context_window_tokens": runtime_config.context_window_tokens,
        "default_max_output_tokens": runtime_config.default_max_output_tokens,
        "max_output_tokens_cap": runtime_config.max_output_tokens_cap,
        "safety_margin_tokens": runtime_config.safety_margin_tokens,
        "chars_per_token_estimate": runtime_config.chars_per_token_estimate,
        "prompt_budget_utilization": runtime_config.prompt_budget_utilization,
        "source": runtime_config.model_profile_source,
        "confidence": runtime_config.model_profile_confidence,
    }


def _execute_candidate_generation(
    *,
    repository: CandidateRunRepositoryProtocol,
    client: CandidateClient,
    audit: AuditLogger,
    question: CandidateQuestionRecord,
    retrieval_result: RagRetrievalResult,
    prompt: CandidatePromptRecord,
    resolved: ResolvedRunCandidatesRag,
    request: RunCandidatesRagRequest,
    runtime_config: ResolvedCandidateRuntimeConfig,
    candidate_run_id: int,
    propagate_retryable_generation_errors: bool = False,
    progress: CandidateProgressReporter | None = None,
) -> CandidateGenerationOutcome:
    initial_budget = _budget_retrieval_for_question(
        question=question,
        retrieval_result=retrieval_result,
        prompt=prompt,
        runtime_config=runtime_config,
    )
    rendered_prompt = _render_prompt(
        question=question,
        retrieval_result=initial_budget.retrieval_result_for_prompt,
        prompt=prompt,
    )
    candidate_budget_metadata = budget_to_metadata(
        budget=initial_budget.budget,
        requested_max_tokens=runtime_config.requested_max_tokens,
    )
    if progress is not None:
        progress.emit(
            "candidate_budget_applied",
            question=question,
            status="ready",
            metadata={
                "final_max_tokens": runtime_config.max_tokens,
                "context_window_tokens": runtime_config.context_window_tokens,
                "included_chunks": initial_budget.budget.included_chunks,
                "truncated_chunks": initial_budget.budget.truncated_chunks,
                "dropped_chunks": initial_budget.budget.dropped_chunks,
                "estimated_prompt_tokens": initial_budget.budget.estimated_prompt_tokens_after_budget,
            },
        )
        progress.emit(
            "candidate_generation_started",
            question=question,
            status="running",
            metadata={
                "model_name": resolved.model_name,
                "provider": resolved.provider,
                "final_max_tokens": runtime_config.max_tokens,
            },
        )
    try:
        raw_response = client.generate(rendered_prompt, model=resolved.model_name)
        audit.event(
            AuditEvent(
                "generation_finished",
                (
                    f"candidate_run_id={candidate_run_id} question_id={question.question_id} "
                    f"latency_ms={raw_response.latency_ms}"
                ),
            )
        )
        if progress is not None:
            progress.emit(
                "candidate_generation_finished",
                question=question,
                status="success",
                metadata={"latency_ms": raw_response.latency_ms},
            )
        stored_answer = CandidateAnswerRecord(
            candidate_answer_id=None,
            candidate_run_id=candidate_run_id,
            question_id=question.question_id,
            model_name=resolved.model_name,
            rendered_prompt=rendered_prompt,
            status="success",
            answer_text=raw_response.text,
            final_choice=_extract_final_choice(raw_response.text, dataset=resolved.dataset),
            latency_ms=raw_response.latency_ms,
            raw_response=_build_candidate_answer_raw_response(
                raw_response=raw_response.raw_response if request.save_raw_response else None,
                retry_metadata=None,
                candidate_budget_metadata=candidate_budget_metadata,
            ),
        )
        return CandidateGenerationOutcome(
            answer=stored_answer,
            retrieval_result_for_prompt=initial_budget.retrieval_result_for_prompt,
            budget=initial_budget.budget,
            runtime_config=runtime_config,
        )
    except Exception as error:
        observation = parse_candidate_runtime_observation(str(error))
        updated_runtime_config = runtime_config
        retry_metadata: dict[str, Any] | None = None
        if observation is not None:
            repository.record_candidate_model_runtime_observation(
                av3_provider=runtime_config.av3_provider,
                provider_model_id=resolved.model_name,
                observed_context_window_tokens=observation.observed_context_window_tokens,
                observed_prompt_tokens=observation.observed_prompt_tokens,
                observed_requested_max_tokens=observation.observed_requested_max_tokens,
                observed_total_tokens=observation.observed_total_tokens,
                error_class=observation.error_class,
                error_message=observation.error_message,
                candidate_run_id=candidate_run_id,
                metadata={
                    "question_id": question.question_id,
                    "status_code": getattr(error, "status_code", None),
                },
            )
            profile_record = repository.upsert_candidate_model_runtime_profile(
                av3_provider=runtime_config.av3_provider,
                provider_model_id=resolved.model_name,
                context_window_tokens=observation.observed_context_window_tokens,
                default_max_output_tokens=runtime_config.default_max_output_tokens,
                safety_margin_tokens=runtime_config.safety_margin_tokens,
                source="db_observed",
                confidence="observed_error",
                metadata={
                    "error_class": observation.error_class,
                    "question_id": question.question_id,
                },
            )
            updated_runtime_config = _runtime_config_from_profile(
                runtime_config=runtime_config,
                profile=resolve_candidate_model_runtime_profile(
                    provider=runtime_config.av3_provider,
                    model_name=resolved.model_name,
                    safety_margin_tokens=runtime_config.safety_margin_tokens,
                    persisted_profile=profile_record,
                ),
                question=question,
                prompt=prompt,
                resolved=resolved,
            )
            _set_client_max_tokens_if_supported(client, updated_runtime_config.max_tokens)
            retry_metadata = {
                "enabled": runtime_config.retry_on_context_window,
                "attempted": False,
                "observed_context_window_tokens": observation.observed_context_window_tokens,
                "first_error_class": observation.error_class,
            }
            audit.event(
                AuditEvent(
                    "runtime_observation_recorded",
                    (
                        f"candidate_run_id={candidate_run_id} question_id={question.question_id} "
                        f"context_window_tokens={observation.observed_context_window_tokens}"
                    ),
                )
            )
            if runtime_config.retry_on_context_window and observation.observed_context_window_tokens is not None:
                retry_metadata["attempted"] = True
                try:
                    retried_budget = _budget_retrieval_for_question(
                        question=question,
                        retrieval_result=retrieval_result,
                        prompt=prompt,
                        runtime_config=updated_runtime_config,
                    )
                    retried_prompt = _render_prompt(
                        question=question,
                        retrieval_result=retried_budget.retrieval_result_for_prompt,
                        prompt=prompt,
                    )
                    candidate_budget_metadata = budget_to_metadata(
                        budget=retried_budget.budget,
                        requested_max_tokens=updated_runtime_config.requested_max_tokens,
                    )
                    if progress is not None:
                        progress.emit(
                            "candidate_budget_applied",
                            question=question,
                            status="ready",
                            metadata={
                                "final_max_tokens": updated_runtime_config.max_tokens,
                                "context_window_tokens": updated_runtime_config.context_window_tokens,
                                "included_chunks": retried_budget.budget.included_chunks,
                                "truncated_chunks": retried_budget.budget.truncated_chunks,
                                "dropped_chunks": retried_budget.budget.dropped_chunks,
                                "estimated_prompt_tokens": retried_budget.budget.estimated_prompt_tokens_after_budget,
                            },
                        )
                        progress.emit(
                            "candidate_generation_started",
                            question=question,
                            status="running",
                            metadata={
                                "model_name": resolved.model_name,
                                "provider": resolved.provider,
                                "final_max_tokens": updated_runtime_config.max_tokens,
                            },
                        )
                    retried_response = client.generate(retried_prompt, model=resolved.model_name)
                    audit.event(
                        AuditEvent(
                            "context_window_retry_succeeded",
                            (
                                f"candidate_run_id={candidate_run_id} question_id={question.question_id} "
                                f"latency_ms={retried_response.latency_ms}"
                            ),
                        )
                    )
                    if progress is not None:
                        progress.emit(
                            "candidate_generation_finished",
                            question=question,
                            status="success",
                            metadata={"latency_ms": retried_response.latency_ms},
                        )
                    stored_answer = CandidateAnswerRecord(
                        candidate_answer_id=None,
                        candidate_run_id=candidate_run_id,
                        question_id=question.question_id,
                        model_name=resolved.model_name,
                        rendered_prompt=retried_prompt,
                        status="success",
                        answer_text=retried_response.text,
                        final_choice=_extract_final_choice(retried_response.text, dataset=resolved.dataset),
                        latency_ms=retried_response.latency_ms,
                        raw_response=_build_candidate_answer_raw_response(
                            raw_response=retried_response.raw_response if request.save_raw_response else None,
                            retry_metadata=retry_metadata,
                            candidate_budget_metadata=candidate_budget_metadata,
                        ),
                    )
                    return CandidateGenerationOutcome(
                        answer=stored_answer,
                        retrieval_result_for_prompt=retried_budget.retrieval_result_for_prompt,
                        budget=retried_budget.budget,
                        runtime_config=updated_runtime_config,
                    )
                except Exception as retry_error:
                    error = retry_error
                    if "retried_prompt" in locals():
                        rendered_prompt = retried_prompt
                    if "retried_budget" in locals():
                        initial_budget = retried_budget
                    if propagate_retryable_generation_errors and classify_candidate_error(error).retryable:
                        raise

        if propagate_retryable_generation_errors and classify_candidate_error(error).retryable:
            raise
        audit.event(
            AuditEvent(
                "generation_failed",
                (
                    f"candidate_run_id={candidate_run_id} question_id={question.question_id} "
                    f"error_type={type(error).__name__}"
                ),
            )
        )
        stored_answer = repository.persist_candidate_answer(
            answer=CandidateAnswerRecord(
                candidate_answer_id=None,
                candidate_run_id=candidate_run_id,
                question_id=question.question_id,
                model_name=resolved.model_name,
                rendered_prompt=rendered_prompt,
                status="failed",
                error_message=str(error),
                raw_response=_build_candidate_answer_raw_response(
                    raw_response=None,
                    retry_metadata=retry_metadata,
                    candidate_budget_metadata=candidate_budget_metadata,
                ),
            )
        )
        return CandidateGenerationOutcome(
            answer=stored_answer,
            retrieval_result_for_prompt=initial_budget.retrieval_result_for_prompt,
            budget=initial_budget.budget,
            runtime_config=updated_runtime_config,
        )


def _build_candidate_answer_raw_response(
    *,
    raw_response: dict[str, Any] | None,
    retry_metadata: dict[str, Any] | None,
    candidate_budget_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_response is None and retry_metadata is None and candidate_budget_metadata is None:
        return None
    payload: dict[str, Any] = {}
    if raw_response is not None:
        payload.update(raw_response)
    if retry_metadata is not None:
        payload["context_window_retry"] = retry_metadata
    if candidate_budget_metadata is not None:
        payload["candidate_budget"] = candidate_budget_metadata
    return payload


def _runtime_config_from_profile(
    *,
    runtime_config: ResolvedCandidateRuntimeConfig,
    profile: CandidateModelRuntimeProfile,
    question: CandidateQuestionRecord,
    prompt: CandidatePromptRecord,
    resolved: ResolvedRunCandidatesRag,
) -> ResolvedCandidateRuntimeConfig:
    requested_max_tokens = int(runtime_config.requested_max_tokens)
    max_tokens = resolve_candidate_max_output_tokens(
        profile=profile,
        requested_max_tokens=requested_max_tokens,
    )
    return replace(
        runtime_config,
        default_max_output_tokens=profile.default_max_output_tokens,
        max_output_tokens_cap=profile.max_output_tokens_cap,
        requested_max_tokens=requested_max_tokens,
        max_tokens=max_tokens,
        context_window_tokens=profile.context_window_tokens,
        safety_margin_tokens=profile.safety_margin_tokens,
        chars_per_token_estimate=profile.chars_per_token_estimate,
        prompt_budget_utilization=profile.prompt_budget_utilization,
        model_profile_source=profile.source,
        model_profile_confidence=profile.confidence,
    )


def _set_client_max_tokens_if_supported(client: CandidateClient, max_tokens: int) -> None:
    config = getattr(client, "config", None)
    if config is None or not hasattr(config, "max_tokens"):
        return
    setattr(client, "config", replace(config, max_tokens=int(max_tokens)))


def _budget_retrieval_for_question(
    *,
    question: CandidateQuestionRecord,
    retrieval_result: RagRetrievalResult,
    prompt: CandidatePromptRecord,
    runtime_config: ResolvedCandidateRuntimeConfig,
) -> BudgetedCandidateRetrievalContext:
    return budget_candidate_retrieval_context(
        question=question,
        retrieval_result=retrieval_result,
        prompt=prompt,
        model_name=runtime_config.model_name,
        av3_provider=runtime_config.av3_provider,
        max_tokens=runtime_config.max_tokens,
        safety_margin_tokens=runtime_config.safety_margin_tokens,
        context_window_tokens=runtime_config.context_window_tokens,
        chars_per_token_estimate=runtime_config.chars_per_token_estimate,
        prompt_budget_utilization=runtime_config.prompt_budget_utilization,
    )


def _format_candidate_prompt_budget(*, question: CandidateQuestionRecord, budget: CandidatePromptBudget) -> str:
    return (
        "Candidate prompt budget:\n"
        f"  question_id: {question.question_id}\n"
        f"  estimated_prompt_tokens_before_budget: {budget.estimated_prompt_tokens_before_budget}\n"
        f"  estimated_prompt_tokens_after_budget: {budget.estimated_prompt_tokens_after_budget}\n"
        f"  target_prompt_budget: {budget.target_prompt_budget or 'unknown'}\n"
        f"  chars_per_token_estimate: {budget.chars_per_token_estimate:g}\n"
        f"  prompt_budget_utilization: {budget.prompt_budget_utilization:g}\n"
        f"  retrieved_chunks: {budget.retrieved_chunks}\n"
        f"  included_chunks: {budget.included_chunks}\n"
        f"  truncated_chunks: {budget.truncated_chunks}\n"
        f"  dropped_chunks: {budget.dropped_chunks}"
    )


def _log_candidate_prompt_budget(
    *,
    audit: AuditLogger,
    question: CandidateQuestionRecord,
    budget: CandidatePromptBudget,
) -> None:
    budget_summary = _format_candidate_prompt_budget(question=question, budget=budget)
    audit.file_event("candidate_prompt_budget", budget_summary.replace("\n", " | "))
    audit.terminal_event(budget_summary)


def _default_retriever_factory(
    repository: CandidateRunRepositoryProtocol,
    settings: Any,
    dataset: str,
) -> RagRetrieverService:
    config = repository.get_rag_embedding_model_config(dataset=dataset)
    api_base_url = "https://api.openai.com/v1"
    if config is not None and getattr(config, "api_base_url", None):
        api_base_url = str(config.api_base_url).strip() or api_base_url
    return RagRetrieverService(
        repository=repository,
        embedding_provider=_QuestionEmbeddingProvider(
            api_key=getattr(settings, "embedding_api_key", None),
            api_base_url=api_base_url,
        ),
    )


@dataclass(frozen=True)
class _QuestionEmbeddingProvider:
    api_key: str | None
    api_base_url: str

    def embed_query(self, text: str, *, model: str) -> list[float]:
        result = request_openai_compatible_embeddings(
            api_base_url=self.api_base_url,
            api_key=(self.api_key or "").strip(),
            model_name=model,
            texts=[text],
            dimensions=None,
        )
        return result.vectors[0]
