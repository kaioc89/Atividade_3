"""Service-layer orchestration for AV3 candidate RAG generation runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

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
    CandidateAnswerRecord,
    CandidateModelAssignment,
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
class CandidateGenerationOutcome:
    """Final per-question execution outcome after optional retry handling."""

    answer: CandidateAnswerRecord
    retrieval_result_for_prompt: RagRetrievalResult
    budget: CandidatePromptBudget | None
    runtime_config: ResolvedCandidateRuntimeConfig


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
    ) -> None:
        self._settings_loader = settings_loader
        self._connect = connect_func
        self._repository_factory = repository_factory
        self._retriever_factory = retriever_factory or _default_retriever_factory
        self._client_factory = client_factory or _default_client_factory
        self._snapshot_service_factory = snapshot_service_factory or _default_snapshot_service_factory

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
            ),
        )

    def run(self, request: RunCandidatesRagRequest) -> RunCandidatesRagResult:
        """Run or dry-run the candidate RAG execution flow."""
        resolved = self.resolve(request)
        animate = False if request.no_audit_animation else None
        with AuditLogger(file_path=resolved.audit_path, animate=animate) as audit:
            with audit.step("Loading configuration"):
                settings = self._settings_loader()
            audit.file_event("execution_summary", resolved.execution_summary.replace("\n", " | "))
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
                runtime_config = _resolve_candidate_runtime_config(
                    repository=repository,
                    settings=settings,
                    request=request,
                    resolved=resolved,
                    questions=questions,
                    require_api_key=not request.dry_run,
                )
                runtime_summary = _format_candidate_runtime_config(
                    runtime_config,
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
                        execution_summary=resolved.execution_summary,
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
                        metadata=_run_metadata(resolved, vector_base, prompt, request, runtime_config),
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
                client = None
                if questions:
                    client_request = request
                    if self._client_factory is _default_client_factory:
                        client_request = _with_remote_candidate_config(request, runtime_config)
                    client = self._client_factory(client_request, settings)
                question_results: list[CandidateQuestionRunResult] = []
                successful_answers = 0
                failed_answers = 0
                skipped_questions = 0
                budget_summaries: list[CandidatePromptBudget] = []

                for question in questions:
                    audit.event(
                        AuditEvent(
                            "question_started",
                            (
                                f"candidate_run_id={run.candidate_run_id} question_id={question.question_id} "
                                f"sequence={question.question_sequence} dataset={question.dataset}"
                            ),
                        )
                    )
                    if resolved.skip_existing_successful and repository.successful_candidate_answer_exists(
                        dataset=resolved.dataset,
                        model_name=resolved.model_name,
                        question_id=question.question_id,
                        exclude_candidate_run_id=run.candidate_run_id,
                    ):
                        skipped_questions += 1
                        question_results.append(
                            CandidateQuestionRunResult(
                                question_id=question.question_id,
                                question_sequence=question.question_sequence,
                                status="skipped",
                                error_message="existing_successful_answer",
                            )
                        )
                        audit.event(
                            AuditEvent(
                                "question_skipped",
                                (
                                    f"candidate_run_id={run.candidate_run_id} question_id={question.question_id} "
                                    "reason=existing_successful_answer"
                                ),
                            )
                        )
                        continue

                    retrieval_result = retriever.retrieve_for_question(
                        question_id=question.question_id,
                        dataset=resolved.dataset,
                    )
                    audit.event(
                        AuditEvent(
                            "retrieval_finished",
                            (
                                f"candidate_run_id={run.candidate_run_id} question_id={question.question_id} "
                                f"status={retrieval_result.status} chunks={len(retrieval_result.chunks)}"
                            ),
                        )
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
                                candidate_run_id=int(run.candidate_run_id),
                                question_id=question.question_id,
                                model_name=resolved.model_name,
                                rendered_prompt=rendered_prompt,
                                status="failed",
                                error_message=f"Retrieval failed: {retrieval_result.status}",
                            )
                        )
                        failed_answers += 1
                        question_results.append(
                            CandidateQuestionRunResult(
                                question_id=question.question_id,
                                question_sequence=question.question_sequence,
                                status="failed",
                                retrieval_status=retrieval_result.status,
                                candidate_answer_id=stored_answer.candidate_answer_id,
                                error_message=stored_answer.error_message,
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
                        continue

                    assert client is not None
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
                        candidate_run_id=int(run.candidate_run_id),
                    )
                    runtime_config = outcome.runtime_config
                    if outcome.budget is not None:
                        budget_summaries.append(outcome.budget)
                        _log_candidate_prompt_budget(
                            audit=audit,
                            question=question,
                            budget=outcome.budget,
                        )
                    stored_answer = outcome.answer
                    if stored_answer.status == "success":
                        successful_answers += 1
                        question_results.append(
                            CandidateQuestionRunResult(
                                question_id=question.question_id,
                                question_sequence=question.question_sequence,
                                status="success",
                                retrieval_status=retrieval_result.status,
                                candidate_answer_id=stored_answer.candidate_answer_id,
                                final_choice=stored_answer.final_choice,
                                latency_ms=stored_answer.latency_ms,
                            )
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
                        failed_answers += 1
                        question_results.append(
                            CandidateQuestionRunResult(
                                question_id=question.question_id,
                                question_sequence=question.question_sequence,
                                status="failed",
                                retrieval_status=retrieval_result.status,
                                candidate_answer_id=stored_answer.candidate_answer_id,
                                error_message=stored_answer.error_message,
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

                    assert stored_answer.candidate_answer_id is not None
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
                    "candidate_runtime_profile": _candidate_runtime_profile_metadata(runtime_config),
                }
                candidate_budget_metadata = aggregate_budget_metadata(
                    budgets=budget_summaries,
                    requested_max_tokens=runtime_config.requested_max_tokens,
                )
                if candidate_budget_metadata:
                    completion_metadata["candidate_budget"] = candidate_budget_metadata
                repository.update_candidate_run_status(
                    candidate_run_id=int(run.candidate_run_id),
                    run_status="completed",
                    finished_at=_utcnow_iso(),
                    metadata=completion_metadata,
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
                    execution_summary=resolved.execution_summary,
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
) -> str:
    return (
        f"Dataset: {dataset}\n"
        f"Candidate model: {model_name}\n"
        f"Provider: {provider}\n"
        f"Batch size: {batch_size}\n"
        f"Question id: {question_id or '-'}\n"
        f"Question range: {_format_question_range(question_sequence_start, question_sequence_end)}"
    )


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
    )


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
        stored_answer = repository.persist_candidate_answer(
            answer=CandidateAnswerRecord(
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
                    stored_answer = repository.persist_candidate_answer(
                        answer=CandidateAnswerRecord(
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

        audit.event(
            AuditEvent(
                "generation_failed",
                (
                    f"candidate_run_id={candidate_run_id} question_id={question.question_id} "
                    f"error={error}"
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
