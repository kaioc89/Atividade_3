"""Service-layer orchestration for AV3 candidate RAG generation runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .audit import AuditEvent, AuditLogger
from .candidate_clients.base import CandidateClient
from .candidate_clients.remote_http import RemoteHttpCandidateClient, RemoteHttpCandidateClientConfig
from .candidate_prompts import build_candidate_prompt
from .config import load_settings
from .contracts import (
    CandidateAnswerRecord,
    CandidatePromptContext,
    CandidatePromptRecord,
    CandidateQuestionRecord,
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
    remote_candidate_temperature: float = 0.0
    remote_candidate_max_tokens: int = 4000
    remote_candidate_top_p: float = 1.0
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
class RunCandidatesRagResult:
    """Service result contract shared by future CLI/Web adapters."""

    dry_run: bool
    audit_log: str
    execution_summary: str
    batch_size: int
    dataset: str
    model_name: str
    provider: str
    candidate_run_id: int | None = None
    retrieval_run_id: int | None = None
    prompt_id: int | None = None
    summary: CandidateRunSummary | None = None


class CandidateRunRepositoryProtocol(Protocol):
    """Repository operations required by the candidate runner service."""

    def ensure_schema(self) -> None:
        """Ensure required AV3 tables exist."""

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
            if request.dry_run:
                audit.file_event("dry_run_finished", "no database rows selected and no remote candidate calls made")
                audit.terminal_event("Dry run: no database rows selected and no remote candidate calls made.")
                return RunCandidatesRagResult(
                    dry_run=True,
                    audit_log=str(resolved.audit_path),
                    execution_summary=resolved.execution_summary,
                    batch_size=resolved.batch_size,
                    dataset=resolved.dataset,
                    model_name=resolved.model_name,
                    provider=resolved.provider,
                )

            with audit.step("Connecting to local PostgreSQL", detail="DATABASE_URL=<redacted>"):
                connection = self._connect(settings.database_url)
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
                        temperature=request.remote_candidate_temperature,
                        max_tokens=request.remote_candidate_max_tokens,
                        top_p=request.remote_candidate_top_p,
                        started_at=started_at,
                        created_by=request.created_by,
                        metadata=_run_metadata(resolved, vector_base, prompt, request),
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
                with audit.step(
                    f"Selecting candidate questions for {resolved.dataset}",
                    detail=(
                        f"dataset={resolved.dataset} batch_size={resolved.batch_size} "
                        f"question_id={resolved.question_id} start={resolved.question_sequence_start} "
                        f"end={resolved.question_sequence_end}"
                    ),
                ):
                    questions = repository.select_candidate_questions(
                        dataset=resolved.dataset,
                        batch_size=resolved.batch_size,
                        question_sequence_start=resolved.question_sequence_start,
                        question_sequence_end=resolved.question_sequence_end,
                        question_id=resolved.question_id,
                    )
                retriever = self._retriever_factory(repository, settings, resolved.dataset)
                client = self._client_factory(request, settings)
                question_results: list[CandidateQuestionRunResult] = []
                successful_answers = 0
                failed_answers = 0
                skipped_questions = 0

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
                    rendered_prompt = _render_prompt(
                        question=question,
                        retrieval_result=retrieval_result,
                        prompt=prompt,
                    )
                    if retrieval_result.status != "success":
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

                    try:
                        raw_response = client.generate(rendered_prompt, model=resolved.model_name)
                        audit.event(
                            AuditEvent(
                                "generation_finished",
                                (
                                    f"candidate_run_id={run.candidate_run_id} question_id={question.question_id} "
                                    f"latency_ms={raw_response.latency_ms}"
                                ),
                            )
                        )
                        stored_answer = repository.persist_candidate_answer(
                            answer=CandidateAnswerRecord(
                                candidate_answer_id=None,
                                candidate_run_id=int(run.candidate_run_id),
                                question_id=question.question_id,
                                model_name=resolved.model_name,
                                rendered_prompt=rendered_prompt,
                                status="success",
                                answer_text=raw_response.text,
                                final_choice=_extract_final_choice(raw_response.text, dataset=resolved.dataset),
                                latency_ms=raw_response.latency_ms,
                                raw_response=raw_response.raw_response if request.save_raw_response else None,
                            )
                        )
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
                    except Exception as error:
                        stored_answer = repository.persist_candidate_answer(
                            answer=CandidateAnswerRecord(
                                candidate_answer_id=None,
                                candidate_run_id=int(run.candidate_run_id),
                                question_id=question.question_id,
                                model_name=resolved.model_name,
                                rendered_prompt=rendered_prompt,
                                status="failed",
                                error_message=str(error),
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
                                "generation_failed",
                                (
                                    f"candidate_run_id={run.candidate_run_id} question_id={question.question_id} "
                                    f"error={error}"
                                ),
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
                        retrieval_result=retrieval_result,
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
                repository.update_candidate_run_status(
                    candidate_run_id=int(run.candidate_run_id),
                    run_status="completed",
                    finished_at=_utcnow_iso(),
                    metadata={
                        "selected_questions": summary.selected_questions,
                        "processed_questions": summary.processed_questions,
                        "successful_answers": summary.successful_answers,
                        "failed_answers": summary.failed_answers,
                        "skipped_questions": summary.skipped_questions,
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
                    execution_summary=resolved.execution_summary,
                    batch_size=resolved.batch_size,
                    dataset=resolved.dataset,
                    model_name=resolved.model_name,
                    provider=resolved.provider,
                    candidate_run_id=run.candidate_run_id,
                    retrieval_run_id=int(vector_base.retrieval_run_id),
                    prompt_id=int(prompt.prompt_id),
                    summary=summary,
                )
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
            temperature=request.remote_candidate_temperature,
            max_tokens=request.remote_candidate_max_tokens,
            top_p=request.remote_candidate_top_p,
            openai_compatible=request.remote_candidate_openai_compatible,
            save_raw_response=request.save_raw_response,
        )
    )


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
