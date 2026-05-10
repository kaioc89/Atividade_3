"""Judge pipeline orchestration and execution-mode policy."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

from .audit import AuditEvent, NullAuditLogger
from .contracts import (
    BatchProgress,
    CandidateAnswerContext,
    EvaluationProgress,
    EvaluationRecord,
    ModelSpec,
    PipelineSummary,
    RuntimeJudgeConfig,
    RemoteJudgeEndpoint,
    StoredJudgeRole,
)
from .judge_clients.base import JudgeClient
from .judge_clients.remote_http import RemoteJudgeError
from .parser import parse_judge_output
from .prompts import allowed_scores_for_context, build_judge_prompt
from .repositories import JudgeRepositoryProtocol


class JudgePipeline:
    """Local orchestration for remote judge execution."""

    def __init__(
        self,
        repository: JudgeRepositoryProtocol,
        client: JudgeClient,
        audit: NullAuditLogger | None = None,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        evaluation_callback: Callable[[EvaluationProgress], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        sleep_func: Callable[[float], None] = time.sleep,
        monotonic_func: Callable[[], float] = time.monotonic,
        jitter_func: Callable[[], float] = random.random,
    ) -> None:
        self.repository = repository
        self.client = client
        self.audit = audit or NullAuditLogger()
        self.progress_callback = progress_callback
        self.evaluation_callback = evaluation_callback
        self.should_stop = should_stop or (lambda: False)
        self.sleep_func = sleep_func
        self.monotonic_func = monotonic_func
        self.jitter_func = jitter_func

    def run(
        self,
        answers: Sequence[CandidateAnswerContext],
        config: RuntimeJudgeConfig,
    ) -> PipelineSummary:
        if config.execution_strategy == "adaptive":
            return self._run_adaptive(answers, config)

        executed = 0
        skipped = 0
        arbiters = 0
        total_answers = len(answers)
        if total_answers == 0:
            self._report_batch_progress(
                BatchProgress(
                    current=0,
                    total=0,
                    percent=100,
                    executed_evaluations=0,
                    skipped_evaluations=0,
                    arbiter_evaluations=0,
                )
            )

        for index, answer in enumerate(answers, start=1):
            if self.should_stop():
                self.audit.event(
                    AuditEvent(
                        "pipeline_cancelled",
                        (
                            f"current={index - 1} total={total_answers} executed={executed} "
                            f"skipped={skipped} arbiters={arbiters}"
                        ),
                    )
                )
                break
            self.audit.terminal_event(f"Running answer {answer.answer_id} ({answer.dataset_name})")
            self.audit.event(
                AuditEvent(
                    "answer_started",
                    (
                        f"answer_id={answer.answer_id} question_id={answer.question_id} "
                        f"dataset={answer.dataset_name} candidate_model={answer.candidate_model}"
                    ),
                )
            )
            result = self._run_answer(answer, config)
            executed += result.executed_evaluations
            skipped += result.skipped_evaluations
            arbiters += result.arbiter_evaluations
            self._report_batch_progress(
                BatchProgress(
                    current=index,
                    total=total_answers,
                    percent=int(index / total_answers * 100) if total_answers else 100,
                    executed_evaluations=executed,
                    skipped_evaluations=skipped,
                    arbiter_evaluations=arbiters,
                )
            )
            self.audit.event(
                AuditEvent(
                    "answer_finished",
                    (
                        f"answer_id={answer.answer_id} executed={result.executed_evaluations} "
                        f"skipped={result.skipped_evaluations} arbiters={result.arbiter_evaluations}"
                    ),
                )
            )

        return PipelineSummary(
            selected_answers=len(answers),
            executed_evaluations=executed,
            skipped_evaluations=skipped,
            arbiter_evaluations=arbiters,
        )

    def _report_batch_progress(self, progress: BatchProgress) -> None:
        self.audit.terminal_event(
            (
                f"Batch progress: {progress.current}/{progress.total} answers ({progress.percent}%) | "
                f"executed={progress.executed_evaluations} skipped={progress.skipped_evaluations} "
                f"arbiters={progress.arbiter_evaluations}"
            )
        )
        self.audit.event(
            AuditEvent(
                "batch_progress",
                (
                    f"current={progress.current} total={progress.total} percent={progress.percent} "
                    f"executed={progress.executed_evaluations} skipped={progress.skipped_evaluations} "
                    f"arbiters={progress.arbiter_evaluations}"
                ),
            )
        )
        if self.progress_callback is not None:
            try:
                self.progress_callback(progress)
            except Exception as error:
                self.audit.event(AuditEvent("batch_progress_callback_failed", f"error={error}"))

    def _run_answer(self, answer: CandidateAnswerContext, config: RuntimeJudgeConfig) -> PipelineSummary:
        if config.panel_mode == "single":
            assert config.single_judge is not None
            executed, skipped = self._execute_if_needed(
                answer=answer,
                config=config,
                judge_model=config.single_judge,
                stored_role="principal",
                trigger_reason="single_mode",
            )
            return PipelineSummary(1, executed, skipped, 0)

        primary_scores: list[int] = []
        executed = 0
        skipped = 0
        roles: tuple[StoredJudgeRole, ...] = ("principal", "controle")
        pending: list[tuple[ModelSpec, StoredJudgeRole]] = []
        for judge_model, stored_role in zip(config.primary_panel, roles, strict=False):
            score_before = self.repository.existing_score(
                answer.answer_id,
                judge_model,
                stored_role,
                config.panel_mode,
            )
            if score_before is not None:
                skipped += 1
                primary_scores.append(score_before)
                self.audit.terminal_event(
                    f"Skipping answer {answer.answer_id} for {judge_model.requested}: existing evaluation"
                )
                self.audit.event(
                    AuditEvent(
                        "evaluation_skipped",
                        (
                            f"answer_id={answer.answer_id} model={judge_model.provider_model} "
                            f"role={stored_role} mode={config.panel_mode} existing_score={score_before}"
                        ),
                    )
                )
                self._report_evaluation_progress(
                    EvaluationProgress(
                        status="skipped",
                        dataset=answer.dataset_name,
                        question_id=answer.question_id,
                        answer_id=answer.answer_id,
                        candidate_model=answer.candidate_model,
                        judge_model=judge_model.provider_model,
                        role=stored_role,
                        panel_mode=config.panel_mode,
                        score=score_before,
                        trigger_reason=f"{config.panel_mode}:existing_evaluation",
                    )
                )
                continue
            pending.append((judge_model, stored_role))

        records = self._execute_primary_judges(answer, config, pending)
        for record in records:
            with self.audit.step(
                f"Persisting evaluation for answer {answer.answer_id}",
                detail=(
                    f"answer_id={answer.answer_id} model={record.judge_model.provider_model} "
                    f"role={record.stored_role} score={record.score}"
                ),
            ):
                self.repository.persist_evaluation(record)
            executed += 1
            primary_scores.append(record.score)

        if config.panel_mode == "primary_only":
            return PipelineSummary(1, executed, skipped, 0)

        if len(primary_scores) != 2:
            return PipelineSummary(1, executed, skipped, 0)

        score_delta = abs(primary_scores[0] - primary_scores[1])
        arbiter_reason = _arbiter_reason(config, score_delta)
        if arbiter_reason is None:
            self.audit.terminal_event(
                f"Arbiter skipped for answer {answer.answer_id}: score delta {score_delta}"
            )
            self.audit.event(
                AuditEvent(
                    "arbiter_skipped",
                    (
                        f"answer_id={answer.answer_id} score_delta={score_delta} "
                        f"threshold={config.arbitration_min_delta}"
                    ),
                )
            )
            return PipelineSummary(1, executed, skipped, 0)

        assert config.arbiter is not None
        arbiter_executed, arbiter_skipped = self._execute_if_needed(
            answer=answer,
            config=config,
            judge_model=config.arbiter,
            stored_role="arbitro",
            trigger_reason=arbiter_reason,
        )
        return PipelineSummary(1, executed + arbiter_executed, skipped + arbiter_skipped, arbiter_executed)

    def _execute_if_needed(
        self,
        *,
        answer: CandidateAnswerContext,
        config: RuntimeJudgeConfig,
        judge_model: ModelSpec,
        stored_role: StoredJudgeRole,
        trigger_reason: str,
    ) -> tuple[int, int]:
        if self.repository.evaluation_exists(
            answer.answer_id,
            judge_model,
            stored_role,
            config.panel_mode,
        ):
            self.audit.terminal_event(
                f"Skipping answer {answer.answer_id} for {judge_model.requested}: existing evaluation"
            )
            self.audit.event(
                AuditEvent(
                    "evaluation_skipped",
                    (
                        f"answer_id={answer.answer_id} model={judge_model.provider_model} "
                        f"role={stored_role} mode={config.panel_mode}"
                    ),
                )
            )
            self._report_evaluation_progress(
                EvaluationProgress(
                    status="skipped",
                    dataset=answer.dataset_name,
                    question_id=answer.question_id,
                    answer_id=answer.answer_id,
                    candidate_model=answer.candidate_model,
                    judge_model=judge_model.provider_model,
                    role=stored_role,
                    panel_mode=config.panel_mode,
                    trigger_reason=f"{config.panel_mode}:existing_evaluation",
                )
            )
            return 0, 1
        record = self._execute_judge(
            answer=answer,
            config=config,
            judge_model=judge_model,
            stored_role=stored_role,
            trigger_reason=trigger_reason,
        )
        with self.audit.step(
            f"Persisting evaluation for answer {answer.answer_id}",
            detail=(
                f"answer_id={answer.answer_id} model={judge_model.provider_model} "
                f"role={stored_role} score={record.score}"
            ),
        ):
            self.repository.persist_evaluation(record)
        return 1, 0

    def _execute_judge(
        self,
        *,
        answer: CandidateAnswerContext,
        config: RuntimeJudgeConfig,
        judge_model: ModelSpec,
        stored_role: StoredJudgeRole,
        trigger_reason: str,
        terminal_progress: bool = True,
    ) -> EvaluationRecord:
        prompt_template = self.repository.get_prompt_template(
            dataset_name=answer.dataset_name,
        )
        prompt = build_judge_prompt(answer, judge_model=judge_model, template=prompt_template)
        detail = (
            f"answer_id={answer.answer_id} question_id={answer.question_id} "
            f"model={judge_model.provider_model} role={stored_role} trigger={trigger_reason}"
        )
        self._report_evaluation_progress(
            EvaluationProgress(
                status="running",
                dataset=answer.dataset_name,
                question_id=answer.question_id,
                answer_id=answer.answer_id,
                candidate_model=answer.candidate_model,
                judge_model=judge_model.provider_model,
                role=stored_role,
                panel_mode=config.panel_mode,
                arbiter_triggered=True if stored_role == "arbitro" else None,
                trigger_reason=f"{config.panel_mode}:{trigger_reason}",
                prompt=prompt,
            )
        )
        raw_response = None
        try:
            with self.audit.step(
                f"Running answer {answer.answer_id} with {judge_model.requested}",
                detail=detail,
                terminal=terminal_progress,
            ):
                raw_response = self.client.judge(
                    prompt=prompt,
                    model=judge_model.provider_model,
                    requested_model=judge_model.requested,
                    endpoint_key=_endpoint_key_for_role(stored_role, config.panel_mode),
                )
            with self.audit.step(
                f"Parsing judge response for answer {answer.answer_id}",
                detail=detail,
                terminal=terminal_progress,
            ):
                parsed = parse_judge_output(
                    raw_response.text,
                    allowed_scores=allowed_scores_for_context(answer),
                )
        except Exception as error:
            self._report_evaluation_progress(
                EvaluationProgress(
                    status="failed",
                    dataset=answer.dataset_name,
                    question_id=answer.question_id,
                    answer_id=answer.answer_id,
                    candidate_model=answer.candidate_model,
                    judge_model=judge_model.provider_model,
                    role=stored_role,
                    panel_mode=config.panel_mode,
                    arbiter_triggered=True if stored_role == "arbitro" else None,
                    trigger_reason=f"{config.panel_mode}:{trigger_reason}",
                    latency_ms=raw_response.latency_ms if raw_response is not None else None,
                    error=str(error),
                    prompt=prompt,
                    raw_response=raw_response.text if raw_response is not None else None,
                )
            )
            raise
        self.audit.event(
            AuditEvent(
                "evaluation_parsed",
                (
                    f"{detail} score={parsed.score} latency_ms={raw_response.latency_ms} "
                    f"status_code={raw_response.status_code}"
                ),
            )
        )
        self._report_evaluation_progress(
            EvaluationProgress(
                status="success",
                dataset=answer.dataset_name,
                question_id=answer.question_id,
                answer_id=answer.answer_id,
                candidate_model=answer.candidate_model,
                judge_model=judge_model.provider_model,
                role=stored_role,
                panel_mode=config.panel_mode,
                score=parsed.score,
                arbiter_triggered=True if stored_role == "arbitro" else None,
                trigger_reason=f"{config.panel_mode}:{trigger_reason}",
                latency_ms=raw_response.latency_ms,
                prompt=prompt,
                raw_response=raw_response.text,
                rationale=parsed.rationale,
            )
        )
        return EvaluationRecord(
            answer_id=answer.answer_id,
            judge_model=judge_model,
            prompt_id=prompt_template.prompt_id if prompt_template is not None else None,
            stored_role=stored_role,
            panel_mode=config.panel_mode,
            trigger_reason=trigger_reason,
            score=parsed.score,
            rationale=parsed.rationale,
            latency_ms=raw_response.latency_ms,
            raw_response=raw_response,
            parsed_evaluation=parsed,
        )

    def _execute_primary_judges(
        self,
        answer: CandidateAnswerContext,
        config: RuntimeJudgeConfig,
        pending: Sequence[tuple[ModelSpec, StoredJudgeRole]],
    ) -> list[EvaluationRecord]:
        if not pending:
            return []
        if config.execution_strategy == "sequential" or len(pending) == 1:
            self.audit.event(
                AuditEvent(
                    "judge_execution_strategy",
                    f"answer_id={answer.answer_id} strategy=sequential calls={len(pending)}",
                )
            )
            return [
                self._execute_judge(
                    answer=answer,
                    config=config,
                    judge_model=judge_model,
                    stored_role=stored_role,
                    trigger_reason="primary_panel",
                )
                for judge_model, stored_role in pending
            ]

        self.audit.event(
            AuditEvent(
                "judge_execution_strategy",
                f"answer_id={answer.answer_id} strategy=parallel calls={len(pending)}",
            )
        )
        with self.audit.step(
            f"Running {len(pending)} primary judges in parallel for answer {answer.answer_id}",
            detail=f"answer_id={answer.answer_id} models={','.join(model.provider_model for model, _ in pending)}",
        ):
            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                futures = [
                    executor.submit(
                        self._execute_judge,
                        answer=answer,
                        config=config,
                        judge_model=judge_model,
                        stored_role=stored_role,
                        trigger_reason="primary_panel",
                        terminal_progress=False,
                    )
                    for judge_model, stored_role in pending
                ]
                return [future.result() for future in futures]

    def _report_evaluation_progress(self, progress: EvaluationProgress) -> None:
        if self.evaluation_callback is None:
            return
        try:
            self.evaluation_callback(progress)
        except Exception as error:
            self.audit.event(AuditEvent("evaluation_progress_callback_failed", f"error={error}"))

    def _run_adaptive(
        self,
        answers: Sequence[CandidateAnswerContext],
        config: RuntimeJudgeConfig,
    ) -> PipelineSummary:
        executed = 0
        skipped = 0
        arbiters = 0
        total_answers = len(answers)
        if total_answers == 0:
            self._report_batch_progress(
                BatchProgress(
                    current=0,
                    total=0,
                    percent=100,
                    executed_evaluations=0,
                    skipped_evaluations=0,
                    arbiter_evaluations=0,
                )
            )
            return PipelineSummary(0, 0, 0, 0)

        scheduler = _AdaptiveScheduler(self, config)
        if config.panel_mode == "single":
            assert config.single_judge is not None
            tasks = self._pending_adaptive_tasks(
                answers=answers,
                config=config,
                judge_model=config.single_judge,
                stored_role="principal",
                trigger_reason="single_mode",
                priority=0,
            )
            skipped += tasks.skipped
            for record in scheduler.run(tasks.pending):
                self._persist_adaptive_record(record)
                executed += 1
                self._report_batch_progress(
                    BatchProgress(
                        current=min(executed + skipped, total_answers),
                        total=total_answers,
                        percent=int(min(executed + skipped, total_answers) / total_answers * 100),
                        executed_evaluations=executed,
                        skipped_evaluations=skipped,
                        arbiter_evaluations=0,
                    )
                )
            if not tasks.pending:
                self._report_batch_progress(
                    BatchProgress(
                        current=total_answers,
                        total=total_answers,
                        percent=100,
                        executed_evaluations=executed,
                        skipped_evaluations=skipped,
                        arbiter_evaluations=0,
                    )
                )
            scheduler.report_final()
            return PipelineSummary(total_answers, executed, skipped, 0)

        answer_by_id = {answer.answer_id: answer for answer in answers}
        primary_scores: dict[int, list[int]] = {answer.answer_id: [] for answer in answers}
        primary_tasks: list[_AdaptiveJudgeTask] = []
        primary_task_sequence = 0
        for priority, (judge_model, stored_role) in enumerate(
            zip(config.primary_panel, ("principal", "controle"), strict=False),
            start=0,
        ):
            tasks = self._pending_adaptive_tasks(
                answers=answers,
                config=config,
                judge_model=judge_model,
                stored_role=stored_role,
                trigger_reason="primary_panel",
                priority=priority,
                existing_scores=primary_scores,
                sequence_offset=primary_task_sequence,
            )
            primary_task_sequence += len(answers)
            skipped += tasks.skipped
            primary_tasks.extend(tasks.pending)
        for record in scheduler.run(primary_tasks):
            self._persist_adaptive_record(record)
            primary_scores[record.answer_id].append(record.score)
            executed += 1
            self._report_batch_progress(
                BatchProgress(
                    current=min(executed + skipped, total_answers),
                    total=total_answers,
                    percent=int(min(executed + skipped, total_answers) / total_answers * 100),
                    executed_evaluations=executed,
                    skipped_evaluations=skipped,
                    arbiter_evaluations=arbiters,
                )
            )

        if config.panel_mode == "2plus1":
            assert config.arbiter is not None
            arbiter_tasks: list[_AdaptiveJudgeTask] = []
            for answer_id, scores in primary_scores.items():
                if len(scores) != 2:
                    continue
                answer = answer_by_id[answer_id]
                score_delta = abs(scores[0] - scores[1])
                arbiter_reason = _arbiter_reason(config, score_delta)
                if arbiter_reason is None:
                    self.audit.terminal_event(
                        f"Arbiter skipped for answer {answer.answer_id}: score delta {score_delta}"
                    )
                    self.audit.event(
                        AuditEvent(
                            "arbiter_skipped",
                            (
                                f"answer_id={answer.answer_id} score_delta={score_delta} "
                                f"threshold={config.arbitration_min_delta}"
                            ),
                        )
                    )
                    continue
                if self.repository.evaluation_exists(answer.answer_id, config.arbiter, "arbitro", config.panel_mode):
                    skipped += 1
                    self._report_existing_skip(answer, config, config.arbiter, "arbitro")
                    continue
                arbiter_tasks.append(
                    _AdaptiveJudgeTask(
                        answer=answer,
                        judge_model=config.arbiter,
                        stored_role="arbitro",
                        trigger_reason=arbiter_reason,
                        priority=2,
                        sequence=len(arbiter_tasks),
                    )
                )
            for record in scheduler.run(arbiter_tasks):
                self._persist_adaptive_record(record)
                executed += 1
                arbiters += 1
                self._report_batch_progress(
                    BatchProgress(
                        current=min(executed + skipped, total_answers),
                        total=total_answers,
                        percent=int(min(executed + skipped, total_answers) / total_answers * 100),
                        executed_evaluations=executed,
                        skipped_evaluations=skipped,
                        arbiter_evaluations=arbiters,
                    )
                )

        self._report_batch_progress(
            BatchProgress(
                current=total_answers,
                total=total_answers,
                percent=100,
                executed_evaluations=executed,
                skipped_evaluations=skipped,
                arbiter_evaluations=arbiters,
            )
        )
        scheduler.report_final()
        return PipelineSummary(total_answers, executed, skipped, arbiters)

    def _pending_adaptive_tasks(
        self,
        *,
        answers: Sequence[CandidateAnswerContext],
        config: RuntimeJudgeConfig,
        judge_model: ModelSpec,
        stored_role: StoredJudgeRole,
        trigger_reason: str,
        priority: int,
        existing_scores: dict[int, list[int]] | None = None,
        sequence_offset: int = 0,
    ) -> "_PendingAdaptiveTasks":
        pending: list[_AdaptiveJudgeTask] = []
        skipped = 0
        for sequence, answer in enumerate(answers):
            score_before = self.repository.existing_score(
                answer.answer_id,
                judge_model,
                stored_role,
                config.panel_mode,
            )
            if score_before is not None:
                skipped += 1
                if existing_scores is not None:
                    existing_scores[answer.answer_id].append(score_before)
                self._report_existing_skip(answer, config, judge_model, stored_role, score_before)
                continue
            pending.append(
                _AdaptiveJudgeTask(
                    answer=answer,
                    judge_model=judge_model,
                    stored_role=stored_role,
                    trigger_reason=trigger_reason,
                    priority=priority,
                    sequence=sequence_offset + sequence,
                )
            )
        return _PendingAdaptiveTasks(pending=pending, skipped=skipped)

    def _report_existing_skip(
        self,
        answer: CandidateAnswerContext,
        config: RuntimeJudgeConfig,
        judge_model: ModelSpec,
        stored_role: StoredJudgeRole,
        score: int | None = None,
    ) -> None:
        self.audit.terminal_event(
            f"Skipping answer {answer.answer_id} for {judge_model.requested}: existing evaluation"
        )
        detail = (
            f"answer_id={answer.answer_id} model={judge_model.provider_model} "
            f"role={stored_role} mode={config.panel_mode}"
        )
        if score is not None:
            detail = f"{detail} existing_score={score}"
        self.audit.event(AuditEvent("evaluation_skipped", detail))
        self._report_evaluation_progress(
            EvaluationProgress(
                status="skipped",
                dataset=answer.dataset_name,
                question_id=answer.question_id,
                answer_id=answer.answer_id,
                candidate_model=answer.candidate_model,
                judge_model=judge_model.provider_model,
                role=stored_role,
                panel_mode=config.panel_mode,
                score=score,
                trigger_reason=f"{config.panel_mode}:existing_evaluation",
            )
        )

    def _persist_adaptive_record(self, record: EvaluationRecord) -> None:
        with self.audit.step(
            f"Persisting evaluation for answer {record.answer_id}",
            detail=(
                f"answer_id={record.answer_id} model={record.judge_model.provider_model} "
                f"role={record.stored_role} score={record.score}"
            ),
        ):
            self.repository.persist_evaluation(record)


def _arbiter_reason(config: RuntimeJudgeConfig, score_delta: int) -> str | None:
    if config.always_run_arbiter:
        return "forced_by_cli_or_env"
    if score_delta >= config.arbitration_min_delta:
        return "score_delta"
    return None


@dataclass(frozen=True)
class _AdaptiveJudgeTask:
    answer: CandidateAnswerContext
    judge_model: ModelSpec
    stored_role: StoredJudgeRole
    trigger_reason: str
    priority: int
    sequence: int


@dataclass(frozen=True)
class _PendingAdaptiveTasks:
    pending: list[_AdaptiveJudgeTask]
    skipped: int


@dataclass(frozen=True)
class _AdaptiveGroupKey:
    provider: str
    base_url: str
    api_key_fingerprint: str
    model_id: str

    @property
    def label(self) -> str:
        return (
            f"provider={self.provider} base_url={self.base_url} "
            f"api_key={self.api_key_fingerprint} model={self.model_id}"
        )


@dataclass
class _AdaptiveGroupState:
    key: _AdaptiveGroupKey
    current_concurrency: int
    max_concurrency: int
    in_flight: int = 0
    cooldown_until: float = 0.0
    successes_since_change: int = 0
    consecutive_failures: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    rate_limits: int = 0
    retries: int = 0
    requeued: int = 0
    wait_for_global_idle: bool = False
    disabled: bool = False


@dataclass
class _QueuedAdaptiveTask:
    task: _AdaptiveJudgeTask
    group_key: _AdaptiveGroupKey
    attempt: int = 0
    ready_at: float = 0.0


@dataclass(frozen=True)
class _CompletedAdaptiveTask:
    queued: _QueuedAdaptiveTask
    record: EvaluationRecord


class _AdaptiveScheduler:
    def __init__(self, pipeline: JudgePipeline, config: RuntimeJudgeConfig) -> None:
        self.pipeline = pipeline
        self.config = config
        self.groups: dict[_AdaptiveGroupKey, _AdaptiveGroupState] = {}

    def run(self, tasks: Sequence[_AdaptiveJudgeTask]) -> list[EvaluationRecord]:
        if not tasks:
            return []

        pending = [
            _QueuedAdaptiveTask(
                task=task,
                group_key=self._group_key(task),
                ready_at=self.pipeline.monotonic_func(),
            )
            for task in sorted(tasks, key=lambda item: (item.priority, item.sequence))
        ]
        for queued in pending:
            self._ensure_group(queued.group_key)

        max_workers = max(1, sum(group.max_concurrency for group in self.groups.values()))
        records: list[EvaluationRecord] = []
        futures: dict[Future[_CompletedAdaptiveTask], _QueuedAdaptiveTask] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pending or futures:
                if self.pipeline.should_stop():
                    self.pipeline.audit.event(
                        AuditEvent(
                            "adaptive_scheduler_stop_requested",
                            f"pending={len(pending)} in_flight={len(futures)}",
                        )
                    )
                    pending.clear()

                submitted = self._submit_ready(pending, futures, executor)
                if submitted:
                    continue
                if not futures:
                    delay = self._next_delay(pending)
                    if delay > 0:
                        self.pipeline.sleep_func(delay)
                    continue

                done, _ = wait(futures.keys(), timeout=self._next_wait_timeout(pending), return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    queued = futures.pop(future)
                    state = self.groups[queued.group_key]
                    state.in_flight -= 1
                    try:
                        completed = future.result()
                    except Exception as error:
                        self._handle_failure(error, queued, pending, state, active_other_tasks=len(futures))
                    else:
                        self._handle_success(state)
                        records.append(completed.record)
        return records

    def report_final(self) -> None:
        for state in self.groups.values():
            self.pipeline.audit.event(
                AuditEvent(
                    "adaptive_concurrency_final",
                    (
                        f"{state.key.label} final_concurrency={state.current_concurrency} "
                        f"successes={state.successes} failures={state.failures} timeouts={state.timeouts} "
                        f"rate_limits={state.rate_limits} retries={state.retries} requeued={state.requeued} "
                        f"disabled={state.disabled}"
                    ),
                )
            )

    def _submit_ready(
        self,
        pending: list[_QueuedAdaptiveTask],
        futures: dict[Future[_CompletedAdaptiveTask], _QueuedAdaptiveTask],
        executor: ThreadPoolExecutor,
    ) -> bool:
        now = self.pipeline.monotonic_func()
        submitted = False
        index = 0
        while index < len(pending):
            queued = pending[index]
            state = self.groups[queued.group_key]
            if state.wait_for_global_idle:
                if futures:
                    index += 1
                    continue
                state.wait_for_global_idle = False
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
                executor.submit(self._execute_queued_task, queued)
            ] = queued
            submitted = True
        return submitted

    def _execute_queued_task(self, queued: _QueuedAdaptiveTask) -> _CompletedAdaptiveTask:
        task = queued.task
        record = self.pipeline._execute_judge(
            answer=task.answer,
            config=self.config,
            judge_model=task.judge_model,
            stored_role=task.stored_role,
            trigger_reason=task.trigger_reason,
            terminal_progress=False,
        )
        return _CompletedAdaptiveTask(queued=queued, record=record)

    def _handle_success(self, state: _AdaptiveGroupState) -> None:
        state.successes += 1
        state.consecutive_failures = 0
        state.successes_since_change += 1
        threshold = self._success_threshold(state)
        if (
            state.successes_since_change >= threshold
            and state.current_concurrency < state.max_concurrency
        ):
            old = state.current_concurrency
            state.current_concurrency += 1
            state.successes_since_change = 0
            self.pipeline.audit.event(
                AuditEvent(
                    "adaptive_concurrency_increased",
                    f"{state.key.label} from={old} to={state.current_concurrency} successes={state.successes}",
                )
            )

    def _success_threshold(self, state: _AdaptiveGroupState) -> int:
        if any(group is not state and group.wait_for_global_idle for group in self.groups.values()):
            return 1
        return self.config.settings.judge_adaptive_success_threshold

    def _handle_failure(
        self,
        error: Exception,
        queued: _QueuedAdaptiveTask,
        pending: list[_QueuedAdaptiveTask],
        state: _AdaptiveGroupState,
        *,
        active_other_tasks: int,
    ) -> None:
        state.failures += 1
        state.consecutive_failures += 1
        state.successes_since_change = 0
        if _is_timeout_error(error):
            state.timeouts += 1
        if isinstance(error, RemoteJudgeError) and error.status_code == 429:
            state.rate_limits += 1

        if _should_wait_for_global_idle(error, state, active_other_tasks):
            state.wait_for_global_idle = True
            pending.append(
                _QueuedAdaptiveTask(
                    task=queued.task,
                    group_key=queued.group_key,
                    attempt=queued.attempt,
                    ready_at=self.pipeline.monotonic_func(),
                )
            )
            pending.sort(key=lambda item: (item.task.priority, item.ready_at, item.task.sequence))
            self.pipeline.audit.event(
                AuditEvent(
                    "adaptive_task_waiting_for_global_idle",
                    (
                        f"{state.key.label} answer_id={queued.task.answer.answer_id} "
                        f"role={queued.task.stored_role} active_other_tasks={active_other_tasks} error={error}"
                    ),
                )
            )
            return

        if not _should_retry(error) or queued.attempt >= self.config.settings.judge_adaptive_max_retries:
            self._disable_group_after_failure(error, queued, pending, state)
            return

        if isinstance(error, RemoteJudgeError) and error.status_code == 429:
            self._reduce_concurrency(state, reason="http_429")
        elif state.consecutive_failures >= 2:
            self._reduce_concurrency(state, reason="recurrent_retryable_failure")

        backoff = self._backoff_seconds(error, queued.attempt)
        state.cooldown_until = max(state.cooldown_until, self.pipeline.monotonic_func() + backoff)
        state.retries += 1
        state.requeued += 1
        retried = _QueuedAdaptiveTask(
            task=queued.task,
            group_key=queued.group_key,
            attempt=queued.attempt + 1,
            ready_at=state.cooldown_until,
        )
        pending.append(retried)
        pending.sort(key=lambda item: (item.task.priority, item.ready_at, item.task.sequence))
        self.pipeline.audit.event(
            AuditEvent(
                "adaptive_task_requeued",
                (
                    f"{state.key.label} answer_id={queued.task.answer.answer_id} role={queued.task.stored_role} "
                    f"attempt={retried.attempt} backoff_seconds={backoff:.3f} error={error}"
                ),
            )
        )

    def _disable_group_after_failure(
        self,
        error: Exception,
        queued: _QueuedAdaptiveTask,
        pending: list[_QueuedAdaptiveTask],
        state: _AdaptiveGroupState,
    ) -> None:
        state.disabled = True
        discarded = 0
        index = 0
        while index < len(pending):
            if pending[index].group_key == queued.group_key:
                pending.pop(index)
                discarded += 1
                continue
            index += 1
        self.pipeline.audit.event(
            AuditEvent(
                "adaptive_task_failed",
                (
                    f"{state.key.label} answer_id={queued.task.answer.answer_id} "
                    f"role={queued.task.stored_role} attempts={queued.attempt + 1} "
                    f"discarded_pending={discarded} error={error}"
                ),
            )
        )
        self.pipeline.audit.event(
            AuditEvent(
                "adaptive_group_disabled",
                f"{state.key.label} discarded_pending={discarded} error={error}",
            )
        )

    def _reduce_concurrency(self, state: _AdaptiveGroupState, *, reason: str) -> None:
        old = state.current_concurrency
        state.current_concurrency = max(1, state.current_concurrency - 1)
        if state.current_concurrency != old:
            self.pipeline.audit.event(
                AuditEvent(
                    "adaptive_concurrency_reduced",
                    f"{state.key.label} from={old} to={state.current_concurrency} reason={reason}",
                )
            )

    def _backoff_seconds(self, error: Exception, attempt: int) -> float:
        if isinstance(error, RemoteJudgeError) and error.retry_after_seconds is not None:
            return min(error.retry_after_seconds, self.config.settings.judge_adaptive_max_backoff_seconds)
        base = self.config.settings.judge_adaptive_base_backoff_seconds
        maximum = self.config.settings.judge_adaptive_max_backoff_seconds
        jitter = self.pipeline.jitter_func()
        return min(maximum, base * (2 ** attempt) + jitter)

    def _next_wait_timeout(self, pending: list[_QueuedAdaptiveTask]) -> float:
        delay = self._next_delay(pending)
        if delay <= 0:
            return 0.05
        return min(delay, 0.05)

    def _next_delay(self, pending: list[_QueuedAdaptiveTask]) -> float:
        if not pending:
            return 0.0
        now = self.pipeline.monotonic_func()
        next_ready = min(
            max(queued.ready_at, self.groups[queued.group_key].cooldown_until)
            for queued in pending
        )
        return max(0.0, next_ready - now)

    def _ensure_group(self, key: _AdaptiveGroupKey) -> None:
        if key in self.groups:
            return
        initial = self.config.settings.judge_adaptive_initial_concurrency
        maximum = self.config.settings.judge_adaptive_max_concurrency
        state = _AdaptiveGroupState(
            key=key,
            current_concurrency=min(initial, maximum),
            max_concurrency=maximum,
        )
        self.groups[key] = state
        self.pipeline.audit.event(
            AuditEvent(
                "adaptive_concurrency_initial",
                f"{key.label} initial_concurrency={state.current_concurrency} max_concurrency={maximum}",
            )
        )

    def _group_key(self, task: _AdaptiveJudgeTask) -> _AdaptiveGroupKey:
        endpoint_key = _endpoint_key_for_role(task.stored_role, self.config.panel_mode)
        endpoint = _resolve_remote_endpoint(
            self.config,
            task.judge_model,
            endpoint_key,
        )
        return _AdaptiveGroupKey(
            provider=self.config.provider,
            base_url=endpoint.base_url or "<missing>",
            api_key_fingerprint=_fingerprint(endpoint.api_key),
            model_id=task.judge_model.provider_model,
        )


def _endpoint_key_for_role(stored_role: StoredJudgeRole, panel_mode: str) -> str:
    if panel_mode == "single":
        return "JUDGE"
    if stored_role == "principal":
        return "JUDGE"
    if stored_role == "controle":
        return "SECONDARY_JUDGE"
    return "ARBITER"


def _resolve_remote_endpoint(
    config: RuntimeJudgeConfig,
    model: ModelSpec,
    endpoint_key: str,
) -> RemoteJudgeEndpoint:
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


def _endpoint_keys(model: str) -> tuple[str, ...]:
    keys = [_endpoint_key(model)]
    if "/" in model:
        keys.append(_endpoint_key(model.rsplit("/", 1)[-1]))
    return tuple(dict.fromkeys(key for key in keys if key))


def _endpoint_key(value: str) -> str:
    import re

    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _fingerprint(value: str | None) -> str:
    if not value:
        return "<missing>"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _should_retry(error: Exception) -> bool:
    return isinstance(error, RemoteJudgeError) and error.retryable


def _should_wait_for_global_idle(
    error: Exception,
    state: _AdaptiveGroupState,
    active_other_tasks: int,
) -> bool:
    return (
        isinstance(error, RemoteJudgeError)
        and error.status_code == 429
        and error.retryable
        and state.current_concurrency <= 1
        and active_other_tasks > 0
        and "concurrency limit exceeded" in str(error).lower()
    )


def _is_timeout_error(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, RemoteJudgeError):
        return "timed out" in str(error).lower() or "timeout" in str(error).lower()
    return False
