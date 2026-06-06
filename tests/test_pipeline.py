from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest

from atividade_2.config import load_settings, resolve_runtime_config
from atividade_2.contracts import (
    BatchProgress,
    CandidateAnswerContext,
    EvaluationProgress,
    EvaluationRecord,
    JudgeRawResponse,
    ModelSpec,
    StoredJudgeRole,
)
from atividade_2.judge_clients.remote_http import RemoteJudgeError
from atividade_2.pipeline import JudgePipeline
from atividade_2.repositories import InMemoryJudgeRepository


BASE_ENV = {
    "REMOTE_JUDGE_BASE_URL": "https://example.invalid/v1",
    "REMOTE_JUDGE_API_KEY": "test-key",
    "REMOTE_JUDGE_MODEL": "gpt-oss-120b",
    "REMOTE_SECONDARY_JUDGE_MODEL": "llama-3.3-70b-instruct",
    "REMOTE_ARBITER_JUDGE_MODEL": "m-prometheus-14b",
}


class FakeJudgeClient:
    def __init__(self, scores: dict[str, int]) -> None:
        self.scores = scores
        self.calls: list[str] = []
        self.endpoint_keys: list[str | None] = []

    def judge(
        self,
        prompt: str,
        model: str,
        *,
        requested_model: str | None = None,
        endpoint_key: str | None = None,
    ) -> JudgeRawResponse:
        self.calls.append(model)
        self.endpoint_keys.append(endpoint_key)
        score = self.scores[model]
        return JudgeRawResponse(
            text=f'{{"score": {score}, "rationale": "nota {score}"}}',
            provider="fake",
            model=model,
            latency_ms=1,
        )


class SlowRecordingJudgeClient(FakeJudgeClient):
    def __init__(self, scores: dict[str, int], *, delay_seconds: float) -> None:
        super().__init__(scores)
        self.delay_seconds = delay_seconds
        self.current_calls = 0
        self.max_seen = 0
        self._lock = threading.Lock()

    def judge(
        self,
        prompt: str,
        model: str,
        *,
        requested_model: str | None = None,
        endpoint_key: str | None = None,
    ) -> JudgeRawResponse:
        with self._lock:
            self.current_calls += 1
            self.max_seen = max(self.max_seen, self.current_calls)
        try:
            time.sleep(self.delay_seconds)
            return super().judge(prompt, model, requested_model=requested_model, endpoint_key=endpoint_key)
        finally:
            with self._lock:
                self.current_calls -= 1


class FlakyRateLimitJudgeClient(FakeJudgeClient):
    def __init__(self, scores: dict[str, int]) -> None:
        super().__init__(scores)
        self.fail_once = True

    def judge(
        self,
        prompt: str,
        model: str,
        *,
        requested_model: str | None = None,
        endpoint_key: str | None = None,
    ) -> JudgeRawResponse:
        self.calls.append(model)
        self.endpoint_keys.append(endpoint_key)
        if self.fail_once:
            self.fail_once = False
            raise RemoteJudgeError(
                "Remote judge returned HTTP 429: rate limited",
                status_code=429,
                retry_after_seconds=0,
                retryable=True,
            )
        score = self.scores[model]
        return JudgeRawResponse(
            text=f'{{"score": {score}, "rationale": "nota {score}"}}',
            provider="fake",
            model=model,
            latency_ms=1,
        )


class DailyTokenQuotaJudgeClient(FakeJudgeClient):
    def judge(
        self,
        prompt: str,
        model: str,
        *,
        requested_model: str | None = None,
        endpoint_key: str | None = None,
    ) -> JudgeRawResponse:
        self.calls.append(model)
        self.endpoint_keys.append(endpoint_key)
        raise RemoteJudgeError(
            (
                "Remote judge returned HTTP 429: Rate limit reached for model "
                "`llama-3.3-70b-versatile` on tokens per day (TPD): Limit 100000, "
                "Used 99787, Requested 1094. Please try again in 12m41.184s."
            ),
            status_code=429,
            retry_after_seconds=761,
            retryable=False,
        )


class ModelSpecificDailyTokenQuotaJudgeClient(FakeJudgeClient):
    def __init__(self, scores: dict[str, int], failing_model: str) -> None:
        super().__init__(scores)
        self.failing_model = failing_model

    def judge(
        self,
        prompt: str,
        model: str,
        *,
        requested_model: str | None = None,
        endpoint_key: str | None = None,
    ) -> JudgeRawResponse:
        self.calls.append(model)
        self.endpoint_keys.append(endpoint_key)
        if model == self.failing_model:
            raise RemoteJudgeError(
                (
                    "Remote judge returned HTTP 429: Rate limit reached for model "
                    f"`{model}` on tokens per day (TPD): Limit 100000, Used 99787, "
                    "Requested 1094. Please try again in 12m41.184s."
                ),
                status_code=429,
                retry_after_seconds=761,
                retryable=False,
            )
        score = self.scores[model]
        return JudgeRawResponse(
            text=f'{{"score": {score}, "rationale": "nota {score}"}}',
            provider="fake",
            model=model,
            latency_ms=1,
        )


class ConcurrencyLimitJudgeClient(FakeJudgeClient):
    def __init__(self, scores: dict[str, int], *, slow_model: str, limited_model: str) -> None:
        super().__init__(scores)
        self.slow_model = slow_model
        self.limited_model = limited_model
        self.limited_fail_once = True
        self.events: list[tuple[str, str]] = []
        self.current_by_model: dict[str, int] = {}
        self.max_by_model: dict[str, int] = {}
        self._lock = threading.Lock()

    def judge(
        self,
        prompt: str,
        model: str,
        *,
        requested_model: str | None = None,
        endpoint_key: str | None = None,
    ) -> JudgeRawResponse:
        with self._lock:
            self.events.append(("start", model))
            current = self.current_by_model.get(model, 0) + 1
            self.current_by_model[model] = current
            self.max_by_model[model] = max(self.max_by_model.get(model, 0), current)
        try:
            if model == self.limited_model and self.limited_fail_once:
                self.limited_fail_once = False
                self.calls.append(model)
                self.endpoint_keys.append(endpoint_key)
                raise RemoteJudgeError(
                    (
                        "Remote judge returned HTTP 429: Concurrency limit exceeded. "
                        "Active concurrent requests: 1 units. This request requires: 4 units."
                    ),
                    status_code=429,
                    retryable=True,
                )
            if model == self.slow_model:
                time.sleep(0.03)
            response = super().judge(prompt, model, requested_model=requested_model, endpoint_key=endpoint_key)
            with self._lock:
                self.events.append(("end", model))
            return response
        finally:
            with self._lock:
                self.current_by_model[model] -= 1


class EventRecordingJudgeClient(FakeJudgeClient):
    def __init__(self, scores: dict[str, int], slow_model: str) -> None:
        super().__init__(scores)
        self.slow_model = slow_model
        self.events: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def judge(
        self,
        prompt: str,
        model: str,
        *,
        requested_model: str | None = None,
        endpoint_key: str | None = None,
    ) -> JudgeRawResponse:
        with self._lock:
            self.events.append(("start", model))
        if model == self.slow_model:
            time.sleep(0.03)
        response = super().judge(prompt, model, requested_model=requested_model, endpoint_key=endpoint_key)
        with self._lock:
            self.events.append(("end", model))
        return response


class RecordingAudit:
    def __init__(self) -> None:
        self.terminal_messages: list[str] = []
        self.file_events: list[tuple[str, str | None]] = []
        self.events: list[tuple[str, str | None]] = []

    def terminal_event(self, message: str) -> None:
        self.terminal_messages.append(message)

    def file_event(self, message: str, detail: str | None = None) -> None:
        self.file_events.append((message, detail))

    def event(self, event) -> None:
        self.events.append((event.message, event.detail))

    @contextmanager
    def step(self, message: str, *, detail: str | None = None, terminal: bool = True):
        yield


def answer() -> CandidateAnswerContext:
    return CandidateAnswerContext(
        av1_answer_id=1,
        candidate_answer_id=None,
        question_id=1,
        dataset_name="OAB_Bench",
        question_text="Enunciado",
        reference_answer="Rubrica",
        candidate_answer="Resposta candidata",
        candidate_model="candidate",
    )


def j2_answer() -> CandidateAnswerContext:
    return CandidateAnswerContext(
        av1_answer_id=1,
        candidate_answer_id=None,
        question_id=101,
        dataset_name="OAB_Exames",
        question_text="Qual alternativa correta?",
        reference_answer="A",
        candidate_answer="Portanto, a opção correta é A.",
        candidate_model="candidate",
    )


def answer_with_id(answer_id: int) -> CandidateAnswerContext:
    base_answer = answer()
    return CandidateAnswerContext(
        av1_answer_id=answer_id,
        candidate_answer_id=None,
        question_id=base_answer.question_id,
        dataset_name=base_answer.dataset_name,
        question_text=base_answer.question_text,
        reference_answer=base_answer.reference_answer,
        candidate_answer=base_answer.candidate_answer,
        candidate_model=base_answer.candidate_model,
    )


def _record_for_existing_score(
    *,
    answer_id: int,
    judge_model: ModelSpec,
    stored_role: StoredJudgeRole,
    score: int = 5,
) -> EvaluationRecord:
    return EvaluationRecord(
        av1_answer_id=answer_id,
        candidate_answer_id=None,
        judge_model=judge_model,
        prompt_id=None,
        stored_role=stored_role,
        panel_mode="2plus1",
        trigger_reason="primary_panel",
        score=score,
        rationale="existing",
        latency_ms=1,
    )


def test_single_mode_runs_one_judge() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="single")
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient({"openai/gpt-oss-120b": 5})

    summary = JudgePipeline(repo, client).run([answer()], config)

    assert summary.executed_evaluations == 1
    assert len(repo.records) == 1
    assert repo.records[0].stored_role == "principal"
    assert client.endpoint_keys == ["JUDGE"]


def test_j2_rejects_non_binary_judge_score() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="single")
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient({"openai/gpt-oss-120b": 3})

    with pytest.raises(Exception, match="one of: 1, 5"):
        JudgePipeline(repo, client).run([j2_answer()], config)

    assert repo.records == []


def test_pipeline_reports_batch_progress_after_each_answer() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="single")
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient({"openai/gpt-oss-120b": 5})
    audit = RecordingAudit()
    progress_events: list[BatchProgress] = []

    JudgePipeline(repo, client, audit=audit, progress_callback=progress_events.append).run(
        [answer_with_id(1), answer_with_id(2)],
        config,
    )

    assert "Batch progress: 1/2 answers (50%) | executed=1 skipped=0 arbiters=0" in audit.terminal_messages
    assert "Batch progress: 2/2 answers (100%) | executed=2 skipped=0 arbiters=0" in audit.terminal_messages
    assert (
        "batch_progress",
        "current=2 total=2 percent=100 executed=2 skipped=0 arbiters=0",
    ) in audit.events
    assert progress_events == [
        BatchProgress(
            current=1,
            total=2,
            percent=50,
            executed_evaluations=1,
            skipped_evaluations=0,
            arbiter_evaluations=0,
        ),
        BatchProgress(
            current=2,
            total=2,
            percent=100,
            executed_evaluations=2,
            skipped_evaluations=0,
            arbiter_evaluations=0,
        ),
    ]


def test_pipeline_stops_between_answers_without_discarding_completed_records() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="single")
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient({"openai/gpt-oss-120b": 5})
    audit = RecordingAudit()
    progress_events: list[BatchProgress] = []
    stop_after_first = False

    def record_progress(progress: BatchProgress) -> None:
        nonlocal stop_after_first
        progress_events.append(progress)
        stop_after_first = True

    summary = JudgePipeline(
        repo,
        client,
        audit=audit,
        progress_callback=record_progress,
        should_stop=lambda: stop_after_first,
    ).run([answer_with_id(1), answer_with_id(2)], config)

    assert summary.executed_evaluations == 1
    assert [record.answer_id for record in repo.records] == [1]
    assert client.calls == ["openai/gpt-oss-120b"]
    assert progress_events == [
        BatchProgress(
            current=1,
            total=2,
            percent=50,
            executed_evaluations=1,
            skipped_evaluations=0,
            arbiter_evaluations=0,
        )
    ]
    assert (
        "pipeline_cancelled",
        "current=1 total=2 executed=1 skipped=0 arbiters=0",
    ) in audit.events


def test_pipeline_reports_evaluation_rows_for_web_table() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="single")
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient({"openai/gpt-oss-120b": 5})
    evaluation_events: list[EvaluationProgress] = []

    JudgePipeline(repo, client, evaluation_callback=evaluation_events.append).run([answer()], config)

    assert [event.status for event in evaluation_events] == ["running", "success"]
    success = evaluation_events[-1]
    assert success.dataset == "OAB_Bench"
    assert success.question_id == 1
    assert success.candidate_model == "candidate"
    assert success.judge_model == "openai/gpt-oss-120b"
    assert success.role == "principal"
    assert success.score == 5
    assert success.latency_ms == 1
    assert success.prompt
    assert success.raw_response == '{"score": 5, "rationale": "nota 5"}'
    assert success.rationale == "nota 5"


def test_pipeline_reports_complete_progress_for_empty_batch() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="single")
    audit = RecordingAudit()
    progress_events: list[BatchProgress] = []

    summary = JudgePipeline(
        InMemoryJudgeRepository(),
        FakeJudgeClient({"openai/gpt-oss-120b": 5}),
        audit=audit,
        progress_callback=progress_events.append,
    ).run([], config)

    assert summary.selected_answers == 0
    assert progress_events == [
        BatchProgress(
            current=0,
            total=0,
            percent=100,
            executed_evaluations=0,
            skipped_evaluations=0,
            arbiter_evaluations=0,
        )
    ]


def test_pipeline_does_not_fail_when_progress_callback_fails() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="single")
    audit = RecordingAudit()

    def fail_progress_callback(progress: BatchProgress) -> None:
        raise RuntimeError("progress sink unavailable")

    summary = JudgePipeline(
        InMemoryJudgeRepository(),
        FakeJudgeClient({"openai/gpt-oss-120b": 5}),
        audit=audit,
        progress_callback=fail_progress_callback,
    ).run([answer()], config)

    assert summary.executed_evaluations == 1
    assert (
        "batch_progress_callback_failed",
        "error=progress sink unavailable",
    ) in audit.events


def test_primary_only_runs_panel_without_arbiter() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="primary_only")
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient(
        {
            "openai/gpt-oss-120b": 5,
            "meta-llama/Llama-3.3-70B-Instruct": 1,
            "Unbabel/M-Prometheus-14B": 4,
        }
    )

    summary = JudgePipeline(repo, client).run([answer()], config)

    assert summary.executed_evaluations == 2
    assert summary.arbiter_evaluations == 0
    assert client.calls == ["openai/gpt-oss-120b", "meta-llama/Llama-3.3-70B-Instruct"]
    assert client.endpoint_keys == ["JUDGE", "SECONDARY_JUDGE"]


def test_2plus1_skips_arbiter_below_threshold() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="2plus1")
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient(
        {
            "openai/gpt-oss-120b": 4,
            "meta-llama/Llama-3.3-70B-Instruct": 3,
            "Unbabel/M-Prometheus-14B": 1,
        }
    )

    summary = JudgePipeline(repo, client).run([answer()], config)

    assert summary.executed_evaluations == 2
    assert summary.arbiter_evaluations == 0


def test_2plus1_does_not_report_skipped_arbiter_as_table_row() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="2plus1")
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient(
        {
            "openai/gpt-oss-120b": 4,
            "meta-llama/Llama-3.3-70B-Instruct": 3,
            "Unbabel/M-Prometheus-14B": 1,
        }
    )
    evaluation_events: list[EvaluationProgress] = []

    summary = JudgePipeline(repo, client, evaluation_callback=evaluation_events.append).run([answer()], config)

    assert summary.executed_evaluations == 2
    assert summary.skipped_evaluations == 0
    assert summary.arbiter_evaluations == 0
    assert [(event.role, event.status) for event in evaluation_events] == [
        ("principal", "running"),
        ("principal", "success"),
        ("controle", "running"),
        ("controle", "success"),
    ]


def test_2plus1_runs_arbiter_at_threshold() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="2plus1")
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient(
        {
            "openai/gpt-oss-120b": 5,
            "meta-llama/Llama-3.3-70B-Instruct": 3,
            "Unbabel/M-Prometheus-14B": 4,
        }
    )

    summary = JudgePipeline(repo, client).run([answer()], config)

    assert summary.executed_evaluations == 3
    assert summary.arbiter_evaluations == 1
    assert client.endpoint_keys == ["JUDGE", "SECONDARY_JUDGE", "ARBITER"]


def test_always_run_arbiter_forces_execution() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="2plus1", always_run_arbiter=True)
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient(
        {
            "openai/gpt-oss-120b": 4,
            "meta-llama/Llama-3.3-70B-Instruct": 4,
            "Unbabel/M-Prometheus-14B": 4,
        }
    )

    summary = JudgePipeline(repo, client).run([answer()], config)

    assert summary.executed_evaluations == 3
    assert summary.arbiter_evaluations == 1


def test_duplicate_evaluation_is_skipped() -> None:
    settings = load_settings(dotenv_path=None, env=BASE_ENV)
    config = resolve_runtime_config(settings, panel_mode="single")
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient({"openai/gpt-oss-120b": 5})
    pipeline = JudgePipeline(repo, client)

    first = pipeline.run([answer()], config)
    second = pipeline.run([answer()], config)

    assert first.executed_evaluations == 1
    assert second.executed_evaluations == 0
    assert second.skipped_evaluations == 1
    assert len(repo.records) == 1


def test_primary_only_supports_parallel_strategy() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "parallel"
    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="primary_only")
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient(
        {
            "openai/gpt-oss-120b": 5,
            "meta-llama/Llama-3.3-70B-Instruct": 4,
        }
    )

    summary = JudgePipeline(repo, client).run([answer()], config)

    assert config.execution_strategy == "parallel"
    assert summary.executed_evaluations == 2
    assert sorted(client.calls) == [
        "meta-llama/Llama-3.3-70B-Instruct",
        "openai/gpt-oss-120b",
    ]


def test_adaptive_single_runs_multiple_answers_with_initial_concurrency_limit() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"
    env["JUDGE_ADAPTIVE_INITIAL_CONCURRENCY"] = "2"
    env["JUDGE_ADAPTIVE_MAX_CONCURRENCY"] = "2"
    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="single")
    repo = InMemoryJudgeRepository()
    client = SlowRecordingJudgeClient({"openai/gpt-oss-120b": 5}, delay_seconds=0.02)

    summary = JudgePipeline(repo, client).run(
        [answer_with_id(1), answer_with_id(2), answer_with_id(3), answer_with_id(4)],
        config,
    )

    assert summary.executed_evaluations == 4
    assert len(repo.records) == 4
    assert client.max_seen == 2


def test_adaptive_scheduler_increases_concurrency_after_success_threshold() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"
    env["JUDGE_ADAPTIVE_INITIAL_CONCURRENCY"] = "1"
    env["JUDGE_ADAPTIVE_MAX_CONCURRENCY"] = "2"
    env["JUDGE_ADAPTIVE_SUCCESS_THRESHOLD"] = "1"
    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="single")
    repo = InMemoryJudgeRepository()
    audit = RecordingAudit()
    client = SlowRecordingJudgeClient({"openai/gpt-oss-120b": 5}, delay_seconds=0.02)

    JudgePipeline(repo, client, audit=audit).run(
        [answer_with_id(1), answer_with_id(2), answer_with_id(3)],
        config,
    )

    assert any(message == "adaptive_concurrency_increased" for message, _ in audit.events)
    assert client.max_seen == 2


def test_adaptive_scheduler_requeues_429_with_backoff_and_reduces_concurrency() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"
    env["JUDGE_ADAPTIVE_INITIAL_CONCURRENCY"] = "2"
    env["JUDGE_ADAPTIVE_MAX_CONCURRENCY"] = "2"
    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="single")
    repo = InMemoryJudgeRepository()
    audit = RecordingAudit()
    client = FlakyRateLimitJudgeClient({"openai/gpt-oss-120b": 5})

    summary = JudgePipeline(repo, client, audit=audit, jitter_func=lambda: 0).run([answer()], config)

    assert summary.executed_evaluations == 1
    assert client.calls == ["openai/gpt-oss-120b", "openai/gpt-oss-120b"]
    assert any(message == "adaptive_concurrency_reduced" for message, _ in audit.events)
    assert any(message == "adaptive_task_requeued" for message, _ in audit.events)


def test_adaptive_scheduler_stops_on_daily_token_quota_429_without_retry() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"
    env["JUDGE_ADAPTIVE_INITIAL_CONCURRENCY"] = "1"
    env["JUDGE_ADAPTIVE_MAX_RETRIES"] = "3"
    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="single", judge_model="llama-3.3-70b-versatile")
    repo = InMemoryJudgeRepository()
    audit = RecordingAudit()
    client = DailyTokenQuotaJudgeClient({"llama-3.3-70b-versatile": 5})

    summary = JudgePipeline(repo, client, audit=audit).run([answer_with_id(1), answer_with_id(2)], config)

    assert summary.executed_evaluations == 0
    assert client.calls == ["llama-3.3-70b-versatile"]
    assert not repo.records
    assert any(message == "adaptive_task_failed" for message, _ in audit.events)
    assert any(message == "adaptive_group_disabled" for message, _ in audit.events)
    assert not any(message == "adaptive_task_requeued" for message, _ in audit.events)


def test_adaptive_scheduler_waits_for_other_judge_on_provider_concurrency_limit() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"
    env["JUDGE_ADAPTIVE_INITIAL_CONCURRENCY"] = "1"
    env["JUDGE_ADAPTIVE_MAX_CONCURRENCY"] = "1"
    env["JUDGE_ADAPTIVE_MAX_RETRIES"] = "3"
    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="primary_only")
    repo = InMemoryJudgeRepository()
    audit = RecordingAudit()
    judge_1 = "openai/gpt-oss-120b"
    judge_2 = "meta-llama/Llama-3.3-70B-Instruct"
    client = ConcurrencyLimitJudgeClient(
        {judge_1: 5, judge_2: 4},
        slow_model=judge_1,
        limited_model=judge_2,
    )

    summary = JudgePipeline(repo, client, audit=audit, jitter_func=lambda: 0).run([answer_with_id(1)], config)

    assert summary.executed_evaluations == 2
    assert client.calls == [judge_2, judge_1, judge_2]
    assert client.events.index(("end", judge_1)) < client.events.index(("start", judge_2), 2)
    assert any(message == "adaptive_task_waiting_for_global_idle" for message, _ in audit.events)
    assert not any(message == "adaptive_task_requeued" for message, _ in audit.events)


def test_adaptive_scheduler_still_scales_healthy_judge_while_other_waits_for_idle() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"
    env["JUDGE_ADAPTIVE_INITIAL_CONCURRENCY"] = "1"
    env["JUDGE_ADAPTIVE_MAX_CONCURRENCY"] = "2"
    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="primary_only")
    repo = InMemoryJudgeRepository()
    audit = RecordingAudit()
    judge_1 = "openai/gpt-oss-120b"
    judge_2 = "meta-llama/Llama-3.3-70B-Instruct"
    client = ConcurrencyLimitJudgeClient(
        {judge_1: 5, judge_2: 4},
        slow_model=judge_1,
        limited_model=judge_2,
    )

    summary = JudgePipeline(repo, client, audit=audit, jitter_func=lambda: 0).run(
        [answer_with_id(1), answer_with_id(2), answer_with_id(3)],
        config,
    )

    assert summary.executed_evaluations == 6
    assert client.max_by_model[judge_1] == 2
    assert any(
        message == "adaptive_concurrency_increased" and f"model={judge_1}" in str(detail)
        for message, detail in audit.events
    )
    assert any(message == "adaptive_task_waiting_for_global_idle" for message, _ in audit.events)


def test_adaptive_scheduler_continues_other_models_after_one_model_quota_failure() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"
    env["JUDGE_ADAPTIVE_INITIAL_CONCURRENCY"] = "1"
    env["JUDGE_ADAPTIVE_MAX_CONCURRENCY"] = "1"
    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="primary_only")
    repo = InMemoryJudgeRepository()
    audit = RecordingAudit()
    judge_1 = "openai/gpt-oss-120b"
    judge_2 = "meta-llama/Llama-3.3-70B-Instruct"
    client = ModelSpecificDailyTokenQuotaJudgeClient({judge_1: 5, judge_2: 4}, failing_model=judge_2)

    summary = JudgePipeline(repo, client, audit=audit).run(
        [answer_with_id(1), answer_with_id(2), answer_with_id(3)],
        config,
    )

    assert summary.executed_evaluations == 3
    assert [record.judge_model.provider_model for record in repo.records] == [judge_1, judge_1, judge_1]
    assert client.calls.count(judge_1) == 3
    assert client.calls.count(judge_2) == 1
    assert any(message == "adaptive_group_disabled" for message, _ in audit.events)


def test_adaptive_2plus1_starts_each_judge_from_its_own_pending_answers() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"
    env["JUDGE_ADAPTIVE_INITIAL_CONCURRENCY"] = "1"
    env["JUDGE_ADAPTIVE_MAX_CONCURRENCY"] = "1"
    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="2plus1")
    repo = InMemoryJudgeRepository()
    judge_1 = "openai/gpt-oss-120b"
    judge_2 = "meta-llama/Llama-3.3-70B-Instruct"
    client = FakeJudgeClient({judge_1: 5, judge_2: 4})

    repo.persist_evaluation(
        _record_for_existing_score(answer_id=1, judge_model=config.primary_panel[0], stored_role="principal")
    )
    repo.persist_evaluation(
        _record_for_existing_score(answer_id=2, judge_model=config.primary_panel[1], stored_role="controle")
    )

    summary = JudgePipeline(repo, client).run([answer_with_id(1), answer_with_id(2)], config)

    assert summary.executed_evaluations == 2
    assert sorted(client.calls) == sorted([judge_1, judge_2])
    assert {(record.answer_id, record.stored_role) for record in repo.records} >= {
        (1, "controle"),
        (2, "principal"),
    }


def test_adaptive_2plus1_prioritizes_judge_1_then_judge_2_then_arbiter() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"
    env["JUDGE_ALWAYS_RUN_ARBITER"] = "true"
    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="2plus1", always_run_arbiter=True)
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient(
        {
            "openai/gpt-oss-120b": 5,
            "meta-llama/Llama-3.3-70B-Instruct": 3,
            "Unbabel/M-Prometheus-14B": 4,
        }
    )

    summary = JudgePipeline(repo, client).run([answer_with_id(1), answer_with_id(2)], config)

    assert summary.executed_evaluations == 6
    assert client.endpoint_keys.count("JUDGE") == 2
    assert client.endpoint_keys.count("SECONDARY_JUDGE") == 2
    assert client.endpoint_keys[-2:] == ["ARBITER", "ARBITER"]


def test_adaptive_2plus1_reports_batch_progress_after_each_evaluation() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"
    env["JUDGE_ALWAYS_RUN_ARBITER"] = "true"
    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="2plus1", always_run_arbiter=True)
    repo = InMemoryJudgeRepository()
    client = FakeJudgeClient(
        {
            "openai/gpt-oss-120b": 5,
            "meta-llama/Llama-3.3-70B-Instruct": 3,
            "Unbabel/M-Prometheus-14B": 4,
        }
    )
    progress_events: list[BatchProgress] = []

    JudgePipeline(repo, client, progress_callback=progress_events.append).run(
        [answer_with_id(1), answer_with_id(2)],
        config,
    )

    assert [event.executed_evaluations for event in progress_events] == [1, 2, 3, 4, 5, 6, 6]
    assert [event.arbiter_evaluations for event in progress_events] == [0, 0, 0, 0, 1, 2, 2]
    assert progress_events[-1] == BatchProgress(
        current=2,
        total=2,
        percent=100,
        executed_evaluations=6,
        skipped_evaluations=0,
        arbiter_evaluations=2,
    )


def test_adaptive_2plus1_starts_judge_2_before_all_judge_1_finish() -> None:
    env = dict(BASE_ENV)
    env["JUDGE_EXECUTION_STRATEGY"] = "adaptive"
    env["JUDGE_ADAPTIVE_INITIAL_CONCURRENCY"] = "1"
    env["JUDGE_ADAPTIVE_MAX_CONCURRENCY"] = "1"
    settings = load_settings(dotenv_path=None, env=env)
    config = resolve_runtime_config(settings, panel_mode="primary_only")
    repo = InMemoryJudgeRepository()
    judge_1 = "openai/gpt-oss-120b"
    judge_2 = "meta-llama/Llama-3.3-70B-Instruct"
    client = EventRecordingJudgeClient({judge_1: 5, judge_2: 4}, slow_model=judge_1)

    JudgePipeline(repo, client).run([answer_with_id(1), answer_with_id(2), answer_with_id(3)], config)

    secondary_start = client.events.index(("start", judge_2))
    judge_1_final_end = max(index for index, event in enumerate(client.events) if event == ("end", judge_1))
    assert secondary_start < judge_1_final_end
