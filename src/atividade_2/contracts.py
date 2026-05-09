"""Typed contracts for the AV2 judge pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PanelMode = Literal["single", "primary_only", "2plus1"]
JudgeProvider = Literal["remote_http"]
JudgeRole = Literal["single", "primary", "arbiter"]
StoredJudgeRole = Literal["principal", "controle", "arbitro"]
JudgeExecutionStrategy = Literal["sequential", "parallel", "adaptive"]
AppEnvironment = Literal["dev", "test", "prod"]

PROMPT_VERSION = "av2-judge-v3"
RUBRIC_VERSION = "av2-legal-rubric-v2"


@dataclass(frozen=True)
class ModelSpec:
    """Resolved judge model identity."""

    requested: str
    provider_model: str


@dataclass(frozen=True)
class RemoteJudgeEndpoint:
    """Per-judge remote endpoint override."""

    base_url: str
    api_key: str


@dataclass(frozen=True)
class JudgeSettings:
    """Settings loaded from ``.env`` and process environment."""

    app_env: AppEnvironment
    database_url: str
    backup_root_file: str
    judge_provider: JudgeProvider
    remote_judge_base_url: str | None
    remote_judge_api_key: str | None
    remote_judge_endpoints: dict[str, RemoteJudgeEndpoint]
    judge_panel_mode: PanelMode
    remote_judge_default_model: str | None
    remote_secondary_judge_model: str | None
    remote_arbiter_judge_model: str | None
    judge_arbitration_min_delta: int
    judge_always_run_arbiter: bool
    remote_judge_timeout_seconds: int
    remote_judge_temperature: float
    remote_judge_max_tokens: int
    remote_judge_top_p: float
    remote_judge_openai_compatible: bool
    judge_save_raw_response: bool
    judge_execution_strategy: JudgeExecutionStrategy
    judge_batch_size: int
    judge_adaptive_initial_concurrency: int
    judge_adaptive_max_concurrency: int
    judge_adaptive_success_threshold: int
    judge_adaptive_max_retries: int
    judge_adaptive_base_backoff_seconds: float
    judge_adaptive_max_backoff_seconds: float


@dataclass(frozen=True)
class RuntimeJudgeConfig:
    """Effective judge execution config after CLI overrides are applied."""

    provider: JudgeProvider
    panel_mode: PanelMode
    single_judge: ModelSpec | None
    primary_panel: tuple[ModelSpec, ...]
    arbiter: ModelSpec | None
    arbitration_min_delta: int
    always_run_arbiter: bool
    execution_strategy: JudgeExecutionStrategy
    settings: JudgeSettings
    model_source: str


@dataclass(frozen=True)
class CandidateAnswerContext:
    """Question, reference, and AV1 answer loaded from PostgreSQL."""

    answer_id: int
    question_id: int
    dataset_name: str
    question_text: str
    reference_answer: str
    candidate_answer: str
    candidate_model: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgePromptTemplate:
    """Custom prompt fields resolved per dataset."""

    prompt_id: int | None
    dataset_name: str
    version: int | None
    created_by: str | None
    prompt_text: str
    persona: str
    context_text: str
    rubric_text: str
    output_text: str


@dataclass(frozen=True)
class JudgePromptConfigRecord:
    """Persisted prompt configuration record exposed to the UI service."""

    prompt_id: int
    dataset: str
    version: int
    created_by: str | None
    active: bool
    prompt: str
    persona: str
    context: str
    rubric: str
    output: str
    created_at: str | None


@dataclass(frozen=True)
class MetaEvaluationRecord:
    """Persisted human meta-evaluation for a judge evaluation."""

    meta_evaluation_id: int
    evaluation_id: int
    evaluator_name: str
    score: int
    rationale: str
    created_at: str | None


@dataclass(frozen=True)
class MetaEvaluationHistoryRecord:
    """Persisted human meta-evaluation enriched with judge context."""

    meta_evaluation_id: int
    evaluation_id: int
    evaluator_name: str
    score: int
    rationale: str
    created_at: str | None
    dataset: str
    question_id: int
    answer_id: int
    candidate_model: str
    judge_model: str
    judge_score: int
    judge_rationale: str
    judge_chain_of_thought: str
    question_text: str
    reference_answer: str
    candidate_answer: str
    evaluated_at: str | None


@dataclass(frozen=True)
class MetaEvaluationSubject:
    """Judge evaluation context shown to the human meta-evaluator."""

    evaluation_id: int
    dataset: str
    question_id: int
    answer_id: int
    candidate_model: str
    judge_model: str
    judge_score: int
    judge_rationale: str
    judge_chain_of_thought: str
    question_text: str
    reference_answer: str
    candidate_answer: str
    evaluated_at: str | None
    prompt_version: int | None
    prompt_created_by: str | None


@dataclass(frozen=True)
class JudgeRawResponse:
    """Raw remote judge response plus provider metadata."""

    text: str
    provider: str
    model: str
    latency_ms: int
    status_code: int | None = None
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True)
class ParsedJudgeEvaluation:
    """Structured, validated judge evaluation."""

    score: int
    rationale: str
    legal_accuracy: str | None = None
    hallucination_risk: str | None = None
    rubric_alignment: str | None = None
    requires_human_review: bool = False
    criteria: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRecord:
    """Evaluation row prepared for persistence."""

    answer_id: int
    judge_model: ModelSpec
    prompt_id: int | None
    stored_role: StoredJudgeRole
    panel_mode: PanelMode
    trigger_reason: str
    score: int
    rationale: str
    latency_ms: int
    raw_response: JudgeRawResponse | None = None


@dataclass(frozen=True)
class PipelineSummary:
    """Concise run result for CLI reporting."""

    selected_answers: int
    executed_evaluations: int
    skipped_evaluations: int
    arbiter_evaluations: int


@dataclass(frozen=True)
class EligibilitySummary:
    """Answer-level eligibility counts before batch execution."""

    missing: int
    failed: int
    successful: int
    batch_size: int
    will_process: int


@dataclass(frozen=True)
class BatchProgress:
    """Structured batch progress for terminal, audit logs, and web UIs."""

    current: int
    total: int
    percent: int
    executed_evaluations: int
    skipped_evaluations: int
    arbiter_evaluations: int


@dataclass(frozen=True)
class EvaluationProgress:
    """Per-evaluation progress row for audit-oriented Web UI execution tables."""

    status: Literal["pending", "running", "success", "failed", "skipped"]
    dataset: str
    question_id: int
    answer_id: int
    candidate_model: str
    judge_model: str
    role: StoredJudgeRole
    panel_mode: PanelMode
    score: int | None = None
    delta: int | None = None
    arbiter_triggered: bool | None = None
    trigger_reason: str | None = None
    latency_ms: int | None = None
    error: str | None = None
    prompt: str | None = None
    raw_response: str | None = None
    rationale: str | None = None
