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
CandidateRunStatus = Literal["created", "running", "completed", "failed", "cancelled"]
CandidateAnswerStatus = Literal["created", "running", "success", "failed", "skipped"]
RagRetrievalStatus = Literal[
    "success",
    "question_not_found",
    "vector_base_not_found",
    "embedding_model_not_configured",
    "no_chunks_found",
]

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
    embedding_api_key: str | None
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
class RagCurationDatasetSummary:
    """Coverage summary for imported RAG curation per dataset."""

    dataset: str
    dataset_name: str
    total_questions: int
    curated_questions: int
    active_run_id: int | None
    active_filename: str | None
    active_imported_by: str | None
    active_imported_at: str | None
    active_item_count: int
    active_article_count: int
    vector_status: str = "nao_materializada"
    vector_retrieval_run_id: int | None = None
    vector_retrieval_name: str | None = None
    vector_document_count: int = 0
    vector_chunk_count: int = 0
    vector_embedding_count: int = 0


@dataclass(frozen=True)
class RagCurationImportRunRecord:
    """Metadata for a versioned RAG curation import run."""

    run_id: int
    dataset: str
    dataset_name: str
    filename: str
    payload_hash: str
    imported_by: str
    imported_at: str | None
    item_count: int
    article_count: int
    active: bool


@dataclass(frozen=True)
class RagCurationItemSummary:
    """Compact list item for a curated question."""

    curation_id: int
    run_id: int
    dataset: str
    question_id: int
    question_external_id: str
    question_sequence: int
    question_type: str
    discipline: str | None
    subject: str | None
    theme: str | None
    curator: str | None
    classified_at: str | None
    primary_norma: str | None
    article_count: int


@dataclass(frozen=True)
class RagCurationItemDetail:
    """Full detail for one curated question."""

    curation_id: int
    run_id: int
    dataset: str
    question_id: int
    question_external_id: str
    question_sequence: int
    question_type: str
    prompt_system: str | None
    question_text: str
    answer_key: Any
    perguntas: Any
    alternativas: Any
    total_points: float | None
    difficulty_level: str | None
    difficulty_scale: int | None
    difficulty_criteria: Any
    discipline: str | None
    subject: str | None
    theme: str | None
    norma: str | None
    lei: str | None
    url: str | None
    urn: str | None
    curator: str | None
    classified_at: str | None
    metadata: dict[str, Any]
    raw_payload: dict[str, Any]
    articles: list[dict[str, Any]]


@dataclass(frozen=True)
class RagBaseMaterializationSummary:
    """Summary of one materialized AV3 RAG base build."""

    dataset: str
    dataset_name: str
    import_run_id: int
    retrieval_run_id: int
    retrieval_name: str
    chunking_strategy: str
    top_k: int
    document_count: int
    chunk_count: int
    embedding_count: int
    vector_extension_enabled: bool
    created_at: str | None


@dataclass(frozen=True)
class RagVectorBaseSummary:
    """Current AV3 vector-base status for one dataset."""

    dataset: str
    dataset_name: str
    import_run_id: int
    active_curation_run_id: int | None
    matches_active_curation: bool
    retrieval_run_id: int
    retrieval_name: str
    retrieval_strategy: str
    embedding_model: str | None
    top_k: int
    vector_enabled: bool
    lexical_enabled: bool
    rerank_enabled: bool
    document_count: int
    chunk_count: int
    embedding_count: int
    status: str
    created_at: str | None


@dataclass(frozen=True)
class RagVectorRunRecord:
    """One persisted retrieval/vector run available for a dataset."""

    run_id: int
    dataset: str
    import_run_id: int
    retrieval_name: str
    retrieval_strategy: str
    embedding_model: str | None
    top_k: int
    active: bool
    document_count: int
    chunk_count: int
    embedding_count: int
    created_at: str | None


@dataclass(frozen=True)
class RagEmbeddingModelConfigRecord:
    """Persisted embedding-model configuration for one dataset."""

    config_id: int
    dataset: str
    dataset_name: str
    provider: str
    model_name: str
    dimensions: int | None
    api_base_url: str | None
    notes: str | None
    updated_by: str
    updated_at: str | None


@dataclass(frozen=True)
class RagEmbeddingGenerationSummary:
    """Summary of one embedding-generation execution for a dataset."""

    dataset: str
    dataset_name: str
    retrieval_run_id: int
    retrieval_name: str
    import_run_id: int
    embedding_model: str
    provider: str
    api_base_url: str | None
    requested_dimensions: int | None
    generated_embeddings: int
    total_chunks: int
    latency_ms: int
    created_at: str | None


@dataclass(frozen=True)
class CandidatePromptRecord:
    """Persisted AV3 candidate prompt configuration."""

    prompt_id: int | None
    dataset: str
    version: int
    persona: str
    context: str
    rag_instruction: str
    output: str
    active: bool = False
    created_by: str = "system"
    created_at: str | None = None


@dataclass(frozen=True)
class CandidateRunRecord:
    """Persisted AV3 candidate generation run metadata."""

    candidate_run_id: int | None
    dataset: str
    retrieval_run_id: int
    prompt_id: int
    model_name: str
    provider: str
    batch_size: int
    run_status: CandidateRunStatus = "created"
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    created_by: str = "system"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


@dataclass(frozen=True)
class CandidateAnswerRecord:
    """Persisted AV3 candidate answer for one question inside a run."""

    candidate_answer_id: int | None
    candidate_run_id: int
    question_id: int
    model_name: str
    rendered_prompt: str
    status: CandidateAnswerStatus = "created"
    answer_text: str | None = None
    final_choice: str | None = None
    error_message: str | None = None
    latency_ms: int | None = None
    raw_response: dict[str, Any] | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class CandidateAnswerContextChunkRecord:
    """Persisted snapshot of one retrieved chunk used for a candidate answer."""

    answer_context_chunk_id: int | None
    candidate_answer_id: int
    chunk_id: int
    rank: int
    chunk_text_snapshot: str
    similarity_score: float | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


@dataclass(frozen=True)
class RagRetrievalQuestion:
    """Candidate-safe question payload used to retrieve AV3 RAG context."""

    question_id: int
    dataset: str
    question_text: str


@dataclass(frozen=True)
class RetrievedRagChunk:
    """One retrieved AV3 RAG chunk safe to expose to candidate prompting."""

    rank: int
    chunk_id: int
    chunk_text: str
    source_kind: str | None
    document_id: int | None
    document_key: str | None
    lei: str | None
    norma: str | None
    url: str | None
    urn: str | None
    artigo: str | None
    topico: str | None
    relevancia: str | None
    tipo: str | None
    distance: float | None
    similarity: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RagRetrievalResult:
    """Structured retrieval result for one AV3 question."""

    question_id: int
    dataset: str
    retrieval_run_id: int | None
    retrieval_name: str | None
    embedding_model: str | None
    top_k: int
    status: RagRetrievalStatus
    chunks: list[RetrievedRagChunk] = field(default_factory=list)


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
    raw_output_jsonb: dict[str, Any] | None = None


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
    parsed_evaluation: ParsedJudgeEvaluation | None = None


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
