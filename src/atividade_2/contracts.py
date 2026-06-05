"""Typed contracts for the AV2 judge pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PanelMode = Literal["single", "primary_only", "2plus1"]
JudgeProvider = Literal["remote_http"]
JudgeRole = Literal["single", "primary", "arbiter"]
StoredJudgeRole = Literal["principal", "controle", "arbitro"]
JudgeExecutionStrategy = Literal["sequential", "parallel", "adaptive"]
CandidateExecutionStrategy = Literal["sequential", "parallel"]
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
CandidateDatasetCode = Literal["J1", "J2"]
Av3Provider = Literal[
    "featherless",
    "openrouter",
    "xai",
    "ollama",
    "huggingface",
    "excluded",
    "unresolved",
]
ArtifactFormat = Literal["api", "hosted", "safetensors", "gguf", "excluded"]
MatchType = Literal[
    "same_model_same_runtime",
    "same_model_different_quantization",
    "same_model_api_reproduction",
    "same_family_newer_version_substitution",
    "same_family_version_needs_subtype_confirmation",
    "proprietary_api_resolved",
    "not_reproduced_provider_unavailable",
]
ValidationStatus = Literal[
    "confirmed_by_owner",
    "confirmed_from_av2_artifacts",
    "approved_version_substitution",
    "needs_owner_confirmation",
    "needs_owner_confirmation_gemini_subtype",
    "needs_provider_resolution",
    "needs_provider_model_id_resolution",
    "excluded_from_av3_run",
    "pending_team_confirmation",
]

DATASET_CODE_VALUES: tuple[CandidateDatasetCode, ...] = ("J1", "J2")
AV3_PROVIDER_VALUES: tuple[Av3Provider, ...] = (
    "featherless",
    "openrouter",
    "xai",
    "ollama",
    "huggingface",
    "excluded",
    "unresolved",
)
ARTIFACT_FORMAT_VALUES: tuple[ArtifactFormat, ...] = (
    "api",
    "hosted",
    "safetensors",
    "gguf",
    "excluded",
)
MATCH_TYPE_VALUES: tuple[MatchType, ...] = (
    "same_model_same_runtime",
    "same_model_different_quantization",
    "same_model_api_reproduction",
    "same_family_newer_version_substitution",
    "same_family_version_needs_subtype_confirmation",
    "proprietary_api_resolved",
    "not_reproduced_provider_unavailable",
)
VALIDATION_STATUS_VALUES: tuple[ValidationStatus, ...] = (
    "confirmed_by_owner",
    "confirmed_from_av2_artifacts",
    "approved_version_substitution",
    "needs_owner_confirmation",
    "needs_owner_confirmation_gemini_subtype",
    "needs_provider_resolution",
    "needs_provider_model_id_resolution",
    "excluded_from_av3_run",
    "pending_team_confirmation",
)

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
    featherless_url: str | None
    featherless_api_key: str | None
    openrouter_url: str | None
    openrouter_api_key: str | None
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
    remote_candidate_temperature: float
    remote_candidate_max_tokens: int | None
    remote_candidate_top_p: float
    remote_candidate_context_safety_margin_tokens: int
    remote_candidate_context_window_tokens: int | None
    remote_candidate_retry_on_context_window: bool
    candidate_execution_strategy: CandidateExecutionStrategy
    candidate_parallel_max_workers: int
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
class CandidateQuestionRecord:
    """Candidate-safe question record selected for one AV3 generation run."""

    question_id: int
    dataset: str
    dataset_name: str
    question_sequence: int
    question_text: str
    alternatives: Any = None


@dataclass(frozen=True)
class CandidateQuestionSelectionSummary:
    """Audit summary for one candidate question selection policy."""

    policy: str
    skip_existing_successful: bool
    selected: int
    failed_retry_candidates: int | None = None
    unanswered_candidates: int | None = None
    successful_excluded: int | None = None


@dataclass(frozen=True)
class CandidateQuestionSelectionResult:
    """Selected candidate-safe questions plus policy-level audit metadata."""

    questions: list[CandidateQuestionRecord]
    summary: CandidateQuestionSelectionSummary


@dataclass(frozen=True)
class CandidatePromptContext:
    """Candidate-safe input required to render one AV3 generation prompt."""

    question_id: int
    dataset_name: str
    question_text: str
    retrieved_chunks: list[RetrievedRagChunk]
    alternatives: Any = None
    retrieval_run_id: int | None = None
    retrieval_name: str | None = None
    top_k: int | None = None


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
class CandidateModelRuntimeProfileRecord:
    """Persisted effective runtime profile for one AV3 provider/model."""

    runtime_profile_id: int
    av3_provider: str
    provider_model_id: str
    provider_model_key: str
    context_window_tokens: int | None
    default_max_output_tokens: int | None
    safety_margin_tokens: int
    source: str
    confidence: str
    active: bool
    first_observed_at: str | None
    last_observed_at: str | None
    observation_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class CandidateModelRuntimeObservationRecord:
    """Historical provider/runtime observation persisted for auditability."""

    runtime_observation_id: int
    av3_provider: str
    provider_model_id: str
    provider_model_key: str
    observed_context_window_tokens: int | None
    observed_prompt_tokens: int | None
    observed_requested_max_tokens: int | None
    observed_total_tokens: int | None
    error_class: str
    error_message: str
    candidate_run_id: int | None
    candidate_answer_id: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    observed_at: str | None = None


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
class CandidateModelAssignmentRange:
    """Question-range scope for one centralized AV3 candidate-model assignment."""

    assignment_range_id: int | None
    assignment_id: int | None
    dataset_code: CandidateDatasetCode
    question_sequence_start: int
    question_sequence_end: int

    def __post_init__(self) -> None:
        if self.dataset_code not in DATASET_CODE_VALUES:
            raise ValueError(f"Unsupported dataset code: {self.dataset_code!r}")
        if self.question_sequence_start < 1:
            raise ValueError("question_sequence_start must be >= 1.")
        if self.question_sequence_end < self.question_sequence_start:
            raise ValueError("question_sequence_end must be >= question_sequence_start.")

    def covers(self, *, dataset: str, question_sequence: int) -> bool:
        return (
            dataset.upper() == self.dataset_code
            and self.question_sequence_start <= question_sequence <= self.question_sequence_end
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_range_id": self.assignment_range_id,
            "assignment_id": self.assignment_id,
            "dataset_code": self.dataset_code,
            "question_sequence_start": self.question_sequence_start,
            "question_sequence_end": self.question_sequence_end,
        }


@dataclass(frozen=True)
class CandidateModelAssignment:
    """AV3 execution mapping anchored to the original AV2 model identity."""

    assignment_id: int | None
    id_modelo_av2: int
    owner: str
    original_provider_model_id: str
    original_runtime: str
    av3_provider: Av3Provider
    artifact_format: ArtifactFormat
    match_type: MatchType
    validation_status: ValidationStatus
    av2_model_name: str | None = None
    av3_provider_model_id: str | None = None
    hf_model_id: str | None = None
    original_quantization: str | None = None
    av3_quantization: str | None = None
    notes: str | None = None
    active: bool = True
    ranges: tuple[CandidateModelAssignmentRange, ...] = field(default_factory=tuple)
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        if self.id_modelo_av2 < 1:
            raise ValueError("id_modelo_av2 must be >= 1.")
        if not self.owner.strip():
            raise ValueError("owner must not be empty.")
        if not self.original_provider_model_id.strip():
            raise ValueError("original_provider_model_id must not be empty.")
        if not self.original_runtime.strip():
            raise ValueError("original_runtime must not be empty.")
        if self.av3_provider not in AV3_PROVIDER_VALUES:
            raise ValueError(f"Unsupported AV3 provider: {self.av3_provider!r}")
        if self.artifact_format not in ARTIFACT_FORMAT_VALUES:
            raise ValueError(f"Unsupported artifact format: {self.artifact_format!r}")
        if self.match_type not in MATCH_TYPE_VALUES:
            raise ValueError(f"Unsupported match type: {self.match_type!r}")
        if self.validation_status not in VALIDATION_STATUS_VALUES:
            raise ValueError(f"Unsupported validation status: {self.validation_status!r}")
        object.__setattr__(self, "ranges", tuple(self.ranges))

    @property
    def warning_message(self) -> str | None:
        if self.validation_status == "needs_owner_confirmation_gemini_subtype":
            return "Pending owner confirmation: Gemini subtype still needs exact confirmation."
        if self.validation_status == "needs_owner_confirmation":
            return "Pending owner confirmation before default execution."
        if self.validation_status == "pending_team_confirmation":
            return "Pending team confirmation before default execution."
        return None

    def covers(self, *, dataset: str, question_sequence: int) -> bool:
        return any(
            assignment_range.covers(dataset=dataset, question_sequence=question_sequence)
            for assignment_range in self.ranges
        )

    def is_runnable(self, *, include_pending_confirmation: bool = False) -> bool:
        if not self.active:
            return False
        if self.av3_provider in {"excluded", "unresolved"}:
            return False
        if not self.av3_provider_model_id:
            return False
        if not self.ranges:
            return False
        if self.validation_status in {"needs_provider_resolution", "excluded_from_av3_run"}:
            return False
        if self.validation_status in {
            "needs_owner_confirmation",
            "needs_owner_confirmation_gemini_subtype",
            "pending_team_confirmation",
        }:
            return include_pending_confirmation
        return True

    def to_dict(self, *, include_pending_confirmation: bool = False) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "id_modelo_av2": self.id_modelo_av2,
            "av2_model_name": self.av2_model_name,
            "owner": self.owner,
            "original_provider_model_id": self.original_provider_model_id,
            "original_runtime": self.original_runtime,
            "av3_provider": self.av3_provider,
            "av3_provider_model_id": self.av3_provider_model_id,
            "hf_model_id": self.hf_model_id,
            "artifact_format": self.artifact_format,
            "original_quantization": self.original_quantization,
            "av3_quantization": self.av3_quantization,
            "match_type": self.match_type,
            "validation_status": self.validation_status,
            "notes": self.notes,
            "active": self.active,
            "ranges": [assignment_range.to_dict() for assignment_range in self.ranges],
            "runnable": self.is_runnable(include_pending_confirmation=include_pending_confirmation),
            "warning_message": self.warning_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


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
class CandidateRawResponse:
    """Raw remote candidate response plus provider metadata."""

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
