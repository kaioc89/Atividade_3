from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from atividade_2.contracts import (
    CandidateAnswerRecord,
    CandidateModelAssignment,
    CandidateModelAssignmentRange,
    CandidateModelRuntimeProfileRecord,
    CandidatePromptRecord,
    CandidateQuestionRecord,
    CandidateRawResponse,
    CandidateRunRecord,
    RagRetrievalResult,
    RagVectorBaseSummary,
    RetrievedRagChunk,
)
from atividade_2.candidate_runtime_learning import parse_candidate_runtime_observation
from atividade_2.run_candidates_rag_service import (
    RunCandidatesRagRequest,
    RunCandidatesRagService,
    _default_client_factory,
    _format_candidate_runtime_config,
    _resolve_candidate_provider_config,
    _resolve_candidate_runtime_config,
    _with_remote_candidate_config,
    resolve_candidate_max_tokens,
)
from atividade_2.repositories import _default_candidate_model_assignments


@dataclass
class FakeSettings:
    database_url: str = "postgresql://example.invalid/app"
    embedding_api_key: str | None = "embedding-test-key"
    featherless_url: str | None = "https://api.featherless.ai/v1"
    featherless_api_key: str | None = "featherless-test-key"
    openrouter_url: str | None = "https://openrouter.ai/api/v1"
    openrouter_api_key: str | None = "openrouter-test-key"
    remote_candidate_temperature: float = 0.2
    remote_candidate_max_tokens: int | None = 1024
    remote_candidate_top_p: float = 0.9
    remote_candidate_context_safety_margin_tokens: int = 512
    remote_candidate_context_window_tokens: int | None = None
    remote_candidate_retry_on_context_window: bool = False


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeRepository:
    def __init__(self) -> None:
        self.ensure_schema_calls = 0
        self.select_calls: list[dict[str, object]] = []
        self.success_exists_calls: list[tuple[str, str, int, int | None]] = []
        self.created_runs: list[CandidateRunRecord] = []
        self.updated_runs: list[dict[str, object]] = []
        self.persisted_answers: list[CandidateAnswerRecord] = []
        self.runtime_profiles: dict[tuple[str, str], CandidateModelRuntimeProfileRecord] = {}
        self.runtime_observations: list[dict[str, object]] = []
        self.next_candidate_run_id = 501
        self.next_candidate_answer_id = 801
        self.next_runtime_profile_id = 901
        self.vector_base = RagVectorBaseSummary(
            dataset="J1",
            dataset_name="OAB_Bench",
            import_run_id=31,
            active_curation_run_id=31,
            matches_active_curation=True,
            retrieval_run_id=21,
            retrieval_name="j1_source_urls_v1",
            retrieval_strategy="source_url_only_v1",
            embedding_model="text-embedding-3-small",
            top_k=5,
            vector_enabled=True,
            lexical_enabled=False,
            rerank_enabled=False,
            document_count=10,
            chunk_count=50,
            embedding_count=50,
            status="pronta_com_embeddings",
            created_at="2026-06-04T12:00:00",
        )
        self.prompt = CandidatePromptRecord(
            prompt_id=7,
            dataset="J1",
            version=1,
            persona="Persona dataset {dataset}",
            context="Questao:\n{pergunta_oab}",
            rag_instruction="Contexto:\n{contexto_rag}",
            output="Saida final",
            active=True,
        )
        self.questions_by_dataset: dict[str, list[CandidateQuestionRecord]] = {
            "J1": [
                CandidateQuestionRecord(
                    question_id=101,
                    dataset="J1",
                    dataset_name="OAB_Bench",
                    question_sequence=1,
                    question_text="Questao J1 101.",
                    alternatives=None,
                )
            ],
            "J2": [
                CandidateQuestionRecord(
                    question_id=202,
                    dataset="J2",
                    dataset_name="OAB_Exames",
                    question_sequence=2,
                    question_text="Qual alternativa correta?",
                    alternatives={"A": "Opcao A", "B": "Opcao B", "C": "Opcao C"},
                )
            ],
        }
        self.existing_successes: set[tuple[str, str, int]] = set()
        self.assignments = _default_candidate_model_assignments()
        self.assignments += (
            CandidateModelAssignment(
                assignment_id=None,
                id_modelo_av2=1001,
                owner="Tests",
                original_provider_model_id="candidate-j1",
                original_runtime="tests",
                av3_provider="featherless",
                artifact_format="api",
                match_type="same_model_api_reproduction",
                validation_status="confirmed_by_owner",
                av3_provider_model_id="candidate-j1",
                hf_model_id="candidate-j1",
                original_quantization="provider_default",
                av3_quantization="provider_default",
                ranges=(
                    CandidateModelAssignmentRange(None, None, "J1", 1, 2000),
                ),
            ),
            CandidateModelAssignment(
                assignment_id=None,
                id_modelo_av2=1002,
                owner="Tests",
                original_provider_model_id="candidate-j2",
                original_runtime="tests",
                av3_provider="featherless",
                artifact_format="api",
                match_type="same_model_api_reproduction",
                validation_status="confirmed_by_owner",
                av3_provider_model_id="candidate-j2",
                hf_model_id="candidate-j2",
                original_quantization="provider_default",
                av3_quantization="provider_default",
                ranges=(
                    CandidateModelAssignmentRange(None, None, "J2", 1, 2000),
                ),
            ),
            CandidateModelAssignment(
                assignment_id=None,
                id_modelo_av2=1003,
                owner="Tests",
                original_provider_model_id="openai/gpt-5.4",
                original_runtime="tests",
                av3_provider="openrouter",
                artifact_format="api",
                match_type="same_model_api_reproduction",
                validation_status="confirmed_by_owner",
                av3_provider_model_id="openai/gpt-5.4",
                hf_model_id="openai/gpt-5.4",
                original_quantization="provider_default",
                av3_quantization="provider_default",
                ranges=(
                    CandidateModelAssignmentRange(None, None, "J1", 1, 2000),
                    CandidateModelAssignmentRange(None, None, "J2", 1, 2000),
                ),
            ),
            CandidateModelAssignment(
                assignment_id=None,
                id_modelo_av2=1004,
                owner="Tests",
                original_provider_model_id="google/gemma-2-2b-it",
                original_runtime="tests",
                av3_provider="featherless",
                artifact_format="api",
                match_type="same_model_api_reproduction",
                validation_status="confirmed_by_owner",
                av3_provider_model_id="google/gemma-2-2b-it",
                hf_model_id="google/gemma-2-2b-it",
                original_quantization="provider_default",
                av3_quantization="provider_default",
                ranges=(
                    CandidateModelAssignmentRange(None, None, "J1", 1, 2000),
                ),
            ),
        )

    def ensure_schema(self) -> None:
        self.ensure_schema_calls += 1

    def list_candidate_model_assignments(self):
        return self.assignments

    def get_rag_vector_base_summary(self, *, dataset: str) -> RagVectorBaseSummary | None:
        if self.vector_base is None:
            return None
        if self.vector_base.dataset != dataset:
            return RagVectorBaseSummary(
                dataset=dataset,
                dataset_name="OAB_Exames" if dataset == "J2" else "OAB_Bench",
                import_run_id=self.vector_base.import_run_id,
                active_curation_run_id=self.vector_base.active_curation_run_id,
                matches_active_curation=self.vector_base.matches_active_curation,
                retrieval_run_id=self.vector_base.retrieval_run_id,
                retrieval_name=f"{dataset.lower()}_source_urls_v1",
                retrieval_strategy=self.vector_base.retrieval_strategy,
                embedding_model=self.vector_base.embedding_model,
                top_k=self.vector_base.top_k,
                vector_enabled=self.vector_base.vector_enabled,
                lexical_enabled=self.vector_base.lexical_enabled,
                rerank_enabled=self.vector_base.rerank_enabled,
                document_count=self.vector_base.document_count,
                chunk_count=self.vector_base.chunk_count,
                embedding_count=self.vector_base.embedding_count,
                status=self.vector_base.status,
                created_at=self.vector_base.created_at,
            )
        return self.vector_base

    def get_or_create_candidate_prompt(
        self,
        *,
        dataset: str,
        prompt_id: int | None = None,
    ) -> CandidatePromptRecord:
        if prompt_id is not None:
            return CandidatePromptRecord(
                prompt_id=prompt_id,
                dataset=dataset,
                version=2,
                persona=self.prompt.persona,
                context=self.prompt.context,
                rag_instruction=self.prompt.rag_instruction,
                output=self.prompt.output,
                active=True,
            )
        if dataset != self.prompt.dataset:
            return CandidatePromptRecord(
                prompt_id=11,
                dataset=dataset,
                version=1,
                persona=self.prompt.persona,
                context=self.prompt.context,
                rag_instruction=self.prompt.rag_instruction,
                output=self.prompt.output,
                active=True,
            )
        return self.prompt

    def create_candidate_run(self, *, run: CandidateRunRecord) -> CandidateRunRecord:
        stored = CandidateRunRecord(
            candidate_run_id=self.next_candidate_run_id,
            dataset=run.dataset,
            retrieval_run_id=run.retrieval_run_id,
            prompt_id=run.prompt_id,
            model_name=run.model_name,
            provider=run.provider,
            batch_size=run.batch_size,
            run_status=run.run_status,
            temperature=run.temperature,
            max_tokens=run.max_tokens,
            top_p=run.top_p,
            started_at=run.started_at,
            finished_at=run.finished_at,
            created_by=run.created_by,
            metadata=dict(run.metadata),
            created_at="2026-06-04T13:00:00",
        )
        self.created_runs.append(stored)
        return stored

    def update_candidate_run_status(
        self,
        *,
        candidate_run_id: int,
        run_status: str,
        finished_at: str | None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.updated_runs.append(
            {
                "candidate_run_id": candidate_run_id,
                "run_status": run_status,
                "finished_at": finished_at,
                "metadata": dict(metadata or {}),
            }
        )

    def select_candidate_questions(
        self,
        *,
        dataset: str,
        batch_size: int,
        question_sequence_start: int | None,
        question_sequence_end: int | None,
        question_id: int | None,
    ) -> list[CandidateQuestionRecord]:
        self.select_calls.append(
            {
                "dataset": dataset,
                "batch_size": batch_size,
                "question_sequence_start": question_sequence_start,
                "question_sequence_end": question_sequence_end,
                "question_id": question_id,
            }
        )
        questions = list(self.questions_by_dataset.get(dataset, []))
        if question_id is not None:
            questions = [question for question in questions if question.question_id == question_id]
        return questions[:batch_size]

    def successful_candidate_answer_exists(
        self,
        *,
        dataset: str,
        model_name: str,
        question_id: int,
        exclude_candidate_run_id: int | None = None,
    ) -> bool:
        self.success_exists_calls.append((dataset, model_name, question_id, exclude_candidate_run_id))
        return (dataset, model_name, question_id) in self.existing_successes

    def persist_candidate_answer(self, *, answer: CandidateAnswerRecord) -> CandidateAnswerRecord:
        stored = CandidateAnswerRecord(
            candidate_answer_id=self.next_candidate_answer_id,
            candidate_run_id=answer.candidate_run_id,
            question_id=answer.question_id,
            model_name=answer.model_name,
            rendered_prompt=answer.rendered_prompt,
            status=answer.status,
            answer_text=answer.answer_text,
            final_choice=answer.final_choice,
            error_message=answer.error_message,
            latency_ms=answer.latency_ms,
            raw_response=answer.raw_response,
            created_at="2026-06-04T13:05:00",
        )
        self.persisted_answers.append(stored)
        self.next_candidate_answer_id += 1
        return stored

    def get_candidate_model_runtime_profile(
        self,
        *,
        av3_provider: str,
        provider_model_id: str,
    ) -> CandidateModelRuntimeProfileRecord | None:
        return self.runtime_profiles.get((av3_provider, provider_model_id.strip().casefold()))

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
        metadata: dict[str, object] | None = None,
    ) -> CandidateModelRuntimeProfileRecord:
        key = (av3_provider, provider_model_id.strip().casefold())
        existing = self.runtime_profiles.get(key)
        if existing is not None and existing.context_window_tokens is not None and context_window_tokens is not None:
            context_window_tokens = min(existing.context_window_tokens, context_window_tokens)
        observation_count = 1 if context_window_tokens is not None else 0
        if existing is not None:
            observation_count = existing.observation_count + (1 if context_window_tokens is not None else 0)
        record = CandidateModelRuntimeProfileRecord(
            runtime_profile_id=existing.runtime_profile_id if existing is not None else self.next_runtime_profile_id,
            av3_provider=av3_provider,
            provider_model_id=provider_model_id,
            provider_model_key=provider_model_id.strip().casefold(),
            context_window_tokens=context_window_tokens if context_window_tokens is not None else (existing.context_window_tokens if existing else None),
            default_max_output_tokens=(
                default_max_output_tokens
                if default_max_output_tokens is not None
                else (existing.default_max_output_tokens if existing else None)
            ),
            safety_margin_tokens=safety_margin_tokens,
            source=source,
            confidence=confidence,
            active=True,
            first_observed_at="2026-06-05T10:00:00" if context_window_tokens is not None else None,
            last_observed_at="2026-06-05T10:00:00" if context_window_tokens is not None else None,
            observation_count=observation_count,
            metadata=dict(metadata or {}),
            created_at="2026-06-05T10:00:00",
            updated_at="2026-06-05T10:00:00",
        )
        self.runtime_profiles[key] = record
        if existing is None:
            self.next_runtime_profile_id += 1
        return record

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
        metadata: dict[str, object] | None = None,
    ):
        self.runtime_observations.append(
            {
                "av3_provider": av3_provider,
                "provider_model_id": provider_model_id,
                "observed_context_window_tokens": observed_context_window_tokens,
                "observed_prompt_tokens": observed_prompt_tokens,
                "observed_requested_max_tokens": observed_requested_max_tokens,
                "observed_total_tokens": observed_total_tokens,
                "error_class": error_class,
                "error_message": error_message,
                "candidate_run_id": candidate_run_id,
                "candidate_answer_id": candidate_answer_id,
                "metadata": dict(metadata or {}),
            }
        )
        return self.runtime_observations[-1]


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int | None]] = []
        self.results: dict[tuple[str, int], RagRetrievalResult] = {}

    def retrieve_for_question(
        self,
        *,
        question_id: int,
        dataset: str,
        top_k: int | None = None,
    ) -> RagRetrievalResult:
        self.calls.append((question_id, dataset, top_k))
        return self.results[(dataset, question_id)]


@dataclass
class FakeClient:
    responses: dict[int, CandidateRawResponse] = field(default_factory=dict)
    failures: dict[int, Exception] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)
    _call_index: int = 0

    def generate(self, prompt: str, *, model: str) -> CandidateRawResponse:
        self.calls.append((prompt, model))
        call_index = self._call_index
        self._call_index += 1
        if call_index in self.failures:
            raise self.failures[call_index]
        return self.responses.get(
            call_index,
            CandidateRawResponse(
                text="Resposta final:\nTexto padrao.",
                provider="fake",
                model=model,
                latency_ms=17,
            ),
        )


class FakeSnapshotService:
    def __init__(self) -> None:
        self.calls: list[tuple[int, RagRetrievalResult]] = []

    def persist_retrieval_snapshot(
        self,
        *,
        candidate_answer_id: int,
        retrieval_result: RagRetrievalResult,
    ) -> list[object]:
        self.calls.append((candidate_answer_id, retrieval_result))
        return []


def _service(
    *,
    repository: FakeRepository | None = None,
    retriever: FakeRetriever | None = None,
    client: FakeClient | None = None,
    client_factory=None,
    snapshot_service: FakeSnapshotService | None = None,
    connect_func=None,
) -> RunCandidatesRagService:
    repository = repository or FakeRepository()
    retriever = retriever or FakeRetriever()
    client = client or FakeClient()
    snapshot_service = snapshot_service or FakeSnapshotService()
    connection = FakeConnection()

    if connect_func is None:
        def connect_func(_: str) -> FakeConnection:
            return connection

    return RunCandidatesRagService(
        settings_loader=FakeSettings,
        connect_func=connect_func,
        repository_factory=lambda _: repository,
        retriever_factory=lambda _repository, _settings, _dataset: retriever,
        client_factory=client_factory or (lambda _request, _settings: client),
        snapshot_service_factory=lambda _repository: snapshot_service,
    )


def _success_result(*, dataset: str, question_id: int, chunk_text: str = "Trecho seguro.") -> RagRetrievalResult:
    return RagRetrievalResult(
        question_id=question_id,
        dataset=dataset,
        retrieval_run_id=21,
        retrieval_name=f"{dataset.lower()}_source_urls_v1",
        embedding_model="text-embedding-3-small",
        top_k=5,
        status="success",
        chunks=[
            RetrievedRagChunk(
                rank=1,
                chunk_id=501,
                chunk_text=chunk_text,
                source_kind="lei",
                document_id=701,
                document_key="doc-701",
                lei="Lei X",
                norma="Norma X",
                url="https://example.test/doc-701",
                urn=None,
                artigo="Art. 5",
                topico="Tema X",
                relevancia="alta",
                tipo="lei",
                distance=0.12,
                similarity=0.88,
                metadata={
                    "guideline": "nao mostrar",
                    "official_answer_key": "B",
                    "judge_prompt": "nao mostrar",
                },
            )
        ],
    )


def test_dry_run_resolves_configuration_without_db_writes_or_client_calls(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    client = FakeClient()
    repository.questions_by_dataset["J1"] = [
        CandidateQuestionRecord(101, "J1", "OAB_Bench", 71, "Questao J1 101.", None)
    ]
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    service = _service(repository=repository, retriever=retriever, client=client)

    result = service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="google/gemma-2-2b-it",
            provider="remote_http",
            batch_size=2,
            question_sequence_start=71,
            question_sequence_end=71,
            dry_run=True,
            audit_log=str(tmp_path / "candidate-dry-run.log"),
            no_audit_animation=True,
        )
    )

    assert result.dry_run is True
    assert result.candidate_run_id is None
    assert result.summary is None
    assert result.runtime_config_summary is not None
    assert "av3 provider: featherless" in result.runtime_config_summary
    assert "api_key: <not required in dry-run>" in result.runtime_config_summary
    assert "final_max_tokens: 1024" in result.runtime_config_summary
    assert "context_window_tokens: 8192" in result.runtime_config_summary
    assert repository.ensure_schema_calls == 1
    assert repository.created_runs == []
    assert retriever.calls == [(101, "J1", None)]
    assert client.calls == []


def test_dry_run_does_not_require_openrouter_or_featherless_keys(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    client = FakeClient()
    repository.questions_by_dataset["J1"] = [
        CandidateQuestionRecord(101, "J1", "OAB_Bench", 119, "Questao J1 101.", None)
    ]
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)

    @dataclass
    class MissingProviderKeysSettings:
        database_url: str = "postgresql://example.invalid/app"
        embedding_api_key: str | None = "embedding-test-key"
        featherless_url: str | None = "https://api.featherless.ai/v1"
        featherless_api_key: str | None = None
        openrouter_url: str | None = "https://openrouter.ai/api/v1"
        openrouter_api_key: str | None = None

    service = RunCandidatesRagService(
        settings_loader=MissingProviderKeysSettings,
        connect_func=lambda _: FakeConnection(),
        repository_factory=lambda _: repository,
        retriever_factory=lambda _repository, _settings, _dataset: retriever,
        client_factory=lambda _request, _settings: client,
        snapshot_service_factory=lambda _repository: FakeSnapshotService(),
    )

    result = service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="x-ai/grok-4.3",
            provider="remote_http",
            batch_size=1,
            question_sequence_start=119,
            question_sequence_end=119,
            dry_run=True,
            audit_log=str(tmp_path / "candidate-dry-run-missing-keys.log"),
            no_audit_animation=True,
        )
    )

    assert result.dry_run is True
    assert result.runtime_config_summary is not None
    assert "av3 provider: openrouter" in result.runtime_config_summary
    assert "api_key: <not required in dry-run>" in result.runtime_config_summary
    assert retriever.calls == [(101, "J1", None)]
    assert client.calls == []


def test_real_run_creates_a_candidate_run(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    service = _service(repository=repository, retriever=retriever)

    result = service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="openai/gpt-5.4",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "candidate-run.log"),
            no_audit_animation=True,
        )
    )

    assert result.candidate_run_id == 501
    assert repository.created_runs[0].dataset == "J1"
    assert repository.created_runs[0].retrieval_run_id == 21
    assert repository.created_runs[0].prompt_id == 7
    assert repository.created_runs[0].metadata["candidate_runtime_profile"]["av3_provider"] == "openrouter"
    assert repository.updated_runs[-1]["run_status"] == "completed"


def test_real_openrouter_candidate_execution_uses_openrouter_env_config(tmp_path) -> None:
    repository = FakeRepository()
    service = _service(repository=repository)
    request = RunCandidatesRagRequest(
        dataset="J1",
        model_name="x-ai/grok-4.3",
        provider="remote_http",
        batch_size=1,
        question_sequence_start=119,
        question_sequence_end=119,
        audit_log=str(tmp_path / "openrouter.log"),
        no_audit_animation=True,
    )
    provider_config = _resolve_candidate_provider_config(
        settings=FakeSettings(),
        assignment=next(
            assignment
            for assignment in repository.assignments
            if assignment.av3_provider_model_id == "x-ai/grok-4.3"
        ),
        require_api_key=True,
    )
    runtime_config = _resolve_candidate_runtime_config(
        repository=repository,
        settings=FakeSettings(),
        request=request,
        resolved=service.resolve(request),
        questions=[CandidateQuestionRecord(101, "J1", "OAB_Bench", 119, "Q", None)],
        require_api_key=True,
    )
    configured_request = _with_remote_candidate_config(request, runtime_config)

    assert configured_request.remote_candidate_base_url == "https://openrouter.ai/api/v1"
    assert configured_request.remote_candidate_api_key == "openrouter-test-key"
    assert configured_request.remote_candidate_temperature == 0.2
    assert configured_request.remote_candidate_top_p == 0.9
    assert configured_request.remote_candidate_max_tokens == 1024


def test_real_featherless_candidate_execution_uses_featherless_env_config(tmp_path) -> None:
    repository = FakeRepository()
    service = _service(repository=repository)
    request = RunCandidatesRagRequest(
        dataset="J1",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        provider="remote_http",
        batch_size=1,
        question_sequence_start=95,
        question_sequence_end=95,
        audit_log=str(tmp_path / "featherless.log"),
        no_audit_animation=True,
    )
    provider_config = _resolve_candidate_provider_config(
        settings=FakeSettings(),
        assignment=next(
            assignment
            for assignment in repository.assignments
            if assignment.av3_provider_model_id == "Qwen/Qwen2.5-7B-Instruct"
        ),
        require_api_key=True,
    )
    runtime_config = _resolve_candidate_runtime_config(
        repository=repository,
        settings=FakeSettings(),
        request=request,
        resolved=service.resolve(request),
        questions=[CandidateQuestionRecord(101, "J1", "OAB_Bench", 95, "Q", None)],
        require_api_key=True,
    )
    configured_request = _with_remote_candidate_config(request, runtime_config)

    assert configured_request.remote_candidate_base_url == "https://api.featherless.ai/v1"
    assert configured_request.remote_candidate_api_key == "featherless-test-key"
    assert configured_request.remote_candidate_temperature == 0.2
    assert configured_request.remote_candidate_top_p == 0.9
    assert configured_request.remote_candidate_max_tokens == 1024


def test_missing_openrouter_key_produces_clear_error(tmp_path) -> None:
    repository = FakeRepository()
    service = _service(repository=repository)

    @dataclass
    class MissingOpenRouterKeySettings(FakeSettings):
        openrouter_api_key: str | None = None

    with pytest.raises(ValueError, match="OPENROUTER_KEY is required for openrouter candidate execution"):
        _resolve_candidate_provider_config(
            settings=MissingOpenRouterKeySettings(),
            assignment=next(
                assignment
                for assignment in repository.assignments
                if assignment.av3_provider_model_id == "x-ai/grok-4.3"
            ),
            require_api_key=True,
        )


def test_real_run_missing_openrouter_key_fails_before_creating_candidate_run(tmp_path) -> None:
    repository = FakeRepository()
    repository.questions_by_dataset["J1"] = [
        CandidateQuestionRecord(101, "J1", "OAB_Bench", 119, "Q", None)
    ]
    retriever = FakeRetriever()

    @dataclass
    class MissingOpenRouterKeySettings(FakeSettings):
        openrouter_api_key: str | None = None

    service = RunCandidatesRagService(
        settings_loader=MissingOpenRouterKeySettings,
        connect_func=lambda _: FakeConnection(),
        repository_factory=lambda _: repository,
        retriever_factory=lambda _repository, _settings, _dataset: retriever,
        client_factory=lambda _request, _settings: FakeClient(),
        snapshot_service_factory=lambda _repository: FakeSnapshotService(),
    )

    with pytest.raises(ValueError, match="OPENROUTER_KEY is required for openrouter candidate execution"):
        service.run(
            RunCandidatesRagRequest(
                dataset="J1",
                model_name="x-ai/grok-4.3",
                provider="remote_http",
                batch_size=1,
                question_sequence_start=119,
                question_sequence_end=119,
                audit_log=str(tmp_path / "missing-openrouter-key.log"),
                no_audit_animation=True,
            )
        )

    assert repository.created_runs == []
    assert repository.updated_runs == []


def test_missing_featherless_key_produces_clear_error(tmp_path) -> None:
    repository = FakeRepository()
    service = _service(repository=repository)

    @dataclass
    class MissingFeatherlessKeySettings(FakeSettings):
        featherless_api_key: str | None = None

    with pytest.raises(ValueError, match="FEATHERLESS_API is required for featherless candidate execution"):
        _resolve_candidate_provider_config(
            settings=MissingFeatherlessKeySettings(),
            assignment=next(
                assignment
                for assignment in repository.assignments
                if assignment.av3_provider_model_id == "Qwen/Qwen2.5-7B-Instruct"
            ),
            require_api_key=True,
        )


def test_real_run_missing_featherless_key_fails_before_creating_candidate_run(tmp_path) -> None:
    repository = FakeRepository()
    repository.questions_by_dataset["J1"] = [
        CandidateQuestionRecord(101, "J1", "OAB_Bench", 95, "Q", None)
    ]
    retriever = FakeRetriever()

    @dataclass
    class MissingFeatherlessKeySettings(FakeSettings):
        featherless_api_key: str | None = None

    service = RunCandidatesRagService(
        settings_loader=MissingFeatherlessKeySettings,
        connect_func=lambda _: FakeConnection(),
        repository_factory=lambda _: repository,
        retriever_factory=lambda _repository, _settings, _dataset: retriever,
        client_factory=lambda _request, _settings: FakeClient(),
        snapshot_service_factory=lambda _repository: FakeSnapshotService(),
    )

    with pytest.raises(ValueError, match="FEATHERLESS_API is required for featherless candidate execution"):
        service.run(
            RunCandidatesRagRequest(
                dataset="J1",
                model_name="Qwen/Qwen2.5-7B-Instruct",
                provider="remote_http",
                batch_size=1,
                question_sequence_start=95,
                question_sequence_end=95,
                audit_log=str(tmp_path / "missing-featherless-key.log"),
                no_audit_animation=True,
            )
        )

    assert repository.created_runs == []
    assert repository.updated_runs == []


def test_provider_resolution_matches_assignment_registry_for_grok_and_qwen() -> None:
    repository = FakeRepository()
    grok_assignment = next(
        assignment
        for assignment in repository.assignments
        if assignment.av3_provider_model_id == "x-ai/grok-4.3"
    )
    qwen_assignment = next(
        assignment
        for assignment in repository.assignments
        if assignment.av3_provider_model_id == "Qwen/Qwen2.5-7B-Instruct"
    )

    assert grok_assignment.av3_provider == "openrouter"
    assert qwen_assignment.av3_provider == "featherless"


def test_resolve_candidate_max_tokens_uses_requested_override_for_gemma() -> None:
    assert resolve_candidate_max_tokens(
        model_name="google/gemma-2-2b-it",
        av3_provider="featherless",
        requested_max_tokens=1024,
    ) == 1024


def test_resolve_candidate_max_tokens_defaults_gemma_to_profile_default() -> None:
    assert resolve_candidate_max_tokens(
        model_name="google/gemma-2-2b-it",
        av3_provider="featherless",
        requested_max_tokens=None,
    ) == 768


def test_resolve_candidate_max_tokens_caps_phi3_requested_tokens() -> None:
    assert resolve_candidate_max_tokens(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        av3_provider="featherless",
        requested_max_tokens=768,
    ) == 512


def test_resolve_candidate_max_tokens_caps_tinyllama_requested_tokens() -> None:
    assert resolve_candidate_max_tokens(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        av3_provider="featherless",
        requested_max_tokens=768,
    ) == 512


def test_resolve_candidate_max_tokens_defaults_openrouter_grok_to_3000() -> None:
    assert resolve_candidate_max_tokens(
        model_name="x-ai/grok-4.3",
        av3_provider="openrouter",
        requested_max_tokens=None,
    ) == 3000


def test_runtime_observation_parser_extracts_platform_records_context_length() -> None:
    observation = parse_candidate_runtime_observation(
        "Remote candidate returned HTTP 400: Platform records context_length as 4096."
    )

    assert observation is not None
    assert observation.error_class == "context_window_exceeded"
    assert observation.observed_context_window_tokens == 4096


def test_runtime_observation_parser_extracts_context_size_of_pattern() -> None:
    observation = parse_candidate_runtime_observation("Requested prompt exceeds context size of 8192 tokens.")

    assert observation is not None
    assert observation.observed_context_window_tokens == 8192


def test_runtime_observation_parser_ignores_non_context_errors() -> None:
    assert parse_candidate_runtime_observation("Provider model is gated for this API key.") is None


def test_runtime_config_uses_db_profile_before_static_fallback() -> None:
    repository = FakeRepository()
    repository.runtime_profiles[("featherless", "google/gemma-2-2b-it")] = CandidateModelRuntimeProfileRecord(
        runtime_profile_id=1,
        av3_provider="featherless",
        provider_model_id="google/gemma-2-2b-it",
        provider_model_key="google/gemma-2-2b-it",
        context_window_tokens=4096,
        default_max_output_tokens=900,
        safety_margin_tokens=256,
        source="db_observed",
        confidence="observed_error",
        active=True,
        first_observed_at="2026-06-05T10:00:00",
        last_observed_at="2026-06-05T10:00:00",
        observation_count=2,
        metadata={},
        created_at="2026-06-05T10:00:00",
        updated_at="2026-06-05T10:00:00",
    )
    service = _service(repository=repository)
    request = RunCandidatesRagRequest(
        dataset="J1",
        model_name="google/gemma-2-2b-it",
        provider="remote_http",
        batch_size=1,
        question_sequence_start=71,
        question_sequence_end=71,
    )

    runtime_config = _resolve_candidate_runtime_config(
        repository=repository,
        settings=FakeSettings(),
        request=request,
        resolved=service.resolve(request),
        questions=[CandidateQuestionRecord(101, "J1", "OAB_Bench", 71, "Q", None)],
        require_api_key=True,
    )

    assert runtime_config.context_window_tokens == 4096
    assert runtime_config.default_max_output_tokens == 900
    assert runtime_config.safety_margin_tokens == 256
    assert runtime_config.model_profile_source == "db_observed"


def test_runtime_config_uses_env_override_before_db_profile() -> None:
    repository = FakeRepository()
    repository.runtime_profiles[("featherless", "google/gemma-2-2b-it")] = CandidateModelRuntimeProfileRecord(
        runtime_profile_id=1,
        av3_provider="featherless",
        provider_model_id="google/gemma-2-2b-it",
        provider_model_key="google/gemma-2-2b-it",
        context_window_tokens=4096,
        default_max_output_tokens=900,
        safety_margin_tokens=256,
        source="db_observed",
        confidence="observed_error",
        active=True,
        first_observed_at="2026-06-05T10:00:00",
        last_observed_at="2026-06-05T10:00:00",
        observation_count=2,
        metadata={},
        created_at="2026-06-05T10:00:00",
        updated_at="2026-06-05T10:00:00",
    )
    service = _service(repository=repository)
    request = RunCandidatesRagRequest(
        dataset="J1",
        model_name="google/gemma-2-2b-it",
        provider="remote_http",
        batch_size=1,
        question_sequence_start=71,
        question_sequence_end=71,
        remote_candidate_context_window_tokens=2048,
    )

    runtime_config = _resolve_candidate_runtime_config(
        repository=repository,
        settings=FakeSettings(),
        request=request,
        resolved=service.resolve(request),
        questions=[CandidateQuestionRecord(101, "J1", "OAB_Bench", 71, "Q", None)],
        require_api_key=True,
    )

    assert runtime_config.context_window_tokens == 2048
    assert runtime_config.model_profile_source == "env_override"


def test_runtime_config_uses_explicit_candidate_max_tokens_for_gemma() -> None:
    repository = FakeRepository()
    service = _service(repository=repository)
    request = RunCandidatesRagRequest(
        dataset="J1",
        model_name="google/gemma-2-2b-it",
        provider="remote_http",
        batch_size=1,
        question_sequence_start=71,
        question_sequence_end=71,
        remote_candidate_max_tokens=1024,
    )

    runtime_config = _resolve_candidate_runtime_config(
        repository=repository,
        settings=FakeSettings(),
        request=request,
        resolved=service.resolve(request),
        questions=[CandidateQuestionRecord(101, "J1", "OAB_Bench", 71, "Q", None)],
        require_api_key=True,
    )

    assert runtime_config.max_tokens == 1024


def test_explicit_candidate_max_tokens_reaches_remote_http_client_config() -> None:
    request = RunCandidatesRagRequest(
        dataset="J1",
        model_name="google/gemma-2-2b-it",
        provider="remote_http",
        remote_candidate_base_url="https://api.featherless.ai/v1",
        remote_candidate_api_key="candidate-secret",
        remote_candidate_temperature=0.2,
        remote_candidate_max_tokens=1024,
        remote_candidate_top_p=0.9,
    )

    client = _default_client_factory(request, FakeSettings())

    assert client.config.max_tokens == 1024


def test_provider_keys_are_not_logged_in_audit_file(tmp_path) -> None:
    repository = FakeRepository()
    repository.questions_by_dataset["J1"] = [
        CandidateQuestionRecord(101, "J1", "OAB_Bench", 119, "Questao J1 101.", None)
    ]
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    service = _service(repository=repository, retriever=retriever)
    audit_path = tmp_path / "candidate-run.log"

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="x-ai/grok-4.3",
            provider="remote_http",
            batch_size=1,
            question_sequence_start=119,
            question_sequence_end=119,
            audit_log=str(audit_path),
            no_audit_animation=True,
        )
    )

    audit_text = audit_path.read_text(encoding="utf-8")
    assert "openrouter-test-key" not in audit_text
    assert "featherless-test-key" not in audit_text
    assert "api_key: <set>" in audit_text


def test_context_window_failure_records_observation_and_upserts_profile(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101, chunk_text="A" * 800)
    client = FakeClient(
        failures={
            0: RuntimeError("Platform records context_length as 4096; prompt contains 3900 tokens; max_tokens 1024.")
        }
    )
    service = _service(repository=repository, retriever=retriever, client=client)

    result = service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="google/gemma-2-2b-it",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "context-window-learning.log"),
            no_audit_animation=True,
        )
    )

    assert result.summary is not None
    assert result.summary.failed_answers == 1
    assert repository.runtime_observations[0]["observed_context_window_tokens"] == 4096
    profile = repository.runtime_profiles[("featherless", "google/gemma-2-2b-it")]
    assert profile.context_window_tokens == 4096
    assert profile.source == "db_observed"
    assert repository.persisted_answers[0].raw_response["context_window_retry"]["attempted"] is False


def test_context_window_failure_retries_once_when_enabled(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101, chunk_text="A" * 800)
    client = FakeClient(
        failures={0: RuntimeError("Platform records context_length as 4096; context size of 4096.")},
        responses={
            1: CandidateRawResponse(
                text="Resposta final apos retry.",
                provider="fake",
                model="google/gemma-2-2b-it",
                latency_ms=11,
            )
        },
    )
    service = _service(repository=repository, retriever=retriever, client=client)

    result = service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="google/gemma-2-2b-it",
            provider="remote_http",
            batch_size=1,
            remote_candidate_retry_on_context_window=True,
            audit_log=str(tmp_path / "context-window-retry.log"),
            no_audit_animation=True,
        )
    )

    assert result.summary is not None
    assert result.summary.successful_answers == 1
    assert len(client.calls) == 2
    assert repository.persisted_answers[0].status == "success"
    assert repository.persisted_answers[0].raw_response["context_window_retry"]["attempted"] is True
    assert repository.updated_runs[-1]["metadata"]["candidate_runtime_profile"]["context_window_tokens"] == 4096


def test_context_window_failure_does_not_retry_when_disabled(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101, chunk_text="A" * 800)
    client = FakeClient(
        failures={0: RuntimeError("Platform records context_length as 4096; context size of 4096.")}
    )
    service = _service(repository=repository, retriever=retriever, client=client)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="google/gemma-2-2b-it",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "context-window-no-retry.log"),
            no_audit_animation=True,
        )
    )

    assert len(client.calls) == 1


def test_non_context_error_does_not_retry(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    client = FakeClient(failures={0: RuntimeError("Provider model is gated for this API key.")})
    service = _service(repository=repository, retriever=retriever, client=client)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="google/gemma-2-2b-it",
            provider="remote_http",
            batch_size=1,
            remote_candidate_retry_on_context_window=True,
            audit_log=str(tmp_path / "non-context-error.log"),
            no_audit_animation=True,
        )
    )

    assert len(client.calls) == 1
    assert repository.runtime_observations == []


def test_runtime_config_formatter_redacts_missing_key_in_dry_run() -> None:
    summary = _format_candidate_runtime_config(
        _resolve_candidate_runtime_config(
            repository=FakeRepository(),
            settings=type(
                "MissingKeysSettings",
                (),
                {
                    "openrouter_url": "https://openrouter.ai/api/v1",
                    "openrouter_api_key": None,
                    "featherless_url": "https://api.featherless.ai/v1",
                    "featherless_api_key": None,
                    "remote_candidate_temperature": 0.2,
                    "remote_candidate_max_tokens": 1024,
                    "remote_candidate_top_p": 0.9,
                },
            )(),
            request=RunCandidatesRagRequest(
                dataset="J1",
                model_name="x-ai/grok-4.3",
                provider="remote_http",
                batch_size=1,
                question_sequence_start=119,
                question_sequence_end=119,
            ),
            resolved=RunCandidatesRagService().resolve(
                RunCandidatesRagRequest(
                    dataset="J1",
                    model_name="x-ai/grok-4.3",
                    provider="remote_http",
                    batch_size=1,
                    question_sequence_start=119,
                    question_sequence_end=119,
                )
            ),
            questions=[CandidateQuestionRecord(101, "J1", "OAB_Bench", 119, "Q", None)],
            require_api_key=False,
        ),
        api_key_state="<not required in dry-run>",
    )

    assert "api_key: <not required in dry-run>" in summary
    assert "openrouter-test-key" not in summary


def test_selects_j1_questions(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    service = _service(repository=repository, retriever=retriever)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="candidate-j1",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "j1.log"),
            no_audit_animation=True,
        )
    )

    assert repository.select_calls[0]["dataset"] == "J1"


def test_selects_j2_questions(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J2", 202)] = _success_result(dataset="J2", question_id=202)
    service = _service(repository=repository, retriever=retriever)

    service.run(
        RunCandidatesRagRequest(
            dataset="J2",
            model_name="candidate-j2",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "j2.log"),
            no_audit_animation=True,
        )
    )

    assert repository.select_calls[0]["dataset"] == "J2"


def test_respects_batch_size(tmp_path) -> None:
    repository = FakeRepository()
    repository.questions_by_dataset["J1"] = [
        CandidateQuestionRecord(101, "J1", "OAB_Bench", 1, "Q1", None),
        CandidateQuestionRecord(102, "J1", "OAB_Bench", 2, "Q2", None),
        CandidateQuestionRecord(103, "J1", "OAB_Bench", 3, "Q3", None),
    ]
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    retriever.results[("J1", 102)] = _success_result(dataset="J1", question_id=102)
    service = _service(repository=repository, retriever=retriever)

    result = service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="candidate-j1",
            provider="remote_http",
            batch_size=2,
            audit_log=str(tmp_path / "batch.log"),
            no_audit_animation=True,
        )
    )

    assert repository.select_calls[0]["batch_size"] == 2
    assert result.summary is not None
    assert result.summary.selected_questions == 2


def test_respects_question_range(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    service = _service(repository=repository, retriever=retriever)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="candidate-j1",
            provider="remote_http",
            batch_size=1,
            question_sequence_start=3,
            question_sequence_end=5,
            audit_log=str(tmp_path / "range.log"),
            no_audit_animation=True,
        )
    )

    assert repository.select_calls[0]["question_sequence_start"] == 3
    assert repository.select_calls[0]["question_sequence_end"] == 5


def test_skips_existing_successful_answers(tmp_path) -> None:
    repository = FakeRepository()
    repository.existing_successes.add(("J1", "candidate-j1", 101))
    retriever = FakeRetriever()
    service = _service(repository=repository, retriever=retriever)

    result = service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="candidate-j1",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "skip.log"),
            no_audit_animation=True,
        )
    )

    assert result.summary is not None
    assert result.summary.skipped_questions == 1
    assert retriever.calls == []
    assert repository.persisted_answers == []


def test_retrieves_context_per_question(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    service = _service(repository=repository, retriever=retriever)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="candidate-j1",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "retrieve.log"),
            no_audit_animation=True,
        )
    )

    assert retriever.calls == [(101, "J1", None)]


def test_renders_prompt_with_retrieved_context(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101, chunk_text="Art. 5 da Lei X.")
    client = FakeClient()
    service = _service(repository=repository, retriever=retriever, client=client)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="candidate-j1",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "prompt.log"),
            no_audit_animation=True,
        )
    )

    prompt, _model = client.calls[0]
    assert "Questao J1 101." in prompt
    assert "Art. 5 da Lei X." in prompt


def test_calls_candidate_client_with_selected_model(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    client = FakeClient()
    service = _service(repository=repository, retriever=retriever, client=client)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="openai/gpt-5.4",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "model.log"),
            no_audit_animation=True,
        )
    )

    assert client.calls[0][1] == "openai/gpt-5.4"


def test_persists_successful_candidate_answer(tmp_path) -> None:
    repository = FakeRepository()
    repository.questions_by_dataset["J2"] = [
        CandidateQuestionRecord(
            question_id=202,
            dataset="J2",
            dataset_name="OAB_Exames",
            question_sequence=2,
            question_text="Qual alternativa correta?",
            alternatives={"A": "Opcao A", "B": "Opcao B"},
        )
    ]
    retriever = FakeRetriever()
    retriever.results[("J2", 202)] = _success_result(dataset="J2", question_id=202)
    client = FakeClient(
        responses={
            0: CandidateRawResponse(
                text="Justificativa breve.\nAlternativa final: B",
                provider="fake",
                model="candidate-j2",
                latency_ms=29,
                raw_response={"text": "Justificativa breve.\nAlternativa final: B"},
            )
        }
    )
    service = _service(repository=repository, retriever=retriever, client=client)

    service.run(
        RunCandidatesRagRequest(
            dataset="J2",
            model_name="candidate-j2",
            provider="remote_http",
            batch_size=1,
            save_raw_response=True,
            audit_log=str(tmp_path / "persist-success.log"),
            no_audit_animation=True,
        )
    )

    answer = repository.persisted_answers[0]
    assert answer.status == "success"
    assert answer.answer_text == "Justificativa breve.\nAlternativa final: B"
    assert answer.final_choice == "B"
    assert answer.raw_response is not None
    assert answer.raw_response["text"] == "Justificativa breve.\nAlternativa final: B"
    assert "candidate_budget" in answer.raw_response


def test_persists_context_snapshot(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    snapshot_service = FakeSnapshotService()
    service = _service(repository=repository, retriever=retriever, snapshot_service=snapshot_service)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="candidate-j1",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "snapshot.log"),
            no_audit_animation=True,
        )
    )

    assert snapshot_service.calls
    candidate_answer_id, retrieval_result = snapshot_service.calls[0]
    assert candidate_answer_id == 801
    assert retrieval_result.status == "success"


def test_budgeted_context_snapshot_uses_included_text_and_metadata(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    repository.questions_by_dataset["J1"] = [
        CandidateQuestionRecord(101, "J1", "OAB_Bench", 71, "Questao J1 101.", None)
    ]
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101, chunk_text="A" * 2000)
    snapshot_service = FakeSnapshotService()
    service = _service(repository=repository, retriever=retriever, snapshot_service=snapshot_service)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="google/gemma-2-2b-it",
            provider="remote_http",
            batch_size=1,
            remote_candidate_max_tokens=1024,
            remote_candidate_context_window_tokens=2400,
            remote_candidate_context_safety_margin_tokens=128,
            audit_log=str(tmp_path / "budgeted-snapshot.log"),
            no_audit_animation=True,
        )
    )

    _candidate_answer_id, retrieval_result = snapshot_service.calls[0]
    chunk = retrieval_result.chunks[0]
    budget_metadata = chunk.metadata["candidate_budget"]
    assert len(chunk.chunk_text) < 2000
    assert budget_metadata["included_in_prompt"] is True
    assert budget_metadata["was_truncated"] is True
    assert budget_metadata["truncation_reason"] == "context_budget"


def test_run_metadata_records_candidate_budget(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    repository.questions_by_dataset["J1"] = [
        CandidateQuestionRecord(101, "J1", "OAB_Bench", 71, "Questao J1 101.", None)
    ]
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101, chunk_text="A" * 2000)
    service = _service(repository=repository, retriever=retriever)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="google/gemma-2-2b-it",
            provider="remote_http",
            batch_size=1,
            remote_candidate_max_tokens=1024,
            remote_candidate_context_window_tokens=2400,
            remote_candidate_context_safety_margin_tokens=128,
            audit_log=str(tmp_path / "budgeted-run-metadata.log"),
            no_audit_animation=True,
        )
    )

    metadata = repository.updated_runs[-1]["metadata"]
    candidate_budget = metadata["candidate_budget"]
    assert candidate_budget["context_window_tokens"] == 2400
    assert candidate_budget["requested_max_tokens"] == 1024
    assert candidate_budget["final_max_tokens"] == 1024
    assert candidate_budget["safety_margin_tokens"] == 512
    assert candidate_budget["chars_per_token_estimate"] == 3
    assert candidate_budget["prompt_budget_utilization"] == 0.85
    assert candidate_budget["safe_prompt_budget"] == 864
    assert candidate_budget["target_prompt_budget"] == 734
    assert candidate_budget["estimated_prompt_tokens_after_budget"] <= candidate_budget["target_prompt_budget"]
    assert candidate_budget["retrieved_chunks"] == 1
    assert candidate_budget["included_chunks"] == 1
    assert candidate_budget["truncated_chunks"] == 1


def test_dry_run_logs_preflight_and_prompt_budget_without_creating_run(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    repository.questions_by_dataset["J1"] = [
        CandidateQuestionRecord(101, "J1", "OAB_Bench", 71, "Questao J1 101.", None)
    ]
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101, chunk_text="A" * 2000)
    audit_path = tmp_path / "gemma-dry-run-budget.log"
    service = _service(repository=repository, retriever=retriever)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="google/gemma-2-2b-it",
            provider="remote_http",
            batch_size=1,
            question_sequence_start=71,
            question_sequence_end=71,
            dry_run=True,
            audit_log=str(audit_path),
            no_audit_animation=True,
        )
    )

    audit_text = audit_path.read_text(encoding="utf-8")
    assert "Candidate runtime preflight:" in audit_text
    assert "api_key: <not required in dry-run>" in audit_text
    assert "final_max_tokens: 1024" in audit_text
    assert "context_window_tokens: 8192" in audit_text
    assert "Candidate prompt budget:" in audit_text
    assert "question_id: 101" in audit_text
    assert repository.created_runs == []


def test_budget_failure_fails_before_creating_run_or_provider_call(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    repository.questions_by_dataset["J1"] = [
        CandidateQuestionRecord(101, "J1", "OAB_Bench", 71, "Q" * 2000, None)
    ]
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101, chunk_text="A" * 2000)
    client = FakeClient()
    service = _service(repository=repository, retriever=retriever, client=client)

    with pytest.raises(ValueError, match="Fixed candidate prompt without retrieved context exceeds"):
        service.run(
            RunCandidatesRagRequest(
                dataset="J1",
                model_name="google/gemma-2-2b-it",
                provider="remote_http",
                batch_size=1,
                remote_candidate_max_tokens=80,
                remote_candidate_context_window_tokens=900,
                remote_candidate_context_safety_margin_tokens=20,
                audit_log=str(tmp_path / "budget-failure.log"),
                no_audit_animation=True,
            )
        )

    assert client.calls == []
    assert repository.created_runs == []
    assert repository.updated_runs == []


def test_records_failure_status_when_retrieval_has_no_chunks(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = RagRetrievalResult(
        question_id=101,
        dataset="J1",
        retrieval_run_id=21,
        retrieval_name="j1_source_urls_v1",
        embedding_model="text-embedding-3-small",
        top_k=5,
        status="no_chunks_found",
        chunks=[],
    )
    client = FakeClient()
    service = _service(repository=repository, retriever=retriever, client=client)

    result = service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="candidate-j1",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "retrieval-failure.log"),
            no_audit_animation=True,
        )
    )

    assert result.summary is not None
    assert result.summary.failed_answers == 1
    assert repository.persisted_answers[0].status == "failed"
    assert repository.persisted_answers[0].error_message == "Retrieval failed: no_chunks_found"
    assert client.calls == []


def test_records_failure_status_when_candidate_client_fails(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    client = FakeClient(failures={0: RuntimeError("candidate timeout")})
    service = _service(repository=repository, retriever=retriever, client=client)

    result = service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="candidate-j1",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "client-failure.log"),
            no_audit_animation=True,
        )
    )

    assert result.summary is not None
    assert result.summary.failed_answers == 1
    assert repository.persisted_answers[0].status == "failed"
    assert "candidate timeout" in str(repository.persisted_answers[0].error_message)


def test_fails_before_creating_run_when_model_has_no_runnable_assignment(tmp_path) -> None:
    repository = FakeRepository()
    repository.questions_by_dataset["J1"] = [
        CandidateQuestionRecord(
            question_id=119,
            dataset="J1",
            dataset_name="OAB_Bench",
            question_sequence=119,
            question_text="Questao J1 119.",
            alternatives=None,
        )
    ]
    retriever = FakeRetriever()
    retriever.results[("J1", 119)] = _success_result(dataset="J1", question_id=119)
    audit_path = tmp_path / "no-runnable-assignment.log"
    service = _service(
        repository=repository,
        retriever=retriever,
        client_factory=_default_client_factory,
    )

    with pytest.raises(
        ValueError,
        match=r"Candidate model google/gemini-3.5-flash has no runnable AV3 assignment for dataset=J1\.",
    ):
        service.run(
            RunCandidatesRagRequest(
                dataset="J1",
                model_name="google/gemini-3.5-flash",
                provider="remote_http",
                batch_size=1,
                audit_log=str(audit_path),
                no_audit_animation=True,
            )
        )

    assert repository.created_runs == []
    assert repository.updated_runs == []
    assert repository.persisted_answers == []


def test_emits_audit_events_for_run_question_retrieval_generation_persistence_skip_and_finish(tmp_path) -> None:
    repository = FakeRepository()
    repository.questions_by_dataset["J1"] = [
        CandidateQuestionRecord(101, "J1", "OAB_Bench", 1, "Q1", None),
        CandidateQuestionRecord(102, "J1", "OAB_Bench", 2, "Q2", None),
    ]
    repository.existing_successes.add(("J1", "candidate-j1", 102))
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    audit_path = tmp_path / "audit.log"
    service = _service(repository=repository, retriever=retriever)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="candidate-j1",
            provider="remote_http",
            batch_size=2,
            audit_log=str(audit_path),
            no_audit_animation=True,
        )
    )

    audit_text = audit_path.read_text(encoding="utf-8")
    assert "run_started" in audit_text
    assert "question_started" in audit_text
    assert "retrieval_finished" in audit_text
    assert "generation_finished" in audit_text
    assert "answer_persisted" in audit_text
    assert "question_skipped" in audit_text
    assert "run_finished" in audit_text


def test_does_not_leak_answer_key_rubric_or_guideline_into_rendered_prompt(tmp_path) -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    retriever.results[("J1", 101)] = _success_result(dataset="J1", question_id=101)
    client = FakeClient()
    service = _service(repository=repository, retriever=retriever, client=client)

    service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="candidate-j1",
            provider="remote_http",
            batch_size=1,
            audit_log=str(tmp_path / "safety.log"),
            no_audit_animation=True,
        )
    )

    prompt = client.calls[0][0].lower()
    assert "official_answer_key" not in prompt
    assert "guideline" not in prompt
    assert "judge_prompt" not in prompt
    assert "rubrica" not in prompt
