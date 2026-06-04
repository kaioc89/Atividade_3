from __future__ import annotations

from dataclasses import dataclass, field

from atividade_2.contracts import (
    CandidateAnswerRecord,
    CandidatePromptRecord,
    CandidateQuestionRecord,
    CandidateRawResponse,
    CandidateRunRecord,
    RagRetrievalResult,
    RagVectorBaseSummary,
    RetrievedRagChunk,
)
from atividade_2.run_candidates_rag_service import RunCandidatesRagRequest, RunCandidatesRagService


@dataclass
class FakeSettings:
    database_url: str = "postgresql://example.invalid/app"
    embedding_api_key: str | None = "embedding-test-key"


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
        self.next_candidate_run_id = 501
        self.next_candidate_answer_id = 801
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

    def ensure_schema(self) -> None:
        self.ensure_schema_calls += 1

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
        client_factory=lambda _request, _settings: client,
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

    def fail_connect(_: str) -> FakeConnection:
        raise AssertionError("dry-run must not connect to PostgreSQL")

    service = _service(repository=repository, retriever=retriever, client=client, connect_func=fail_connect)

    result = service.run(
        RunCandidatesRagRequest(
            dataset="J1",
            model_name="openai/gpt-5.4",
            provider="remote_http",
            batch_size=2,
            dry_run=True,
            audit_log=str(tmp_path / "candidate-dry-run.log"),
            no_audit_animation=True,
        )
    )

    assert result.dry_run is True
    assert result.candidate_run_id is None
    assert result.summary is None
    assert repository.ensure_schema_calls == 0
    assert repository.created_runs == []
    assert retriever.calls == []
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
    assert repository.updated_runs[-1]["run_status"] == "completed"


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
    assert answer.raw_response == {"text": "Justificativa breve.\nAlternativa final: B"}


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
