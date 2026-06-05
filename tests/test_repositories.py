from __future__ import annotations

from datetime import datetime
from pathlib import Path

from atividade_2.contracts import (
    CandidateAnswerContextChunkRecord,
    CandidateAnswerRecord,
    CandidateQuestionRecord,
    CandidateRunRecord,
    EvaluationRecord,
    ModelSpec,
    ParsedJudgeEvaluation,
    RagVectorBaseSummary,
)
from atividade_2.evaluation_details import EvaluationDetails
from atividade_2.repositories import JudgeRepository, _default_prompt_config


class RecordingCursor:
    def __init__(self) -> None:
        self.query = ""
        self.params = []

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, query, params=None) -> None:
        self.query = query
        self.params = list(params or [])

    def fetchall(self):
        return []


class RecordingConnection:
    def __init__(self) -> None:
        self.cursor_instance = RecordingCursor()

    def cursor(self) -> RecordingCursor:
        return self.cursor_instance


class TransactionConnection:
    def __init__(self, cursor) -> None:
        self.cursor_instance = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def cursor(self):
        return self.cursor_instance


class MultiRecordingCursor:
    def __init__(self, *, fetchone_rows=None, fetchall_rows=None) -> None:
        self.queries = []
        self.params = []
        self.fetchone_rows = list(fetchone_rows or [])
        self.fetchall_rows = list(fetchall_rows or [])
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, query, params=None) -> None:
        self.queries.append(query)
        self.params.append(list(params or []))

    def fetchone(self):
        if self.fetchone_rows:
            return self.fetchone_rows.pop(0)
        return None

    def fetchall(self):
        if self.fetchall_rows:
            return self.fetchall_rows.pop(0)
        return []


def test_pending_answer_selection_takes_a_batch_per_required_judge() -> None:
    connection = RecordingConnection()
    repository = JudgeRepository(connection)
    repository.ensure_judge_model = lambda model: 10 if model.requested == "judge-1" else 20  # type: ignore[method-assign]

    repository.select_pending_candidate_answers(
        dataset="J2",
        batch_size=2,
        required_evaluations=(
            (ModelSpec(requested="judge-1", provider_model="provider/judge-1"), "principal", "2plus1"),
            (ModelSpec(requested="judge-2", provider_model="provider/judge-2"), "controle", "2plus1"),
        ),
    )

    query = connection.cursor_instance.query
    assert "ROW_NUMBER() OVER" in query
    assert "PARTITION BY" in query
    assert "required.id_modelo_juiz" in query
    assert "required.papel_juiz" in query
    assert "WHERE required_rank <= %s" in query
    assert connection.cursor_instance.params[-1] == 2


def test_default_j1_prompt_matches_professor_style_persona_and_rubric() -> None:
    defaults = _default_prompt_config("OAB_Bench")
    assert "Desembargador" in defaults["persona"]
    assert "densidade de informacao correta" in defaults["persona"]
    assert "Rubrica de avaliacao (1 a 5)" in defaults["rubric"]
    assert "Ignore o tamanho do texto" in defaults["rubric"]


def test_rag_vector_search_does_not_hide_duplicate_chunk_text() -> None:
    cursor = MultiRecordingCursor()
    repository = JudgeRepository(TransactionConnection(cursor))
    repository.get_rag_vector_base_summary = lambda dataset: RagVectorBaseSummary(  # type: ignore[method-assign]
        dataset=dataset,
        dataset_name="OAB_Bench",
        import_run_id=7,
        active_curation_run_id=7,
        matches_active_curation=True,
        retrieval_run_id=21,
        retrieval_name="j1_source_urls_v1",
        retrieval_strategy="source_url_only_v1",
        embedding_model="Qwen/Qwen3-Embedding-8B",
        top_k=5,
        vector_enabled=True,
        lexical_enabled=False,
        rerank_enabled=False,
        document_count=70,
        chunk_count=433,
        embedding_count=433,
        status="pronta_com_embeddings",
        created_at="2026-06-02T01:20:00",
    )

    repository.search_rag_chunks_by_embedding(
        dataset="J1",
        embedding_model="Qwen/Qwen3-Embedding-8B",
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
    )

    query = cursor.queries[0]
    assert "PARTITION BY md5(c.chunk_text)" not in query
    assert "WHERE duplicate_rank = 1" not in query
    assert "ORDER BY e.embedding_vector <=> %s::vector ASC, c.id_chunk ASC" in query


def test_rag_source_chunk_replacement_skips_duplicate_text_chunks() -> None:
    cursor = MultiRecordingCursor(fetchall_rows=[[]])
    repository = JudgeRepository(TransactionConnection(cursor))
    repository.get_rag_vector_base_summary = lambda dataset: RagVectorBaseSummary(  # type: ignore[method-assign]
        dataset=dataset,
        dataset_name="OAB_Bench",
        import_run_id=7,
        active_curation_run_id=7,
        matches_active_curation=True,
        retrieval_run_id=21,
        retrieval_name="j1_source_urls_v1",
        retrieval_strategy="source_url_only_v1",
        embedding_model="Qwen/Qwen3-Embedding-8B",
        top_k=5,
        vector_enabled=True,
        lexical_enabled=False,
        rerank_enabled=False,
        document_count=70,
        chunk_count=433,
        embedding_count=433,
        status="pronta_com_embeddings",
        created_at="2026-06-02T01:20:00",
    )

    inserted = repository.replace_rag_source_content_chunks_for_active_vector_base(
        dataset="J1",
        source_contents=[
            {
                "document_id": 10,
                "url": "https://fonte.example/a",
                "content_type": "text/html",
                "content": "Texto normativo repetido.",
            },
            {
                "document_id": 11,
                "url": "https://fonte.example/b",
                "content_type": "text/html",
                "content": "Texto   normativo\nrepetido.",
            },
        ],
    )

    insert_queries = [
        query
        for query in cursor.queries
        if "INSERT INTO av3.rag_chunks" in query
    ]
    assert inserted == 2
    assert len(insert_queries) == 2


def test_get_question_for_rag_retrieval_loads_only_candidate_safe_fields() -> None:
    cursor = MultiRecordingCursor(fetchone_rows=[(41, "OAB_Bench", "Enunciado seguro.")])
    repository = JudgeRepository(TransactionConnection(cursor))

    result = repository.get_question_for_rag_retrieval(question_id=41, dataset="J1")

    assert result is not None
    assert result.question_id == 41
    assert result.dataset == "J1"
    assert result.question_text == "Enunciado seguro."
    query = cursor.queries[0]
    assert "p.enunciado" in query
    assert "p.resposta_ouro" not in query
    assert "gabarito_jsonb" not in query
    assert cursor.params[0] == [41, "OAB_Bench"]


def test_evaluation_details_schema_is_auxiliary_and_unique_by_evaluation() -> None:
    cursor = MultiRecordingCursor()
    repository = JudgeRepository(TransactionConnection(cursor))

    repository._ensure_evaluation_details_schema(cursor)

    schema_sql = cursor.queries[0]
    assert "CREATE TABLE IF NOT EXISTS avaliacao_juiz_detalhes" in schema_sql
    assert "id_avaliacao INTEGER NOT NULL UNIQUE" in schema_sql
    assert "REFERENCES avaliacoes_juiz(id_avaliacao) ON DELETE CASCADE" in schema_sql
    assert "raw_output_jsonb JSONB" in schema_sql
    assert "ALTER TABLE avaliacoes_juiz" not in schema_sql


def test_persist_evaluation_writes_details_after_official_evaluation_insert() -> None:
    cursor = MultiRecordingCursor(fetchone_rows=[(123,)])
    repository = JudgeRepository(TransactionConnection(cursor))
    repository.ensure_judge_model = lambda model: 10  # type: ignore[method-assign]

    repository.persist_evaluation(
        EvaluationRecord(
            answer_id=1,
            judge_model=ModelSpec(requested="judge", provider_model="provider/judge"),
            prompt_id=2,
            stored_role="principal",
            panel_mode="single",
            trigger_reason="single_mode",
            score=5,
            rationale="ok",
            latency_ms=10,
            parsed_evaluation=ParsedJudgeEvaluation(
                score=5,
                rationale="ok",
                legal_accuracy="alta",
                hallucination_risk="baixo",
                rubric_alignment="aderente",
                requires_human_review=False,
                criteria={"citation_quality": "boa"},
                raw_output_jsonb={"score": 5, "api_key": "<redacted>"},
            ),
        )
    )

    official_sql = cursor.queries[0]
    details_sql = cursor.queries[1]
    assert "INSERT INTO avaliacoes_juiz" in official_sql
    assert "RETURNING id_avaliacao" in official_sql
    assert "INSERT INTO avaliacao_juiz_detalhes" in details_sql
    assert "ON CONFLICT (id_avaliacao) DO UPDATE" in details_sql
    assert cursor.params[1][0] == 123
    assert cursor.params[1][1:5] == ["alta", "baixo", "aderente", False]
    assert "citation_quality" in cursor.params[1][5]
    assert "<redacted>" in cursor.params[1][6]


def test_details_rollback_drops_only_auxiliary_table() -> None:
    cursor = MultiRecordingCursor()
    repository = JudgeRepository(TransactionConnection(cursor))

    repository.rollback_evaluation_details_schema()

    assert cursor.queries == ["DROP TABLE IF EXISTS avaliacao_juiz_detalhes;"]


def test_find_evaluation_id_returns_only_unique_match() -> None:
    unique_cursor = MultiRecordingCursor(fetchall_rows=[[(55,)]])
    repository = JudgeRepository(TransactionConnection(unique_cursor))

    matched = repository.find_evaluation_id_for_details(
        answer_id=1,
        judge_model="provider/judge",
        role="principal",
        panel_mode="single",
        trigger_reason="single_mode",
        score=5,
    )

    assert matched == 55
    assert "JOIN modelos" in unique_cursor.queries[0]
    assert unique_cursor.params[0] == [
        1,
        "provider/judge",
        "provider/judge",
        "principal",
        "single:%",
        "%:single_mode",
        5,
    ]

    ambiguous_cursor = MultiRecordingCursor(fetchall_rows=[[(55,), (56,)]])
    ambiguous_repository = JudgeRepository(TransactionConnection(ambiguous_cursor))
    assert (
        ambiguous_repository.find_evaluation_id_for_details(
            answer_id=1,
            judge_model="provider/judge",
            role=None,
            panel_mode=None,
            trigger_reason=None,
            score=None,
        )
        is None
    )


def test_persist_evaluation_details_uses_controlled_upsert() -> None:
    cursor = MultiRecordingCursor()
    repository = JudgeRepository(TransactionConnection(cursor))

    repository.persist_evaluation_details(
        evaluation_id=99,
        details=EvaluationDetails(
            legal_accuracy=None,
            hallucination_risk="baixo",
            criteria={"answer_completeness": "completa"},
            raw_output_jsonb=None,
        ),
    )

    query = cursor.queries[0]
    assert "ON CONFLICT (id_avaliacao) DO UPDATE" in query
    assert "COALESCE(EXCLUDED.legal_accuracy" in query
    assert "criteria = avaliacao_juiz_detalhes.criteria || EXCLUDED.criteria" in query
    assert "UPDATE avaliacoes_juiz" not in query


def test_candidate_rag_schema_creates_prompt_table_with_active_prompt_index() -> None:
    cursor = MultiRecordingCursor()
    repository = JudgeRepository(TransactionConnection(cursor))

    repository._ensure_candidate_rag_schema(cursor)

    sql_statements = "\n".join(cursor.queries)
    assert "CREATE TABLE IF NOT EXISTS av3.prompt_candidatos" in sql_statements
    assert "id_prompt_candidato SERIAL PRIMARY KEY" in sql_statements
    assert "UNIQUE (dataset_code, versao)" in sql_statements
    assert "ds_instrucao_rag TEXT NOT NULL" in sql_statements
    assert "created_by VARCHAR(120) NOT NULL DEFAULT 'system'" in sql_statements
    assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_prompt_candidatos_active_dataset" in sql_statements
    assert "ON av3.prompt_candidatos (dataset_code)" in sql_statements
    assert "WHERE ativo;" in sql_statements


def test_candidate_assignment_schema_creates_registry_and_range_tables() -> None:
    cursor = MultiRecordingCursor()
    repository = JudgeRepository(TransactionConnection(cursor))

    repository._ensure_candidate_rag_schema(cursor)

    sql_statements = "\n".join(cursor.queries)
    assert "CREATE TABLE IF NOT EXISTS av3.candidate_model_assignments" in sql_statements
    assert "REFERENCES public.modelos(id_modelo)" in sql_statements
    assert "UNIQUE (id_modelo_av2, owner, original_provider_model_id)" in sql_statements
    assert "CREATE TABLE IF NOT EXISTS av3.candidate_model_assignment_ranges" in sql_statements
    assert "REFERENCES av3.candidate_model_assignments(id_assignment) ON DELETE CASCADE" in sql_statements
    assert "CHECK (dataset_code IN ('J1', 'J2'))" in sql_statements
    assert "CHECK (question_sequence_end >= question_sequence_start)" in sql_statements
    assert "CREATE INDEX IF NOT EXISTS idx_candidate_model_assignments_owner_model" in sql_statements
    assert "CREATE INDEX IF NOT EXISTS idx_candidate_model_assignment_ranges_dataset_sequence" in sql_statements


def test_candidate_rag_schema_creates_run_answer_and_chunk_constraints() -> None:
    cursor = MultiRecordingCursor()
    repository = JudgeRepository(TransactionConnection(cursor))

    repository._ensure_candidate_rag_schema(cursor)

    sql_statements = "\n".join(cursor.queries)
    assert "CREATE TABLE IF NOT EXISTS av3.candidate_runs" in sql_statements
    assert "REFERENCES av3.retrieval_runs(id_retrieval_run)" in sql_statements
    assert "REFERENCES av3.prompt_candidatos(id_prompt_candidato)" in sql_statements
    assert "batch_size INTEGER NOT NULL CHECK (batch_size >= 1)" in sql_statements
    assert "run_status VARCHAR(30) NOT NULL DEFAULT 'created'" in sql_statements
    assert "run_status IN ('created', 'running', 'completed', 'failed', 'cancelled')" in sql_statements
    assert "CREATE TABLE IF NOT EXISTS av3.candidate_answers" in sql_statements
    assert "REFERENCES av3.candidate_runs(id_candidate_run) ON DELETE CASCADE" in sql_statements
    assert "REFERENCES public.perguntas(id_pergunta)" in sql_statements
    assert "UNIQUE (id_candidate_run, id_pergunta)" in sql_statements
    assert "status VARCHAR(30) NOT NULL DEFAULT 'created'" in sql_statements
    assert "status IN ('created', 'running', 'success', 'failed', 'skipped')" in sql_statements
    assert "latency_ms INTEGER CHECK (latency_ms IS NULL OR latency_ms >= 0)" in sql_statements
    assert "CREATE TABLE IF NOT EXISTS av3.candidate_answer_context_chunks" in sql_statements
    assert "REFERENCES av3.candidate_answers(id_candidate_answer) ON DELETE CASCADE" in sql_statements
    assert "REFERENCES av3.rag_chunks(id_chunk)" in sql_statements
    assert "UNIQUE (id_candidate_answer, rank)" in sql_statements
    assert "UNIQUE (id_candidate_answer, id_chunk)" in sql_statements
    assert "rank INTEGER NOT NULL CHECK (rank >= 1)" in sql_statements


def test_candidate_rag_schema_is_idempotent_on_repeated_calls() -> None:
    cursor = MultiRecordingCursor()
    repository = JudgeRepository(TransactionConnection(cursor))

    repository._ensure_candidate_rag_schema(cursor)
    repository._ensure_candidate_rag_schema(cursor)

    assert len(cursor.queries) == 28
    assert all("IF NOT EXISTS" in query for query in cursor.queries)
    assert all("DROP TABLE" not in query for query in cursor.queries)
    assert all("ALTER TABLE" not in query for query in cursor.queries)


def test_ensure_schema_invokes_candidate_rag_schema_after_vector_schema() -> None:
    cursor = MultiRecordingCursor()
    repository = JudgeRepository(TransactionConnection(cursor))
    calls: list[str] = []

    repository._ensure_prompt_schema = lambda inner_cursor: calls.append("prompt")  # type: ignore[method-assign]
    repository._ensure_evaluation_prompt_fk = lambda inner_cursor: calls.append("evaluation-prompt-fk")  # type: ignore[method-assign]
    repository._ensure_meta_evaluation_schema = lambda inner_cursor: calls.append("meta-evaluation")  # type: ignore[method-assign]
    repository._ensure_evaluation_details_schema = lambda inner_cursor: calls.append("evaluation-details")  # type: ignore[method-assign]
    repository._ensure_rag_curation_schema = lambda inner_cursor: calls.append("rag-curation")  # type: ignore[method-assign]
    repository._ensure_rag_vector_schema = lambda inner_cursor: calls.append("rag-vector")  # type: ignore[method-assign]
    repository._ensure_candidate_rag_schema = lambda inner_cursor: calls.append("candidate-rag")  # type: ignore[method-assign]

    repository.ensure_schema()

    assert calls == [
        "prompt",
        "evaluation-prompt-fk",
        "meta-evaluation",
        "evaluation-details",
        "rag-curation",
        "rag-vector",
        "candidate-rag",
    ]


def test_candidate_schema_is_present_in_project_ddl() -> None:
    ddl = Path("database/ddl_banco/ddl_atividade_2.sql").read_text(encoding="utf-8")

    assert "CREATE TABLE av3.prompt_candidatos (" in ddl
    assert "CREATE TABLE av3.candidate_model_assignments (" in ddl
    assert "CREATE TABLE av3.candidate_model_assignment_ranges (" in ddl
    assert "CREATE TABLE av3.candidate_runs (" in ddl
    assert "CREATE TABLE av3.candidate_answers (" in ddl
    assert "CREATE TABLE av3.candidate_answer_context_chunks (" in ddl
    assert "CREATE UNIQUE INDEX idx_prompt_candidatos_active_dataset" in ddl
    assert "CREATE INDEX idx_candidate_model_assignments_provider_status" in ddl
    assert "CREATE INDEX idx_candidate_answers_run_status" in ddl


def test_create_candidate_run_persists_metadata_json_and_returns_record() -> None:
    created_at = datetime(2026, 6, 4, 13, 20, 0)
    cursor = MultiRecordingCursor(
        fetchone_rows=[
            (
                17,
                "J1",
                31,
                7,
                "candidate-model",
                "openai",
                0.2,
                512,
                0.95,
                25,
                "created",
                None,
                None,
                "tester",
                '{"mode": "rag", "top_k": 5}',
                created_at,
            )
        ]
    )
    repository = JudgeRepository(TransactionConnection(cursor))

    record = repository.create_candidate_run(
        run=CandidateRunRecord(
            candidate_run_id=None,
            dataset="j1",
            retrieval_run_id=31,
            prompt_id=7,
            model_name="candidate-model",
            provider="openai",
            batch_size=25,
            temperature=0.2,
            max_tokens=512,
            top_p=0.95,
            created_by="tester",
            metadata={"top_k": 5, "mode": "rag"},
        )
    )

    assert "INSERT INTO av3.candidate_runs" in cursor.queries[0]
    assert "metadata_jsonb" in cursor.queries[0]
    assert cursor.params[0] == [
        "J1",
        31,
        7,
        "candidate-model",
        "openai",
        0.2,
        512,
        0.95,
        25,
        "created",
        None,
        None,
        "tester",
        '{"mode": "rag", "top_k": 5}',
    ]
    assert record == CandidateRunRecord(
        candidate_run_id=17,
        dataset="J1",
        retrieval_run_id=31,
        prompt_id=7,
        model_name="candidate-model",
        provider="openai",
        batch_size=25,
        run_status="created",
        temperature=0.2,
        max_tokens=512,
        top_p=0.95,
        created_by="tester",
        metadata={"mode": "rag", "top_k": 5},
        created_at=created_at.isoformat(),
    )


def test_persist_candidate_answer_upserts_by_run_and_question() -> None:
    created_at = datetime(2026, 6, 4, 13, 45, 0)
    cursor = MultiRecordingCursor(
        fetchone_rows=[
            (
                41,
                17,
                77,
                "candidate-model",
                "Resposta final",
                "B",
                "prompt renderizado",
                "success",
                None,
                812,
                '{"answer": "Resposta final", "usage": {"tokens": 11}}',
                created_at,
            )
        ]
    )
    repository = JudgeRepository(TransactionConnection(cursor))

    record = repository.persist_candidate_answer(
        answer=CandidateAnswerRecord(
            candidate_answer_id=None,
            candidate_run_id=17,
            question_id=77,
            model_name="candidate-model",
            rendered_prompt="prompt renderizado",
            status="success",
            answer_text="Resposta final",
            final_choice="B",
            latency_ms=812,
            raw_response={"usage": {"tokens": 11}, "answer": "Resposta final"},
        )
    )

    assert "INSERT INTO av3.candidate_answers" in cursor.queries[0]
    assert "ON CONFLICT (id_candidate_run, id_pergunta) DO UPDATE" in cursor.queries[0]
    assert cursor.params[0] == [
        17,
        77,
        "candidate-model",
        "Resposta final",
        "B",
        "prompt renderizado",
        "success",
        None,
        812,
        '{"answer": "Resposta final", "usage": {"tokens": 11}}',
    ]
    assert record == CandidateAnswerRecord(
        candidate_answer_id=41,
        candidate_run_id=17,
        question_id=77,
        model_name="candidate-model",
        rendered_prompt="prompt renderizado",
        status="success",
        answer_text="Resposta final",
        final_choice="B",
        latency_ms=812,
        raw_response={"answer": "Resposta final", "usage": {"tokens": 11}},
        created_at=created_at.isoformat(),
    )


def test_persist_candidate_answer_context_chunks_replaces_existing_snapshots() -> None:
    first_created_at = datetime(2026, 6, 4, 14, 0, 0)
    second_created_at = datetime(2026, 6, 4, 14, 0, 1)
    cursor = MultiRecordingCursor(
        fetchone_rows=[
            (
                91,
                41,
                501,
                1,
                0.912345,
                "Trecho 1",
                "https://fonte.example/1",
                '{"source_kind": "lei"}',
                first_created_at,
            ),
            (
                92,
                41,
                502,
                2,
                0.812345,
                "Trecho 2",
                None,
                '{"source_kind": "questao"}',
                second_created_at,
            ),
        ]
    )
    repository = JudgeRepository(TransactionConnection(cursor))

    records = repository.persist_candidate_answer_context_chunks(
        candidate_answer_id=41,
        chunks=[
            CandidateAnswerContextChunkRecord(
                answer_context_chunk_id=None,
                candidate_answer_id=41,
                chunk_id=501,
                rank=1,
                chunk_text_snapshot="Trecho 1",
                similarity_score=0.912345,
                source_url="https://fonte.example/1",
                metadata={"source_kind": "lei"},
            ),
            CandidateAnswerContextChunkRecord(
                answer_context_chunk_id=None,
                candidate_answer_id=41,
                chunk_id=502,
                rank=2,
                chunk_text_snapshot="Trecho 2",
                similarity_score=0.812345,
                metadata={"source_kind": "questao"},
            ),
        ],
    )

    assert "DELETE FROM av3.candidate_answer_context_chunks" in cursor.queries[0]
    assert cursor.params[0] == [41]
    assert "INSERT INTO av3.candidate_answer_context_chunks" in cursor.queries[1]
    assert "INSERT INTO av3.candidate_answer_context_chunks" in cursor.queries[2]
    assert records == [
        CandidateAnswerContextChunkRecord(
            answer_context_chunk_id=91,
            candidate_answer_id=41,
            chunk_id=501,
            rank=1,
            chunk_text_snapshot="Trecho 1",
            similarity_score=0.912345,
            source_url="https://fonte.example/1",
            metadata={"source_kind": "lei"},
            created_at=first_created_at.isoformat(),
        ),
        CandidateAnswerContextChunkRecord(
            answer_context_chunk_id=92,
            candidate_answer_id=41,
            chunk_id=502,
            rank=2,
            chunk_text_snapshot="Trecho 2",
            similarity_score=0.812345,
            source_url=None,
            metadata={"source_kind": "questao"},
            created_at=second_created_at.isoformat(),
        ),
    ]


def test_list_candidate_runs_filters_by_dataset_and_status() -> None:
    created_at = datetime(2026, 6, 4, 14, 10, 0)
    cursor = MultiRecordingCursor(
        fetchall_rows=[
            [
                (
                    17,
                    "J2",
                    31,
                    7,
                    "candidate-model",
                    "openai",
                    None,
                    256,
                    None,
                    10,
                    "running",
                    None,
                    None,
                    "system",
                    "{}",
                    created_at,
                )
            ]
        ]
    )
    repository = JudgeRepository(TransactionConnection(cursor))

    records = repository.list_candidate_runs(dataset="j2", run_status="running", limit=5)

    assert "FROM av3.candidate_runs" in cursor.queries[0]
    assert "WHERE dataset_code = %s AND run_status = %s" in cursor.queries[0]
    assert cursor.params[0] == ["J2", "running", 5]
    assert records == [
        CandidateRunRecord(
            candidate_run_id=17,
            dataset="J2",
            retrieval_run_id=31,
            prompt_id=7,
            model_name="candidate-model",
            provider="openai",
            batch_size=10,
            run_status="running",
            temperature=None,
            max_tokens=256,
            top_p=None,
            created_by="system",
            metadata={},
            created_at=created_at.isoformat(),
        )
    ]


def test_list_candidate_answers_filters_by_run_and_status() -> None:
    created_at = datetime(2026, 6, 4, 14, 25, 0)
    cursor = MultiRecordingCursor(
        fetchall_rows=[
            [
                (
                    41,
                    17,
                    77,
                    "candidate-model",
                    "Resposta final",
                    "B",
                    "prompt renderizado",
                    "success",
                    None,
                    812,
                    '{"answer": "Resposta final"}',
                    created_at,
                )
            ]
        ]
    )
    repository = JudgeRepository(TransactionConnection(cursor))

    records = repository.list_candidate_answers(candidate_run_id=17, status="success")

    assert "FROM av3.candidate_answers" in cursor.queries[0]
    assert "WHERE id_candidate_run = %s" in cursor.queries[0]
    assert "AND status = %s" in cursor.queries[0]
    assert cursor.params[0] == [17, "success"]
    assert records == [
        CandidateAnswerRecord(
            candidate_answer_id=41,
            candidate_run_id=17,
            question_id=77,
            model_name="candidate-model",
            rendered_prompt="prompt renderizado",
            status="success",
            answer_text="Resposta final",
            final_choice="B",
            latency_ms=812,
            raw_response={"answer": "Resposta final"},
            created_at=created_at.isoformat(),
        )
    ]


def test_get_or_create_candidate_prompt_returns_existing_active_prompt() -> None:
    created_at = datetime(2026, 6, 4, 14, 40, 0)
    cursor = MultiRecordingCursor(
        fetchone_rows=[
            (
                12,
                "J2",
                3,
                "persona",
                "contexto",
                "instrucao",
                "saida",
                True,
                "tester",
                created_at,
            )
        ]
    )
    repository = JudgeRepository(TransactionConnection(cursor))

    record = repository.get_or_create_candidate_prompt(dataset="j2")

    assert "FROM av3.prompt_candidatos" in cursor.queries[0]
    assert "ativo = TRUE" in cursor.queries[0]
    assert cursor.params[0] == ["J2"]
    assert record.prompt_id == 12
    assert record.dataset == "J2"
    assert record.version == 3


def test_get_or_create_candidate_prompt_seeds_default_active_prompt_when_missing() -> None:
    created_at = datetime(2026, 6, 4, 14, 45, 0)
    cursor = MultiRecordingCursor(
        fetchone_rows=[
            None,
            (1,),
            (
                19,
                "J1",
                1,
                "Você é um candidato do exame da OAB respondendo uma questão discursiva.",
                "Questão original:\n```text\n{pergunta_oab}\n```",
                "{contexto_rag}\n\nUse os trechos recuperados apenas como apoio para fundamentar a resposta.\n- Responda como candidato da OAB, em português.\n- Se o contexto não for suficiente, reconheça a limitação sem inventar normas, fatos ou jurisprudência.\n- Não mencione critérios de correção, respostas de referência ou avaliação.",
                "Entregue uma resposta objetiva e juridicamente fundamentada.\nFinalize com o bloco:\nResposta final:\n<sua resposta>",
                True,
                "system",
                created_at,
            ),
        ]
    )
    repository = JudgeRepository(TransactionConnection(cursor))

    record = repository.get_or_create_candidate_prompt(dataset="J1")

    assert "SELECT COALESCE(MAX(versao), 0) + 1" in cursor.queries[1]
    assert "INSERT INTO av3.prompt_candidatos" in cursor.queries[3]
    assert record.prompt_id == 19
    assert record.active is True
    assert record.created_by == "system"


def test_select_candidate_questions_uses_active_vector_base_scope_and_filters() -> None:
    cursor = MultiRecordingCursor(
        fetchall_rows=[
            [
                (101, "J2", 8, "Qual alternativa correta?", '{"A": "Opcao A", "B": "Opcao B"}'),
            ]
        ]
    )
    repository = JudgeRepository(TransactionConnection(cursor))
    repository.get_rag_vector_base_summary = lambda dataset: RagVectorBaseSummary(  # type: ignore[method-assign]
        dataset=dataset,
        dataset_name="OAB_Exames",
        import_run_id=31,
        active_curation_run_id=31,
        matches_active_curation=True,
        retrieval_run_id=21,
        retrieval_name="j2_source_urls_v1",
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

    records = repository.select_candidate_questions(
        dataset="J2",
        batch_size=2,
        question_sequence_start=5,
        question_sequence_end=9,
        question_id=101,
    )

    assert "FROM av3.curadoria_questoes q" in cursor.queries[0]
    assert "q.id_import_run = %s" in cursor.queries[0]
    assert "q.id_pergunta = %s" in cursor.queries[0]
    assert "q.question_sequence >= %s" in cursor.queries[0]
    assert "q.question_sequence <= %s" in cursor.queries[0]
    assert cursor.params[0] == ["J2", 31, 101, 5, 9, 2]
    assert records == [
        CandidateQuestionRecord(
            question_id=101,
            dataset="J2",
            dataset_name="OAB_Exames",
            question_sequence=8,
            question_text="Qual alternativa correta?",
            alternatives={"A": "Opcao A", "B": "Opcao B"},
        )
    ]


def test_successful_candidate_answer_exists_filters_by_dataset_model_question_and_status() -> None:
    cursor = MultiRecordingCursor(fetchone_rows=[(1,)])
    repository = JudgeRepository(TransactionConnection(cursor))

    exists = repository.successful_candidate_answer_exists(
        dataset="j1",
        model_name="candidate-model",
        question_id=77,
        exclude_candidate_run_id=17,
    )

    assert exists is True
    assert "FROM av3.candidate_answers a" in cursor.queries[0]
    assert "JOIN av3.candidate_runs r" in cursor.queries[0]
    assert "a.status = 'success'" in cursor.queries[0]
    assert "r.id_candidate_run <> %s" in cursor.queries[0]
    assert cursor.params[0] == ["J1", "candidate-model", 77, 17]
