from __future__ import annotations

from atividade_2.contracts import EvaluationRecord, ModelSpec, ParsedJudgeEvaluation, RagVectorBaseSummary
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
        retrieval_name="j1_curated_v1",
        retrieval_strategy="curated_articles_v1",
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
        retrieval_name="j1_curated_v1",
        retrieval_strategy="curated_articles_v1",
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
    assert inserted == 1
    assert len(insert_queries) == 1


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
