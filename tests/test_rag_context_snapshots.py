"""Tests for AV3 RAG retrieval snapshot persistence orchestration."""

from __future__ import annotations

from atividade_2.contracts import (
    CandidateAnswerContextChunkRecord,
    RagRetrievalResult,
    RetrievedRagChunk,
)
from atividade_2.rag_context_snapshots import RagContextSnapshotService


class FakeRepository:
    def __init__(
        self,
        *,
        returned_records: list[CandidateAnswerContextChunkRecord] | None = None,
    ) -> None:
        self.calls: list[tuple[int, list[CandidateAnswerContextChunkRecord]]] = []
        self.returned_records = returned_records

    def persist_candidate_answer_context_chunks(
        self,
        *,
        candidate_answer_id: int,
        chunks: list[CandidateAnswerContextChunkRecord],
    ) -> list[CandidateAnswerContextChunkRecord]:
        self.calls.append((candidate_answer_id, chunks))
        if self.returned_records is not None:
            return list(self.returned_records)
        return list(chunks)


def test_persist_retrieval_snapshot_persists_successful_j1_chunks() -> None:
    repository = FakeRepository()
    service = RagContextSnapshotService(repository=repository)

    result = service.persist_retrieval_snapshot(
        candidate_answer_id=41,
        retrieval_result=RagRetrievalResult(
            question_id=101,
            dataset="J1",
            retrieval_run_id=123,
            retrieval_name="j1_source_urls_v1",
            embedding_model="Qwen/Qwen3-Embedding-8B",
            top_k=5,
            status="success",
            chunks=[
                RetrievedRagChunk(
                    rank=1,
                    chunk_id=501,
                    chunk_text="Texto exato do chunk J1.",
                    source_kind="source_url_content",
                    document_id=10,
                    document_key="doc-j1-10",
                    lei="Lei 8.112/1990",
                    norma="Estatuto",
                    url="https://example.test/j1/10",
                    urn="urn:lex:br:federal:lei:1990-12-11;8112",
                    artigo="Art. 10",
                    topico="Servidor publico",
                    relevancia="alta",
                    tipo="lei",
                    distance=0.12,
                    similarity=0.88,
                    metadata={
                        "sequence": 22,
                        "guideline": "nao persistir",
                        "judge_prompt": "nao persistir",
                        "nested": {
                            "keep": "ok",
                            "answer_key": "A",
                        },
                    },
                )
            ],
        ),
    )

    assert len(repository.calls) == 1
    persisted_candidate_answer_id, persisted_chunks = repository.calls[0]
    assert persisted_candidate_answer_id == 41
    assert result == persisted_chunks
    assert persisted_chunks == [
        CandidateAnswerContextChunkRecord(
            answer_context_chunk_id=None,
            candidate_answer_id=41,
            chunk_id=501,
            rank=1,
            chunk_text_snapshot="Texto exato do chunk J1.",
            similarity_score=0.88,
            source_url="https://example.test/j1/10",
            metadata={
                "sequence": 22,
                "nested": {"keep": "ok"},
                "dataset": "J1",
                "retrieval_run_id": 123,
                "retrieval_name": "j1_source_urls_v1",
                "embedding_model": "Qwen/Qwen3-Embedding-8B",
                "top_k": 5,
                "source_kind": "source_url_content",
                "document_id": 10,
                "document_key": "doc-j1-10",
                "lei": "Lei 8.112/1990",
                "norma": "Estatuto",
                "urn": "urn:lex:br:federal:lei:1990-12-11;8112",
                "artigo": "Art. 10",
                "topico": "Servidor publico",
                "relevancia": "alta",
                "tipo": "lei",
                "distance": 0.12,
                "similarity": 0.88,
            },
        )
    ]
    assert "guideline" not in persisted_chunks[0].metadata
    assert "judge_prompt" not in persisted_chunks[0].metadata
    assert "answer_key" not in persisted_chunks[0].metadata["nested"]


def test_persist_retrieval_snapshot_persists_successful_j2_chunks_and_returns_repository_rows() -> None:
    returned_records = [
        CandidateAnswerContextChunkRecord(
            answer_context_chunk_id=91,
            candidate_answer_id=77,
            chunk_id=900,
            rank=1,
            chunk_text_snapshot="Chunk J2.",
            similarity_score=0.93,
            source_url="https://example.test/j2",
            metadata={
                "dataset": "J2",
                "retrieval_run_id": 222,
                "retrieval_name": "j2_source_urls_v1",
                "embedding_model": "Qwen/Qwen3-Embedding-8B",
                "top_k": 3,
                "source_kind": "source_url_content",
                "document_id": 901,
                "document_key": "doc-j2",
                "norma": "Norma J2",
                "topico": "Tema J2",
                "relevancia": "media",
                "tipo": "jurisprudencia",
                "distance": 0.07,
                "similarity": 0.93,
                "keep": "value",
            },
            created_at="2026-06-04T14:00:00",
        )
    ]
    repository = FakeRepository(returned_records=returned_records)
    service = RagContextSnapshotService(repository=repository)

    result = service.persist_retrieval_snapshot(
        candidate_answer_id=77,
        retrieval_result=RagRetrievalResult(
            question_id=202,
            dataset="J2",
            retrieval_run_id=222,
            retrieval_name="j2_source_urls_v1",
            embedding_model="Qwen/Qwen3-Embedding-8B",
            top_k=3,
            status="success",
            chunks=[
                RetrievedRagChunk(
                    rank=1,
                    chunk_id=900,
                    chunk_text="Chunk J2.",
                    source_kind="source_url_content",
                    document_id=901,
                    document_key="doc-j2",
                    lei=None,
                    norma="Norma J2",
                    url="https://example.test/j2",
                    urn=None,
                    artigo=None,
                    topico="Tema J2",
                    relevancia="media",
                    tipo="jurisprudencia",
                    distance=0.07,
                    similarity=0.93,
                    metadata={
                        "keep": "value",
                        "official_answer_key": "B",
                        "correct_alternative": "B",
                        "judge_score": 5,
                    },
                )
            ],
        ),
    )

    assert len(repository.calls) == 1
    persisted_candidate_answer_id, persisted_chunks = repository.calls[0]
    assert persisted_candidate_answer_id == 77
    assert result == returned_records
    assert persisted_chunks == [
        CandidateAnswerContextChunkRecord(
            answer_context_chunk_id=None,
            candidate_answer_id=77,
            chunk_id=900,
            rank=1,
            chunk_text_snapshot="Chunk J2.",
            similarity_score=0.93,
            source_url="https://example.test/j2",
            metadata={
                "keep": "value",
                "dataset": "J2",
                "retrieval_run_id": 222,
                "retrieval_name": "j2_source_urls_v1",
                "embedding_model": "Qwen/Qwen3-Embedding-8B",
                "top_k": 3,
                "source_kind": "source_url_content",
                "document_id": 901,
                "document_key": "doc-j2",
                "norma": "Norma J2",
                "topico": "Tema J2",
                "relevancia": "media",
                "tipo": "jurisprudencia",
                "distance": 0.07,
                "similarity": 0.93,
            },
        )
    ]
    assert "official_answer_key" not in persisted_chunks[0].metadata
    assert "correct_alternative" not in persisted_chunks[0].metadata
    assert "judge_score" not in persisted_chunks[0].metadata


def test_persist_retrieval_snapshot_returns_empty_list_for_expected_fallback_statuses() -> None:
    repository = FakeRepository()
    service = RagContextSnapshotService(repository=repository)

    for status in (
        "question_not_found",
        "vector_base_not_found",
        "embedding_model_not_configured",
        "no_chunks_found",
    ):
        result = service.persist_retrieval_snapshot(
            candidate_answer_id=55,
            retrieval_result=RagRetrievalResult(
                question_id=999,
                dataset="J1",
                retrieval_run_id=None,
                retrieval_name=None,
                embedding_model=None,
                top_k=5,
                status=status,
            ),
        )

        assert result == []

    assert repository.calls == []
