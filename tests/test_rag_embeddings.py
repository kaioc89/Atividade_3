"""Tests for RAG embedding generation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atividade_2.contracts import (
    RagEmbeddingGenerationSummary,
    RagEmbeddingModelConfigRecord,
    RagVectorBaseSummary,
)
from atividade_2.rag_embedding_client import EmbeddingBatchResult
from atividade_2.rag_embeddings import RagEmbeddingGenerationService
from atividade_2.rag_source_fetch import SourceUrlContent, SourceUrlFailure, SourceUrlFetchReport


@dataclass(frozen=True)
class FakeSettings:
    database_url: str = "postgresql://example/app"
    embedding_api_key: str = "embedding-secret"


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeRepository:
    def __init__(self, *, has_vector_base: bool) -> None:
        self.has_vector_base = has_vector_base
        self.calls: list[str] = []
        self.source_chunk_count = 0
        self.resolved_ranges: list[tuple[int | None, int | None]] = []

    def ensure_schema(self) -> None:
        self.calls.append("ensure_schema")

    def get_rag_embedding_model_config(self, *, dataset: str) -> RagEmbeddingModelConfigRecord:
        self.calls.append(f"get_config:{dataset}")
        return RagEmbeddingModelConfigRecord(
            config_id=1,
            dataset=dataset,
            dataset_name="OAB_Bench",
            provider="Qwen",
            model_name="Qwen/Qwen3-Embedding-8B",
            dimensions=None,
            api_base_url="https://api.featherless.ai/v1",
            notes=None,
            updated_by="tester",
            updated_at="2026-06-02T01:00:00",
        )

    def get_rag_vector_base_summary(self, *, dataset: str) -> RagVectorBaseSummary | None:
        self.calls.append(f"get_vector_base:{dataset}")
        if not self.has_vector_base:
            return None
        return _vector_base_summary(dataset=dataset)

    def materialize_rag_base_from_active_curation(self, *, dataset: str) -> None:
        self.calls.append(f"materialize:{dataset}")
        self.has_vector_base = True

    def resolve_rag_question_sequence_range_for_active_vector_base(
        self,
        *,
        dataset: str,
        question_sequence_start: int | None = None,
        question_sequence_end: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append(f"resolve_range:{dataset}:{question_sequence_start}:{question_sequence_end}")
        return {
            "start": 71 if question_sequence_start == 1 else question_sequence_start,
            "end": 72 if question_sequence_end == 2 else question_sequence_end,
            "mapped_from_dataset_position": question_sequence_start == 1 and question_sequence_end == 2,
        }

    def list_rag_source_documents_for_active_vector_base(
        self,
        *,
        dataset: str,
        question_sequence_start: int | None = None,
        question_sequence_end: int | None = None,
    ) -> list[dict[str, Any]]:
        self.resolved_ranges.append((question_sequence_start, question_sequence_end))
        self.calls.append(f"list_source_documents:{dataset}")
        return [
            {
                "document_id": 11,
                "url": "https://fonte.example/lei-14133, https://fonte.example/codigo-penal",
                "title": "Lei 14133",
            },
            {"document_id": 12, "url": "https://fonte.example/fora", "title": "Fonte fora"},
        ]

    def replace_rag_source_content_chunks_for_active_vector_base(
        self,
        *,
        dataset: str,
        source_contents: list[dict[str, Any]],
        question_sequence_start: int | None = None,
        question_sequence_end: int | None = None,
    ) -> int:
        self.resolved_ranges.append((question_sequence_start, question_sequence_end))
        self.calls.append(f"replace_source_chunks:{dataset}:{len(source_contents)}")
        self.source_chunk_count = len(source_contents)
        return self.source_chunk_count

    def list_rag_chunks_for_active_vector_base(
        self,
        *,
        dataset: str,
        question_sequence_start: int | None = None,
        question_sequence_end: int | None = None,
    ) -> list[dict[str, Any]]:
        self.resolved_ranges.append((question_sequence_start, question_sequence_end))
        self.calls.append(f"list_chunks:{dataset}")
        chunks = [
            {"chunk_id": 101, "chunk_text": "texto juridico um"},
            {"chunk_id": 102, "chunk_text": "texto juridico dois"},
        ]
        chunks.extend(
            {"chunk_id": 200 + index, "chunk_text": f"conteudo fonte {index}"}
            for index in range(1, self.source_chunk_count + 1)
        )
        return chunks

    def replace_rag_embeddings_for_active_vector_base(
        self,
        *,
        dataset: str,
        embedding_model: str,
        embedding_dimensions: int | None,
        provider: str,
        api_base_url: str | None,
        embeddings: list[dict[str, Any]],
        latency_ms: int,
        question_sequence_start: int | None = None,
        question_sequence_end: int | None = None,
    ) -> RagEmbeddingGenerationSummary:
        self.resolved_ranges.append((question_sequence_start, question_sequence_end))
        self.calls.append(f"replace_embeddings:{dataset}:{len(embeddings)}")
        return RagEmbeddingGenerationSummary(
            dataset=dataset,
            dataset_name="OAB_Bench",
            retrieval_run_id=21,
            retrieval_name="j1_curated_v1",
            import_run_id=7,
            embedding_model=embedding_model,
            provider=provider,
            api_base_url=api_base_url,
            requested_dimensions=embedding_dimensions,
            generated_embeddings=len(embeddings),
            total_chunks=2,
            latency_ms=latency_ms,
            created_at="2026-06-02T01:30:00",
        )


def test_generate_embeddings_materializes_vector_base_when_missing(monkeypatch) -> None:
    connection = FakeConnection()
    repository = FakeRepository(has_vector_base=False)

    monkeypatch.setattr(
        "atividade_2.rag_embeddings.request_openai_compatible_embeddings",
        _fake_embedding_request,
    )
    service = RagEmbeddingGenerationService(
        settings_loader=FakeSettings,
        connect_func=lambda _database_url: connection,
        repository_factory=lambda _connection: repository,
        source_fetcher=_fake_source_fetcher,
    )
    events: list[dict[str, Any]] = []

    result = service.run(dataset="J1", progress_callback=events.append)

    assert result["materialized_base"] is True
    assert result["summary"].generated_embeddings == 4
    assert any("Enviando lote" in event["message"] for event in events)
    assert events[-1]["state"] == "done"
    assert result["source_url_summary"] == {
        "attempted": 3,
        "succeeded": 2,
        "failed": 1,
        "inserted_chunks": 2,
        "failures": [{"url": "https://fonte.example/fora", "reason": "HTTP 404."}],
    }
    assert repository.calls == [
        "ensure_schema",
        "get_config:J1",
        "get_vector_base:J1",
        "materialize:J1",
        "list_source_documents:J1",
        "replace_source_chunks:J1:2",
        "list_chunks:J1",
        "replace_embeddings:J1:4",
    ]
    assert connection.closed is True


def test_generate_embeddings_reuses_existing_vector_base(monkeypatch) -> None:
    connection = FakeConnection()
    repository = FakeRepository(has_vector_base=True)

    monkeypatch.setattr(
        "atividade_2.rag_embeddings.request_openai_compatible_embeddings",
        _fake_embedding_request,
    )
    service = RagEmbeddingGenerationService(
        settings_loader=FakeSettings,
        connect_func=lambda _database_url: connection,
        repository_factory=lambda _connection: repository,
        source_fetcher=_fake_source_fetcher,
    )

    result = service.run(dataset="J1")

    assert result["materialized_base"] is False
    assert "materialize:J1" not in repository.calls
    assert result["summary"].generated_embeddings == 4


def test_generate_embeddings_maps_dataset_local_question_range(monkeypatch) -> None:
    connection = FakeConnection()
    repository = FakeRepository(has_vector_base=True)

    monkeypatch.setattr(
        "atividade_2.rag_embeddings.request_openai_compatible_embeddings",
        _fake_embedding_request,
    )
    service = RagEmbeddingGenerationService(
        settings_loader=FakeSettings,
        connect_func=lambda _database_url: connection,
        repository_factory=lambda _connection: repository,
        source_fetcher=_fake_source_fetcher,
    )
    events: list[dict[str, Any]] = []

    result = service.run(
        dataset="J1",
        question_sequence_start=1,
        question_sequence_end=2,
        progress_callback=events.append,
    )

    assert result["question_sequence_range"] == {
        "requested_start": 1,
        "requested_end": 2,
        "effective_start": 71,
        "effective_end": 72,
    }
    assert repository.resolved_ranges == [(71, 72), (71, 72), (71, 72), (71, 72)]
    assert any("posicao local do dataset" in event["message"] for event in events)


def _fake_embedding_request(**kwargs: Any) -> EmbeddingBatchResult:
    texts = kwargs["texts"]
    return EmbeddingBatchResult(
        vectors=[[0.1, 0.2, 0.3] for _text in texts],
        latency_ms=123,
        endpoint_url="https://api.featherless.ai/v1/embeddings",
        endpoint_host="api.featherless.ai",
    )


def _fake_source_fetcher(urls: list[str]) -> SourceUrlFetchReport:
    assert urls == [
        "https://fonte.example/lei-14133",
        "https://fonte.example/codigo-penal",
        "https://fonte.example/fora",
    ]
    return SourceUrlFetchReport(
        successes=[
            SourceUrlContent(
                url="https://fonte.example/lei-14133",
                content="Conteudo recuperado da Lei 14133.",
                content_type="text/html",
                title="Lei 14133",
            ),
            SourceUrlContent(
                url="https://fonte.example/codigo-penal",
                content="Conteudo recuperado do Codigo Penal.",
                content_type="text/html",
                title="Codigo Penal",
            )
        ],
        failures=[SourceUrlFailure(url="https://fonte.example/fora", reason="HTTP 404.")],
    )


def _vector_base_summary(*, dataset: str) -> RagVectorBaseSummary:
    return RagVectorBaseSummary(
        dataset=dataset,
        dataset_name="OAB_Bench",
        import_run_id=7,
        active_curation_run_id=7,
        matches_active_curation=True,
        retrieval_run_id=21,
        retrieval_name="j1_curated_v1",
        retrieval_strategy="curated_articles_v1",
        embedding_model=None,
        top_k=5,
        vector_enabled=True,
        lexical_enabled=False,
        rerank_enabled=False,
        document_count=70,
        chunk_count=2,
        embedding_count=0,
        status="materializada_sem_embeddings",
        created_at="2026-06-02T01:20:00",
    )
