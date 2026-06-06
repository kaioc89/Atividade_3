"""Tests for RAG embedding generation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.error import URLError

from atividade_2.contracts import (
    RagEmbeddingGenerationSummary,
    RagEmbeddingModelConfigRecord,
    RagVectorBaseSummary,
)
from atividade_2.rag_embedding_client import EmbeddingBatchResult, EmbeddingProviderError, request_openai_compatible_embeddings
from atividade_2.rag_embeddings import RagEmbeddingGenerationService
from atividade_2.rag_source_fetch import SourceUrlContent, SourceUrlFailure, SourceUrlFetchReport
from atividade_2.repositories import _split_source_content


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
    def __init__(
        self,
        *,
        has_vector_base: bool,
        vector_base_matches_active_curation: bool = True,
        retrieval_name: str = "j1_source_urls_v2",
        retrieval_strategy: str = "source_url_only_v2",
    ) -> None:
        self.has_vector_base = has_vector_base
        self.vector_base_matches_active_curation = vector_base_matches_active_curation
        self.retrieval_name = retrieval_name
        self.retrieval_strategy = retrieval_strategy
        self.calls: list[str] = []
        self.source_chunk_count = 0
        self.resolved_ranges: list[tuple[int | None, int | None]] = []
        self.saved_embedding_batches: list[list[dict[str, Any]]] = []

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
        return _vector_base_summary(
            dataset=dataset,
            import_run_id=7 if self.vector_base_matches_active_curation else 6,
            active_curation_run_id=7,
            matches_active_curation=self.vector_base_matches_active_curation,
            retrieval_name=self.retrieval_name,
            retrieval_strategy=self.retrieval_strategy,
        )

    def materialize_rag_base_from_active_curation(self, *, dataset: str) -> None:
        self.calls.append(f"materialize:{dataset}")
        self.has_vector_base = True
        self.vector_base_matches_active_curation = True
        self.retrieval_name = f"{dataset.lower()}_source_urls_v2"
        self.retrieval_strategy = "source_url_only_v2"

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
                "url": (
                    "https://fonte.example/lei-14133, "
                    "http://fonte.example/lei-14133, "
                    "https://fonte.example/codigo-penal"
                ),
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
        self.source_chunk_count = len({str(item["content"]) for item in source_contents})
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
        return [
            {
                "chunk_id": 200 + index,
                "chunk_text": f"conteudo fonte {index}",
                "source_kind": "source_url_content",
            }
            for index in range(1, self.source_chunk_count + 1)
        ]

    def clear_rag_embeddings_for_active_vector_base(
        self,
        *,
        dataset: str,
        embedding_model: str,
        question_sequence_start: int | None = None,
        question_sequence_end: int | None = None,
    ) -> None:
        self.resolved_ranges.append((question_sequence_start, question_sequence_end))
        self.calls.append(f"clear_embeddings:{dataset}:{embedding_model}")

    def upsert_rag_embedding_batch_for_active_vector_base(
        self,
        *,
        dataset: str,
        embedding_model: str,
        embedding_dimensions: int | None,
        provider: str,
        api_base_url: str | None,
        embeddings: list[dict[str, Any]],
    ) -> None:
        self.calls.append(f"upsert_batch:{dataset}:{len(embeddings)}")
        self.saved_embedding_batches.append(embeddings)

    def build_rag_embedding_generation_summary(
        self,
        *,
        dataset: str,
        embedding_model: str,
        embedding_dimensions: int | None,
        provider: str,
        api_base_url: str | None,
        generated_embeddings: int,
        latency_ms: int,
    ) -> RagEmbeddingGenerationSummary:
        self.calls.append(f"build_summary:{dataset}:{generated_embeddings}")
        return RagEmbeddingGenerationSummary(
            dataset=dataset,
            dataset_name="OAB_Bench",
            retrieval_run_id=21,
            retrieval_name=self.retrieval_name,
            import_run_id=7,
            embedding_model=embedding_model,
            provider=provider,
            api_base_url=api_base_url,
            requested_dimensions=embedding_dimensions,
            generated_embeddings=generated_embeddings,
            total_chunks=generated_embeddings,
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
    assert result["summary"].generated_embeddings == 2
    assert any("Enviando lote" in event["message"] for event in events)
    assert any("fonte/URL recuperada=2" in event["message"] for event in events)
    assert events[-1]["state"] == "done"
    assert result["chunk_summary"] == {
        "total": 2,
        "by_source_kind": {"source_url_content": 2},
        "curation_chunks": 0,
        "source_url_chunks": 2,
    }
    assert result["source_url_summary"] == {
        "references": 4,
        "attempted": 3,
        "deduplicated": 1,
        "succeeded": 2,
        "failed": 1,
        "inserted_chunks": 2,
        "failures": [{"url": "https://fonte.example/fora", "reason": "HTTP 404."}],
    }
    assert any("1 duplicada(s) por normalizacao de URL" in event["message"] for event in events)
    assert repository.calls == [
        "ensure_schema",
        "get_config:J1",
        "get_vector_base:J1",
        "materialize:J1",
        "list_source_documents:J1",
        "replace_source_chunks:J1:2",
        "list_chunks:J1",
        "clear_embeddings:J1:Qwen/Qwen3-Embedding-8B",
        "upsert_batch:J1:2",
        "build_summary:J1:2",
    ]
    assert len(repository.saved_embedding_batches) == 1
    assert len(repository.saved_embedding_batches[0]) == 2
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
    assert result["vector_base_trace"] == {"action": "reused"}
    assert "materialize:J1" not in repository.calls
    assert result["summary"].generated_embeddings == 2


def test_generate_embeddings_rematerializes_legacy_vector_base(monkeypatch) -> None:
    connection = FakeConnection()
    repository = FakeRepository(
        has_vector_base=True,
        retrieval_name="j1_source_urls_v1",
        retrieval_strategy="source_url_only_v1",
    )

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
    assert result["vector_base_trace"] == {
        "action": "refreshed_legacy_strategy",
        "previous_retrieval_run_id": 21,
        "previous_retrieval_name": "j1_source_urls_v1",
        "previous_retrieval_strategy": "source_url_only_v1",
        "expected_retrieval_strategy": "source_url_only_v2",
    }
    assert "materialize:J1" in repository.calls
    assert any("estrategia anterior" in event["message"] for event in events)
    assert result["summary"].retrieval_name == "j1_source_urls_v2"
    assert result["summary"].generated_embeddings == 2


def test_generate_embeddings_rematerializes_stale_vector_base(monkeypatch) -> None:
    connection = FakeConnection()
    repository = FakeRepository(has_vector_base=True, vector_base_matches_active_curation=False)

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
    assert result["vector_base_trace"] == {
        "action": "refreshed_stale",
        "previous_import_run_id": 6,
        "active_curation_run_id": 7,
        "previous_retrieval_run_id": 21,
    }
    assert "materialize:J1" in repository.calls
    assert any("Base vetorial ativa desatualizada" in event["message"] for event in events)
    assert any(
        event.get("previous_import_run_id") == 6 and event.get("active_curation_run_id") == 7
        for event in events
    )
    assert result["summary"].generated_embeddings == 2


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


def test_generate_embeddings_logs_batch_failure_context(monkeypatch) -> None:
    connection = FakeConnection()
    repository = FakeRepository(has_vector_base=True)

    def _broken_embedding_request(**kwargs: Any) -> EmbeddingBatchResult:
        raise EmbeddingProviderError("Embedding request failed after 3 attempt(s): <urlopen error [Errno 32] Broken pipe>")

    monkeypatch.setattr(
        "atividade_2.rag_embeddings.request_openai_compatible_embeddings",
        _broken_embedding_request,
    )
    service = RagEmbeddingGenerationService(
        settings_loader=FakeSettings,
        connect_func=lambda _database_url: connection,
        repository_factory=lambda _connection: repository,
        source_fetcher=_fake_source_fetcher,
    )
    events: list[dict[str, Any]] = []

    try:
        service.run(dataset="J1", progress_callback=events.append)
    except RuntimeError as error:
        assert "Falha ao gerar embeddings no lote 1/1" in str(error)
    else:
        raise AssertionError("Expected embedding generation to fail.")

    assert any("Falha no lote 1/1" in event["message"] for event in events)
    assert any("provider=Qwen" in event["message"] for event in events)
    assert any(event["state"] == "error" for event in events)


def test_generate_embeddings_persists_incrementally_by_batch(monkeypatch) -> None:
    connection = FakeConnection()
    repository = FakeRepository(has_vector_base=True)

    def _batch_embedding_request(**kwargs: Any) -> EmbeddingBatchResult:
        texts = kwargs["texts"]
        return EmbeddingBatchResult(
            vectors=[[0.1, 0.2, 0.3] for _ in texts],
            latency_ms=50,
            endpoint_url="https://api.featherless.ai/v1/embeddings",
            endpoint_host="api.featherless.ai",
        )

    monkeypatch.setattr(
        "atividade_2.rag_embeddings.request_openai_compatible_embeddings",
        _batch_embedding_request,
    )
    service = RagEmbeddingGenerationService(
        settings_loader=FakeSettings,
        connect_func=lambda _database_url: connection,
        repository_factory=lambda _connection: repository,
        source_fetcher=_fake_source_fetcher,
        batch_size=2,
    )

    result = service.run(dataset="J1")

    assert result["summary"].generated_embeddings == 2
    assert repository.calls.count("upsert_batch:J1:2") == 1
    assert len(repository.saved_embedding_batches) == 1
    assert [len(batch) for batch in repository.saved_embedding_batches] == [2]


def test_split_source_content_aligns_overlap_to_word_start() -> None:
    content = " ".join(f"palavra{index:03d}" for index in range(90))

    chunks = _split_source_content(content=content, max_chunk_chars=500, overlap_chars=100)

    assert len(chunks) > 1
    assert chunks[1].startswith("palavra035 ")
    assert not chunks[1].startswith("5 palavra")


def test_embedding_client_retries_broken_pipe(monkeypatch) -> None:
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return b'{"data":[{"embedding":[0.1,0.2]}]}'

    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise URLError(OSError(32, "Broken pipe"))
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    monkeypatch.setattr("atividade_2.rag_embedding_client.time.sleep", sleep_calls.append)

    result = request_openai_compatible_embeddings(
        api_base_url="https://api.example/v1",
        api_key="secret",
        model_name="demo-embedding",
        texts=["texto"],
        dimensions=None,
    )

    assert result.endpoint_url == "https://api.example/v1/embeddings"
    assert result.vectors == [[0.1, 0.2]]
    assert attempts["count"] == 3
    assert sleep_calls == [0.75, 1.5]


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


def _vector_base_summary(
    *,
    dataset: str,
    import_run_id: int = 7,
    active_curation_run_id: int | None = 7,
    matches_active_curation: bool = True,
    retrieval_name: str = "j1_source_urls_v2",
    retrieval_strategy: str = "source_url_only_v2",
) -> RagVectorBaseSummary:
    return RagVectorBaseSummary(
        dataset=dataset,
        dataset_name="OAB_Bench",
        import_run_id=import_run_id,
        active_curation_run_id=active_curation_run_id,
        matches_active_curation=matches_active_curation,
        retrieval_run_id=21,
        retrieval_name=retrieval_name,
        retrieval_strategy=retrieval_strategy,
        embedding_model=None,
        top_k=5,
        vector_enabled=True,
        lexical_enabled=False,
        rerank_enabled=False,
        document_count=70,
        chunk_count=0,
        embedding_count=0,
        status="materializada_sem_embeddings",
        created_at="2026-06-02T01:20:00",
    )
