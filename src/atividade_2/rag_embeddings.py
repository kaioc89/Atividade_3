"""Embedding generation pipeline for the materialized AV3 RAG base."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import load_settings
from .db import connect
from .rag_curation import resolve_rag_curation_dataset
from .rag_embedding_client import request_openai_compatible_embeddings


class RagEmbeddingGenerationService:
    """Generate embeddings for active materialized RAG chunks."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], Any] = load_settings,
        connect_func: Callable[[str], Any] = connect,
        repository_factory: Callable[[Any], Any] | None = None,
        request_timeout_seconds: int = 90,
        batch_size: int = 32,
    ) -> None:
        self._settings_loader = settings_loader
        self._connect = connect_func
        self._repository_factory = repository_factory
        self._request_timeout_seconds = request_timeout_seconds
        self._batch_size = batch_size

    def run(self, *, dataset: str, batch_size: int | None = None) -> dict[str, Any]:
        dataset_code = resolve_rag_curation_dataset(dataset)
        settings = self._settings_loader()
        api_key = getattr(settings, "embedding_api_key", None)
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY is required to generate RAG embeddings.")

        connection = self._connect(settings.database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            config = repository.get_rag_embedding_model_config(dataset=dataset_code)
            if config is None:
                raise RuntimeError(f"Nenhuma configuracao de embedding encontrada para {dataset_code}.")
            chunks = repository.list_rag_chunks_for_active_vector_base(dataset=dataset_code)
            if not chunks:
                raise RuntimeError(f"Nenhum chunk materializado encontrado para {dataset_code}.")

            effective_batch_size = max(1, int(batch_size or self._batch_size))
            generated: list[dict[str, Any]] = []
            total_latency_ms = 0
            for chunk_batch in _chunked(chunks, effective_batch_size):
                texts = [str(item["chunk_text"]) for item in chunk_batch]
                batch_result = request_openai_compatible_embeddings(
                    api_base_url=(config.api_base_url or "").strip() or "https://api.openai.com/v1",
                    api_key=api_key,
                    model_name=config.model_name,
                    texts=texts,
                    dimensions=config.dimensions,
                    timeout_seconds=self._request_timeout_seconds,
                )
                total_latency_ms += batch_result.latency_ms
                for item, vector in zip(chunk_batch, batch_result.vectors, strict=True):
                    generated.append(
                        {
                            "chunk_id": int(item["chunk_id"]),
                            "embedding": vector,
                        }
                    )

            summary = repository.replace_rag_embeddings_for_active_vector_base(
                dataset=dataset_code,
                embedding_model=config.model_name,
                embedding_dimensions=config.dimensions,
                provider=config.provider,
                api_base_url=config.api_base_url,
                embeddings=generated,
                latency_ms=total_latency_ms,
            )
        finally:
            connection.close()
        return {"summary": summary}

    def _make_repository(self, connection: Any) -> Any:
        if self._repository_factory is not None:
            return self._repository_factory(connection)
        from .repositories import JudgeRepository

        return JudgeRepository(connection)


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]
