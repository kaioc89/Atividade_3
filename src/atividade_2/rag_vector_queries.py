"""Preview and semantic-query helpers for the AV3 vector base."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import load_settings
from .db import connect
from .rag_curation import resolve_rag_curation_dataset
from .rag_embedding_client import request_openai_compatible_embeddings


class RagVectorQueryService:
    """Inspect and query the active vector base for one dataset."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], Any] = load_settings,
        connect_func: Callable[[str], Any] = connect,
        repository_factory: Callable[[Any], Any] | None = None,
        request_timeout_seconds: int = 60,
    ) -> None:
        self._settings_loader = settings_loader
        self._connect = connect_func
        self._repository_factory = repository_factory
        self._request_timeout_seconds = request_timeout_seconds

    def preview(self, *, dataset: str, limit: int = 8) -> dict[str, Any]:
        dataset_code = resolve_rag_curation_dataset(dataset)
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            vector_base = repository.get_rag_vector_base_summary(dataset=dataset_code)
            documents = repository.list_rag_vector_documents_preview(dataset=dataset_code, limit=limit)
            chunks = repository.list_rag_vector_chunks_preview(dataset=dataset_code, limit=limit)
        finally:
            connection.close()
        return {
            "dataset": dataset_code,
            "vector_base": vector_base,
            "documents": documents,
            "chunks": chunks,
        }

    def search(self, *, dataset: str, query_text: str, top_k: int = 5) -> dict[str, Any]:
        dataset_code = resolve_rag_curation_dataset(dataset)
        normalized_query = query_text.strip()
        if not normalized_query:
            raise ValueError("Informe um texto para consultar a base vetorial.")

        settings = self._settings_loader()
        api_key = getattr(settings, "embedding_api_key", None)
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY is required for vector search.")

        connection = self._connect(settings.database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            config = repository.get_rag_embedding_model_config(dataset=dataset_code)
            if config is None:
                raise RuntimeError(f"Nenhuma configuracao de embedding encontrada para {dataset_code}.")
            vector_base = repository.get_rag_vector_base_summary(dataset=dataset_code)
            if vector_base is None:
                raise RuntimeError(f"Nenhuma base vetorial ativa encontrada para {dataset_code}.")
            if vector_base.embedding_count <= 0:
                raise RuntimeError(f"A base vetorial de {dataset_code} ainda nao possui embeddings gerados.")

            batch = request_openai_compatible_embeddings(
                api_base_url=(config.api_base_url or "").strip() or "https://api.openai.com/v1",
                api_key=api_key,
                model_name=config.model_name,
                texts=[normalized_query],
                dimensions=config.dimensions,
                timeout_seconds=self._request_timeout_seconds,
            )
            query_vector = batch.vectors[0]
            results = repository.search_rag_chunks_by_embedding(
                dataset=dataset_code,
                embedding_model=config.model_name,
                query_vector=query_vector,
                top_k=max(1, int(top_k)),
            )
        finally:
            connection.close()

        return {
            "dataset": dataset_code,
            "query": normalized_query,
            "top_k": max(1, int(top_k)),
            "latency_ms": batch.latency_ms,
            "returned_dimensions": len(query_vector),
            "results": results,
        }

    def _make_repository(self, connection: Any) -> Any:
        if self._repository_factory is not None:
            return self._repository_factory(connection)
        from .repositories import JudgeRepository

        return JudgeRepository(connection)
