"""Embedding-model configuration service for AV3 RAG."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from .config import load_settings
from .db import connect
from .rag_curation import resolve_rag_curation_dataset


class RagEmbeddingConfigService:
    """Manage per-dataset embedding-model settings for the AV3 RAG pipeline."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], Any] = load_settings,
        connect_func: Callable[[str], Any] = connect,
        repository_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self._settings_loader = settings_loader
        self._connect = connect_func
        self._repository_factory = repository_factory

    def get(self, *, dataset: str) -> dict[str, Any]:
        dataset_code = resolve_rag_curation_dataset(dataset)
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            config = repository.get_rag_embedding_model_config(dataset=dataset_code)
        finally:
            connection.close()
        return {"record": asdict(config) if config is not None else None}

    def save(
        self,
        *,
        dataset: str,
        provider: str,
        model_name: str,
        dimensions: int | None,
        api_base_url: str | None,
        notes: str | None,
        updated_by: str,
    ) -> dict[str, Any]:
        dataset_code = resolve_rag_curation_dataset(dataset)
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            config = repository.upsert_rag_embedding_model_config(
                dataset=dataset_code,
                provider=provider,
                model_name=model_name,
                dimensions=dimensions,
                api_base_url=api_base_url,
                notes=notes,
                updated_by=updated_by,
            )
        finally:
            connection.close()
        return {"record": asdict(config)}

    def _make_repository(self, connection: Any) -> Any:
        if self._repository_factory is not None:
            return self._repository_factory(connection)
        from .repositories import JudgeRepository

        return JudgeRepository(connection)
