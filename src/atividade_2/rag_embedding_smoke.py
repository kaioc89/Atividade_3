"""Smoke-test helpers for AV3 embedding-provider connectivity."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .config import load_settings
from .db import connect
from .rag_embedding_client import request_openai_compatible_embeddings
from .rag_curation import resolve_rag_curation_dataset


class RagEmbeddingSmokeTestService:
    """Execute a lightweight embedding request using the saved dataset config."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], Any] = load_settings,
        connect_func: Callable[[str], Any] = connect,
        repository_factory: Callable[[Any], Any] | None = None,
        request_timeout_seconds: int = 30,
    ) -> None:
        self._settings_loader = settings_loader
        self._connect = connect_func
        self._repository_factory = repository_factory
        self._request_timeout_seconds = request_timeout_seconds

    def run(self, *, dataset: str, sample_text: str | None = None) -> dict[str, Any]:
        dataset_code = resolve_rag_curation_dataset(dataset)
        settings = self._settings_loader()
        api_key = getattr(settings, "embedding_api_key", None)
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY is required for the embedding smoke test.")

        connection = self._connect(settings.database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            config = repository.get_rag_embedding_model_config(dataset=dataset_code)
        finally:
            connection.close()

        if config is None:
            raise RuntimeError(f"Nenhuma configuracao de embedding encontrada para {dataset_code}.")

        api_base_url = (config.api_base_url or "").strip() or "https://api.openai.com/v1"
        text = (sample_text or _default_sample_text(dataset_code)).strip()
        result = request_openai_compatible_embeddings(
            api_base_url=api_base_url,
            api_key=api_key,
            model_name=config.model_name,
            texts=[text],
            dimensions=config.dimensions,
            timeout_seconds=self._request_timeout_seconds,
        )
        vector = result.vectors[0]
        return {
            "result": {
                "dataset": config.dataset,
                "dataset_name": config.dataset_name,
                "provider": config.provider,
                "model_name": config.model_name,
                "requested_dimensions": config.dimensions,
                "returned_dimensions": len(vector),
                "endpoint_url": result.endpoint_url,
                "endpoint_host": result.endpoint_host,
                "latency_ms": result.latency_ms,
                "sample_text": text,
                "tested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        }

    def _make_repository(self, connection: Any) -> Any:
        if self._repository_factory is not None:
            return self._repository_factory(connection)
        from .repositories import JudgeRepository

        return JudgeRepository(connection)


def _default_sample_text(dataset: str) -> str:
    if dataset == "J1":
        return "Improbidade administrativa, principio da legalidade e controle da administracao publica."
    return "Competencia tributaria, anterioridade e interpretacao constitucional em prova objetiva da OAB."
