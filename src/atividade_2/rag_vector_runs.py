"""Manage activation and deletion of persisted AV3 vector runs."""

from __future__ import annotations

from collections.abc import Callable

from .config import load_settings
from .db import connect


class RagVectorRunService:
    """Operate on retrieval/vector runs for one dataset."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], object] = load_settings,
        connect_func: Callable[[str], object] = connect,
        repository_factory: Callable[[object], object] | None = None,
    ) -> None:
        self._settings_loader = settings_loader
        self._connect = connect_func
        self._repository_factory = repository_factory

    def activate(self, *, run_id: int, dataset: str) -> dict:
        dataset_code = str(dataset).upper()
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            repository.activate_rag_vector_run(run_id=run_id, dataset=dataset_code)
            vector_base = repository.get_rag_vector_base_summary(dataset=dataset_code)
            runs = repository.list_rag_vector_runs(dataset=dataset_code, limit=20)
        finally:
            connection.close()
        return {
            "action": "activated",
            "dataset": dataset_code,
            "vector_base": vector_base,
            "runs": runs,
        }

    def delete(self, *, run_id: int, dataset: str) -> dict:
        dataset_code = str(dataset).upper()
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            repository.delete_rag_vector_run(run_id=run_id, dataset=dataset_code)
            vector_base = repository.get_rag_vector_base_summary(dataset=dataset_code)
            runs = repository.list_rag_vector_runs(dataset=dataset_code, limit=20)
        finally:
            connection.close()
        return {
            "action": "deleted",
            "dataset": dataset_code,
            "vector_base": vector_base,
            "runs": runs,
        }

    def _make_repository(self, connection: object) -> object:
        if self._repository_factory is not None:
            return self._repository_factory(connection)
        from .repositories import JudgeRepository

        return JudgeRepository(connection)
