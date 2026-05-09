"""Meta-Avaliacao persistence and UI service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from .config import load_settings
from .db import connect


class MetaEvaluationService:
    """Manage human meta-evaluation records for J1 judge evaluations."""

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

    def options(self) -> dict[str, list[dict[str, Any]]]:
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            return {"evaluations": repository.list_meta_evaluation_targets(dataset="J1")}
        finally:
            connection.close()

    def get(self, *, evaluation_id: int) -> dict[str, Any]:
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            subject = repository.get_meta_evaluation_subject(evaluation_id=evaluation_id, dataset="J1")
            records = repository.list_meta_evaluations(evaluation_id=evaluation_id)
        finally:
            connection.close()
        return {
            "subject": asdict(subject) if subject is not None else None,
            "records": [asdict(record) for record in records],
        }

    def history(self) -> dict[str, list[dict[str, Any]]]:
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            records = repository.list_meta_evaluation_history(dataset="J1")
        finally:
            connection.close()
        return {"records": [asdict(record) for record in records]}

    def save(
        self,
        *,
        meta_evaluation_id: int | None,
        evaluation_id: int,
        evaluator_name: str,
        score: int,
        rationale: str,
    ) -> dict[str, Any]:
        evaluator_name = evaluator_name.strip()
        rationale = rationale.strip()
        if not evaluator_name:
            raise ValueError("Informe o nome do avaliador.")
        if not rationale:
            raise ValueError("Informe a justificativa da meta-avaliacao.")
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            subject = repository.get_meta_evaluation_subject(evaluation_id=evaluation_id, dataset="J1")
            if subject is None:
                raise ValueError("Avaliacao J1 nao encontrada para meta-avaliacao.")
            if meta_evaluation_id is None:
                saved = repository.create_meta_evaluation(
                    evaluation_id=evaluation_id,
                    evaluator_name=evaluator_name,
                    score=score,
                    rationale=rationale,
                )
                action = "created"
            else:
                saved = repository.update_meta_evaluation(
                    meta_evaluation_id=meta_evaluation_id,
                    evaluation_id=evaluation_id,
                    evaluator_name=evaluator_name,
                    score=score,
                    rationale=rationale,
                )
                action = "updated"
            records = repository.list_meta_evaluations(evaluation_id=evaluation_id)
        finally:
            connection.close()
        return {
            "action": action,
            "record": asdict(saved),
            "subject": asdict(subject) if subject is not None else None,
            "records": [asdict(record) for record in records],
        }

    def delete(self, *, meta_evaluation_id: int, evaluation_id: int) -> dict[str, Any]:
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            subject = repository.get_meta_evaluation_subject(evaluation_id=evaluation_id, dataset="J1")
            if subject is None:
                raise ValueError("Avaliacao J1 nao encontrada para meta-avaliacao.")
            repository.delete_meta_evaluation(meta_evaluation_id=meta_evaluation_id, evaluation_id=evaluation_id)
            records = repository.list_meta_evaluations(evaluation_id=evaluation_id)
        finally:
            connection.close()
        return {
            "action": "deleted",
            "subject": asdict(subject),
            "records": [asdict(record) for record in records],
        }

    def _make_repository(self, connection: Any) -> Any:
        if self._repository_factory is not None:
            return self._repository_factory(connection)
        from .repositories import JudgeRepository

        return JudgeRepository(connection)
