"""RAG curation import, versioning, and UI service."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .config import load_settings
from .db import connect

DATASET_ALIASES = {
    "J1": "OAB_Bench",
    "J2": "OAB_Exames",
}


@dataclass(frozen=True)
class NormalizedRagCurationArticle:
    ordem: int
    artigo: str
    topico: str | None
    relevancia: str | None
    tipo: str | None


@dataclass(frozen=True)
class NormalizedRagCurationItem:
    dataset: str
    dataset_name: str
    question_id: int
    question_external_id: str
    question_sequence: int
    question_type: str
    prompt_system: str | None
    question_text: str
    answer_key: Any
    perguntas: Any
    alternativas: Any
    total_points: float | None
    difficulty_level: str | None
    difficulty_scale: int | None
    difficulty_criteria: list[str]
    discipline: str | None
    subject: str | None
    theme: str | None
    norma: str | None
    lei: str | None
    url: str | None
    urn: str | None
    curator: str | None
    classified_at: str | None
    metadata: dict[str, Any]
    raw_payload: dict[str, Any]
    payload_hash: str
    articles: list[NormalizedRagCurationArticle]


class RagCurationService:
    """Manage versioned AV3 RAG curation imports sourced from exported JSON."""

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

    def options(self) -> dict[str, Any]:
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            return {"datasets": [asdict(item) for item in repository.list_rag_curation_datasets()]}
        finally:
            connection.close()

    def get(self, *, dataset: str) -> dict[str, Any]:
        dataset_code = resolve_rag_curation_dataset(dataset)
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            active = repository.get_rag_curation_dataset_summary(dataset=dataset_code)
            vector_base = repository.get_rag_vector_base_summary(dataset=dataset_code)
            vector_runs = repository.list_rag_vector_runs(dataset=dataset_code, limit=20)
            items = repository.list_rag_curation_items(dataset=dataset_code, active_only=True)
            runs = repository.list_rag_curation_runs(dataset=dataset_code, limit=20)
        finally:
            connection.close()
        return {
            "dataset": dataset_code,
            "active": asdict(active) if active is not None else None,
            "vector_base": asdict(vector_base) if vector_base is not None else None,
            "vector_runs": [asdict(run) for run in vector_runs],
            "items": [asdict(item) for item in items],
            "runs": [asdict(run) for run in runs],
        }

    def detail(self, *, curation_id: int, dataset: str) -> dict[str, Any]:
        dataset_code = resolve_rag_curation_dataset(dataset)
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            detail = repository.get_rag_curation_detail(curation_id=curation_id, dataset=dataset_code)
        finally:
            connection.close()
        return {"detail": asdict(detail) if detail is not None else None}

    def activate_run(self, *, run_id: int, dataset: str) -> dict[str, Any]:
        dataset_code = resolve_rag_curation_dataset(dataset)
        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            repository.activate_rag_curation_run(run_id=run_id, dataset=dataset_code)
            active = repository.get_rag_curation_dataset_summary(dataset=dataset_code)
            vector_base = repository.get_rag_vector_base_summary(dataset=dataset_code)
            vector_runs = repository.list_rag_vector_runs(dataset=dataset_code, limit=20)
            items = repository.list_rag_curation_items(dataset=dataset_code, active_only=True)
            runs = repository.list_rag_curation_runs(dataset=dataset_code, limit=20)
        finally:
            connection.close()
        return {
            "dataset": dataset_code,
            "active": asdict(active) if active is not None else None,
            "vector_base": asdict(vector_base) if vector_base is not None else None,
            "vector_runs": [asdict(run) for run in vector_runs],
            "items": [asdict(item) for item in items],
            "runs": [asdict(run) for run in runs],
        }

    def import_json(self, *, filename: str, imported_by: str, raw_text: str) -> dict[str, Any]:
        imported_by = imported_by.strip()
        if not imported_by:
            raise ValueError("Informe quem está importando a curadoria.")
        if not filename.lower().endswith(".json"):
            raise ValueError("Selecione um arquivo .json.")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON inválido: {error.msg}.") from error
        if not isinstance(payload, list) or not payload:
            raise ValueError("O arquivo de curadoria deve conter uma lista JSON não vazia.")
        dataset_code = self._infer_dataset(payload)
        payload_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        connection = self._connect(self._settings_loader().database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            dataset_name = repository.get_dataset_name_for_code(dataset_code)
            if dataset_name is None:
                raise ValueError(f"Dataset não encontrado no banco para {dataset_code}.")
            existing = repository.get_rag_curation_run_by_hash(dataset=dataset_code, payload_hash=payload_hash)
            if existing is not None:
                repository.activate_rag_curation_run(run_id=existing.run_id, dataset=dataset_code)
                active = repository.get_rag_curation_dataset_summary(dataset=dataset_code)
                vector_base = repository.get_rag_vector_base_summary(dataset=dataset_code)
                vector_runs = repository.list_rag_vector_runs(dataset=dataset_code, limit=20)
                items = repository.list_rag_curation_items(dataset=dataset_code, active_only=True)
                runs = repository.list_rag_curation_runs(dataset=dataset_code, limit=20)
                return {
                    "action": "activated_existing",
                    "dataset": dataset_code,
                    "active": asdict(active) if active is not None else None,
                    "vector_base": asdict(vector_base) if vector_base is not None else None,
                    "vector_runs": [asdict(run) for run in vector_runs],
                    "items": [asdict(item) for item in items],
                    "runs": [asdict(run) for run in runs],
                }
            question_map = repository.list_question_sequence_map(dataset=dataset_code)
            normalized = [
                self._normalize_item(
                    item=item,
                    dataset=dataset_code,
                    dataset_name=dataset_name,
                    question_map=question_map,
                )
                for item in payload
            ]
            run = repository.create_rag_curation_import_run(
                dataset=dataset_code,
                dataset_name=dataset_name,
                filename=filename,
                payload_hash=payload_hash,
                imported_by=imported_by,
                items=normalized,
            )
            active = repository.get_rag_curation_dataset_summary(dataset=dataset_code)
            vector_base = repository.get_rag_vector_base_summary(dataset=dataset_code)
            vector_runs = repository.list_rag_vector_runs(dataset=dataset_code, limit=20)
            items = repository.list_rag_curation_items(dataset=dataset_code, active_only=True)
            runs = repository.list_rag_curation_runs(dataset=dataset_code, limit=20)
        finally:
            connection.close()
        return {
            "action": "imported",
            "dataset": dataset_code,
            "run": asdict(run),
            "active": asdict(active) if active is not None else None,
            "vector_base": asdict(vector_base) if vector_base is not None else None,
            "vector_runs": [asdict(run) for run in vector_runs],
            "items": [asdict(item) for item in items],
            "runs": [asdict(run) for run in runs],
        }

    def _make_repository(self, connection: Any) -> Any:
        if self._repository_factory is not None:
            return self._repository_factory(connection)
        from .repositories import JudgeRepository

        return JudgeRepository(connection)

    def _infer_dataset(self, payload: list[dict[str, Any]]) -> str:
        dataset_values = {
            str(item.get("metadados", {}).get("dataset", "")).strip().upper()
            for item in payload
            if isinstance(item.get("metadados"), dict)
        }
        dataset_values.discard("")
        if len(dataset_values) != 1:
            raise ValueError("O arquivo deve conter exatamente um dataset coerente em metadados.dataset.")
        dataset_code = next(iter(dataset_values))
        return resolve_rag_curation_dataset(dataset_code)

    def _normalize_item(
        self,
        *,
        item: dict[str, Any],
        dataset: str,
        dataset_name: str,
        question_map: dict[int, int],
    ) -> NormalizedRagCurationItem:
        required_root = {"id", "tipo_questao", "questao", "gabarito", "classificacao", "metadados"}
        missing_root = sorted(required_root - set(item.keys()))
        if missing_root:
            raise ValueError(f"Item de curadoria sem campos obrigatórios: {', '.join(missing_root)}.")
        metadata = item.get("metadados")
        if not isinstance(metadata, dict):
            raise ValueError("metadados deve ser um objeto JSON.")
        raw_dataset = str(metadata.get("dataset", "")).strip().upper()
        if resolve_rag_curation_dataset(raw_dataset) != dataset:
            raise ValueError("O arquivo mistura datasets ou contém dataset inconsistente.")
        question_sequence = metadata.get("numero_questao_sequencial")
        if not isinstance(question_sequence, int):
            raise ValueError("metadados.numero_questao_sequencial deve ser inteiro.")
        question_id = question_map.get(question_sequence)
        if question_id is None:
            raise ValueError(
                f"Questão {dataset}/{question_sequence} não encontrada no banco local para reconciliação."
            )
        classification = item.get("classificacao")
        if not isinstance(classification, dict):
            raise ValueError("classificacao deve ser um objeto JSON.")
        difficulty = classification.get("dificuldade", {})
        specialty = classification.get("especialidade", {})
        legislation = classification.get("legislacao", {})
        if not isinstance(difficulty, dict) or not isinstance(specialty, dict) or not isinstance(legislation, dict):
            raise ValueError("dificuldade, especialidade e legislacao devem ser objetos JSON.")
        articles_payload = legislation.get("artigos", [])
        if not isinstance(articles_payload, list):
            raise ValueError("classificacao.legislacao.artigos deve ser uma lista.")
        articles: list[NormalizedRagCurationArticle] = []
        for index, article in enumerate(articles_payload, start=1):
            if not isinstance(article, dict):
                raise ValueError("Cada item de classificacao.legislacao.artigos deve ser um objeto.")
            article_text = str(article.get("artigo", "")).strip()
            if not article_text:
                raise ValueError("Cada artigo curado deve informar o campo artigo.")
            articles.append(
                NormalizedRagCurationArticle(
                    ordem=index,
                    artigo=article_text,
                    topico=_normalize_optional_text(article.get("topico")),
                    relevancia=_normalize_optional_text(article.get("relevancia")),
                    tipo=_normalize_optional_text(article.get("tipo")),
                )
            )
        question_external_id = str(item.get("id", "")).strip()
        if not question_external_id:
            raise ValueError("Cada item deve informar um id externo.")
        question_text = str(item.get("questao", "")).strip()
        if not question_text:
            raise ValueError("Cada item deve informar o enunciado da questão.")
        return NormalizedRagCurationItem(
            dataset=dataset,
            dataset_name=dataset_name,
            question_id=question_id,
            question_external_id=question_external_id,
            question_sequence=question_sequence,
            question_type=str(item.get("tipo_questao", "")).strip(),
            prompt_system=_normalize_optional_text(item.get("prompt_system")),
            question_text=question_text,
            answer_key=item.get("gabarito"),
            perguntas=item.get("perguntas"),
            alternativas=item.get("alternativas"),
            total_points=_normalize_optional_float(item.get("pontuacao_total")),
            difficulty_level=_normalize_optional_text(difficulty.get("nivel")),
            difficulty_scale=_normalize_optional_int(difficulty.get("escala")),
            difficulty_criteria=_normalize_optional_string_list(difficulty.get("criterios")),
            discipline=_normalize_optional_text(specialty.get("disciplina")),
            subject=_normalize_optional_text(specialty.get("assunto")),
            theme=_normalize_optional_text(specialty.get("tema")),
            norma=_normalize_optional_text(legislation.get("norma")),
            lei=_normalize_optional_text(legislation.get("lei")),
            url=_normalize_optional_text(legislation.get("url")),
            urn=_normalize_optional_text(legislation.get("urn")),
            curator=_normalize_optional_text(classification.get("curador")),
            classified_at=_normalize_optional_text(classification.get("dt_classificacao")),
            metadata=metadata,
            raw_payload=item,
            payload_hash=hashlib.sha256(
                json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            articles=articles,
        )


def resolve_rag_curation_dataset(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in DATASET_ALIASES:
        return normalized
    reverse = {target.upper(): source for source, target in DATASET_ALIASES.items()}
    if normalized in reverse:
        return reverse[normalized]
    raise ValueError(f"Dataset de curadoria inválido: {value}.")


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except ValueError as error:
        raise ValueError(f"Valor inteiro inválido: {value!r}.") from error


def _normalize_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError as error:
        raise ValueError(f"Valor numérico inválido: {value!r}.") from error


def _normalize_optional_string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("Lista de critérios deve ser um array JSON.")
    normalized = []
    for item in value:
        text = _normalize_optional_text(item)
        if text is not None:
            normalized.append(text)
    return normalized
