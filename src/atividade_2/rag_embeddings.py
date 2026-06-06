"""Embedding generation pipeline for the materialized AV3 RAG base."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .config import load_settings
from .db import connect
from .rag_curation import resolve_rag_curation_dataset
from .rag_embedding_client import request_openai_compatible_embeddings
from .rag_source_fetch import SourceUrlFetchReport, fetch_source_url_contents, split_source_urls


EXPECTED_RETRIEVAL_STRATEGY = "source_url_only_v2"


class RagEmbeddingGenerationService:
    """Generate embeddings for active materialized RAG chunks."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], Any] = load_settings,
        connect_func: Callable[[str], Any] = connect,
        repository_factory: Callable[[Any], Any] | None = None,
        source_fetcher: Callable[[list[str]], SourceUrlFetchReport] = fetch_source_url_contents,
        request_timeout_seconds: int = 90,
        batch_size: int = 32,
    ) -> None:
        self._settings_loader = settings_loader
        self._connect = connect_func
        self._repository_factory = repository_factory
        self._source_fetcher = source_fetcher
        self._request_timeout_seconds = request_timeout_seconds
        self._batch_size = batch_size

    def run(
        self,
        *,
        dataset: str,
        batch_size: int | None = None,
        question_sequence_start: int | None = None,
        question_sequence_end: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        dataset_code = resolve_rag_curation_dataset(dataset)
        question_sequence_start, question_sequence_end = _normalize_question_range(
            question_sequence_start,
            question_sequence_end,
        )

        def emit(message: str, *, state: str = "running", **metadata: Any) -> None:
            if progress_callback is not None:
                progress_callback({"message": message, "state": state, **metadata})

        range_label = _format_question_range(question_sequence_start, question_sequence_end)
        emit(f"Iniciando geracao de embeddings para {dataset_code}{range_label}.")
        settings = self._settings_loader()
        api_key = getattr(settings, "embedding_api_key", None)
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY is required to generate RAG embeddings.")

        connection = self._connect(settings.database_url)
        try:
            repository = self._make_repository(connection)
            repository.ensure_schema()
            emit("Schema validado.")
            config = repository.get_rag_embedding_model_config(dataset=dataset_code)
            if config is None:
                raise RuntimeError(f"Nenhuma configuracao de embedding encontrada para {dataset_code}.")
            emit(f"Modelo de embedding carregado: {config.model_name}.")
            materialized_base = False
            vector_base_trace: dict[str, Any] = {"action": "reused"}
            vector_base_summary = repository.get_rag_vector_base_summary(dataset=dataset_code)
            if vector_base_summary is None:
                emit("Nenhuma base vetorial ativa encontrada; materializando curadoria ativa.")
                repository.materialize_rag_base_from_active_curation(dataset=dataset_code)
                materialized_base = True
                vector_base_trace = {"action": "created_missing"}
                emit("Base vetorial materializada.", state="done")
            elif not vector_base_summary.matches_active_curation:
                previous_import_run_id = vector_base_summary.import_run_id
                active_curation_run_id = vector_base_summary.active_curation_run_id
                emit(
                    "Base vetorial ativa desatualizada; materializando curadoria ativa "
                    f"(base atual import_run_id={previous_import_run_id}, "
                    f"curadoria ativa import_run_id={active_curation_run_id}).",
                    previous_import_run_id=previous_import_run_id,
                    active_curation_run_id=active_curation_run_id,
                )
                repository.materialize_rag_base_from_active_curation(dataset=dataset_code)
                materialized_base = True
                vector_base_trace = {
                    "action": "refreshed_stale",
                    "previous_import_run_id": previous_import_run_id,
                    "active_curation_run_id": active_curation_run_id,
                    "previous_retrieval_run_id": vector_base_summary.retrieval_run_id,
                }
                emit("Base vetorial rematerializada a partir da curadoria ativa.", state="done")
            elif _is_legacy_vector_base(vector_base_summary):
                emit(
                    "Base vetorial ativa usa estrategia anterior; rematerializando com a versao atual.",
                    previous_retrieval_run_id=vector_base_summary.retrieval_run_id,
                    previous_retrieval_name=vector_base_summary.retrieval_name,
                    previous_retrieval_strategy=vector_base_summary.retrieval_strategy,
                    expected_retrieval_strategy=EXPECTED_RETRIEVAL_STRATEGY,
                )
                repository.materialize_rag_base_from_active_curation(dataset=dataset_code)
                materialized_base = True
                vector_base_trace = {
                    "action": "refreshed_legacy_strategy",
                    "previous_retrieval_run_id": vector_base_summary.retrieval_run_id,
                    "previous_retrieval_name": vector_base_summary.retrieval_name,
                    "previous_retrieval_strategy": vector_base_summary.retrieval_strategy,
                    "expected_retrieval_strategy": EXPECTED_RETRIEVAL_STRATEGY,
                }
                emit("Base vetorial rematerializada com a estrategia atual.", state="done")
            else:
                emit("Base vetorial ativa encontrada.")
            requested_question_sequence_start = question_sequence_start
            requested_question_sequence_end = question_sequence_end
            if question_sequence_start is not None or question_sequence_end is not None:
                resolved_range = repository.resolve_rag_question_sequence_range_for_active_vector_base(
                    dataset=dataset_code,
                    question_sequence_start=question_sequence_start,
                    question_sequence_end=question_sequence_end,
                )
                question_sequence_start = resolved_range["start"]
                question_sequence_end = resolved_range["end"]
                if resolved_range["mapped_from_dataset_position"]:
                    emit(
                        "Intervalo informado interpretado como posicao local do dataset: "
                        f"{requested_question_sequence_start or 1}-{requested_question_sequence_end or 'fim'} "
                        f"-> sequencias reais {question_sequence_start}-{question_sequence_end}.",
                        state="done",
                    )
                else:
                    emit(
                        f"Intervalo de sequencias reais confirmado: {question_sequence_start or 1}-"
                        f"{question_sequence_end or 'fim'}.",
                        state="done",
                    )
            else:
                requested_question_sequence_start = None
                requested_question_sequence_end = None
            source_documents = repository.list_rag_source_documents_for_active_vector_base(
                dataset=dataset_code,
                question_sequence_start=question_sequence_start,
                question_sequence_end=question_sequence_end,
            )
            emit(f"{len(source_documents)} documento(s) com URL de fonte no escopo selecionado.")
            for document in source_documents[:20]:
                emit(
                    "Documento fonte selecionado: "
                    f"id_document={document.get('document_id')}, "
                    f"document_key={_short_value(document.get('document_key'))}, "
                    f"titulo={_short_value(document.get('title'), 80)}, "
                    f"tipo={_short_value(document.get('source_type'))}.",
                )
            if len(source_documents) > 20:
                emit(f"{len(source_documents) - 20} documento(s) adicional(is) omitido(s) no log detalhado.")
            source_document_urls = [
                {**document, "url": url, "canonical_url_key": _canonical_source_url_key(url)}
                for document in source_documents
                for url in split_source_urls([str(document["url"])])
            ]
            source_url_by_key: dict[str, str] = {}
            for item in source_document_urls:
                key = str(item["canonical_url_key"])
                url = str(item["url"])
                current_url = source_url_by_key.get(key)
                source_url_by_key[key] = _preferred_fetch_url(current=current_url, candidate=url)
            source_urls = list(source_url_by_key.values())
            duplicate_url_count = max(0, len(source_document_urls) - len(source_urls))
            emit(
                f"{len(source_document_urls)} referencia(s) de URL extraida(s); "
                f"{len(source_urls)} URL(s) unica(s) selecionada(s) para consulta; "
                f"{duplicate_url_count} duplicada(s) por normalizacao de URL.",
                state="done" if duplicate_url_count else "running",
            )
            for item in source_document_urls[:30]:
                fetch_url = source_url_by_key[str(item["canonical_url_key"])]
                fetch_suffix = f", consulta={_short_value(fetch_url, 140)}" if fetch_url != str(item["url"]) else ""
                emit(
                    "URL preparada: "
                    f"id_document={item.get('document_id')}, "
                    f"document_key={_short_value(item.get('document_key'))}, "
                    f"url={_short_value(item.get('url'), 140)}, "
                    f"normalizada={_short_value(item.get('canonical_url_key'), 140)}"
                    f"{fetch_suffix}.",
                )
            if len(source_document_urls) > 30:
                emit(f"{len(source_document_urls) - 30} URL(s) adicional(is) omitida(s) no log detalhado.")
            source_report = self._source_fetcher(source_urls) if source_urls else SourceUrlFetchReport([], [])
            if source_urls:
                emit(
                    f"Fontes consultadas: {len(source_report.successes)} sucesso(s), "
                    f"{len(source_report.failures)} falha(s).",
                    state="error" if source_report.failures else "done",
                )
                for success in source_report.successes[:20]:
                    emit(
                        "Fonte recuperada: "
                        f"url={_short_value(success.url, 140)}, "
                        f"content_type={_short_value(success.content_type)}, "
                        f"chars={len(success.content)}.",
                        state="done",
                    )
                for failure in source_report.failures[:20]:
                    emit(
                        "Falha ao recuperar fonte: "
                        f"url={_short_value(failure.url, 140)}, "
                        f"motivo={_short_value(failure.reason, 140)}.",
                        state="error",
                    )
            source_contents_by_url = {_canonical_source_url_key(item.url): item for item in source_report.successes}
            source_contents = []
            source_content_keys_seen: set[str] = set()
            for document in source_document_urls:
                canonical_url_key = str(document["canonical_url_key"])
                fetched_content = source_contents_by_url.get(canonical_url_key)
                if fetched_content is None or canonical_url_key in source_content_keys_seen:
                    continue
                source_content_keys_seen.add(canonical_url_key)
                source_contents.append(
                    {
                        **document,
                        "url": fetched_content.url,
                        "original_url": document["url"],
                        "content": fetched_content.content,
                        "content_type": fetched_content.content_type,
                    }
                )
            for item in source_contents[:20]:
                emit(
                    "Conteudo de fonte pronto para chunking: "
                    f"id_document={item.get('document_id')}, "
                    f"document_key={_short_value(item.get('document_key'))}, "
                    f"url={_short_value(item.get('url'), 140)}, "
                    f"normalizada={_short_value(item.get('canonical_url_key'), 140)}, "
                    f"chars={len(str(item.get('content') or ''))}.",
                )
            source_chunk_count = repository.replace_rag_source_content_chunks_for_active_vector_base(
                dataset=dataset_code,
                source_contents=source_contents,
                question_sequence_start=question_sequence_start,
                question_sequence_end=question_sequence_end,
            )
            emit(f"{source_chunk_count} chunk(s) de fonte recuperada gravado(s).", state="done")
            chunks = repository.list_rag_chunks_for_active_vector_base(
                dataset=dataset_code,
                question_sequence_start=question_sequence_start,
                question_sequence_end=question_sequence_end,
            )
            if not chunks:
                raise RuntimeError(f"Nenhum chunk materializado encontrado para {dataset_code}.")
            chunk_stats = _chunk_stats(chunks)
            emit(
                f"{len(chunks)} chunk(s) selecionado(s) para embedding: "
                f"{_format_chunk_stats(chunk_stats)}"
            )
            for chunk in chunks[:20]:
                emit(
                    "Chunk selecionado: "
                    f"id_chunk={chunk.get('chunk_id')}, "
                    f"id_document={chunk.get('document_id')}, "
                    f"document_key={_short_value(chunk.get('document_key'))}, "
                    f"kind={_short_value(chunk.get('source_kind'))}, "
                    f"index={chunk.get('chunk_index')}, "
                    f"hash={_short_value(chunk.get('content_hash'), 12)}, "
                    f"chars={len(str(chunk.get('chunk_text') or ''))}.",
                )
            if len(chunks) > 20:
                emit(f"{len(chunks) - 20} chunk(s) adicional(is) omitido(s) no log detalhado.")

            effective_batch_size = max(1, int(batch_size or self._batch_size))
            generated_count = 0
            total_latency_ms = 0
            emit("Limpando embeddings antigos do escopo selecionado antes da geracao incremental.")
            repository.clear_rag_embeddings_for_active_vector_base(
                dataset=dataset_code,
                embedding_model=config.model_name,
                question_sequence_start=question_sequence_start,
                question_sequence_end=question_sequence_end,
            )
            emit("Escopo limpo; embeddings serao gravados incrementalmente por lote.", state="done")
            batches = _chunked(chunks, effective_batch_size)
            for batch_index, chunk_batch in enumerate(batches, start=1):
                chunk_ids = [int(item["chunk_id"]) for item in chunk_batch]
                batch_kinds = _format_chunk_stats(_chunk_stats(chunk_batch))
                batch_characters = sum(len(str(item["chunk_text"] or "")) for item in chunk_batch)
                emit(
                    f"Enviando lote {batch_index}/{len(batches)} com {len(chunk_batch)} chunk(s): "
                    f"id_chunk {min(chunk_ids)}-{max(chunk_ids)}. {batch_kinds}",
                    batch_index=batch_index,
                    batch_count=len(batches),
                    generated_embeddings=generated_count,
                    total_chunks=len(chunks),
                )
                texts = [str(item["chunk_text"]) for item in chunk_batch]
                try:
                    batch_result = request_openai_compatible_embeddings(
                        api_base_url=(config.api_base_url or "").strip() or "https://api.openai.com/v1",
                        api_key=api_key,
                        model_name=config.model_name,
                        texts=texts,
                        dimensions=config.dimensions,
                        timeout_seconds=self._request_timeout_seconds,
                    )
                except Exception as error:
                    emit(
                        f"Falha no lote {batch_index}/{len(batches)}: id_chunk {min(chunk_ids)}-{max(chunk_ids)}, "
                        f"{len(chunk_batch)} chunk(s), {batch_characters} caractere(s), "
                        f"{batch_kinds} provider={config.provider}, modelo={config.model_name}, "
                        f"api_base={_short_value(config.api_base_url, 120)}. Erro: {error}",
                        state="error",
                        batch_index=batch_index,
                        batch_count=len(batches),
                        generated_embeddings=generated_count,
                        total_chunks=len(chunks),
                        chunk_id_start=min(chunk_ids),
                        chunk_id_end=max(chunk_ids),
                        chunk_characters=batch_characters,
                        chunk_kind_summary=batch_kinds,
                        provider=config.provider,
                        model_name=config.model_name,
                        api_base_url=config.api_base_url,
                        error=str(error),
                    )
                    raise RuntimeError(
                        f"Falha ao gerar embeddings no lote {batch_index}/{len(batches)} "
                        f"(id_chunk {min(chunk_ids)}-{max(chunk_ids)}): {error}"
                    ) from error
                total_latency_ms += batch_result.latency_ms
                generated_batch = [
                    {
                        "chunk_id": int(item["chunk_id"]),
                        "embedding": vector,
                    }
                    for item, vector in zip(chunk_batch, batch_result.vectors, strict=True)
                ]
                repository.upsert_rag_embedding_batch_for_active_vector_base(
                    dataset=dataset_code,
                    embedding_model=config.model_name,
                    embedding_dimensions=config.dimensions,
                    provider=config.provider,
                    api_base_url=config.api_base_url,
                    embeddings=generated_batch,
                )
                generated_count += len(generated_batch)
                returned_dimensions = len(batch_result.vectors[0]) if batch_result.vectors else 0
                emit(
                    f"Lote {batch_index}/{len(batches)} concluido em {batch_result.latency_ms} ms; "
                    f"{len(batch_result.vectors)} embedding(s), {returned_dimensions} dimensao(oes), "
                    f"total gravado ate agora={generated_count}.",
                    state="done",
                    batch_index=batch_index,
                    batch_count=len(batches),
                    generated_embeddings=generated_count,
                    total_chunks=len(chunks),
                )

            emit("Consolidando resumo final da geracao incremental.")
            summary = repository.build_rag_embedding_generation_summary(
                dataset=dataset_code,
                embedding_model=config.model_name,
                embedding_dimensions=config.dimensions,
                provider=config.provider,
                api_base_url=config.api_base_url,
                generated_embeddings=generated_count,
                latency_ms=total_latency_ms,
            )
            emit(f"{summary.generated_embeddings} embedding(s) gravado(s).", state="done")
        finally:
            connection.close()
        return {
            "summary": summary,
            "materialized_base": materialized_base,
            "vector_base_trace": vector_base_trace,
            "question_sequence_range": {
                "requested_start": requested_question_sequence_start,
                "requested_end": requested_question_sequence_end,
                "effective_start": question_sequence_start,
                "effective_end": question_sequence_end,
            },
            "source_url_summary": {
                "references": len(source_document_urls),
                "attempted": len(source_urls),
                "deduplicated": duplicate_url_count,
                "succeeded": len(source_report.successes),
                "failed": len(source_report.failures),
                "inserted_chunks": source_chunk_count,
                "failures": [{"url": item.url, "reason": item.reason} for item in source_report.failures],
            },
            "chunk_summary": {
                "total": len(chunks),
                "by_source_kind": chunk_stats,
                "curation_chunks": _curation_chunk_count(chunk_stats),
                "source_url_chunks": chunk_stats.get("source_url_content", 0),
            },
        }

    def _make_repository(self, connection: Any) -> Any:
        if self._repository_factory is not None:
            return self._repository_factory(connection)
        from .repositories import JudgeRepository

        return JudgeRepository(connection)


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _is_legacy_vector_base(vector_base_summary: Any) -> bool:
    return (
        vector_base_summary.retrieval_strategy != EXPECTED_RETRIEVAL_STRATEGY
        or str(vector_base_summary.retrieval_name).endswith("_source_urls_v1")
    )


def _normalize_question_range(
    question_sequence_start: int | None,
    question_sequence_end: int | None,
) -> tuple[int | None, int | None]:
    start = int(question_sequence_start) if question_sequence_start is not None else None
    end = int(question_sequence_end) if question_sequence_end is not None else None
    if start is not None and start < 1:
        raise ValueError("A questao inicial deve ser maior ou igual a 1.")
    if end is not None and end < 1:
        raise ValueError("A questao final deve ser maior ou igual a 1.")
    if start is not None and end is not None and start > end:
        raise ValueError("A questao inicial nao pode ser maior que a questao final.")
    return start, end


def _format_question_range(question_sequence_start: int | None, question_sequence_end: int | None) -> str:
    if question_sequence_start is None and question_sequence_end is None:
        return ""
    if question_sequence_start is not None and question_sequence_end is not None:
        return f" no intervalo de questoes {question_sequence_start}-{question_sequence_end}"
    if question_sequence_start is not None:
        return f" a partir da questao {question_sequence_start}"
    return f" ate a questao {question_sequence_end}"


def _chunk_stats(chunks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        kind = str(chunk.get("source_kind") or "desconhecido")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _format_chunk_stats(counts: dict[str, int]) -> str:
    if not counts:
        return "sem chunks."
    friendly_parts = []
    for kind, count in sorted(counts.items()):
        friendly_parts.append(f"{_source_kind_label(kind)}={count}")
    return "origem dos chunks: " + ", ".join(friendly_parts) + "."


def _curation_chunk_count(counts: dict[str, int]) -> int:
    return sum(count for kind, count in counts.items() if kind != "source_url_content")


def _source_kind_label(kind: str) -> str:
    labels = {
        "curated_article": "curadoria/artigos curados",
        "curation_summary": "curadoria/resumo",
        "source_url_content": "fonte/URL recuperada",
    }
    return labels.get(kind, kind)


def _canonical_source_url_key(url: str) -> str:
    parsed = urlsplit(url.strip())
    host = parsed.hostname.lower() if parsed.hostname else parsed.netloc.lower()
    port = ""
    if parsed.port is not None and parsed.port not in {80, 443}:
        port = f":{parsed.port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{port}{path}{query}"


def _preferred_fetch_url(*, current: str | None, candidate: str) -> str:
    if current is None:
        return _normalized_fetch_url(candidate)
    current_scheme = urlsplit(current).scheme.lower()
    candidate_scheme = urlsplit(candidate).scheme.lower()
    if current_scheme == "http" and candidate_scheme == "https":
        return _normalized_fetch_url(candidate)
    return current


def _normalized_fetch_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def _short_value(value: Any, max_length: int = 32) -> str:
    text = str(value or "-")
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
