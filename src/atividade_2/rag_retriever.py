"""Focused AV3 retrieval service for loading top-k RAG chunks by question."""

from __future__ import annotations

from typing import Any, Protocol

from .contracts import RagRetrievalQuestion, RagRetrievalResult, RetrievedRagChunk
from .rag_curation import resolve_rag_curation_dataset

DEFAULT_RAG_TOP_K = 5


class RagRetrieverRepositoryProtocol(Protocol):
    """Repository operations required for candidate-safe RAG retrieval."""

    def get_question_for_rag_retrieval(
        self,
        *,
        question_id: int,
        dataset: str,
    ) -> RagRetrievalQuestion | None:
        """Return candidate-safe question text for retrieval."""

    def get_rag_vector_base_summary(self, *, dataset: str) -> Any | None:
        """Return the active AV3 vector-base summary for one dataset."""

    def search_rag_chunks_by_embedding(
        self,
        *,
        dataset: str,
        embedding_model: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Perform vector search over the active AV3 vector base."""


class QuestionEmbeddingProviderProtocol(Protocol):
    """Question-embedding provider used by the retriever service."""

    def embed_query(self, text: str, *, model: str) -> list[float]:
        """Return one embedding vector for the provided question text."""


class RagRetrieverService:
    """Retrieve top-k RAG chunks for one question without persistence side effects."""

    def __init__(
        self,
        *,
        repository: RagRetrieverRepositoryProtocol,
        embedding_provider: QuestionEmbeddingProviderProtocol,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider

    def retrieve_for_question(
        self,
        *,
        question_id: int,
        dataset: str,
        top_k: int | None = None,
    ) -> RagRetrievalResult:
        dataset_code = resolve_rag_curation_dataset(dataset)
        requested_top_k = _normalize_top_k(top_k)
        fallback_top_k = requested_top_k or DEFAULT_RAG_TOP_K

        question = self._repository.get_question_for_rag_retrieval(
            question_id=question_id,
            dataset=dataset_code,
        )
        if question is None:
            return RagRetrievalResult(
                question_id=question_id,
                dataset=dataset_code,
                retrieval_run_id=None,
                retrieval_name=None,
                embedding_model=None,
                top_k=fallback_top_k,
                status="question_not_found",
            )

        vector_base = self._repository.get_rag_vector_base_summary(dataset=dataset_code)
        if vector_base is None:
            return RagRetrievalResult(
                question_id=question.question_id,
                dataset=dataset_code,
                retrieval_run_id=None,
                retrieval_name=None,
                embedding_model=None,
                top_k=fallback_top_k,
                status="vector_base_not_found",
            )

        embedding_model = (vector_base.embedding_model or "").strip() or None
        resolved_top_k = requested_top_k or max(1, int(vector_base.top_k or DEFAULT_RAG_TOP_K))
        if embedding_model is None:
            return RagRetrievalResult(
                question_id=question.question_id,
                dataset=dataset_code,
                retrieval_run_id=vector_base.retrieval_run_id,
                retrieval_name=vector_base.retrieval_name,
                embedding_model=None,
                top_k=resolved_top_k,
                status="embedding_model_not_configured",
            )

        query_vector = self._embedding_provider.embed_query(
            question.question_text,
            model=embedding_model,
        )
        raw_chunks = self._repository.search_rag_chunks_by_embedding(
            dataset=dataset_code,
            embedding_model=embedding_model,
            query_vector=query_vector,
            top_k=resolved_top_k,
        )
        chunks = _build_chunks(raw_chunks)
        status = "success" if chunks else "no_chunks_found"
        return RagRetrievalResult(
            question_id=question.question_id,
            dataset=dataset_code,
            retrieval_run_id=vector_base.retrieval_run_id,
            retrieval_name=vector_base.retrieval_name,
            embedding_model=embedding_model,
            top_k=resolved_top_k,
            status=status,
            chunks=chunks,
        )


def _normalize_top_k(top_k: int | None) -> int | None:
    if top_k is None:
        return None
    normalized = int(top_k)
    if normalized <= 0:
        raise ValueError("top_k must be greater than zero.")
    return normalized


def _build_chunks(raw_chunks: list[dict[str, Any]]) -> list[RetrievedRagChunk]:
    chunks: list[RetrievedRagChunk] = []
    for raw in sorted(raw_chunks, key=lambda item: (int(item.get("rank", 0)), int(item.get("chunk_id", 0)))):
        metadata = {
            key: value
            for key, value in raw.items()
            if key
            not in {
                "rank",
                "chunk_id",
                "chunk_text",
                "source_kind",
                "chunk_kind",
                "document_id",
                "document_key",
                "lei",
                "norma",
                "url",
                "urn",
                "artigo",
                "topico",
                "relevancia",
                "tipo",
                "distance",
                "similarity",
            }
        }
        chunks.append(
            RetrievedRagChunk(
                rank=int(raw["rank"]),
                chunk_id=int(raw["chunk_id"]),
                chunk_text=str(raw["chunk_text"]),
                source_kind=_optional_text(raw.get("source_kind") or raw.get("chunk_kind")),
                document_id=_optional_int(raw.get("document_id")),
                document_key=_optional_text(raw.get("document_key")),
                lei=_optional_text(raw.get("lei")),
                norma=_optional_text(raw.get("norma")),
                url=_optional_text(raw.get("url")),
                urn=_optional_text(raw.get("urn")),
                artigo=_optional_text(raw.get("artigo")),
                topico=_optional_text(raw.get("topico")),
                relevancia=_optional_text(raw.get("relevancia")),
                tipo=_optional_text(raw.get("tipo")),
                distance=_optional_float(raw.get("distance")),
                similarity=_optional_float(raw.get("similarity")),
                metadata=metadata,
            )
        )
    return chunks


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
