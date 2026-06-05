"""Persist immutable AV3 RAG chunk snapshots for candidate answers."""

from __future__ import annotations

from typing import Any, Protocol

from .contracts import (
    CandidateAnswerContextChunkRecord,
    RagRetrievalResult,
    RetrievedRagChunk,
)
from .candidate_context_budget import budget_metadata_for_chunk

_UNSAFE_METADATA_KEYS = {
    "answerkey",
    "correctalternative",
    "correctoption",
    "correctanswer",
    "expectedanswer",
    "goldanswer",
    "goldreferenceanswer",
    "officialanswerkey",
    "referenceanswer",
    "rubric",
    "guideline",
    "judgeprompt",
    "judgerubric",
    "judgescore",
    "humanevaluation",
    "previousmodelranking",
    "previousmodelscore",
    "gabarito",
    "gabaritooficial",
    "alternativacorreta",
    "respostaouro",
    "respostaoficial",
    "rubrica",
    "diretriz",
    "promptjuiz",
    "rubricajuiz",
    "notajuiz",
    "avaliacaohumana",
    "rankingmodeloanterior",
    "notamodeloanterior",
}


class RagContextSnapshotRepositoryProtocol(Protocol):
    """Repository operations required to persist retrieval snapshots."""

    def persist_candidate_answer_context_chunks(
        self,
        *,
        candidate_answer_id: int,
        chunks: list[CandidateAnswerContextChunkRecord],
    ) -> list[CandidateAnswerContextChunkRecord]:
        """Replace the stored chunk snapshot set for one candidate answer."""


class RagContextSnapshotService:
    """Convert successful retrieval results into persisted snapshot rows."""

    def __init__(
        self,
        *,
        repository: RagContextSnapshotRepositoryProtocol,
    ) -> None:
        self._repository = repository

    def persist_retrieval_snapshot(
        self,
        *,
        candidate_answer_id: int,
        retrieval_result: RagRetrievalResult,
    ) -> list[CandidateAnswerContextChunkRecord]:
        if retrieval_result.status != "success":
            return []

        chunks = [
            _to_context_chunk_record(
                candidate_answer_id=candidate_answer_id,
                retrieval_result=retrieval_result,
                chunk=chunk,
            )
            for chunk in retrieval_result.chunks
        ]
        return self._repository.persist_candidate_answer_context_chunks(
            candidate_answer_id=candidate_answer_id,
            chunks=chunks,
        )


def _to_context_chunk_record(
    *,
    candidate_answer_id: int,
    retrieval_result: RagRetrievalResult,
    chunk: RetrievedRagChunk,
) -> CandidateAnswerContextChunkRecord:
    return CandidateAnswerContextChunkRecord(
        answer_context_chunk_id=None,
        candidate_answer_id=candidate_answer_id,
        chunk_id=chunk.chunk_id,
        rank=chunk.rank,
        chunk_text_snapshot=chunk.chunk_text,
        similarity_score=chunk.similarity,
        source_url=chunk.url,
        metadata=_build_snapshot_metadata(
            retrieval_result=retrieval_result,
            chunk=chunk,
        ),
    )


def _build_snapshot_metadata(
    *,
    retrieval_result: RagRetrievalResult,
    chunk: RetrievedRagChunk,
) -> dict[str, Any]:
    sanitized_chunk_metadata = _sanitize_metadata(chunk.metadata)
    snapshot_metadata = _compact_dict(
        {
            "dataset": retrieval_result.dataset,
            "retrieval_run_id": retrieval_result.retrieval_run_id,
            "retrieval_name": retrieval_result.retrieval_name,
            "embedding_model": retrieval_result.embedding_model,
            "top_k": retrieval_result.top_k,
            "source_kind": chunk.source_kind,
            "document_id": chunk.document_id,
            "document_key": chunk.document_key,
            "lei": chunk.lei,
            "norma": chunk.norma,
            "urn": chunk.urn,
            "artigo": chunk.artigo,
            "topico": chunk.topico,
            "relevancia": chunk.relevancia,
            "tipo": chunk.tipo,
            "distance": chunk.distance,
            "similarity": chunk.similarity,
        }
    )
    return {
        **sanitized_chunk_metadata,
        **snapshot_metadata,
        **budget_metadata_for_chunk(chunk),
    }


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _normalize_metadata_key(key)
            if normalized_key in _UNSAFE_METADATA_KEYS:
                continue
            sanitized_item = _sanitize_metadata(item)
            if sanitized_item is None:
                continue
            sanitized[str(key)] = sanitized_item
        return sanitized
    if isinstance(value, list):
        sanitized_list = []
        for item in value:
            sanitized_item = _sanitize_metadata(item)
            if sanitized_item is None:
                continue
            sanitized_list.append(sanitized_item)
        return sanitized_list
    return value


def _normalize_metadata_key(key: Any) -> str:
    return "".join(char for char in str(key).lower() if char.isalnum())


def _compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value is not None
    }
