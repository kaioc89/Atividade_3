"""Tests for question-scoped AV3 RAG retrieval."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from atividade_2.contracts import RagRetrievalQuestion, RagVectorBaseSummary
from atividade_2.rag_retriever import RagRetrieverService


class FakeEmbeddingProvider:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.11, 0.22, 0.33]
        self.calls: list[tuple[str, str]] = []

    def embed_query(self, text: str, *, model: str) -> list[float]:
        self.calls.append((text, model))
        return list(self.vector)


class FakeRepository:
    def __init__(self) -> None:
        self.questions: dict[tuple[str, int], RagRetrievalQuestion] = {}
        self.vector_bases: dict[str, RagVectorBaseSummary | None] = {}
        self.search_results: dict[str, list[dict[str, object]]] = {}
        self.calls: list[tuple[str, object]] = []

    def get_question_for_rag_retrieval(
        self,
        *,
        question_id: int,
        dataset: str,
    ) -> RagRetrievalQuestion | None:
        self.calls.append(("get_question", dataset, question_id))
        return self.questions.get((dataset, question_id))

    def get_rag_vector_base_summary(self, *, dataset: str) -> RagVectorBaseSummary | None:
        self.calls.append(("get_vector_base", dataset))
        return self.vector_bases.get(dataset)

    def search_rag_chunks_by_embedding(
        self,
        *,
        dataset: str,
        embedding_model: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        self.calls.append(("search", dataset, embedding_model, list(query_vector), top_k))
        return list(self.search_results.get(dataset, []))


def test_retrieve_for_question_works_for_j1() -> None:
    repository = FakeRepository()
    provider = FakeEmbeddingProvider()
    repository.questions[("J1", 101)] = RagRetrievalQuestion(
        question_id=101,
        dataset="J1",
        question_text="Texto original da questao J1.",
    )
    repository.vector_bases["J1"] = _vector_base(dataset="J1", embedding_model="text-embedding-3-small", top_k=4)
    repository.search_results["J1"] = [
        {
            "rank": 2,
            "chunk_id": 501,
            "chunk_text": "Segundo chunk.",
            "chunk_kind": "source_url_content",
            "document_id": 801,
            "document_key": "lei-1",
            "lei": "Lei 8.112",
            "norma": "Estatuto",
            "url": "https://example.test/lei-1",
            "urn": "urn:lei:1",
            "artigo": "Art. 10",
            "topico": "Tema B",
            "relevancia": "alta",
            "tipo": "lei",
            "distance": 0.18,
            "similarity": 0.82,
            "sequence": 22,
        },
        {
            "rank": 1,
            "chunk_id": 500,
            "chunk_text": "Primeiro chunk.",
            "chunk_kind": "source_url_content",
            "document_id": 800,
            "document_key": "lei-0",
            "lei": "Lei 8.666",
            "norma": "Licitações",
            "url": "https://example.test/lei-0",
            "urn": "urn:lei:0",
            "artigo": "Art. 5",
            "topico": "Tema A",
            "relevancia": "alta",
            "tipo": "lei",
            "distance": 0.12,
            "similarity": 0.88,
            "sequence": 21,
        },
    ]

    result = RagRetrieverService(repository=repository, embedding_provider=provider).retrieve_for_question(
        question_id=101,
        dataset="j1",
    )

    assert result.status == "success"
    assert result.dataset == "J1"
    assert result.question_id == 101
    assert result.retrieval_run_id == 21
    assert result.retrieval_name == "j1_source_urls_v1"
    assert result.embedding_model == "text-embedding-3-small"
    assert result.top_k == 4
    assert [chunk.rank for chunk in result.chunks] == [1, 2]
    assert [chunk.chunk_id for chunk in result.chunks] == [500, 501]
    assert result.chunks[0].source_kind == "source_url_content"
    assert result.chunks[0].metadata == {"sequence": 21}
    assert provider.calls == [("Texto original da questao J1.", "text-embedding-3-small")]
    assert repository.calls == [
        ("get_question", "J1", 101),
        ("get_vector_base", "J1"),
        ("search", "J1", "text-embedding-3-small", [0.11, 0.22, 0.33], 4),
    ]
    serialized = asdict(result)
    assert "answer_key" not in serialized
    assert "reference_answer" not in serialized
    assert "rubric" not in serialized
    assert "guideline" not in serialized


def test_retrieve_for_question_works_for_j2_with_explicit_top_k() -> None:
    repository = FakeRepository()
    provider = FakeEmbeddingProvider(vector=[0.9, 0.1])
    repository.questions[("J2", 202)] = RagRetrievalQuestion(
        question_id=202,
        dataset="J2",
        question_text="Qual alternativa correta?",
    )
    repository.vector_bases["J2"] = _vector_base(dataset="J2", embedding_model="qwen-embed-8b", top_k=8)
    repository.search_results["J2"] = [
        {
            "rank": 1,
            "chunk_id": 900,
            "chunk_text": "Chunk J2.",
            "chunk_kind": "source_url_content",
            "document_id": 901,
            "document_key": "doc-j2",
            "lei": None,
            "norma": "Norma J2",
            "url": "https://example.test/j2",
            "urn": None,
            "artigo": None,
            "topico": "Tema J2",
            "relevancia": "media",
            "tipo": "jurisprudencia",
            "distance": 0.07,
            "similarity": 0.93,
        }
    ]

    result = RagRetrieverService(repository=repository, embedding_provider=provider).retrieve_for_question(
        question_id=202,
        dataset="OAB_Exames",
        top_k=3,
    )

    assert result.status == "success"
    assert result.dataset == "J2"
    assert result.embedding_model == "qwen-embed-8b"
    assert result.top_k == 3
    assert provider.calls == [("Qual alternativa correta?", "qwen-embed-8b")]
    assert repository.calls == [
        ("get_question", "J2", 202),
        ("get_vector_base", "J2"),
        ("search", "J2", "qwen-embed-8b", [0.9, 0.1], 3),
    ]


def test_retrieve_for_question_returns_question_not_found() -> None:
    repository = FakeRepository()
    provider = FakeEmbeddingProvider()

    result = RagRetrieverService(repository=repository, embedding_provider=provider).retrieve_for_question(
        question_id=404,
        dataset="J1",
    )

    assert result.status == "question_not_found"
    assert result.dataset == "J1"
    assert result.retrieval_run_id is None
    assert result.embedding_model is None
    assert result.chunks == []
    assert provider.calls == []
    assert repository.calls == [("get_question", "J1", 404)]


def test_retrieve_for_question_returns_vector_base_not_found() -> None:
    repository = FakeRepository()
    provider = FakeEmbeddingProvider()
    repository.questions[("J1", 111)] = RagRetrievalQuestion(
        question_id=111,
        dataset="J1",
        question_text="Questao sem base vetorial.",
    )

    result = RagRetrieverService(repository=repository, embedding_provider=provider).retrieve_for_question(
        question_id=111,
        dataset="J1",
    )

    assert result.status == "vector_base_not_found"
    assert result.dataset == "J1"
    assert result.retrieval_run_id is None
    assert result.embedding_model is None
    assert result.chunks == []
    assert provider.calls == []


def test_retrieve_for_question_returns_embedding_model_not_configured() -> None:
    repository = FakeRepository()
    provider = FakeEmbeddingProvider()
    repository.questions[("J2", 222)] = RagRetrievalQuestion(
        question_id=222,
        dataset="J2",
        question_text="Questao sem modelo configurado.",
    )
    repository.vector_bases["J2"] = _vector_base(dataset="J2", embedding_model=None, top_k=6)

    result = RagRetrieverService(repository=repository, embedding_provider=provider).retrieve_for_question(
        question_id=222,
        dataset="J2",
    )

    assert result.status == "embedding_model_not_configured"
    assert result.dataset == "J2"
    assert result.retrieval_run_id == 21
    assert result.embedding_model is None
    assert result.top_k == 6
    assert result.chunks == []
    assert provider.calls == []


def test_retrieve_for_question_returns_no_chunks_found() -> None:
    repository = FakeRepository()
    provider = FakeEmbeddingProvider()
    repository.questions[("J2", 333)] = RagRetrievalQuestion(
        question_id=333,
        dataset="J2",
        question_text="Questao sem resultado.",
    )
    repository.vector_bases["J2"] = _vector_base(dataset="J2", embedding_model="text-embedding-3-small", top_k=5)
    repository.search_results["J2"] = []

    result = RagRetrieverService(repository=repository, embedding_provider=provider).retrieve_for_question(
        question_id=333,
        dataset="J2",
    )

    assert result.status == "no_chunks_found"
    assert result.dataset == "J2"
    assert result.top_k == 5
    assert result.chunks == []
    assert provider.calls == [("Questao sem resultado.", "text-embedding-3-small")]


def test_retrieve_for_question_fallbacks_are_dataset_scoped_for_j1() -> None:
    repository = FakeRepository()
    provider = FakeEmbeddingProvider()

    result = RagRetrieverService(repository=repository, embedding_provider=provider).retrieve_for_question(
        question_id=1,
        dataset="J1",
    )

    assert result.dataset == "J1"
    assert result.status == "question_not_found"
    assert repository.calls == [("get_question", "J1", 1)]


def test_retrieve_for_question_fallbacks_are_dataset_scoped_for_j2() -> None:
    repository = FakeRepository()
    provider = FakeEmbeddingProvider()

    result = RagRetrieverService(repository=repository, embedding_provider=provider).retrieve_for_question(
        question_id=1,
        dataset="J2",
    )

    assert result.dataset == "J2"
    assert result.status == "question_not_found"
    assert repository.calls == [("get_question", "J2", 1)]


def test_retrieve_for_question_validates_top_k() -> None:
    repository = FakeRepository()
    provider = FakeEmbeddingProvider()

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        RagRetrieverService(repository=repository, embedding_provider=provider).retrieve_for_question(
            question_id=1,
            dataset="J1",
            top_k=0,
        )


def _vector_base(*, dataset: str, embedding_model: str | None, top_k: int) -> RagVectorBaseSummary:
    return RagVectorBaseSummary(
        dataset=dataset,
        dataset_name="OAB_Bench" if dataset == "J1" else "OAB_Exames",
        import_run_id=7,
        active_curation_run_id=7,
        matches_active_curation=True,
        retrieval_run_id=21,
        retrieval_name=f"{dataset.lower()}_source_urls_v1",
        retrieval_strategy="source_url_only_v1",
        embedding_model=embedding_model,
        top_k=top_k,
        vector_enabled=True,
        lexical_enabled=False,
        rerank_enabled=False,
        document_count=10,
        chunk_count=20,
        embedding_count=20,
        status="pronta_com_embeddings",
        created_at="2026-06-04T10:00:00",
    )
