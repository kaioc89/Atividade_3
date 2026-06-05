from __future__ import annotations

import pytest

from atividade_2.candidate_context_budget import (
    budget_candidate_retrieval_context,
    clamp_max_output_tokens_to_fixed_prompt,
    resolve_candidate_max_output_tokens,
    resolve_candidate_model_runtime_profile,
)
from atividade_2.candidate_prompts import build_candidate_prompt
from atividade_2.contracts import (
    CandidatePromptContext,
    CandidatePromptRecord,
    CandidateQuestionRecord,
    RagRetrievalResult,
    RetrievedRagChunk,
)


def _prompt() -> CandidatePromptRecord:
    return CandidatePromptRecord(
        prompt_id=1,
        dataset="J1",
        version=1,
        persona="Core instruction: answer as an OAB candidate.",
        context="Question:\n{pergunta_oab}",
        rag_instruction="Retrieved context:\n{contexto_rag}",
        output="Core output instruction: provide the final answer.",
        active=True,
    )


def _question(question_text: str = "What is the legal consequence?") -> CandidateQuestionRecord:
    return CandidateQuestionRecord(
        question_id=71,
        dataset="J1",
        dataset_name="OAB_Bench",
        question_sequence=71,
        question_text=question_text,
    )


def _chunk(*, rank: int, chunk_id: int, text: str) -> RetrievedRagChunk:
    return RetrievedRagChunk(
        rank=rank,
        chunk_id=chunk_id,
        chunk_text=text,
        source_kind="lei",
        document_id=chunk_id + 100,
        document_key=f"doc-{chunk_id}",
        lei="Lei X",
        norma="Norma X",
        url="https://example.test/source",
        urn=None,
        artigo="Art. 1",
        topico="Tema",
        relevancia="alta",
        tipo="lei",
        distance=0.1,
        similarity=0.9,
        metadata={},
    )


def _retrieval_result(chunks: list[RetrievedRagChunk]) -> RagRetrievalResult:
    return RagRetrievalResult(
        question_id=71,
        dataset="J1",
        retrieval_run_id=21,
        retrieval_name="j1_source_urls_v1",
        embedding_model="text-embedding-3-small",
        top_k=5,
        status="success",
        chunks=chunks,
    )


def test_gemma_profile_defaults_to_1024_and_known_context_window() -> None:
    profile = resolve_candidate_model_runtime_profile(
        provider="featherless",
        model_name="google/gemma-2-2b-it",
        safety_margin_tokens=512,
    )

    assert profile.context_window_tokens == 8192
    assert resolve_candidate_max_output_tokens(profile=profile, requested_max_tokens=None) == 1024


def test_context_window_override_replaces_static_profile_window() -> None:
    profile = resolve_candidate_model_runtime_profile(
        provider="featherless",
        model_name="google/gemma-2-2b-it",
        safety_margin_tokens=512,
        context_window_tokens_override=4096,
    )

    assert profile.context_window_tokens == 4096
    assert profile.source == "env_override"


def test_known_context_window_clamps_output_budget() -> None:
    fixed_prompt = "x" * 1200

    max_tokens = clamp_max_output_tokens_to_fixed_prompt(
        model_name="google/gemma-2-2b-it",
        context_window_tokens=1000,
        fixed_prompt_text=fixed_prompt,
        max_output_tokens=900,
        safety_margin_tokens=100,
    )

    assert max_tokens == 600


def test_retrieved_context_chunks_are_truncated_and_dropped_to_fit_budget() -> None:
    result = budget_candidate_retrieval_context(
        question=_question("Question text must remain intact."),
        retrieval_result=_retrieval_result(
            [
                _chunk(rank=1, chunk_id=501, text="A" * 80),
                _chunk(rank=2, chunk_id=502, text="B" * 80),
                _chunk(rank=3, chunk_id=503, text="C" * 80),
            ]
        ),
        prompt=_prompt(),
        model_name="google/gemma-2-2b-it",
        av3_provider="featherless",
        max_tokens=10,
        safety_margin_tokens=10,
        context_window_tokens=160,
    )

    assert result.budget.retrieved_chunks == 3
    assert result.budget.included_chunks == 2
    assert result.budget.truncated_chunks == 1
    assert result.budget.dropped_chunks == 1
    assert [chunk.rank for chunk in result.retrieval_result_for_prompt.chunks] == [1, 2]
    assert result.retrieval_result_for_prompt.chunks[1].chunk_text.startswith("B")
    assert len(result.retrieval_result_for_prompt.chunks[1].chunk_text) < 80


def test_core_question_and_instructions_are_not_truncated() -> None:
    question = _question("Question text must remain intact.")
    prompt = _prompt()
    result = budget_candidate_retrieval_context(
        question=question,
        retrieval_result=_retrieval_result(
            [
                _chunk(rank=1, chunk_id=501, text="A" * 200),
                _chunk(rank=2, chunk_id=502, text="B" * 200),
            ]
        ),
        prompt=prompt,
        model_name="google/gemma-2-2b-it",
        av3_provider="featherless",
        max_tokens=10,
        safety_margin_tokens=10,
        context_window_tokens=88,
    )

    rendered = build_candidate_prompt(
        CandidatePromptContext(
            question_id=question.question_id,
            dataset_name=question.dataset,
            question_text=question.question_text,
            retrieved_chunks=result.retrieval_result_for_prompt.chunks,
        ),
        template=prompt,
    )

    assert "Question text must remain intact." in rendered
    assert "Core instruction: answer as an OAB candidate." in rendered
    assert "Core output instruction: provide the final answer." in rendered


def test_chunk_metadata_records_truncation() -> None:
    result = budget_candidate_retrieval_context(
        question=_question(),
        retrieval_result=_retrieval_result([_chunk(rank=1, chunk_id=501, text="A" * 200)]),
        prompt=_prompt(),
        model_name="google/gemma-2-2b-it",
        av3_provider="featherless",
        max_tokens=10,
        safety_margin_tokens=10,
        context_window_tokens=100,
    )

    metadata = result.retrieval_result_for_prompt.chunks[0].metadata["candidate_budget"]

    assert metadata["included_in_prompt"] is True
    assert metadata["was_truncated"] is True
    assert metadata["original_estimated_tokens"] == 50
    assert metadata["included_estimated_tokens"] < metadata["original_estimated_tokens"]
    assert metadata["truncation_reason"] == "context_budget"


def test_budget_fails_when_fixed_prompt_cannot_fit() -> None:
    with pytest.raises(ValueError, match="Fixed candidate prompt plus max output exceeds"):
        budget_candidate_retrieval_context(
            question=_question("Q" * 2000),
            retrieval_result=_retrieval_result([]),
            prompt=_prompt(),
            model_name="google/gemma-2-2b-it",
            av3_provider="featherless",
            max_tokens=100,
            safety_margin_tokens=100,
            context_window_tokens=200,
        )
