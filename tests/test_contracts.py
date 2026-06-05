"""Tests for baseline package and contracts imports."""

from __future__ import annotations

import atividade_2
from atividade_2 import contracts
from atividade_2.contracts import (
    MATCH_TYPE_VALUES,
    CandidateAnswerContextChunkRecord,
    CandidateAnswerRecord,
    CandidateModelAssignment,
    CandidateModelAssignmentRange,
    CandidatePromptContext,
    CandidatePromptRecord,
    CandidateQuestionRecord,
    CandidateRawResponse,
    CandidateRunRecord,
    RetrievedRagChunk,
)


def test_package_can_be_imported() -> None:
    """The installed package should expose a version string."""
    assert isinstance(atividade_2.__version__, str)


def test_contracts_module_can_be_imported() -> None:
    """The contracts module should exist without fake domain models."""
    assert contracts.__doc__


def test_candidate_rag_contracts_capture_optional_fields_and_defaults() -> None:
    prompt = CandidatePromptRecord(
        prompt_id=7,
        dataset="J1",
        version=2,
        persona="persona",
        context="contexto",
        rag_instruction="instrucao",
        output="saida",
        active=True,
        created_by="tester",
        created_at="2026-06-04T09:00:00",
    )
    run = CandidateRunRecord(
        candidate_run_id=11,
        dataset="J1",
        retrieval_run_id=21,
        prompt_id=7,
        model_name="candidate-model",
        provider="openai",
        batch_size=25,
        temperature=0.2,
        max_tokens=512,
        top_p=0.95,
        metadata={"mode": "rag"},
    )
    answer = CandidateAnswerRecord(
        candidate_answer_id=31,
        candidate_run_id=11,
        question_id=41,
        model_name="candidate-model",
        rendered_prompt="prompt renderizado",
        status="success",
        answer_text="resposta",
        raw_response={"answer": "resposta"},
    )
    chunk = CandidateAnswerContextChunkRecord(
        answer_context_chunk_id=51,
        candidate_answer_id=31,
        chunk_id=61,
        rank=1,
        chunk_text_snapshot="trecho",
        similarity_score=0.812345,
        metadata={"source_kind": "lei"},
    )
    prompt_context = CandidatePromptContext(
        question_id=41,
        dataset_name="J2",
        question_text="Qual alternativa correta?",
        retrieved_chunks=[
            RetrievedRagChunk(
                rank=1,
                chunk_id=61,
                chunk_text="Trecho seguro.",
                source_kind="lei",
                document_id=71,
                document_key="doc-1",
                lei="Lei 8.666",
                norma="Licitacoes",
                url="https://example.test/lei",
                urn=None,
                artigo="Art. 10",
                topico="Tema",
                relevancia="alta",
                tipo="lei",
                distance=0.12,
                similarity=0.88,
            )
        ],
        alternatives={"A": "Opcao A", "B": "Opcao B"},
        retrieval_run_id=21,
        retrieval_name="j2_source_urls_v1",
        top_k=5,
    )
    raw_response = CandidateRawResponse(
        text="Alternativa final: B",
        provider="remote_http",
        model="candidate-model",
        latency_ms=321,
    )
    question = CandidateQuestionRecord(
        question_id=41,
        dataset="J2",
        dataset_name="OAB_Exames",
        question_sequence=12,
        question_text="Qual a alternativa correta?",
        alternatives={"A": "Opcao A", "B": "Opcao B"},
    )

    assert prompt.active is True
    assert question.question_sequence == 12
    assert run.metadata == {"mode": "rag"}
    assert answer.raw_response == {"answer": "resposta"}
    assert chunk.metadata["source_kind"] == "lei"
    assert prompt_context.alternatives == {"A": "Opcao A", "B": "Opcao B"}
    assert raw_response.text == "Alternativa final: B"


def test_candidate_model_assignment_contract_serializes_av2_and_av3_identity_without_mutation() -> None:
    assignment = CandidateModelAssignment(
        assignment_id=17,
        id_modelo_av2=14,
        av2_model_name="chatgpt-5.3",
        owner="José Bruno",
        original_provider_model_id="GPT-5",
        original_runtime="ChatGPT UI",
        av3_provider="openrouter",
        av3_provider_model_id="openai/gpt-5",
        hf_model_id=None,
        artifact_format="api",
        original_quantization=None,
        av3_quantization="proprietary_api",
        match_type="same_model_api_reproduction",
        validation_status="confirmed_by_owner",
        ranges=(
            CandidateModelAssignmentRange(
                assignment_range_id=31,
                assignment_id=17,
                dataset_code="J1",
                question_sequence_start=119,
                question_sequence_end=130,
            ),
        ),
    )

    payload = assignment.to_dict()
    payload["ranges"][0]["dataset_code"] = "J2"

    assert payload["assignment_id"] == 17
    assert payload["id_modelo_av2"] == 14
    assert payload["av2_model_name"] == "chatgpt-5.3"
    assert payload["av3_provider"] == "openrouter"
    assert payload["av3_provider_model_id"] == "openai/gpt-5"
    assert payload["runnable"] is True
    assert assignment.ranges[0].dataset_code == "J1"


def test_candidate_model_assignment_contract_validates_ranges_and_match_types() -> None:
    assert set(MATCH_TYPE_VALUES) == {
        "same_model_same_runtime",
        "same_model_different_quantization",
        "same_model_api_reproduction",
        "same_family_version_needs_subtype_confirmation",
        "same_family_newer_version_substitution",
        "proprietary_api_resolved",
        "not_reproduced_provider_unavailable",
    }

    invalid_range_error = None
    try:
        CandidateModelAssignmentRange(
            assignment_range_id=None,
            assignment_id=None,
            dataset_code="J1",
            question_sequence_start=10,
            question_sequence_end=9,
        )
    except ValueError as exc:
        invalid_range_error = str(exc)

    pending_assignment = CandidateModelAssignment(
        assignment_id=18,
        id_modelo_av2=13,
        av2_model_name="gemini-3-pro",
        owner="José Bruno",
        original_provider_model_id="Gemini 3.5",
        original_runtime="Gemini UI",
        av3_provider="openrouter",
        av3_provider_model_id="google/gemini-3.5-flash",
        hf_model_id=None,
        artifact_format="api",
        original_quantization=None,
        av3_quantization="proprietary_api",
        match_type="same_family_version_needs_subtype_confirmation",
        validation_status="needs_owner_confirmation_gemini_subtype",
        ranges=(
            CandidateModelAssignmentRange(
                assignment_range_id=32,
                assignment_id=18,
                dataset_code="J2",
                question_sequence_start=1231,
                question_sequence_end=1353,
            ),
        ),
    )

    assert invalid_range_error == "question_sequence_end must be >= question_sequence_start."
    assert pending_assignment.is_runnable() is False
    assert pending_assignment.is_runnable(include_pending_confirmation=True) is True
    assert pending_assignment.warning_message == (
        "Pending owner confirmation: Gemini subtype still needs exact confirmation."
    )
