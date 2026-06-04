"""Tests for baseline package and contracts imports."""

from __future__ import annotations

import atividade_2
from atividade_2 import contracts
from atividade_2.contracts import (
    CandidateAnswerContextChunkRecord,
    CandidateAnswerRecord,
    CandidatePromptRecord,
    CandidateRunRecord,
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

    assert prompt.active is True
    assert run.metadata == {"mode": "rag"}
    assert answer.raw_response == {"answer": "resposta"}
    assert chunk.metadata["source_kind"] == "lei"
