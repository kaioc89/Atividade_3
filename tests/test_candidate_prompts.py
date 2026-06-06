from __future__ import annotations

from atividade_2.candidate_prompts import build_candidate_prompt
from atividade_2.contracts import CandidatePromptContext, CandidatePromptRecord, RetrievedRagChunk


def _chunk() -> RetrievedRagChunk:
    return RetrievedRagChunk(
        rank=1,
        chunk_id=501,
        chunk_text="Art. 5 da Lei X.",
        source_kind="lei",
        document_id=701,
        document_key="lei-x",
        lei="Lei X",
        norma="Norma X",
        url="https://example.test/lei-x",
        urn=None,
        artigo="Art. 5",
        topico="Tema A",
        relevancia="alta",
        tipo="lei",
        distance=0.12,
        similarity=0.88,
    )


def test_j1_piece_prompt_renders_piece_specific_instructions() -> None:
    prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=101,
            dataset_name="J1",
            question_text="Elabore a medida judicial cabível.",
            question_type="PEÇA PRÁTICO-PROFISSIONAL",
            retrieved_chunks=[_chunk()],
        )
    )

    assert "peça prático-profissional" in prompt
    assert "segunda fase do exame da OAB" in prompt
    assert "somente a peça prático-profissional final" in prompt
    assert "Estruture o texto como documento processual." in prompt
    assert "Resposta final:\n<peça>" in prompt


def test_j1_piece_prompt_forbids_fabrication_and_rag_mentions() -> None:
    prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=101,
            dataset_name="J1",
            question_text="Elabore a medida judicial cabível.",
            question_type="peça profissional",
            retrieved_chunks=[],
        )
    )

    assert "Não mencione RAG, chunks recuperados, contexto fornecido ou trechos recuperados." in prompt
    assert "Não invente fatos, partes, números de processo, números de OAB" in prompt
    assert "não fornecidas" in prompt
    assert "`Local, ...`" in prompt
    assert "`OAB/UF: ...`" in prompt


def test_j1_piece_prompt_uses_template_override_when_persisted_template_is_generic() -> None:
    prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=101,
            dataset_name="J1",
            question_text="Elabore a medida judicial cabível.",
            question_type="PECA PRATICO-PROFISSIONAL",
            retrieved_chunks=[_chunk()],
        ),
        template=CandidatePromptRecord(
            prompt_id=1,
            dataset="J1",
            version=1,
            persona="Você é um candidato do exame da OAB respondendo uma questão discursiva.",
            context="Questão original:\n{pergunta_oab}",
            rag_instruction="{contexto_rag}",
            output="Entregue uma resposta objetiva e juridicamente fundamentada.",
            active=True,
        ),
    )

    assert "segunda fase do exame da OAB" in prompt
    assert "questão discursiva" not in prompt
    assert "somente a peça prático-profissional final" in prompt


def test_j1_piece_template_placeholders_render_question_type_and_instruction() -> None:
    prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=101,
            dataset_name="J1",
            question_text="Elabore a medida judicial cabível.",
            question_type="PEÇA PROFISSIONAL",
            retrieved_chunks=[],
        ),
        template=CandidatePromptRecord(
            prompt_id=1,
            dataset="J1",
            version=2,
            persona="Tipo: {question_type} / {tipo_questao}",
            context="Pergunta:\n{pergunta_oab}",
            rag_instruction="{candidate_prompt_type}\n{candidate_prompt_type_instruction}",
            output="Saída",
            active=True,
        ),
    )

    assert "Tipo: PEÇA PROFISSIONAL / PEÇA PROFISSIONAL" in prompt
    assert "j1_peca_profissional" in prompt
    assert "Não mencione RAG" in prompt


def test_j1_question_prompt_remains_behaviorally_equivalent() -> None:
    base_prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=101,
            dataset_name="J1",
            question_text="Questão aberta.",
            retrieved_chunks=[],
        )
    )
    typed_prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=101,
            dataset_name="J1",
            question_text="Questão aberta.",
            question_type="QUESTÃO",
            retrieved_chunks=[],
        )
    )

    assert typed_prompt == base_prompt
    assert "questão discursiva" in typed_prompt
    assert "Resposta final:\n<sua resposta>" in typed_prompt


def test_unknown_j1_type_falls_back_to_discursive_prompt() -> None:
    prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=101,
            dataset_name="J1",
            question_text="Questão aberta.",
            question_type="parecer oral",
            retrieved_chunks=[],
        ),
        template=CandidatePromptRecord(
            prompt_id=1,
            dataset="J1",
            version=2,
            persona="Tipo efetivo: {candidate_prompt_type}",
            context="Pergunta:\n{pergunta_oab}",
            rag_instruction="",
            output="Saída",
            active=True,
        ),
    )

    assert "j1_unknown_fallback_discursive" in prompt
    assert "peça prático-profissional" not in prompt


def test_j1_candidate_prompt_excludes_judge_only_material() -> None:
    prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=101,
            dataset_name="J1",
            question_text="Questão aberta.",
            retrieved_chunks=[],
        )
    ).lower()

    assert "guideline" not in prompt
    assert "rubric" not in prompt
    assert "reference answer" not in prompt
    assert "answer key" not in prompt
    assert "judge prompt" not in prompt
    assert "judge score" not in prompt
    assert "gabarito" not in prompt
    assert "rubrica" not in prompt
    assert "nota do juiz" not in prompt


def test_j2_candidate_prompt_includes_question_alternatives_chunks_and_final_choice_instruction() -> None:
    prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=202,
            dataset_name="J2",
            question_text="Qual alternativa está correta?",
            question_type="questao objetiva",
            alternatives={
                "A": "Primeira opção.",
                "B": "Segunda opção.",
                "C": "Terceira opção.",
            },
            retrieved_chunks=[_chunk()],
        )
    )

    assert "Qual alternativa está correta?" in prompt
    assert "- A: Primeira opção." in prompt
    assert "- B: Segunda opção." in prompt
    assert "Art. 5 da Lei X." in prompt
    assert "Alternativa final: X" in prompt
    assert "escolher exatamente uma alternativa" in prompt


def test_j2_prompt_remains_unchanged_even_if_question_type_is_present() -> None:
    base_prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=202,
            dataset_name="J2",
            question_text="Questão objetiva.",
            alternatives={"A": "Opcao A", "B": "Opcao B"},
            retrieved_chunks=[],
        )
    )
    typed_prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=202,
            dataset_name="J2",
            question_text="Questão objetiva.",
            question_type="PEÇA PRÁTICO-PROFISSIONAL",
            alternatives={"A": "Opcao A", "B": "Opcao B"},
            retrieved_chunks=[],
        )
    )

    assert typed_prompt == base_prompt


def test_j2_candidate_prompt_excludes_official_answer_key_and_correct_alternative() -> None:
    prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=202,
            dataset_name="J2",
            question_text="Questão objetiva.",
            alternatives={"A": "Opcao A", "B": "Opcao B"},
            retrieved_chunks=[],
        )
    ).lower()

    assert "gabarito oficial" not in prompt
    assert "alternativa correta" not in prompt
    assert "correct alternative" not in prompt
    assert "answer key" not in prompt
    assert "judge score" not in prompt
