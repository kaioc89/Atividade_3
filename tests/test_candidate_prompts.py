from __future__ import annotations

from atividade_2.candidate_prompts import build_candidate_prompt
from atividade_2.contracts import CandidatePromptContext, RetrievedRagChunk


def test_j1_candidate_prompt_includes_question_and_retrieved_chunks() -> None:
    prompt = build_candidate_prompt(
        CandidatePromptContext(
            question_id=101,
            dataset_name="J1",
            question_text="Elabore a medida judicial cabível.",
            retrieved_chunks=[
                RetrievedRagChunk(
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
                ),
                RetrievedRagChunk(
                    rank=2,
                    chunk_id=502,
                    chunk_text="Precedente relevante.",
                    source_kind="jurisprudencia",
                    document_id=702,
                    document_key="precedente-y",
                    lei=None,
                    norma="Jurisprudencia",
                    url="https://example.test/precedente-y",
                    urn=None,
                    artigo=None,
                    topico="Tema B",
                    relevancia="media",
                    tipo="jurisprudencia",
                    distance=0.2,
                    similarity=0.8,
                ),
            ],
        )
    )

    assert "Elabore a medida judicial cabível." in prompt
    assert "Art. 5 da Lei X." in prompt
    assert "Precedente relevante." in prompt
    assert "Resposta final:" in prompt


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
            alternatives={
                "A": "Primeira opção.",
                "B": "Segunda opção.",
                "C": "Terceira opção.",
            },
            retrieved_chunks=[
                RetrievedRagChunk(
                    rank=1,
                    chunk_id=801,
                    chunk_text="Trecho de apoio para J2.",
                    source_kind="lei",
                    document_id=901,
                    document_key="doc-j2",
                    lei="Lei Y",
                    norma="Norma Y",
                    url="https://example.test/lei-y",
                    urn=None,
                    artigo="Art. 7",
                    topico="Tema J2",
                    relevancia="alta",
                    tipo="lei",
                    distance=0.09,
                    similarity=0.91,
                )
            ],
        )
    )

    assert "Qual alternativa está correta?" in prompt
    assert "- A: Primeira opção." in prompt
    assert "- B: Segunda opção." in prompt
    assert "Trecho de apoio para J2." in prompt
    assert "Alternativa final: X" in prompt
    assert "escolher exatamente uma alternativa" in prompt


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
