from __future__ import annotations

from atividade_2.candidate_answer_normalization import (
    normalize_candidate_answer_context_for_judge,
    normalize_candidate_answer_for_judge,
)
from atividade_2.contracts import CandidateAnswerContext
from atividade_2.prompts import build_judge_prompt


def _j1_context(candidate_answer: str, *, question_text: str = "QUESTÃO\n\nExplique a medida cabível.") -> CandidateAnswerContext:
    return CandidateAnswerContext(
        av1_answer_id=1,
        candidate_answer_id=None,
        question_id=101,
        dataset_name="OAB_Bench",
        question_text=question_text,
        reference_answer="Gabarito jurídico.",
        candidate_answer=candidate_answer,
        candidate_model="candidate-model",
    )


def test_normalize_candidate_answer_keeps_clean_answer_unchanged() -> None:
    raw_answer = "O habeas corpus é cabível quando há coação ilegal à liberdade de locomoção."

    result = normalize_candidate_answer_for_judge(raw_answer)

    assert result.normalized_answer == raw_answer
    assert result.normalization_flags == {
        "raw_answer_preserved": True,
        "removed_resposta_final_label": False,
        "removed_english_final_answer_wrapper": False,
        "removed_special_tokens": False,
        "removed_markdown_fences": False,
        "removed_placeholder_answer": False,
        "collapsed_excess_blank_lines": False,
        "detected_degenerate_output": False,
    }


def test_normalize_candidate_answer_removes_leading_resposta_final_label() -> None:
    result = normalize_candidate_answer_for_judge("Resposta final:\nMandado de segurança.")

    assert result.normalized_answer == "Mandado de segurança."
    assert result.normalization_flags["removed_resposta_final_label"] is True


def test_normalize_candidate_answer_removes_leading_english_wrapper() -> None:
    result = normalize_candidate_answer_for_judge("Here is the final answer:\nA ação cabível é apelação.")

    assert result.normalized_answer == "A ação cabível é apelação."
    assert result.normalization_flags["removed_english_final_answer_wrapper"] is True


def test_normalize_candidate_answer_removes_markdown_fences_and_preserves_inner_content() -> None:
    result = normalize_candidate_answer_for_judge("```text\nMandado de segurança com pedido liminar.\n```")

    assert result.normalized_answer == "Mandado de segurança com pedido liminar."
    assert result.normalization_flags["removed_markdown_fences"] is True


def test_normalize_candidate_answer_removes_special_tokens() -> None:
    result = normalize_candidate_answer_for_judge("Fundamentação jurídica.<|im_end|>")

    assert result.normalized_answer == "Fundamentação jurídica."
    assert result.normalization_flags["removed_special_tokens"] is True


def test_normalize_candidate_answer_removes_placeholder_literal() -> None:
    result = normalize_candidate_answer_for_judge("Resposta final:\n<sua resposta>\nMandado de segurança.")

    assert result.normalized_answer == "Mandado de segurança."
    assert result.normalization_flags["removed_placeholder_answer"] is True


def test_normalize_candidate_answer_collapses_excessive_blank_lines() -> None:
    result = normalize_candidate_answer_for_judge("Linha 1\n\n\n\nLinha 2\n\n\nLinha 3")

    assert result.normalized_answer == "Linha 1\n\nLinha 2\n\nLinha 3"
    assert result.normalization_flags["collapsed_excess_blank_lines"] is True


def test_normalize_candidate_answer_flags_degenerate_repeated_character_output_without_removing_it() -> None:
    result = normalize_candidate_answer_for_judge("@@@@@")

    assert result.normalized_answer == "@@@@@"
    assert result.normalization_flags["detected_degenerate_output"] is True


def test_normalize_candidate_answer_preserves_piece_style_content() -> None:
    raw_answer = (
        "Resposta final:\n"
        "Excelentíssimo Senhor Doutor Juiz de Direito,\n\n"
        "Trata-se de mandado de segurança com pedido liminar."
    )

    result = normalize_candidate_answer_for_judge(raw_answer)

    assert result.normalized_answer == (
        "Excelentíssimo Senhor Doutor Juiz de Direito,\n\n"
        "Trata-se de mandado de segurança com pedido liminar."
    )


def test_normalize_candidate_answer_preserves_discursive_question_content() -> None:
    raw_answer = "A medida cabível é a apelação, com fundamento no art. 1.009 do CPC."

    result = normalize_candidate_answer_for_judge(raw_answer)

    assert result.normalized_answer == raw_answer


def test_normalize_candidate_answer_does_not_rewrite_meaningful_legal_content() -> None:
    raw_answer = "A peça cabível é o mandado de segurança, e não habeas data, porque há direito líquido e certo."

    result = normalize_candidate_answer_for_judge(raw_answer)

    assert result.normalized_answer == raw_answer
    assert result.normalization_flags["detected_degenerate_output"] is False


def test_normalize_candidate_answer_flags_placeholder_only_or_nearly_empty_output() -> None:
    result = normalize_candidate_answer_for_judge("  <sua resposta>  ")

    assert result.normalized_answer == ""
    assert result.normalization_flags["removed_placeholder_answer"] is True
    assert result.normalization_flags["detected_degenerate_output"] is True


def test_normalize_candidate_answer_context_for_judge_preserves_raw_answer_and_exposes_normalized_prompt_input() -> None:
    raw_answer = "Resposta final:\n```text\nMandado de segurança.\n```\n<|im_end|>"
    context = _j1_context(raw_answer, question_text="PEÇA PRÁTICO-PROFISSIONAL\n\nElabore a peça cabível.")

    normalized_context = normalize_candidate_answer_context_for_judge(context)
    prompt = build_judge_prompt(normalized_context)

    assert context.candidate_answer == raw_answer
    assert normalized_context.raw_candidate_answer == raw_answer
    assert normalized_context.candidate_answer == "Mandado de segurança."
    assert normalized_context.candidate_answer_normalization_flags["removed_resposta_final_label"] is True
    assert normalized_context.candidate_answer_normalization_flags["removed_markdown_fences"] is True
    assert normalized_context.candidate_answer_normalization_flags["removed_special_tokens"] is True
    assert "Mandado de segurança." in prompt
    assert "Resposta final:" not in prompt
    assert "Resposta da IA a ser avaliada:\n```text\nMandado de segurança.\n```" in prompt
    assert "Resposta da IA a ser avaliada:\n```text\n```text" not in prompt
    assert "<|im_end|>" not in prompt
