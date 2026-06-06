"""Candidate-safe AV3 prompt rendering for Com_RAG answer generation."""

from __future__ import annotations

import json
import string
import unicodedata
from dataclasses import dataclass
from typing import Any

from .contracts import CandidatePromptContext, CandidatePromptRecord, RetrievedRagChunk


@dataclass(frozen=True)
class CandidatePromptStrategy:
    """Normalized prompt subtype resolved from dataset and question type."""

    candidate_prompt_type: str
    question_type: str | None
    question_type_normalized: str | None
    candidate_prompt_type_instruction: str
    persona_override: str | None = None
    output_override: str | None = None


def build_candidate_prompt(
    context: CandidatePromptContext,
    *,
    template: CandidatePromptRecord | None = None,
) -> str:
    """Render one candidate prompt from safe question data and retrieved chunks."""
    strategy = resolve_candidate_prompt_strategy(
        dataset_name=context.dataset_name,
        question_type=context.question_type,
    )
    if template is not None:
        return _render_template_prompt(context=context, template=template, strategy=strategy)
    if is_j2_candidate_context(context):
        return _build_default_j2_prompt(context)
    if strategy.candidate_prompt_type == "j1_peca_profissional":
        return _build_default_j1_piece_prompt(context)
    return _build_default_j1_discursive_prompt(context)


def resolve_candidate_prompt_strategy(
    *,
    dataset_name: str,
    question_type: str | None,
) -> CandidatePromptStrategy:
    """Resolve the effective prompt subtype used for one candidate question."""
    normalized_question_type = _normalize_question_type(question_type)
    if dataset_name.upper() in {"J2", "OAB_EXAMES"}:
        return CandidatePromptStrategy(
            candidate_prompt_type="j2_multiple_choice",
            question_type=question_type,
            question_type_normalized=normalized_question_type,
            candidate_prompt_type_instruction="",
        )
    if _is_piece_question_type(normalized_question_type):
        return CandidatePromptStrategy(
            candidate_prompt_type="j1_peca_profissional",
            question_type=question_type,
            question_type_normalized=normalized_question_type,
            candidate_prompt_type_instruction=_piece_instruction_block(),
            persona_override="Você é um candidato da segunda fase do exame da OAB elaborando uma peça prático-profissional.",
            output_override=(
                "Entregue somente a peça prático-profissional final.\n"
                "Finalize com o bloco:\n"
                "Resposta final:\n"
                "<peça>"
            ),
        )
    if _is_discursive_question_type(normalized_question_type):
        prompt_type = "j1_questao_discursiva"
    else:
        prompt_type = "j1_unknown_fallback_discursive"
    return CandidatePromptStrategy(
        candidate_prompt_type=prompt_type,
        question_type=question_type,
        question_type_normalized=normalized_question_type,
        candidate_prompt_type_instruction="",
    )


def is_j2_candidate_context(context: CandidatePromptContext) -> bool:
    """Return whether the prompt targets the objective multiple-choice dataset."""
    return context.dataset_name.upper() in {"J2", "OAB_EXAMES"}


def build_candidate_prompt_metadata(context: CandidatePromptContext) -> dict[str, Any]:
    """Return audit metadata for the prompt subtype used to render one question."""
    strategy = resolve_candidate_prompt_strategy(
        dataset_name=context.dataset_name,
        question_type=context.question_type,
    )
    return {
        "question_type": context.question_type,
        "candidate_prompt_type": strategy.candidate_prompt_type,
    }


def _build_default_j1_discursive_prompt(context: CandidatePromptContext) -> str:
    return "\n\n".join(
        [
            "Você é um candidato do exame da OAB respondendo uma questão discursiva.",
            _question_block(context),
            _retrieved_context_block(context.retrieved_chunks),
            (
                "Use os trechos recuperados apenas como apoio para fundamentar a resposta.\n"
                "- Responda como candidato da OAB, em português.\n"
                "- Se o contexto não for suficiente, reconheça a limitação sem inventar normas, fatos ou jurisprudência.\n"
                "- Não mencione critérios de correção, respostas de referência ou avaliação."
            ),
            (
                "Entregue uma resposta objetiva e juridicamente fundamentada.\n"
                "Finalize com o bloco:\n"
                "Resposta final:\n"
                "<sua resposta>"
            ),
        ]
    ).strip()


def _build_default_j1_piece_prompt(context: CandidatePromptContext) -> str:
    return "\n\n".join(
        [
            "Você é um candidato da segunda fase do exame da OAB elaborando uma peça prático-profissional.",
            _question_block(context),
            _retrieved_context_block(context.retrieved_chunks),
            _piece_instruction_block(),
            (
                "Entregue somente a peça prático-profissional final.\n"
                "Finalize com o bloco:\n"
                "Resposta final:\n"
                "<peça>"
            ),
        ]
    ).strip()


def _build_default_j2_prompt(context: CandidatePromptContext) -> str:
    return "\n\n".join(
        [
            "Você é um candidato do exame da OAB respondendo uma questão de múltipla escolha.",
            _question_block(context),
            _alternatives_block(context.alternatives),
            _retrieved_context_block(context.retrieved_chunks),
            (
                "Use os trechos recuperados apenas como apoio para escolher exatamente uma alternativa.\n"
                "- Considere o enunciado e as alternativas apresentadas.\n"
                "- Se houver incerteza, escolha a melhor alternativa com base no contexto disponível.\n"
                "- Não invente normas, fatos ou jurisprudência."
            ),
            (
                "Explique sua escolha de forma breve.\n"
                "Ao final, inclua exatamente uma linha no formato:\n"
                "Alternativa final: X"
            ),
        ]
    ).strip()


def _render_template_prompt(
    *,
    context: CandidatePromptContext,
    template: CandidatePromptRecord,
    strategy: CandidatePromptStrategy,
) -> str:
    template_uses_type_placeholders = any(
        placeholder in text
        for placeholder in (
            "{candidate_prompt_type_instruction}",
            "{candidate_prompt_type}",
            "{question_type}",
            "{tipo_questao}",
        )
        for text in (
            template.persona,
            template.context,
            template.rag_instruction,
            template.output,
        )
    )
    rendered_persona = _fill_template_placeholders(
        (
            template.persona
            if template_uses_type_placeholders
            else strategy.persona_override if strategy.persona_override is not None else template.persona
        ),
        context=context,
        strategy=strategy,
    )
    rendered_context = _fill_template_placeholders(template.context, context=context, strategy=strategy)
    rendered_rag_instruction = _fill_template_placeholders(
        template.rag_instruction,
        context=context,
        strategy=strategy,
    )
    rendered_output = _fill_template_placeholders(
        (
            template.output
            if template_uses_type_placeholders
            else strategy.output_override if strategy.output_override is not None else template.output
        ),
        context=context,
        strategy=strategy,
    )
    sections = [rendered_persona, rendered_context, rendered_rag_instruction]
    if strategy.candidate_prompt_type_instruction and not template_uses_type_placeholders:
        sections.append(strategy.candidate_prompt_type_instruction)
    sections.append(rendered_output)
    return "\n\n".join(section.strip() for section in sections if section.strip())


def _question_block(context: CandidatePromptContext) -> str:
    return f"Questão original:\n```text\n{context.question_text}\n```"


def _alternatives_block(alternatives: Any) -> str:
    formatted = _format_alternatives(alternatives)
    if formatted is None:
        return "Alternativas:\n- não informadas."
    return f"Alternativas:\n{formatted}"


def _retrieved_context_block(chunks: list[RetrievedRagChunk]) -> str:
    if not chunks:
        return "Contexto jurídico recuperado:\n- nenhum trecho recuperado."
    rendered_chunks = []
    for chunk in chunks:
        header_parts = [f"[{chunk.rank}]"]
        for label, value in (
            ("Lei", chunk.lei),
            ("Norma", chunk.norma),
            ("Artigo", chunk.artigo),
            ("Tópico", chunk.topico),
            ("Fonte", chunk.url),
        ):
            if value:
                header_parts.append(f"{label}: {value}")
        header = " | ".join(header_parts)
        rendered_chunks.append(f"{header}\n{chunk.chunk_text}")
    return "Contexto jurídico recuperado:\n\n" + "\n\n".join(rendered_chunks)


def _fill_template_placeholders(
    text: str,
    *,
    context: CandidatePromptContext,
    strategy: CandidatePromptStrategy,
) -> str:
    values = {
        "{dataset}": context.dataset_name,
        "{id_pergunta}": str(context.question_id),
        "{pergunta_oab}": context.question_text,
        "{questao_original}": context.question_text,
        "{alternativas}": _format_alternatives(context.alternatives) or "- não informadas.",
        "{contexto_rag}": _retrieved_context_block(context.retrieved_chunks),
        "{retrieval_run_id}": "" if context.retrieval_run_id is None else str(context.retrieval_run_id),
        "{retrieval_name}": context.retrieval_name or "",
        "{top_k}": "" if context.top_k is None else str(context.top_k),
        "{tipo_questao}": context.question_type or "",
        "{question_type}": context.question_type or "",
        "{candidate_prompt_type}": strategy.candidate_prompt_type,
        "{candidate_prompt_type_instruction}": strategy.candidate_prompt_type_instruction,
    }
    rendered = text
    for placeholder, value in values.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _piece_instruction_block() -> str:
    return (
        "Use os trechos recuperados apenas como apoio jurídico para fundamentar a peça.\n"
        "- Atue como examinando da segunda fase da OAB e redija a medida processual mais adequada com base apenas nos fatos do caso.\n"
        "- Escreva somente a peça prático-profissional final, em português brasileiro formal e objetivo.\n"
        "- Estruture o texto como documento processual.\n"
        "- Não repita instruções, enunciado, contexto recuperado, rótulos, critérios de avaliação ou respostas de referência.\n"
        "- Não mencione RAG, chunks recuperados, contexto fornecido ou trechos recuperados.\n"
        "- Não adicione comentários metatextuais.\n"
        "- Não invente fatos, partes, números de processo, números de OAB, datas reais, endereços, documentos, artigos de lei, súmulas, precedentes, jurisprudência ou outras informações não fornecidas.\n"
        "- Se o número exato do artigo for desconhecido, fundamente com a lei aplicável ou entendimento jurídico pertinente sem inventar a numeração.\n"
        "- Se o contexto recuperado for insuficiente, prossiga apenas com o que puder ser inferido do caso, sem fabricar conteúdo.\n"
        "- Inclua apenas elementos compatíveis com a medida escolhida.\n"
        "- Quando aplicável, organize a peça com endereçamento, qualificação essencial das partes, cabimento ou tempestividade, fatos, fundamentos jurídicos, pedidos ou requerimentos e fechamento formal.\n"
        "- Use placeholders compatíveis com a OAB quando necessário: `Local, ...`, `Data, ...`, `Advogado(a): ...`, `OAB/UF: ...`."
    )


def _normalize_question_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(character for character in normalized if not unicodedata.combining(character))
    cleaned = " ".join(
        ascii_only.lower().replace("-", " ").replace("/", " ").replace("_", " ").split()
    )
    return cleaned or None


def _is_piece_question_type(normalized_question_type: str | None) -> bool:
    if normalized_question_type is None:
        return False
    tokens = set(normalized_question_type.split())
    return "peca" in tokens and "profissional" in tokens


def _is_discursive_question_type(normalized_question_type: str | None) -> bool:
    if normalized_question_type is None:
        return False
    return "questao" in normalized_question_type


def _format_alternatives(alternatives: Any) -> str | None:
    if alternatives is None:
        return None
    if isinstance(alternatives, dict):
        lines = []
        for key, value in alternatives.items():
            label = str(key).strip()
            if not label:
                continue
            lines.append(f"- {label}: {value}")
        return "\n".join(lines) if lines else None
    if isinstance(alternatives, list):
        if not alternatives:
            return None
        if all(isinstance(item, str) for item in alternatives):
            lines = []
            for index, item in enumerate(alternatives):
                option = string.ascii_uppercase[index] if index < len(string.ascii_uppercase) else str(index + 1)
                lines.append(f"- {option}: {item}")
            return "\n".join(lines)
        return json.dumps(alternatives, ensure_ascii=False, indent=2)
    return str(alternatives).strip() or None
