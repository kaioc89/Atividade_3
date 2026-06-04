"""Candidate-safe AV3 prompt rendering for Com_RAG answer generation."""

from __future__ import annotations

import json
import string
from typing import Any

from .contracts import CandidatePromptContext, CandidatePromptRecord, RetrievedRagChunk


def build_candidate_prompt(
    context: CandidatePromptContext,
    *,
    template: CandidatePromptRecord | None = None,
) -> str:
    """Render one candidate prompt from safe question data and retrieved chunks."""
    if template is not None:
        return _render_template_prompt(context=context, template=template)
    if is_j2_candidate_context(context):
        return _build_default_j2_prompt(context)
    return _build_default_j1_prompt(context)


def is_j2_candidate_context(context: CandidatePromptContext) -> bool:
    """Return whether the prompt targets the objective multiple-choice dataset."""
    return context.dataset_name.upper() in {"J2", "OAB_EXAMES"}


def _build_default_j1_prompt(context: CandidatePromptContext) -> str:
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
) -> str:
    sections = [
        _fill_template_placeholders(template.persona, context=context),
        _fill_template_placeholders(template.context, context=context),
        _fill_template_placeholders(template.rag_instruction, context=context),
        _fill_template_placeholders(template.output, context=context),
    ]
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


def _fill_template_placeholders(text: str, *, context: CandidatePromptContext) -> str:
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
    }
    rendered = text
    for placeholder, value in values.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


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
