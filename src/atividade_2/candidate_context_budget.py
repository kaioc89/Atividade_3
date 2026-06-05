"""Context-window budgeting for AV3 candidate RAG prompts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Any

from .candidate_prompts import build_candidate_prompt
from .contracts import (
    CandidatePromptContext,
    CandidatePromptRecord,
    CandidateQuestionRecord,
    RagRetrievalResult,
    RetrievedRagChunk,
)


@dataclass(frozen=True)
class CandidateModelRuntimeProfile:
    """Runtime limits and output defaults for one candidate model."""

    provider: str
    model_name: str
    context_window_tokens: int | None
    default_max_output_tokens: int
    safety_margin_tokens: int
    source: str


@dataclass(frozen=True)
class CandidatePromptBudget:
    """Audit summary for one budgeted candidate prompt."""

    context_window_tokens: int | None
    estimated_fixed_prompt_tokens: int
    estimated_context_tokens_before_budget: int
    estimated_context_tokens_after_budget: int
    max_output_tokens: int
    safety_margin_tokens: int
    available_context_tokens: int | None
    retrieved_chunks: int
    included_chunks: int
    truncated_chunks: int
    dropped_chunks: int
    was_truncated: bool
    truncation_reason: str | None
    estimated_prompt_tokens_before_budget: int
    estimated_prompt_tokens_after_budget: int


@dataclass(frozen=True)
class BudgetedRetrievedChunk:
    """Retrieved chunk plus the exact text included in the prompt."""

    chunk: RetrievedRagChunk
    included_chunk_text: str
    included_in_prompt: bool
    was_truncated: bool
    original_estimated_tokens: int
    included_estimated_tokens: int
    truncation_reason: str | None


@dataclass(frozen=True)
class BudgetedCandidateRetrievalContext:
    """Budgeted retrieval result and prompt budget metadata."""

    retrieval_result_for_prompt: RagRetrievalResult
    budget: CandidatePromptBudget
    budgeted_chunks: list[BudgetedRetrievedChunk]


_MODEL_DEFAULTS: dict[str, tuple[int | None, int]] = {
    "google/gemma-2-2b-it": (8192, 1024),
    "qwen/qwen2.5-3b-instruct": (None, 1024),
    "qwen/qwen2.5-7b-instruct": (None, 1500),
    "x-ai/grok-4.3": (None, 3000),
    "openai/gpt-5": (None, 3000),
    "google/gemini-3.5-flash": (None, 2000),
}


def estimate_tokens(text: str) -> int:
    """Estimate tokens conservatively without adding tokenizer dependencies."""
    return ceil(len(text) / 4)


def resolve_candidate_model_runtime_profile(
    *,
    provider: str,
    model_name: str,
    safety_margin_tokens: int,
    context_window_tokens_override: int | None = None,
) -> CandidateModelRuntimeProfile:
    """Resolve known model runtime defaults without inventing unknown windows."""
    normalized = model_name.strip().casefold()
    context_window, default_max_output = _MODEL_DEFAULTS.get(
        normalized,
        (None, _default_output_tokens_for_unknown_model(provider=provider, model_name=model_name)),
    )
    source = "static_profile" if normalized in _MODEL_DEFAULTS else "fallback_default"
    if context_window_tokens_override is not None:
        context_window = int(context_window_tokens_override)
        source = "env_override"
    return CandidateModelRuntimeProfile(
        provider=provider,
        model_name=model_name,
        context_window_tokens=context_window,
        default_max_output_tokens=default_max_output,
        safety_margin_tokens=safety_margin_tokens,
        source=source,
    )


def resolve_candidate_max_output_tokens(
    *,
    profile: CandidateModelRuntimeProfile,
    requested_max_tokens: int | None,
) -> int:
    """Resolve requested/default max output before prompt-specific clamping."""
    if requested_max_tokens is not None:
        return int(requested_max_tokens)
    return int(profile.default_max_output_tokens)


def clamp_max_output_tokens_to_fixed_prompt(
    *,
    model_name: str,
    context_window_tokens: int | None,
    fixed_prompt_text: str,
    max_output_tokens: int,
    safety_margin_tokens: int,
) -> int:
    """Clamp output tokens when fixed instructions and question already consume budget."""
    if context_window_tokens is None:
        return int(max_output_tokens)
    estimated_fixed_prompt_tokens = estimate_tokens(fixed_prompt_text)
    safe_output_budget = context_window_tokens - estimated_fixed_prompt_tokens - safety_margin_tokens
    if safe_output_budget <= 0:
        raise ValueError(
            f"Fixed candidate prompt exceeds the known context window for {model_name}. "
            f"estimated_fixed_prompt_tokens={estimated_fixed_prompt_tokens} "
            f"context_window_tokens={context_window_tokens} safety_margin_tokens={safety_margin_tokens}."
        )
    return max(1, min(int(max_output_tokens), safe_output_budget))


def budget_candidate_retrieval_context(
    *,
    question: CandidateQuestionRecord,
    retrieval_result: RagRetrievalResult,
    prompt: CandidatePromptRecord,
    model_name: str,
    av3_provider: str,
    max_tokens: int,
    safety_margin_tokens: int,
    context_window_tokens: int | None,
) -> BudgetedCandidateRetrievalContext:
    """Apply the known context budget after retrieval and before prompt rendering."""
    fixed_prompt = _render_prompt(question=question, retrieval_result=retrieval_result, prompt=prompt, chunks=[])
    prompt_before_budget = _render_prompt(
        question=question,
        retrieval_result=retrieval_result,
        prompt=prompt,
        chunks=list(retrieval_result.chunks),
    )
    estimated_fixed_prompt_tokens = estimate_tokens(fixed_prompt)
    estimated_prompt_tokens_before_budget = estimate_tokens(prompt_before_budget)
    estimated_context_tokens_before_budget = sum(estimate_tokens(chunk.chunk_text) for chunk in retrieval_result.chunks)

    available_context_tokens: int | None = None
    budgeted_chunks: list[BudgetedRetrievedChunk]
    if context_window_tokens is None:
        budgeted_chunks = [
            BudgetedRetrievedChunk(
                chunk=chunk,
                included_chunk_text=chunk.chunk_text,
                included_in_prompt=True,
                was_truncated=False,
                original_estimated_tokens=estimate_tokens(chunk.chunk_text),
                included_estimated_tokens=estimate_tokens(chunk.chunk_text),
                truncation_reason=None,
            )
            for chunk in retrieval_result.chunks
        ]
    else:
        available_context_tokens = (
            context_window_tokens - estimated_fixed_prompt_tokens - int(max_tokens) - int(safety_margin_tokens)
        )
        if available_context_tokens < 0:
            raise ValueError(
                f"Fixed candidate prompt plus max output exceeds the known context window for {model_name}. "
                f"estimated_fixed_prompt_tokens={estimated_fixed_prompt_tokens} max_tokens={max_tokens} "
                f"safety_margin_tokens={safety_margin_tokens} context_window_tokens={context_window_tokens}."
            )
        budgeted_chunks = _budget_chunks(
            chunks=retrieval_result.chunks,
            available_context_tokens=available_context_tokens,
        )

    retrieval_result_for_prompt, estimated_prompt_tokens_after_budget = _build_result_and_estimate(
        question=question,
        retrieval_result=retrieval_result,
        prompt=prompt,
        budgeted_chunks=budgeted_chunks,
    )
    if context_window_tokens is not None:
        total_estimated_tokens = estimated_prompt_tokens_after_budget + int(max_tokens) + int(safety_margin_tokens)
        if total_estimated_tokens > context_window_tokens:
            budgeted_chunks, retrieval_result_for_prompt, estimated_prompt_tokens_after_budget = (
                _shrink_budgeted_chunks_to_fit_rendered_prompt(
                    question=question,
                    retrieval_result=retrieval_result,
                    prompt=prompt,
                    budgeted_chunks=budgeted_chunks,
                    context_window_tokens=context_window_tokens,
                    max_tokens=int(max_tokens),
                    safety_margin_tokens=int(safety_margin_tokens),
                )
            )

    truncated_chunks = sum(1 for chunk in budgeted_chunks if chunk.was_truncated)
    dropped_chunks = sum(1 for chunk in budgeted_chunks if not chunk.included_in_prompt)
    estimated_context_tokens_after_budget = sum(
        chunk.included_estimated_tokens for chunk in budgeted_chunks if chunk.included_in_prompt
    )
    was_truncated = truncated_chunks > 0 or dropped_chunks > 0
    budget = CandidatePromptBudget(
        context_window_tokens=context_window_tokens,
        estimated_fixed_prompt_tokens=estimated_fixed_prompt_tokens,
        estimated_context_tokens_before_budget=estimated_context_tokens_before_budget,
        estimated_context_tokens_after_budget=estimated_context_tokens_after_budget,
        max_output_tokens=int(max_tokens),
        safety_margin_tokens=int(safety_margin_tokens),
        available_context_tokens=available_context_tokens,
        retrieved_chunks=len(retrieval_result.chunks),
        included_chunks=sum(1 for chunk in budgeted_chunks if chunk.included_in_prompt),
        truncated_chunks=truncated_chunks,
        dropped_chunks=dropped_chunks,
        was_truncated=was_truncated,
        truncation_reason="context_budget" if was_truncated else None,
        estimated_prompt_tokens_before_budget=estimated_prompt_tokens_before_budget,
        estimated_prompt_tokens_after_budget=estimated_prompt_tokens_after_budget,
    )
    return BudgetedCandidateRetrievalContext(
        retrieval_result_for_prompt=retrieval_result_for_prompt,
        budget=budget,
        budgeted_chunks=budgeted_chunks,
    )


def budget_metadata_for_chunk(chunk: RetrievedRagChunk) -> dict[str, Any]:
    """Return budget metadata embedded in a prompt chunk, if present."""
    value = chunk.metadata.get("candidate_budget")
    return value if isinstance(value, dict) else {}


def budget_to_metadata(
    *,
    budget: CandidatePromptBudget,
    requested_max_tokens: int,
) -> dict[str, Any]:
    """Serialize a prompt budget for candidate run metadata."""
    return {
        "context_window_tokens": budget.context_window_tokens,
        "safety_margin_tokens": budget.safety_margin_tokens,
        "requested_max_tokens": int(requested_max_tokens),
        "final_max_tokens": budget.max_output_tokens,
        "estimated_prompt_tokens_before_budget": budget.estimated_prompt_tokens_before_budget,
        "estimated_prompt_tokens_after_budget": budget.estimated_prompt_tokens_after_budget,
        "retrieved_chunks": budget.retrieved_chunks,
        "included_chunks": budget.included_chunks,
        "truncated_chunks": budget.truncated_chunks,
        "dropped_chunks": budget.dropped_chunks,
    }


def aggregate_budget_metadata(
    *,
    budgets: list[CandidatePromptBudget],
    requested_max_tokens: int,
) -> dict[str, Any]:
    """Aggregate per-question prompt budgets into run-level metadata."""
    if not budgets:
        return {}
    return {
        "context_window_tokens": budgets[0].context_window_tokens,
        "safety_margin_tokens": budgets[0].safety_margin_tokens,
        "requested_max_tokens": int(requested_max_tokens),
        "final_max_tokens": budgets[0].max_output_tokens,
        "estimated_prompt_tokens_before_budget": max(
            budget.estimated_prompt_tokens_before_budget for budget in budgets
        ),
        "estimated_prompt_tokens_after_budget": max(
            budget.estimated_prompt_tokens_after_budget for budget in budgets
        ),
        "retrieved_chunks": sum(budget.retrieved_chunks for budget in budgets),
        "included_chunks": sum(budget.included_chunks for budget in budgets),
        "truncated_chunks": sum(budget.truncated_chunks for budget in budgets),
        "dropped_chunks": sum(budget.dropped_chunks for budget in budgets),
    }


def _budget_chunks(
    *,
    chunks: list[RetrievedRagChunk],
    available_context_tokens: int,
) -> list[BudgetedRetrievedChunk]:
    remaining_tokens = available_context_tokens
    budgeted: list[BudgetedRetrievedChunk] = []
    for chunk in chunks:
        original_estimated_tokens = estimate_tokens(chunk.chunk_text)
        if remaining_tokens <= 0:
            budgeted.append(
                BudgetedRetrievedChunk(
                    chunk=chunk,
                    included_chunk_text="",
                    included_in_prompt=False,
                    was_truncated=False,
                    original_estimated_tokens=original_estimated_tokens,
                    included_estimated_tokens=0,
                    truncation_reason="context_budget",
                )
            )
            continue
        if original_estimated_tokens <= remaining_tokens:
            budgeted.append(
                BudgetedRetrievedChunk(
                    chunk=chunk,
                    included_chunk_text=chunk.chunk_text,
                    included_in_prompt=True,
                    was_truncated=False,
                    original_estimated_tokens=original_estimated_tokens,
                    included_estimated_tokens=original_estimated_tokens,
                    truncation_reason=None,
                )
            )
            remaining_tokens -= original_estimated_tokens
            continue

        included_text = chunk.chunk_text[: remaining_tokens * 4].rstrip()
        included_estimated_tokens = estimate_tokens(included_text)
        budgeted.append(
            BudgetedRetrievedChunk(
                chunk=chunk,
                included_chunk_text=included_text,
                included_in_prompt=bool(included_text),
                was_truncated=bool(included_text),
                original_estimated_tokens=original_estimated_tokens,
                included_estimated_tokens=included_estimated_tokens,
                truncation_reason="context_budget",
            )
        )
        remaining_tokens = 0
    return budgeted


def _copy_chunk_for_prompt(budgeted: BudgetedRetrievedChunk) -> RetrievedRagChunk:
    budget_metadata = {
        "included_in_prompt": budgeted.included_in_prompt,
        "was_truncated": budgeted.was_truncated,
        "original_estimated_tokens": budgeted.original_estimated_tokens,
        "included_estimated_tokens": budgeted.included_estimated_tokens,
        "truncation_reason": budgeted.truncation_reason,
    }
    return RetrievedRagChunk(
        rank=budgeted.chunk.rank,
        chunk_id=budgeted.chunk.chunk_id,
        chunk_text=budgeted.included_chunk_text,
        source_kind=budgeted.chunk.source_kind,
        document_id=budgeted.chunk.document_id,
        document_key=budgeted.chunk.document_key,
        lei=budgeted.chunk.lei,
        norma=budgeted.chunk.norma,
        url=budgeted.chunk.url,
        urn=budgeted.chunk.urn,
        artigo=budgeted.chunk.artigo,
        topico=budgeted.chunk.topico,
        relevancia=budgeted.chunk.relevancia,
        tipo=budgeted.chunk.tipo,
        distance=budgeted.chunk.distance,
        similarity=budgeted.chunk.similarity,
        metadata={**budgeted.chunk.metadata, "candidate_budget": budget_metadata},
    )


def _build_result_and_estimate(
    *,
    question: CandidateQuestionRecord,
    retrieval_result: RagRetrievalResult,
    prompt: CandidatePromptRecord,
    budgeted_chunks: list[BudgetedRetrievedChunk],
) -> tuple[RagRetrievalResult, int]:
    included_chunks = [_copy_chunk_for_prompt(chunk) for chunk in budgeted_chunks if chunk.included_in_prompt]
    retrieval_result_for_prompt = RagRetrievalResult(
        question_id=retrieval_result.question_id,
        dataset=retrieval_result.dataset,
        retrieval_run_id=retrieval_result.retrieval_run_id,
        retrieval_name=retrieval_result.retrieval_name,
        embedding_model=retrieval_result.embedding_model,
        top_k=retrieval_result.top_k,
        status=retrieval_result.status,
        chunks=included_chunks,
    )
    prompt_after_budget = _render_prompt(
        question=question,
        retrieval_result=retrieval_result_for_prompt,
        prompt=prompt,
        chunks=included_chunks,
    )
    return retrieval_result_for_prompt, estimate_tokens(prompt_after_budget)


def _shrink_budgeted_chunks_to_fit_rendered_prompt(
    *,
    question: CandidateQuestionRecord,
    retrieval_result: RagRetrievalResult,
    prompt: CandidatePromptRecord,
    budgeted_chunks: list[BudgetedRetrievedChunk],
    context_window_tokens: int,
    max_tokens: int,
    safety_margin_tokens: int,
) -> tuple[list[BudgetedRetrievedChunk], RagRetrievalResult, int]:
    adjusted_chunks = list(budgeted_chunks)
    while True:
        retrieval_result_for_prompt, estimated_prompt_tokens = _build_result_and_estimate(
            question=question,
            retrieval_result=retrieval_result,
            prompt=prompt,
            budgeted_chunks=adjusted_chunks,
        )
        total_estimated_tokens = estimated_prompt_tokens + max_tokens + safety_margin_tokens
        if total_estimated_tokens <= context_window_tokens:
            return adjusted_chunks, retrieval_result_for_prompt, estimated_prompt_tokens

        last_included_index = _last_included_chunk_index(adjusted_chunks)
        if last_included_index is None:
            raise ValueError(
                "Budgeted candidate prompt still exceeds the known context window after dropping all retrieved context. "
                f"estimated_prompt_tokens={estimated_prompt_tokens} max_tokens={max_tokens} "
                f"safety_margin_tokens={safety_margin_tokens} context_window_tokens={context_window_tokens}."
            )

        excess_tokens = total_estimated_tokens - context_window_tokens
        adjusted_chunks[last_included_index] = _shrink_or_drop_chunk(
            adjusted_chunks[last_included_index],
            excess_tokens=excess_tokens,
        )


def _last_included_chunk_index(chunks: list[BudgetedRetrievedChunk]) -> int | None:
    for index in range(len(chunks) - 1, -1, -1):
        if chunks[index].included_in_prompt:
            return index
    return None


def _shrink_or_drop_chunk(chunk: BudgetedRetrievedChunk, *, excess_tokens: int) -> BudgetedRetrievedChunk:
    excess_chars = max(4, excess_tokens * 4)
    if len(chunk.included_chunk_text) <= excess_chars:
        return replace(
            chunk,
            included_chunk_text="",
            included_in_prompt=False,
            was_truncated=False,
            included_estimated_tokens=0,
            truncation_reason="context_budget",
        )
    included_text = chunk.included_chunk_text[:-excess_chars].rstrip()
    if not included_text:
        return replace(
            chunk,
            included_chunk_text="",
            included_in_prompt=False,
            was_truncated=False,
            included_estimated_tokens=0,
            truncation_reason="context_budget",
        )
    return replace(
        chunk,
        included_chunk_text=included_text,
        included_in_prompt=True,
        was_truncated=True,
        included_estimated_tokens=estimate_tokens(included_text),
        truncation_reason="context_budget",
    )


def _render_prompt(
    *,
    question: CandidateQuestionRecord,
    retrieval_result: RagRetrievalResult,
    prompt: CandidatePromptRecord,
    chunks: list[RetrievedRagChunk],
) -> str:
    return build_candidate_prompt(
        CandidatePromptContext(
            question_id=question.question_id,
            dataset_name=question.dataset,
            question_text=question.question_text,
            retrieved_chunks=chunks,
            alternatives=question.alternatives,
            retrieval_run_id=retrieval_result.retrieval_run_id,
            retrieval_name=retrieval_result.retrieval_name,
            top_k=retrieval_result.top_k,
        ),
        template=prompt,
    )


def _default_output_tokens_for_unknown_model(*, provider: str, model_name: str) -> int:
    upper_name = model_name.upper()
    if any(token in upper_name for token in ("1.5B", "1B", "2B", "3B")):
        return 1024
    if any(token in upper_name for token in ("7B", "8B")):
        return 1500
    if provider.casefold() == "openrouter":
        return 3000
    return 1024
