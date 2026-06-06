"""Context-window budgeting for AV3 candidate RAG prompts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, floor
from typing import Any

from .candidate_prompts import build_candidate_prompt
from .contracts import (
    CandidateModelRuntimeProfileRecord,
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
    max_output_tokens_cap: int | None
    safety_margin_tokens: int
    chars_per_token_estimate: float
    prompt_budget_utilization: float
    source: str
    confidence: str


@dataclass(frozen=True)
class CandidatePromptBudget:
    """Audit summary for one budgeted candidate prompt."""

    context_window_tokens: int | None
    estimated_fixed_prompt_tokens: int
    estimated_context_tokens_before_budget: int
    estimated_context_tokens_after_budget: int
    max_output_tokens: int
    safety_margin_tokens: int
    chars_per_token_estimate: float
    prompt_budget_utilization: float
    safe_prompt_budget: int | None
    target_prompt_budget: int | None
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


_MODEL_DEFAULTS: dict[str, dict[str, int | float | None]] = {
    "google/gemma-2-2b-it": {
        "context_window_tokens": 8192,
        "default_max_output_tokens": 768,
        "max_output_tokens_cap": 1024,
        "safety_margin_tokens": 512,
        "chars_per_token_estimate": 3.0,
        "prompt_budget_utilization": 0.85,
    },
    "microsoft/phi-3-mini-4k-instruct": {
        "context_window_tokens": 4096,
        "default_max_output_tokens": 512,
        "max_output_tokens_cap": 512,
        "safety_margin_tokens": 512,
        "chars_per_token_estimate": 3.0,
        "prompt_budget_utilization": 0.80,
    },
    "tinyllama/tinyllama-1.1b-chat-v1.0": {
        "context_window_tokens": 2048,
        "default_max_output_tokens": 512,
        "max_output_tokens_cap": 512,
        "safety_margin_tokens": 512,
        "chars_per_token_estimate": 3.0,
        "prompt_budget_utilization": 0.75,
    },
    "qwen/qwen2.5-3b-instruct": {
        "context_window_tokens": None,
        "default_max_output_tokens": 1024,
        "max_output_tokens_cap": None,
        "safety_margin_tokens": 512,
        "chars_per_token_estimate": 4.0,
        "prompt_budget_utilization": 1.0,
    },
    "qwen/qwen2.5-7b-instruct": {
        "context_window_tokens": None,
        "default_max_output_tokens": 1500,
        "max_output_tokens_cap": None,
        "safety_margin_tokens": 512,
        "chars_per_token_estimate": 4.0,
        "prompt_budget_utilization": 1.0,
    },
    "x-ai/grok-4.3": {
        "context_window_tokens": None,
        "default_max_output_tokens": 3000,
        "max_output_tokens_cap": None,
        "safety_margin_tokens": 512,
        "chars_per_token_estimate": 4.0,
        "prompt_budget_utilization": 1.0,
    },
    "openai/gpt-5": {
        "context_window_tokens": None,
        "default_max_output_tokens": 3000,
        "max_output_tokens_cap": None,
        "safety_margin_tokens": 512,
        "chars_per_token_estimate": 4.0,
        "prompt_budget_utilization": 1.0,
    },
    "google/gemini-3.5-flash": {
        "context_window_tokens": None,
        "default_max_output_tokens": 2000,
        "max_output_tokens_cap": None,
        "safety_margin_tokens": 512,
        "chars_per_token_estimate": 4.0,
        "prompt_budget_utilization": 1.0,
    },
}

_MIN_PLAUSIBLE_CONTEXT_WINDOW_TOKENS = 1024


def estimate_tokens(text: str, *, chars_per_token_estimate: float = 4.0) -> int:
    """Estimate tokens conservatively without adding tokenizer dependencies."""
    if chars_per_token_estimate <= 0:
        raise ValueError("chars_per_token_estimate must be greater than zero.")
    return ceil(len(text) / chars_per_token_estimate)


def resolve_candidate_model_runtime_profile(
    *,
    provider: str,
    model_name: str,
    safety_margin_tokens: int,
    context_window_tokens_override: int | None = None,
    persisted_profile: CandidateModelRuntimeProfileRecord | None = None,
    catalog_context_window_tokens: int | None = None,
) -> CandidateModelRuntimeProfile:
    """Resolve runtime limits with env override, DB profile, and static fallback."""
    normalized = model_name.strip().casefold()
    static_profile = _MODEL_DEFAULTS.get(normalized)
    default_max_output = (
        int(static_profile["default_max_output_tokens"])
        if static_profile is not None
        else _default_output_tokens_for_unknown_model(provider=provider, model_name=model_name)
    )
    max_output_tokens_cap = (
        int(static_profile["max_output_tokens_cap"])
        if static_profile is not None and static_profile["max_output_tokens_cap"] is not None
        else None
    )
    context_window = (
        int(static_profile["context_window_tokens"])
        if static_profile is not None and static_profile["context_window_tokens"] is not None
        else None
    )
    resolved_safety_margin = (
        int(static_profile["safety_margin_tokens"])
        if static_profile is not None
        else int(safety_margin_tokens)
    )
    chars_per_token_estimate = (
        float(static_profile["chars_per_token_estimate"]) if static_profile is not None else 4.0
    )
    prompt_budget_utilization = (
        float(static_profile["prompt_budget_utilization"]) if static_profile is not None else 1.0
    )
    source = "static_seed" if static_profile is not None else "fallback_default"
    confidence = "seeded" if static_profile is not None else "heuristic"
    if persisted_profile is not None:
        if (
            persisted_profile.context_window_tokens is not None
            and int(persisted_profile.context_window_tokens) >= _MIN_PLAUSIBLE_CONTEXT_WINDOW_TOKENS
        ):
            context_window = int(persisted_profile.context_window_tokens)
        if persisted_profile.default_max_output_tokens is not None:
            default_max_output = int(persisted_profile.default_max_output_tokens)
        resolved_safety_margin = int(persisted_profile.safety_margin_tokens)
        max_output_tokens_cap = _metadata_optional_int(
            persisted_profile.metadata,
            "max_output_tokens_cap",
            default=max_output_tokens_cap,
        )
        chars_per_token_estimate = _metadata_float(
            persisted_profile.metadata,
            "chars_per_token_estimate",
            default=chars_per_token_estimate,
        )
        prompt_budget_utilization = _metadata_float(
            persisted_profile.metadata,
            "prompt_budget_utilization",
            default=prompt_budget_utilization,
        )
        source = persisted_profile.source
        confidence = persisted_profile.confidence
    elif catalog_context_window_tokens is not None:
        context_window = int(catalog_context_window_tokens)
        source = "provider_catalog"
        confidence = "catalog"
    if context_window_tokens_override is not None:
        context_window = int(context_window_tokens_override)
        source = "env_override"
        confidence = "explicit"
    return CandidateModelRuntimeProfile(
        provider=provider,
        model_name=model_name,
        context_window_tokens=context_window,
        default_max_output_tokens=default_max_output,
        max_output_tokens_cap=max_output_tokens_cap,
        safety_margin_tokens=max(1, resolved_safety_margin),
        chars_per_token_estimate=max(0.01, chars_per_token_estimate),
        prompt_budget_utilization=min(1.0, max(0.01, prompt_budget_utilization)),
        source=source,
        confidence=confidence,
    )


def resolve_candidate_max_output_tokens(
    *,
    profile: CandidateModelRuntimeProfile,
    requested_max_tokens: int | None,
) -> int:
    """Resolve requested/default max output before prompt-specific clamping."""
    requested_or_default = (
        int(requested_max_tokens)
        if requested_max_tokens is not None
        else int(profile.default_max_output_tokens)
    )
    if profile.max_output_tokens_cap is not None:
        requested_or_default = min(requested_or_default, int(profile.max_output_tokens_cap))
    return max(1, requested_or_default)


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
    chars_per_token_estimate: float | None = None,
    prompt_budget_utilization: float | None = None,
) -> BudgetedCandidateRetrievalContext:
    """Apply the known context budget after retrieval and before prompt rendering."""
    resolved_chars_per_token_estimate, resolved_prompt_budget_utilization = _resolve_budgeting_parameters(
        model_name=model_name,
        av3_provider=av3_provider,
        safety_margin_tokens=safety_margin_tokens,
        chars_per_token_estimate=chars_per_token_estimate,
        prompt_budget_utilization=prompt_budget_utilization,
    )
    fixed_prompt = _render_prompt(question=question, retrieval_result=retrieval_result, prompt=prompt, chunks=[])
    prompt_before_budget = _render_prompt(
        question=question,
        retrieval_result=retrieval_result,
        prompt=prompt,
        chunks=list(retrieval_result.chunks),
    )
    estimated_fixed_prompt_tokens = estimate_tokens(
        fixed_prompt,
        chars_per_token_estimate=resolved_chars_per_token_estimate,
    )
    estimated_prompt_tokens_before_budget = estimate_tokens(
        prompt_before_budget,
        chars_per_token_estimate=resolved_chars_per_token_estimate,
    )
    estimated_context_tokens_before_budget = sum(
        estimate_tokens(chunk.chunk_text, chars_per_token_estimate=resolved_chars_per_token_estimate)
        for chunk in retrieval_result.chunks
    )

    available_context_tokens: int | None = None
    safe_prompt_budget: int | None = None
    target_prompt_budget: int | None = None
    budgeted_chunks: list[BudgetedRetrievedChunk]
    if context_window_tokens is None:
        budgeted_chunks = [
            BudgetedRetrievedChunk(
                chunk=chunk,
                included_chunk_text=chunk.chunk_text,
                included_in_prompt=True,
                was_truncated=False,
                original_estimated_tokens=estimate_tokens(
                    chunk.chunk_text,
                    chars_per_token_estimate=resolved_chars_per_token_estimate,
                ),
                included_estimated_tokens=estimate_tokens(
                    chunk.chunk_text,
                    chars_per_token_estimate=resolved_chars_per_token_estimate,
                ),
                truncation_reason=None,
            )
            for chunk in retrieval_result.chunks
        ]
    else:
        safe_prompt_budget = int(context_window_tokens) - int(max_tokens) - int(safety_margin_tokens)
        if safe_prompt_budget <= 0:
            raise ValueError(
                f"Candidate max output plus safety margin exceeds the known context window for {model_name}. "
                f"max_tokens={max_tokens} "
                f"safety_margin_tokens={safety_margin_tokens} context_window_tokens={context_window_tokens}."
            )
        target_prompt_budget = floor(safe_prompt_budget * resolved_prompt_budget_utilization)
        if target_prompt_budget <= 0:
            raise ValueError(
                f"Candidate target prompt budget is not positive for {model_name}. "
                f"safe_prompt_budget={safe_prompt_budget} "
                f"prompt_budget_utilization={resolved_prompt_budget_utilization}."
            )
        available_context_tokens = target_prompt_budget - estimated_fixed_prompt_tokens
        if available_context_tokens < 0:
            raise ValueError(
                f"Fixed candidate prompt without retrieved context exceeds the target prompt budget for {model_name}. "
                f"estimated_fixed_prompt_tokens={estimated_fixed_prompt_tokens} "
                f"target_prompt_budget={target_prompt_budget} safe_prompt_budget={safe_prompt_budget} "
                f"max_tokens={max_tokens} safety_margin_tokens={safety_margin_tokens} "
                f"context_window_tokens={context_window_tokens}."
            )
        budgeted_chunks = _budget_chunks(
            chunks=retrieval_result.chunks,
            available_context_tokens=available_context_tokens,
            chars_per_token_estimate=resolved_chars_per_token_estimate,
        )

    retrieval_result_for_prompt, estimated_prompt_tokens_after_budget = _build_result_and_estimate(
        question=question,
        retrieval_result=retrieval_result,
        prompt=prompt,
        budgeted_chunks=budgeted_chunks,
        chars_per_token_estimate=resolved_chars_per_token_estimate,
    )
    if context_window_tokens is not None:
        assert target_prompt_budget is not None
        if estimated_prompt_tokens_after_budget > target_prompt_budget:
            budgeted_chunks, retrieval_result_for_prompt, estimated_prompt_tokens_after_budget = (
                _shrink_budgeted_chunks_to_fit_rendered_prompt(
                    question=question,
                    retrieval_result=retrieval_result,
                    prompt=prompt,
                    budgeted_chunks=budgeted_chunks,
                    target_prompt_budget=target_prompt_budget,
                    chars_per_token_estimate=resolved_chars_per_token_estimate,
                )
            )

    truncated_chunks = sum(1 for chunk in budgeted_chunks if chunk.was_truncated)
    dropped_chunks = sum(1 for chunk in budgeted_chunks if not chunk.included_in_prompt)
    included_chunks = sum(1 for chunk in budgeted_chunks if chunk.included_in_prompt)
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
        chars_per_token_estimate=resolved_chars_per_token_estimate,
        prompt_budget_utilization=resolved_prompt_budget_utilization,
        safe_prompt_budget=safe_prompt_budget,
        target_prompt_budget=target_prompt_budget,
        available_context_tokens=available_context_tokens,
        retrieved_chunks=len(retrieval_result.chunks),
        included_chunks=included_chunks,
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
        "prompt_budget_utilization": budget.prompt_budget_utilization,
        "chars_per_token_estimate": budget.chars_per_token_estimate,
        "safe_prompt_budget": budget.safe_prompt_budget,
        "target_prompt_budget": budget.target_prompt_budget,
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
        "prompt_budget_utilization": budgets[0].prompt_budget_utilization,
        "chars_per_token_estimate": budgets[0].chars_per_token_estimate,
        "safe_prompt_budget": budgets[0].safe_prompt_budget,
        "target_prompt_budget": budgets[0].target_prompt_budget,
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
    chars_per_token_estimate: float,
) -> list[BudgetedRetrievedChunk]:
    remaining_tokens = available_context_tokens
    budgeted: list[BudgetedRetrievedChunk] = []
    for chunk in chunks:
        original_estimated_tokens = estimate_tokens(
            chunk.chunk_text,
            chars_per_token_estimate=chars_per_token_estimate,
        )
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

        included_char_budget = max(1, int(remaining_tokens * chars_per_token_estimate))
        included_text = chunk.chunk_text[:included_char_budget].rstrip()
        included_estimated_tokens = estimate_tokens(
            included_text,
            chars_per_token_estimate=chars_per_token_estimate,
        )
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
    chars_per_token_estimate: float,
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
    return retrieval_result_for_prompt, estimate_tokens(
        prompt_after_budget,
        chars_per_token_estimate=chars_per_token_estimate,
    )


def _shrink_budgeted_chunks_to_fit_rendered_prompt(
    *,
    question: CandidateQuestionRecord,
    retrieval_result: RagRetrievalResult,
    prompt: CandidatePromptRecord,
    budgeted_chunks: list[BudgetedRetrievedChunk],
    target_prompt_budget: int,
    chars_per_token_estimate: float,
) -> tuple[list[BudgetedRetrievedChunk], RagRetrievalResult, int]:
    adjusted_chunks = list(budgeted_chunks)
    while True:
        retrieval_result_for_prompt, estimated_prompt_tokens = _build_result_and_estimate(
            question=question,
            retrieval_result=retrieval_result,
            prompt=prompt,
            budgeted_chunks=adjusted_chunks,
            chars_per_token_estimate=chars_per_token_estimate,
        )
        if estimated_prompt_tokens <= target_prompt_budget:
            return adjusted_chunks, retrieval_result_for_prompt, estimated_prompt_tokens

        last_included_index = _last_included_chunk_index(adjusted_chunks)
        if last_included_index is None:
            raise ValueError(
                "Fixed candidate prompt without retrieved context exceeds the target prompt budget. "
                f"estimated_prompt_tokens={estimated_prompt_tokens} "
                f"target_prompt_budget={target_prompt_budget}."
            )

        excess_tokens = estimated_prompt_tokens - target_prompt_budget
        adjusted_chunks[last_included_index] = _shrink_or_drop_chunk(
            adjusted_chunks[last_included_index],
            excess_tokens=excess_tokens,
            chars_per_token_estimate=chars_per_token_estimate,
        )


def _last_included_chunk_index(chunks: list[BudgetedRetrievedChunk]) -> int | None:
    for index in range(len(chunks) - 1, -1, -1):
        if chunks[index].included_in_prompt:
            return index
    return None


def _shrink_or_drop_chunk(
    chunk: BudgetedRetrievedChunk,
    *,
    excess_tokens: int,
    chars_per_token_estimate: float,
) -> BudgetedRetrievedChunk:
    excess_chars = max(1, int(ceil(excess_tokens * chars_per_token_estimate)))
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
        included_estimated_tokens=estimate_tokens(
            included_text,
            chars_per_token_estimate=chars_per_token_estimate,
        ),
        truncation_reason="context_budget",
    )


def _resolve_budgeting_parameters(
    *,
    model_name: str,
    av3_provider: str,
    safety_margin_tokens: int,
    chars_per_token_estimate: float | None,
    prompt_budget_utilization: float | None,
) -> tuple[float, float]:
    profile = resolve_candidate_model_runtime_profile(
        provider=av3_provider,
        model_name=model_name,
        safety_margin_tokens=safety_margin_tokens,
    )
    resolved_chars = (
        float(chars_per_token_estimate)
        if chars_per_token_estimate is not None
        else float(profile.chars_per_token_estimate)
    )
    resolved_utilization = (
        float(prompt_budget_utilization)
        if prompt_budget_utilization is not None
        else float(profile.prompt_budget_utilization)
    )
    return max(0.01, resolved_chars), min(1.0, max(0.01, resolved_utilization))


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
            question_type=question.question_type,
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


def _metadata_optional_int(metadata: dict[str, Any], key: str, *, default: int | None) -> int | None:
    value = metadata.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _metadata_float(metadata: dict[str, Any], key: str, *, default: float) -> float:
    value = metadata.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
