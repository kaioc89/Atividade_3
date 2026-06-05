"""Provider runtime observation parsing for AV3 candidate execution."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateRuntimeObservation:
    """Parsed context-window observation extracted from one provider error."""

    error_class: str
    error_message: str
    observed_context_window_tokens: int | None
    observed_prompt_tokens: int | None
    observed_requested_max_tokens: int | None
    observed_total_tokens: int | None


_CONTEXT_PATTERNS = (
    re.compile(r"platform records context_length as\s*(\d+)", re.IGNORECASE),
    re.compile(r"context_length as\s*(\d+)", re.IGNORECASE),
    re.compile(r"context size of\s*(\d+)", re.IGNORECASE),
    re.compile(r"context window of\s*(\d+)", re.IGNORECASE),
)
_PROMPT_TOKENS_PATTERNS = (
    re.compile(r"prompt(?: contains| has| is)?\s*(\d+)\s*tokens", re.IGNORECASE),
    re.compile(r"input(?: contains| has| is)?\s*(\d+)\s*tokens", re.IGNORECASE),
)
_REQUESTED_MAX_PATTERNS = (
    re.compile(r"max_tokens(?: requested| is| of)?\s*(\d+)", re.IGNORECASE),
    re.compile(r"requested max(?:_output)?_tokens(?: is| of)?\s*(\d+)", re.IGNORECASE),
)
_TOTAL_TOKENS_PATTERNS = (
    re.compile(r"total(?: requested)?\s*(?:tokens)?\s*(?:of|is)?\s*(\d+)", re.IGNORECASE),
    re.compile(r"combined(?: total)?\s*(?:tokens)?\s*(?:of|is)?\s*(\d+)", re.IGNORECASE),
)


def normalize_provider_model_key(provider_model_id: str) -> str:
    """Normalize provider model ids for unique/profile keys."""
    return provider_model_id.strip().casefold()


def parse_candidate_runtime_observation(error_message: str) -> CandidateRuntimeObservation | None:
    """Parse provider context-window errors into structured observations."""
    observed_context_window_tokens = _first_int_match(error_message, _CONTEXT_PATTERNS)
    if observed_context_window_tokens is None:
        return None
    observed_prompt_tokens = _first_int_match(error_message, _PROMPT_TOKENS_PATTERNS)
    observed_requested_max_tokens = _first_int_match(error_message, _REQUESTED_MAX_PATTERNS)
    observed_total_tokens = _first_int_match(error_message, _TOTAL_TOKENS_PATTERNS)
    if observed_total_tokens is None and (
        observed_prompt_tokens is not None and observed_requested_max_tokens is not None
    ):
        observed_total_tokens = observed_prompt_tokens + observed_requested_max_tokens
    return CandidateRuntimeObservation(
        error_class="context_window_exceeded",
        error_message=error_message,
        observed_context_window_tokens=observed_context_window_tokens,
        observed_prompt_tokens=observed_prompt_tokens,
        observed_requested_max_tokens=observed_requested_max_tokens,
        observed_total_tokens=observed_total_tokens,
    )


def _first_int_match(text: str, patterns: tuple[re.Pattern[str], ...]) -> int | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match is not None:
            return int(match.group(1))
    return None
