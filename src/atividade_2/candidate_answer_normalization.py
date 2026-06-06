"""Normalization helpers for AV3 candidate answers used as judge input."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .contracts import CandidateAnswerContext

_LEADING_RESPOSTA_FINAL_PATTERN = re.compile(r"^\s*resposta\s+final:\s*", re.IGNORECASE)
_LEADING_ENGLISH_FINAL_ANSWER_PATTERN = re.compile(r"^\s*here\s+is\s+the\s+final\s+answer:\s*", re.IGNORECASE)
_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|im_end\|>")
_PLACEHOLDER_PATTERN = re.compile(r"<sua resposta>", re.IGNORECASE)
_MARKDOWN_FENCE_LINE_PATTERN = re.compile(r"(?m)^[ \t]*```[^\n`]*\n?|^[ \t]*```[ \t]*$")
_EXCESS_BLANK_LINES_PATTERN = re.compile(r"\n(?:[ \t]*\n){2,}")
_REPEATED_SYMBOL_PATTERN = re.compile(r"([^\w\s])\1{4,}")
_SHORT_TOKEN_LOOP_PATTERN = re.compile(r"\b(\w{1,12})(?:\s+\1){4,}\b", re.IGNORECASE)


@dataclass(frozen=True)
class CandidateAnswerNormalizationResult:
    """Deterministic judge-input normalization output."""

    normalized_answer: str
    normalization_flags: dict[str, bool]


def normalize_candidate_answer_for_judge(raw_answer: str) -> CandidateAnswerNormalizationResult:
    """Remove presentation artifacts without repairing or improving answer content."""
    normalized_answer = raw_answer.strip()
    removed_resposta_final_label = False
    removed_english_final_answer_wrapper = False
    removed_special_tokens = False
    removed_markdown_fences = False
    removed_placeholder_answer = False
    collapsed_excess_blank_lines = False

    normalized_answer, removed_resposta_final_label = _remove_leading_pattern(
        normalized_answer,
        _LEADING_RESPOSTA_FINAL_PATTERN,
    )
    normalized_answer, removed_english_final_answer_wrapper = _remove_leading_pattern(
        normalized_answer,
        _LEADING_ENGLISH_FINAL_ANSWER_PATTERN,
    )
    normalized_answer, removed_special_tokens = _remove_pattern(normalized_answer, _SPECIAL_TOKEN_PATTERN)
    normalized_answer, removed_placeholder_answer = _remove_pattern(normalized_answer, _PLACEHOLDER_PATTERN)
    normalized_answer, removed_markdown_fences = _remove_markdown_fences(normalized_answer)
    normalized_answer, collapsed_excess_blank_lines = _collapse_excess_blank_lines(normalized_answer)
    normalized_answer = normalized_answer.strip()

    normalization_flags = {
        "raw_answer_preserved": True,
        "removed_resposta_final_label": removed_resposta_final_label,
        "removed_english_final_answer_wrapper": removed_english_final_answer_wrapper,
        "removed_special_tokens": removed_special_tokens,
        "removed_markdown_fences": removed_markdown_fences,
        "removed_placeholder_answer": removed_placeholder_answer,
        "collapsed_excess_blank_lines": collapsed_excess_blank_lines,
        "detected_degenerate_output": _detect_degenerate_output(normalized_answer),
    }
    return CandidateAnswerNormalizationResult(
        normalized_answer=normalized_answer,
        normalization_flags=normalization_flags,
    )


def normalize_candidate_answer_context_for_judge(context: CandidateAnswerContext) -> CandidateAnswerContext:
    """Return a judge-input clone with normalized answer text and preserved raw answer."""
    result = normalize_candidate_answer_for_judge(context.candidate_answer)
    raw_candidate_answer = context.raw_candidate_answer or context.candidate_answer
    return replace(
        context,
        candidate_answer=result.normalized_answer,
        raw_candidate_answer=raw_candidate_answer,
        candidate_answer_normalization_flags=result.normalization_flags,
    )


def _remove_leading_pattern(value: str, pattern: re.Pattern[str]) -> tuple[str, bool]:
    updated_value, count = pattern.subn("", value, count=1)
    return updated_value, count > 0


def _remove_pattern(value: str, pattern: re.Pattern[str]) -> tuple[str, bool]:
    updated_value, count = pattern.subn("", value)
    return updated_value, count > 0


def _remove_markdown_fences(value: str) -> tuple[str, bool]:
    updated_value, count = _MARKDOWN_FENCE_LINE_PATTERN.subn("", value)
    return updated_value, count > 0


def _collapse_excess_blank_lines(value: str) -> tuple[str, bool]:
    updated_value, count = _EXCESS_BLANK_LINES_PATTERN.subn("\n\n", value)
    return updated_value, count > 0


def _detect_degenerate_output(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return True
    if _REPEATED_SYMBOL_PATTERN.search(stripped):
        return True
    if _SHORT_TOKEN_LOOP_PATTERN.search(stripped):
        return True
    return False
