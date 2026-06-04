"""Candidate client protocol."""

from __future__ import annotations

from typing import Protocol

from atividade_2.contracts import CandidateRawResponse


class CandidateClient(Protocol):
    """Provider-agnostic candidate generation boundary."""

    def generate(
        self,
        prompt: str,
        *,
        model: str,
    ) -> CandidateRawResponse:
        """Execute one candidate prompt with the resolved provider model id."""
