"""Fake candidate client for deterministic tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from atividade_2.contracts import CandidateRawResponse


@dataclass
class FakeCandidateClient:
    """Return a controlled response and capture generation calls."""

    response: CandidateRawResponse | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def generate(
        self,
        prompt: str,
        *,
        model: str,
    ) -> CandidateRawResponse:
        self.calls.append((prompt, model))
        if self.response is not None:
            return self.response
        return CandidateRawResponse(
            text="",
            provider="fake",
            model=model,
            latency_ms=0,
        )
