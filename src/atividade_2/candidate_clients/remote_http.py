"""Generic remote HTTP candidate client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from atividade_2.contracts import CandidateRawResponse


class RemoteCandidateError(RuntimeError):
    """Raised for remote candidate transport or response errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class HttpTransport(Protocol):
    """Small transport seam for offline tests."""

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: int,
    ) -> tuple[int, dict[str, Any]]:
        """POST JSON and return ``(status_code, parsed_json)``."""


@dataclass(frozen=True)
class UrllibHttpTransport:
    """stdlib urllib transport implementation."""

    max_response_bytes: int = 1_000_000

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: int,
    ) -> tuple[int, dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status_code = response.getcode()
                raw_body = response.read(self.max_response_bytes + 1)
        except urllib.error.HTTPError as error:
            raw_body = error.read(self.max_response_bytes + 1)
            raise RemoteCandidateError(
                f"Remote candidate returned HTTP {error.code}: {_safe_response_message(raw_body)}",
                status_code=error.code,
            ) from error
        except urllib.error.URLError as error:
            raise RemoteCandidateError(f"Remote candidate request failed: {error.reason}") from error
        except TimeoutError as error:
            raise RemoteCandidateError("Remote candidate request timed out.") from error

        if len(raw_body) > self.max_response_bytes:
            raise RemoteCandidateError("Remote candidate response exceeded the maximum allowed size.")
        return status_code, _parse_json_body(raw_body)


@dataclass(frozen=True)
class RemoteHttpCandidateClientConfig:
    """Configuration for the remote candidate client."""

    base_url: str
    api_key: str
    provider: str = "remote_http"
    timeout_seconds: int = 120
    temperature: float = 0.0
    max_tokens: int = 4000
    top_p: float = 1.0
    openai_compatible: bool = True
    save_raw_response: bool = False


@dataclass
class RemoteHttpCandidateClient:
    """Remote HTTP client for AV3 candidate generation."""

    config: RemoteHttpCandidateClientConfig
    transport: HttpTransport | None = None

    def generate(
        self,
        prompt: str,
        *,
        model: str,
    ) -> CandidateRawResponse:
        if not self.config.base_url:
            raise RemoteCandidateError("Candidate base_url is required.")
        if not self.config.api_key:
            raise RemoteCandidateError("Candidate api_key is required.")

        transport = self.transport or UrllibHttpTransport()
        url = _resolve_url(
            self.config.base_url,
            openai_compatible=self.config.openai_compatible,
        )
        payload = self._build_payload(prompt=prompt, model=model)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "User-Agent": "atividade-2-candidate/0.1",
        }

        started = time.monotonic()
        status_code, raw_response = transport.post(
            url,
            headers=headers,
            payload=payload,
            timeout=self.config.timeout_seconds,
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        if status_code < 200 or status_code >= 300:
            raise RemoteCandidateError(
                f"Remote candidate returned HTTP {status_code}.",
                status_code=status_code,
            )

        return CandidateRawResponse(
            text=_extract_response_text(raw_response),
            provider=self.config.provider,
            model=model,
            latency_ms=latency_ms,
            status_code=status_code,
            raw_response=raw_response if self.config.save_raw_response else None,
        )

    def _build_payload(self, *, prompt: str, model: str) -> dict[str, Any]:
        if self.config.openai_compatible:
            return {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }
        return {
            "model": model,
            "prompt": prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }


def _resolve_url(base_url: str, *, openai_compatible: bool) -> str:
    stripped = base_url.rstrip("/")
    if not openai_compatible:
        return stripped
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def _extract_response_text(raw_response: dict[str, Any]) -> str:
    choices = raw_response.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first_choice.get("text"), str):
                return first_choice["text"]

    for key in ("text", "output", "response"):
        value = raw_response.get(key)
        if isinstance(value, str):
            return value

    raise RemoteCandidateError("Remote candidate response did not contain model text.")


def _parse_json_body(raw_body: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise RemoteCandidateError("Remote candidate returned invalid JSON.") from error
    if not isinstance(parsed, dict):
        raise RemoteCandidateError("Remote candidate JSON response must be an object.")
    return parsed


def _safe_response_message(raw_body: bytes) -> str:
    try:
        parsed = json.loads(raw_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "<non-json response>"
    if isinstance(parsed, dict):
        for key in ("error", "message", "detail"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value[:300]
            if isinstance(value, dict) and isinstance(value.get("message"), str):
                return value["message"][:300]
    return "<response omitted>"
