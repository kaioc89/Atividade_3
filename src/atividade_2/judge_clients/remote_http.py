"""Generic remote HTTP judge client."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from atividade_2.contracts import JudgeRawResponse, JudgeSettings


class RemoteJudgeError(RuntimeError):
    """Raised for remote judge transport or response errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        headers: dict[str, str] | None = None,
        retry_after_seconds: float | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}
        self.retry_after_seconds = retry_after_seconds
        self.retryable = retryable


def is_remote_model_capacity_error(error: Exception) -> bool:
    """Return True for provider capacity errors that should skip the model for this run."""

    if not isinstance(error, RemoteJudgeError):
        return False
    normalized = str(error).lower()
    has_capacity_phrase = "temporarily at capacity" in normalized
    has_retry_phrase = "please try again shortly" in normalized
    if error.status_code == 503:
        return has_capacity_phrase or has_retry_phrase
    return has_capacity_phrase and has_retry_phrase


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
            message = _safe_response_message(raw_body)
            headers = dict(error.headers.items()) if error.headers is not None else {}
            raise RemoteJudgeError(
                f"Remote judge returned HTTP {error.code}: {message}",
                status_code=error.code,
                headers=headers,
                retry_after_seconds=_parse_retry_after(headers.get("Retry-After")),
                retryable=_is_retryable_http_error(error.code, message),
            ) from error
        except urllib.error.URLError as error:
            raise RemoteJudgeError(f"Remote judge request failed: {error.reason}", retryable=True) from error
        except TimeoutError as error:
            raise RemoteJudgeError("Remote judge request timed out.", retryable=True) from error

        if len(raw_body) > self.max_response_bytes:
            raise RemoteJudgeError("Remote judge response exceeded the maximum allowed size.")
        return status_code, _parse_json_body(raw_body)


@dataclass
class RemoteHttpJudgeClient:
    """Remote HTTP judge client with OpenAI-compatible request support."""

    settings: JudgeSettings
    transport: HttpTransport | None = None

    def judge(
        self,
        prompt: str,
        model: str,
        *,
        requested_model: str | None = None,
        endpoint_key: str | None = None,
    ) -> JudgeRawResponse:
        endpoint = self._resolve_endpoint(
            model=model,
            requested_model=requested_model,
            endpoint_key=endpoint_key,
        )
        if not endpoint.base_url:
            raise RemoteJudgeError("REMOTE_JUDGE_BASE_URL is required.")
        if not endpoint.api_key:
            raise RemoteJudgeError("REMOTE_JUDGE_API_KEY is required.")

        transport = self.transport or UrllibHttpTransport()
        url = _resolve_url(
            endpoint.base_url,
            openai_compatible=self.settings.remote_judge_openai_compatible,
        )
        payload = self._build_payload(prompt=prompt, model=model)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {endpoint.api_key}",
            "User-Agent": "atividade-2-judge/0.1",
        }

        started = time.monotonic()
        status_code, raw_response = transport.post(
            url,
            headers=headers,
            payload=payload,
            timeout=self.settings.remote_judge_timeout_seconds,
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        if status_code < 200 or status_code >= 300:
            message = _response_message(raw_response)
            raise RemoteJudgeError(
                f"Remote judge returned HTTP {status_code}: {message}",
                status_code=status_code,
                retryable=_is_retryable_http_error(status_code, message),
            )

        text = _extract_response_text(raw_response)
        return JudgeRawResponse(
            text=text,
            provider="remote_http",
            model=model,
            latency_ms=latency_ms,
            status_code=status_code,
            raw_response=raw_response if self.settings.judge_save_raw_response else None,
        )

    def _build_payload(self, *, prompt: str, model: str) -> dict[str, Any]:
        if self.settings.remote_judge_openai_compatible:
            return {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Você é um avaliador jurídico. Responda somente JSON válido.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.settings.remote_judge_temperature,
                "max_tokens": self.settings.remote_judge_max_tokens,
                "top_p": self.settings.remote_judge_top_p,
            }
        return {
            "model": model,
            "prompt": prompt,
            "temperature": self.settings.remote_judge_temperature,
            "max_tokens": self.settings.remote_judge_max_tokens,
            "top_p": self.settings.remote_judge_top_p,
        }

    def _resolve_endpoint(
        self,
        *,
        model: str,
        requested_model: str | None,
        endpoint_key: str | None,
    ) -> "_RemoteEndpoint":
        if endpoint_key:
            normalized_endpoint_key = _endpoint_key(endpoint_key)
            if normalized_endpoint_key == "JUDGE":
                return _RemoteEndpoint(
                    base_url=self.settings.remote_judge_base_url,
                    api_key=self.settings.remote_judge_api_key,
                )
            endpoint = self.settings.remote_judge_endpoints.get(normalized_endpoint_key)
            if endpoint is not None:
                return _RemoteEndpoint(base_url=endpoint.base_url, api_key=endpoint.api_key)
        for candidate in (requested_model, model):
            if not candidate:
                continue
            for endpoint_key in _endpoint_keys(candidate):
                endpoint = self.settings.remote_judge_endpoints.get(endpoint_key)
                if endpoint is not None:
                    return _RemoteEndpoint(base_url=endpoint.base_url, api_key=endpoint.api_key)
        return _RemoteEndpoint(
            base_url=self.settings.remote_judge_base_url,
            api_key=self.settings.remote_judge_api_key,
        )


@dataclass(frozen=True)
class _RemoteEndpoint:
    base_url: str | None
    api_key: str | None


def _resolve_url(base_url: str, *, openai_compatible: bool) -> str:
    stripped = base_url.rstrip("/")
    if not openai_compatible:
        return stripped
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def _endpoint_keys(model: str) -> tuple[str, ...]:
    keys = [_endpoint_key(model)]
    if "/" in model:
        keys.append(_endpoint_key(model.rsplit("/", 1)[-1]))
    return tuple(dict.fromkeys(key for key in keys if key))


def _endpoint_key(model: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", model.upper()).strip("_")


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

    raise RemoteJudgeError("Remote judge response did not contain model text.")


def _parse_json_body(raw_body: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise RemoteJudgeError("Remote judge returned invalid JSON.") from error
    if not isinstance(parsed, dict):
        raise RemoteJudgeError("Remote judge JSON response must be an object.")
    return parsed


def _safe_response_message(raw_body: bytes) -> str:
    try:
        parsed = json.loads(raw_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "<non-json response>"
    if isinstance(parsed, dict):
        return _response_message(parsed)
    return "<response omitted>"


def _response_message(payload: dict[str, Any]) -> str:
    for key in ("error", "message", "detail"):
        value = payload.get(key)
        if isinstance(value, str):
            return value[:300]
        if isinstance(value, dict) and isinstance(value.get("message"), str):
            return value["message"][:300]
    return "<response omitted>"


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def _is_retryable_status(status_code: int | None) -> bool:
    return status_code == 429 or (status_code is not None and 500 <= status_code <= 599)


def _is_retryable_http_error(status_code: int | None, message: str) -> bool:
    if status_code == 429 and _is_daily_token_quota_error(message):
        return False
    if status_code == 503 and (
        "temporarily at capacity" in message.lower()
        or "please try again shortly" in message.lower()
    ):
        return False
    return _is_retryable_status(status_code)


def _is_daily_token_quota_error(message: str) -> bool:
    normalized = message.lower()
    return (
        "tokens per day" in normalized
        or "(tpd)" in normalized
        or " on tokens per day " in normalized
    )
