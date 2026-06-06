from __future__ import annotations

import io
import urllib.error
from typing import Any

import pytest

from atividade_2.config import load_settings
from atividade_2.judge_clients.remote_http import (
    RemoteHttpJudgeClient,
    RemoteJudgeError,
    UrllibHttpTransport,
    is_remote_model_capacity_error,
)


class FakeTransport:
    def __init__(self, status_code: int, response: dict[str, Any]) -> None:
        self.status_code = status_code
        self.response = response
        self.payload: dict[str, Any] | None = None
        self.headers: dict[str, str] | None = None
        self.url: str | None = None

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: int,
    ) -> tuple[int, dict[str, Any]]:
        self.url = url
        self.payload = payload
        self.headers = headers
        return self.status_code, self.response


def test_remote_client_sends_effective_model() -> None:
    settings = load_settings(
        dotenv_path=None,
        env={
            "REMOTE_JUDGE_BASE_URL": "https://example.invalid/v1",
            "REMOTE_JUDGE_API_KEY": "secret",
        },
    )
    transport = FakeTransport(
        200,
        {"choices": [{"message": {"content": '{"score": 5, "rationale": "ok"}'}}]},
    )
    client = RemoteHttpJudgeClient(settings, transport=transport)

    response = client.judge("prompt", "provider/model")

    assert response.text == '{"score": 5, "rationale": "ok"}'
    assert transport.payload is not None
    assert transport.payload["model"] == "provider/model"
    assert transport.headers is not None
    assert transport.headers["Authorization"] == "Bearer secret"
    assert transport.headers["User-Agent"] == "atividade-2-judge/0.1"


def test_remote_client_uses_per_judge_endpoint_for_requested_alias() -> None:
    settings = load_settings(
        dotenv_path=None,
        env={
            "REMOTE_JUDGE_BASE_URL": "https://default.example.invalid/v1",
            "REMOTE_JUDGE_API_KEY": "default-secret",
            "REMOTE_JUDGE_GPT_OSS_120B_BASE_URL": "https://gpt.example.invalid/v1",
            "REMOTE_JUDGE_GPT_OSS_120B_API_KEY": "gpt-secret",
        },
    )
    transport = FakeTransport(
        200,
        {"choices": [{"message": {"content": '{"score": 5, "rationale": "ok"}'}}]},
    )
    client = RemoteHttpJudgeClient(settings, transport=transport)

    client.judge("prompt", "openai/gpt-oss-120b", requested_model="gpt-oss-120b")

    assert transport.url == "https://gpt.example.invalid/v1/chat/completions"
    assert transport.headers is not None
    assert transport.headers["Authorization"] == "Bearer gpt-secret"


def test_remote_client_matches_per_judge_endpoint_by_provider_model_leaf() -> None:
    settings = load_settings(
        dotenv_path=None,
        env={
            "REMOTE_JUDGE_BASE_URL": "https://default.example.invalid/v1",
            "REMOTE_JUDGE_API_KEY": "default-secret",
            "REMOTE_JUDGE_GPT_OSS_120B_BASE_URL": "https://gpt.example.invalid/v1",
            "REMOTE_JUDGE_GPT_OSS_120B_API_KEY": "gpt-secret",
        },
    )
    transport = FakeTransport(
        200,
        {"choices": [{"message": {"content": '{"score": 5, "rationale": "ok"}'}}]},
    )
    client = RemoteHttpJudgeClient(settings, transport=transport)

    client.judge("prompt", "openai/gpt-oss-120b", requested_model="openai/gpt-oss-120b")

    assert transport.url == "https://gpt.example.invalid/v1/chat/completions"
    assert transport.headers is not None
    assert transport.headers["Authorization"] == "Bearer gpt-secret"


def test_remote_client_prefers_judge_slot_endpoint_key_over_model_endpoint() -> None:
    settings = load_settings(
        dotenv_path=None,
        env={
            "REMOTE_JUDGE_BASE_URL": "https://judge.example.invalid/v1",
            "REMOTE_JUDGE_API_KEY": "judge-secret",
            "REMOTE_JUDGE_GPT_OSS_120B_BASE_URL": "https://gpt.example.invalid/v1",
            "REMOTE_JUDGE_GPT_OSS_120B_API_KEY": "gpt-secret",
        },
    )
    transport = FakeTransport(
        200,
        {"choices": [{"message": {"content": '{"score": 5, "rationale": "ok"}'}}]},
    )
    client = RemoteHttpJudgeClient(settings, transport=transport)

    client.judge(
        "prompt",
        "openai/gpt-oss-120b",
        requested_model="openai/gpt-oss-120b",
        endpoint_key="JUDGE",
    )

    assert transport.url == "https://judge.example.invalid/v1/chat/completions"
    assert transport.headers is not None
    assert transport.headers["Authorization"] == "Bearer judge-secret"


def test_remote_client_routes_2plus1_models_to_role_endpoints() -> None:
    settings = load_settings(
        dotenv_path=None,
        env={
            "REMOTE_JUDGE_BASE_URL": "https://openrouter.ai/api/v1",
            "REMOTE_JUDGE_API_KEY": "openrouter-secret",
            "REMOTE_SECONDARY_JUDGE_BASE_URL": "https://api.groq.com/openai/v1",
            "REMOTE_SECONDARY_JUDGE_API_KEY": "groq-secret",
            "REMOTE_ARBITER_JUDGE_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai",
            "REMOTE_ARBITER_JUDGE_API_KEY": "gemini-secret",
        },
    )
    transport = FakeTransport(
        200,
        {"choices": [{"message": {"content": '{"score": 5, "rationale": "ok"}'}}]},
    )
    client = RemoteHttpJudgeClient(settings, transport=transport)

    client.judge(
        "prompt",
        "gemini-2.5-flash",
        requested_model="gemini-2.5-flash",
        endpoint_key="ARBITER",
    )

    assert transport.url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert transport.payload is not None
    assert transport.payload["model"] == "gemini-2.5-flash"
    assert transport.headers is not None
    assert transport.headers["Authorization"] == "Bearer gemini-secret"

    client.judge(
        "prompt",
        "llama-3.3-70b-versatile",
        requested_model="llama-3.3-70b-versatile",
        endpoint_key="SECONDARY_JUDGE",
    )

    assert transport.url == "https://api.groq.com/openai/v1/chat/completions"
    assert transport.payload["model"] == "llama-3.3-70b-versatile"
    assert transport.headers["Authorization"] == "Bearer groq-secret"

    client.judge(
        "prompt",
        "openai/gpt-oss-120b:free",
        requested_model="openai/gpt-oss-120b:free",
        endpoint_key="JUDGE",
    )

    assert transport.url == "https://openrouter.ai/api/v1/chat/completions"
    assert transport.payload["model"] == "openai/gpt-oss-120b:free"
    assert transport.headers["Authorization"] == "Bearer openrouter-secret"


def test_remote_client_falls_back_to_global_endpoint_without_role_endpoint() -> None:
    settings = load_settings(
        dotenv_path=None,
        env={
            "REMOTE_JUDGE_BASE_URL": "https://api.groq.com/openai/v1",
            "REMOTE_JUDGE_API_KEY": "global-secret",
        },
    )
    transport = FakeTransport(
        200,
        {"choices": [{"message": {"content": '{"score": 5, "rationale": "ok"}'}}]},
    )
    client = RemoteHttpJudgeClient(settings, transport=transport)

    client.judge("prompt", "llama-3.3-70b-versatile", endpoint_key="SECONDARY_JUDGE")

    assert transport.url == "https://api.groq.com/openai/v1/chat/completions"
    assert transport.headers is not None
    assert transport.headers["Authorization"] == "Bearer global-secret"


def test_remote_client_handles_non_2xx() -> None:
    settings = load_settings(
        dotenv_path=None,
        env={
            "REMOTE_JUDGE_BASE_URL": "https://example.invalid/v1",
            "REMOTE_JUDGE_API_KEY": "secret",
        },
    )
    client = RemoteHttpJudgeClient(settings, transport=FakeTransport(500, {"error": "down"}))

    with pytest.raises(RemoteJudgeError, match="HTTP 500") as error:
        client.judge("prompt", "provider/model")
    assert error.value.status_code == 500
    assert error.value.retryable is True


def test_remote_client_classifies_capacity_503_as_non_retryable() -> None:
    settings = load_settings(
        dotenv_path=None,
        env={
            "REMOTE_JUDGE_BASE_URL": "https://example.invalid/v1",
            "REMOTE_JUDGE_API_KEY": "secret",
        },
    )
    client = RemoteHttpJudgeClient(
        settings,
        transport=FakeTransport(
            503,
            {
                "error": {
                    "message": (
                        "Unbabel/M-Prometheus-14B is temporarily at capacity. "
                        "Please try again shortly."
                    )
                }
            },
        ),
    )

    with pytest.raises(RemoteJudgeError, match="HTTP 503") as error:
        client.judge("prompt", "Unbabel/M-Prometheus-14B")
    assert error.value.status_code == 503
    assert error.value.retryable is False
    assert is_remote_model_capacity_error(error.value) is True


def test_capacity_error_detection_accepts_retry_later_wording_for_503() -> None:
    error = RemoteJudgeError(
        "Remote judge returned HTTP 503: Please try again shortly.",
        status_code=503,
        retryable=True,
    )

    assert is_remote_model_capacity_error(error) is True


def test_capacity_error_detection_does_not_match_generic_503() -> None:
    error = RemoteJudgeError(
        "Remote judge returned HTTP 503: upstream unavailable",
        status_code=503,
        retryable=True,
    )

    assert is_remote_model_capacity_error(error) is False


def test_remote_client_classifies_auth_and_missing_model_errors_as_non_retryable() -> None:
    settings = load_settings(
        dotenv_path=None,
        env={
            "REMOTE_JUDGE_BASE_URL": "https://example.invalid/v1",
            "REMOTE_JUDGE_API_KEY": "secret",
        },
    )
    client = RemoteHttpJudgeClient(settings, transport=FakeTransport(403, {"error": "forbidden"}))

    with pytest.raises(RemoteJudgeError, match="HTTP 403") as error:
        client.judge("prompt", "provider/model")
    assert error.value.status_code == 403
    assert error.value.retryable is False


def test_urllib_transport_preserves_retry_after_for_429(monkeypatch: pytest.MonkeyPatch) -> None:
    http_error = urllib.error.HTTPError(
        "https://example.invalid/v1/chat/completions",
        429,
        "Too Many Requests",
        {"Retry-After": "7"},
        io.BytesIO(b'{"error":"rate limited"}'),
    )

    def raise_http_error(*args: Any, **kwargs: Any) -> None:
        raise http_error

    monkeypatch.setattr(urllib.request, "urlopen", raise_http_error)

    with pytest.raises(RemoteJudgeError, match="HTTP 429") as error:
        UrllibHttpTransport().post(
            "https://example.invalid/v1/chat/completions",
            headers={},
            payload={"model": "provider/model"},
            timeout=1,
        )
    assert error.value.status_code == 429
    assert error.value.retry_after_seconds == 7
    assert error.value.retryable is True


def test_urllib_transport_classifies_daily_token_quota_429_as_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (
        b'{"error":{"message":"Rate limit reached for model `llama-3.3-70b-versatile` '
        b'in organization `org_123` service tier `on_demand` on tokens per day (TPD): '
        b'Limit 100000, Used 99787, Requested 1094. Please try again in 12m41.184s."}}'
    )
    http_error = urllib.error.HTTPError(
        "https://example.invalid/v1/chat/completions",
        429,
        "Too Many Requests",
        {"Retry-After": "761"},
        io.BytesIO(body),
    )

    def raise_http_error(*args: Any, **kwargs: Any) -> None:
        raise http_error

    monkeypatch.setattr(urllib.request, "urlopen", raise_http_error)

    with pytest.raises(RemoteJudgeError, match="HTTP 429") as error:
        UrllibHttpTransport().post(
            "https://example.invalid/v1/chat/completions",
            headers={},
            payload={"model": "llama-3.3-70b-versatile"},
            timeout=1,
        )
    assert error.value.status_code == 429
    assert error.value.retry_after_seconds == 761
    assert error.value.retryable is False


def test_remote_client_requires_text_in_response() -> None:
    settings = load_settings(
        dotenv_path=None,
        env={
            "REMOTE_JUDGE_BASE_URL": "https://example.invalid/v1",
            "REMOTE_JUDGE_API_KEY": "secret",
        },
    )
    client = RemoteHttpJudgeClient(settings, transport=FakeTransport(200, {"choices": [{}]}))

    with pytest.raises(RemoteJudgeError, match="model text"):
        client.judge("prompt", "provider/model")
