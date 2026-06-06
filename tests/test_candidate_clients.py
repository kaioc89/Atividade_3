from __future__ import annotations

from atividade_2.candidate_clients.fake import FakeCandidateClient
from atividade_2.candidate_clients.remote_http import RemoteHttpCandidateClient, RemoteHttpCandidateClientConfig
from atividade_2.contracts import CandidateRawResponse


class FakeTransport:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self.payload = payload
        self.url: str | None = None
        self.headers: dict[str, str] | None = None
        self.request_payload: dict[str, object] | None = None
        self.timeout: int | None = None

    def post(self, url: str, *, headers: dict[str, str], payload: dict[str, object], timeout: int) -> tuple[int, dict]:
        self.url = url
        self.headers = headers
        self.request_payload = payload
        self.timeout = timeout
        return self.status_code, dict(self.payload)


def test_fake_candidate_client_returns_controlled_raw_response() -> None:
    response = CandidateRawResponse(
        text="Resposta final:\nTexto controlado.",
        provider="fake",
        model="candidate-test-model",
        latency_ms=17,
        status_code=200,
        raw_response={"id": "resp-1"},
    )
    client = FakeCandidateClient(response=response)

    result = client.generate("prompt candidato", model="candidate-test-model")

    assert result == response
    assert client.calls == [("prompt candidato", "candidate-test-model")]


def test_remote_candidate_client_openai_payload_uses_only_user_message() -> None:
    transport = FakeTransport(
        200,
        {"choices": [{"message": {"content": "Alternativa final: B"}}]},
    )
    client = RemoteHttpCandidateClient(
        config=RemoteHttpCandidateClientConfig(
            base_url="https://candidate.example.invalid/v1",
            api_key="candidate-secret",
            temperature=0.2,
            max_tokens=512,
            top_p=0.95,
            openai_compatible=True,
            save_raw_response=True,
        ),
        transport=transport,
    )

    response = client.generate("prompt candidato", model="openai/gpt-5.4")

    assert transport.url == "https://candidate.example.invalid/v1/chat/completions"
    assert transport.headers is not None
    assert transport.headers["Authorization"] == "Bearer candidate-secret"
    assert transport.request_payload is not None
    assert transport.request_payload["model"] == "openai/gpt-5.4"
    assert transport.request_payload["messages"] == [{"role": "user", "content": "prompt candidato"}]
    assert response == CandidateRawResponse(
        text="Alternativa final: B",
        provider="remote_http",
        model="openai/gpt-5.4",
        latency_ms=response.latency_ms,
        status_code=200,
        raw_response={"choices": [{"message": {"content": "Alternativa final: B"}}]},
    )
    assert response.latency_ms >= 0


def test_remote_candidate_client_omits_authorization_header_when_api_key_is_empty() -> None:
    transport = FakeTransport(
        200,
        {"choices": [{"message": {"content": "Resposta final"}}]},
    )
    client = RemoteHttpCandidateClient(
        config=RemoteHttpCandidateClientConfig(
            base_url="http://localhost:8080/v1",
            api_key="",
            provider="llama_cpp",
            openai_compatible=True,
        ),
        transport=transport,
    )

    response = client.generate("prompt candidato", model="jurema-7b-q4_k_m")

    assert transport.url == "http://localhost:8080/v1/chat/completions"
    assert transport.headers is not None
    assert "Authorization" not in transport.headers
    assert response.provider == "llama_cpp"
