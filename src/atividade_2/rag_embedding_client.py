"""OpenAI-compatible embedding client helpers for AV3."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from socket import timeout as SocketTimeoutError
from urllib.parse import urlparse


@dataclass(frozen=True)
class EmbeddingBatchResult:
    vectors: list[list[float]]
    latency_ms: int
    endpoint_url: str
    endpoint_host: str


class EmbeddingProviderError(RuntimeError):
    """Raised when the embedding provider request fails."""


def request_openai_compatible_embeddings(
    *,
    api_base_url: str,
    api_key: str,
    model_name: str,
    texts: list[str],
    dimensions: int | None,
    timeout_seconds: int = 60,
    retry_attempts: int = 3,
    retry_backoff_seconds: float = 0.75,
) -> EmbeddingBatchResult:
    if not api_key:
        raise EmbeddingProviderError("Embedding API key is required.")
    if not texts:
        raise EmbeddingProviderError("At least one text is required to request embeddings.")

    url = _embeddings_url(api_base_url)
    payload: dict[str, object] = {
        "input": texts,
        "model": model_name,
    }
    if dimensions is not None:
        payload["dimensions"] = dimensions

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "atividade-2-rag-embeddings/0.1",
        },
        method="POST",
    )
    started_at = time.perf_counter()
    retry_attempts = max(1, int(retry_attempts))
    last_error: Exception | None = None
    raw_body = b""
    for attempt in range(1, retry_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw_body = response.read(8_000_001)
            break
        except urllib.error.HTTPError as error:
            detail = _read_http_error_body(error)
            raise EmbeddingProviderError(
                f"Embedding provider returned HTTP {error.code}{': ' + detail if detail else ''}."
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError, SocketTimeoutError) as error:
            last_error = error
            if attempt < retry_attempts and _is_retryable_transport_error(error):
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
                continue
            attempt_label = f" after {attempt} attempt(s)" if attempt > 1 else ""
            raise EmbeddingProviderError(f"Embedding request failed{attempt_label}: {error}") from error
    else:
        attempt_label = f" after {retry_attempts} attempt(s)"
        raise EmbeddingProviderError(f"Embedding request failed{attempt_label}: {last_error}")

    latency_ms = int((time.perf_counter() - started_at) * 1000)

    if len(raw_body) > 8_000_000:
        raise EmbeddingProviderError("Embedding response exceeded the maximum allowed size.")
    try:
        parsed = json.loads(raw_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EmbeddingProviderError("Embedding provider returned invalid JSON.") from error

    vectors = _extract_embedding_vectors(parsed)
    if len(vectors) != len(texts):
        raise EmbeddingProviderError(
            f"Embedding provider returned {len(vectors)} vectors for {len(texts)} inputs."
        )
    return EmbeddingBatchResult(
        vectors=vectors,
        latency_ms=latency_ms,
        endpoint_url=url,
        endpoint_host=urlparse(url).netloc,
    )


def _embeddings_url(api_base_url: str) -> str:
    normalized = api_base_url.rstrip("/")
    if normalized.endswith("/embeddings"):
        return normalized
    return f"{normalized}/embeddings"


def _extract_embedding_vectors(payload: object) -> list[list[float]]:
    if not isinstance(payload, dict):
        raise EmbeddingProviderError("Embedding provider JSON response must be an object.")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise EmbeddingProviderError("Embedding provider response does not contain a data array.")
    vectors: list[list[float]] = []
    for item in data:
        if not isinstance(item, dict):
            raise EmbeddingProviderError("Embedding provider response item must be an object.")
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingProviderError("Embedding provider response does not contain a valid embedding vector.")
        vectors.append(embedding)
    return vectors


def _read_http_error_body(error: urllib.error.HTTPError) -> str | None:
    try:
        raw = error.read(10_001)
    except Exception:
        return None
    if not raw:
        return None
    try:
        text = raw.decode("utf-8", errors="replace").strip()
    except Exception:
        return None
    return text[:400]


def _is_retryable_transport_error(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, SocketTimeoutError)):
        return True
    if isinstance(error, urllib.error.URLError):
        return _is_retryable_transport_error(error.reason if isinstance(error.reason, Exception) else OSError(str(error.reason)))
    if isinstance(error, OSError):
        if getattr(error, "errno", None) == 32:
            return True
        message = str(error).lower()
        return "broken pipe" in message or "timed out" in message or "timeout" in message
    message = str(error).lower()
    return "broken pipe" in message or "timed out" in message or "timeout" in message
