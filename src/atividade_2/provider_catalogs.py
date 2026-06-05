"""Provider catalog clients used to validate AV3 assignment model ids."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .provider_validation_contracts import ProviderModelCatalogEntry

OPENROUTER_PROVIDER = "openrouter"
FEATHERLESS_PROVIDER = "featherless"
DEFAULT_OPENROUTER_CATALOG_MAX_RESPONSE_BYTES = 5_000_000
DEFAULT_FEATHERLESS_CATALOG_MAX_RESPONSE_BYTES = 50_000_000


class ProviderCatalogError(RuntimeError):
    """Raised when a provider catalog cannot be fetched or parsed."""


class ProviderCatalogClient(Protocol):
    """Read-only boundary for provider model catalogs."""

    def list_models(self) -> tuple[ProviderModelCatalogEntry, ...]:
        """Return the provider catalog entries used for exact id validation."""


class CatalogHttpTransport(Protocol):
    """Minimal GET transport seam for offline tests."""

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: int,
    ) -> tuple[int, Any]:
        """GET JSON and return ``(status_code, parsed_json)``."""


@dataclass(frozen=True)
class UrllibCatalogHttpTransport:
    """stdlib urllib transport for provider catalog fetches."""

    max_response_bytes: int = 1_000_000

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: int,
    ) -> tuple[int, Any]:
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status_code = response.getcode()
                raw_body = response.read(self.max_response_bytes + 1)
        except urllib.error.HTTPError as error:
            raw_body = error.read(self.max_response_bytes + 1)
            message = _safe_response_message(raw_body)
            raise ProviderCatalogError(
                f"Provider catalog returned HTTP {error.code}: {message}"
            ) from error
        except urllib.error.URLError as error:
            raise ProviderCatalogError(f"Provider catalog request failed: {error.reason}") from error
        except TimeoutError as error:
            raise ProviderCatalogError("Provider catalog request timed out.") from error

        if len(raw_body) > self.max_response_bytes:
            raise ProviderCatalogError("Provider catalog response exceeded the maximum allowed size.")
        return status_code, _parse_json_body(raw_body)


@dataclass(frozen=True)
class OpenRouterCatalogClient:
    """OpenRouter model catalog client."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str | None = None
    timeout_seconds: int = 30
    max_response_bytes: int = DEFAULT_OPENROUTER_CATALOG_MAX_RESPONSE_BYTES
    transport: CatalogHttpTransport | None = None

    def list_models(self) -> tuple[ProviderModelCatalogEntry, ...]:
        transport = self.transport or UrllibCatalogHttpTransport(max_response_bytes=self.max_response_bytes)
        status_code, payload = transport.get(
            _resolve_catalog_url(self.base_url, "/models"),
            headers=_build_headers(self.api_key),
            timeout=self.timeout_seconds,
        )
        if status_code < 200 or status_code >= 300:
            raise ProviderCatalogError(f"OpenRouter catalog returned HTTP {status_code}.")
        if not isinstance(payload, dict):
            raise ProviderCatalogError("OpenRouter catalog JSON response must be an object.")
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ProviderCatalogError("OpenRouter catalog JSON response must contain a data list.")
        return tuple(
            entry
            for entry in (_parse_catalog_entry(OPENROUTER_PROVIDER, row) for row in rows)
            if entry is not None
        )


@dataclass(frozen=True)
class FeatherlessCatalogClient:
    """Featherless model catalog client."""

    base_url: str = "https://api.featherless.ai/v1"
    api_key: str | None = None
    timeout_seconds: int = 30
    max_response_bytes: int = DEFAULT_FEATHERLESS_CATALOG_MAX_RESPONSE_BYTES
    transport: CatalogHttpTransport | None = None

    def list_models(self) -> tuple[ProviderModelCatalogEntry, ...]:
        transport = self.transport or UrllibCatalogHttpTransport(max_response_bytes=self.max_response_bytes)
        status_code, payload = transport.get(
            _resolve_catalog_url(self.base_url, "/models"),
            headers=_build_headers(self.api_key),
            timeout=self.timeout_seconds,
        )
        if status_code < 200 or status_code >= 300:
            raise ProviderCatalogError(f"Featherless catalog returned HTTP {status_code}.")
        rows = _extract_featherless_rows(payload)
        return tuple(
            entry
            for entry in (_parse_catalog_entry(FEATHERLESS_PROVIDER, row) for row in rows)
            if entry is not None
        )


@dataclass(frozen=True)
class FakeProviderCatalogClient:
    """Deterministic fake catalog client for offline tests."""

    entries: tuple[ProviderModelCatalogEntry, ...] = ()
    error: Exception | None = None

    def list_models(self) -> tuple[ProviderModelCatalogEntry, ...]:
        if self.error is not None:
            raise self.error
        return tuple(self.entries)


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "atividade-2-provider-validation/0.1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _resolve_catalog_url(base_url: str, path: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith(path):
        return stripped
    return f"{stripped}{path}"


def _extract_featherless_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("data")
        if isinstance(rows, list):
            return rows
    raise ProviderCatalogError("Featherless catalog JSON response must be a list or an object with a data list.")


def _parse_catalog_entry(provider: str, row: Any) -> ProviderModelCatalogEntry | None:
    if not isinstance(row, dict):
        return None
    model_id = _optional_str(row.get("id"))
    if not model_id:
        return None
    return ProviderModelCatalogEntry(
        provider=provider,
        model_id=model_id,
        name=_optional_str(row.get("name")),
        canonical_slug=_optional_str(row.get("canonical_slug")),
        hugging_face_id=_optional_str(row.get("hugging_face_id")),
        context_length=_optional_int(row.get("context_length")),
        raw=dict(row),
    )


def _parse_json_body(raw_body: bytes) -> Any:
    try:
        return json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise ProviderCatalogError("Provider catalog returned invalid JSON.") from error


def _safe_response_message(raw_body: bytes) -> str:
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "<non-json response>"
    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if isinstance(value, str):
                return value[:300]
            if isinstance(value, dict) and isinstance(value.get("message"), str):
                return value["message"][:300]
    return "<response omitted>"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
