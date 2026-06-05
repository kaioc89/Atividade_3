from __future__ import annotations

import io
import urllib.error
import urllib.request
from dataclasses import replace

import pytest

from atividade_2.contracts import CandidateModelAssignment
from atividade_2.provider_catalogs import (
    DEFAULT_FEATHERLESS_CATALOG_MAX_RESPONSE_BYTES,
    DEFAULT_OPENROUTER_CATALOG_MAX_RESPONSE_BYTES,
    FakeProviderCatalogClient,
    FeatherlessCatalogClient,
    OpenRouterCatalogClient,
    ProviderCatalogError,
)
from atividade_2.provider_model_validation import ProviderModelValidationService
from atividade_2.provider_validation_contracts import ProviderModelCatalogEntry
from atividade_2.repositories import _default_candidate_model_assignments


class FakeAssignmentRepository:
    def __init__(self, assignments: tuple[CandidateModelAssignment, ...]) -> None:
        self._assignments = assignments

    def list_candidate_model_assignments(self) -> tuple[CandidateModelAssignment, ...]:
        return self._assignments


class FakeTransport:
    def __init__(self, payload, *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.calls: list[tuple[str, dict[str, str], int]] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: int):
        self.calls.append((url, headers, timeout))
        return self.status_code, self.payload


def _openrouter_entries() -> tuple[ProviderModelCatalogEntry, ...]:
    return (
        ProviderModelCatalogEntry(provider="openrouter", model_id="openai/gpt-5", name="GPT-5"),
        ProviderModelCatalogEntry(
            provider="openrouter",
            model_id="google/gemini-3.5-flash",
            name="Gemini 3.5 Flash",
        ),
        ProviderModelCatalogEntry(
            provider="openrouter",
            model_id="x-ai/grok-4.3",
            name="Grok 4.3",
        ),
    )


def _featherless_entries() -> tuple[ProviderModelCatalogEntry, ...]:
    return tuple(
        ProviderModelCatalogEntry(provider="featherless", model_id=model_id)
        for model_id in (
            "google/gemma-2-2b-it",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "google/gemma-3-12b-it",
            "meta-llama/Llama-3.1-8B-Instruct",
        )
    )


def _default_service(
    assignments: tuple[CandidateModelAssignment, ...] | None = None,
    *,
    openrouter_client: FakeProviderCatalogClient | None = None,
    featherless_client: FakeProviderCatalogClient | None = None,
) -> ProviderModelValidationService:
    return ProviderModelValidationService(
        assignment_repository=FakeAssignmentRepository(assignments or _default_candidate_model_assignments()),
        catalog_clients={
            "openrouter": openrouter_client or FakeProviderCatalogClient(entries=_openrouter_entries()),
            "featherless": featherless_client or FakeProviderCatalogClient(entries=_featherless_entries()),
        },
    )


def test_service_validates_openrouter_and_featherless_assignments_against_fake_catalogs() -> None:
    report = _default_service().validate()

    jose_gpt5 = next(item for item in report.items if item.id_modelo_av2 == 14)

    assert report.missing == 0
    assert jose_gpt5.status == "found"
    assert jose_gpt5.matched_model is not None
    assert jose_gpt5.matched_model.model_id == "openai/gpt-5"
    assert all(
        item.status == "found"
        for item in report.items
        if item.av3_provider == "featherless"
    )


def test_found_models_are_reported_as_found() -> None:
    report = _default_service().validate()

    statuses = {item.id_modelo_av2: item.status for item in report.items}

    assert statuses[14] == "found"
    assert statuses[1] == "found"


def test_missing_provider_model_ids_are_reported_as_missing() -> None:
    assignments = tuple(
        replace(assignment, av3_provider_model_id="openai/gpt-5-missing")
        if assignment.id_modelo_av2 == 14
        else assignment
        for assignment in _default_candidate_model_assignments()
    )

    report = _default_service(assignments).validate()

    jose_gpt5 = next(item for item in report.items if item.id_modelo_av2 == 14)

    assert report.missing == 1
    assert jose_gpt5.status == "missing"
    assert jose_gpt5.matched_model is None


def test_jose_grok_is_found_when_fake_openrouter_contains_team_approved_substitution() -> None:
    report = _default_service().validate(include_pending_confirmation=True)

    jose_grok = next(item for item in report.items if item.id_modelo_av2 == 15)

    assert jose_grok.av3_provider == "openrouter"
    assert jose_grok.original_provider_model_id == "Grok 3"
    assert jose_grok.av3_provider_model_id == "x-ai/grok-4.3"
    assert jose_grok.status == "found"
    assert jose_grok.matched_model is not None
    assert jose_grok.matched_model.model_id == "x-ai/grok-4.3"


def test_excluded_assignments_are_skipped_when_requested() -> None:
    report = _default_service().validate(include_excluded=True)

    excluded = next(item for item in report.items if item.id_modelo_av2 == 11)

    assert excluded.status == "skipped_excluded"


def test_assignments_with_empty_provider_model_id_are_skipped() -> None:
    assignments = tuple(
        replace(assignment, av3_provider_model_id="  ")
        if assignment.id_modelo_av2 == 14
        else assignment
        for assignment in _default_candidate_model_assignments()
    )

    report = _default_service(assignments).validate()

    jose_gpt5 = next(item for item in report.items if item.id_modelo_av2 == 14)

    assert jose_gpt5.status == "skipped_missing_model_id"


def test_unsupported_providers_are_skipped_with_explicit_reason() -> None:
    assignments = tuple(
        replace(assignment, av3_provider="ollama", av3_provider_model_id="llama3.1:8b")
        if assignment.id_modelo_av2 == 14
        else assignment
        for assignment in _default_candidate_model_assignments()
    )

    report = _default_service(assignments).validate()

    jose_gpt5 = next(item for item in report.items if item.id_modelo_av2 == 14)

    assert jose_gpt5.status == "skipped_unsupported_provider"
    assert "not supported" in jose_gpt5.message


def test_provider_client_error_produces_provider_error_without_crashing() -> None:
    report = _default_service(
        openrouter_client=FakeProviderCatalogClient(error=ProviderCatalogError("OpenRouter unavailable"))
    ).validate()

    jose_gpt5 = next(item for item in report.items if item.id_modelo_av2 == 14)
    diego_gemma = next(item for item in report.items if item.id_modelo_av2 == 6)
    jose_grok = next(item for item in report.items if item.id_modelo_av2 == 15)

    assert report.provider_errors == 2
    assert jose_gpt5.status == "provider_error"
    assert jose_grok.status == "provider_error"
    assert diego_gemma.status == "found"


def test_jose_gpt5_is_found_when_fake_openrouter_contains_openai_gpt5() -> None:
    report = _default_service().validate()

    jose_gpt5 = next(item for item in report.items if item.id_modelo_av2 == 14)

    assert jose_gpt5.status == "found"
    assert jose_gpt5.matched_model is not None
    assert jose_gpt5.matched_model.name == "GPT-5"


def test_jose_gemini_is_excluded_by_default_when_pending_confirmation_is_disabled() -> None:
    report = _default_service().validate()

    assert all(item.id_modelo_av2 != 13 for item in report.items)


def test_jose_gemini_is_checked_when_pending_confirmation_is_enabled() -> None:
    report = _default_service().validate(include_pending_confirmation=True)

    jose_gemini = next(item for item in report.items if item.id_modelo_av2 == 13)

    assert jose_gemini.status == "found"
    assert jose_gemini.matched_model is not None
    assert jose_gemini.matched_model.model_id == "google/gemini-3.5-flash"


def test_openrouter_provider_filter_includes_jose_grok_pending_assignment() -> None:
    report = _default_service().validate(
        providers=("openrouter",),
        include_pending_confirmation=True,
    )

    statuses = {item.id_modelo_av2: item.status for item in report.items}

    assert report.total_assignments == 3
    assert set(statuses) == {13, 14, 15}
    assert statuses[14] == "found"
    assert statuses[13] == "found"
    assert statuses[15] == "found"


def test_all_checked_featherless_ids_can_be_found_in_fake_catalog() -> None:
    report = _default_service().validate()

    checked_featherless_ids = {
        item.av3_provider_model_id
        for item in report.items
        if item.av3_provider == "featherless"
    }

    assert checked_featherless_ids == {
        "google/gemma-2-2b-it",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "google/gemma-3-12b-it",
        "meta-llama/Llama-3.1-8B-Instruct",
    }


def test_openrouter_catalog_client_parses_data_array() -> None:
    transport = FakeTransport(
        {
            "data": [
                {
                    "id": "openai/gpt-5",
                    "name": "GPT-5",
                    "canonical_slug": "openai/gpt-5",
                    "hugging_face_id": "openai/gpt-5",
                    "context_length": 128000,
                }
            ]
        }
    )

    entries = OpenRouterCatalogClient(transport=transport).list_models()

    assert entries == (
        ProviderModelCatalogEntry(
            provider="openrouter",
            model_id="openai/gpt-5",
            name="GPT-5",
            canonical_slug="openai/gpt-5",
            hugging_face_id="openai/gpt-5",
            context_length=128000,
            raw={
                "id": "openai/gpt-5",
                "name": "GPT-5",
                "canonical_slug": "openai/gpt-5",
                "hugging_face_id": "openai/gpt-5",
                "context_length": 128000,
            },
        ),
    )
    assert transport.calls[0][0] == "https://openrouter.ai/api/v1/models"


def test_featherless_catalog_client_accepts_top_level_list_shape() -> None:
    transport = FakeTransport(
        [
            {"id": "meta-llama/Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B"},
            {"id": "google/gemma-3-12b-it"},
        ]
    )

    entries = FeatherlessCatalogClient(transport=transport).list_models()

    assert [entry.model_id for entry in entries] == [
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-3-12b-it",
    ]
    assert transport.calls[0][0] == "https://api.featherless.ai/v1/models"


def test_openrouter_catalog_client_uses_provider_default_response_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_limits: list[int] = []

    class FakeHttpResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def getcode(self) -> int:
            return 200

        def read(self, limit: int) -> bytes:
            read_limits.append(limit)
            return b'{"data":[]}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=0: FakeHttpResponse())

    OpenRouterCatalogClient().list_models()

    assert read_limits == [DEFAULT_OPENROUTER_CATALOG_MAX_RESPONSE_BYTES + 1]


def test_featherless_catalog_client_uses_provider_default_response_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_limits: list[int] = []

    class FakeHttpResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def getcode(self) -> int:
            return 200

        def read(self, limit: int) -> bytes:
            read_limits.append(limit)
            return b"[]"

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=0: FakeHttpResponse())

    FeatherlessCatalogClient().list_models()

    assert read_limits == [DEFAULT_FEATHERLESS_CATALOG_MAX_RESPONSE_BYTES + 1]


def test_openrouter_catalog_client_raises_provider_catalog_error_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHttpResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def getcode(self) -> int:
            return 200

        def read(self, _limit: int) -> bytes:
            return b"{invalid-json"

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=0: FakeHttpResponse())

    with pytest.raises(ProviderCatalogError, match="invalid JSON"):
        OpenRouterCatalogClient().list_models()


def test_featherless_catalog_client_raises_provider_catalog_error_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http_error = urllib.error.HTTPError(
        "https://api.featherless.ai/v1/models",
        503,
        "Unavailable",
        hdrs=None,
        fp=io.BytesIO(b'{"error":"down"}'),
    )

    def raise_http_error(request, timeout=0):
        raise http_error

    monkeypatch.setattr(urllib.request, "urlopen", raise_http_error)

    with pytest.raises(ProviderCatalogError, match="HTTP 503"):
        FeatherlessCatalogClient().list_models()


def test_openrouter_catalog_client_raises_provider_catalog_error_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(request, timeout=0):
        raise TimeoutError()

    monkeypatch.setattr(urllib.request, "urlopen", raise_timeout)

    with pytest.raises(ProviderCatalogError, match="timed out"):
        OpenRouterCatalogClient().list_models()
