"""Read-only validation for AV3 provider model assignments."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Protocol

from .contracts import CandidateModelAssignment
from .provider_catalogs import (
    FEATHERLESS_PROVIDER,
    OPENROUTER_PROVIDER,
    ProviderCatalogClient,
    ProviderCatalogError,
)
from .provider_validation_contracts import (
    ProviderModelCatalogEntry,
    ProviderModelValidationItem,
    ProviderModelValidationReport,
    ProviderModelValidationStatus,
)

SUPPORTED_PROVIDER_VALIDATION_PROVIDERS: tuple[str, ...] = (
    OPENROUTER_PROVIDER,
    FEATHERLESS_PROVIDER,
)
PENDING_CONFIRMATION_STATUSES = {
    "needs_owner_confirmation",
    "needs_owner_confirmation_gemini_subtype",
    "pending_team_confirmation",
}


class CandidateAssignmentRepositoryProtocol(Protocol):
    """Repository reads required by the provider validation service."""

    def list_candidate_model_assignments(self) -> tuple[CandidateModelAssignment, ...]:
        """Return the full AV3 assignment registry without mutating it."""


class ProviderModelValidationService:
    """Validate configured AV3 provider model ids against provider catalogs."""

    def __init__(
        self,
        *,
        assignment_repository: CandidateAssignmentRepositoryProtocol,
        catalog_clients: Mapping[str, ProviderCatalogClient],
    ) -> None:
        self._assignment_repository = assignment_repository
        self._catalog_clients = {
            provider.strip().casefold(): client
            for provider, client in catalog_clients.items()
            if provider.strip()
        }

    def validate(
        self,
        *,
        providers: Iterable[str] | None = None,
        include_pending_confirmation: bool = False,
        include_excluded: bool = False,
    ) -> ProviderModelValidationReport:
        provider_filter = _normalize_provider_filter(providers)
        assignments = tuple(
            assignment
            for assignment in self._assignment_repository.list_candidate_model_assignments()
            if _include_assignment(
                assignment,
                provider_filter=provider_filter,
                include_pending_confirmation=include_pending_confirmation,
                include_excluded=include_excluded,
            )
        )
        catalogs, provider_errors = self._load_catalogs(assignments)
        items = tuple(
            self._validate_assignment(
                assignment,
                catalogs=catalogs,
                provider_errors=provider_errors,
                include_pending_confirmation=include_pending_confirmation,
            )
            for assignment in assignments
        )
        found = sum(1 for item in items if item.status == "found")
        missing = sum(1 for item in items if item.status == "missing")
        skipped = sum(1 for item in items if item.status.startswith("skipped_"))
        provider_errors_count = sum(1 for item in items if item.status == "provider_error")
        return ProviderModelValidationReport(
            total_assignments=len(items),
            checked=found + missing,
            found=found,
            missing=missing,
            skipped=skipped,
            provider_errors=provider_errors_count,
            items=items,
        )

    def _load_catalogs(
        self,
        assignments: tuple[CandidateModelAssignment, ...],
    ) -> tuple[dict[str, dict[str, ProviderModelCatalogEntry]], dict[str, str]]:
        providers_to_fetch = {
            assignment.av3_provider.casefold()
            for assignment in assignments
            if _is_provider_checkable(assignment)
        }
        catalogs: dict[str, dict[str, ProviderModelCatalogEntry]] = {}
        provider_errors: dict[str, str] = {}
        for provider in sorted(providers_to_fetch):
            client = self._catalog_clients.get(provider)
            if client is None:
                provider_errors[provider] = f"No catalog client configured for provider {provider}."
                continue
            try:
                entries = client.list_models()
            except ProviderCatalogError as error:
                provider_errors[provider] = str(error)
                continue
            except Exception as error:  # pragma: no cover - defensive parity with CLI/runtime boundaries
                provider_errors[provider] = str(error) or error.__class__.__name__
                continue
            catalogs[provider] = {entry.model_id: entry for entry in entries}
        return catalogs, provider_errors

    def _validate_assignment(
        self,
        assignment: CandidateModelAssignment,
        *,
        catalogs: Mapping[str, Mapping[str, ProviderModelCatalogEntry]],
        provider_errors: Mapping[str, str],
        include_pending_confirmation: bool,
    ) -> ProviderModelValidationItem:
        provider = assignment.av3_provider.casefold()
        model_id = _normalized_model_id(assignment.av3_provider_model_id)
        if not assignment.active:
            return _build_item(assignment, status="skipped_inactive", message="Assignment is inactive.")
        if assignment.validation_status in PENDING_CONFIRMATION_STATUSES and not include_pending_confirmation:
            return _build_item(
                assignment,
                status="skipped_pending_confirmation",
                message="Assignment still requires pending confirmation before validation.",
            )
        if provider == "excluded" or assignment.validation_status == "excluded_from_av3_run":
            return _build_item(assignment, status="skipped_excluded", message="Assignment is excluded from AV3 validation.")
        if provider == "unresolved" or assignment.validation_status == "needs_provider_resolution":
            return _build_item(
                assignment,
                status="skipped_unresolved",
                message="Assignment remains unresolved and must not be replaced automatically.",
            )
        if not model_id:
            return _build_item(
                assignment,
                status="skipped_missing_model_id",
                message="Assignment does not define an AV3 provider model id.",
            )
        if provider not in SUPPORTED_PROVIDER_VALIDATION_PROVIDERS:
            return _build_item(
                assignment,
                status="skipped_unsupported_provider",
                message=f"Provider {assignment.av3_provider} is not supported by this validation helper.",
            )
        if provider in provider_errors:
            return _build_item(
                assignment,
                status="provider_error",
                message=f"Provider catalog error for {assignment.av3_provider}: {provider_errors[provider]}",
            )
        matched_model = catalogs.get(provider, {}).get(model_id)
        if matched_model is None:
            return _build_item(
                assignment,
                status="missing",
                message=f"Provider model id {model_id} was not found in the {assignment.av3_provider} catalog.",
            )
        return _build_item(
            assignment,
            status="found",
            message=f"Provider model id {model_id} was found in the {assignment.av3_provider} catalog.",
            matched_model=matched_model,
        )


def provider_model_validation_exit_code(report: ProviderModelValidationReport) -> int:
    """Map one validation report to the CLI exit-code contract."""
    if report.provider_errors:
        return 2
    if report.missing:
        return 1
    return 0


def format_provider_model_validation_report(
    report: ProviderModelValidationReport,
    *,
    as_json: bool = False,
) -> str:
    """Render the validation report for the CLI."""
    if as_json:
        return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    lines = [
        "Provider model validation report",
        f"Assignments: {report.total_assignments}",
        f"Checked: {report.checked}",
        f"Found: {report.found}",
        f"Missing: {report.missing}",
        f"Skipped: {report.skipped}",
        f"Provider errors: {report.provider_errors}",
    ]
    for item in report.items:
        lines.append(
            " | ".join(
                (
                    str(item.assignment_id) if item.assignment_id is not None else "-",
                    str(item.id_modelo_av2),
                    item.owner,
                    item.av3_provider,
                    item.av3_provider_model_id or "-",
                    item.status,
                    item.message,
                )
            )
        )
    return "\n".join(lines)


def _normalize_provider_filter(providers: Iterable[str] | None) -> set[str] | None:
    if providers is None:
        return None
    normalized = {provider.strip().casefold() for provider in providers if provider.strip()}
    return normalized or None


def _include_assignment(
    assignment: CandidateModelAssignment,
    *,
    provider_filter: set[str] | None,
    include_pending_confirmation: bool,
    include_excluded: bool,
) -> bool:
    provider = assignment.av3_provider.casefold()
    if provider_filter is not None and provider not in provider_filter:
        return False
    if not include_pending_confirmation and assignment.validation_status in PENDING_CONFIRMATION_STATUSES:
        return False
    if not include_excluded and (provider == "excluded" or assignment.validation_status == "excluded_from_av3_run"):
        return False
    return True


def _is_provider_checkable(assignment: CandidateModelAssignment) -> bool:
    provider = assignment.av3_provider.casefold()
    model_id = _normalized_model_id(assignment.av3_provider_model_id)
    return provider in SUPPORTED_PROVIDER_VALIDATION_PROVIDERS and bool(model_id)


def _normalized_model_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _build_item(
    assignment: CandidateModelAssignment,
    *,
    status: ProviderModelValidationStatus,
    message: str,
    matched_model: ProviderModelCatalogEntry | None = None,
) -> ProviderModelValidationItem:
    return ProviderModelValidationItem(
        assignment_id=assignment.assignment_id,
        id_modelo_av2=assignment.id_modelo_av2,
        owner=assignment.owner,
        av2_model_name=assignment.av2_model_name,
        original_provider_model_id=assignment.original_provider_model_id,
        av3_provider=assignment.av3_provider,
        av3_provider_model_id=assignment.av3_provider_model_id,
        status=status,
        message=message,
        matched_model=matched_model,
    )
