"""Typed contracts for AV3 provider-model validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ProviderModelValidationStatus = Literal[
    "found",
    "missing",
    "skipped_excluded",
    "skipped_inactive",
    "skipped_missing_model_id",
    "skipped_pending_confirmation",
    "skipped_unresolved",
    "skipped_unsupported_provider",
    "provider_error",
]

PROVIDER_MODEL_VALIDATION_STATUS_VALUES: tuple[ProviderModelValidationStatus, ...] = (
    "found",
    "missing",
    "skipped_excluded",
    "skipped_inactive",
    "skipped_missing_model_id",
    "skipped_pending_confirmation",
    "skipped_unresolved",
    "skipped_unsupported_provider",
    "provider_error",
)


@dataclass(frozen=True)
class ProviderModelCatalogEntry:
    """One provider-catalog model entry used for AV3 assignment validation."""

    provider: str
    model_id: str
    name: str | None = None
    canonical_slug: str | None = None
    hugging_face_id: str | None = None
    context_length: int | None = None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must not be empty.")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty.")
        if self.context_length is not None and self.context_length < 1:
            raise ValueError("context_length must be >= 1 when provided.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "name": self.name,
            "canonical_slug": self.canonical_slug,
            "hugging_face_id": self.hugging_face_id,
            "context_length": self.context_length,
            "raw": dict(self.raw) if self.raw is not None else None,
        }


@dataclass(frozen=True)
class ProviderModelValidationItem:
    """Validation outcome for one AV3 candidate-model assignment."""

    assignment_id: int | None
    id_modelo_av2: int
    owner: str
    av2_model_name: str | None
    original_provider_model_id: str
    av3_provider: str
    av3_provider_model_id: str | None
    status: ProviderModelValidationStatus
    message: str
    matched_model: ProviderModelCatalogEntry | None = None

    def __post_init__(self) -> None:
        if self.id_modelo_av2 < 1:
            raise ValueError("id_modelo_av2 must be >= 1.")
        if not self.owner.strip():
            raise ValueError("owner must not be empty.")
        if not self.original_provider_model_id.strip():
            raise ValueError("original_provider_model_id must not be empty.")
        if self.status not in PROVIDER_MODEL_VALIDATION_STATUS_VALUES:
            raise ValueError(f"Unsupported provider model validation status: {self.status!r}")
        if not self.message.strip():
            raise ValueError("message must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "id_modelo_av2": self.id_modelo_av2,
            "owner": self.owner,
            "av2_model_name": self.av2_model_name,
            "original_provider_model_id": self.original_provider_model_id,
            "av3_provider": self.av3_provider,
            "av3_provider_model_id": self.av3_provider_model_id,
            "status": self.status,
            "message": self.message,
            "matched_model": self.matched_model.to_dict() if self.matched_model is not None else None,
        }


@dataclass(frozen=True)
class ProviderModelValidationReport:
    """Aggregated provider-model validation report for AV3 assignments."""

    total_assignments: int
    checked: int
    found: int
    missing: int
    skipped: int
    provider_errors: int
    items: tuple[ProviderModelValidationItem, ...]

    def __post_init__(self) -> None:
        if min(
            self.total_assignments,
            self.checked,
            self.found,
            self.missing,
            self.skipped,
            self.provider_errors,
        ) < 0:
            raise ValueError("report counters must be >= 0.")
        object.__setattr__(self, "items", tuple(self.items))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_assignments": self.total_assignments,
            "checked": self.checked,
            "found": self.found,
            "missing": self.missing,
            "skipped": self.skipped,
            "provider_errors": self.provider_errors,
            "items": [item.to_dict() for item in self.items],
        }
