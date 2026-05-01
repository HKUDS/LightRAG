from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Iterable, Mapping


class ModelProfile(StrEnum):
    """Governed routing profiles used by enterprise workflows."""

    PREMIUM_REASONING = "premium_reasoning"
    BALANCED_GENERAL = "balanced_general"
    CHEAP_HIGH_VOLUME = "cheap_high_volume"
    LOCAL_PRIVATE = "local_private"


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def infer_provider(model_id: str, name: str = "") -> str:
    """Infer provider/lab from a dynamic model id without relying on a fixed catalog."""

    if "/" in model_id:
        return model_id.split("/", 1)[0].strip().lower()
    if ":" in name:
        return name.split(":", 1)[0].strip().lower()
    return "unknown"


def infer_family(model_id: str, name: str = "") -> str:
    """Infer a coarse model family for filtering and UI grouping.

    This is a heuristic over runtime catalog values, not a fixed model list. New
    labs remain visible even when the family resolves to the provider prefix.
    """

    text = f"{model_id} {name}".lower()
    known_family_markers = (
        "gemini",
        "claude",
        "gpt",
        "chatgpt",
        "grok",
        "perplexity",
        "minimax",
        "llama",
        "mistral",
        "qwen",
        "deepseek",
        "command",
    )
    for marker in known_family_markers:
        if marker in text:
            return marker
    provider = infer_provider(model_id, name)
    slug = model_id.split("/", 1)[-1]
    return slug.split("-", 1)[0].lower() if slug else provider


@dataclass(frozen=True)
class ModelCatalogEntry:
    """Normalized model metadata from a hosted or local catalog."""

    model_id: str
    slug: str
    provider: str
    family: str
    context_window: int | None
    modalities: dict[str, list[str]]
    capabilities: set[str] = field(default_factory=set)
    tool_calling: bool = False
    structured_output: bool = False
    input_price: Decimal | None = None
    output_price: Decimal | None = None
    request_price: Decimal | None = None
    image_price: Decimal | None = None
    privacy_flags: dict[str, Any] = field(default_factory=dict)
    synced_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_openrouter_model(
        cls, payload: Mapping[str, Any], synced_at: datetime | None = None
    ) -> "ModelCatalogEntry":
        model_id = str(payload.get("id") or "")
        slug = str(payload.get("canonical_slug") or model_id)
        name = str(payload.get("name") or model_id)
        architecture = payload.get("architecture") or {}
        pricing = payload.get("pricing") or {}
        top_provider = payload.get("top_provider") or {}
        supported_parameters = set(payload.get("supported_parameters") or [])

        capabilities = set(supported_parameters)
        input_modalities = list(architecture.get("input_modalities") or [])
        output_modalities = list(architecture.get("output_modalities") or [])
        for modality in input_modalities:
            capabilities.add(f"input:{modality}")
        for modality in output_modalities:
            capabilities.add(f"output:{modality}")

        return cls(
            model_id=model_id,
            slug=slug,
            provider=infer_provider(model_id, name),
            family=infer_family(model_id, name),
            context_window=payload.get("context_length")
            or top_provider.get("context_length"),
            modalities={
                "input": input_modalities,
                "output": output_modalities,
                "raw": [str(architecture.get("modality") or "")],
            },
            capabilities=capabilities,
            tool_calling="tools" in supported_parameters,
            structured_output=(
                "structured_outputs" in supported_parameters
                or "response_format" in supported_parameters
            ),
            input_price=_decimal_or_none(pricing.get("prompt")),
            output_price=_decimal_or_none(pricing.get("completion")),
            request_price=_decimal_or_none(pricing.get("request")),
            image_price=_decimal_or_none(pricing.get("image")),
            privacy_flags={
                "hosted": True,
                "local": False,
                "moderated": top_provider.get("is_moderated"),
                "per_request_limits": payload.get("per_request_limits"),
                "expiration_date": payload.get("expiration_date"),
            },
            synced_at=synced_at or datetime.now(timezone.utc),
            raw=dict(payload),
        )

    @classmethod
    def local(
        cls,
        model_id: str,
        *,
        family: str = "local",
        context_window: int | None = None,
        capabilities: Iterable[str] = (),
    ) -> "ModelCatalogEntry":
        capability_set = set(capabilities)
        return cls(
            model_id=model_id,
            slug=model_id,
            provider="local",
            family=family,
            context_window=context_window,
            modalities={"input": ["text"], "output": ["text"], "raw": ["text->text"]},
            capabilities=capability_set,
            tool_calling="tools" in capability_set,
            structured_output=(
                "structured_outputs" in capability_set
                or "response_format" in capability_set
            ),
            privacy_flags={"hosted": False, "local": True, "private": True},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "slug": self.slug,
            "provider": self.provider,
            "family": self.family,
            "context_window": self.context_window,
            "modalities": self.modalities,
            "capabilities": sorted(self.capabilities),
            "tool_calling": self.tool_calling,
            "structured_output": self.structured_output,
            "input_price": str(self.input_price)
            if self.input_price is not None
            else None,
            "output_price": str(self.output_price)
            if self.output_price is not None
            else None,
            "request_price": str(self.request_price)
            if self.request_price is not None
            else None,
            "image_price": str(self.image_price)
            if self.image_price is not None
            else None,
            "privacy_flags": self.privacy_flags,
            "synced_at": self.synced_at.isoformat(),
        }


@dataclass(frozen=True)
class ModelCatalogFilter:
    provider: str | None = None
    family: str | None = None
    max_input_price: Decimal | None = None
    max_output_price: Decimal | None = None
    min_context_window: int | None = None
    capabilities: set[str] = field(default_factory=set)
    requires_tools: bool | None = None
    requires_structured_output: bool | None = None
    require_private: bool = False
    include_hosted: bool = True
    include_local: bool = True


@dataclass
class ModelCatalog:
    """Runtime model catalog with visible-vs-allowed filtering helpers."""

    entries: list[ModelCatalogEntry] = field(default_factory=list)
    synced_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "runtime"

    def filter(
        self, criteria: ModelCatalogFilter | None = None
    ) -> list[ModelCatalogEntry]:
        criteria = criteria or ModelCatalogFilter()
        result: list[ModelCatalogEntry] = []
        for entry in self.entries:
            if criteria.provider and entry.provider != criteria.provider:
                continue
            if criteria.family and entry.family != criteria.family:
                continue
            if not criteria.include_hosted and entry.privacy_flags.get("hosted"):
                continue
            if not criteria.include_local and entry.privacy_flags.get("local"):
                continue
            if criteria.require_private and not entry.privacy_flags.get("private"):
                continue
            if (
                criteria.min_context_window is not None
                and (entry.context_window or 0) < criteria.min_context_window
            ):
                continue
            if (
                criteria.max_input_price is not None
                and entry.input_price is not None
                and entry.input_price > criteria.max_input_price
            ):
                continue
            if (
                criteria.max_output_price is not None
                and entry.output_price is not None
                and entry.output_price > criteria.max_output_price
            ):
                continue
            if criteria.requires_tools is True and not entry.tool_calling:
                continue
            if criteria.requires_tools is False and entry.tool_calling:
                continue
            if (
                criteria.requires_structured_output is True
                and not entry.structured_output
            ):
                continue
            if criteria.capabilities and not criteria.capabilities.issubset(
                entry.capabilities
            ):
                continue
            result.append(entry)
        return result

    def by_id(self, model_id: str) -> ModelCatalogEntry | None:
        return next(
            (entry for entry in self.entries if entry.model_id == model_id), None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "synced_at": self.synced_at.isoformat(),
            "count": len(self.entries),
            "data": [entry.to_dict() for entry in self.entries],
        }
