from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable

from .catalog import ModelCatalog, ModelCatalogEntry, ModelCatalogFilter, ModelProfile


ESCALATION_ORDER = (
    ModelProfile.LOCAL_PRIVATE,
    ModelProfile.CHEAP_HIGH_VOLUME,
    ModelProfile.BALANCED_GENERAL,
    ModelProfile.PREMIUM_REASONING,
)


@dataclass(frozen=True)
class ModelRoutingContext:
    tenant_id: str
    workspace: str
    purpose: str
    role: str = "user"
    contains_private_data: bool = False
    requires_tools: bool = False
    requires_structured_output: bool = False
    min_context_window: int | None = None
    requested_profile: ModelProfile | None = None


@dataclass(frozen=True)
class ModelPolicy:
    allow_hosted: bool = True
    allow_local: bool = True
    allowed_providers: set[str] = field(default_factory=set)
    denied_providers: set[str] = field(default_factory=set)
    max_input_price: Decimal | None = None
    max_output_price: Decimal | None = None
    require_private_for_private_data: bool = True
    pinned_models: dict[ModelProfile, str] = field(default_factory=dict)
    visible_providers: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ModelRouteDecision:
    model: ModelCatalogEntry | None
    profile: ModelProfile
    allowed: bool
    reason: str
    fallback_chain: list[ModelProfile]


class PolicyModelRouter:
    """Selects runtime-visible models using explicit governance policy."""

    def __init__(self, catalog: ModelCatalog, policy: ModelPolicy | None = None):
        self.catalog = catalog
        self.policy = policy or ModelPolicy()

    def visible_models(self) -> list[ModelCatalogEntry]:
        entries = self.catalog.entries
        if self.policy.visible_providers:
            entries = [
                entry
                for entry in entries
                if entry.provider in self.policy.visible_providers
            ]
        return entries

    def permitted_models(self, context: ModelRoutingContext) -> list[ModelCatalogEntry]:
        require_private = (
            context.contains_private_data
            and self.policy.require_private_for_private_data
        )
        filtered = ModelCatalog(entries=self.visible_models()).filter(
            ModelCatalogFilter(
                include_hosted=self.policy.allow_hosted and not require_private,
                include_local=self.policy.allow_local,
                require_private=require_private,
                min_context_window=context.min_context_window,
                max_input_price=self.policy.max_input_price,
                max_output_price=self.policy.max_output_price,
                requires_tools=True if context.requires_tools else None,
                requires_structured_output=True
                if context.requires_structured_output
                else None,
            )
        )
        if self.policy.allowed_providers:
            filtered = [
                entry
                for entry in filtered
                if entry.provider in self.policy.allowed_providers
            ]
        if self.policy.denied_providers:
            filtered = [
                entry
                for entry in filtered
                if entry.provider not in self.policy.denied_providers
            ]
        return filtered

    def route(self, context: ModelRoutingContext) -> ModelRouteDecision:
        chain = self._profile_chain(context.requested_profile)
        permitted = self.permitted_models(context)
        if not permitted:
            return ModelRouteDecision(
                model=None,
                profile=chain[0],
                allowed=False,
                reason="No permitted model satisfies policy and runtime catalog.",
                fallback_chain=list(chain),
            )

        for profile in chain:
            pinned = self.policy.pinned_models.get(profile)
            if pinned:
                match = next((entry for entry in permitted if entry.model_id == pinned), None)
                if match:
                    return ModelRouteDecision(
                        model=match,
                        profile=profile,
                        allowed=True,
                        reason="Pinned production model selected.",
                        fallback_chain=list(chain),
                    )
            candidates = self._candidates_for_profile(permitted, profile)
            if candidates:
                return ModelRouteDecision(
                    model=candidates[0],
                    profile=profile,
                    allowed=True,
                    reason="Selected first policy-compliant runtime model for profile.",
                    fallback_chain=list(chain),
                )

        return ModelRouteDecision(
            model=None,
            profile=chain[-1],
            allowed=False,
            reason="Catalog has permitted models, but none matched profile constraints.",
            fallback_chain=list(chain),
        )

    def _profile_chain(self, requested: ModelProfile | None) -> tuple[ModelProfile, ...]:
        if requested is None:
            return ESCALATION_ORDER
        start = ESCALATION_ORDER.index(requested)
        return ESCALATION_ORDER[start:]

    def _candidates_for_profile(
        self, entries: Iterable[ModelCatalogEntry], profile: ModelProfile
    ) -> list[ModelCatalogEntry]:
        candidates = list(entries)
        if profile == ModelProfile.LOCAL_PRIVATE:
            candidates = [entry for entry in candidates if entry.privacy_flags.get("local")]
        elif profile == ModelProfile.CHEAP_HIGH_VOLUME:
            candidates = [
                entry
                for entry in candidates
                if entry.input_price is not None or entry.privacy_flags.get("local")
            ]
            candidates.sort(key=lambda e: (e.input_price is None, e.input_price or Decimal("0")))
        elif profile == ModelProfile.BALANCED_GENERAL:
            candidates.sort(key=lambda e: (e.context_window or 0), reverse=True)
        elif profile == ModelProfile.PREMIUM_REASONING:
            candidates = [
                entry
                for entry in candidates
                if "reasoning" in entry.capabilities
                or "include_reasoning" in entry.capabilities
                or "reasoning" in entry.family
            ] or candidates
            candidates.sort(key=lambda e: (e.context_window or 0), reverse=True)
        return candidates
