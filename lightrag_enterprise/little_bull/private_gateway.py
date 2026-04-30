from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import status

from lightrag_enterprise.model_gateway import (
    ModelCatalog,
    ModelCatalogEntry,
    ModelPolicy,
    ModelProfile,
    ModelRoutingContext,
    PolicyModelRouter,
)
from lightrag_enterprise.system.policy_keys import (
    PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY,
    stable_policy_hash,
)


LITTLE_BULL_PROFILE_MAP = {
    "rapido": ModelProfile.CHEAP_HIGH_VOLUME,
    "equilibrado": ModelProfile.BALANCED_GENERAL,
    "inteligente": ModelProfile.PREMIUM_REASONING,
    "privado": ModelProfile.LOCAL_PRIVATE,
}


@dataclass(frozen=True)
class PrivateLocalDecision:
    allowed: bool
    result: str
    reason: str
    status_code: int | None
    requested_profile: str
    routed_profile: str
    selected_model_id: str | None
    selected_provider: str | None
    model_func: Callable[..., object] | None
    contains_private_data: bool
    requires_private_runtime: bool
    hosted_private_exception: bool = False
    hosted_private_provider: str | None = None
    hosted_private_approval_id: str | None = None
    hosted_private_reason: str | None = None
    hosted_private_policy_hash: str | None = None
    hosted_private_policy_status: str | None = None
    hosted_private_policy_key: str | None = None

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "reason": self.result,
            "message": self.reason,
            "requested_profile": self.requested_profile,
            "routed_profile": self.routed_profile,
            "selected_model_id": self.selected_model_id,
            "selected_provider": self.selected_provider,
            "contains_private_data": self.contains_private_data,
            "requires_private_runtime": self.requires_private_runtime,
            "hosted_private_exception": self.hosted_private_exception,
            "hosted_private_provider": self.hosted_private_provider,
            "hosted_private_approval_id": self.hosted_private_approval_id,
            "hosted_private_reason": self.hosted_private_reason,
            "hosted_private_policy_hash": self.hosted_private_policy_hash,
            "hosted_private_policy_status": self.hosted_private_policy_status,
            "hosted_private_policy_key": self.hosted_private_policy_key,
        }


class PrivateLocalGateway:
    def __init__(self, rag: Any) -> None:
        self.rag = rag

    def evaluate(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        confidentiality: str,
        requested_profile: str,
        workspace_contains_private_data: bool,
        strict: bool = True,
        hosted_private_policy: dict[str, Any] | bool | None = None,
    ) -> PrivateLocalDecision:
        normalized_profile = requested_profile.strip().lower()
        requested_gateway_profile = LITTLE_BULL_PROFILE_MAP.get(
            normalized_profile,
            ModelProfile.BALANCED_GENERAL,
        )
        contains_private_data = (
            confidentiality in {"sensivel", "privado"} or workspace_contains_private_data
        )
        effective_confidentiality = (
            confidentiality if confidentiality in {"sensivel", "privado"} else "privado"
        )
        explicitly_private_profile = requested_gateway_profile == ModelProfile.LOCAL_PRIVATE
        hosted_private_exception = (
            not explicitly_private_profile
            and self._hosted_private_exception_allowed(
            hosted_private_policy,
            contains_private_data=contains_private_data or explicitly_private_profile,
                confidentiality=effective_confidentiality,
        )
        )
        requires_private_runtime = (
            strict
            and (contains_private_data or explicitly_private_profile)
            and not hosted_private_exception
        )

        if strict and contains_private_data and not explicitly_private_profile and not hosted_private_exception:
            return self._blocked(
                result="private_local_required",
                reason="Private/local profile is required for sensitive or private documents.",
                status_code=status.HTTP_403_FORBIDDEN,
                requested_profile=normalized_profile,
                routed_profile=requested_gateway_profile,
                contains_private_data=contains_private_data,
                requires_private_runtime=requires_private_runtime,
            )

        route_context = ModelRoutingContext(
            tenant_id=tenant_id or "unknown",
            workspace=workspace_id,
            purpose="little_bull_query",
            contains_private_data=requires_private_runtime,
            requested_profile=requested_gateway_profile,
        )
        route_policy = ModelPolicy(require_private_for_private_data=requires_private_runtime)
        route = PolicyModelRouter(self._runtime_catalog(), route_policy).route(route_context)

        if requires_private_runtime and not route.allowed:
            return self._blocked(
                result="private_local_unavailable",
                reason="Private/local model is unavailable for sensitive or private documents.",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                requested_profile=normalized_profile,
                routed_profile=route.profile,
                contains_private_data=contains_private_data,
                requires_private_runtime=requires_private_runtime,
            )

        return PrivateLocalDecision(
            allowed=True,
            result="allowed",
            reason=route.reason,
            status_code=None,
            requested_profile=normalized_profile,
            routed_profile=route.profile.value,
            selected_model_id=route.model.model_id if route.model else None,
            selected_provider=route.model.provider if route.model else None,
            model_func=self._model_func_for(route.model),
            contains_private_data=contains_private_data,
            requires_private_runtime=requires_private_runtime,
            hosted_private_exception=hosted_private_exception,
            hosted_private_provider=self._active_provider() if hosted_private_exception else None,
            hosted_private_approval_id=self._hosted_private_policy_approval_id(hosted_private_policy)
            if hosted_private_exception
            else None,
            hosted_private_reason=self._hosted_private_policy_reason(hosted_private_policy)
            if hosted_private_exception
            else None,
            hosted_private_policy_hash=stable_policy_hash(hosted_private_policy)
            if hosted_private_exception
            else None,
            hosted_private_policy_status="valid" if hosted_private_exception else None,
            hosted_private_policy_key=PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY
            if hosted_private_exception
            else None,
        )

    def _blocked(
        self,
        *,
        result: str,
        reason: str,
        status_code: int,
        requested_profile: str,
        routed_profile: ModelProfile,
        contains_private_data: bool,
        requires_private_runtime: bool,
    ) -> PrivateLocalDecision:
        return PrivateLocalDecision(
            allowed=False,
            result=result,
            reason=reason,
            status_code=status_code,
            requested_profile=requested_profile,
            routed_profile=routed_profile.value,
            selected_model_id=None,
            selected_provider=None,
            model_func=None,
            contains_private_data=contains_private_data,
            requires_private_runtime=requires_private_runtime,
        )

    def _hosted_private_exception_allowed(
        self,
        policy: dict[str, Any] | bool | None,
        *,
        contains_private_data: bool,
        confidentiality: str,
    ) -> bool:
        if not contains_private_data:
            return False
        if not isinstance(policy, dict) or not policy.get("enabled"):
            return False
        if policy.get("schema_version") != 1:
            return False
        if str(policy.get("provider", "")).strip().lower() != self._active_provider():
            return False
        if str(policy.get("binding", "")).strip().lower() != self._active_binding():
            return False
        policy_host = str(policy.get("binding_host", "")).strip().rstrip("/")
        if policy_host != self._active_host().rstrip("/"):
            return False
        allowed_models = {str(model).strip() for model in policy.get("allowed_model_ids", []) if str(model).strip()}
        if self._active_model_name() not in allowed_models:
            return False
        allowed_confidentiality = {
            str(item).strip().lower()
            for item in policy.get("allowed_confidentiality", [])
            if str(item).strip()
        }
        if confidentiality not in allowed_confidentiality:
            return False
        if not policy.get("approved_by") or not policy.get("approved_at") or not policy.get("reason"):
            return False
        expires_at = self._parse_datetime(policy.get("expires_at"))
        if expires_at is None or expires_at <= datetime.now(timezone.utc):
            return False
        return True

    @staticmethod
    def _hosted_private_policy_reason(policy: dict[str, Any] | bool | None) -> str | None:
        if isinstance(policy, dict):
            raw = policy.get("reason")
            return str(raw) if raw else None
        return None

    @staticmethod
    def _hosted_private_policy_approval_id(policy: dict[str, Any] | bool | None) -> str | None:
        if isinstance(policy, dict):
            raw = policy.get("approval_id")
            return str(raw) if raw else None
        return None

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _runtime_catalog(self) -> ModelCatalog:
        entries: list[ModelCatalogEntry] = []
        binding = self._active_binding()
        provider = self._active_provider()
        model_name = self._active_model_name()
        model_id = f"{binding}/{model_name}" if "/" not in model_name else model_name
        if binding in self._local_bindings():
            entries.append(
                ModelCatalogEntry.local(
                    model_id,
                    family=binding,
                    context_window=getattr(self.rag, "summary_context_size", None),
                )
            )
        else:
            entries.append(
                ModelCatalogEntry(
                    model_id=model_id,
                    slug=model_id,
                    provider=provider,
                    family=binding,
                    context_window=getattr(self.rag, "summary_context_size", None),
                    modalities={"input": ["text"], "output": ["text"], "raw": ["text->text"]},
                    privacy_flags={"hosted": True, "local": False, "private": False},
                )
            )
        configured_private_model = self._configured_private_local_model()
        if configured_private_model:
            entries.append(
                ModelCatalogEntry.local(
                    configured_private_model,
                    family=self._configured_private_local_binding(),
                    context_window=getattr(self.rag, "summary_context_size", None),
                )
            )
        return ModelCatalog(entries=entries, source="lightrag-runtime")

    def _model_func_for(self, model: ModelCatalogEntry | None) -> Callable[..., object] | None:
        configured_private_model = self._configured_private_local_model()
        if model is None or not configured_private_model:
            return None
        if model.model_id != configured_private_model:
            return None
        if self._configured_private_local_binding() != "ollama":
            return None

        async def private_ollama_model_complete(
            prompt,
            system_prompt=None,
            history_messages=[],
            enable_cot: bool = False,
            keyword_extraction=False,
            **kwargs,
        ):
            from lightrag.llm.ollama import _ollama_model_if_cache

            keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
            if keyword_extraction:
                kwargs["format"] = "json"
            kwargs.pop("hashing_kv", None)
            kwargs["host"] = os.getenv("LITTLE_BULL_PRIVATE_LOCAL_HOST", "http://localhost:11434")
            kwargs["timeout"] = int(os.getenv("LITTLE_BULL_PRIVATE_LOCAL_TIMEOUT", "60"))
            api_key = os.getenv("LITTLE_BULL_PRIVATE_LOCAL_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key
            return await _ollama_model_if_cache(
                configured_private_model.split("/", 1)[-1],
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                enable_cot=enable_cot,
                **kwargs,
            )

        return private_ollama_model_complete

    def _active_binding(self) -> str:
        return str(
            getattr(self.rag, "little_bull_llm_binding", None)
            or os.getenv("LLM_BINDING")
            or "ollama"
        ).strip().lower()

    def _active_model_name(self) -> str:
        return str(
            getattr(self.rag, "little_bull_llm_model", None)
            or getattr(self.rag, "llm_model_name", None)
            or os.getenv("LLM_MODEL")
            or "unknown"
        ).strip()

    def _active_provider(self) -> str:
        host = self._active_host().lower()
        if "openrouter.ai" in host:
            return "openrouter"
        return self._active_binding()

    def _active_host(self) -> str:
        return str(
            getattr(self.rag, "little_bull_llm_host", None)
            or os.getenv("LLM_BINDING_HOST")
            or ""
        ).strip()

    @staticmethod
    def _configured_private_local_model() -> str | None:
        raw = os.getenv("LITTLE_BULL_PRIVATE_LOCAL_MODEL")
        if not raw:
            return None
        model = raw.strip()
        if not model:
            return None
        return model if "/" in model else f"{PrivateLocalGateway._configured_private_local_binding()}/{model}"

    @staticmethod
    def _configured_private_local_binding() -> str:
        return os.getenv("LITTLE_BULL_PRIVATE_LOCAL_BINDING", "ollama").strip().lower()

    @staticmethod
    def _local_bindings() -> set[str]:
        raw = os.getenv("LITTLE_BULL_PRIVATE_LOCAL_BINDINGS", "ollama,lollms")
        return {item.strip().lower() for item in raw.split(",") if item.strip()}
