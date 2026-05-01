from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp

from .catalog import ModelCatalog, ModelCatalogEntry


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class OpenRouterCatalogClient:
    """OpenRouter model catalog client.

    `/models/user` is preferred when an API key is present because it reflects
    provider preferences, privacy settings, and guardrails for the account. The
    public `/models` endpoint is retained as a safe fallback.
    """

    api_key: str | None = None
    base_url: str = OPENROUTER_BASE_URL
    app_referer: str | None = None
    app_title: str = "LightRAG Enterprise"
    timeout_seconds: int = 20
    ttl_seconds: int = 3600
    _catalog: ModelCatalog | None = field(default=None, init=False, repr=False)
    _expires_at: datetime | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(cls) -> "OpenRouterCatalogClient":
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_BINDING_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            app_referer=os.getenv("OPENROUTER_APP_REFERER"),
            app_title=os.getenv("OPENROUTER_APP_TITLE", "LightRAG Enterprise"),
            ttl_seconds=int(os.getenv("MODEL_CATALOG_TTL_SECONDS", "3600")),
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "LightRAG Enterprise Model Gateway",
            "X-OpenRouter-Title": self.app_title,
        }
        if self.app_referer:
            headers["HTTP-Referer"] = self.app_referer
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def fetch_catalog(
        self, *, force: bool = False, account_scoped: bool = True
    ) -> ModelCatalog:
        now = datetime.now(timezone.utc)
        if (
            not force
            and self._catalog is not None
            and self._expires_at is not None
            and now < self._expires_at
        ):
            return self._catalog

        payload = await self._fetch_payload(account_scoped=account_scoped)
        entries = [
            ModelCatalogEntry.from_openrouter_model(item, synced_at=now)
            for item in payload.get("data", [])
        ]
        source = (
            "openrouter:/models/user"
            if account_scoped and self.api_key
            else "openrouter:/models"
        )
        catalog = ModelCatalog(entries=entries, synced_at=now, source=source)
        self._catalog = catalog
        self._expires_at = now + timedelta(seconds=self.ttl_seconds)
        return catalog

    async def _fetch_payload(self, *, account_scoped: bool) -> dict[str, Any]:
        endpoints = []
        if account_scoped and self.api_key:
            endpoints.append("/models/user")
        endpoints.append("/models")

        last_error: Exception | None = None
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
        ) as session:
            for endpoint in endpoints:
                url = f"{self.base_url.rstrip('/')}{endpoint}"
                try:
                    async with session.get(url, headers=self._headers()) as response:
                        if response.status == 401 and endpoint == "/models/user":
                            continue
                        response.raise_for_status()
                        return await response.json()
                except (
                    Exception
                ) as exc:  # pragma: no cover - exercised by fallback tests
                    last_error = exc
                    continue
        if self._catalog is not None:
            return self._catalog.to_dict()
        raise RuntimeError(f"Unable to sync OpenRouter catalog: {last_error}")


async def sync_openrouter_catalog(
    output_path: str | Path,
    *,
    client: OpenRouterCatalogClient | None = None,
    account_scoped: bool = True,
    force: bool = False,
) -> ModelCatalog:
    """Sync catalog to disk for jobs, admin APIs, and offline fallback."""

    catalog_client = client or OpenRouterCatalogClient.from_env()
    catalog = await catalog_client.fetch_catalog(
        force=force, account_scoped=account_scoped
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(catalog.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return catalog
