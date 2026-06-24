"""
LightRAG FastAPI Server
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
import json
import os
import re
import logging
import logging.config
import sys
import textwrap
import uvicorn
import pipmaster as pm
from typing import Any
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path
from ascii_colors import ASCIIColors
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from lightrag.api.utils_api import (
    get_combined_auth_dependency,
    display_splash_screen,
    check_env_file,
)
from .config import (
    global_args,
    update_uvicorn_mode_config,
    get_default_host,
    resolve_asymmetric_embedding_opt_in,
    PREFIX_ASYMMETRIC_EMBEDDING_BINDINGS,
)
from lightrag.utils import get_env_value
from lightrag import LightRAG, ROLES, RoleLLMConfig, __version__ as core_version
from lightrag.api import __api_version__
from lightrag.utils import EmbeddingFunc
from lightrag.constants import (
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_FILENAME,
)
from lightrag.api.routers.document_routes import (
    DocumentManager,
    create_document_routes,
)
from lightrag.parser.plugins import load_third_party_parsers
from lightrag.parser.routing import (
    parser_rules_from_env,
    validate_parser_routing_config,
)
from lightrag.parser.external.mineru.cache import MinerUParserOptions
from lightrag.api.routers.query_routes import create_query_routes
from lightrag.api.routers.graph_routes import create_graph_routes
from lightrag.api.routers.ollama_api import OllamaAPI

from lightrag.utils import logger, set_verbose_debug
from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_default_workspace,
    # set_default_workspace,
    cleanup_keyed_lock,
    finalize_share_data,
)
from fastapi.security import OAuth2PasswordRequestForm
from lightrag.api.auth import auth_handler

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


webui_title = os.getenv("WEBUI_TITLE")
webui_description = os.getenv("WEBUI_DESCRIPTION")

# Global authentication configuration
auth_configured = bool(auth_handler.accounts)


def _inject_swagger_theme(html: str, theme: str) -> str:
    if theme not in {"dark", "light"}:
        theme = "auto"

    # The script resolves dark / light / (auto + prefers-color-scheme) into a
    # single boolean attribute `data-lightrag-docs-dark` on <html>. CSS below
    # only matches when that attribute is present, so light/auto-light paths
    # leave Swagger UI's default palette untouched.
    theme_snippet = textwrap.dedent(
        f"""
        <script>
          (function () {{
            var ALLOWED = {{ dark: 1, light: 1, auto: 1 }};
            var currentTheme = {json.dumps(theme)};
            var mql = window.matchMedia('(prefers-color-scheme: dark)');
            function resolveDark(value) {{
              if (value === 'dark') return true;
              if (value === 'auto') return mql.matches;
              return false;
            }}
            function apply(value) {{
              currentTheme = ALLOWED[value] ? value : 'auto';
              var root = document.documentElement;
              if (resolveDark(currentTheme)) {{
                root.setAttribute('data-lightrag-docs-dark', '1');
              }} else {{
                root.removeAttribute('data-lightrag-docs-dark');
              }}
            }}
            apply(currentTheme);
            // Re-resolve when the OS theme flips while `theme=auto` is active.
            var onMqlChange = function () {{ apply(currentTheme); }};
            if (mql.addEventListener) mql.addEventListener('change', onMqlChange);
            else if (mql.addListener) mql.addListener(onMqlChange);
            window.addEventListener('message', function (event) {{
              var data = event.data;
              if (!data || data.type !== 'lightrag:set-docs-theme') return;
              apply(data.theme);
            }});
          }})();
        </script>
        <style>
          html[data-lightrag-docs-dark="1"] {{
            color-scheme: dark;
          }}

          html[data-lightrag-docs-dark="1"] body,
          html[data-lightrag-docs-dark="1"] .swagger-ui,
          html[data-lightrag-docs-dark="1"] .swagger-ui .scheme-container,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model-box,
          html[data-lightrag-docs-dark="1"] .swagger-ui .opblock,
          html[data-lightrag-docs-dark="1"] .swagger-ui .dialog-ux .modal-ux,
          html[data-lightrag-docs-dark="1"] .swagger-ui .auth-container {{
            background: #0f172a;
            color: #e5e7eb;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .info .title,
          html[data-lightrag-docs-dark="1"] .swagger-ui .opblock-tag,
          html[data-lightrag-docs-dark="1"] .swagger-ui .opblock .opblock-summary-description,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model-title,
          html[data-lightrag-docs-dark="1"] .swagger-ui .parameter__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .parameter__type,
          html[data-lightrag-docs-dark="1"] .swagger-ui .response-col_status,
          html[data-lightrag-docs-dark="1"] .swagger-ui .response-col_description,
          html[data-lightrag-docs-dark="1"] .swagger-ui .auth-container h4,
          html[data-lightrag-docs-dark="1"] .swagger-ui .auth-container label,
          html[data-lightrag-docs-dark="1"] .swagger-ui .auth-container p,
          html[data-lightrag-docs-dark="1"] .swagger-ui .markdown p,
          html[data-lightrag-docs-dark="1"] .swagger-ui .markdown li,
          html[data-lightrag-docs-dark="1"] .swagger-ui .renderedMarkdown p,
          html[data-lightrag-docs-dark="1"] .swagger-ui .renderedMarkdown li,
          html[data-lightrag-docs-dark="1"] .swagger-ui table thead tr th,
          html[data-lightrag-docs-dark="1"] .swagger-ui table tbody tr td,
          html[data-lightrag-docs-dark="1"] .swagger-ui .tab li,
          html[data-lightrag-docs-dark="1"] .swagger-ui .tab li button.tablinks {{
            color: #e5e7eb;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .opblock-description-wrapper p,
          html[data-lightrag-docs-dark="1"] .swagger-ui .opblock-external-docs-wrapper p,
          html[data-lightrag-docs-dark="1"] .swagger-ui .response-col_links {{
            color: #cbd5f5;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui input,
          html[data-lightrag-docs-dark="1"] .swagger-ui textarea,
          html[data-lightrag-docs-dark="1"] .swagger-ui select {{
            background: #020617;
            border-color: #334155;
            color: #f8fafc;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .markdown code,
          html[data-lightrag-docs-dark="1"] .swagger-ui .renderedMarkdown code,
          html[data-lightrag-docs-dark="1"] .swagger-ui .highlight-code,
          html[data-lightrag-docs-dark="1"] .swagger-ui .highlight-code pre,
          html[data-lightrag-docs-dark="1"] .swagger-ui .microlight,
          html[data-lightrag-docs-dark="1"] .swagger-ui .body-param__example,
          html[data-lightrag-docs-dark="1"] .swagger-ui .example,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model-example pre {{
            background: #020617;
            color: #e2e8f0;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui table thead tr th,
          html[data-lightrag-docs-dark="1"] .swagger-ui table tbody tr td {{
            border-color: #334155;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .tab li.active button.tablinks,
          html[data-lightrag-docs-dark="1"] .swagger-ui .tab li.tabitem.active {{
            color: #f8fafc;
            border-bottom-color: #34d399;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .btn.authorize,
          html[data-lightrag-docs-dark="1"] .swagger-ui .auth-wrapper .authorize {{
            background: #064e3b;
            border-color: #34d399;
            color: #d1fae5;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .btn.authorize svg {{
            fill: #d1fae5;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui .dialog-ux .modal-ux,
          html[data-lightrag-docs-dark="1"] .swagger-ui .scheme-container,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models,
          html[data-lightrag-docs-dark="1"] .swagger-ui .opblock {{
            border-color: #334155;
            box-shadow: none;
          }}

          /* Schemas panel: section.models contains its own grey-on-grey
             buttons (`Schemas` header, each model row, "Expand all") that
             ignore the top-level body color. Force the whole subtree to
             use surface backgrounds and bright text. */
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models.is-open,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .model-container,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .models-control,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model-box {{
            background: #111827;
            border-color: #334155;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4 button,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4 a,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4 span,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .models-control,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .models-control button,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .model-toggle,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model .model-title__text,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model .property,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model .prop-name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model .prop-type,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model .prop-format,
          html[data-lightrag-docs-dark="1"] .swagger-ui .expand-operation,
          html[data-lightrag-docs-dark="1"] .swagger-ui .expand-operation span {{
            color: #e5e7eb;
          }}

          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4 svg,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .models-control svg,
          html[data-lightrag-docs-dark="1"] .swagger-ui .model-toggle::after,
          html[data-lightrag-docs-dark="1"] .swagger-ui .expand-operation svg,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-accordion__icon svg,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-accordion__icon svg path {{
            fill: #e5e7eb;
          }}

          /* The "Expand all" pill and per-row toggle buttons inherit a light
             grey background from Swagger; clear it so they don't punch a
             pale rectangle into the dark panel. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .expand-operation,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models h4 button,
          html[data-lightrag-docs-dark="1"] .swagger-ui section.models .models-control button {{
            background: transparent;
          }}

          /* Swagger's new JSON Schema 2020-12 renderer hard-codes light-mode
             greys (#505050 / #3b4151 / #afaeae / #6b6b6b) on every title,
             keyword, attribute and json-viewer node — completely independent
             from the .model / .swagger-ui ancestors we already restyled.
             Override the whole renderer subtree so model/property names,
             types, and the per-row "Expand all" button stay readable. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12__title,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-property .json-schema-2020-12__title,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-expand-deep-button,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-accordion,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__name--primary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__value--primary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12__attribute,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12__attribute--primary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__name--primary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__value--primary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--const .json-schema-2020-12-json-viewer__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--const .json-schema-2020-12-json-viewer__value,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--default .json-schema-2020-12-json-viewer__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--default .json-schema-2020-12-json-viewer__value,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--enum .json-schema-2020-12-json-viewer__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--enum .json-schema-2020-12-json-viewer__value,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--examples .json-schema-2020-12-json-viewer__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--examples .json-schema-2020-12-json-viewer__value {{
            color: #e5e7eb;
          }}

          /* Secondary / extension / description text — keep them visible but
             dimmer than primary titles. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__name--secondary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__value,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__value--secondary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__name--extension,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__value--extension,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword--description,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__name--secondary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__value,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__value--secondary,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__name--extension,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__value--extension,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer-extension-keyword .json-schema-2020-12-json-viewer__name,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer-extension-keyword .json-schema-2020-12-json-viewer__value,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12__attribute--muted {{
            color: #cbd5f5;
          }}

          /* The deep-expand button inside each schemas row has its own
             background and shouldn't paint a pale capsule on dark surface. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-expand-deep-button,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-accordion {{
            background-color: transparent;
          }}

          /* Restore Swagger's red warning palette. The broad keyword__value /
             __attribute / json-viewer__value overrides above otherwise win
             the cascade over `.json-schema-2020-12-*--warning` (higher
             specificity), flattening deprecated/schema-warning markers into
             plain text. Re-declared *after* the generic rules so equal-
             specificity selectors lose to these explicit ones. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-keyword__value--warning,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12-json-viewer__value--warning,
          html[data-lightrag-docs-dark="1"] .swagger-ui .json-schema-2020-12__attribute--warning {{
            color: #fca5a5;
            border-color: #fca5a5;
          }}

          /* `.model-toggle::after` paints its caret with a `background:url(
             data:image/svg+xml,…<path d=…/>)` embedded SVG whose path has no
             fill attribute and no currentColor reference — `fill` rules can't
             touch it. Invert the rendered pixels so the black arrow flips to
             white on the dark schema surface. The glyph is single-color, so
             invert has no perceptible side effect. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .model-toggle::after {{
            filter: invert(1);
          }}

          /* Per-operation Authorize lock icon. Swagger renders it via
             `<symbol id="locked|unlocked">` whose <path> has no fill attr
             and no currentColor reference; Swagger's CSS also never sets
             fill on .authorization__btn svg, leaving the path at the SVG
             default (black). Set fill on the outer <svg> — fill is inherited
             through <use> into the referenced symbol because the path itself
             is unstyled, so one declaration colors both locked and unlocked
             states. */
          html[data-lightrag-docs-dark="1"] .swagger-ui .authorization__btn svg,
          html[data-lightrag-docs-dark="1"] .swagger-ui .authorization__btn .locked,
          html[data-lightrag-docs-dark="1"] .swagger-ui .authorization__btn .unlocked {{
            fill: #e5e7eb;
          }}
        </style>
        """
    ).strip()

    needle = "</head>"
    if needle not in html:
        logger.warning(
            "Swagger UI HTML missing </head> tag; theme patch was skipped. "
            "FastAPI's swagger template may have changed."
        )
        return html
    return html.replace(needle, f"{theme_snippet}\n{needle}", 1)


# Fixed WebUI mount path. Used as `app.mount(WEBUI_PATH, ...)` and as the
# in-app component of `webuiPrefix` injected into window.__LIGHTRAG_CONFIG__
# (which the browser sees as `LIGHTRAG_API_PREFIX + WEBUI_PATH + "/"`).
# Not user-configurable: a single mount path simplifies the operator surface
# and matches how LightRAG is deployed in practice. See
# docs/MultiSiteDeployment.md.
WEBUI_PATH = "/webui"


def _normalize_api_prefix(value: str | None) -> str:
    """Canonicalize an API prefix before handing it to FastAPI's ``root_path``.

    Strips surrounding whitespace, ensures a leading slash, drops a trailing
    slash, and treats empty/"/" as "no prefix". Raw CLI/env input like
    ``"site01"`` or ``"/site01/"`` would otherwise feed an invalid form to
    FastAPI and to the WebUI prefix injection.
    """
    if value is None:
        return ""
    value = value.strip()
    if not value or value == "/":
        return ""
    if not value.startswith("/"):
        value = "/" + value
    return value.rstrip("/")


class _RootPathNormalizationMiddleware:
    """Make Mount sub-apps work when the reverse proxy strips the API prefix.

    When ``LIGHTRAG_API_PREFIX=/site01`` and nginx strips ``/site01`` before
    forwarding, the backend sees ``scope["path"]="/webui/"`` while FastAPI's
    ``__call__`` sets ``scope["root_path"]="/site01"``. Starlette's outer
    Mount.matches still hits via ``get_route_path`` 's fallback branch (path
    not starting with root_path is returned unchanged), but it mutates the
    child scope to ``root_path="/site01/webui"`` without touching
    ``scope["path"]``. The inner ``StaticFiles.get_path`` then sees a
    non-overlapping pair and falls through to a literal ``webui`` filename
    lookup → 404 on the actual file system.

    Prepending ``root_path`` to a non-prefixed ``scope["path"]`` restores the
    canonical ASGI form (path always contains root_path), matching what a
    verbatim-forwarding proxy produces natively. Plain Routes are unaffected
    because their handlers do not redo nested ``get_route_path`` resolution.

    See docs/MultiSiteDeployment.md for the deployment modes this enables.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            root_path = scope.get("root_path", "")
            path = scope.get("path", "")
            if root_path and not path.startswith(root_path):
                scope = {**scope, "path": root_path + path}
                raw_path = scope.get("raw_path")
                if isinstance(raw_path, (bytes, bytearray)):
                    scope["raw_path"] = root_path.encode("ascii") + bytes(raw_path)
        await self.app(scope, receive, send)


def _clean_workspace_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get_storage_workspace(storage: Any) -> str | None:
    if storage is None:
        return None

    effective_workspace = _clean_workspace_value(
        getattr(storage, "effective_workspace", None)
    )
    if effective_workspace:
        return effective_workspace

    final_namespace = _clean_workspace_value(getattr(storage, "final_namespace", None))
    namespace = _clean_workspace_value(getattr(storage, "namespace", None))
    if final_namespace and namespace:
        suffix = f"_{namespace}"
        if final_namespace.endswith(suffix):
            workspace = final_namespace[: -len(suffix)]
            if workspace:
                return workspace

    return _clean_workspace_value(getattr(storage, "workspace", None))


def _get_storage_workspaces(rag: Any) -> dict[str, str | None]:
    return {
        "kv_storage": _get_storage_workspace(getattr(rag, "full_docs", None)),
        "doc_status_storage": _get_storage_workspace(getattr(rag, "doc_status", None)),
        "graph_storage": _get_storage_workspace(
            getattr(rag, "chunk_entity_relation_graph", None)
        ),
        "vector_storage": _get_storage_workspace(getattr(rag, "entities_vdb", None)),
    }


def _build_mineru_status() -> dict[str, Any]:
    """Snapshot MinerU-related env vars for the /health endpoint.

    Reads env directly (no MinerURawClient instantiation — that has
    side effects like token validation). Reuses MinerUParserOptions to
    share defaulting logic with the actual parser path.
    """
    api_mode_raw = os.getenv("MINERU_API_MODE", "").strip().lower()
    api_mode: str | None = api_mode_raw or None
    endpoint = ""
    if api_mode == "official":
        endpoint = os.getenv("MINERU_OFFICIAL_ENDPOINT", "").strip()
    elif api_mode == "local":
        endpoint = os.getenv("MINERU_LOCAL_ENDPOINT", "").strip()

    options: dict[str, Any] = {}
    if api_mode in ("official", "local"):
        try:
            opts = MinerUParserOptions.from_env(api_mode=api_mode)
        except Exception:
            opts = None
        if opts is not None:
            options = {
                "language": opts.language,
                "enable_table": opts.enable_table,
                "enable_formula": opts.enable_formula,
            }
            if opts.api_mode == "official":
                options["model_version"] = opts.model_version
                options["is_ocr"] = opts.is_ocr
            else:
                options["local_backend"] = opts.local_backend
                options["local_parse_method"] = opts.local_parse_method
                options["local_image_analysis"] = opts.local_image_analysis

    return {"endpoint": endpoint, "api_mode": api_mode, "options": options}


def _build_docling_status() -> dict[str, Any]:
    """Snapshot Docling-related env vars for the /health endpoint."""
    endpoint = os.getenv("DOCLING_ENDPOINT", "").strip()
    if not endpoint:
        return {"endpoint": "", "options": {}}
    return {
        "endpoint": endpoint,
        "options": {
            "do_ocr": get_env_value("DOCLING_DO_OCR", True, bool),
            "force_ocr": get_env_value("DOCLING_FORCE_OCR", True, bool),
            "ocr_engine": os.getenv("DOCLING_OCR_ENGINE", "auto").strip() or "auto",
            "ocr_lang": os.getenv("DOCLING_OCR_LANG", "").strip(),
            "do_formula_enrichment": get_env_value(
                "DOCLING_DO_FORMULA_ENRICHMENT", False, bool
            ),
        },
    }


class LLMConfigCache:
    """Smart LLM and Embedding configuration cache class"""

    def __init__(self, args):
        self.args = args

        # Initialize configurations based on binding conditions
        self.openai_llm_options = None
        self.gemini_llm_options = None
        self.gemini_embedding_options = None
        self.ollama_llm_options = None
        self.ollama_embedding_options = None
        self.bedrock_llm_options = None

        # Only initialize and log OpenAI options when using OpenAI-related bindings
        if args.llm_binding in ["openai", "azure_openai"]:
            from lightrag.llm.binding_options import OpenAILLMOptions

            self.openai_llm_options = OpenAILLMOptions.options_dict(args)
            logger.info(f"OpenAI LLM Options: {self.openai_llm_options}")

        if args.llm_binding == "gemini":
            from lightrag.llm.binding_options import GeminiLLMOptions

            self.gemini_llm_options = GeminiLLMOptions.options_dict(args)
            logger.info(f"Gemini LLM Options: {self.gemini_llm_options}")

        if args.llm_binding == "bedrock":
            from lightrag.llm.binding_options import BedrockLLMOptions

            self.bedrock_llm_options = BedrockLLMOptions.options_dict(args)
            logger.info(f"Bedrock LLM Options: {self.bedrock_llm_options}")

        # Only initialize and log Ollama LLM options when using Ollama LLM binding
        if args.llm_binding == "ollama":
            try:
                from lightrag.llm.binding_options import OllamaLLMOptions

                self.ollama_llm_options = OllamaLLMOptions.options_dict(args)
                logger.info(f"Ollama LLM Options: {self.ollama_llm_options}")
            except ImportError:
                logger.warning(
                    "OllamaLLMOptions not available, using default configuration"
                )
                self.ollama_llm_options = {}

        # Only initialize and log Ollama Embedding options when using Ollama Embedding binding
        if args.embedding_binding == "ollama":
            try:
                from lightrag.llm.binding_options import OllamaEmbeddingOptions

                self.ollama_embedding_options = OllamaEmbeddingOptions.options_dict(
                    args
                )
                logger.info(
                    f"Ollama Embedding Options: {self.ollama_embedding_options}"
                )
            except ImportError:
                logger.warning(
                    "OllamaEmbeddingOptions not available, using default configuration"
                )
                self.ollama_embedding_options = {}

        # Only initialize and log Gemini Embedding options when using Gemini Embedding binding
        if args.embedding_binding == "gemini":
            try:
                from lightrag.llm.binding_options import GeminiEmbeddingOptions

                self.gemini_embedding_options = GeminiEmbeddingOptions.options_dict(
                    args
                )
                logger.info(
                    f"Gemini Embedding Options: {self.gemini_embedding_options}"
                )
            except ImportError:
                logger.warning(
                    "GeminiEmbeddingOptions not available, using default configuration"
                )
                self.gemini_embedding_options = {}


_PROVIDER_LOG_LABELS = {
    "azure_openai": "Azure OpenAI",
    "bedrock": "Bedrock",
    "gemini": "Gemini",
    "lollms": "Lollms",
    "ollama": "Ollama",
    "openai": "OpenAI",
}


def create_optimized_embedding_function(
    config_cache: LLMConfigCache,
    binding,
    model,
    host,
    api_key,
    args,
    document_prefix=None,
    query_prefix=None,
) -> EmbeddingFunc:
    """
    Create optimized embedding function and return an EmbeddingFunc instance
    with proper max_token_size inheritance from provider defaults.

    This function:
    1. Imports the provider embedding function
    2. Extracts max_token_size and embedding_dim from provider if it's an EmbeddingFunc
    3. Creates an optimized wrapper that calls the underlying function directly (avoiding double-wrapping)
    4. Returns a properly configured EmbeddingFunc instance

    Configuration Rules:
    - When EMBEDDING_MODEL is not set: Uses provider's default model and dimension
      (e.g., jina-embeddings-v4 with 2048 dims, text-embedding-3-small with 1536 dims)
    - When EMBEDDING_MODEL is set to a custom model: User MUST also set EMBEDDING_DIM
      to match the custom model's dimension (e.g., for jina-embeddings-v3, set EMBEDDING_DIM=1024)

    Note: The embedding_dim parameter is automatically injected by EmbeddingFunc wrapper
    when send_dimensions=True (enabled for Jina and Gemini bindings). This wrapper calls
    the underlying provider function directly (.func) to avoid double-wrapping, so we must
    explicitly pass embedding_dim to the provider's underlying function.
    """

    # Step 1: Import provider function and extract default attributes
    provider_func = None
    provider_max_token_size = None
    provider_embedding_dim = None
    provider_supports_asymmetric = False

    try:
        if binding == "openai":
            from lightrag.llm.openai import openai_embed

            provider_func = openai_embed
        elif binding == "ollama":
            from lightrag.llm.ollama import ollama_embed

            provider_func = ollama_embed
        elif binding == "gemini":
            from lightrag.llm.gemini import gemini_embed

            provider_func = gemini_embed
        elif binding == "jina":
            from lightrag.llm.jina import jina_embed

            provider_func = jina_embed
        elif binding == "azure_openai":
            from lightrag.llm.azure_openai import azure_openai_embed

            provider_func = azure_openai_embed
        elif binding == "bedrock":
            from lightrag.llm.bedrock import bedrock_embed

            provider_func = bedrock_embed
        elif binding == "lollms":
            from lightrag.llm.lollms import lollms_embed

            provider_func = lollms_embed
        elif binding == "voyageai":
            from lightrag.llm.voyageai import voyageai_embed

            provider_func = voyageai_embed
        # Extract attributes if provider is an EmbeddingFunc
        if provider_func and isinstance(provider_func, EmbeddingFunc):
            provider_max_token_size = provider_func.max_token_size
            provider_embedding_dim = provider_func.embedding_dim
            provider_supports_asymmetric = provider_func.supports_asymmetric
            logger.debug(
                f"Extracted from {binding} provider: "
                f"max_token_size={provider_max_token_size}, "
                f"embedding_dim={provider_embedding_dim}, "
                f"supports_asymmetric={provider_supports_asymmetric}"
            )
    except ImportError as e:
        logger.warning(f"Could not import provider function for {binding}: {e}")

    # Step 2: Apply priority (user config > provider default)
    # For max_token_size: explicit env var > provider default > None
    final_max_token_size = args.embedding_token_limit or provider_max_token_size
    # For embedding_dim: user config (always has value) takes priority
    # Only use provider default if user config is explicitly None (which shouldn't happen)
    final_embedding_dim = (
        args.embedding_dim if args.embedding_dim else provider_embedding_dim
    )
    # Asymmetric embedding is explicit opt-in only. Provider-specific
    # validation decides whether task parameters or prefixes are required.
    asymmetric_opt_in = resolve_asymmetric_embedding_opt_in(
        binding=binding,
        embedding_asymmetric=args.embedding_asymmetric,
        embedding_asymmetric_configured=args.embedding_asymmetric_configured,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        query_prefix_configured=args.embedding_query_prefix_configured,
        document_prefix_configured=args.embedding_document_prefix_configured,
    )

    # Step 3: Create optimized embedding function (calls underlying function directly)
    # Note: When model is None, each binding will use its own default model
    async def optimized_embedding_function(
        texts, embedding_dim=None, context="document"
    ):
        try:
            if binding == "lollms":
                from lightrag.llm.lollms import lollms_embed

                # Get real function, skip EmbeddingFunc wrapper if present
                actual_func = (
                    lollms_embed.func
                    if isinstance(lollms_embed, EmbeddingFunc)
                    else lollms_embed
                )
                # lollms embed_model is not used (server uses configured vectorizer)
                # Only pass base_url and api_key
                return await actual_func(texts, base_url=host, api_key=api_key)
            elif binding == "ollama":
                from lightrag.llm.ollama import ollama_embed

                # Get real function, skip EmbeddingFunc wrapper if present
                actual_func = (
                    ollama_embed.func
                    if isinstance(ollama_embed, EmbeddingFunc)
                    else ollama_embed
                )

                # Use pre-processed configuration if available
                if config_cache.ollama_embedding_options is not None:
                    ollama_options = config_cache.ollama_embedding_options
                else:
                    from lightrag.llm.binding_options import OllamaEmbeddingOptions

                    ollama_options = OllamaEmbeddingOptions.options_dict(args)

                # Pass embed_model only if provided, let function use its default (bge-m3:latest)
                kwargs = {
                    "texts": texts,
                    "host": host,
                    "api_key": api_key,
                    "options": ollama_options,
                }
                if provider_supports_asymmetric and asymmetric_opt_in:
                    kwargs["context"] = context
                    if query_prefix:
                        kwargs["query_prefix"] = query_prefix
                    if document_prefix:
                        kwargs["document_prefix"] = document_prefix
                if model:
                    kwargs["embed_model"] = model
                return await actual_func(**kwargs)
            elif binding == "azure_openai":
                from lightrag.llm.azure_openai import azure_openai_embed

                actual_func = (
                    azure_openai_embed.func
                    if isinstance(azure_openai_embed, EmbeddingFunc)
                    else azure_openai_embed
                )
                # Pass model only if provided, let function use its default otherwise
                kwargs = {
                    "texts": texts,
                    "api_key": api_key,
                    "embedding_dim": embedding_dim,
                }
                if model:
                    kwargs["model"] = model
                if provider_supports_asymmetric and asymmetric_opt_in:
                    kwargs["context"] = context
                    if query_prefix:
                        kwargs["query_prefix"] = query_prefix
                    if document_prefix:
                        kwargs["document_prefix"] = document_prefix
                return await actual_func(**kwargs)
            elif binding == "bedrock":
                from lightrag.llm.bedrock import bedrock_embed

                actual_func = (
                    bedrock_embed.func
                    if isinstance(bedrock_embed, EmbeddingFunc)
                    else bedrock_embed
                )
                # Pass model only if provided, let function use its default otherwise
                kwargs = {
                    "texts": texts,
                    "aws_region": getattr(args, "aws_region", None),
                    "aws_access_key_id": getattr(args, "aws_access_key_id", None),
                    "aws_secret_access_key": getattr(
                        args, "aws_secret_access_key", None
                    ),
                    "aws_session_token": getattr(args, "aws_session_token", None),
                }
                if host is not None:
                    kwargs["endpoint_url"] = host
                if model:
                    kwargs["model"] = model
                return await actual_func(**kwargs)
            elif binding == "jina":
                from lightrag.llm.jina import jina_embed

                actual_func = (
                    jina_embed.func
                    if isinstance(jina_embed, EmbeddingFunc)
                    else jina_embed
                )
                # Pass model only if provided, let function use its default (jina-embeddings-v4)
                kwargs = {
                    "texts": texts,
                    "embedding_dim": embedding_dim,
                    "base_url": host,
                    "api_key": api_key,
                }
                if model:
                    kwargs["model"] = model
                if provider_supports_asymmetric and asymmetric_opt_in:
                    kwargs["context"] = context
                    kwargs["task"] = None
                return await actual_func(**kwargs)
            elif binding == "gemini":
                from lightrag.llm.gemini import gemini_embed

                actual_func = (
                    gemini_embed.func
                    if isinstance(gemini_embed, EmbeddingFunc)
                    else gemini_embed
                )

                # Use pre-processed configuration if available
                if config_cache.gemini_embedding_options is not None:
                    gemini_options = config_cache.gemini_embedding_options
                else:
                    from lightrag.llm.binding_options import GeminiEmbeddingOptions

                    gemini_options = GeminiEmbeddingOptions.options_dict(args)
                # Pass model only if provided, let function use its default (gemini-embedding-001)
                kwargs = {
                    "texts": texts,
                    "base_url": host,
                    "api_key": api_key,
                    "embedding_dim": embedding_dim,
                }
                if model:
                    kwargs["model"] = model
                task_type = gemini_options.get("task_type")
                if task_type is not None:
                    kwargs["task_type"] = task_type
                if provider_supports_asymmetric and asymmetric_opt_in:
                    kwargs["context"] = context
                return await actual_func(**kwargs)
            elif binding == "voyageai":
                from lightrag.llm.voyageai import voyageai_embed

                actual_func = (
                    voyageai_embed.func
                    if isinstance(voyageai_embed, EmbeddingFunc)
                    else voyageai_embed
                )
                kwargs = {
                    "texts": texts,
                    "api_key": api_key,
                    "embedding_dim": embedding_dim,
                }
                if model:
                    kwargs["model"] = model
                if provider_supports_asymmetric and asymmetric_opt_in:
                    kwargs["context"] = context
                return await actual_func(**kwargs)
            else:  # openai and compatible
                from lightrag.llm.openai import openai_embed

                actual_func = (
                    openai_embed.func
                    if isinstance(openai_embed, EmbeddingFunc)
                    else openai_embed
                )
                # Pass model only if provided, let function use its default (text-embedding-3-small)
                kwargs = {
                    "texts": texts,
                    "base_url": host,
                    "api_key": api_key,
                    "embedding_dim": embedding_dim,
                }
                if model:
                    kwargs["model"] = model
                if provider_supports_asymmetric and asymmetric_opt_in:
                    kwargs["context"] = context
                    if query_prefix:
                        kwargs["query_prefix"] = query_prefix
                    if document_prefix:
                        kwargs["document_prefix"] = document_prefix
                return await actual_func(**kwargs)
        except ImportError as e:
            raise Exception(f"Failed to import {binding} embedding: {e}")

    # Step 4: Wrap in EmbeddingFunc and return
    embedding_func_instance = EmbeddingFunc(
        embedding_dim=final_embedding_dim,
        func=optimized_embedding_function,
        max_token_size=final_max_token_size,
        send_dimensions=False,  # Will be set later based on binding requirements
        model_name=model,
        supports_asymmetric=provider_supports_asymmetric and asymmetric_opt_in,
    )

    # Log final embedding configuration. Only include prefix info when
    # prefixes will actually be applied (prefix-based asymmetric mode).
    prefix_info = ""
    if (
        asymmetric_opt_in
        and binding in PREFIX_ASYMMETRIC_EMBEDDING_BINDINGS
        and (document_prefix or query_prefix)
    ):
        prefix_info = f" document_prefix={repr(document_prefix)} query_prefix={repr(query_prefix)}"
    logger.info(
        f"Embedding config: binding={binding} model={model} "
        f"embedding_dim={final_embedding_dim} max_token_size={final_max_token_size}{prefix_info}"
    )

    return embedding_func_instance


def create_embedding_function_from_args(
    args, config_cache: LLMConfigCache | None = None
) -> EmbeddingFunc:
    """Build the fully configured EmbeddingFunc used by the LightRAG server.

    Combines the provider embedding factory with the send_dimensions policy
    so that offline tools (e.g. lightrag-rebuild-vdb) can embed in exactly
    the same vector space as the running server.
    """
    import inspect

    if config_cache is None:
        config_cache = LLMConfigCache(args)

    # Create the EmbeddingFunc instance (now returns complete EmbeddingFunc with max_token_size)
    embedding_func = create_optimized_embedding_function(
        config_cache=config_cache,
        binding=args.embedding_binding,
        model=args.embedding_model,
        host=args.embedding_binding_host,
        api_key=None
        if args.embedding_binding == "bedrock"
        else args.embedding_binding_api_key,
        args=args,
        document_prefix=args.embedding_document_prefix,
        query_prefix=args.embedding_query_prefix,
    )

    # Get embedding_send_dim from centralized configuration
    embedding_send_dim = args.embedding_send_dim

    # Check if the underlying function signature has embedding_dim parameter
    sig = inspect.signature(embedding_func.func)
    has_embedding_dim_param = "embedding_dim" in sig.parameters

    # Determine send_dimensions value based on binding type
    # Jina and Gemini REQUIRE dimension parameter (forced to True)
    # OpenAI and others: controlled by EMBEDDING_SEND_DIM environment variable
    if args.embedding_binding in ["jina", "gemini"]:
        # Jina and Gemini APIs require dimension parameter - always send it
        send_dimensions = has_embedding_dim_param
        dimension_control = f"forced by {args.embedding_binding.title()} API"
    else:
        # For OpenAI and other bindings, respect EMBEDDING_SEND_DIM setting
        send_dimensions = embedding_send_dim and has_embedding_dim_param
        if send_dimensions or not embedding_send_dim:
            dimension_control = "by env var"
        else:
            dimension_control = "by not hasparam"

    # Set send_dimensions on the EmbeddingFunc instance
    embedding_func.send_dimensions = send_dimensions

    logger.info(
        f"Send embedding dimension: {send_dimensions} {dimension_control} "
        f"(dimensions={embedding_func.embedding_dim}, has_param={has_embedding_dim_param}, "
        f"binding={args.embedding_binding})"
    )

    # Log max_token_size source
    if embedding_func.max_token_size:
        source = (
            "env variable"
            if args.embedding_token_limit
            else f"{args.embedding_binding} provider default"
        )
        logger.info(
            f"Embedding max_token_size: {embedding_func.max_token_size} (from {source})"
        )
    else:
        logger.info(
            "Embedding max_token_size: None (Embedding token limit is disabled)."
        )

    return embedding_func


def _provider_log_label(binding: Any) -> str:
    binding_name = str(binding)
    return _PROVIDER_LOG_LABELS.get(
        binding_name, binding_name.replace("_", " ").title()
    )


def _log_role_provider_options(rag: Any) -> None:
    """Log sanitized provider options for every role LLM."""
    try:
        role_configs = rag.get_llm_role_config()
    except Exception as e:
        logger.warning(f"Failed to read role LLM configuration for logging: {e}")
        return

    logger.info("Role LLM Option:")

    for spec in ROLES:
        role_config = role_configs.get(spec.name)
        if not isinstance(role_config, dict):
            continue

        metadata = role_config.get("metadata") or {}
        binding = role_config.get("binding") or metadata.get("binding")
        if not binding:
            continue

        provider_options = metadata.get("provider_options") or {}
        logger.info(
            " - %s: %s %s",
            spec.name,
            _provider_log_label(binding),
            provider_options,
        )


def check_frontend_build():
    """Check if frontend is built and optionally check if source is up-to-date

    Returns:
        tuple: (assets_exist: bool, is_outdated: bool)
            - assets_exist: True if WebUI build files exist
            - is_outdated: True if source is newer than build (only in dev environment)
    """
    webui_dir = Path(__file__).parent / "webui"
    index_html = webui_dir / "index.html"

    # 1. Check if build files exist
    if not index_html.exists():
        ASCIIColors.yellow("\n" + "=" * 80)
        ASCIIColors.yellow("WARNING: Frontend Not Built")
        ASCIIColors.yellow("=" * 80)
        ASCIIColors.yellow("The WebUI frontend has not been built yet.")
        ASCIIColors.yellow("The API server will start without the WebUI interface.")
        ASCIIColors.yellow(
            "\nTo enable WebUI, build the frontend using these commands:\n"
        )
        ASCIIColors.cyan("    cd lightrag_webui")
        ASCIIColors.cyan("    bun install --frozen-lockfile")
        ASCIIColors.cyan("    bun run build")
        ASCIIColors.cyan("    cd ..")
        ASCIIColors.yellow("\nThen restart the service.\n")
        ASCIIColors.cyan(
            "Note: Make sure you have Bun installed. Visit https://bun.sh for installation."
        )
        ASCIIColors.yellow("=" * 80 + "\n")
        return (False, False)  # Assets don't exist, not outdated

    # 2. Check if this is a development environment (source directory exists)
    try:
        source_dir = Path(__file__).parent.parent.parent / "lightrag_webui"
        src_dir = source_dir / "src"

        # Determine if this is a development environment: source directory exists and contains src directory
        if not source_dir.exists() or not src_dir.exists():
            # Production environment, skip source code check
            logger.debug(
                "Production environment detected, skipping source freshness check"
            )
            return (True, False)  # Assets exist, not outdated (prod environment)

        # Development environment, perform source code timestamp check
        logger.debug("Development environment detected, checking source freshness")

        # Source code file extensions (files to check)
        source_extensions = {
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".mjs",
            ".cjs",  # TypeScript/JavaScript
            ".css",
            ".scss",
            ".sass",
            ".less",  # Style files
            ".json",
            ".jsonc",  # Configuration/data files
            ".html",
            ".htm",  # Template files
            ".md",
            ".mdx",  # Markdown
        }

        # Key configuration files (in lightrag_webui root directory)
        key_files = [
            source_dir / "package.json",
            source_dir / "bun.lock",
            source_dir / "vite.config.ts",
            source_dir / "tsconfig.json",
            source_dir / "tailraid.config.js",
            source_dir / "index.html",
        ]

        # Get the latest modification time of source code
        latest_source_time = 0

        # Check source code files in src directory
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                # Only check source code files, ignore temporary files and logs
                if file_path.suffix.lower() in source_extensions:
                    mtime = file_path.stat().st_mtime
                    latest_source_time = max(latest_source_time, mtime)

        # Check key configuration files
        for key_file in key_files:
            if key_file.exists():
                mtime = key_file.stat().st_mtime
                latest_source_time = max(latest_source_time, mtime)

        # Get build time
        build_time = index_html.stat().st_mtime

        # Compare timestamps (5 second tolerance to avoid file system time precision issues)
        if latest_source_time > build_time + 5:
            ASCIIColors.yellow("\n" + "=" * 80)
            ASCIIColors.yellow("WARNING: Frontend Source Code Has Been Updated")
            ASCIIColors.yellow("=" * 80)
            ASCIIColors.yellow(
                "The frontend source code is newer than the current build."
            )
            ASCIIColors.yellow(
                "This might happen after 'git pull' or manual code changes.\n"
            )
            ASCIIColors.cyan(
                "Recommended: Rebuild the frontend to use the latest changes:"
            )
            ASCIIColors.cyan("    cd lightrag_webui")
            ASCIIColors.cyan("    bun install --frozen-lockfile")
            ASCIIColors.cyan("    bun run build")
            ASCIIColors.cyan("    cd ..")
            ASCIIColors.yellow("\nThe server will continue with the current build.")
            ASCIIColors.yellow("=" * 80 + "\n")
            return (True, True)  # Assets exist, outdated
        else:
            logger.info("Frontend build is up-to-date")
            return (True, False)  # Assets exist, up-to-date

    except Exception as e:
        # If check fails, log warning but don't affect startup
        logger.warning(f"Failed to check frontend source freshness: {e}")
        return (True, False)  # Assume assets exist and up-to-date on error


def create_app(args):
    # Check frontend build first and get status
    webui_assets_exist, is_frontend_outdated = check_frontend_build()

    # Create unified API version display with warning symbol if frontend is outdated
    api_version_display = (
        f"{__api_version__}⚠️" if is_frontend_outdated else __api_version__
    )

    # Setup logging
    logger.setLevel(args.log_level)
    set_verbose_debug(args.verbose)
    # Discover third-party parser engines (``lightrag.parsers`` entry points)
    # BEFORE validating routing rules, so LIGHTRAG_PARSER may reference them.
    load_third_party_parsers()
    validate_parser_routing_config()

    # Create configuration cache (this will output configuration logs)
    config_cache = LLMConfigCache(args)

    # Verify that bindings are correctly setup
    if args.llm_binding not in [
        "lollms",
        "ollama",
        "openai",
        "azure_openai",
        "bedrock",
        "gemini",
    ]:
        raise Exception("llm binding not supported")

    if args.embedding_binding not in [
        "lollms",
        "ollama",
        "openai",
        "azure_openai",
        "bedrock",
        "jina",
        "gemini",
        "voyageai",
    ]:
        raise Exception(f"embedding binding '{args.embedding_binding}' not supported")

    # Set default hosts if not provided
    if args.llm_binding_host is None:
        args.llm_binding_host = get_default_host(args.llm_binding)

    if args.embedding_binding_host is None:
        args.embedding_binding_host = get_default_host(args.embedding_binding)

    # Add SSL validation
    if args.ssl:
        if not args.ssl_certfile or not args.ssl_keyfile:
            raise Exception(
                "SSL certificate and key files must be provided when SSL is enabled"
            )
        if not os.path.exists(args.ssl_certfile):
            raise Exception(f"SSL certificate file not found: {args.ssl_certfile}")
        if not os.path.exists(args.ssl_keyfile):
            raise Exception(f"SSL key file not found: {args.ssl_keyfile}")

    # Check if API key is provided either through env var or args
    api_key = os.getenv("LIGHTRAG_API_KEY") or args.key

    # Initialize document manager with workspace support for data isolation
    doc_manager = DocumentManager(args.input_dir, workspace=args.workspace)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        # Store background tasks
        app.state.background_tasks = set()

        try:
            # Initialize database connections
            # Note: initialize_storages() now auto-initializes pipeline_status for rag.workspace
            await rag.initialize_storages()

            # Data migration regardless of storage implementation
            await rag.check_and_migrate_data()

            ASCIIColors.green("\nServer is ready to accept connections! 🚀\n")

            yield

        finally:
            # Clean up database connections
            await rag.finalize_storages()

            if "LIGHTRAG_GUNICORN_MODE" not in os.environ:
                # Only perform cleanup in Uvicorn single-process mode
                logger.debug("Unvicorn Mode: finalizing shared storage...")
                finalize_share_data()
            else:
                # In Gunicorn mode with preload_app=True, cleanup is handled by on_exit hooks
                logger.debug(
                    "Gunicorn Mode: postpone shared storage finalization to master process"
                )

    base_description = (
        "Providing API for LightRAG core, Web UI and Ollama Model Emulation"
    )
    swagger_description = (
        base_description
        + (" (API-Key Enabled)" if api_key else "")
        + "\n\n[View ReDoc documentation](/redoc)"
    )

    # The WebUI mount path is fixed at "/webui" — see
    # docs/MultiSiteDeployment.md for the rationale.
    api_prefix = _normalize_api_prefix(getattr(args, "api_prefix", None))
    webui_path = WEBUI_PATH

    app_kwargs = {
        "title": "LightRAG Server API",
        "description": swagger_description,
        "version": __api_version__,
        "openapi_url": "/openapi.json",
        "docs_url": None,  # custom endpoint for offline Swagger support
        "redoc_url": "/redoc",
        "root_path": api_prefix if api_prefix else None,
        "lifespan": lifespan,
    }

    # Configure Swagger UI parameters
    # Enable persistAuthorization and tryItOutEnabled for better user experience
    app_kwargs["swagger_ui_parameters"] = {
        "persistAuthorization": True,
        "tryItOutEnabled": True,
    }

    app = FastAPI(**app_kwargs)

    # Add custom validation error handler for /query/data endpoint
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        # Check if this is a request to /query/data endpoint
        if request.url.path.endswith("/query/data"):
            # Extract error details
            error_details = []
            for error in exc.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                error_details.append(f"{field_path}: {error['msg']}")

            error_message = "; ".join(error_details)

            # Return in the expected format for /query/data
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failure",
                    "message": f"Validation error: {error_message}",
                    "data": {},
                    "metadata": {},
                },
            )
        else:
            # For other endpoints, return the default FastAPI validation error
            return JSONResponse(status_code=422, content={"detail": exc.errors()})

    def get_cors_origins():
        """Get allowed origins from global_args
        Returns a list of allowed origins, defaults to ["*"] if not set
        """
        origins_str = global_args.cors_origins
        if origins_str == "*":
            return ["*"]
        return [origin.strip() for origin in origins_str.split(",")]

    # Normalize scope["path"] for proxy-strip deployments so the WebUI
    # Mount (and any other Mount) routes correctly. Added before CORS so it
    # runs first in the middleware stack — see _RootPathNormalizationMiddleware
    # docstring.
    if api_prefix:
        app.add_middleware(_RootPathNormalizationMiddleware)

    # Add CORS middleware
    cors_origins = get_cors_origins()
    # Per the Fetch spec, the wildcard origin "*" and credentialed requests are
    # mutually exclusive: a server must not pair "Access-Control-Allow-Origin: *"
    # with "Access-Control-Allow-Credentials: true". LightRAG authenticates via
    # the Authorization (Bearer) and X-API-Key request headers, never via cookies
    # or other ambient credentials, so credentials are only ever meaningful for an
    # explicit origin allowlist. When origins are wildcarded we therefore disable
    # credentials to keep the configuration spec-compliant and avoid the permissive
    # "reflect any origin with credentials" behavior that Starlette would otherwise
    # apply to cookie-bearing cross-origin requests.
    allow_credentials = cors_origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-New-Token"
        ],  # Expose token renewal header for cross-origin requests
    )

    # Create combined auth dependency for all endpoints
    combined_auth = get_combined_auth_dependency(api_key)

    def get_workspace_from_request(request: Request) -> str | None:
        """
        Extract workspace from HTTP request header or use default.

        This enables multi-workspace API support by checking the custom
        'LIGHTRAG-WORKSPACE' header. If not present, falls back to the
        server's default workspace configuration.

        Args:
            request: FastAPI Request object

        Returns:
            Workspace identifier (may be empty string for global namespace)
        """
        # Check custom header first
        workspace = request.headers.get("LIGHTRAG-WORKSPACE", "").strip()

        if not workspace:
            workspace = None
        else:
            sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", workspace)
            if sanitized != workspace:
                logger.warning(
                    f"Workspace header '{workspace}' contains invalid characters. "
                    f"Sanitized to '{sanitized}'."
                )
                workspace = sanitized

        return workspace

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)

    def create_optimized_openai_llm_func(
        config_cache: LLMConfigCache, args, llm_timeout: int
    ):
        """Create optimized OpenAI LLM function with pre-processed configuration"""

        async def optimized_openai_alike_model_complete(
            prompt,
            system_prompt=None,
            history_messages=None,
            **kwargs,
        ) -> str:
            from lightrag.llm.openai import openai_complete_if_cache

            if history_messages is None:
                history_messages = []

            # Use pre-processed configuration to avoid repeated parsing.
            # response_format and legacy keyword_extraction/entity_extraction
            # flags flow through **kwargs; openai_complete_if_cache handles
            # the deprecation shim for the legacy booleans.
            kwargs["timeout"] = llm_timeout
            if config_cache.openai_llm_options:
                kwargs.update(config_cache.openai_llm_options)

            return await openai_complete_if_cache(
                args.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=args.llm_binding_host,
                api_key=args.llm_binding_api_key,
                **kwargs,
            )

        return optimized_openai_alike_model_complete

    def create_optimized_azure_openai_llm_func(
        config_cache: LLMConfigCache, args, llm_timeout: int
    ):
        """Create optimized Azure OpenAI LLM function with pre-processed configuration"""

        async def optimized_azure_openai_model_complete(
            prompt,
            system_prompt=None,
            history_messages=None,
            **kwargs,
        ) -> str:
            from lightrag.llm.azure_openai import azure_openai_complete_if_cache

            if history_messages is None:
                history_messages = []

            # response_format and legacy extraction booleans flow through kwargs
            # to azure_openai_complete_if_cache, which handles deprecation shims.
            kwargs["timeout"] = llm_timeout
            if config_cache.openai_llm_options:
                kwargs.update(config_cache.openai_llm_options)

            return await azure_openai_complete_if_cache(
                args.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=args.llm_binding_host,
                api_key=os.getenv("AZURE_OPENAI_API_KEY", args.llm_binding_api_key),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                **kwargs,
            )

        return optimized_azure_openai_model_complete

    def create_optimized_gemini_llm_func(
        config_cache: LLMConfigCache, args, llm_timeout: int
    ):
        """Create optimized Gemini LLM function with cached configuration"""

        async def optimized_gemini_model_complete(
            prompt,
            system_prompt=None,
            history_messages=None,
            **kwargs,
        ) -> str:
            from lightrag.llm.gemini import gemini_complete_if_cache

            if history_messages is None:
                history_messages = []

            # response_format and legacy extraction booleans flow through kwargs
            # to gemini_complete_if_cache, which handles deprecation shims.
            kwargs["timeout"] = llm_timeout
            if (
                config_cache.gemini_llm_options is not None
                and "generation_config" not in kwargs
            ):
                kwargs["generation_config"] = dict(config_cache.gemini_llm_options)

            return await gemini_complete_if_cache(
                args.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=args.llm_binding_api_key,
                base_url=args.llm_binding_host,
                **kwargs,
            )

        return optimized_gemini_model_complete

    def create_llm_model_func(binding: str):
        """
        Create LLM model function based on binding type.
        Uses optimized functions for OpenAI bindings and lazy import for others.
        """
        try:
            if binding == "lollms":
                from lightrag.llm.lollms import lollms_model_complete

                return lollms_model_complete
            elif binding == "ollama":
                from lightrag.llm.ollama import ollama_model_complete

                return ollama_model_complete
            elif binding == "bedrock":
                return bedrock_model_complete  # Already defined locally
            elif binding == "azure_openai":
                # Use optimized function with pre-processed configuration
                return create_optimized_azure_openai_llm_func(
                    config_cache, args, llm_timeout
                )
            elif binding == "gemini":
                return create_optimized_gemini_llm_func(config_cache, args, llm_timeout)
            else:  # openai and compatible
                # Use optimized function with pre-processed configuration
                return create_optimized_openai_llm_func(config_cache, args, llm_timeout)
        except ImportError as e:
            raise Exception(f"Failed to import {binding} LLM binding: {e}")

    def create_llm_model_kwargs(binding: str, args, llm_timeout: int) -> dict:
        """
        Create LLM model kwargs based on binding type.
        Uses lazy import for binding-specific options.
        """
        if binding in ["lollms", "ollama"]:
            try:
                from lightrag.llm.binding_options import OllamaLLMOptions

                return {
                    "host": args.llm_binding_host,
                    "timeout": llm_timeout,
                    "options": OllamaLLMOptions.options_dict(args),
                    "api_key": args.llm_binding_api_key,
                }
            except ImportError as e:
                raise Exception(f"Failed to import {binding} options: {e}")
        return {}

    def resolve_role_llm_settings(
        role: str, override_meta: dict | None = None
    ) -> dict[str, Any]:
        attr = role.lower()
        override_meta = override_meta or {}

        role_binding = (
            override_meta.get("binding")
            or getattr(args, f"{attr}_llm_binding", None)
            or args.llm_binding
        )
        role_model = (
            override_meta.get("model")
            or getattr(args, f"{attr}_llm_model", None)
            or args.llm_model
        )
        role_host = (
            override_meta.get("host")
            or getattr(args, f"{attr}_llm_binding_host", None)
            or args.llm_binding_host
        )
        explicit_role_apikey = override_meta.get("api_key") or getattr(
            args, f"{attr}_llm_binding_api_key", None
        )
        if role_binding == "bedrock":
            if explicit_role_apikey:
                raise ValueError(
                    f"Bedrock role '{role}' does not support role-specific "
                    "LLM_BINDING_API_KEY; use role-specific SigV4 AWS_* "
                    "variables or process-level AWS_BEARER_TOKEN_BEDROCK."
                )
            role_apikey = None
        else:
            role_apikey = explicit_role_apikey or args.llm_binding_api_key
        role_timeout = (
            override_meta.get("timeout")
            or getattr(args, f"{attr}_llm_timeout", None)
            or llm_timeout
        )
        role_max_async = override_meta.get("max_async")
        if role_max_async is None:
            role_max_async = getattr(args, f"{attr}_llm_max_async", None)
        is_cross_provider = role_binding != args.llm_binding

        role_provider_options = override_meta.get("provider_options")
        if role_provider_options is None:
            if role_binding in ["openai", "azure_openai"]:
                from lightrag.llm.binding_options import OpenAILLMOptions

                role_provider_options = OpenAILLMOptions.options_dict_for_role(
                    args, role, is_cross_provider
                )
            elif role_binding == "gemini":
                from lightrag.llm.binding_options import GeminiLLMOptions

                role_provider_options = GeminiLLMOptions.options_dict_for_role(
                    args, role, is_cross_provider
                )
            elif role_binding in ["lollms", "ollama"]:
                from lightrag.llm.binding_options import OllamaLLMOptions

                role_provider_options = OllamaLLMOptions.options_dict_for_role(
                    args, role, is_cross_provider
                )
            elif role_binding == "bedrock":
                from lightrag.llm.binding_options import BedrockLLMOptions

                role_provider_options = BedrockLLMOptions.options_dict_for_role(
                    args, role, is_cross_provider
                )
            else:
                role_provider_options = {}

        bedrock_aws_options = {}
        if role_binding == "bedrock":
            override_bedrock_aws_options = override_meta.get("bedrock_aws_options", {})
            bedrock_aws_options = {
                "aws_region": override_meta.get("aws_region")
                or override_bedrock_aws_options.get("aws_region")
                or getattr(args, f"{attr}_aws_region", None)
                or getattr(args, "aws_region", None),
                "aws_access_key_id": override_meta.get("aws_access_key_id")
                or override_bedrock_aws_options.get("aws_access_key_id")
                or getattr(args, f"{attr}_aws_access_key_id", None)
                or getattr(args, "aws_access_key_id", None),
                "aws_secret_access_key": override_meta.get("aws_secret_access_key")
                or override_bedrock_aws_options.get("aws_secret_access_key")
                or getattr(args, f"{attr}_aws_secret_access_key", None)
                or getattr(args, "aws_secret_access_key", None),
                "aws_session_token": override_meta.get("aws_session_token")
                or override_bedrock_aws_options.get("aws_session_token")
                or getattr(args, f"{attr}_aws_session_token", None)
                or getattr(args, "aws_session_token", None),
            }

        return {
            "binding": role_binding,
            "model": role_model,
            "host": role_host,
            "api_key": role_apikey,
            "timeout": role_timeout,
            "max_async": role_max_async,
            "provider_options": role_provider_options,
            "is_cross_provider": is_cross_provider,
            "bedrock_aws_options": bedrock_aws_options,
        }

    def create_role_llm_func(role: str, override_meta: dict | None = None):
        """Create an independent raw LLM function for a role."""
        settings = resolve_role_llm_settings(role, override_meta)
        role_binding = settings["binding"]
        role_model = settings["model"]
        role_host = settings["host"]
        role_apikey = settings["api_key"]
        role_timeout = settings["timeout"]
        role_provider_options = settings["provider_options"]
        bedrock_aws_options = settings["bedrock_aws_options"]

        try:
            if role_binding == "ollama":
                from lightrag.llm.ollama import _ollama_model_if_cache

                async def role_ollama_complete(
                    prompt,
                    system_prompt=None,
                    history_messages=None,
                    enable_cot: bool = False,
                    **kwargs,
                ):
                    # response_format and legacy extraction booleans flow
                    # through kwargs to _ollama_model_if_cache, which handles
                    # the deprecation shim and emits a single warning.
                    if history_messages is None:
                        history_messages = []
                    if role_provider_options:
                        kwargs.setdefault("options", dict(role_provider_options))
                    return await _ollama_model_if_cache(
                        role_model,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        enable_cot=enable_cot,
                        host=role_host,
                        timeout=role_timeout,
                        api_key=role_apikey,
                        **kwargs,
                    )

                return role_ollama_complete
            if role_binding == "lollms":
                from lightrag.llm.lollms import lollms_model_if_cache

                async def role_lollms_complete(
                    prompt,
                    system_prompt=None,
                    history_messages=None,
                    enable_cot: bool = False,
                    **kwargs,
                ):
                    # response_format and legacy extraction booleans flow
                    # through kwargs to lollms_model_if_cache, which drops
                    # them and emits deprecation warnings when booleans are set.
                    if history_messages is None:
                        history_messages = []
                    if role_provider_options:
                        kwargs = {**role_provider_options, **kwargs}
                    return await lollms_model_if_cache(
                        role_model,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        enable_cot=enable_cot,
                        base_url=role_host,
                        api_key=role_apikey,
                        timeout=role_timeout,
                        **kwargs,
                    )

                return role_lollms_complete
            if role_binding == "bedrock":
                from lightrag.llm.bedrock import bedrock_complete_if_cache

                async def role_bedrock_complete(
                    prompt,
                    system_prompt=None,
                    history_messages=None,
                    **kwargs,
                ) -> str:
                    if history_messages is None:
                        history_messages = []
                    if role_provider_options:
                        kwargs = {**role_provider_options, **kwargs}
                    return await bedrock_complete_if_cache(
                        role_model,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        endpoint_url=role_host,
                        **bedrock_aws_options,
                        **kwargs,
                    )

                return role_bedrock_complete
            if role_binding == "azure_openai":
                from lightrag.llm.azure_openai import azure_openai_complete_if_cache

                async def role_azure_openai_complete(
                    prompt,
                    system_prompt=None,
                    history_messages=None,
                    **kwargs,
                ) -> str:
                    if history_messages is None:
                        history_messages = []
                    kwargs["timeout"] = role_timeout
                    if role_provider_options:
                        kwargs.update(role_provider_options)
                    return await azure_openai_complete_if_cache(
                        role_model,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        base_url=role_host,
                        api_key=role_apikey or os.getenv("AZURE_OPENAI_API_KEY"),
                        api_version=os.getenv(
                            "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
                        ),
                        **kwargs,
                    )

                return role_azure_openai_complete
            if role_binding == "gemini":
                from lightrag.llm.gemini import gemini_complete_if_cache

                async def role_gemini_complete(
                    prompt,
                    system_prompt=None,
                    history_messages=None,
                    **kwargs,
                ) -> str:
                    if history_messages is None:
                        history_messages = []
                    kwargs["timeout"] = role_timeout
                    if role_provider_options and "generation_config" not in kwargs:
                        kwargs["generation_config"] = dict(role_provider_options)
                    return await gemini_complete_if_cache(
                        role_model,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        api_key=role_apikey,
                        base_url=role_host,
                        **kwargs,
                    )

                return role_gemini_complete

            from lightrag.llm.openai import openai_complete_if_cache

            async def role_openai_complete(
                prompt,
                system_prompt=None,
                history_messages=None,
                **kwargs,
            ) -> str:
                if history_messages is None:
                    history_messages = []
                kwargs["timeout"] = role_timeout
                if role_provider_options:
                    kwargs.update(role_provider_options)
                return await openai_complete_if_cache(
                    role_model,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    base_url=role_host,
                    api_key=role_apikey,
                    **kwargs,
                )

            return role_openai_complete
        except ImportError as e:
            raise Exception(f"Failed to create LLM for role '{role}': {e}")

    def create_role_llm_model_kwargs(
        role: str, override_meta: dict | None = None
    ) -> dict[str, Any] | None:
        """Create role-specific kwargs for runtime wrapper injection.

        Role functions built above already encapsulate provider host/model/api_key/options,
        so we intentionally return an empty dict here to prevent base kwargs inheritance
        from polluting cross-provider role calls.
        """
        _ = role
        _ = override_meta
        return {}

    llm_timeout = args.llm_timeout
    embedding_timeout = args.embedding_timeout

    async def bedrock_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        **kwargs,
    ) -> str:
        # Lazy import
        from lightrag.llm.bedrock import bedrock_complete_if_cache

        if history_messages is None:
            history_messages = []

        # Bedrock Converse API has no JSON mode; response_format and the legacy
        # extraction booleans flow through kwargs to bedrock_complete_if_cache,
        # which drops them and emits deprecation warnings when booleans are set.
        if config_cache.bedrock_llm_options:
            kwargs = {**config_cache.bedrock_llm_options, **kwargs}

        return await bedrock_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            endpoint_url=args.llm_binding_host,
            aws_region=getattr(args, "aws_region", None),
            aws_access_key_id=getattr(args, "aws_access_key_id", None),
            aws_secret_access_key=getattr(args, "aws_secret_access_key", None),
            aws_session_token=getattr(args, "aws_session_token", None),
            **kwargs,
        )

    # Create embedding function with optimized configuration and max_token_size inheritance
    import inspect

    embedding_func = create_embedding_function_from_args(args, config_cache)

    # Configure rerank function based on args.rerank_bindingparameter
    rerank_model_func = None
    if args.rerank_binding != "null":
        from lightrag.rerank import cohere_rerank, jina_rerank, ali_rerank

        # Map rerank binding to corresponding function
        rerank_functions = {
            "cohere": cohere_rerank,
            "jina": jina_rerank,
            "aliyun": ali_rerank,
        }

        # Select the appropriate rerank function based on binding
        selected_rerank_func = rerank_functions.get(args.rerank_binding)
        if not selected_rerank_func:
            logger.error(f"Unsupported rerank binding: {args.rerank_binding}")
            raise ValueError(f"Unsupported rerank binding: {args.rerank_binding}")

        # Get default values from selected_rerank_func if args values are None
        if args.rerank_model is None or args.rerank_binding_host is None:
            sig = inspect.signature(selected_rerank_func)

            # Set default model if args.rerank_model is None
            if args.rerank_model is None and "model" in sig.parameters:
                default_model = sig.parameters["model"].default
                if default_model != inspect.Parameter.empty:
                    args.rerank_model = default_model

            # Set default base_url if args.rerank_binding_host is None
            if args.rerank_binding_host is None and "base_url" in sig.parameters:
                default_base_url = sig.parameters["base_url"].default
                if default_base_url != inspect.Parameter.empty:
                    args.rerank_binding_host = default_base_url

        async def server_rerank_func(
            query: str, documents: list, top_n: int = None, extra_body: dict = None
        ):
            """Server rerank function with configuration from environment variables"""
            # Prepare kwargs for rerank function
            kwargs = {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "api_key": args.rerank_binding_api_key,
                "model": args.rerank_model,
                "base_url": args.rerank_binding_host,
            }

            # Add Cohere-specific parameters if using cohere binding
            if args.rerank_binding == "cohere":
                # Enable chunking if configured (useful for models with token limits like ColBERT)
                kwargs["enable_chunking"] = (
                    os.getenv("RERANK_ENABLE_CHUNKING", "false").lower() == "true"
                )
                kwargs["max_tokens_per_doc"] = int(
                    os.getenv("RERANK_MAX_TOKENS_PER_DOC", "4096")
                )

            return await selected_rerank_func(**kwargs, extra_body=extra_body)

        rerank_model_func = server_rerank_func
        logger.info(
            f"Reranking is enabled: {args.rerank_model or 'default model'} using {args.rerank_binding} provider"
        )
    else:
        logger.info("Reranking is disabled")

    # Create ollama_server_infos from command line arguments
    from lightrag.api.config import OllamaServerInfos

    ollama_server_infos = OllamaServerInfos(
        name=args.simulated_model_name, tag=args.simulated_model_tag
    )

    # LightRAG.__post_init__ normalizes addon_params and backfills env-based defaults
    # (SUMMARY_LANGUAGE, ENTITY_TYPE_PROMPT_FILE, ...), so we only need to pass the
    # API-level overrides here.
    addon_params = {
        "language": args.summary_language,
    }

    role_llm_configs = {
        spec.name: {
            **resolve_role_llm_settings(spec.name),
            "func": create_role_llm_func(spec.name),
            "kwargs": create_role_llm_model_kwargs(spec.name),
        }
        for spec in ROLES
    }

    # Initialize RAG with unified configuration
    try:
        rag = LightRAG(
            working_dir=args.working_dir,
            workspace=args.workspace,
            llm_model_func=create_llm_model_func(args.llm_binding),
            llm_model_name=args.llm_model,
            llm_model_max_async=args.max_async,
            summary_max_tokens=args.summary_max_tokens,
            summary_context_size=args.summary_context_size,
            chunk_token_size=int(args.chunk_size),
            chunk_overlap_token_size=int(args.chunk_overlap_size),
            llm_model_kwargs=create_llm_model_kwargs(
                args.llm_binding, args, llm_timeout
            ),
            embedding_func=embedding_func,
            default_llm_timeout=llm_timeout,
            default_embedding_timeout=embedding_timeout,
            kv_storage=args.kv_storage,
            graph_storage=args.graph_storage,
            vector_storage=args.vector_storage,
            doc_status_storage=args.doc_status_storage,
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": args.cosine_threshold
            },
            enable_llm_cache_for_entity_extract=args.enable_llm_cache_for_extract,
            enable_llm_cache=args.enable_llm_cache,
            vlm_process_enable=args.vlm_process_enable,
            rerank_model_func=rerank_model_func,
            rerank_model_max_async=args.rerank_max_async,
            default_rerank_timeout=args.rerank_timeout,
            max_parallel_insert=args.max_parallel_insert,
            max_graph_nodes=args.max_graph_nodes,
            addon_params=addon_params,
            ollama_server_infos=ollama_server_infos,
            role_llm_configs={
                spec.name: RoleLLMConfig(
                    func=role_llm_configs[spec.name]["func"],
                    kwargs=role_llm_configs[spec.name]["kwargs"],
                    max_async=role_llm_configs[spec.name]["max_async"],
                    timeout=role_llm_configs[spec.name]["timeout"],
                    metadata={
                        "base_binding": args.llm_binding,
                        "binding": role_llm_configs[spec.name]["binding"],
                        "model": role_llm_configs[spec.name]["model"],
                        "host": role_llm_configs[spec.name]["host"],
                        "api_key": role_llm_configs[spec.name]["api_key"],
                        "provider_options": role_llm_configs[spec.name][
                            "provider_options"
                        ],
                        "bedrock_aws_options": role_llm_configs[spec.name][
                            "bedrock_aws_options"
                        ],
                        "is_cross_provider": role_llm_configs[spec.name][
                            "is_cross_provider"
                        ],
                    },
                )
                for spec in ROLES
            },
        )
    except Exception as e:
        logger.error(f"Failed to initialize LightRAG: {e}")
        raise

    _log_role_provider_options(rag)

    rag.register_role_llm_builder(
        lambda role, meta: (
            create_role_llm_func(role, meta),
            create_role_llm_model_kwargs(role, meta),
        )
    )

    # Add routes
    # root_path is set on the app for reverse proxy support;
    # routes stay at their natural paths and are prefixed by the proxy or uvicorn --root-path
    app.include_router(create_document_routes(rag, doc_manager, api_key))
    app.include_router(create_query_routes(rag, api_key, args.top_k))
    app.include_router(create_graph_routes(rag, api_key))

    # Add Ollama API routes
    ollama_api = OllamaAPI(rag, top_k=args.top_k, api_key=api_key)
    app.include_router(ollama_api.router, prefix="/api")

    # Custom Swagger UI endpoint for offline support
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html(request: Request):
        """Custom Swagger UI HTML with local static files"""
        response = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url="/docs/oauth2-redirect",
            swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui/swagger-ui.css",
            swagger_favicon_url="/static/swagger-ui/favicon-32x32.png",
            swagger_ui_parameters=app.swagger_ui_parameters,
        )
        html = response.body.decode("utf-8")
        html = _inject_swagger_theme(
            html, request.query_params.get("theme", "auto").lower()
        )
        return HTMLResponse(content=html)

    @app.get("/docs/oauth2-redirect", include_in_schema=False)
    async def swagger_ui_redirect():
        """OAuth2 redirect for Swagger UI"""
        return get_swagger_ui_oauth2_redirect_html()

    @app.get("/")
    async def redirect_to_webui(request: Request):
        """Redirect root path based on WebUI availability.

        Prepend the ASGI root_path so that, behind a reverse proxy, the
        absolute redirect target keeps the configured prefix instead of
        bypassing it.
        """
        root = request.scope.get("root_path", "")
        if webui_assets_exist:
            return RedirectResponse(url=f"{root}{webui_path}/")
        else:
            return RedirectResponse(url=f"{root}/docs")

    @app.get("/auth-status")
    async def get_auth_status():
        """Get authentication status and guest token if auth is not configured"""

        if not auth_handler.accounts:
            # Authentication not configured, return guest token
            guest_token = auth_handler.create_token(
                username="guest", role="guest", metadata={"auth_mode": "disabled"}
            )
            return {
                "auth_configured": False,
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }

        return {
            "auth_configured": True,
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": api_version_display,
            "webui_title": webui_title,
            "webui_description": webui_description,
        }

    @app.post("/login")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        if not auth_handler.accounts:
            # Authentication not configured, return guest token
            guest_token = auth_handler.create_token(
                username="guest", role="guest", metadata={"auth_mode": "disabled"}
            )
            return {
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }
        username = form_data.username
        if not auth_handler.verify_password(username, form_data.password):
            raise HTTPException(status_code=401, detail="Incorrect credentials")

        # Regular user login
        user_token = auth_handler.create_token(
            username=username, role="user", metadata={"auth_mode": "enabled"}
        )
        return {
            "access_token": user_token,
            "token_type": "bearer",
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": api_version_display,
            "webui_title": webui_title,
            "webui_description": webui_description,
        }

    @app.get(
        "/health",
        dependencies=[Depends(combined_auth)],
        summary="Get system health and configuration status",
        description="Returns comprehensive system status including WebUI availability, configuration, and operational metrics",
        response_description="System health status with configuration details",
        responses={
            200: {
                "description": "Successful response with system status",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "healthy",
                            "webui_available": True,
                            "working_directory": "/path/to/working/dir",
                            "input_directory": "/path/to/input/dir",
                            "configuration": {
                                "llm_binding": "openai",
                                "llm_model": "gpt-4",
                                "embedding_binding": "openai",
                                "embedding_model": "text-embedding-ada-002",
                                "workspace": "default",
                                "storage_workspaces": {
                                    "kv_storage": "default",
                                    "doc_status_storage": "default",
                                    "graph_storage": "default",
                                    "vector_storage": "default",
                                },
                                "parser_routing": "pdf:mineru",
                                "mineru": {
                                    "endpoint": "http://localhost:8080",
                                    "api_mode": "local",
                                    "options": {
                                        "language": "ch",
                                        "enable_table": True,
                                        "enable_formula": True,
                                        "local_backend": "pipeline",
                                        "local_parse_method": "auto",
                                        "local_image_analysis": False,
                                    },
                                },
                                "docling": {
                                    "endpoint": "",
                                    "options": {},
                                },
                            },
                            "auth_mode": "enabled",
                            "pipeline_busy": False,
                            "core_version": "0.0.1",
                            "api_version": "0.0.1",
                        }
                    }
                },
            }
        },
    )
    async def get_status(request: Request):
        """Get current system status including WebUI availability"""
        try:
            workspace = get_workspace_from_request(request)
            default_workspace = get_default_workspace()
            if workspace is None:
                workspace = default_workspace
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=workspace
            )

            pipeline_busy = bool(pipeline_status.get("busy", False))
            pipeline_scanning = bool(pipeline_status.get("scanning", False))
            pipeline_destructive_busy = bool(
                pipeline_status.get("destructive_busy", False)
            )
            pipeline_pending_enqueues = int(
                pipeline_status.get("pending_enqueues", 0) or 0
            )
            pipeline_active = (
                pipeline_busy
                or pipeline_scanning
                or pipeline_destructive_busy
                or pipeline_pending_enqueues > 0
            )

            if not auth_configured:
                auth_mode = "disabled"
            else:
                auth_mode = "enabled"

            # Cleanup expired keyed locks and get status
            keyed_lock_info = cleanup_keyed_lock()

            return {
                "status": "healthy",
                "webui_available": webui_assets_exist,
                "working_directory": str(args.working_dir),
                "input_directory": str(args.input_dir),
                "configuration": {
                    # LLM configuration binding/host address (if applicable)/model (if applicable)
                    "llm_binding": args.llm_binding,
                    "llm_binding_host": args.llm_binding_host,
                    "llm_model": args.llm_model,
                    # embedding model configuration binding/host address (if applicable)/model (if applicable)
                    "embedding_binding": args.embedding_binding,
                    "embedding_binding_host": args.embedding_binding_host,
                    "embedding_model": args.embedding_model,
                    "summary_max_tokens": args.summary_max_tokens,
                    "summary_context_size": args.summary_context_size,
                    "kv_storage": args.kv_storage,
                    "doc_status_storage": args.doc_status_storage,
                    "graph_storage": args.graph_storage,
                    "vector_storage": args.vector_storage,
                    "enable_llm_cache_for_extract": args.enable_llm_cache_for_extract,
                    "enable_llm_cache": args.enable_llm_cache,
                    "vlm_process_enable": args.vlm_process_enable,
                    "workspace": default_workspace,
                    "storage_workspaces": _get_storage_workspaces(rag),
                    "max_graph_nodes": args.max_graph_nodes,
                    # Rerank configuration
                    "enable_rerank": rerank_model_func is not None,
                    "rerank_binding": args.rerank_binding,
                    "rerank_model": args.rerank_model if rerank_model_func else None,
                    "rerank_binding_host": args.rerank_binding_host
                    if rerank_model_func
                    else None,
                    "rerank_max_async": args.rerank_max_async,
                    "rerank_timeout": args.rerank_timeout,
                    # Environment variable status (requested configuration)
                    "summary_language": args.summary_language,
                    "force_llm_summary_on_merge": args.force_llm_summary_on_merge,
                    "max_parallel_insert": args.max_parallel_insert,
                    "cosine_threshold": args.cosine_threshold,
                    "min_rerank_score": args.min_rerank_score,
                    "related_chunk_number": args.related_chunk_number,
                    "max_async": args.max_async,
                    "llm_timeout": args.llm_timeout,
                    "embedding_func_max_async": args.embedding_func_max_async,
                    "embedding_batch_num": args.embedding_batch_num,
                    "embedding_timeout": args.embedding_timeout,
                    "role_llm_config": rag.get_llm_role_config(),
                    # Parser routing snapshot — surfaced in the WebUI status card
                    "parser_routing": parser_rules_from_env(),
                    "mineru": _build_mineru_status(),
                    "docling": _build_docling_status(),
                },
                "auth_mode": auth_mode,
                "server_mode": "gunicorn"
                if os.environ.get("LIGHTRAG_GUNICORN_MODE")
                else "uvicorn",
                "workers": getattr(args, "workers", 1),
                "pipeline_busy": pipeline_busy,
                "pipeline_active": pipeline_active,
                "pipeline_scanning": pipeline_scanning,
                "pipeline_destructive_busy": pipeline_destructive_busy,
                "pipeline_pending_enqueues": pipeline_pending_enqueues,
                "keyed_locks": keyed_lock_info,
                "llm_queue_status": await rag.get_llm_queue_status(include_base=True),
                "embedding_queue_status": await rag.get_embedding_queue_status(),
                "rerank_queue_status": await rag.get_rerank_queue_status(),
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    # Pre-render the runtime-config <script> once. The browser-visible URL
    # prefixes are NOT baked into the bundle anymore — index.html ships with
    # a placeholder comment that we replace with this snippet on every HTML
    # response, so one build serves any reverse-proxy mount point.
    #
    # `</` → `<\/` escaping prevents an embedded "</script>" sequence from
    # breaking out of the inline script (defense-in-depth — values come from
    # admin config, not user input).
    _runtime_config_payload = json.dumps(
        {
            "apiPrefix": api_prefix,
            "webuiPrefix": f"{api_prefix}{webui_path}/",
        }
    ).replace("</", "<\\/")
    runtime_config_script = (
        f"<script>window.__LIGHTRAG_CONFIG__ = {_runtime_config_payload};</script>"
    )

    # Custom StaticFiles class for smart caching + runtime config injection
    class SmartStaticFiles(StaticFiles):  # Renamed from NoCacheStaticFiles
        # Replaced in index.html on every request. Keep in sync with
        # lightrag_webui/index.html.
        RUNTIME_CONFIG_PLACEHOLDER = b"<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->"

        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)

            # `path` is empty when accessing the mount root (StaticFiles
            # rewrites it to index.html internally) — match on media_type
            # too so we still inject in that case.
            is_html = (
                path.endswith(".html")
                or path == ""
                or path.endswith("/")
                or getattr(response, "media_type", None) == "text/html"
            )

            if (
                is_html
                and getattr(response, "status_code", 0) == 200
                and isinstance(response, FileResponse)
            ):
                response = self._inject_runtime_config(response)

            if is_html:
                response.headers["Cache-Control"] = (
                    "no-cache, no-store, must-revalidate"
                )
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            elif (
                "/assets/" in path
            ):  # Assets (JS, CSS, images, fonts) generated by Vite with hash in filename
                response.headers["Cache-Control"] = (
                    "public, max-age=31536000, immutable"
                )
            # Add other rules here if needed for non-HTML, non-asset files

            # Ensure correct Content-Type
            if path.endswith(".js"):
                response.headers["Content-Type"] = "application/javascript"
            elif path.endswith(".css"):
                response.headers["Content-Type"] = "text/css"

            return response

        def _inject_runtime_config(self, response: FileResponse) -> Response:
            """Replace the runtime-config placeholder in index.html.

            Returns the original FileResponse if the placeholder is absent
            (older build, or a non-index HTML file) — avoids breaking
            previously-working bundles during upgrades.
            """
            try:
                content = Path(response.path).read_bytes()
            except OSError as e:
                logger.warning(
                    "Could not read %s for runtime config injection: %s",
                    response.path,
                    e,
                )
                return response

            if self.RUNTIME_CONFIG_PLACEHOLDER not in content:
                return response

            new_content = content.replace(
                self.RUNTIME_CONFIG_PLACEHOLDER,
                runtime_config_script.encode("utf-8"),
            )
            return Response(content=new_content, media_type="text/html")

    # Mount Swagger UI static files for offline support
    swagger_static_dir = Path(__file__).parent / "static" / "swagger-ui"
    if swagger_static_dir.exists():
        app.mount(
            "/static/swagger-ui",
            StaticFiles(directory=swagger_static_dir),
            name="swagger-ui-static",
        )

    # Conditionally mount WebUI only if assets exist
    if webui_assets_exist:
        static_dir = Path(__file__).parent / "webui"
        static_dir.mkdir(exist_ok=True)
        app.mount(
            webui_path,
            SmartStaticFiles(
                directory=static_dir, html=True, check_dir=True
            ),  # Use SmartStaticFiles
            name="webui",
        )
        logger.info(f"WebUI assets mounted at {webui_path}")
    else:
        logger.info("WebUI assets not available, WebUI route not mounted")

        # Add redirect for WebUI path when assets are not available
        @app.get(webui_path)
        @app.get(f"{webui_path}/")
        async def webui_redirect_to_docs(request: Request):
            """Redirect WebUI path to /docs when WebUI is not available."""
            root = request.scope.get("root_path", "")
            return RedirectResponse(url=f"{root}/docs")

    return app


def get_application(args=None):
    """Factory function for creating the FastAPI application"""
    if args is None:
        args = global_args
    return create_app(args)


def configure_logging():
    """Configure logging for uvicorn startup"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.filters = []

    # Get log directory path from environment variable
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, DEFAULT_LOG_FILENAME))

    print(f"\nLightRAG log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = get_env_value("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES, int)
    log_backup_count = get_env_value("LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT, int)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                # Configure all uvicorn related loggers
                "uvicorn": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["path_filter"],
                },
                "uvicorn.error": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["path_filter"],
                },
            },
            "filters": {
                "path_filter": {
                    "()": "lightrag.utils.LightragPathFilter",
                },
            },
        }
    )


def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        "uvicorn",
        "tiktoken",
        "fastapi",
        # Add other required packages here
    ]

    for package in required_packages:
        if not pm.is_installed(package):
            print(f"Installing {package}...")
            pm.install(package)
            print(f"{package} installed successfully")


def main():
    # On Windows, ProactorEventLoop (default since Python 3.8) has known
    # race conditions with uvicorn's socket binding that can cause the server
    # to report it's running while the port is never actually bound.
    # Using SelectorEventLoop resolves this issue.
    # See: https://github.com/HKUDS/LightRAG/issues/2438
    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Explicitly initialize configuration for clarity
    # (The proxy will auto-initialize anyway, but this makes intent clear)
    from .config import initialize_config

    initialize_config()

    # Check if running under Gunicorn
    if "GUNICORN_CMD_ARGS" in os.environ:
        # If started with Gunicorn, return directly as Gunicorn will call get_application
        print("Running under Gunicorn - worker management handled by Gunicorn")
        return

    # Check .env file
    if not check_env_file():
        sys.exit(1)

    # Check and install dependencies
    check_and_install_dependencies()

    from multiprocessing import freeze_support

    freeze_support()

    # Configure logging before parsing args
    configure_logging()
    update_uvicorn_mode_config()
    display_splash_screen(global_args)

    # Note: Signal handlers are NOT registered here because:
    # - Uvicorn has built-in signal handling that properly calls lifespan shutdown
    # - Custom signal handlers can interfere with uvicorn's graceful shutdown
    # - Cleanup is handled by the lifespan context manager's finally block

    # Create application instance directly instead of using factory function
    app = create_app(global_args)

    # Start Uvicorn in single process mode. Do not pass root_path here;
    # the prefix lives only on FastAPI's app.root_path. See
    # docs/MultiSiteDeployment.md.
    uvicorn_config = {
        "app": app,  # Pass application instance directly instead of string path
        "host": global_args.host,
        "port": global_args.port,
        "log_config": None,  # Disable default config
    }

    if global_args.ssl:
        uvicorn_config.update(
            {
                "ssl_certfile": global_args.ssl_certfile,
                "ssl_keyfile": global_args.ssl_keyfile,
            }
        )

    print(
        f"Starting Uvicorn server in single-process mode on {global_args.host}:{global_args.port}"
    )
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
