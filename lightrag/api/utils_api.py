"""
Utility functions for the LightRAG API.
"""

import os
import argparse
from collections import OrderedDict
from typing import Any, Mapping, Optional, List, Tuple
import sys
import time
import uuid
import logging
from ascii_colors import ASCIIColors
from .._version import __api_version__ as api_version
from .._version import __version__ as core_version
from lightrag.constants import (
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
)
from lightrag.api.runtime_validation import validate_runtime_target_from_env_file
from fastapi import HTTPException, Security, Request, Response, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from starlette.status import HTTP_403_FORBIDDEN
from ..utils import safe_log_value
from .auth import auth_handler
from .config import (
    ollama_server_infos,
    global_args,
    get_env_value,
    normalize_api_prefix,
)

logger = logging.getLogger("lightrag")

# Generic body returned to clients for any HTTP 500 so that raw exception text
# (which can carry DB hosts, credentials, filesystem paths, or SQL fragments) is
# never disclosed in the response — see CWE-209.
_INTERNAL_SERVER_ERROR_MESSAGE = "Internal server error"


def internal_server_error(exc: Exception) -> HTTPException:
    """Build a client-safe HTTP 500 that never exposes raw exception text (CWE-209).

    Callers are expected to have already logged the full exception (message and
    traceback) server-side. This mints a short correlation id, emits one more log
    line carrying it, and returns an ``HTTPException`` whose ``detail`` is only a
    generic message plus that id. An operator can join the client-visible id to
    the server log, while the response body never reveals internal infrastructure
    such as database hosts, credentials, filesystem paths, or SQL fragments.

    Usage::

        except Exception as e:
            logger.error(f"Error doing X: {e}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)
    """
    error_id = uuid.uuid4().hex[:12]
    logger.error(
        f"Returning HTTP 500 to client [error_id={error_id}] ({type(exc).__name__})"
    )
    return HTTPException(
        status_code=500,
        detail=f"{_INTERNAL_SERVER_ERROR_MESSAGE} (error_id: {error_id})",
    )


# ========== Token Renewal Rate Limiting ==========
# Cache to track last renewal time per user (username as key)
# Format: {username: last_renewal_timestamp}
#
# Bounded on purpose (CWE-770, GHSA-3wg5-5w54-3rfm). The key is the JWT "sub"
# claim, which is attacker-chosen in any profile running on the default
# guest-mode JWT secret, so an unbounded dict here is a memory-exhaustion
# primitive. Three independent bounds now apply: the claim itself is capped at
# MAX_TOKEN_SUBJECT_LENGTH by AuthHandler.validate_token, only authenticated
# requests reach _record_token_renewal (see _renew_token_if_needed), and the
# table is bounded below.
#
# Note the contrast with LoginRateLimiter, the sibling control on this same
# request path: it must never evict a live record, because evicting an active
# lockout would clear it. Here a "live" entry only suppresses an early renewal,
# so evicting one costs a single extra token mint and carries no security
# consequence. The cap can therefore be enforced unconditionally, which is what
# keeps the table genuinely bounded instead of merely capped-then-full.
_token_renewal_cache: "OrderedDict[str, float]" = OrderedDict()
_RENEWAL_MIN_INTERVAL = 60  # Minimum 60 seconds between renewals for same user
_MAX_TRACKED_RENEWALS = 10_000  # Hard ceiling on distinct tracked usernames

# ========== Token Renewal Path Exclusions ==========
# Paths that should NOT trigger token auto-renewal
# - /health: Health check endpoint, no login required
# - /documents/paginated: Client polls this frequently (5-30s), renewal not needed
# - /documents/pipeline_status: Client polls this very frequently (2s), renewal not needed
_TOKEN_RENEWAL_SKIP_PATHS = [
    "/health",
    "/documents/paginated",
    "/documents/pipeline_status",
]


def _record_token_renewal(username: str, renewed_at: float) -> None:
    """Record a renewal timestamp, keeping ``_token_renewal_cache`` bounded.

    Insertion order is maintained as time order, so the oldest entries sit at the
    head: entries older than ``_RENEWAL_MIN_INTERVAL`` can no longer influence any
    decision and are dropped, then ``_MAX_TRACKED_RENEWALS`` is enforced as a hard
    ceiling. Both passes are amortized O(1). See the declaration for why evicting
    a still-live entry is safe here.
    """
    # Purge dead entries from the head. A backwards clock step (time.time() is not
    # monotonic) makes the delta negative and simply stops the purge early; the
    # hard ceiling below is the backstop.
    while _token_renewal_cache:
        oldest_username, oldest_at = next(iter(_token_renewal_cache.items()))
        if renewed_at - oldest_at < _RENEWAL_MIN_INTERVAL:
            break
        del _token_renewal_cache[oldest_username]

    # Re-insert at the tail so insertion order stays time order even when a
    # username is refreshed.
    _token_renewal_cache.pop(username, None)
    _token_renewal_cache[username] = renewed_at

    while len(_token_renewal_cache) > _MAX_TRACKED_RENEWALS:
        _token_renewal_cache.popitem(last=False)


def _renew_token_if_needed(path: str, response: Response, token_info: dict) -> None:
    """Attach a refreshed token via the ``X-New-Token`` header when near expiry.

    Callers MUST invoke this only after the request has actually authenticated.
    This bookkeeping writes process-wide server-side state keyed on the JWT "sub"
    claim; running it on a request that is about to be rejected let an
    unauthenticated caller grow ``_token_renewal_cache`` and forge log lines on
    every 403 (GHSA-3wg5-5w54-3rfm). Keeping the call at the authenticated exits
    of ``combined_dependency`` -- rather than next to token validation -- is what
    makes that unreachable, so do not hoist it back up.
    """
    from lightrag.api.config import global_args
    from datetime import datetime, timezone

    if not global_args.token_auto_renew:
        return

    # Check if current path should skip token renewal
    skip_renewal = any(
        path == skip_path or path.startswith(skip_path + "/")
        for skip_path in _TOKEN_RENEWAL_SKIP_PATHS
    )
    if skip_renewal:
        logger.debug(f"Token auto-renewal skipped for path: {safe_log_value(path)}")
        return

    try:
        expire_time = token_info.get("exp")
        if not expire_time:
            return

        # Calculate remaining time ratio
        now = datetime.now(timezone.utc)
        remaining_seconds = (expire_time - now).total_seconds()

        # Get original token expiration duration
        role = token_info.get("role", "user")
        total_hours = (
            auth_handler.guest_expire_hours
            if role == "guest"
            else auth_handler.expire_hours
        )
        total_seconds = total_hours * 3600

        # Issue new token if remaining time < threshold
        if remaining_seconds >= total_seconds * global_args.token_renew_threshold:
            return

        # ========== Rate Limiting Check ==========
        username = token_info["username"]
        current_time = time.time()
        last_renewal = _token_renewal_cache.get(username, 0)
        time_since_last_renewal = current_time - last_renewal

        # Only renew if enough time has passed since last renewal
        if time_since_last_renewal < _RENEWAL_MIN_INTERVAL:
            logger.debug(
                f"Token renewal skipped for {safe_log_value(username)} "
                f"(rate limit: last renewal {time_since_last_renewal:.0f}s ago)"
            )
            return

        new_token = auth_handler.create_token(
            username=username,
            role=role,
            metadata=token_info.get("metadata", {}),
        )
        # Return new token via response header
        response.headers["X-New-Token"] = new_token

        # Update renewal cache
        _record_token_renewal(username, current_time)

        # Optional: log renewal. The claim is bounded by validate_token but still
        # caller-supplied, so it goes through safe_log_value -- a raw CR/LF in it
        # forged whole log records (CWE-117).
        logger.info(
            f"Token auto-renewed for user {safe_log_value(username)} "
            f"(role: {safe_log_value(role)}, remaining: {remaining_seconds:.0f}s)"
        )
    except Exception as e:
        # Renewal failure should not affect normal request, just log
        logger.warning(f"Token auto-renew failed: {e}")


def check_env_file():
    """
    Check if .env file exists and handle user confirmation if needed.
    Returns True if should continue, False if should exit.
    """
    env_path = ".env"

    if not os.path.exists(env_path):
        warning_msg = "Warning: Startup directory must contain .env file for multi-instance support."
        ASCIIColors.yellow(warning_msg)

        # Check if running in interactive terminal
        if sys.stdin.isatty():
            response = input("Do you want to continue? (yes/NO): ")
            if response.lower() != "yes":
                ASCIIColors.red("Server startup cancelled")
                return False
        return True

    is_valid, error_message = validate_runtime_target_from_env_file(env_path)
    if not is_valid:
        for line in error_message.splitlines():
            ASCIIColors.red(line)
        return False

    return True


# Get whitelist paths from global_args, only once during initialization
whitelist_paths = global_args.whitelist_paths.split(",")

# Pre-compile path matching patterns
whitelist_patterns: List[Tuple[str, bool]] = []
for path in whitelist_paths:
    path = path.strip()
    if path:
        # If path ends with /*, match all paths with that prefix
        if path.endswith("/*"):
            prefix = path[:-2]
            whitelist_patterns.append((prefix, True))  # (prefix, is_prefix_match)
        else:
            whitelist_patterns.append((path, False))  # (exact_path, is_prefix_match)

# Global authentication configuration
auth_configured = bool(auth_handler.accounts)


def get_route_path(scope: Mapping[str, Any], mount_prefix: str = "") -> str:
    """The route path a matcher must compare against: ``scope["path"]`` minus the
    ASGI mount prefix.

    In canonical ASGI form ``scope["path"]`` *includes* ``root_path``, so a
    request routed to ``/documents`` on a server mounted at ``/site01`` carries
    ``path == "/site01/documents"``. Anything comparing against configured route
    paths (``WHITELIST_PATHS``, ``ADMISSION_PATHS``,
    ``_TOKEN_RENEWAL_SKIP_PATHS``) must subtract the prefix first, or it compares
    two different kinds of path.

    ``root_path`` is authoritative when present: ``FastAPI.__call__`` writes it
    from ``app.root_path`` before the middleware stack runs, so every layer sees
    it. ``mount_prefix`` is the fallback for callers whose scope was not built by
    FastAPI (hand-rolled ASGI scopes in tests, for instance).

    The subtraction is deliberately guarded rather than an unconditional slice,
    because the two callers do not see the same scope shape. The admission
    middleware runs *outside* ``_RootPathNormalizationMiddleware`` (that ordering
    is intentional, so CORS can wrap its refusals), so in a proxy-strip
    deployment it sees a bare ``/documents/upload`` while ``root_path`` is
    already ``/site01``. An unguarded ``path[len(prefix):]`` would silently
    mangle that into ``ments``.

    Mirrors ``starlette._utils.get_route_path`` — inlined rather than imported
    because that module is private — with one documented divergence: an
    exact-prefix hit returns ``"/"`` where Starlette returns ``""``. Configured
    paths are always written with a leading slash, so ``"/"`` is the normal form
    the matchers here expect.
    """
    path = scope.get("path") or ""
    prefix = (scope.get("root_path") or mount_prefix or "").rstrip("/")
    if not prefix or not path.startswith(prefix):
        return path
    remainder = path[len(prefix) :]
    if not remainder:
        return "/"
    if remainder.startswith("/"):
        return remainder
    # Mid-segment collision (prefix ``/api`` against path ``/apikeys``): the
    # prefix is not actually a path-segment prefix here, so nothing is removed.
    return path


def path_is_whitelisted(scope: Mapping[str, Any], *, mount_prefix: str = "") -> bool:
    """Whether ``WHITELIST_PATHS`` exempts this request from authentication.

    Shared so every layer that has to answer "may this request skip auth?" uses
    one matcher: the enforcing route dependency
    (:func:`get_combined_auth_dependency`) and the pure-ASGI admission
    middleware, which must pre-authenticate before reading a request body and
    would otherwise 401 a path the route itself lets through (LR2 §9.3).

    Takes the ASGI scope, not a path string, so the mount-prefix normalization
    cannot be forgotten at a call site. ``WHITELIST_PATHS`` entries are route
    paths written without ``LIGHTRAG_API_PREFIX``; matching the raw
    ``request.url.path`` against them made the shipped default ``/health,/api/*``
    exempt *every* route as soon as the mount prefix itself started with
    ``/api`` — the whole API, unauthenticated, including ``DELETE /documents``.

    A ``/*`` entry matches on path-segment boundaries only, so ``/api/*`` exempts
    ``/api`` and everything under ``/api/`` but not a sibling route that merely
    starts with those characters (``/graph/*`` must not exempt ``/graphs``). The
    catch-all ``/*`` compiles to an empty prefix and still matches everything,
    since every path starts with ``/``.
    """
    route = get_route_path(scope, mount_prefix)
    for pattern, is_prefix in whitelist_patterns:
        if is_prefix:
            if route == pattern or route.startswith(pattern + "/"):
                return True
        elif route == pattern:
            return True
    return False


def credentials_accepted(
    *,
    token: Optional[str],
    api_key_header_value: Optional[str],
    api_key: Optional[str],
) -> bool:
    """Whether these credentials authenticate, per the configured auth mode.

    The acceptance rules, in one place, for every caller that needs the answer
    as a boolean rather than as an exception:

      - fully open (no ``AUTH_ACCOUNTS``, no API key): nothing is protected
        anywhere, so the request is authenticated;
      - a valid API key authenticates in any mode where one is configured;
      - password auth (``AUTH_ACCOUNTS`` set): a valid non-guest token
        authenticates. A guest token never does — in API-key-only mode it is
        forgeable by anyone (GHSA-f4vv-55c2-5789 / GHSA-xr5c-v5r6-c9f9), so it
        must not substitute for the API key.

    Deliberately path-agnostic: ``WHITELIST_PATHS`` is a separate question
    (:func:`path_is_whitelisted`), because a whitelisted path is unauthenticated
    but reachable — conflating the two would let ``/health`` report itself as an
    authenticated caller and reveal configuration.
    """
    api_key_configured = bool(api_key)
    if not auth_configured and not api_key_configured:
        return True
    if api_key_configured and api_key_header_value and api_key_header_value == api_key:
        return True
    if token:
        try:
            token_info = auth_handler.validate_token(token)
        except Exception:
            token_info = None
        if token_info and auth_configured and token_info.get("role") != "guest":
            return True
    return False


def get_combined_auth_dependency(api_key: Optional[str] = None):
    """
    Create a combined authentication dependency that implements authentication logic
    based on API key, OAuth2 token, and whitelist paths.

    Args:
        api_key (Optional[str]): API key for validation

    Returns:
        Callable: A dependency function that implements the authentication logic
    """
    # Use global whitelist_patterns and auth_configured variables
    # whitelist_patterns and auth_configured are already initialized at module level

    # Only calculate api_key_configured as it depends on the function parameter
    api_key_configured = bool(api_key)

    # Create security dependencies with proper descriptions for Swagger UI
    oauth2_scheme = OAuth2PasswordBearer(
        tokenUrl="login", auto_error=False, description="OAuth2 Password Authentication"
    )

    # If API key is configured, create an API key header security
    api_key_header = None
    if api_key_configured:
        api_key_header = APIKeyHeader(
            name="X-API-Key", auto_error=False, description="API Key Authentication"
        )

    async def combined_dependency(
        request: Request,
        response: Response,  # Added: needed to return new token via response header
        token: str = Security(oauth2_scheme),
        api_key_header_value: Optional[str] = None
        if api_key_header is None
        else Security(api_key_header),
    ):
        # 1. Check if path is in whitelist.
        # The route path, not request.url.path: the latter still carries the
        # mount prefix (see get_route_path), and both the whitelist and the
        # renewal skip list below are written as unprefixed route paths.
        path = get_route_path(request.scope)
        if path_is_whitelisted(request.scope):
            return  # Whitelist path, allow access

        # 2. Validate token first if provided in the request (Ensure 401 error if token is invalid)
        # Kept in scope for step 4: in API-key-only mode a valid guest token
        # authenticates nothing and falls through to the API key check, which is
        # the only exit that can still renew it.
        token_info = None
        if token:
            try:
                token_info = auth_handler.validate_token(token)

                # Auto-renewal deliberately runs AFTER the authorization decision,
                # at each exit that accepts the request (below, and the API key
                # check in step 4). It writes process-wide state keyed on the "sub"
                # claim, so running it here let an unauthenticated caller grow that
                # state on requests the server rejects (GHSA-3wg5-5w54-3rfm).

                # A token only authenticates when it matches the configured auth mode:
                #   - password auth (AUTH_ACCOUNTS set): accept non-guest user tokens
                #   - fully open (no AUTH_ACCOUNTS, no API key): accept guest tokens
                # In the API-key-only profile (API key set, no AUTH_ACCOUNTS) a guest
                # token must NOT authenticate: anyone can obtain one (via /auth-status,
                # /login, or by signing it with the public default secret), so honoring
                # it here would let a forged guest token bypass the X-API-Key check
                # below (GHSA-f4vv-55c2-5789 / GHSA-xr5c-v5r6-c9f9). Instead, fall
                # through so the API key stays mandatory in that mode.
                if not auth_configured and token_info.get("role") == "guest":
                    if not api_key_configured:
                        _renew_token_if_needed(path, response, token_info)
                        return
                    # API-key-only mode: ignore the guest token; the X-API-Key check
                    # below is the sole authority. Fall through (no return, no raise)
                    # and, critically, without renewal bookkeeping: this token did
                    # not authenticate anything.
                elif auth_configured and token_info.get("role") != "guest":
                    # Accept non-guest token if password auth is configured
                    _renew_token_if_needed(path, response, token_info)
                    return
                else:
                    # Token present but not valid for the configured auth mode.
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid token. Please login again.",
                    )
            except HTTPException as e:
                # If already a 401 error, re-raise it
                if e.status_code == status.HTTP_401_UNAUTHORIZED:
                    raise
                # For other exceptions, continue processing

        # 3. Acept all request if no API protection needed
        if not auth_configured and not api_key_configured:
            return

        # 4. Validate API key if provided and API-Key authentication is configured
        if (
            api_key_configured
            and api_key_header_value
            and api_key_header_value == api_key
        ):
            # The API key authenticated this request. If the caller ALSO presented
            # a valid token, renew it here. The WebUI sends Authorization and
            # X-API-Key together, and in API-key-only mode its guest token
            # authenticates nothing (see step 2), so this is the only exit that can
            # keep it fresh -- letting it lapse makes step 2 reject every request
            # with 401 until the client notices and re-fetches /auth-status.
            # Still strictly post-authorization, so no unauthenticated caller
            # reaches the bookkeeping (GHSA-3wg5-5w54-3rfm).
            if token_info is not None:
                _renew_token_if_needed(path, response, token_info)
            return  # API key validation successful

        ### Authentication failed ####

        # if password authentication is configured but not provided, ensure 401 error if auth_configured
        if auth_configured and not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No credentials provided. Please login.",
            )

        # if api key is provided but validation failed
        if api_key_header_value:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Invalid API Key",
            )

        # if api_key_configured but not provided
        if api_key_configured and not api_key_header_value:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="API Key required",
            )

        # Otherwise: refuse access and return 403 error
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="API Key required or login authentication required.",
        )

    return combined_dependency


def get_auth_status_dependency(api_key: Optional[str] = None):
    """Create a dependency that reports whether the request carries accepted
    credentials, WITHOUT enforcing authentication (it never raises).

    Used by endpoints such as ``/health`` that must stay reachable for
    unauthenticated liveness probes (always HTTP 200) while only revealing
    sensitive configuration to authenticated callers. The acceptance rules
    mirror ``get_combined_auth_dependency`` exactly:

      - fully open (no AUTH_ACCOUNTS, no API key): nothing is protected
        anywhere, so the request is treated as authenticated.
      - password auth (AUTH_ACCOUNTS set): a valid non-guest token, or a
        valid API key when one is configured, authenticates.
      - API-key-only (API key set, no AUTH_ACCOUNTS): only a valid API key
        authenticates; a guest token is forgeable and must NOT count
        (GHSA-f4vv-55c2-5789 / GHSA-xr5c-v5r6-c9f9).
    """
    api_key_configured = bool(api_key)
    oauth2_scheme = OAuth2PasswordBearer(
        tokenUrl="login", auto_error=False, description="OAuth2 Password Authentication"
    )
    api_key_header = None
    if api_key_configured:
        api_key_header = APIKeyHeader(
            name="X-API-Key", auto_error=False, description="API Key Authentication"
        )

    async def auth_status_dependency(
        token: str = Security(oauth2_scheme),
        api_key_header_value: Optional[str] = None
        if api_key_header is None
        else Security(api_key_header),
    ) -> bool:
        # Path-agnostic on purpose: /health is itself whitelisted, so folding
        # the whitelist in here would report every caller as authenticated.
        return credentials_accepted(
            token=token,
            api_key_header_value=api_key_header_value,
            api_key=api_key,
        )

    return auth_status_dependency


def whitelist_exposes_api_routes(whitelist_paths: str) -> bool:
    """Return True if WHITELIST_PATHS exempts any Ollama-compatible /api route.

    Mirrors the prefix/exact matching in get_combined_auth_dependency so that a
    catch-all entry such as ``/*`` (which strips to an empty prefix and matches
    every request path, including ``/api/chat``) is recognized as exposing the
    /api routes — not just literal ``/api...`` entries.
    """
    for entry in whitelist_paths.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if entry.endswith("/*"):
            # Prefix match: this entry exempts an /api route when some /api path
            # starts with the prefix ("/api".startswith(prefix) also covers the
            # empty catch-all prefix from "/*") or the prefix is itself under
            # /api/. The "/api/" boundary matters: "/apiary/*" only exempts
            # /apiary..., not /api/chat, so it must NOT be flagged.
            prefix = entry[:-2]
            if "/api".startswith(prefix) or prefix.startswith("/api/"):
                return True
        else:
            # Exact match: only the literal path is exempted.
            if entry == "/api" or entry.startswith("/api/"):
                return True
    return False


def display_splash_screen(args: argparse.Namespace) -> None:
    """
    Display a colorful splash screen showing LightRAG server configuration

    Args:
        args: Parsed command line arguments
    """
    # Banner
    # Banner
    top_border = "╔══════════════════════════════════════════════════════════════╗"
    bottom_border = "╚══════════════════════════════════════════════════════════════╝"
    width = len(top_border) - 4  # width inside the borders

    line1_text = f"LightRAG Server v{core_version}/{api_version}"
    line2_text = "Fast, Lightweight RAG Server Implementation"

    line1 = f"║ {line1_text.center(width)} ║"
    line2 = f"║ {line2_text.center(width)} ║"

    banner = f"""
    {top_border}
    {line1}
    {line2}
    {bottom_border}
    """
    ASCIIColors.cyan(banner)

    # Server Configuration
    ASCIIColors.magenta("\n📡 Server Configuration:")
    ASCIIColors.white("    ├─ Host: ", end="")
    ASCIIColors.yellow(f"{args.host}")
    ASCIIColors.white("    ├─ Port: ", end="")
    ASCIIColors.yellow(f"{args.port}")
    ASCIIColors.white("    ├─ Workers: ", end="")
    ASCIIColors.yellow(f"{args.workers}")
    ASCIIColors.white("    ├─ Timeout: ", end="")
    ASCIIColors.yellow(f"{args.timeout}")
    ASCIIColors.white("    ├─ CORS Origins: ", end="")
    ASCIIColors.yellow(f"{args.cors_origins}")
    ASCIIColors.white("    ├─ SSL Enabled: ", end="")
    ASCIIColors.yellow(f"{args.ssl}")
    if args.ssl:
        ASCIIColors.white("    ├─ SSL Cert: ", end="")
        ASCIIColors.yellow(f"{args.ssl_certfile}")
        ASCIIColors.white("    ├─ SSL Key: ", end="")
        ASCIIColors.yellow(f"{args.ssl_keyfile}")
    ASCIIColors.white("    ├─ Ollama Emulating Model: ", end="")
    ASCIIColors.yellow(f"{ollama_server_infos.LIGHTRAG_MODEL}")
    ASCIIColors.white("    ├─ Log Level: ", end="")
    ASCIIColors.yellow(f"{args.log_level}")
    ASCIIColors.white("    ├─ Verbose Debug: ", end="")
    ASCIIColors.yellow(f"{args.verbose}")
    ASCIIColors.white("    ├─ API Key: ", end="")
    ASCIIColors.yellow("Set" if args.key else "Not Set")
    ASCIIColors.white("    ├─ JWT Auth: ", end="")
    ASCIIColors.yellow("Enabled" if args.auth_accounts else "Disabled")
    # Request limits: previously invisible at startup, which made it hard to tell
    # a deployment that had bounded its request surface from one that had not.
    ASCIIColors.white("    ├─ Max Request Body: ", end="")
    ASCIIColors.yellow(
        f"{args.max_request_body_bytes} bytes"
        if getattr(args, "max_request_body_bytes", 0)
        else "Unlimited"
    )
    ASCIIColors.white("    ├─ Max Upload Size: ", end="")
    ASCIIColors.yellow(
        f"{args.max_upload_size} bytes"
        if getattr(args, "max_upload_size", None)
        else "Unlimited"
    )
    ASCIIColors.white("    └─ Max Pending Documents: ", end="")
    ASCIIColors.yellow(
        f"{args.max_pending_documents}"
        if getattr(args, "max_pending_documents", 0)
        else "Unlimited"
    )

    # Directory Configuration
    ASCIIColors.magenta("\n📂 Directory Configuration:")
    ASCIIColors.white("    ├─ Working Directory: ", end="")
    ASCIIColors.yellow(f"{args.working_dir}")
    ASCIIColors.white("    └─ Input Directory: ", end="")
    ASCIIColors.yellow(f"{args.input_dir}")
    # Embedding Configuration
    ASCIIColors.magenta("\n📊 Embedding Configuration:")
    ASCIIColors.white("    ├─ Binding: ", end="")
    ASCIIColors.yellow(f"{args.embedding_binding}")
    ASCIIColors.white("    ├─ Host: ", end="")
    ASCIIColors.yellow(f"{args.embedding_binding_host}")
    ASCIIColors.white("    ├─ Model: ", end="")
    ASCIIColors.yellow(f"{args.embedding_model}")
    ASCIIColors.white("    ├─ Dimensions: ", end="")
    ASCIIColors.yellow(f"{args.embedding_dim}")
    ASCIIColors.white("    └─ Asymmetric: ", end="")
    ASCIIColors.yellow(f"{args.embedding_asymmetric}")

    # RAG Configuration
    ASCIIColors.magenta("\n⚙️ RAG Configuration:")
    ASCIIColors.white("    ├─ Summary Language: ", end="")
    ASCIIColors.yellow(f"{args.summary_language}")
    ASCIIColors.white("    ├─ Max Parallel Insert: ", end="")
    ASCIIColors.yellow(f"{args.max_parallel_insert}")
    ASCIIColors.white("    ├─ Chunk Size: ", end="")
    ASCIIColors.yellow(f"{args.chunk_size}")
    ASCIIColors.white("    ├─ Chunk Overlap Size: ", end="")
    ASCIIColors.yellow(f"{args.chunk_overlap_size}")
    ASCIIColors.white("    ├─ Cosine Threshold: ", end="")
    ASCIIColors.yellow(f"{args.cosine_threshold}")
    ASCIIColors.white("    ├─ Top-K: ", end="")
    ASCIIColors.yellow(f"{args.top_k}")
    ASCIIColors.white("    └─ Force LLM Summary on Merge: ", end="")
    ASCIIColors.yellow(
        f"{get_env_value('FORCE_LLM_SUMMARY_ON_MERGE', DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, int)}"
    )

    # System Configuration
    ASCIIColors.magenta("\n💾 Storage Configuration:")
    ASCIIColors.white("    ├─ KV Storage: ", end="")
    ASCIIColors.yellow(f"{args.kv_storage}")
    ASCIIColors.white("    ├─ Vector Storage: ", end="")
    ASCIIColors.yellow(f"{args.vector_storage}")
    ASCIIColors.white("    ├─ Graph Storage: ", end="")
    ASCIIColors.yellow(f"{args.graph_storage}")
    ASCIIColors.white("    ├─ Document Status Storage: ", end="")
    ASCIIColors.yellow(f"{args.doc_status_storage}")
    ASCIIColors.white("    └─ Workspace: ", end="")
    ASCIIColors.yellow(f"{args.workspace if args.workspace else '-'}")

    # Server Status
    ASCIIColors.green("\n✨ Server starting up...\n")

    # Server Access Information
    protocol = "https" if args.ssl else "http"
    if args.host == "0.0.0.0":
        ASCIIColors.magenta("\n🌐 Server Access Information:")
        ASCIIColors.white("    ├─ WebUI (local): ", end="")
        ASCIIColors.yellow(f"{protocol}://localhost:{args.port}")
        ASCIIColors.white("    ├─ Remote Access: ", end="")
        ASCIIColors.yellow(f"{protocol}://<your-ip-address>:{args.port}")
        ASCIIColors.white("    ├─ API Documentation (local): ", end="")
        ASCIIColors.yellow(f"{protocol}://localhost:{args.port}/docs")
        ASCIIColors.white("    └─ Alternative Documentation (local): ", end="")
        ASCIIColors.yellow(f"{protocol}://localhost:{args.port}/redoc")

        ASCIIColors.magenta("\n📝 Note:")
        ASCIIColors.cyan("""    Since the server is running on 0.0.0.0:
    - Use 'localhost' or '127.0.0.1' for local access
    - Use your machine's IP address for remote access
    - To find your IP address:
      • Windows: Run 'ipconfig' in terminal
      • Linux/Mac: Run 'ifconfig' or 'ip addr' in terminal
    """)
    else:
        base_url = f"{protocol}://{args.host}:{args.port}"
        ASCIIColors.magenta("\n🌐 Server Access Information:")
        ASCIIColors.white("    ├─ WebUI (local): ", end="")
        ASCIIColors.yellow(f"{base_url}")
        ASCIIColors.white("    ├─ API Documentation: ", end="")
        ASCIIColors.yellow(f"{base_url}/docs")
        ASCIIColors.white("    └─ Alternative Documentation: ", end="")
        ASCIIColors.yellow(f"{base_url}/redoc")

    # Security Notice
    if args.key:
        ASCIIColors.white("✅  Security Notice:")
        ASCIIColors.white("""    API Key authentication is enabled.
    Make sure to include the X-API-Key header in all your requests.
    """)
    if args.auth_accounts:
        ASCIIColors.white("✅  Security Notice:")
        ASCIIColors.white("""    JWT authentication is enabled.
    Make sure to login before making the request, and include the 'Authorization' in the header.
    """)

    # Warn when the server runs without any authentication. In this mode every
    # endpoint is publicly reachable (see get_combined_auth_dependency: with
    # neither AUTH_ACCOUNTS nor LIGHTRAG_API_KEY set, all requests are allowed).
    if not args.key and not args.auth_accounts:
        loopback_hosts = {"127.0.0.1", "::1", "localhost"}
        if args.host in loopback_hosts:
            ASCIIColors.yellow("\n⚠️  Security Warning:")
            ASCIIColors.white(f"""    No authentication is configured (no API Key, no login accounts).
    The server is bound to a loopback address ('{args.host}'), so it is only
    reachable from this machine. Set LIGHTRAG_API_KEY, or AUTH_ACCOUNTS together
    with TOKEN_SECRET, before binding to a non-loopback address (e.g. HOST=0.0.0.0).
    """)
        else:
            ASCIIColors.red("\n🔴 SECURITY ALERT:")
            ASCIIColors.white(f"""    The server is listening on '{args.host}' WITHOUT any authentication.
    Every endpoint (document upload, query, knowledge graph, deletion) is
    publicly accessible to anyone who can reach this address.

    Secure the server before exposing it to a network by setting at least one of:
      - LIGHTRAG_API_KEY=<a-strong-secret>   (X-API-Key header authentication)
      - AUTH_ACCOUNTS=user:password together with TOKEN_SECRET=<a-strong-secret>
                                             (JWT login authentication; AUTH_ACCOUNTS
                                              without TOKEN_SECRET fails to start)
    Or restrict access by binding to loopback only: HOST=127.0.0.1
    """)

    # When authentication IS configured but the server is reachable from a
    # network, warn that the default whitelist still exempts the
    # Ollama-compatible /api/* routes (kept open for Ollama-client compatibility).
    # Those routes invoke the LLM and read the knowledge base, so they stay
    # public unless the operator narrows WHITELIST_PATHS (e.g. to /health).
    #
    # A configured LIGHTRAG_API_PREFIX counts as reachable regardless of the bind
    # address: the prefix exists to be served behind a reverse proxy, so the
    # canonical multi-site deployment (proxy on the public interface, backend on
    # loopback) is exposed while a loopback-only test on the bind address is not.
    # The prefix is normalized first, because LIGHTRAG_API_PREFIX=/ means "no
    # prefix" and must not read as a proxied deployment.
    if args.key or args.auth_accounts:
        loopback_hosts = {"127.0.0.1", "::1", "localhost"}
        ollama_open = whitelist_exposes_api_routes(args.whitelist_paths)
        behind_proxy = bool(normalize_api_prefix(getattr(args, "api_prefix", None)))
        reachable = args.host not in loopback_hosts or behind_proxy
        if reachable and ollama_open:
            where = (
                f"behind the '{args.api_prefix}' reverse-proxy prefix"
                if behind_proxy
                else f"on '{args.host}'"
            )
            ASCIIColors.yellow("\n⚠️  Security Warning:")
            ASCIIColors.white(f"""    WHITELIST_PATHS ('{args.whitelist_paths}') exempts the Ollama-compatible
    /api/* routes (/api/chat, /api/generate, ...) from authentication, so they
    remain publicly accessible {where} even though auth is enabled.
    These routes invoke the LLM and read your knowledge base. If you do not need
    open Ollama access, set WHITELIST_PATHS=/health to require authentication.
    """)

    _warn_about_body_limits(args)
    _warn_about_svg_rasterizer()

    # Ensure splash output flush to system log
    sys.stdout.flush()


def _warn_about_body_limits(args) -> None:
    """Flag body-limit settings that leave a route unbounded.

    The ceilings themselves are printed in the Server Configuration block; this
    only speaks up when a value turns protection off or weakens it in a way an
    operator is unlikely to have intended.
    """
    from lightrag.constants import (
        DEFAULT_MAX_INGEST_BODY_BYTES,
        MULTIPART_OVERHEAD_BYTES,
    )

    body_limit = getattr(args, "max_request_body_bytes", 0) or 0
    max_upload_size = getattr(args, "max_upload_size", None)

    if body_limit <= 0:
        ASCIIColors.yellow("\n⚠️  Security Warning:")
        ASCIIColors.white("""    MAX_REQUEST_BODY_BYTES=0 disables every raw request-body ceiling, including
    the one /documents/upload derives from MAX_UPLOAD_SIZE. A single large body
    can then occupy memory and CPU for as long as it takes to parse.
    """)
        return

    if body_limit > DEFAULT_MAX_INGEST_BODY_BYTES:
        ASCIIColors.yellow("\n⚠️  Configuration Notice:")
        ASCIIColors.white(f"""    MAX_REQUEST_BODY_BYTES is set to {body_limit} bytes, which now applies to
    EVERY non-upload route including /query and the Ollama-compatible /api/*
    endpoints. /documents/upload no longer needs it: that route derives its own
    ceiling from MAX_UPLOAD_SIZE (+{MULTIPART_OVERHEAD_BYTES} bytes of multipart
    overhead). If this value was raised for uploads, removing it restores a
    tighter bound on the rest of the API.
    """)

    if not (isinstance(max_upload_size, int) and max_upload_size > 0):
        ASCIIColors.yellow("\n⚠️  Security Warning:")
        ASCIIColors.white("""    MAX_UPLOAD_SIZE is unset or unlimited, so /documents/upload has no raw
    request-body ceiling to derive and accepts a body of any size. Set
    MAX_UPLOAD_SIZE to the largest file you intend to accept.
    """)


def _warn_about_svg_rasterizer() -> None:
    """Flag a missing/broken cairosvg->libcairo rasterization path.

    ``cairosvg`` is a cffi binding: ``pip install cairosvg`` always succeeds,
    but rendering only works if the native ``libcairo`` shared library is also
    present on the host, which pip/uv cannot install. Without it, SVG images in
    markdown/textpack documents are silently skipped (the rest of the document
    is unaffected) — surfacing the gap here, once, is cheaper for an operator
    to notice than a per-document warning buried in later processing logs.
    """
    from lightrag.parser.markdown.parser import check_svg_rasterizer

    error = check_svg_rasterizer()
    if error is None:
        return
    ASCIIColors.yellow("\n⚠️  SVG Rasterization Warning:")
    ASCIIColors.white(f"""    {error}
    Install the native libcairo library to fix this:
      Debian/Ubuntu : sudo apt-get install -y libcairo2
      RHEL/Fedora   : sudo dnf install -y cairo
      macOS         : brew install cairo
      Windows       : install the GTK3 runtime (bundles libcairo-2.dll)
    """)
