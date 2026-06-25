"""
Utility functions for the LightRAG API.
"""

import os
import argparse
from typing import Optional, List, Tuple
import sys
import time
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
from .auth import auth_handler
from .config import ollama_server_infos, global_args, get_env_value

logger = logging.getLogger("lightrag")

# ========== Token Renewal Rate Limiting ==========
# Cache to track last renewal time per user (username as key)
# Format: {username: last_renewal_timestamp}
_token_renewal_cache: dict[str, float] = {}
_RENEWAL_MIN_INTERVAL = 60  # Minimum 60 seconds between renewals for same user

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
        # 1. Check if path is in whitelist
        path = request.url.path
        for pattern, is_prefix in whitelist_patterns:
            if (is_prefix and path.startswith(pattern)) or (
                not is_prefix and path == pattern
            ):
                return  # Whitelist path, allow access

        # 2. Validate token first if provided in the request (Ensure 401 error if token is invalid)
        if token:
            try:
                token_info = auth_handler.validate_token(token)

                # ========== Token Auto-Renewal Logic ==========
                from lightrag.api.config import global_args
                from datetime import datetime, timezone

                if global_args.token_auto_renew:
                    # Check if current path should skip token renewal
                    skip_renewal = any(
                        path == skip_path or path.startswith(skip_path + "/")
                        for skip_path in _TOKEN_RENEWAL_SKIP_PATHS
                    )

                    if skip_renewal:
                        logger.debug(f"Token auto-renewal skipped for path: {path}")
                    else:
                        try:
                            expire_time = token_info.get("exp")
                            if expire_time:
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
                                if (
                                    remaining_seconds
                                    < total_seconds * global_args.token_renew_threshold
                                ):
                                    # ========== Rate Limiting Check ==========
                                    username = token_info["username"]
                                    current_time = time.time()
                                    last_renewal = _token_renewal_cache.get(username, 0)
                                    time_since_last_renewal = (
                                        current_time - last_renewal
                                    )

                                    # Only renew if enough time has passed since last renewal
                                    if time_since_last_renewal >= _RENEWAL_MIN_INTERVAL:
                                        new_token = auth_handler.create_token(
                                            username=username,
                                            role=role,
                                            metadata=token_info.get("metadata", {}),
                                        )
                                        # Return new token via response header
                                        response.headers["X-New-Token"] = new_token

                                        # Update renewal cache
                                        _token_renewal_cache[username] = current_time

                                        # Optional: log renewal
                                        logger.info(
                                            f"Token auto-renewed for user {username} "
                                            f"(role: {role}, remaining: {remaining_seconds:.0f}s)"
                                        )
                                    else:
                                        # Log skip due to rate limit
                                        logger.debug(
                                            f"Token renewal skipped for {username} "
                                            f"(rate limit: last renewal {time_since_last_renewal:.0f}s ago)"
                                        )
                                    # ========== End of Rate Limiting Check ==========
                        except Exception as e:
                            # Renewal failure should not affect normal request, just log
                            logger.warning(f"Token auto-renew failed: {e}")
                # ========== End of Token Auto-Renewal Logic ==========

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
                        return
                    # API-key-only mode: ignore the guest token; the X-API-Key check
                    # below is the sole authority. Fall through (no return, no raise).
                elif auth_configured and token_info.get("role") != "guest":
                    # Accept non-guest token if password auth is configured
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
        # Fully-open mode: nothing is protected anywhere, so reveal config too.
        if not auth_configured and not api_key_configured:
            return True

        # A valid API key authenticates in any mode where one is configured.
        if (
            api_key_configured
            and api_key_header_value
            and api_key_header_value == api_key
        ):
            return True

        if token:
            try:
                token_info = auth_handler.validate_token(token)
            except Exception:
                token_info = None
            if token_info:
                role = token_info.get("role")
                # Password auth: accept a non-guest token. A guest token never
                # authenticates here (in API-key-only mode it is forgeable).
                if auth_configured and role != "guest":
                    return True

        return False

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
    ASCIIColors.white("    └─ JWT Auth: ", end="")
    ASCIIColors.yellow("Enabled" if args.auth_accounts else "Disabled")

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

    # When authentication IS configured but the server is exposed on a
    # non-loopback address, warn that the default whitelist still exempts the
    # Ollama-compatible /api/* routes (kept open for Ollama-client compatibility).
    # Those routes invoke the LLM and read the knowledge base, so they stay
    # public unless the operator narrows WHITELIST_PATHS (e.g. to /health).
    if args.key or args.auth_accounts:
        loopback_hosts = {"127.0.0.1", "::1", "localhost"}
        ollama_open = whitelist_exposes_api_routes(args.whitelist_paths)
        if args.host not in loopback_hosts and ollama_open:
            ASCIIColors.yellow("\n⚠️  Security Warning:")
            ASCIIColors.white(f"""    WHITELIST_PATHS ('{args.whitelist_paths}') exempts the Ollama-compatible
    /api/* routes (/api/chat, /api/generate, ...) from authentication, so they
    remain publicly accessible on '{args.host}' even though auth is enabled.
    These routes invoke the LLM and read your knowledge base. If you do not need
    open Ollama access, set WHITELIST_PATHS=/health to require authentication.
    """)

    # Ensure splash output flush to system log
    sys.stdout.flush()
