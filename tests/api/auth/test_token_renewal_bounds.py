"""Token auto-renewal: GHSA-3wg5-5w54-3rfm regressions, plus the renewal decision.

Every test here drives the REAL ``get_combined_auth_dependency`` /
``AuthHandler.validate_token`` and asserts on the real
``lightrag.api.utils_api._token_renewal_cache``.

This file replaces ``test_token_auto_renewal.py``, which imported none of the
production code: it declared its own ``_token_renewal_cache = {}``, re-derived
the renewal arithmetic inside each test, and asserted on values it had just
written to a ``Mock`` -- so it stayed green throughout the defect below, and one
of its cases (``test_token_renewal_disabled``) asserted on a header dict nothing
had ever written to and could not fail at all. ``TestRenewalDecision`` at the
bottom re-covers the branches it named, against the real dependency.

The defect had three parts, all reachable by an attacker holding no credential in
any profile that runs on the default guest-mode JWT secret:

1. the renewal bookkeeping ran BEFORE the authorization decision, so a request
   answered 403 still wrote a permanent, process-wide cache entry;
2. the cache key was the raw ``sub`` claim, unbounded in both count and size;
3. the renewal log line interpolated that same raw claim, so CR/LF in it forged
   whole log records.
"""

import importlib
import logging
import sys
import time
from types import SimpleNamespace

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

# ``lightrag.api.config`` resolves its args lazily on first attribute access and
# would otherwise argparse pytest's own argv. Import under a clean argv (same
# approach as test_shared_credential_check.py) rather than reloading, so module
# identity stays intact for the routers that captured these objects by value.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
config_module = importlib.import_module("lightrag.api.config")
utils_api = importlib.import_module("lightrag.api.utils_api")
_auth = importlib.import_module("lightrag.api.auth")
sys.argv = _original_argv

auth_handler = _auth.auth_handler

# Symbols introduced by the fix are referenced INSIDE tests, never at module
# level: a module-level reference would turn every behavioral assertion below
# into a collection-time AttributeError, which proves only that the new names
# exist and not that the defect is closed.

API_KEY = "the-operators-secret-api-key"


@pytest.fixture(autouse=True)
def _clean_renewal_state(monkeypatch):
    """Neutralize import-time auth state and start from an empty cache."""
    utils_api._token_renewal_cache.clear()
    # No whitelist, so the dependency never short-circuits at step 1.
    monkeypatch.setattr(utils_api, "whitelist_patterns", [])
    monkeypatch.setattr(config_module.global_args, "token_auto_renew", True)
    monkeypatch.setattr(config_module.global_args, "token_renew_threshold", 0.5)
    monkeypatch.setattr(auth_handler, "guest_expire_hours", 24)
    monkeypatch.setattr(auth_handler, "expire_hours", 24)
    yield
    utils_api._token_renewal_cache.clear()


def _client(monkeypatch, *, auth_configured: bool, api_key: str | None):
    """Mount the real dependency under a chosen auth profile."""
    monkeypatch.setattr(utils_api, "auth_configured", auth_configured)
    app = FastAPI()
    dependency = utils_api.get_combined_auth_dependency(api_key)

    @app.get("/documents", dependencies=[Depends(dependency)])
    async def documents():
        return {"ok": True}

    return TestClient(app)


def _near_expiry_token(username: str, role: str = "guest") -> str:
    """A signature-valid token 1h from expiry, i.e. inside the renewal window.

    Minted through the real ``create_token`` so it is signed with whatever secret
    this process loaded -- the same capability the advisory's attacker gets from
    the public ``DEFAULT_TOKEN_SECRET`` in the API-key-only profile.
    """
    return auth_handler.create_token(
        username=username, role=role, custom_expire_hours=1
    )


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _handler_with_accounts(monkeypatch, auth_accounts: str):
    """Build a fresh AuthHandler over a chosen AUTH_ACCOUNTS string.

    ``auth.py`` imported ``global_args`` by value, so the module attribute is what
    ``AuthHandler.__init__`` reads. Patching it beats reloading the module, which
    would hand out a second ``auth_handler`` identity to everything that already
    captured the first one.
    """
    monkeypatch.setattr(
        _auth,
        "global_args",
        SimpleNamespace(
            auth_accounts=auth_accounts,
            token_secret="test-jwt-secret",
            jwt_algorithm="HS256",
            token_expire_hours=48,
            guest_token_expire_hours=24,
        ),
    )
    return _auth.AuthHandler()


@pytest.mark.offline
class TestNoPreAuthorizationWrite:
    """Part 1: rejected requests must not write server-side state."""

    def test_rejected_request_writes_no_renewal_state(self, monkeypatch):
        """API-key-only profile: a forged guest token is refused and leaves nothing.

        Pre-fix this request was answered 403 *after* the renewal block had already
        inserted into ``_token_renewal_cache``, which is the whole DoS primitive.
        """
        client = _client(monkeypatch, auth_configured=False, api_key=API_KEY)

        response = client.get(
            "/documents", headers=_bearer(_near_expiry_token("alice"))
        )

        assert response.status_code == 403
        assert utils_api._token_renewal_cache == {}
        assert "X-New-Token" not in response.headers

    def test_repeated_rejections_do_not_accumulate(self, monkeypatch):
        """The unbounded-growth loop itself: every request rejected, nothing retained."""
        client = _client(monkeypatch, auth_configured=False, api_key=API_KEY)

        statuses = {
            client.get(
                "/documents", headers=_bearer(_near_expiry_token(f"attacker-{i}"))
            ).status_code
            for i in range(50)
        }

        assert statuses == {403}
        assert len(utils_api._token_renewal_cache) == 0


@pytest.mark.offline
class TestSubjectClaimBounds:
    """Part 2: the claim used as cache key / log payload is bounded and well-typed."""

    def test_oversized_subject_is_rejected(self):
        """A ~32 KB ``sub`` must not survive validation.

        Pre-fix ``validate_token`` returned it verbatim, so each request could
        retain tens of kilobytes indefinitely.
        """
        token = auth_handler.create_token(
            username="B" * 32_000, role="guest", custom_expire_hours=1
        )

        with pytest.raises(Exception) as excinfo:
            auth_handler.validate_token(token)

        assert getattr(excinfo.value, "status_code", None) == 401

    def test_subject_at_limit_is_accepted(self):
        """The bound is a cap, not a tightening that breaks real usernames."""
        username = "u" * _auth.MAX_TOKEN_SUBJECT_LENGTH
        token = auth_handler.create_token(
            username=username, role="guest", custom_expire_hours=1
        )

        assert auth_handler.validate_token(token)["username"] == username

    def test_oversized_subject_never_reaches_the_cache(self, monkeypatch):
        """End-to-end: the fully-open profile accepts guest tokens, but not huge ones."""
        client = _client(monkeypatch, auth_configured=False, api_key=None)
        token = auth_handler.create_token(
            username="B" * 32_000, role="guest", custom_expire_hours=1
        )

        response = client.get("/documents", headers=_bearer(token))

        assert response.status_code == 401
        assert utils_api._token_renewal_cache == {}

    @pytest.mark.parametrize("claim", ["sub", "exp"])
    def test_missing_required_claim_is_401_not_500(self, claim):
        """A signature-valid token missing a claim is invalid input, not a crash.

        Pre-fix the bare ``payload[claim]`` raised ``KeyError``, which
        ``except jwt.PyJWTError`` does not cover, so it escaped ``validate_token``
        and surfaced to an unauthenticated caller as HTTP 500.
        """
        payload = {
            "sub": "alice",
            "exp": int(time.time()) + 3600,
            "role": "guest",
            "metadata": {},
        }
        del payload[claim]
        token = jwt.encode(
            payload, auth_handler.secret, algorithm=auth_handler.algorithm
        )

        with pytest.raises(Exception) as excinfo:
            auth_handler.validate_token(token)

        assert getattr(excinfo.value, "status_code", None) == 401

    def test_out_of_range_expiry_is_401_not_500(self):
        """A numeric but absurd ``exp`` passes PyJWT and then overflows datetime."""
        token = jwt.encode(
            {"sub": "alice", "exp": 10**20, "role": "guest", "metadata": {}},
            auth_handler.secret,
            algorithm=auth_handler.algorithm,
        )

        with pytest.raises(Exception) as excinfo:
            auth_handler.validate_token(token)

        assert getattr(excinfo.value, "status_code", None) == 401


@pytest.mark.offline
class TestConfiguredAccountsRespectTheSameBound:
    """The claim cap must be shared with account config, not only enforced on read.

    ``validate_token`` capping ``sub`` is only sound if nothing can mint a token
    above that cap. ``AUTH_ACCOUNTS`` accepted any non-empty username, so a
    257-character account could authenticate at /login, receive a token, and then
    be rejected on every subsequent request -- an account that is configured,
    logs in, and cannot be used.
    """

    def test_oversized_configured_username_is_refused_at_startup(self, monkeypatch):
        """Fail fast with an actionable message instead of at first API call."""
        with pytest.raises(ValueError, match="at most"):
            _handler_with_accounts(
                monkeypatch, f"{'u' * (_auth.MAX_TOKEN_SUBJECT_LENGTH + 1)}:secret"
            )

    def test_oversized_username_error_does_not_log_the_password(
        self, monkeypatch, caplog
    ):
        """The rejected entry carries a password, so only lengths may be logged."""
        lightrag_logger = logging.getLogger("lightrag")
        monkeypatch.setattr(lightrag_logger, "propagate", True)
        username = "u" * (_auth.MAX_TOKEN_SUBJECT_LENGTH + 1)

        with caplog.at_level(logging.ERROR, logger="lightrag"):
            with pytest.raises(ValueError):
                _handler_with_accounts(monkeypatch, f"{username}:sup3r-s3cret-pw")

        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert "sup3r-s3cret-pw" not in logged
        assert username not in logged
        assert str(len(username)) in logged

    @pytest.mark.parametrize("length", [1, 64, 256, 257, 1000])
    def test_accepted_account_always_yields_a_usable_token(self, monkeypatch, length):
        """The invariant itself: whatever is accepted must be able to log in AND work.

        Pre-fix the 257 and 1000 cases broke it -- the handler accepted the
        account, ``create_token`` signed the username into ``sub``, and
        ``validate_token`` then rejected that very token with 401.

        Deliberately expressed as an implication rather than pinned to the current
        cap: an account refused at configuration time satisfies it vacuously, so
        this stays meaningful whatever MAX_TOKEN_SUBJECT_LENGTH is set to.
        """
        username = "u" * length
        try:
            handler = _handler_with_accounts(monkeypatch, f"{username}:secret")
        except ValueError:
            return  # Refused up front; there is no unusable account to speak of.

        assert handler.verify_password(username, "secret")
        token = handler.create_token(username=username, role="user")

        assert handler.validate_token(token)["username"] == username

    def test_login_then_request_succeeds_for_a_long_username(self, monkeypatch):
        """Same invariant driven through the real dependency, not just the handler."""
        username = "u" * _auth.MAX_TOKEN_SUBJECT_LENGTH
        handler = _handler_with_accounts(monkeypatch, f"{username}:secret")
        # The dependency reads the module-level singleton, so point it at ours.
        monkeypatch.setattr(utils_api, "auth_handler", handler)
        monkeypatch.setattr(_auth, "auth_handler", handler)
        client = _client(monkeypatch, auth_configured=True, api_key=None)

        token = handler.create_token(username=username, role="user")
        response = client.get("/documents", headers=_bearer(token))

        assert response.status_code == 200

    def test_valid_accounts_are_still_accepted(self, monkeypatch):
        """The new rejection must not swallow ordinary configurations."""
        handler = _handler_with_accounts(monkeypatch, "admin:pw1,alice:pw2")

        assert set(handler.accounts) == {"admin", "alice"}

    def test_malformed_entry_still_reports_the_original_error(self, monkeypatch):
        """The pre-existing format check keeps priority over the new one."""
        with pytest.raises(ValueError, match="user:password pairs"):
            _handler_with_accounts(monkeypatch, "no-colon-here")


@pytest.mark.offline
class TestApiKeyExitStillRenews:
    """A request the API key authenticated must still get its token refreshed.

    The WebUI sends Authorization and X-API-Key together, and /auth-status hands
    out a guest token whenever AUTH_ACCOUNTS is unset -- which includes the
    API-key-only profile. That guest token authenticates nothing there (it falls
    through to the mandatory API key check), but it is still validated at step 2,
    so once it expires every request 401s until the client re-fetches it. Moving
    renewal off the pre-authorization path must not take this exit with it.
    """

    def test_api_key_plus_guest_token_renews(self, monkeypatch):
        client = _client(monkeypatch, auth_configured=False, api_key=API_KEY)

        response = client.get(
            "/documents",
            headers={**_bearer(_near_expiry_token("guest")), "X-API-Key": API_KEY},
        )

        assert response.status_code == 200
        assert response.headers.get("X-New-Token")
        assert set(utils_api._token_renewal_cache) == {"guest"}

    def test_expired_token_rejects_even_with_a_correct_api_key(self, monkeypatch):
        """Why the renewal exit is load-bearing, not a nicety.

        Step 2 validates any presented token before the API key is ever examined,
        so an expired one 401s the request outright. That is pre-existing behavior;
        it is the reason a lapsed guest token must not be allowed to happen.
        """
        client = _client(monkeypatch, auth_configured=False, api_key=API_KEY)
        expired = auth_handler.create_token(
            username="guest", role="guest", custom_expire_hours=-1
        )

        response = client.get(
            "/documents", headers={**_bearer(expired), "X-API-Key": API_KEY}
        )

        assert response.status_code == 401

    def test_api_key_without_token_renews_nothing(self, monkeypatch):
        """No token presented means there is nothing to refresh, and no write."""
        client = _client(monkeypatch, auth_configured=False, api_key=API_KEY)

        response = client.get("/documents", headers={"X-API-Key": API_KEY})

        assert response.status_code == 200
        assert "X-New-Token" not in response.headers
        assert utils_api._token_renewal_cache == {}

    def test_wrong_api_key_with_valid_token_still_writes_nothing(self, monkeypatch):
        """The step 4 exit is reached only on a correct key -- the fix must hold."""
        client = _client(monkeypatch, auth_configured=False, api_key=API_KEY)

        response = client.get(
            "/documents",
            headers={**_bearer(_near_expiry_token("alice")), "X-API-Key": "wrong"},
        )

        assert response.status_code == 403
        assert utils_api._token_renewal_cache == {}


@pytest.mark.offline
class TestRenewalCacheIsBounded:
    """Part 2 (cont.): the table has a hard ceiling, enforced behaviorally."""

    def test_hard_ceiling_evicts_oldest(self, monkeypatch):
        """Distinct authenticated subjects cannot grow the table past the cap.

        The cap is lowered so this runs behaviorally through the real dependency:
        pre-fix all 20 subjects were retained, because nothing evicted anything.
        ``raising=False`` keeps that a failed length assertion rather than an
        AttributeError on the constant the fix introduced.
        """
        monkeypatch.setattr(utils_api, "_MAX_TRACKED_RENEWALS", 5, raising=False)
        client = _client(monkeypatch, auth_configured=False, api_key=None)

        for i in range(20):
            response = client.get(
                "/documents", headers=_bearer(_near_expiry_token(f"guest-{i}"))
            )
            assert response.status_code == 200

        assert len(utils_api._token_renewal_cache) == 5
        # Eviction is oldest-first, so the survivors are the most recent subjects.
        assert set(utils_api._token_renewal_cache) == {
            f"guest-{i}" for i in range(15, 20)
        }

    def test_entries_past_the_interval_are_purged(self):
        """Entries that can no longer suppress a renewal are dropped eagerly.

        Mechanism test, not fix-proof: it exercises the helper the fix introduced
        (the behavioral proof is ``test_hard_ceiling_evicts_oldest`` above).
        """
        now = time.time()
        utils_api._record_token_renewal("stale", now)
        utils_api._record_token_renewal(
            "fresh", now + utils_api._RENEWAL_MIN_INTERVAL + 1
        )

        assert set(utils_api._token_renewal_cache) == {"fresh"}

    def test_refreshed_subject_moves_to_the_tail(self):
        """Insertion order must stay time order, or purging from the head misfires."""
        now = time.time()
        utils_api._record_token_renewal("a", now)
        utils_api._record_token_renewal("b", now + 1)
        utils_api._record_token_renewal("a", now + 2)

        assert list(utils_api._token_renewal_cache) == ["b", "a"]


@pytest.mark.offline
class TestRenewalLogSanitization:
    """Part 3: CR/LF in the claim must not forge log records (CWE-117)."""

    def test_crlf_in_subject_is_neutralized(self, monkeypatch, caplog):
        """The renewal INFO line must not carry raw newlines from the claim.

        Pre-fix the emitted record read as three lines, the middle one a
        fabricated ``CRITICAL:lightrag:`` entry.
        """
        client = _client(monkeypatch, auth_configured=False, api_key=None)
        # The lightrag logger does not propagate, so caplog sees nothing by default.
        lightrag_logger = logging.getLogger("lightrag")
        monkeypatch.setattr(lightrag_logger, "propagate", True)
        injected = "alice\nCRITICAL:lightrag: SECURITY AUDIT PASSED - admin login\nbob"

        with caplog.at_level(logging.INFO, logger="lightrag"):
            response = client.get(
                "/documents", headers=_bearer(_near_expiry_token(injected))
            )

        assert response.status_code == 200
        renewal_records = [
            record.getMessage()
            for record in caplog.records
            if "auto-renewed" in record.getMessage()
        ]
        assert renewal_records, "renewal did not happen; the test proves nothing"
        for message in renewal_records:
            assert "\n" not in message
            assert "\r" not in message
        # The raw value is still what the cache and the new token are keyed on.
        assert injected in utils_api._token_renewal_cache


@pytest.mark.offline
class TestRenewalStillWorks:
    """Stability: the fix must not disable the feature it hardens."""

    def test_password_profile_renews_authenticated_user(self, monkeypatch):
        client = _client(monkeypatch, auth_configured=True, api_key=None)

        response = client.get(
            "/documents", headers=_bearer(_near_expiry_token("realuser", role="user"))
        )

        assert response.status_code == 200
        assert response.headers.get("X-New-Token")
        assert set(utils_api._token_renewal_cache) == {"realuser"}

    def test_guest_profile_renews_guest(self, monkeypatch):
        client = _client(monkeypatch, auth_configured=False, api_key=None)

        response = client.get(
            "/documents", headers=_bearer(_near_expiry_token("guest"))
        )

        assert response.status_code == 200
        assert response.headers.get("X-New-Token")

    def test_rate_limit_still_suppresses_second_renewal(self, monkeypatch):
        client = _client(monkeypatch, auth_configured=False, api_key=None)

        first = client.get("/documents", headers=_bearer(_near_expiry_token("guest")))
        second = client.get("/documents", headers=_bearer(_near_expiry_token("guest")))

        assert first.headers.get("X-New-Token")
        assert "X-New-Token" not in second.headers

    def test_skip_paths_still_skip(self, monkeypatch):
        monkeypatch.setattr(utils_api, "auth_configured", False)
        app = FastAPI()
        dependency = utils_api.get_combined_auth_dependency(None)

        @app.get("/documents/paginated", dependencies=[Depends(dependency)])
        async def paginated():
            return {"ok": True}

        response = TestClient(app).get(
            "/documents/paginated", headers=_bearer(_near_expiry_token("guest"))
        )

        assert response.status_code == 200
        assert "X-New-Token" not in response.headers
        assert utils_api._token_renewal_cache == {}


@pytest.mark.offline
class TestRenewalDecision:
    """When renewal fires, driven through the real dependency.

    These re-cover the branches ``test_token_auto_renewal.py`` named but never
    reached. Rather than pinning the exact threshold instant -- ``exp`` is minted
    relative to a clock that moves between mint and check, so an at-the-boundary
    assertion is inherently flaky -- each case sits clearly on one side of it.
    """

    def _get(
        self,
        monkeypatch,
        *,
        expire_hours,
        role="guest",
        username="guest",
        auth_configured=False,
    ):
        client = _client(monkeypatch, auth_configured=auth_configured, api_key=None)
        token = auth_handler.create_token(
            username=username, role=role, custom_expire_hours=expire_hours
        )
        return client.get("/documents", headers=_bearer(token))

    def test_renews_below_threshold(self, monkeypatch):
        """11h left of a nominal 24h token is under the 50% mark."""
        response = self._get(monkeypatch, expire_hours=11)

        assert response.status_code == 200
        assert response.headers.get("X-New-Token")

    def test_does_not_renew_above_threshold(self, monkeypatch):
        """13h left is over the mark, so nothing is issued and nothing recorded."""
        response = self._get(monkeypatch, expire_hours=13)

        assert response.status_code == 200
        assert "X-New-Token" not in response.headers
        assert utils_api._token_renewal_cache == {}

    def test_disabled_globally(self, monkeypatch):
        """TOKEN_AUTO_RENEW=false suppresses renewal even deep inside the window."""
        monkeypatch.setattr(config_module.global_args, "token_auto_renew", False)

        response = self._get(monkeypatch, expire_hours=1)

        assert response.status_code == 200
        assert "X-New-Token" not in response.headers
        assert utils_api._token_renewal_cache == {}

    def test_role_selects_which_expiry_budget_applies(self, monkeypatch):
        """The threshold is a fraction of the role's OWN configured lifetime.

        Constructed so the two branches disagree: with guest_expire_hours=2 and
        expire_hours=24, a token 3h from expiry is above the guest threshold (1h)
        but below the user one (12h). A guest token must therefore NOT renew --
        reading the wrong budget here would renew it.
        """
        monkeypatch.setattr(auth_handler, "guest_expire_hours", 2)
        monkeypatch.setattr(auth_handler, "expire_hours", 24)

        guest = self._get(monkeypatch, expire_hours=3, role="guest")
        assert "X-New-Token" not in guest.headers

        # Same token lifetime, non-guest role: now under the 12h mark, so it renews.
        # Needs the password profile, since that is where a user token authenticates.
        user = self._get(
            monkeypatch,
            expire_hours=3,
            role="user",
            username="realuser",
            auth_configured=True,
        )
        assert user.headers.get("X-New-Token")

    def test_rate_limit_allows_renewal_after_the_interval(self, monkeypatch):
        """The 60s interval is a delay, not a permanent block.

        The prior timestamp is pre-seeded rather than waited out; that is state
        setup, not a re-implementation of the decision under test.
        """
        utils_api._token_renewal_cache["guest"] = (
            time.time() - utils_api._RENEWAL_MIN_INTERVAL - 1
        )

        response = self._get(monkeypatch, expire_hours=1)

        assert response.headers.get("X-New-Token")

    def test_rate_limits_are_per_subject(self, monkeypatch):
        """One subject's recent renewal must not suppress another's."""
        client = _client(monkeypatch, auth_configured=False, api_key=None)

        first = client.get("/documents", headers=_bearer(_near_expiry_token("alice")))
        second = client.get("/documents", headers=_bearer(_near_expiry_token("bob")))

        assert first.headers.get("X-New-Token")
        assert second.headers.get("X-New-Token")
        assert set(utils_api._token_renewal_cache) == {"alice", "bob"}


@pytest.mark.offline
class TestRenewalSkipUnderApiPrefix:
    """``_TOKEN_RENEWAL_SKIP_PATHS`` is a list of route paths too.

    It shares the path input with the whitelist check, so it shared that bug: the
    dependency passed ``request.url.path``, which still carries the mount prefix,
    and under ``LIGHTRAG_API_PREFIX`` no skip entry matched. The frequently-polled
    endpoints the list exists to protect went back to minting a token on every
    poll.

    A local app is built here instead of reusing ``_client``: that helper mounts a
    bare ``FastAPI()`` with only ``GET /documents`` and no ``root_path``, so a
    request to ``/api/v1/documents/paginated`` would 404 without ever running the
    dependency -- and an assertion that no ``X-New-Token`` came back would pass
    for the wrong reason.
    """

    @staticmethod
    def _client(monkeypatch, api_prefix: str) -> TestClient:
        monkeypatch.setattr(utils_api, "auth_configured", False)
        server = importlib.import_module("lightrag.api.lightrag_server")

        dependency = utils_api.get_combined_auth_dependency(None)
        app = FastAPI(root_path=api_prefix or None)
        if api_prefix:
            app.add_middleware(server._RootPathNormalizationMiddleware)

        # In the skip list, and its sibling that is not.
        @app.get("/documents/paginated", dependencies=[Depends(dependency)])
        async def paginated():
            return {"documents": []}

        @app.get("/documents", dependencies=[Depends(dependency)])
        async def documents():
            return {"ok": True}

        return TestClient(app)

    @pytest.mark.parametrize("mode", ["verbatim", "strip"])
    def test_skip_list_still_applies_under_a_mount_prefix(self, monkeypatch, mode):
        client = self._client(monkeypatch, "/api/v1")
        token = _near_expiry_token("guest")
        prefix = "" if mode == "strip" else "/api/v1"

        skipped = client.get(f"{prefix}/documents/paginated", headers=_bearer(token))
        renewed = client.get(f"{prefix}/documents", headers=_bearer(token))

        # The route ran: a 404 here would mean the dependency never executed and
        # the header assertions below would prove nothing.
        assert skipped.status_code == 200
        assert renewed.status_code == 200

        assert skipped.headers.get("X-New-Token") is None
        # Positive control: renewal is genuinely enabled in this app, so the
        # missing header above is the skip list working, not renewal being off.
        assert renewed.headers.get("X-New-Token")
