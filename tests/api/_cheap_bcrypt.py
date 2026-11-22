"""Cheap bcrypt for tests that count verifications rather than time them.

Production hashes at cost 12, about 0.3 s per hash or check on a CI core, and
``AuthHandler.verify_password`` runs one on every login path. Tests of which
path ran, how often, and with what result do not depend on the cost, so they
can use bcrypt's minimum, cost 4, where one check takes about a millisecond.
A test that judges the cost or the timing itself must not use this.

``cheap_bcrypt_salts`` makes new hashes cheap (``hash_password`` looks
``bcrypt.gensalt`` up at call time). The timing-equalizing dummy spec is a
module global of ``lightrag.api.auth``, which some tests reload, so each test
installs ``CHEAP_DUMMY_SPEC`` on the module object it actually uses.
"""

from __future__ import annotations

import bcrypt
import pytest

from lightrag.api.passwords import BCRYPT_PASSWORD_PREFIX

CHEAP_ROUNDS = 4
_real_gensalt = bcrypt.gensalt

# A cost-4 stand-in for ``lightrag.api.auth._DUMMY_VERIFY_SPEC``.
CHEAP_DUMMY_SPEC = BCRYPT_PASSWORD_PREFIX + bcrypt.hashpw(
    b"lightrag-test-dummy", _real_gensalt(CHEAP_ROUNDS)
).decode("utf-8")


@pytest.fixture
def cheap_bcrypt_salts(monkeypatch) -> None:
    monkeypatch.setattr(
        bcrypt,
        "gensalt",
        lambda rounds=CHEAP_ROUNDS, prefix=b"2b": _real_gensalt(rounds, prefix),
    )
