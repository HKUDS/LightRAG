"""ContentPolicyViolationError must survive message-only reconstruction.

``lightrag/utils.py`` re-raises binding errors as ``type(e)(error_msg)`` with
no fallback, and ``create_prefixed_exception`` reconstructs via
``type(e)(*args)``. A constructor with keyword-only structured fields would
degrade both to ``TypeError`` and lose the type in doc_status, so the error
must stay constructible from a single positional message string, like
``EmptyTruncatedResponseError``.
"""

import pytest

from lightrag.exceptions import ContentPolicyViolationError
from lightrag.utils import create_prefixed_exception

pytestmark = pytest.mark.offline


def test_message_only_constructor_survives_type_reconstruction():
    error = ContentPolicyViolationError("provider moderation rejected the request")

    assert isinstance(error, RuntimeError)
    assert str(error) == "provider moderation rejected the request"

    # The utils.py re-raise shape: type(e)(error_msg), no keyword arguments.
    rebuilt = type(error)("rewritten message")
    assert type(rebuilt) is ContentPolicyViolationError
    assert str(rebuilt) == "rewritten message"


def test_create_prefixed_exception_preserves_type():
    error = ContentPolicyViolationError("data_inspection_failed")

    prefixed = create_prefixed_exception(error, "chunk-1")

    assert type(prefixed) is ContentPolicyViolationError
    assert str(prefixed) == "chunk-1: data_inspection_failed"