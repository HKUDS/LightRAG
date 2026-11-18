"""Shared aiohttp session mock for the rerank provider wrapper tests.

``generic_rerank_api`` opens ``aiohttp.ClientSession()`` and posts inside two
nested ``async with`` blocks; patch ``lightrag.rerank.aiohttp.ClientSession``
with ``return_value=session`` to drive it without a network.
"""

from unittest.mock import AsyncMock, Mock


def mock_rerank_session(response_json):
    """Return ``(session, captured)`` for one HTTP 200 rerank call.

    ``session.post(...)`` resolves, as an async context manager, to a response
    whose ``json()`` returns ``response_json``. ``captured`` is filled with the
    ``args`` and ``kwargs`` of that ``post`` call, so a test can assert on the
    URL (``captured["args"][0]``), the payload (``captured["kwargs"]["json"]``)
    and the headers (``captured["kwargs"]["headers"]``).
    """
    captured = {}
    mock_response = Mock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_json)
    mock_response.text = AsyncMock(return_value=str(response_json))
    mock_response.request_info = None
    mock_response.history = None
    mock_response.headers = {}
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    def capture_post(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return mock_response

    mock_session = Mock()
    mock_session.post = Mock(side_effect=capture_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    return mock_session, captured
