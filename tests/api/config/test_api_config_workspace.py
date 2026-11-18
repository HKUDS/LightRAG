"""The server's ``WORKSPACE`` parsing: a reserved name is a clean refusal.

A leading ``$`` is reserved for internal configuration keys, so the server
refuses it rather than rewriting it -- and refuses it like every other bad
setting this parser rejects: one ``SystemExit`` line naming the setting,
never a traceback.
"""

from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = pytest.mark.offline


def _parse(monkeypatch, workspace: str):
    monkeypatch.setenv("WORKSPACE", workspace)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    config = importlib.import_module("lightrag.api.config")
    return config.parse_args()


@pytest.mark.parametrize("workspace", ["$prod", "$default"])
def test_a_reserved_workspace_exits_with_one_line(monkeypatch, workspace):
    with pytest.raises(SystemExit) as error:
        _parse(monkeypatch, workspace)
    message = str(error.value.code)
    assert message.startswith("WORKSPACE: ")
    assert "reserved" in message


def test_other_characters_are_still_rewritten(monkeypatch):
    assert _parse(monkeypatch, "team-a.prod").workspace == "team_a_prod"
