"""The rendered timeout help must describe values the parser accepts."""

import sys

import pytest

from lightrag.api.config import parse_args


pytestmark = pytest.mark.offline


def test_timeout_help_does_not_advertise_none_or_infinite(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["lightrag-server", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    timeout_help = help_text.partition("\n  --timeout TIMEOUT")[2].split("\n  --", 1)[0]
    assert timeout_help, "the --timeout entry was not found in the help output"
    assert "None" not in timeout_help
    assert "infinite" not in timeout_help
