from __future__ import annotations

from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP


def split_relation_tokens(value: Any) -> list[str]:
    text = str(value or "")
    for separator in (
        GRAPH_FIELD_SEP,
        "<SEP>",
        ",",
        "\uff0c",
        ";",
        "\uff1b",
        "|",
        "/",
        "\n",
        "\r",
    ):
        text = text.replace(separator, "\n")
    return [token.strip().casefold() for token in text.splitlines() if token.strip()]
