from typing import TYPE_CHECKING, Any

from ._version import __version__ as __version__

__all__ = ["LightRAG", "QueryParam", "__version__"]

if TYPE_CHECKING:
    from .lightrag import LightRAG as LightRAG, QueryParam as QueryParam


def __getattr__(name: str) -> Any:
    if name in {"LightRAG", "QueryParam"}:
        from .lightrag import LightRAG, QueryParam

        value = {"LightRAG": LightRAG, "QueryParam": QueryParam}[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__author__ = "Zirui Guo"
__url__ = "https://github.com/HKUDS/LightRAG"
