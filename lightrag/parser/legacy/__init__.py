"""Legacy parser engine: simple in-process text extraction (no sidecar).

Produces ``raw``-format plain text.  The extraction helpers
(:func:`extract_text` and the per-format ``_extract_*`` functions) were moved
here from the API layer so the core parser owns them (the API layer imports
from here instead of the other way round).
"""

from lightrag.parser.legacy.extractors import (
    LegacyExtractionError,
    extract_text,
)

__all__ = ["LegacyExtractionError", "extract_text"]
