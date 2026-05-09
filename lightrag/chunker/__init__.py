"""LightRAG chunking strategies.

Two contracts coexist intentionally:

  - **Legacy contract** — :func:`chunking_by_token_size` keeps its
    historical 6-positional-arg signature

        ``(tokenizer, content, split_by_character,
            split_by_character_only, chunk_overlap_token_size,
            chunk_token_size)``

    so externally-supplied :attr:`lightrag.LightRAG.chunking_func`
    implementations continue to work unchanged. The legacy contract is
    only invoked for the **non-file text insert path** (e.g.
    :meth:`LightRAG.ainsert` with raw text), preserving the historical
    customization point.

  - **File-chunker contract** — for documents that went through the
    native parser (``parse_format == "lightrag"``), the file-based
    dispatcher in ``_PipelineMixin._process_single_document`` reads
    ``doc_process_opts.chunking`` and routes to a chunker following the
    standardized signature

        ``(tokenizer, content, chunk_token_size, *,
            <strategy-specific kwargs>)``

    Currently shipped file chunkers:

      - :func:`chunking_by_fixed_token` — the ``"F"`` strategy (same
        algorithm as :func:`chunking_by_token_size`, surfaced under the
        new contract).
      - :func:`chunking_by_paragraph_semantic` — the ``"P"`` strategy
        (heading-aware semantic chunker; consumes the docx-native
        ``.blocks.jsonl`` sidecar).

See ``docs/ParagraphSemanticChunking-zh.md`` for the algorithm behind
the ``"P"`` strategy.
"""

from lightrag.chunker.paragraph_semantic import chunking_by_paragraph_semantic
from lightrag.chunker.token_size import (
    chunking_by_fixed_token,
    chunking_by_token_size,
)

__all__ = [
    "chunking_by_fixed_token",
    "chunking_by_paragraph_semantic",
    "chunking_by_token_size",
]
