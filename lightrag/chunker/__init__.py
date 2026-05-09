"""LightRAG chunking strategies.

Two contracts coexist intentionally:

  - **Legacy contract** — :func:`chunking_by_token_size` keeps its
    historical 6-positional-arg signature

        ``(tokenizer, content, split_by_character,
            split_by_character_only, chunk_overlap_token_size,
            chunk_token_size)``

    so externally-supplied :attr:`lightrag.LightRAG.chunking_func`
    implementations continue to work unchanged. The legacy contract is
    only invoked when ``process_options`` does NOT specify a chunking
    selector (i.e. ``chunking_explicit`` is False) — typically direct
    :meth:`LightRAG.ainsert` calls with raw text.

  - **File-chunker contract** — for documents whose ``process_options``
    explicitly selects a chunking strategy, the file-based dispatcher in
    ``_PipelineMixin._process_single_document`` reads
    ``doc_process_opts.chunking`` and routes to a chunker following the
    standardized signature

        ``(tokenizer, content, chunk_token_size, *,
            <strategy-specific kwargs>)``

    Currently shipped file chunkers:

      - :func:`chunking_by_fixed_token` — the ``"F"`` strategy. Same
        algorithm as :func:`chunking_by_token_size`, surfaced under the
        new contract.
      - :func:`chunking_by_recursive_character` — the ``"R"`` strategy.
        Wraps LangChain ``RecursiveCharacterTextSplitter``; recursively
        splits on a separator cascade with token-aware sizing.
      - :func:`chunking_by_semantic_vector` — the ``"V"`` strategy.
        Wraps LangChain ``SemanticChunker``; sentence-level embedding
        similarity finds breakpoints. Async; needs an
        :class:`~lightrag.utils.EmbeddingFunc`.
      - :func:`chunking_by_paragraph_semantic` — the ``"P"`` strategy.
        Heading-aware semantic chunker; consumes the docx-native
        ``.blocks.jsonl`` sidecar. Falls back to R when the sidecar is
        missing or unreadable.

See ``docs/ParagraphSemanticChunking-zh.md`` for the algorithm behind
the ``"P"`` strategy and ``docs/FileProcessingConfiguration-zh.md`` for
how ``process_options`` and the new ``chunk_options`` snapshot drive
chunker selection per document.
"""

from lightrag.chunker.paragraph_semantic import chunking_by_paragraph_semantic
from lightrag.chunker.recursive_character import (
    chunking_by_recursive_character,
)
from lightrag.chunker.semantic_vector import chunking_by_semantic_vector
from lightrag.chunker.token_size import (
    chunking_by_fixed_token,
    chunking_by_token_size,
)

__all__ = [
    "chunking_by_fixed_token",
    "chunking_by_paragraph_semantic",
    "chunking_by_recursive_character",
    "chunking_by_semantic_vector",
    "chunking_by_token_size",
]
