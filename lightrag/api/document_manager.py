"""Document manager for LightRAG API — per-workspace input-directory management.

Extracted from ``lightrag/api/routers/document_routes.py`` so that
``workspace_pool.py`` can import ``DocumentManager`` without creating a
circular import with the router module.
"""

from pathlib import Path
from typing import List

from lightrag.utils import logger, validate_workspace


class DocumentManager:
    def __init__(
        self,
        input_dir: str,
        workspace: str = "",  # New parameter for workspace isolation
    ):
        # Reject path traversal before using workspace in the upload path
        validate_workspace(workspace)
        # Store the base input directory and workspace
        self.base_input_dir = Path(input_dir)
        self.workspace = workspace
        self.indexed_files = set()

        # Create workspace-specific input directory
        # If workspace is provided, create a subdirectory for data isolation
        if workspace:
            self.input_dir = self.base_input_dir / workspace
        else:
            self.input_dir = self.base_input_dir

        # Create input directory if it doesn't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)

    @property
    def supported_extensions(self) -> tuple:
        """Suffixes accepted for an unhinted filename, derived live.

        A suffix is advertised only when it is *routable without extra
        directives*: the engine that ``resolve_file_parser_engine`` picks for
        a bare ``x.<suffix>`` (filename hint absent; ``LIGHTRAG_PARSER``
        rules + default apply) must itself support the suffix. This keeps
        "uploadable" aligned with "will actually parse": e.g. mineru's
        ``png`` joins only when its endpoint is configured AND a routing
        rule (or per-file hint, see ``is_supported_file``) sends pngs to it
        — otherwise the default ``legacy`` engine would fail the suffix gate
        at the parse stage. A default deployment equals the local engines'
        (legacy ∪ native) types; no hardcoded list to keep in sync.
        """
        from lightrag.parser.registry import available_engine_suffixes
        from lightrag.parser.routing import (
            parser_engine_supports_suffix,
            resolve_file_parser_engine,
        )

        out = []
        for s in sorted(available_engine_suffixes()):
            engine = resolve_file_parser_engine(f"x.{s}")
            if parser_engine_supports_suffix(engine, s):
                out.append(f".{s}")
        return tuple(out)

    def scan_directory_for_new_files(self) -> List[Path]:
        """Scan input directory for new, routable files.

        Globs over every *available* engine suffix (capability surface, so a
        hint-carrying file like ``img.[mineru].png`` is discoverable even
        when bare ``.png`` is not advertised), then keeps only files whose
        resolved engine actually supports them (``is_supported_file``).
        """
        from lightrag.parser.registry import available_engine_suffixes
        from lightrag.parser.routing import FilenameParserHintError

        new_files = []
        for s in sorted(available_engine_suffixes()):
            ext = f".{s}"
            logger.debug(f"Scanning for {ext} files in {self.input_dir}")
            for file_path in self.input_dir.glob(f"*{ext}"):
                if file_path in self.indexed_files:
                    continue
                try:
                    if not self.is_supported_file(file_path.name):
                        continue
                except FilenameParserHintError:
                    # Malformed hint: pass the file through — the enqueue
                    # path reports a detailed error document, instead of the
                    # scan silently ignoring the user's file.
                    pass
                new_files.append(file_path)
        return new_files

    def mark_as_indexed(self, file_path: Path):
        self.indexed_files.add(file_path)

    def is_supported_file(self, filename: str) -> bool:
        """True when THIS filename routes to an engine that can parse it.

        Resolves the engine for the concrete name — so a per-file hint
        (``img.[mineru].png``) is honoured — and checks the resolved engine
        supports the suffix. A bare suffix that would fall through to the
        default ``legacy`` engine is rejected here instead of failing later
        at the parse worker's suffix gate.

        Raises :class:`FilenameParserHintError` for a malformed hint —
        callers surface it (upload → HTTP 400 with the detailed message;
        scan passes the file through so enqueue emits an error document).
        """
        from lightrag.parser.routing import (
            parser_engine_supports_suffix,
            parser_suffix,
            resolve_file_parser_engine,
        )

        engine = resolve_file_parser_engine(filename)
        return parser_engine_supports_suffix(engine, parser_suffix(filename))
