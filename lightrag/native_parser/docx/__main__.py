"""Allow ``python -m lightrag.native_parser.docx <file.docx>`` to invoke the
LightRAG-format debug CLI shipped in :mod:`lightrag_adapter`.

The CLI produces canonical LightRAG Document artifacts
(``.blocks.jsonl`` + sidecar JSONs + ``.blocks.assets/``) under
``./parse_output/<stem>.parsed/`` so the on-disk format can be inspected
directly without spinning up the full pipeline.
"""

from .lightrag_adapter import main


if __name__ == "__main__":
    main()
