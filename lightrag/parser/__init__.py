"""Document parsing layer.

Sub-packages and modules:

- ``routing``: parser engine selection and per-file parser directives.
- ``debug``: minimal ``LightRAG`` stand-in for offline parser debugging.
- ``cli``: ``python -m lightrag.parser.cli`` entry point for single-file
  parser debugging across all engines.
- ``docx``: native ``.docx`` parser. Additional native format parsers
  should live as sibling sub-packages here (e.g. ``parser/pdf/``).
- ``external``: adapters for external parsing services (``mineru``,
  ``docling``) that post to a remote API and cache the raw bundle.
"""
