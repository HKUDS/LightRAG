"""Opt-in smart heading discovery for the native docx parser.

Activated per file via the ``native(smart_heading=true)`` engine parameter.
The package is deliberately isolated: the smart-off parse path must not
import anything from here (its optional spaCy dependency would otherwise
become a hard dependency of every native parse).
"""
