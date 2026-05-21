"""
ABOUTME: OMML (Office Math Markup Language) to LaTeX conversion
"""

from .ommlparser import OMMLParser


def convert_omml_to_latex(omml_element) -> str:
    """Convert an m:oMath XML element to a LaTeX string."""
    return OMMLParser().parse(omml_element)
