"""
XML namespace helpers for the OMML to LaTeX parser.
"""

ns_map = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def qn(tag):
    """
    A utility function to turn a namespace
    prefixed tag name into a Clark-notation qualified tag name for lxml. For
    example, qn('m:oMath') returns '{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath'

    :param tag:str - A namespace-prefixed tag name
    :return qn:str - A Clark-notation qualified name tag for lxml.
    """
    prefix, tag_root = tag.split(":")
    uri = ns_map[prefix]
    return "{{{}}}{}".format(uri, tag_root)
