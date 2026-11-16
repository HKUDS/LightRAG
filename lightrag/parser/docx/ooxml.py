"""WordprocessingML value helpers shared by the native DOCX parser.

Kept free of python-docx and ElementTree imports so that both the
python-docx (lxml) code paths and the defusedxml readers can use it.
"""

from __future__ import annotations

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

_W_VAL = f"{{{W_NS}}}val"

# ST_OnOff values that switch a property off. The transitional schema also
# accepts "1" / "true" / "on" (and strict only "1" / "true" / "0" / "false");
# anything that is not one of these three spellings reads as on.
_OFF_VALUES = frozenset({"0", "false", "off"})


def is_on(elem) -> bool:
    """Whether an on/off property element (``<w:b/>``, ``<w:tblHeader/>`` ...) is on.

    Pass the element itself, after checking that it is present: an absent
    element means the property is not set here (inherited or off), which only
    the caller can decide. A present element without ``w:val`` is on, and so is
    any ``w:val`` other than ``"0"``, ``"false"`` or ``"off"``.

    Not for on/off *attributes* such as ``w:style/@w:default``: there an absent
    attribute means off.
    """
    return elem.get(_W_VAL) not in _OFF_VALUES
