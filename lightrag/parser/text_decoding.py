"""Decode an uploaded text file's bytes, honouring a leading byte order mark.

Shared by the legacy text extractor and the native markdown parser so both
accept the same encodings and fail with the same message.
"""

from __future__ import annotations

import codecs

# Longest BOM first: the UTF-32 LE BOM (FF FE 00 00) starts with the UTF-16 LE
# BOM (FF FE), so testing UTF-16 first would decode a UTF-32 LE file as UTF-16
# with interleaved NULs instead of refusing or decoding it.
_BOM_ENCODINGS: tuple[tuple[bytes, str, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32", "UTF-32"),
    (codecs.BOM_UTF32_BE, "utf-32", "UTF-32"),
    (codecs.BOM_UTF8, "utf-8-sig", "UTF-8"),
    (codecs.BOM_UTF16_LE, "utf-16", "UTF-16"),
    (codecs.BOM_UTF16_BE, "utf-16", "UTF-16"),
)


class TextDecodeError(ValueError):
    """Raised when file bytes are not text in a supported encoding."""


def decode_text_bytes(data: bytes) -> str:
    """Decode ``data`` as UTF-8, or as UTF-16 / UTF-32 when it starts with a BOM.

    - The BOM selects the codec and is dropped from the result; a U+FEFF later
      in the text is content and is kept.
    - Without a BOM the bytes must be UTF-8. UTF-16 / UTF-32 without a BOM is
      not detected: guessing the byte order from the content is unreliable.
    - Raises :class:`TextDecodeError` with a message fit to show the user.
      Emptiness and other content checks are the caller's job.
    """
    codec, label = "utf-8", "UTF-8"
    for bom, bom_codec, bom_label in _BOM_ENCODINGS:
        if data.startswith(bom):
            codec, label = bom_codec, bom_label
            break
    try:
        return data.decode(codec)
    except UnicodeDecodeError as e:
        hint = (
            " (UTF-16 and UTF-32 files are accepted only with a byte order mark)"
            if label == "UTF-8"
            else ""
        )
        raise TextDecodeError(
            f"File is not valid {label} encoded text{hint}. Please convert it to "
            f"UTF-8 before processing: {e}"
        ) from e
