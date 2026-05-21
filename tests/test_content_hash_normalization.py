"""Cross-filename content_hash dedup via merged_text normalization.

Sidecar-rendered bodies embed ``tb-<doc_hash>-NNNN`` / ``im-<doc_hash>-NNNN`` /
``eq-<doc_hash>-NNNN`` ids and ``path="<base>.blocks.assets/..."`` asset
references — both derive from the filename for pending_parse uploads.
``compute_text_content_hash`` normalizes those surfaces before hashing so
the same content under two filenames produces the same ``content_hash``
and post-parse dedup fires.
"""

from __future__ import annotations

from lightrag.utils_pipeline import (
    compute_text_content_hash,
    normalize_merged_text_for_hash,
)


def _render(doc_hash: str, base_name: str) -> str:
    """Approximate the sidecar writer's merged_text for a doc with one table,
    one drawing, and one block equation."""
    return (
        "标题 1\n\n"
        f'正文段落引用表格 <table id="tb-{doc_hash}-0001" format="json">[[]]</table>。\n\n'
        f'<drawing id="im-{doc_hash}-0001" format="png" '
        f'path="{base_name}.blocks.assets/image1.png" src="rId4" />\n\n'
        f'<equation id="eq-{doc_hash}-0001" format="latex">E=mc^2</equation>'
    )


def test_same_content_different_filename_dedupes():
    """Same merged_text with two different doc_hash / base names hashes to
    the same content_hash."""
    text_a = _render("a" * 32, "report-A")
    text_b = _render("b" * 32, "report-B")

    assert text_a != text_b, "sanity: raw bodies must differ"
    assert compute_text_content_hash(text_a) == compute_text_content_hash(text_b)


def test_different_content_still_distinguishes():
    """Distinct bodies (different block text) still produce distinct hashes
    after normalization."""
    text_a = _render("a" * 32, "doc-A") + "\n\n附加段落 X"
    text_b = _render("a" * 32, "doc-A") + "\n\n附加段落 Y"

    assert compute_text_content_hash(text_a) != compute_text_content_hash(text_b)


def test_plain_text_unaffected():
    """RAW text without sidecar markup is passed through unchanged so its
    hash matches the legacy ``MD5(text)`` value."""
    plain = "纯文本上传，没有任何 sidecar 标签。"
    assert normalize_merged_text_for_hash(plain) == plain


def test_asset_filename_still_distinguishes():
    """The asset filename suffix is preserved — only the ``<base>.blocks.assets/``
    prefix is stripped — so two drawings pointing at different images still
    yield different hashes."""
    h = "c" * 32
    text_a = (
        f'<drawing id="im-{h}-0001" format="png" '
        f'path="doc.blocks.assets/image1.png" src="r1" />'
    )
    text_b = (
        f'<drawing id="im-{h}-0001" format="png" '
        f'path="doc.blocks.assets/image2.png" src="r1" />'
    )
    assert compute_text_content_hash(text_a) != compute_text_content_hash(text_b)
