"""Rerank chunks must preserve source characters at token boundaries."""

import pytest

from lightrag.rerank import chunk_documents_for_rerank
from lightrag.utils import TiktokenTokenizer, TokenBudgetError

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "text", ["你好🙂世界" * 8, "🧑🏽‍💻" * 8, "日本語の文章です" * 8]
)
@pytest.mark.parametrize("overlap", [0, 2])
def test_rerank_chunks_preserve_unicode_and_token_budget(text, overlap):
    chunks, indices = chunk_documents_for_rerank(
        [text, "", "short"], max_tokens=4, overlap_tokens=overlap
    )
    tokenizer = TiktokenTokenizer()
    document_chunks = [chunk for chunk, index in zip(chunks, indices) if index == 0]
    assert all(chunk in text for chunk in document_chunks)
    assert all(len(tokenizer.encode(chunk)) <= 4 for chunk in document_chunks)
    assert chunks[-2:] == ["", "short"]
    assert indices[-2:] == [1, 2]
    if overlap == 0:
        assert "".join(document_chunks) == text


def test_rerank_rejects_budget_smaller_than_one_codepoint():
    assert len(TiktokenTokenizer().encode("🧑")) > 1
    with pytest.raises(TokenBudgetError):
        chunk_documents_for_rerank(["🧑" * 3], max_tokens=1, overlap_tokens=0)
