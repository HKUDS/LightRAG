"""Tests for remove_think_tags utility function.

Covers the fix for issue #2895: responses truncated when retrieved chunks
contain <think> tags.
"""

try:
    from lightrag.utils import remove_think_tags
except ImportError:
    # Fallback for environments without full lightrag dependencies (numpy,
    # httpx, etc.).  CI should always use the real import above.
    import re

    def remove_think_tags(text: str) -> str:  # type: ignore[misc]
        text = re.sub(r"^((?!<think>).)*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()


class TestRemoveThinkTags:
    """Test cases for the remove_think_tags function."""

    def test_standard_think_block_at_start(self):
        """Model reasoning wrapped in <think> at the beginning is removed."""
        text = "<think>Let me reason about this...</think>The actual answer."
        assert remove_think_tags(text) == "The actual answer."

    def test_orphaned_close_tag_at_start(self):
        """Streaming scenario: partial reasoning ending with </think>."""
        text = "partial reasoning content</think>The actual answer."
        assert remove_think_tags(text) == "The actual answer."

    def test_no_think_tags(self):
        """Text without any think tags is returned unchanged (stripped)."""
        text = "Just a normal response with no tags."
        assert remove_think_tags(text) == "Just a normal response with no tags."

    def test_think_tags_in_middle_of_text(self):
        """Bug #2895: <think> tags embedded in retrieved chunks must not
        truncate the surrounding content."""
        text = "Answer about xxx<think>reasoning</think>xxx more content"
        assert remove_think_tags(text) == "Answer about xxxxxx more content"

    def test_multiple_think_blocks(self):
        """Multiple <think> blocks are all removed."""
        text = "<think>r1</think>Answer<think>r2</think> more"
        assert remove_think_tags(text) == "Answer more"

    def test_multiline_think_block(self):
        """Think blocks spanning multiple lines are fully removed."""
        text = "<think>line1\nline2\nline3</think>Result."
        assert remove_think_tags(text) == "Result."

    def test_empty_think_block(self):
        """Empty think block is removed."""
        text = "<think></think>Content."
        assert remove_think_tags(text) == "Content."

    def test_only_think_block(self):
        """Text that is only a think block returns empty string."""
        text = "<think>only reasoning</think>"
        assert remove_think_tags(text) == ""

    def test_whitespace_stripping(self):
        """Leading/trailing whitespace is stripped after tag removal."""
        text = "  <think>reasoning</think>  Answer  "
        assert remove_think_tags(text) == "Answer"

    def test_xpath_like_content_preserved(self):
        """XPath-like strings that happen to be near think tags are preserved."""
        text = (
            "<think>thinking</think>"
            "The path is .//postTransactionAmounts/sharesOwnedFollowingTransaction/value"
        )
        assert (
            remove_think_tags(text)
            == "The path is .//postTransactionAmounts/sharesOwnedFollowingTransaction/value"
        )

    def test_orphaned_close_tag_with_angle_brackets_in_content(self):
        """Orphaned </think> prefix containing '<' chars is still removed."""
        text = "2 < 3 reasoning</think>final answer"
        assert remove_think_tags(text) == "final answer"

    def test_orphaned_close_tag_with_html_in_content(self):
        """Orphaned prefix with HTML/XML-like content is fully removed."""
        text = "check <b>bold</b> reasoning</think>The answer."
        assert remove_think_tags(text) == "The answer."
