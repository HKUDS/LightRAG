"""
Simple tokenizer implementation for offline integration testing.

This tokenizer doesn't require internet access and provides a basic
word-based tokenization suitable for testing purposes.
"""

from typing import List
import re


class SimpleTokenizerImpl:
    """
    A simple word-based tokenizer that works offline.

    This tokenizer:
    - Splits text into words and punctuation
    - Doesn't require downloading any external files
    - Provides deterministic token IDs based on a vocabulary
    """

    def __init__(self):
        # Build a simple vocabulary for common tokens
        # This is a simplified approach - real tokenizers have much larger vocabularies
        self.vocab = self._build_vocab()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token_id = len(self.vocab)

    def _build_vocab(self) -> dict:
        """Build a basic vocabulary of common tokens."""
        vocab = {}
        current_id = 0

        # Add common words and symbols
        common_tokens = [
            # Whitespace and punctuation
            " ",
            "\n",
            "\t",
            ".",
            ",",
            "!",
            "?",
            ";",
            ":",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            '"',
            "'",
            "-",
            "_",
            "/",
            "\\",
            "@",
            "#",
            "$",
            "%",
            "&",
            "*",
            "+",
            "=",
            # Common programming keywords (for C++ code)
            "class",
            "struct",
            "public",
            "private",
            "protected",
            "void",
            "int",
            "double",
            "float",
            "char",
            "bool",
            "if",
            "else",
            "for",
            "while",
            "return",
            "include",
            "namespace",
            "using",
            "const",
            "static",
            "virtual",
            "new",
            "delete",
            "this",
            "nullptr",
            "true",
            "false",
            # Common English words
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "from",
            "with",
            "by",
            "for",
            "of",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "can",
            "may",
            "might",
            "must",
            "not",
            "no",
            "yes",
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
        ]

        for token in common_tokens:
            vocab[token.lower()] = current_id
            current_id += 1

        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens (words and punctuation)."""
        # Simple pattern to split on whitespace and keep punctuation separate
        pattern = r"\w+|[^\w\s]"
        tokens = re.findall(pattern, text)
        return tokens

    def encode(self, content: str) -> List[int]:
        """
        Encode a string into a list of token IDs.

        Args:
            content: The string to encode.

        Returns:
            A list of integer token IDs.
        """
        if not content:
            return []

        tokens = self._tokenize(content)
        token_ids = []

        for token in tokens:
            token_lower = token.lower()
            if token_lower in self.vocab:
                token_ids.append(self.vocab[token_lower])
            else:
                # For unknown tokens, use a hash-based ID to be deterministic
                # Offset by vocab size to avoid collisions
                hash_id = abs(hash(token)) % 10000 + len(self.vocab)
                token_ids.append(hash_id)

        return token_ids

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs into a string.

        Args:
            tokens: The list of token IDs to decode.

        Returns:
            The decoded string.
        """
        if not tokens:
            return ""

        words = []
        for token_id in tokens:
            if token_id in self.inverse_vocab:
                words.append(self.inverse_vocab[token_id])
            else:
                # For unknown IDs, use a placeholder
                words.append(f"<unk_{token_id}>")

        # Simple reconstruction - join words with spaces
        # This is a simplification; real tokenizers preserve exact spacing
        return " ".join(words)


def create_simple_tokenizer():
    """
    Create a simple tokenizer for offline use.

    Returns:
        A Tokenizer instance using SimpleTokenizerImpl.
    """
    from lightrag.utils import Tokenizer

    return Tokenizer("simple-tokenizer", SimpleTokenizerImpl())
