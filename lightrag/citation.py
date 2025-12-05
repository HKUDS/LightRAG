"""
Citation extraction and footnote generation for LightRAG.

This module provides post-processing capabilities to extract citations from LLM responses
by matching response sentences to source chunks using embedding similarity.
"""

import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Configuration
CITATION_MIN_SIMILARITY = float(os.getenv('CITATION_MIN_SIMILARITY', '0.5'))
CITATION_MAX_PER_SENTENCE = int(os.getenv('CITATION_MAX_PER_SENTENCE', '3'))


@dataclass
class CitationSpan:
    """Represents a span of text that should be attributed to a source."""

    start_char: int  # Start position in response text
    end_char: int  # End position in response text
    text: str  # The actual text span
    reference_ids: list[str]  # List of reference IDs supporting this claim
    confidence: float  # 0.0-1.0 confidence that this claim is supported


@dataclass
class SourceReference:
    """Enhanced reference with full metadata for footnotes."""

    reference_id: str
    file_path: str
    document_title: str | None = None
    section_title: str | None = None
    page_range: str | None = None
    excerpt: str | None = None
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class CitationResult:
    """Complete citation analysis result."""

    original_response: str  # Raw LLM response
    annotated_response: str  # Response with [n] markers inserted
    footnotes: list[str]  # Formatted footnote strings
    citations: list[CitationSpan]  # Detailed citation spans
    references: list[SourceReference]  # Enhanced reference list
    uncited_claims: list[str] = field(default_factory=list)  # Claims without sources


def extract_title_from_path(file_path: str) -> str:
    """Extract a human-readable title from a file path."""
    if not file_path:
        return 'Unknown Source'

    path = Path(file_path)
    # Get filename without extension
    name = path.stem

    # Convert snake_case or kebab-case to Title Case
    name = name.replace('_', ' ').replace('-', ' ')
    return name.title()


def split_into_sentences(text: str) -> list[dict[str, Any]]:
    """Split text into sentences with their positions.

    Returns list of dicts with:
        - text: The sentence text
        - start: Start character position
        - end: End character position
    """
    # Improved sentence splitting that handles common edge cases
    # Matches: .!? followed by space and capital letter, or end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'

    sentences = []
    current_pos = 0

    # Split on sentence boundaries
    parts = re.split(sentence_pattern, text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Find the actual position in original text
        start = text.find(part, current_pos)
        if start == -1:
            start = current_pos

        end = start + len(part)

        sentences.append({'text': part, 'start': start, 'end': end})

        current_pos = end

    return sentences


def compute_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


class CitationExtractor:
    """Post-processor to extract and format citations from LLM responses."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        references: list[dict[str, str]],
        embedding_func: Callable,
        min_similarity: float = CITATION_MIN_SIMILARITY,
    ):
        """Initialize the citation extractor.

        Args:
            chunks: List of chunk dictionaries with 'content', 'file_path', etc.
            references: List of reference dicts with 'reference_id', 'file_path'
            embedding_func: Async function to compute embeddings
            min_similarity: Minimum similarity threshold for citation matching
        """
        self.chunks = chunks
        self.references = references
        self.embedding_func = embedding_func
        self.min_similarity = min_similarity

        # Build lookup structures
        self._build_chunk_index()

    def _build_chunk_index(self):
        """Build index mapping chunk content to reference IDs."""
        self.chunk_to_ref: dict[str, str] = {}
        self.ref_to_chunks: dict[str, list[dict]] = {}
        self.path_to_ref: dict[str, str] = {}

        # Map file_path to reference_id
        for ref in self.references:
            path = ref.get('file_path', '')
            if path:
                self.path_to_ref[path] = ref.get('reference_id', '')

        # Index chunks by reference
        for chunk in self.chunks:
            file_path = chunk.get('file_path', '')
            ref_id = self.path_to_ref.get(file_path, '')

            if ref_id:
                chunk_id = chunk.get('id') or chunk.get('chunk_id') or chunk.get('content', '')[:100]
                self.chunk_to_ref[chunk_id] = ref_id

                if ref_id not in self.ref_to_chunks:
                    self.ref_to_chunks[ref_id] = []
                self.ref_to_chunks[ref_id].append(chunk)

    def _compute_content_overlap(self, sentence: str, chunk_content: str) -> float:
        """Compute a lexical overlap score to verify vector matches.

        Returns:
            float: 0.0 to 1.0 representing how much of the sentence's key terms
                   are present in the chunk.
        """

        # Simple tokenizer: lowercase and split by non-alphanumeric
        def tokenize(text):
            return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))

        sent_tokens = tokenize(sentence)
        if not sent_tokens:
            return 0.0

        chunk_tokens = tokenize(chunk_content)

        # Calculate overlap
        common = sent_tokens.intersection(chunk_tokens)
        return len(common) / len(sent_tokens)

    async def _find_supporting_chunks(self, sentence: str, sentence_embedding: list[float]) -> list[dict[str, Any]]:
        """Find chunks that support a given sentence.

        Args:
            sentence: The sentence text
            sentence_embedding: Pre-computed embedding for the sentence

        Returns:
            List of matches with reference_id and similarity score
        """
        matches = []

        for chunk in self.chunks:
            chunk_content = chunk.get('content', '')
            chunk_embedding = chunk.get('embedding')

            # Skip chunks without embeddings (handle both None and empty arrays)
            if chunk_embedding is None or (hasattr(chunk_embedding, '__len__') and len(chunk_embedding) == 0):
                continue

            # 1. Vector Similarity (The "Vibe" Check)
            vector_score = compute_similarity(sentence_embedding, chunk_embedding)

            # Quick filter
            if vector_score < self.min_similarity:
                continue

            # 2. Content Verification (The "Fact" Check)
            # We penalize the vector score if the actual words are missing.
            # This reduces "hallucinated" citations where the topic is same but facts differ.
            overlap_score = self._compute_content_overlap(sentence, chunk_content)

            # Weighted Score:
            # We trust the vector more (70%), but allow the overlap to boost/penalty (30%)
            # If overlap is 0, max score is ~0.7 * vector_score
            # If overlap is 1, score is full.
            final_score = (vector_score * 0.7) + (overlap_score * 0.3)

            if final_score >= self.min_similarity:
                file_path = chunk.get('file_path', '')
                ref_id = self.path_to_ref.get(file_path)

                if ref_id:
                    matches.append(
                        {
                            'reference_id': ref_id,
                            'similarity': final_score,
                            'chunk_excerpt': chunk_content[:100],
                        }
                    )

        # Sort by similarity and deduplicate by reference_id
        matches.sort(key=lambda x: x['similarity'], reverse=True)

        seen_refs = set()
        unique_matches = []
        for match in matches:
            if match['reference_id'] not in seen_refs:
                seen_refs.add(match['reference_id'])
                unique_matches.append(match)
                if len(unique_matches) >= CITATION_MAX_PER_SENTENCE:
                    break

        return unique_matches

    async def extract_citations(
        self, response: str, chunk_embeddings: dict[str, list[float]] | None = None
    ) -> CitationResult:
        """Extract citations by matching response sentences to source chunks.

        Algorithm:
        1. Split response into sentences
        2. For each sentence, compute embedding similarity to all chunks
        3. Assign reference_id from best-matching chunk(s) above threshold
        4. Generate inline markers and footnotes

        Args:
            response: The LLM response text
            chunk_embeddings: Optional pre-computed chunk embeddings keyed by chunk_id

        Returns:
            CitationResult with annotated response and footnotes
        """
        sentences = split_into_sentences(response)
        citations: list[CitationSpan] = []
        used_refs: set[str] = set()
        uncited_claims: list[str] = []

        # Compute embeddings for all sentences at once (batch)
        sentence_texts = [s['text'] for s in sentences]
        if sentence_texts:
            try:
                sentence_embeddings = await self.embedding_func(sentence_texts)
            except Exception as e:
                logger.warning(f'Failed to compute sentence embeddings: {e}')
                sentence_embeddings = [None] * len(sentence_texts)
        else:
            sentence_embeddings = []

        # Pre-compute or use provided chunk embeddings
        if chunk_embeddings is not None and len(chunk_embeddings) > 0:
            for chunk in self.chunks:
                chunk_id = chunk.get('id', chunk.get('content', '')[:50])
                if chunk_id in chunk_embeddings:
                    chunk['embedding'] = chunk_embeddings[chunk_id]
        else:
            # Compute chunk embeddings if not provided
            chunk_contents = [c.get('content', '') for c in self.chunks]
            if chunk_contents:
                try:
                    computed_embeddings = await self.embedding_func(chunk_contents)
                    for i, chunk in enumerate(self.chunks):
                        chunk['embedding'] = computed_embeddings[i]
                except Exception as e:
                    logger.warning(f'Failed to compute chunk embeddings: {e}')

        # Match sentences to chunks
        for i, sentence in enumerate(sentences):
            sentence_emb = sentence_embeddings[i] if i < len(sentence_embeddings) else None

            if sentence_emb is None:
                uncited_claims.append(sentence['text'])
                continue

            matches = await self._find_supporting_chunks(sentence['text'], sentence_emb)

            if matches:
                ref_ids = [m['reference_id'] for m in matches]
                confidence = matches[0]['similarity'] if matches else 0.0

                citations.append(
                    CitationSpan(
                        start_char=sentence['start'],
                        end_char=sentence['end'],
                        text=sentence['text'],
                        reference_ids=ref_ids,
                        confidence=confidence,
                    )
                )

                used_refs.update(ref_ids)
            else:
                uncited_claims.append(sentence['text'])

        # Generate annotated response with inline markers
        annotated = self._insert_citation_markers(response, citations)

        # Build enhanced references
        enhanced_refs = self._enhance_references(used_refs)

        # Format footnotes
        footnotes = self._format_footnotes(enhanced_refs)

        return CitationResult(
            original_response=response,
            annotated_response=annotated,
            footnotes=footnotes,
            citations=citations,
            references=enhanced_refs,
            uncited_claims=uncited_claims,
        )

    def _insert_citation_markers(self, response: str, citations: list[CitationSpan]) -> str:
        """Insert [n] citation markers into response text.

        Processes citations in reverse order to preserve character positions.
        """
        # Sort by position (descending) to insert from end to beginning
        sorted_citations = sorted(citations, key=lambda c: c.end_char, reverse=True)

        result = response
        for citation in sorted_citations:
            if not citation.reference_ids:
                continue

            # Create marker like [1] or [1,2] for multiple refs
            marker = '[' + ','.join(citation.reference_ids) + ']'

            # Insert marker after the sentence (at end_char position)
            result = result[: citation.end_char] + marker + result[citation.end_char :]

        return result

    def _enhance_references(self, used_refs: set[str]) -> list[SourceReference]:
        """Build enhanced reference objects with metadata."""
        enhanced = []

        for ref in self.references:
            ref_id = ref.get('reference_id', '')
            if ref_id not in used_refs:
                continue

            file_path = ref.get('file_path', '')
            chunks = self.ref_to_chunks.get(ref_id, [])

            # Extract first chunk as excerpt
            excerpt = None
            if chunks:
                first_chunk = chunks[0]
                content = first_chunk.get('content', '')
                excerpt = content[:150] + '...' if len(content) > 150 else content

            enhanced.append(
                SourceReference(
                    reference_id=ref_id,
                    file_path=file_path,
                    document_title=ref.get('document_title') or extract_title_from_path(file_path),
                    section_title=ref.get('section_title'),
                    page_range=ref.get('page_range'),
                    excerpt=excerpt,
                    chunk_ids=[c.get('id', '') for c in chunks if c.get('id')],
                )
            )

        return enhanced

    def _format_footnotes(self, references: list[SourceReference]) -> list[str]:
        """Format references as footnote strings.

        Format: [n] "Document Title", Section X, pp. Y-Z. "Excerpt..."
        """
        footnotes = []

        for ref in sorted(references, key=lambda r: int(r.reference_id or '0')):
            parts = [f'[{ref.reference_id}] "{ref.document_title}"']

            if ref.section_title:
                parts.append(f'Section: {ref.section_title}')

            if ref.page_range:
                parts.append(f'pp. {ref.page_range}')

            footnote = ', '.join(parts)

            if ref.excerpt:
                footnote += f'. "{ref.excerpt}"'

            footnotes.append(footnote)

        return footnotes


async def extract_citations_from_response(
    response: str,
    chunks: list[dict[str, Any]],
    references: list[dict[str, str]],
    embedding_func: Callable,
    min_similarity: float = CITATION_MIN_SIMILARITY,
) -> CitationResult:
    """Convenience function to extract citations from a response.

    Args:
        response: The LLM response text
        chunks: List of chunk dictionaries
        references: List of reference dicts
        embedding_func: Async function to compute embeddings
        min_similarity: Minimum similarity threshold

    Returns:
        CitationResult with annotated response and footnotes
    """
    extractor = CitationExtractor(
        chunks=chunks,
        references=references,
        embedding_func=embedding_func,
        min_similarity=min_similarity,
    )

    return await extractor.extract_citations(response)
