"""
Text Chunking Utility for the Core RAG System.

Handles splitting text into overlapping chunks with metadata.
Uses NLTK for TRUE sentence-aware chunking - never breaks mid-sentence.
"""

import os
import re
from typing import List, Dict, Any
from datetime import datetime

import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

# Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default")
DEFAULT_WORKSPACE_ID = os.getenv("DEFAULT_WORKSPACE_ID", "default")


class TextChunker:
    """Text chunking with sentence boundary awareness and metadata."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Characters per chunk (default from env)
            chunk_overlap: Overlap between chunks (default from env)
        """
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    def chunk_text(
        self,
        text: str,
        document_id: str,
        document_name: str,
        document_type: str = "pdf",
        user_id: str = None,
        workspace_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with rich metadata.

        Args:
            text: Full document text
            document_id: Unique document identifier
            document_name: Original filename
            document_type: Type of document (pdf, pptx, etc.)
            user_id: User ID (for multi-user, defaults to config)
            workspace_id: Workspace ID (for multi-workspace, defaults to config)

        Returns:
            List of chunk dicts with text and metadata
        """
        user_id = user_id or DEFAULT_USER_ID
        workspace_id = workspace_id or DEFAULT_WORKSPACE_ID

        # Clean the text
        text = self._clean_text(text)

        if not text.strip():
            return []

        # Split into chunks
        raw_chunks = self._split_text(text)

        # Build chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            # Try to extract page number from markers like [Page 5]
            page_number = self._extract_page_number(chunk_text)

            chunk = {
                "text": chunk_text,
                "metadata": {
                    "document_id": document_id,
                    "document_name": document_name,
                    "document_type": document_type,
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "page_number": page_number,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "user_id": user_id,
                    "workspace_id": workspace_id,
                    "created_at": datetime.now().isoformat()
                }
            }
            chunks.append(chunk)

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing excessive whitespace.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks with TRUE sentence awareness.
        Uses NLTK sentence tokenizer - never breaks mid-sentence.
        Always includes at least 1 sentence overlap for context continuity.
        """
        sentences = sent_tokenize(text)

        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            # Edge case: single sentence exceeds chunk_size
            if sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long sentence (rare)
                for i in range(0, sentence_length, self.chunk_size - self.chunk_overlap):
                    chunks.append(sentence[i:i + self.chunk_size])
                continue

            # Check if adding sentence exceeds chunk_size
            if current_length + sentence_length + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))

                # Calculate overlap: take last N chars worth of sentences
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) + 1 <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s) + 1
                    else:
                        break

                # IMPORTANT: Always include at least 1 sentence for context continuity
                # Even if it exceeds overlap limit, we need semantic connection
                if not overlap_sentences and current_chunk:
                    overlap_sentences = [current_chunk[-1]]

                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + (1 if current_chunk else 0)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _extract_page_number(self, chunk_text: str) -> int:
        """
        Extract page number from chunk text if present.

        Looks for markers like [Page 5] or [Page: 5]

        Args:
            chunk_text: Chunk text

        Returns:
            Page number or 0 if not found
        """
        # Match patterns like [Page 5], [Page: 5], [page 5]
        match = re.search(r'\[Page[:\s]*(\d+)\]', chunk_text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Match patterns like --- Page 5 ---
        match = re.search(r'---\s*Page\s*(\d+)\s*---', chunk_text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return 0

    def estimate_chunks(self, text_length: int) -> int:
        """
        Estimate number of chunks for a given text length.

        Args:
            text_length: Length of text in characters

        Returns:
            Estimated number of chunks
        """
        if text_length <= 0:
            return 0

        effective_step = self.chunk_size - self.chunk_overlap
        return max(1, (text_length + effective_step - 1) // effective_step)
