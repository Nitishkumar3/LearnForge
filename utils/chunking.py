"""
Text Chunking Utility for the Core RAG System.

Handles splitting text into overlapping chunks with metadata.
Attempts to break at sentence boundaries for better context.
"""

import os
import re
from typing import List, Dict, Any
from datetime import datetime

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
        Split text into overlapping chunks with sentence boundary awareness.

        Args:
            text: Full text to split

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary if not at end
            if end < len(text):
                # Look for sentence endings in the second half of chunk
                best_break = -1
                for sep in ['. ', '? ', '! ', '.\n', '?\n', '!\n', '\n\n', '\n']:
                    pos = chunk.rfind(sep)
                    if pos > self.chunk_size // 2:  # Only if in second half
                        if pos > best_break:
                            best_break = pos + len(sep)

                if best_break > 0:
                    chunk = chunk[:best_break]
                    end = start + best_break

            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)

            # Move forward with overlap
            start = end - self.chunk_overlap

            # Prevent infinite loop
            if start <= 0 and end >= len(text):
                break
            if start < 0:
                start = 0

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
