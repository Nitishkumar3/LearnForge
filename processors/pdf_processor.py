"""
PDF Processor for the Core RAG System.

Handles PDF text extraction with page-level metadata.
"""

from PyPDF2 import PdfReader
from typing import Dict, Any, List
import os


class PDFProcessor:
    """PDF document processor with page-level extraction."""

    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Process a PDF file and extract text with page metadata.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dict with:
                - text: Full document text with page markers
                - total_pages: Number of pages
                - pages: List of {page_number, text, char_count}
                - file_size: File size in bytes

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid PDF or has no text
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            reader = PdfReader(file_path)
        except Exception as e:
            raise ValueError(f"Invalid PDF file: {e}")

        pages = []
        total_text_parts = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            # Clean the page text
            page_text = self._clean_text(page_text)

            pages.append({
                "page_number": page_num,
                "text": page_text,
                "char_count": len(page_text)
            })

            # Add page marker and text to full document
            if page_text.strip():
                total_text_parts.append(f"[Page {page_num}]\n{page_text}")

        # Combine all pages
        full_text = "\n\n".join(total_text_parts)

        if not full_text.strip():
            raise ValueError("Could not extract any text from PDF. The PDF might be scanned or image-based.")

        # Get file size
        file_size = os.path.getsize(file_path)

        return {
            "text": full_text,
            "total_pages": len(reader.pages),
            "pages": pages,
            "file_size": file_size,
            "extracted_pages": len([p for p in pages if p["char_count"] > 0])
        }

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove null characters
        text = text.replace('\x00', '')

        # Normalize whitespace
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Remove excessive spaces within line
            line = ' '.join(line.split())
            if line:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def get_page_count(self, file_path: str) -> int:
        """
        Get the page count of a PDF without full extraction.

        Args:
            file_path: Path to PDF file

        Returns:
            Number of pages
        """
        try:
            reader = PdfReader(file_path)
            return len(reader.pages)
        except Exception:
            return 0

    def extract_page(self, file_path: str, page_number: int) -> str:
        """
        Extract text from a specific page.

        Args:
            file_path: Path to PDF file
            page_number: Page number (1-indexed)

        Returns:
            Page text
        """
        try:
            reader = PdfReader(file_path)
            if 1 <= page_number <= len(reader.pages):
                page = reader.pages[page_number - 1]
                return self._clean_text(page.extract_text() or "")
            return ""
        except Exception:
            return ""
