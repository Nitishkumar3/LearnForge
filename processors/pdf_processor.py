"""PDF processing utilities."""

import os
from PyPDF2 import PdfReader


def clean_text(text):
    text = text.replace('\x00', '')
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = ' '.join(line.split())
        if line:
            cleaned.append(line)
    return '\n'.join(cleaned)


def process(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise ValueError(f"Invalid PDF file: {e}")

    pages = []
    text_parts = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            print(f"Error extracting page {page_num}: {e}")
            page_text = ""

        page_text = clean_text(page_text)

        pages.append({
            "page_number": page_num,
            "text": page_text,
            "char_count": len(page_text)
        })

        if page_text.strip():
            text_parts.append(f"[Page {page_num}]\n{page_text}")

    full_text = "\n\n".join(text_parts)

    if not full_text.strip():
        raise ValueError("Could not extract any text from PDF. The PDF might be scanned or image-based.")

    return {
        "text": full_text,
        "total_pages": len(reader.pages),
        "pages": pages,
        "file_size": os.path.getsize(file_path),
        "extracted_pages": len([p for p in pages if p["char_count"] > 0])
    }


def get_page_count(file_path):
    try:
        reader = PdfReader(file_path)
        return len(reader.pages)
    except Exception as e:
        print(f"Error getting page count: {e}")
        return 0


def extract_page(file_path, page_number):
    try:
        reader = PdfReader(file_path)
        if 1 <= page_number <= len(reader.pages):
            page = reader.pages[page_number - 1]
            return clean_text(page.extract_text() or "")
        return ""
    except Exception as e:
        print(f"Error extracting page {page_number}: {e}")
        return ""
