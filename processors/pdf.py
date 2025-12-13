"""PDF processing utilities using PyMuPDF with OCR fallback."""

import os
import pymupdf


def clean_text(text):
    """Clean extracted text."""
    text = text.replace('\x00', '')
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = ' '.join(line.split())
        if line:
            cleaned.append(line)
    return '\n'.join(cleaned)


def get_page_image_ratio(page):
    """Calculate ratio of image area to page area."""
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    if page_area == 0:
        return 0

    image_area = 0
    for img in page.get_images():
        try:
            bbox = page.get_image_bbox(img)
            if bbox:
                image_area += bbox.width * bbox.height
        except:
            pass

    return image_area / page_area


def should_use_ocr(doc, use_ocr=None):
    """
    Determine if OCR should be used.

    Args:
        doc: PyMuPDF document
        use_ocr: None (auto), True (force), False (skip)

    Returns:
        (should_ocr, reason)
    """
    if use_ocr is True:
        return True, "User requested OCR"

    if use_ocr is False:
        return False, "OCR disabled by user"

    total_chars = 0
    total_pages = len(doc)
    high_image_pages = 0

    for page in doc:
        text = page.get_text()
        total_chars += len(text.strip())

        img_ratio = get_page_image_ratio(page)
        if img_ratio > 0.8:
            high_image_pages += 1

    chars_per_page = total_chars / total_pages if total_pages > 0 else 0

    if chars_per_page < 100:
        return True, f"Low text density ({chars_per_page:.0f} chars/page)"

    if high_image_pages / total_pages > 0.8:
        return True, f"Image-heavy PDF ({high_image_pages}/{total_pages} pages with 80%+ images)"

    return False, "Sufficient text content"


def extract_text_direct(doc):
    """Extract text directly from PDF."""
    pages = []
    text_parts = []

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        page_text = clean_text(page_text)

        pages.append({
            "page_number": page_num,
            "text": page_text,
            "char_count": len(page_text)
        })

        if page_text.strip():
            text_parts.append(f"[Page {page_num}]\n{page_text}")

    return '\n\n'.join(text_parts), pages


def extract_text_with_ocr(file_path, num_pages):
    """Extract text using Gemini OCR with single file upload."""
    from processors import gemini_ocr

    print(f"[PDF] Starting OCR extraction for {num_pages} pages...")

    # Use single file upload for entire PDF
    full_text = gemini_ocr.extract_text_from_pdf(file_path)
    print(f"[PDF] OCR returned {len(full_text)} chars")

    full_text = clean_text(full_text)
    print(f"[PDF] After cleaning: {len(full_text)} chars")

    # Create a single page entry for metadata
    pages = [{
        "page_number": 1,
        "text": full_text,
        "char_count": len(full_text)
    }]

    return full_text, pages


def process(file_path, use_ocr=None):
    """
    Extract text from PDF and convert to Markdown.

    Args:
        file_path: Path to PDF file
        use_ocr: None (auto-detect), True (force OCR), False (direct only)

    Returns:
        {
            "text": str,
            "file_size": int,
            "num_pages": int,
            "processing_method": str,
            "metadata": dict
        }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        doc = pymupdf.open(file_path)
    except Exception as e:
        raise ValueError(f"Invalid PDF file: {e}")

    file_size = os.path.getsize(file_path)
    num_pages = len(doc)

    should_ocr, ocr_reason = should_use_ocr(doc, use_ocr)
    doc.close()

    if should_ocr:
        full_text, pages = extract_text_with_ocr(file_path, num_pages)
        processing_method = 'ocr'
    else:
        doc = pymupdf.open(file_path)
        full_text, pages = extract_text_direct(doc)
        doc.close()
        processing_method = 'direct'

    if not full_text.strip():
        raise ValueError("Could not extract any text from PDF.")

    return {
        "text": full_text,
        "file_size": file_size,
        "num_pages": num_pages,
        "processing_method": processing_method,
        "metadata": {
            "ocr_reason": ocr_reason,
            "pages": pages,
            "extracted_pages": len([p for p in pages if p["char_count"] > 0])
        }
    }


def analyze_pdf(file_path):
    """
    Analyze PDF to determine if OCR is recommended.

    Args:
        file_path: Path to PDF

    Returns:
        {
            "num_pages": int,
            "ocr_recommended": bool,
            "ocr_reason": str,
            "text_density": float
        }
    """
    try:
        doc = pymupdf.open(file_path)
        num_pages = len(doc)
        should_ocr, reason = should_use_ocr(doc, None)

        total_chars = sum(len(page.get_text().strip()) for page in doc)
        text_density = total_chars / num_pages if num_pages > 0 else 0

        doc.close()

        return {
            "num_pages": num_pages,
            "ocr_recommended": should_ocr,
            "ocr_reason": reason,
            "text_density": text_density
        }
    except Exception as e:
        return {
            "num_pages": 0,
            "ocr_recommended": True,
            "ocr_reason": f"Analysis failed: {str(e)}",
            "text_density": 0
        }
