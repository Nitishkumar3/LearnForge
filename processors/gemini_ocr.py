"""Gemini Flash OCR service for image-based text extraction."""

import os
from google import genai
from google.genai import types

_client = None

MODEL = "gemini-flash-latest"
PROMPT = """Extract all text from this document and convert it into well-structured Markdown format.

Instructions:
1. **Structure the content intelligently**: Identify logical sections, topics, and subtopics. Create appropriate headings (# ## ###) based on content hierarchy, not page numbers.
2. **Format mathematical content**: Use LaTeX notation for equations and formulas (e.g., $y = mx + c$ for inline, $$equation$$ for block).
3. **Create proper lists**: Convert bullet points, numbered items, and enumerations into proper Markdown lists.
4. **Format tables**: If tabular data exists, use Markdown table syntax.
5. **Preserve all information**: Do not omit, summarize, or condense any content. Include everything but organize it logically.
6. **Clean up noise**: Fix OCR artifacts, merge broken words/sentences, but keep all actual content.
7. **No page markers**: Do not include "Page 1", "Page 2" etc. Treat the document as one continuous, well-organized text.

Output clean, readable, well-structured Markdown that a student would find easy to study from."""


def get_client():
    """Get or create Gemini client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    return _client


def extract_text_from_pdf(file_path):
    """
    Extract text from PDF using Gemini Flash with file upload.

    Args:
        file_path: Path to PDF file

    Returns:
        Extracted text as string
    """
    print(f"[OCR] Starting PDF extraction for: {file_path}")

    try:
        client = get_client()
        print(f"[OCR] Got Gemini client, uploading file...")

        # Upload entire PDF as a single file
        uploaded_file = client.files.upload(file=file_path)
        print(f"[OCR] File uploaded. URI: {uploaded_file.uri}")

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type="application/pdf",
                    ),
                    types.Part.from_text(text=PROMPT),
                ],
            ),
        ]

        config = types.GenerateContentConfig(
            temperature=1.0,
            max_output_tokens=65536,
        )

        print(f"[OCR] Calling Gemini API with model: {MODEL}")
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )
        print(f"[OCR] Got response, text length: {len(response.text) if response.text else 0}")
    except Exception as e:
        print(f"[OCR] ERROR: {type(e).__name__}: {e}")
        raise

    # Clean up uploaded file
    try:
        client.files.delete(name=uploaded_file.name)
    except:
        pass

    return response.text


def extract_text_from_image(image_bytes, mime_type='image/png'):
    """
    Extract text from image using Gemini Flash vision.

    Args:
        image_bytes: Raw image bytes
        mime_type: MIME type of image

    Returns:
        Extracted text as string
    """
    client = get_client()

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part.from_text(text=PROMPT),
            ],
        ),
    ]

    config = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=65536,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )

    return response.text


def extract_text_from_image_file(file_path):
    """
    Extract text from image file.

    Args:
        file_path: Path to image file

    Returns:
        Extracted text as string
    """
    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff'
    }
    mime_type = mime_map.get(ext, 'image/png')

    with open(file_path, 'rb') as f:
        image_bytes = f.read()

    return extract_text_from_image(image_bytes, mime_type)
