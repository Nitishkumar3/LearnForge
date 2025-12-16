"""Gemini Flash OCR service for image-based text extraction."""

import os
from google import genai
from google.genai import types

client = None

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
    global client
    if client is None:
        client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    return client

def extract_text_from_pdf(file_path):
    """Extract text from PDF using Gemini Flash with file upload."""
    client = get_client()
    uploaded_file = client.files.upload(file=file_path)

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

    config = types.GenerateContentConfig(temperature=1.0, max_output_tokens=65536,)
    response = client.models.generate_content(model=MODEL, contents=contents, config=config,)

    # Clean up uploaded file
    try:
        client.files.delete(name=uploaded_file.name)
    except:
        pass

    return response.text

def extract_text_from_image(image_bytes, mime_type='image/png'):
    """Extract text from image using Gemini Flash vision."""
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

    config = types.GenerateContentConfig(temperature=1.0, max_output_tokens=65536,)
    response = client.models.generate_content(model=MODEL, contents=contents, config=config,)
    return response.text