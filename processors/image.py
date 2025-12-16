"""Image processing utilities with OCR."""

import os
from PIL import Image
import io

def process(file_path, use_ocr=None):
    """Extract text from image using OCR."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    file_size = os.path.getsize(file_path)

    try:
        img = Image.open(file_path)
        width, height = img.size
        format_name = img.format or 'Unknown'

        if hasattr(img, 'n_frames') and img.n_frames > 1:
            return process_multiframe(file_path, img, file_size)

        img.close()
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")

    from processors import gemini_ocr

    with open(file_path, 'rb') as f:
        image_bytes = f.read()

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

    try:
        extracted_text = gemini_ocr.extract_text_from_image(image_bytes, mime_type)
    except Exception as e:
        raise ValueError(f"OCR failed: {str(e)}")

    full_text = f"""# Image Analysis

**Dimensions:** {width} x {height}
**Format:** {format_name}

---

## Extracted Content

{extracted_text}
"""

    return {
        "text": full_text,
        "file_size": file_size,
        "num_pages": 1,
        "processing_method": "ocr",
        "metadata": {
            "width": width,
            "height": height,
            "format": format_name
        }
    }

def process_multiframe(file_path, img, file_size):
    """Process multi-frame images (GIF, TIFF)."""
    from processors import gemini_ocr

    n_frames = img.n_frames
    width, height = img.size
    format_name = img.format or 'Unknown'

    md_parts = [f"# Multi-Page Image\n\n**Pages:** {n_frames}\n**Dimensions:** {width} x {height}\n\n---"]

    for frame_num in range(min(n_frames, 20)):
        img.seek(frame_num)

        buffer = io.BytesIO()
        frame = img.convert('RGB')
        frame.save(buffer, format='PNG')
        frame_bytes = buffer.getvalue()

        try:
            frame_text = gemini_ocr.extract_text_from_image(frame_bytes, 'image/png')
        except Exception as e:
            frame_text = f"[OCR Error: {str(e)}]"

        md_parts.append(f"## Page {frame_num + 1}\n\n{frame_text}")

    img.close()

    if n_frames > 20:
        md_parts.append(f"\n*Note: Only first 20 of {n_frames} pages processed*")

    full_text = '\n\n'.join(md_parts)

    return {
        "text": full_text,
        "file_size": file_size,
        "num_pages": n_frames,
        "processing_method": "ocr",
        "metadata": {
            "width": width,
            "height": height,
            "format": format_name,
            "frames": n_frames
        }
    }

def analyze_image(file_path):
    """Analyze image file for processing."""
    try:
        img = Image.open(file_path)
        n_frames = getattr(img, 'n_frames', 1)
        img.close()

        return {
            "num_pages": n_frames,
            "ocr_recommended": True,
            "ocr_reason": "Images always require OCR for text extraction"
        }
    except Exception as e:
        return {
            "num_pages": 1,
            "ocr_recommended": True,
            "ocr_reason": f"Analysis failed: {str(e)}"
        }