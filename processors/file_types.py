"""Document processors registry and utilities."""

import os
import hashlib

from processors import pdf
from processors import word
from processors import presentation
from processors import excel
from processors import image
from processors import audio


FILE_TYPE_MAP = {
    '.pdf': {'type': 'pdf', 'processor': pdf, 'ocr_available': True},
    '.docx': {'type': 'word', 'processor': word, 'ocr_available': False},
    '.doc': {'type': 'word', 'processor': word, 'ocr_available': False},
    '.pptx': {'type': 'presentation', 'processor': presentation, 'ocr_available': True},
    '.ppt': {'type': 'presentation', 'processor': presentation, 'ocr_available': True},
    '.xlsx': {'type': 'spreadsheet', 'processor': excel, 'ocr_available': False},
    '.xls': {'type': 'spreadsheet', 'processor': excel, 'ocr_available': False},
    '.csv': {'type': 'spreadsheet', 'processor': excel, 'ocr_available': False},
    '.png': {'type': 'image', 'processor': image, 'ocr_available': True, 'ocr_default': True},
    '.jpg': {'type': 'image', 'processor': image, 'ocr_available': True, 'ocr_default': True},
    '.jpeg': {'type': 'image', 'processor': image, 'ocr_available': True, 'ocr_default': True},
    '.gif': {'type': 'image', 'processor': image, 'ocr_available': True, 'ocr_default': True},
    '.bmp': {'type': 'image', 'processor': image, 'ocr_available': True, 'ocr_default': True},
    '.webp': {'type': 'image', 'processor': image, 'ocr_available': True, 'ocr_default': True},
    '.tiff': {'type': 'image', 'processor': image, 'ocr_available': True, 'ocr_default': True},
    '.tif': {'type': 'image', 'processor': image, 'ocr_available': True, 'ocr_default': True},
    '.mp3': {'type': 'audio', 'processor': audio, 'ocr_available': False},
    '.wav': {'type': 'audio', 'processor': audio, 'ocr_available': False},
    '.ogg': {'type': 'audio', 'processor': audio, 'ocr_available': False},
    '.m4a': {'type': 'audio', 'processor': audio, 'ocr_available': False},
    '.flac': {'type': 'audio', 'processor': audio, 'ocr_available': False},
    '.webm': {'type': 'audio', 'processor': audio, 'ocr_available': False},
    '.mp4': {'type': 'video', 'processor': audio, 'ocr_available': False, 'extract_audio': True},
    '.mov': {'type': 'video', 'processor': audio, 'ocr_available': False, 'extract_audio': True},
    '.avi': {'type': 'video', 'processor': audio, 'ocr_available': False, 'extract_audio': True},
    '.mkv': {'type': 'video', 'processor': audio, 'ocr_available': False, 'extract_audio': True},
}

MIME_TYPES = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/msword',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.ppt': 'application/vnd.ms-powerpoint',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.xls': 'application/vnd.ms-excel',
    '.csv': 'text/csv',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.webp': 'image/webp',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff',
    '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav',
    '.ogg': 'audio/ogg',
    '.m4a': 'audio/mp4',
    '.flac': 'audio/flac',
    '.webm': 'audio/webm',
    '.mp4': 'video/mp4',
    '.mov': 'video/quicktime',
    '.avi': 'video/x-msvideo',
    '.mkv': 'video/x-matroska',
}

ALLOWED_EXTENSIONS = set(FILE_TYPE_MAP.keys())


def get_extension(filename):
    """Get lowercase file extension."""
    return os.path.splitext(filename)[1].lower()


def is_allowed_file(filename):
    """Check if file extension is supported."""
    return get_extension(filename) in ALLOWED_EXTENSIONS


def get_file_type(filename):
    """Get file type category from filename."""
    ext = get_extension(filename)
    info = FILE_TYPE_MAP.get(ext)
    return info['type'] if info else None


def get_mime_type(filename):
    """Get MIME type from filename."""
    ext = get_extension(filename)
    return MIME_TYPES.get(ext, 'application/octet-stream')


def is_ocr_available(filename):
    """Check if OCR is available for file type."""
    ext = get_extension(filename)
    info = FILE_TYPE_MAP.get(ext)
    return info.get('ocr_available', False) if info else False


def is_ocr_default(filename):
    """Check if OCR is enabled by default for file type."""
    ext = get_extension(filename)
    info = FILE_TYPE_MAP.get(ext)
    return info.get('ocr_default', False) if info else False


def requires_audio_extraction(filename):
    """Check if file is video and needs audio extraction."""
    ext = get_extension(filename)
    info = FILE_TYPE_MAP.get(ext)
    return info.get('extract_audio', False) if info else False


def get_processor(filename):
    """Get processor module for file type."""
    ext = get_extension(filename)
    info = FILE_TYPE_MAP.get(ext)
    return info['processor'] if info else None


def calculate_file_hash(file_path):
    """Calculate SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def process_file(file_path, use_ocr=None):
    """
    Process file and extract text.

    Args:
        file_path: Path to file
        use_ocr: None (auto), True (force), False (skip)

    Returns:
        Processing result dict
    """
    processor = get_processor(file_path)
    if not processor:
        raise ValueError(f"Unsupported file type: {get_extension(file_path)}")

    return processor.process(file_path, use_ocr=use_ocr)


def analyze_file(file_path):
    """
    Analyze file to determine processing recommendations.

    Args:
        file_path: Path to file

    Returns:
        {
            "file_type": str,
            "mime_type": str,
            "ocr_available": bool,
            "ocr_recommended": bool,
            "ocr_reason": str,
            "num_pages": int
        }
    """
    ext = get_extension(file_path)
    info = FILE_TYPE_MAP.get(ext)

    if not info:
        return {
            "file_type": None,
            "mime_type": "application/octet-stream",
            "ocr_available": False,
            "ocr_recommended": False,
            "ocr_reason": "Unsupported file type",
            "num_pages": 0
        }

    file_type = info['type']
    processor = info['processor']

    analysis = {"num_pages": 1, "ocr_recommended": False, "ocr_reason": ""}

    if hasattr(processor, 'analyze_pdf'):
        analysis = processor.analyze_pdf(file_path)
    elif hasattr(processor, 'analyze_word'):
        analysis = processor.analyze_word(file_path)
    elif hasattr(processor, 'analyze_presentation'):
        analysis = processor.analyze_presentation(file_path)
    elif hasattr(processor, 'analyze_excel'):
        analysis = processor.analyze_excel(file_path)
    elif hasattr(processor, 'analyze_image'):
        analysis = processor.analyze_image(file_path)
    elif hasattr(processor, 'analyze_audio'):
        analysis = processor.analyze_audio(file_path)

    if info.get('ocr_default'):
        analysis['ocr_recommended'] = True
        analysis['ocr_reason'] = "Images always require OCR"

    return {
        "file_type": file_type,
        "mime_type": get_mime_type(file_path),
        "ocr_available": info.get('ocr_available', False),
        "ocr_recommended": analysis.get('ocr_recommended', False),
        "ocr_reason": analysis.get('ocr_reason', ''),
        "num_pages": analysis.get('num_pages', 1)
    }


def get_supported_formats():
    """Get list of supported file formats grouped by type."""
    formats = {}
    for ext, info in FILE_TYPE_MAP.items():
        file_type = info['type']
        if file_type not in formats:
            formats[file_type] = []
        formats[file_type].append(ext)
    return formats
