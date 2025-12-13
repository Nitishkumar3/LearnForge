"""Audio processing utilities with Whisper transcription."""

import os


def process(file_path, use_ocr=None):
    """
    Transcribe audio file to text.

    Args:
        file_path: Path to audio file
        use_ocr: Ignored (always uses Whisper for audio)

    Returns:
        {
            "text": str,
            "file_size": int,
            "num_pages": int,
            "processing_method": str,
            "metadata": dict,
            "duration_seconds": int
        }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    file_size = os.path.getsize(file_path)

    from processors import whisper

    try:
        md_text, duration = whisper.transcribe_audio_to_markdown(
            file_path,
            original_filename=os.path.basename(file_path)
        )
    except Exception as e:
        raise ValueError(f"Transcription failed: {str(e)}")

    return {
        "text": md_text,
        "file_size": file_size,
        "num_pages": 1,
        "processing_method": "transcription",
        "duration_seconds": int(duration),
        "metadata": {
            "duration_seconds": duration,
            "duration_formatted": format_duration(duration)
        }
    }


def format_duration(seconds):
    """Format seconds to human readable duration."""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def analyze_audio(file_path):
    """
    Analyze audio file.

    Args:
        file_path: Path to audio

    Returns:
        {
            "num_pages": int,
            "ocr_recommended": bool,
            "ocr_reason": str,
            "file_size_mb": float
        }
    """
    try:
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        max_size_mb = 19

        if file_size_mb > max_size_mb:
            return {
                "num_pages": 1,
                "ocr_recommended": False,
                "ocr_reason": f"File too large ({file_size_mb:.1f}MB > {max_size_mb}MB limit)",
                "file_size_mb": file_size_mb
            }

        return {
            "num_pages": 1,
            "ocr_recommended": False,
            "ocr_reason": "Audio uses Whisper transcription (not OCR)",
            "file_size_mb": file_size_mb
        }
    except Exception as e:
        return {
            "num_pages": 1,
            "ocr_recommended": False,
            "ocr_reason": f"Analysis failed: {str(e)}",
            "file_size_mb": 0
        }
