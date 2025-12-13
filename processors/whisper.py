"""Groq Whisper transcription service for audio files."""

import os
from groq import Groq

WHISPER_MAX_FILE_SIZE = 19 * 1024 * 1024  # 19MB hard limit

_client = None


def get_client():
    """Get or create Groq client."""
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    return _client


def transcribe_audio(file_path):
    """
    Transcribe audio file using Groq Whisper.

    Args:
        file_path: Path to audio file

    Returns:
        {
            "text": str,
            "duration_seconds": float,
            "language": str
        }
    """
    file_size = os.path.getsize(file_path)

    if file_size > WHISPER_MAX_FILE_SIZE:
        raise Exception(f"Audio file too large. Max size: 19MB, got: {file_size / (1024*1024):.1f}MB")

    client = get_client()

    with open(file_path, 'rb') as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), audio_file),
            model="whisper-large-v3-turbo",
            response_format="verbose_json"
        )

    duration_seconds = getattr(transcription, 'duration', 0)

    return {
        "text": transcription.text,
        "duration_seconds": duration_seconds,
        "language": getattr(transcription, 'language', 'unknown')
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


def transcribe_audio_to_markdown(file_path, original_filename=None):
    """
    Transcribe audio and format as markdown.

    Args:
        file_path: Path to audio file
        original_filename: Original filename for header

    Returns:
        (markdown_text, duration_seconds)
    """
    result = transcribe_audio(file_path)

    filename = original_filename or os.path.basename(file_path)
    duration_str = format_duration(result['duration_seconds'])

    md = f"""# Audio Transcription

**File:** {filename}
**Duration:** {duration_str}
**Language:** {result['language']}

---

{result['text']}
"""

    return md, result['duration_seconds']
