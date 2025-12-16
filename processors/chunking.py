"""Text chunking with sentence awareness."""

import os
import re
from datetime import datetime

import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default")
DEFAULT_WORKSPACE_ID = os.getenv("DEFAULT_WORKSPACE_ID", "default")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def split_into_chunks(text, chunk_size=None, chunk_overlap=None):
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    sentences = sent_tokenize(text)
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_length = len(sentence)

        # Long sentence edge case
        if sentence_length > chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            for i in range(0, sentence_length, chunk_size - chunk_overlap):
                chunks.append(sentence[i:i + chunk_size])
            continue

        if current_length + sentence_length + 1 > chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            # Calculate overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) + 1 <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1
                else:
                    break

            if not overlap_sentences and current_chunk:
                overlap_sentences = [current_chunk[-1]]

            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
        else:
            current_chunk.append(sentence)
            current_length += sentence_length + (1 if current_chunk else 0)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def extract_page_number(text):
    match = re.search(r'\[Page[:\s]*(\d+)\]', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r'---\s*Page\s*(\d+)\s*---', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def chunk_text(text, document_id, document_name, document_type="pdf", user_id=None, workspace_id=None):
    user_id = user_id or DEFAULT_USER_ID
    workspace_id = workspace_id or DEFAULT_WORKSPACE_ID

    text = clean_text(text)
    if not text.strip():
        return []

    raw_chunks = split_into_chunks(text)

    chunks = []
    for i, chunk_content in enumerate(raw_chunks):
        page_number = extract_page_number(chunk_content)

        chunk = {
            "text": chunk_content,
            "metadata": {
                "document_id": document_id,
                "document_name": document_name,
                "document_type": document_type,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "page_number": page_number,
                "char_count": len(chunk_content),
                "word_count": len(chunk_content.split()),
                "user_id": user_id,
                "workspace_id": workspace_id,
                "created_at": datetime.now().isoformat()
            }
        }
        chunks.append(chunk)

    return chunks
