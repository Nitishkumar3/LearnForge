"""Embedding utilities using local BGE-M3 server."""

import os
import time
import requests

SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:5001")
EMBED_ENDPOINT = f"{SERVER_URL}/embed"


def embed_text(text):
    """Generate embedding for a single text. Returns list of 1024 floats."""
    try:
        response = requests.post(EMBED_ENDPOINT, json={"text": text}, timeout=30)
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to get embedding: {str(e)}")


def embed_texts(texts, batch_size=10):
    """Generate embeddings for multiple texts with batching."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            response = requests.post(EMBED_ENDPOINT, json={"texts": batch}, timeout=60)
            response.raise_for_status()
            all_embeddings.extend(response.json()["embeddings"])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get embeddings: {str(e)}")

        if i + batch_size < len(texts):
            time.sleep(0.1)

    return all_embeddings


def embed_query(query):
    """Generate embedding for a query. Alias for embed_text."""
    return embed_text(query)
