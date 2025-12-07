"""
Embedding Service for the Core RAG System.

Handles text embedding using local BGE-M3 embedding server.
Supports both single text and batch embedding operations.
"""

import os
import requests
from typing import List
import time

# Configuration
EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:5001")


class EmbeddingService:
    """Service for generating text embeddings using local BGE-M3 server."""

    def __init__(self):
        """
        Initialize the embedding service.

        No API key needed - uses local embedding server.
        """
        self.server_url = EMBEDDING_SERVER_URL
        self.embed_endpoint = f"{self.server_url}/embed"

        # Verify server is running
        try:
            health_response = requests.get(f"{self.server_url}/health", timeout=5)
            if health_response.status_code == 200:
                print(f"✓ Connected to local embedding server at {self.server_url}")
            else:
                print(f"⚠ Warning: Embedding server returned status {health_response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠ Warning: Could not connect to embedding server at {self.server_url}")
            print(f"  Make sure embedserver.py is running on port 5001")
            print(f"  Error: {str(e)}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats (1024 dimensions for BGE-M3)
        """
        try:
            response = requests.post(
                self.embed_endpoint,
                json={"text": text},
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result["embedding"]

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get embedding from local server: {str(e)}")

    def embed_texts(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.

        Batches requests to improve efficiency.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch

        Returns:
            List of embeddings (each embedding is 1024 floats for BGE-M3)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Send batch request
                response = requests.post(
                    self.embed_endpoint,
                    json={"texts": batch},
                    timeout=60  # Longer timeout for batches
                )
                response.raise_for_status()

                result = response.json()
                batch_embeddings = result["embeddings"]
                all_embeddings.extend(batch_embeddings)

            except requests.exceptions.RequestException as e:
                raise Exception(f"Failed to get embeddings from local server: {str(e)}")

            # Small delay between batches
            if i + batch_size < len(texts):
                time.sleep(0.1)

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.

        This is semantically the same as embed_text but named
        differently for clarity in the retrieval pipeline.

        Args:
            query: Query text to embed

        Returns:
            List of floats (1024 dimensions for BGE-M3)
        """
        return self.embed_text(query)
