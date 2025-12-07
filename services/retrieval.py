"""
Retrieval Service for the Core RAG System.

Simple retrieval - get top K most similar chunks for a query.
"""

import os
from typing import Dict, Any, List

# Configuration
TOP_K = int(os.getenv("TOP_K", "15"))


def retrieve(vector_store, embedding_service, query: str) -> Dict[str, Any]:
    """
    Retrieve relevant chunks for a query.

    Args:
        vector_store: Vector store instance
        embedding_service: Embedding service instance
        query: User query

    Returns:
        Dict with chunks, metadatas, and total_retrieved
    """
    # Get query embedding
    query_embedding = embedding_service.embed_query(query)

    # Query vector store
    results = vector_store.query(
        query_embedding=query_embedding,
        n_results=TOP_K
    )

    # Deduplicate overlapping chunks
    chunks = results["documents"]
    metadatas = results["metadatas"]

    unique_chunks, unique_metadatas = _deduplicate(chunks, metadatas)

    return {
        "chunks": unique_chunks,
        "metadatas": unique_metadatas,
        "total_retrieved": len(unique_chunks)
    }


def _deduplicate(chunks: List[str], metadatas: List[Dict]) -> tuple:
    """
    Remove duplicate/overlapping chunks.

    Uses first 100 characters as fingerprint to detect duplicates.
    """
    seen = set()
    unique_chunks = []
    unique_metadatas = []

    for chunk, meta in zip(chunks, metadatas):
        # Create fingerprint from first 100 chars
        fingerprint = chunk[:100].lower().strip().replace(' ', '').replace('\n', '')

        if fingerprint not in seen:
            seen.add(fingerprint)
            unique_chunks.append(chunk)
            unique_metadatas.append(meta)

    return unique_chunks, unique_metadatas
