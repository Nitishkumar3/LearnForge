"""
Vector Store Service for the Core RAG System.

Handles persistent storage of document chunks and embeddings using ChromaDB.
Supports multi-document storage with metadata filtering.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "documents"


class VectorStore:
    """Persistent vector store using ChromaDB."""

    def __init__(self):
        """Initialize the persistent ChromaDB client and collection."""
        # Use PersistentClient for data that survives restarts
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create the documents collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )

    def add_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        document_id: str
    ) -> int:
        """
        Add document chunks with embeddings to the vector store.

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts for each chunk
            document_id: Unique document identifier

        Returns:
            Number of chunks added
        """
        # Generate unique IDs for each chunk
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        return len(chunks)

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        filters: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar chunks.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            filters: Optional metadata filters (e.g., {"user_id": "user123", "workspace_id": "ws123"})

        Returns:
            Dict with 'documents', 'metadatas', 'distances', 'ids'
        """
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": n_results
        }

        # Add filters if provided - handle multiple filters with $and
        if filters:
            if len(filters) == 1:
                # Single filter, use directly
                query_params["where"] = filters
            else:
                # Multiple filters, use $and
                query_params["where"] = {
                    "$and": [{k: v} for k, v in filters.items()]
                }

        results = self.collection.query(**query_params)

        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks belonging to a document.

        Args:
            document_id: Document identifier

        Returns:
            True if deletion successful
        """
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )

            if results["ids"]:
                self.collection.delete(ids=results["ids"])

            return True
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.

        Args:
            document_id: Document identifier

        Returns:
            List of chunk dicts with text and metadata
        """
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )

        chunks = []
        for i, doc in enumerate(results["documents"]):
            chunks.append({
                "text": doc,
                "metadata": results["metadatas"][i] if results["metadatas"] else {}
            })

        return chunks

    def get_all_document_ids(self) -> List[str]:
        """
        Get unique document IDs in the store.

        Returns:
            List of unique document IDs
        """
        try:
            results = self.collection.get(include=["metadatas"])

            if not results["metadatas"]:
                return []

            # Extract unique document IDs
            doc_ids = set()
            for metadata in results["metadatas"]:
                if "document_id" in metadata:
                    doc_ids.add(metadata["document_id"])

            return list(doc_ids)
        except Exception:
            return []

    def count(self) -> int:
        """
        Get total number of chunks in the store.

        Returns:
            Number of chunks
        """
        return self.collection.count()

    def clear_all(self) -> bool:
        """
        Delete all data from the collection.

        Returns:
            True if successful
        """
        try:
            # Delete and recreate collection
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
