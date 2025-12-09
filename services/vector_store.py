"""
Vector Store Service for the Core RAG System.

Handles persistent storage of document chunks and embeddings using ChromaDB.
Supports multi-document storage with metadata filtering.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import json
import os
from datetime import datetime

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
DOCUMENTS_REGISTRY = os.path.join(DATA_DIR, "documents.json")
COLLECTION_NAME = "documents"
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default")
DEFAULT_WORKSPACE_ID = os.getenv("DEFAULT_WORKSPACE_ID", "default")


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


class DocumentRegistry:
    """JSON-based document metadata registry."""

    def __init__(self, registry_path: str = None):
        """
        Initialize the document registry.

        Args:
            registry_path: Path to the JSON registry file
        """
        self.registry_path = registry_path or DOCUMENTS_REGISTRY
        self._ensure_registry()

    def _ensure_registry(self):
        """Create registry file if it doesn't exist."""
        if not os.path.exists(self.registry_path):
            self._save({
                "documents": [],
                "metadata": {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "last_updated": datetime.now().isoformat()
                }
            })

    def _load(self) -> Dict[str, Any]:
        """Load registry from file."""
        with open(self.registry_path, 'r') as f:
            return json.load(f)

    def _save(self, data: Dict[str, Any]):
        """Save registry to file."""
        data["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_document(self, doc_info: Dict[str, Any]) -> str:
        """
        Add a document to the registry.

        Args:
            doc_info: Document information dict

        Returns:
            Document ID
        """
        data = self._load()

        # Generate document ID if not provided
        if "id" not in doc_info:
            import uuid
            doc_info["id"] = f"doc_{uuid.uuid4().hex[:12]}"

        # Set defaults for multi-user fields
        doc_info.setdefault("user_id", DEFAULT_USER_ID)
        doc_info.setdefault("workspace_id", DEFAULT_WORKSPACE_ID)
        doc_info.setdefault("upload_time", datetime.now().isoformat())
        doc_info.setdefault("status", "pending")

        data["documents"].append(doc_info)
        data["metadata"]["total_documents"] = len(data["documents"])

        self._save(data)
        return doc_info["id"]

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document dict or None if not found
        """
        data = self._load()
        for doc in data["documents"]:
            if doc["id"] == document_id:
                return doc
        return None

    def list_documents(
        self,
        user_id: str = None,
        workspace_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        List all documents, optionally filtered.

        Args:
            user_id: Filter by user (for future multi-user)
            workspace_id: Filter by workspace (for future multi-workspace)

        Returns:
            List of document dicts
        """
        data = self._load()
        documents = data["documents"]

        # Apply filters if provided
        if user_id:
            documents = [d for d in documents if d.get("user_id") == user_id]
        if workspace_id:
            documents = [d for d in documents if d.get("workspace_id") == workspace_id]

        return documents

    def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a document's metadata.

        Args:
            document_id: Document identifier
            updates: Dict of fields to update

        Returns:
            True if successful
        """
        data = self._load()

        for i, doc in enumerate(data["documents"]):
            if doc["id"] == document_id:
                data["documents"][i].update(updates)
                self._save(data)
                return True

        return False

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the registry.

        Args:
            document_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        data = self._load()
        original_count = len(data["documents"])

        data["documents"] = [d for d in data["documents"] if d["id"] != document_id]

        if len(data["documents"]) < original_count:
            data["metadata"]["total_documents"] = len(data["documents"])
            self._save(data)
            return True

        return False

    def clear_all(self) -> bool:
        """
        Clear all documents from the registry.

        Returns:
            True if successful
        """
        self._save({
            "documents": [],
            "metadata": {
                "total_documents": 0,
                "total_chunks": 0,
                "last_updated": datetime.now().isoformat()
            }
        })
        return True

    def update_total_chunks(self, total_chunks: int):
        """Update the total chunks count in metadata."""
        data = self._load()
        data["metadata"]["total_chunks"] = total_chunks
        self._save(data)
