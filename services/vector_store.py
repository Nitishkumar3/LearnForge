"""Vector store using ChromaDB."""

import os
import chromadb
from chromadb.config import Settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
COLLECTION_NAME = "documents"

db_client = None
collection = None

def get_collection():
    global db_client, collection
    if collection is None:
        db_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = db_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return collection

def add_chunks(chunks, embeddings, metadatas, document_id):
    ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    get_collection().add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)

def query(query_embedding, n_results=10, filters=None):
    params = {"query_embeddings": [query_embedding], "n_results": n_results}

    if filters:
        if len(filters) == 1:
            params["where"] = filters
        else:
            params["where"] = {"$and": [{k: v} for k, v in filters.items()]}

    results = get_collection().query(**params)

    return {
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
        "ids": results["ids"][0] if results["ids"] else []
    }

def delete_document(document_id):
    try:
        results = get_collection().get(where={"document_id": document_id})
        if results["ids"]:
            get_collection().delete(ids=results["ids"])
        return True
    except Exception as e:
        print(f"Error deleting document {document_id}: {e}")
        return False

def get_document_chunks(document_id):
    results = get_collection().get(where={"document_id": document_id}, include=["documents", "metadatas"])
    chunks = []
    for i, doc in enumerate(results["documents"]):
        chunks.append({
            "text": doc,
            "metadata": results["metadatas"][i] if results["metadatas"] else {}
        })
    return chunks

def get_all_document_ids():
    try:
        results = get_collection().get(include=["metadatas"])
        if not results["metadatas"]:
            return []
        doc_ids = set()
        for meta in results["metadatas"]:
            if "document_id" in meta:
                doc_ids.add(meta["document_id"])
        return list(doc_ids)
    except Exception as e:
        print(f"Error getting document IDs: {e}")
        return []

def count():
    return get_collection().count()

def clear_all():
    global collection
    try:
        db_client.delete_collection(COLLECTION_NAME)
        collection = db_client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        return True
    except Exception as e:
        print(f"Error clearing collection: {e}")
        return False

def delete_by_metadata(filters):
    """Delete chunks matching metadata filters."""
    try:
        if len(filters) == 1:
            where_clause = filters
        else:
            where_clause = {"$and": [{k: v} for k, v in filters.items()]}

        results = get_collection().get(where=where_clause)
        if results["ids"]:
            get_collection().delete(ids=results["ids"])
            return len(results["ids"])
        return 0
    except Exception as e:
        print(f"Error deleting by metadata: {e}")
        return 0