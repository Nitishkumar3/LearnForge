"""
Enhanced Retrieval Service for LearnForge RAG System.

Features:
- Hybrid Search (BM25 + Vector with RRF)
- Query Rewriting via LLM
- Reranking with score threshold
- Token-based chunk selection (not static TOP_K)
"""

import os
import requests
import tiktoken
from typing import Dict, Any, List, Optional
from rank_bm25 import BM25Okapi
import re

# Configuration
RERANK_SERVER_URL = os.getenv("RERANK_SERVER_URL", "http://localhost:5002")
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "7000"))

# Initial retrieval count (before reranking filters)
INITIAL_RETRIEVAL_K = 30

# Token counter (cl100k_base works for most models)
_tokenizer = tiktoken.get_encoding("cl100k_base")

# BM25 cache
_bm25_index: Optional[BM25Okapi] = None
_bm25_corpus: List[str] = []
_bm25_metadatas: List[Dict] = []
_corpus_hash: Optional[str] = None


def _count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(_tokenizer.encode(text))


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if len(t) > 2]


def _build_bm25_index(chunks: List[str], metadatas: List[Dict]) -> BM25Okapi:
    """Build BM25 index from chunks."""
    tokenized_corpus = [_tokenize(chunk) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def _get_bm25_index(vector_store) -> tuple:
    """Get or rebuild BM25 index."""
    global _bm25_index, _bm25_corpus, _bm25_metadatas, _corpus_hash

    current_count = vector_store.count()
    current_hash = f"count_{current_count}"

    if _corpus_hash != current_hash or _bm25_index is None:
        all_results = vector_store.collection.get(
            include=["documents", "metadatas"]
        )

        _bm25_corpus = all_results["documents"] if all_results["documents"] else []
        _bm25_metadatas = all_results["metadatas"] if all_results["metadatas"] else []

        if _bm25_corpus:
            _bm25_index = _build_bm25_index(_bm25_corpus, _bm25_metadatas)
        else:
            _bm25_index = None

        _corpus_hash = current_hash

    return _bm25_index, _bm25_corpus, _bm25_metadatas


def _bm25_search(query: str, vector_store, n_results: int) -> Dict[str, Any]:
    """Perform BM25 keyword search."""
    bm25_index, corpus, metadatas = _get_bm25_index(vector_store)

    if bm25_index is None or not corpus:
        return {"documents": [], "metadatas": [], "scores": []}

    query_tokens = _tokenize(query)
    scores = bm25_index.get_scores(query_tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    return {
        "documents": [corpus[i] for i in top_indices],
        "metadatas": [metadatas[i] for i in top_indices],
        "scores": [float(scores[i]) for i in top_indices]
    }


def _reciprocal_rank_fusion(
    vector_results: Dict[str, Any],
    bm25_results: Dict[str, Any],
    k: int = 60
) -> Dict[str, Any]:
    """Merge results using Reciprocal Rank Fusion (RRF)."""
    fused_scores = {}

    def fingerprint(doc: str) -> str:
        return doc[:100].lower().strip().replace(' ', '').replace('\n', '')

    for rank, (doc, meta) in enumerate(zip(
        vector_results["documents"],
        vector_results["metadatas"]
    )):
        fp = fingerprint(doc)
        if fp not in fused_scores:
            fused_scores[fp] = {"score": 0, "doc": doc, "metadata": meta}
        fused_scores[fp]["score"] += 1 / (k + rank + 1)

    for rank, (doc, meta) in enumerate(zip(
        bm25_results["documents"],
        bm25_results["metadatas"]
    )):
        fp = fingerprint(doc)
        if fp not in fused_scores:
            fused_scores[fp] = {"score": 0, "doc": doc, "metadata": meta}
        fused_scores[fp]["score"] += 1 / (k + rank + 1)

    sorted_results = sorted(
        fused_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return {
        "documents": [r["doc"] for r in sorted_results],
        "metadatas": [r["metadata"] for r in sorted_results],
        "scores": [r["score"] for r in sorted_results]
    }


def _rewrite_query(query: str, llm_manager) -> str:
    """Expand query using LLM for better retrieval."""
    try:
        prompt = f"""Rewrite this search query to improve document retrieval.
Add synonyms, related terms, and alternative phrasings.
Keep it concise (under 50 words). Output ONLY the rewritten query.

Original query: {query}

Rewritten query:"""

        expanded = llm_manager.generate(prompt, use_search=False, use_thinking=False)
        return f"{query} {expanded.strip()}"
    except Exception as e:
        print(f"Query rewrite failed: {e}")
        return query


def _rerank_and_select(
    query: str,
    chunks: List[str],
    metadatas: List[Dict]
) -> tuple:
    """
    Rerank chunks and select based on:
    1. Score threshold (filter out low-relevance chunks)
    2. Token limit (select chunks until MAX_CONTEXT_TOKENS)

    Edge case: If current tokens < limit, always add one more chunk
    even if it exceeds the limit (ensures we don't waste context space).

    Returns: (selected_chunks, selected_metadatas)
    """
    if not chunks:
        return [], []

    try:
        # Call rerank server
        response = requests.post(
            f"{RERANK_SERVER_URL}/rerank",
            json={"query": query, "documents": chunks},
            timeout=30
        )

        if response.status_code != 200:
            print(f"Rerank server error: {response.status_code}")
            return chunks[:10], metadatas[:10]  # Fallback

        results = response.json()["results"]

        # Step 1: Filter by score threshold
        filtered = [
            r for r in results
            if r["score"] >= RERANK_SCORE_THRESHOLD
        ]

        if not filtered:
            # If all filtered out, take top 3 regardless of score
            filtered = results[:3]

        # Step 2: Select chunks until token limit
        selected_chunks = []
        selected_metadatas = []
        total_tokens = 0

        for r in filtered:
            idx = r["index"]
            chunk_text = chunks[idx]
            chunk_tokens = _count_tokens(chunk_text)

            # If we haven't reached limit, add this chunk
            if total_tokens < MAX_CONTEXT_TOKENS:
                selected_chunks.append(chunk_text)
                selected_metadatas.append(metadatas[idx])
                total_tokens += chunk_tokens

                # Edge case: if we just went over limit, this is the last chunk
                # (we already added it, so we stop here)
                if total_tokens >= MAX_CONTEXT_TOKENS:
                    break
            else:
                break

        return selected_chunks, selected_metadatas

    except requests.exceptions.ConnectionError:
        print("Rerank server not available, using fallback selection")
        # Fallback: just take first chunks up to token limit
        return _select_by_tokens(chunks, metadatas)
    except Exception as e:
        print(f"Rerank failed: {e}")
        return _select_by_tokens(chunks, metadatas)


def _select_by_tokens(chunks: List[str], metadatas: List[Dict]) -> tuple:
    """Fallback: select chunks by token limit without reranking."""
    selected_chunks = []
    selected_metadatas = []
    total_tokens = 0

    for chunk, meta in zip(chunks, metadatas):
        chunk_tokens = _count_tokens(chunk)

        if total_tokens < MAX_CONTEXT_TOKENS:
            selected_chunks.append(chunk)
            selected_metadatas.append(meta)
            total_tokens += chunk_tokens

            if total_tokens >= MAX_CONTEXT_TOKENS:
                break
        else:
            break

    return selected_chunks, selected_metadatas


def _deduplicate(chunks: List[str], metadatas: List[Dict]) -> tuple:
    """Remove duplicate chunks using fingerprint."""
    seen = set()
    unique_chunks = []
    unique_metadatas = []

    for chunk, meta in zip(chunks, metadatas):
        fingerprint = chunk[:100].lower().strip().replace(' ', '').replace('\n', '')

        if fingerprint not in seen:
            seen.add(fingerprint)
            unique_chunks.append(chunk)
            unique_metadatas.append(meta)

    return unique_chunks, unique_metadatas


def retrieve(
    vector_store,
    embedding_service,
    query: str,
    llm_manager=None
) -> Dict[str, Any]:
    """
    Enhanced retrieval pipeline:

    1. Query Rewriting (LLM expands query)
    2. Hybrid Search (Vector + BM25 with RRF)
    3. Deduplication
    4. Reranking + Token-Based Selection
       - Filter by score threshold
       - Select chunks until token limit
       - Edge case: always add one more if under limit

    Args:
        vector_store: VectorStore instance
        embedding_service: EmbeddingService instance
        query: User query
        llm_manager: LLMManager for query rewriting

    Returns:
        Dict with chunks, metadatas, total_retrieved
    """
    # Step 1: Query Rewriting
    search_query = query
    if llm_manager is not None:
        search_query = _rewrite_query(query, llm_manager)

    # Step 2: Vector Search
    query_embedding = embedding_service.embed_query(search_query)
    vector_results = vector_store.query(
        query_embedding=query_embedding,
        n_results=INITIAL_RETRIEVAL_K
    )

    # Step 3: BM25 Search + Fusion
    bm25_results = _bm25_search(search_query, vector_store, INITIAL_RETRIEVAL_K)

    fused_results = _reciprocal_rank_fusion(
        {"documents": vector_results["documents"], "metadatas": vector_results["metadatas"]},
        bm25_results
    )

    chunks = fused_results["documents"]
    metadatas = fused_results["metadatas"]

    # Step 4: Deduplicate
    chunks, metadatas = _deduplicate(chunks, metadatas)

    # Step 5: Rerank + Token-Based Selection
    chunks, metadatas = _rerank_and_select(query, chunks, metadatas)

    return {
        "chunks": chunks,
        "metadatas": metadatas,
        "total_retrieved": len(chunks)
    }


def invalidate_bm25_cache():
    """Call when documents change to force BM25 rebuild."""
    global _corpus_hash
    _corpus_hash = None
