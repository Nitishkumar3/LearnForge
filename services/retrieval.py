"""Retrieval service with hybrid search (BM25 + Vector) and reranking."""

import os
import requests
import tiktoken
import re
from rank_bm25 import BM25Okapi

from services import vector_store
from services import embedding
from services import llm_providers as llm

RERANK_SERVER_URL = os.getenv("RERANK_SERVER_URL", "http://localhost:5002")
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "7000"))
INITIAL_RETRIEVAL_K = 30

tokenizer = tiktoken.get_encoding("cl100k_base")

# BM25 cache
bm25_index = None
bm25_corpus = []
bm25_metadatas = []
corpus_hash = None


def count_tokens(text):
    return len(tokenizer.encode(text))


def tokenize_for_bm25(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if len(t) > 2]


def build_bm25_index(chunks):
    tokenized = [tokenize_for_bm25(chunk) for chunk in chunks]
    return BM25Okapi(tokenized)


def get_bm25_index():
    global bm25_index, bm25_corpus, bm25_metadatas, corpus_hash

    current_count = vector_store.count()
    current_hash = f"count_{current_count}"

    if corpus_hash != current_hash or bm25_index is None:
        coll = vector_store.get_collection()
        all_results = coll.get(include=["documents", "metadatas"])

        bm25_corpus = all_results["documents"] if all_results["documents"] else []
        bm25_metadatas = all_results["metadatas"] if all_results["metadatas"] else []

        if bm25_corpus:
            bm25_index = build_bm25_index(bm25_corpus)
        else:
            bm25_index = None

        corpus_hash = current_hash

    return bm25_index, bm25_corpus, bm25_metadatas


def bm25_search(query, n_results, user_id=None, workspace_id=None):
    index, corpus, metadatas = get_bm25_index()

    if index is None or not corpus:
        return {"documents": [], "metadatas": [], "scores": []}

    query_tokens = tokenize_for_bm25(query)
    scores = index.get_scores(query_tokens)

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    docs = []
    metas = []
    score_list = []

    for i in sorted_indices:
        meta = metadatas[i]
        if user_id and meta.get("user_id") != user_id:
            continue
        if workspace_id and meta.get("workspace_id") != workspace_id:
            continue

        docs.append(corpus[i])
        metas.append(meta)
        score_list.append(float(scores[i]))

        if len(docs) >= n_results:
            break

    return {"documents": docs, "metadatas": metas, "scores": score_list}


def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    fused = {}

    def fp(doc):
        return doc[:100].lower().strip().replace(' ', '').replace('\n', '')

    for rank, (doc, meta) in enumerate(zip(vector_results["documents"], vector_results["metadatas"])):
        key = fp(doc)
        if key not in fused:
            fused[key] = {"score": 0, "doc": doc, "metadata": meta}
        fused[key]["score"] += 1 / (k + rank + 1)

    for rank, (doc, meta) in enumerate(zip(bm25_results["documents"], bm25_results["metadatas"])):
        key = fp(doc)
        if key not in fused:
            fused[key] = {"score": 0, "doc": doc, "metadata": meta}
        fused[key]["score"] += 1 / (k + rank + 1)

    sorted_results = sorted(fused.values(), key=lambda x: x["score"], reverse=True)

    return {
        "documents": [r["doc"] for r in sorted_results],
        "metadatas": [r["metadata"] for r in sorted_results],
        "scores": [r["score"] for r in sorted_results]
    }


def rewrite_query(query):
    try:
        prompt = f"""Rewrite this search query to improve document retrieval.
Add synonyms, related terms, and alternative phrasings.
Keep it concise (under 50 words). Output ONLY the rewritten query.

Original query: {query}

Rewritten query:"""
        expanded = llm.generate(prompt, use_search=False, use_thinking=False)
        return f"{query} {expanded.strip()}"
    except Exception as e:
        print(f"Error rewriting query: {e}")
        return query


def rerank_chunks(query, chunks, metadatas):
    if not chunks:
        return [], []

    try:
        response = requests.post(
            f"{RERANK_SERVER_URL}/rerank",
            json={"query": query, "documents": chunks},
            timeout=30
        )

        if response.status_code != 200:
            return chunks[:10], metadatas[:10]

        results = response.json()["results"]

        filtered = [r for r in results if r["score"] >= RERANK_SCORE_THRESHOLD]
        if not filtered:
            filtered = results[:3]

        selected_chunks = []
        selected_metadatas = []
        total_tokens = 0

        for r in filtered:
            idx = r["index"]
            chunk_text = chunks[idx]
            chunk_tokens = count_tokens(chunk_text)

            if total_tokens < MAX_CONTEXT_TOKENS:
                selected_chunks.append(chunk_text)
                selected_metadatas.append(metadatas[idx])
                total_tokens += chunk_tokens
                if total_tokens >= MAX_CONTEXT_TOKENS:
                    break

        return selected_chunks, selected_metadatas

    except requests.exceptions.ConnectionError:
        return select_by_tokens(chunks, metadatas)
    except Exception as e:
        print(f"Error reranking chunks: {e}")
        return select_by_tokens(chunks, metadatas)


def select_by_tokens(chunks, metadatas):
    selected_chunks = []
    selected_metadatas = []
    total_tokens = 0

    for chunk, meta in zip(chunks, metadatas):
        chunk_tokens = count_tokens(chunk)
        if total_tokens < MAX_CONTEXT_TOKENS:
            selected_chunks.append(chunk)
            selected_metadatas.append(meta)
            total_tokens += chunk_tokens
            if total_tokens >= MAX_CONTEXT_TOKENS:
                break

    return selected_chunks, selected_metadatas


def deduplicate(chunks, metadatas):
    seen = set()
    unique_chunks = []
    unique_metadatas = []

    for chunk, meta in zip(chunks, metadatas):
        fp = chunk[:100].lower().strip().replace(' ', '').replace('\n', '')
        if fp not in seen:
            seen.add(fp)
            unique_chunks.append(chunk)
            unique_metadatas.append(meta)

    return unique_chunks, unique_metadatas


def retrieve(query, user_id=None, workspace_id=None, do_rewrite=True):
    # Query rewriting
    search_query = rewrite_query(query) if do_rewrite else query

    # Build filters
    filters = {}
    if user_id:
        filters["user_id"] = user_id
    if workspace_id:
        filters["workspace_id"] = workspace_id

    # Vector search
    query_embedding = embedding.embed_query(search_query)
    vector_results = vector_store.query(
        query_embedding=query_embedding,
        n_results=INITIAL_RETRIEVAL_K,
        filters=filters if filters else None
    )

    # BM25 search + fusion
    bm25_results = bm25_search(search_query, INITIAL_RETRIEVAL_K, user_id, workspace_id)
    fused = reciprocal_rank_fusion(
        {"documents": vector_results["documents"], "metadatas": vector_results["metadatas"]},
        bm25_results
    )

    chunks = fused["documents"]
    metadatas = fused["metadatas"]

    # Deduplicate
    chunks, metadatas = deduplicate(chunks, metadatas)

    # Rerank
    chunks, metadatas = rerank_chunks(query, chunks, metadatas)

    return {"chunks": chunks, "metadatas": metadatas, "total_retrieved": len(chunks)}


def invalidate_bm25_cache():
    global corpus_hash
    corpus_hash = None
    