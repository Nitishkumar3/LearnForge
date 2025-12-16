"""
Exa.ai Deep Search Service
Provides web search capabilities with on-the-fly chunking and reranking
"""

from exa_py import Exa
import os

EXA_API_KEY = os.getenv("EXA_API_KEY")
exa = Exa(api_key=EXA_API_KEY)

def search_web(query, num_results=10):
    """Perform deep web search using Exa.ai."""
    try:
        result = exa.search_and_contents(
            query,
            num_results=num_results,
            text=True,
            type="auto"
        )

        web_sources = []
        full_texts = []

        for r in result.results:
            web_sources.append({
                "title": r.title or "Untitled",
                "url": r.url,
                "favicon": r.favicon or f"https://www.google.com/s2/favicons?domain={r.url}&sz=32"
            })
            if r.text:
                full_texts.append({
                    "text": r.text,
                    "source": r.title or r.url,
                    "url": r.url
                })

        return {
            "success": True,
            "web_sources": web_sources,
            "full_texts": full_texts,
            "result_count": len(web_sources)
        }

    except Exception as e:
        print(f"Exa search error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "web_sources": [],
            "full_texts": []
        }

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

def chunk_and_prepare_web_results(full_texts, chunk_size=500):
    """Chunk web results and prepare for embedding/reranking."""
    all_chunks = []

    for item in full_texts:
        text = item.get("text", "")
        source = item.get("source", "Unknown")
        url = item.get("url", "")

        chunks = chunk_text(text, chunk_size=chunk_size)
        for chunk in chunks:
            all_chunks.append({
                "chunk": chunk,
                "source": source,
                "url": url
            })

    return all_chunks

def rerank_chunks_simple(query, chunks, top_k=5):
    """Simple keyword-based reranking fallback."""
    query_words = set(query.lower().split())

    scored_chunks = []
    for chunk_data in chunks:
        chunk_text = chunk_data.get("chunk", "").lower()
        score = sum(1 for word in query_words if word in chunk_text)
        scored_chunks.append((score, chunk_data))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return [chunk_data for score, chunk_data in scored_chunks[:top_k]]

def get_web_context_for_llm(query, num_results=8, top_chunks=5):
    """Search web, chunk, rerank, and return context for LLM."""
    # Step 1: Search web
    search_result = search_web(query, num_results=num_results)

    if not search_result["success"]:
        return {
            "context": "",
            "web_sources": [],
            "error": search_result.get("error")
        }

    # Step 2: Chunk the results
    chunks = chunk_and_prepare_web_results(search_result["full_texts"])

    if not chunks:
        return {
            "context": "",
            "web_sources": search_result["web_sources"]
        }

    # Step 3: Rerank chunks
    top_chunks_data = rerank_chunks_simple(query, chunks, top_k=top_chunks)

    # Step 4: Build context string for LLM
    context_parts = []
    for i, chunk_data in enumerate(top_chunks_data, 1):
        source = chunk_data.get("source", "Web")
        chunk = chunk_data.get("chunk", "")
        context_parts.append(f"[Web Source {i}: {source}]\n{chunk}")

    context = "\n\n---\n\n".join(context_parts)

    return {
        "context": context,
        "web_sources": search_result["web_sources"]
    }