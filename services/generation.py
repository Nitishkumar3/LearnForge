"""Answer generation using LLM."""

import re
from services import llm_providers as llm


def build_context(chunks, metadatas):
    parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
        doc_name = meta.get("document_name", "Unknown")
        page = meta.get("page_number", 0)
        label = f"[Source {i}: {doc_name} - Page {page}]" if page else f"[Source {i}: {doc_name}]"
        parts.append(f"{label}\n{chunk}")
    return "\n\n---\n\n".join(parts)


def build_history(chat_history):
    if not chat_history:
        return ""
    recent = chat_history[-5:]
    lines = []
    for h in recent:
        lines.append(f"User: {h.get('question', '')}")
        lines.append(f"Assistant: {h.get('answer', '')}")
    return "\n\nPrevious conversation:\n" + "\n".join(lines)


def build_prompt(query, context, history_text):
    return f"""You are a helpful study assistant. Answer any question - both general and document-related.

CONTEXT (for reference only):
{context}
{history_text}

QUESTION: {query}

RESPONSE GUIDELINES:
- If the question is related to the documents, use the context and cite sources as [1], [2], etc.
- If the question is unrelated to the provided context, answer it directly without referencing the documents
- Do NOT force document references or add document-related information when the question is clearly unrelated
- Be clear, helpful, and well-structured

FORMATTING (Markdown + LaTeX):

Structure:
- Headings: ## Main Section, ### Subsection
- Bold: **important text**
- Italic: *emphasized text*
- Lists: Use - for bullets, 1. 2. 3. for numbered
- Code: `inline code` or ```code blocks```

Math & Science (use LaTeX with $ delimiters):
- Inline formulas: $E = mc^2$, $a^2 + b^2 = c^2$
- Chemical formulas: $CO_2$, $H_2O$, $NaCl$, $C_6H_{{12}}O_6$
- Subscripts: $x_1$, $A_{{max}}$
- Superscripts: $x^2$, $e^{{-x}}$
- Fractions: $\\frac{{a}}{{b}}$
- Greek letters: $\\alpha$, $\\beta$, $\\Delta$
- Block equations (own line): $$\\sum_{{i=1}}^{{n}} x_i$$

IMPORTANT:
- Always use $ for math/chemistry, never plain text for formulas
- Use double braces for multi-character subscripts/superscripts: $x_{{10}}$ not $x_10$
- Keep formatting consistent throughout response

Provide your response:"""


def process_citations(answer, metadatas):
    if not answer:
        return answer, []

    if not metadatas:
        return re.sub(r'\[\d+\]', '', answer), []

    # Find all citation numbers
    all_cited = []
    for match in re.finditer(r'\[(\d+)\]', answer):
        num = int(match.group(1))
        if num not in all_cited:
            all_cited.append(num)

    if not all_cited:
        return answer, []

    # Filter valid citations
    valid_cited = [n for n in all_cited if 1 <= n <= len(metadatas)]
    if not valid_cited:
        return re.sub(r'\[\d+\]', '', answer), []

    # Build unique sources
    source_key_to_num = {}
    unique_sources = []
    old_to_new = {}

    for old_num in sorted(valid_cited):
        meta = metadatas[old_num - 1]
        doc_name = meta.get("document_name", "Unknown")
        page = meta.get("page_number", 0)
        source_key = f"{doc_name}|{page}"

        if source_key not in source_key_to_num:
            new_num = len(unique_sources) + 1
            source_key_to_num[source_key] = new_num
            unique_sources.append(f"{doc_name} - Page {page}" if page else doc_name)

        old_to_new[old_num] = source_key_to_num[source_key]

    # Replace citations
    def replace_citation(match):
        num = int(match.group(1))
        return f"[{old_to_new[num]}]" if num in old_to_new else ""

    return re.sub(r'\[(\d+)\]', replace_citation, answer), unique_sources


def generate_answer(query, chunks, metadatas, chat_history=None, use_search=False, use_thinking=False):
    if not chunks:
        return {
            "answer": "I don't have any documents to search. Please upload some PDFs first.",
            "sources": [],
            "chunks_used": 0
        }

    context = build_context(chunks, metadatas)
    history_text = build_history(chat_history)
    prompt = build_prompt(query, context, history_text)

    try:
        answer = llm.generate(prompt, use_search=use_search, use_thinking=use_thinking)
    except Exception as e:
        answer = f"Error generating response: {str(e)}"

    answer, sources = process_citations(answer, metadatas)

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(sources) if sources else 0
    }


def generate_answer_stream(query, chunks, metadatas, chat_history=None, use_search=False, use_thinking=False):
    if not chunks:
        yield {
            "type": "done",
            "content": "I don't have any documents to search. Please upload some PDFs first.",
            "sources": [],
            "thinking": ""
        }
        return

    context = build_context(chunks, metadatas)
    history_text = build_history(chat_history)
    prompt = build_prompt(query, context, history_text)

    full_answer = ""
    full_thinking = ""

    try:
        for event in llm.generate_stream(prompt, use_search=use_search, use_thinking=use_thinking):
            if event["type"] == "thinking":
                full_thinking += event["content"]
                yield {"type": "thinking", "content": event["content"]}
            elif event["type"] == "text":
                full_answer += event["content"]
                yield {"type": "chunk", "content": event["content"]}
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return

    processed_answer, sources = process_citations(full_answer, metadatas)

    yield {
        "type": "done",
        "content": processed_answer,
        "sources": sources,
        "thinking": full_thinking
    }
