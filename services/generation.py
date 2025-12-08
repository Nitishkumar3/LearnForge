"""
Generation Service for LearnForge.

Handles answer generation using LLM providers with retrieved context.
"""

import re
from typing import List, Dict, Any, Generator
from services.llm_providers import get_llm_manager


class GenerationService:
    """Service for generating answers using LLM with retrieved context."""

    def __init__(self):
        """Initialize the generation service."""
        self.llm = get_llm_manager()

    def generate_answer(
        self,
        query: str,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]] = None,
        use_search: bool = False,
        use_thinking: bool = False
    ) -> Dict[str, Any]:
        """
        Generate an answer using the retrieved context.

        Args:
            query: User question
            chunks: Retrieved text chunks
            metadatas: Chunk metadata (for citations)
            chat_history: Previous Q&A pairs
            use_search: Enable web search if model supports it
            use_thinking: Enable thinking mode if model supports it

        Returns:
            Dict with:
                - answer: Generated answer text
                - sources: List of source citations
                - chunks_used: Number of chunks in context
        """
        if not chunks:
            return {
                "answer": "I don't have any documents to search. Please upload some PDFs first.",
                "sources": [],
                "chunks_used": 0
            }

        # Build context from chunks with source info
        context = self._build_context(chunks, metadatas)

        # Build chat history context
        history_text = self._build_history(chat_history)

        # Create the prompt
        prompt = self._build_prompt(query, context, history_text)

        # Generate response with optional features
        try:
            answer = self.llm.generate(
                prompt,
                use_search=use_search,
                use_thinking=use_thinking
            )
        except Exception as e:
            answer = f"Error generating response: {str(e)}"

        # Process citations - renumber and filter sources
        answer, sources = self._process_citations(answer, metadatas)

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(sources) if sources else 0
        }

    def _build_context(
        self,
        chunks: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> str:
        """
        Build context string from chunks with source information.

        Args:
            chunks: Text chunks
            metadatas: Chunk metadata

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
            doc_name = meta.get("document_name", "Unknown")
            page = meta.get("page_number", 0)

            if page:
                source_label = f"[Source {i}: {doc_name} - Page {page}]"
            else:
                source_label = f"[Source {i}: {doc_name}]"

            context_parts.append(f"{source_label}\n{chunk}")

        return "\n\n---\n\n".join(context_parts)

    def _build_history(
        self,
        chat_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Build chat history context.

        Args:
            chat_history: List of previous Q&A pairs

        Returns:
            Formatted history string
        """
        if not chat_history:
            return ""

        # Use last 5 exchanges
        recent_history = chat_history[-5:]

        history_parts = []
        for h in recent_history:
            history_parts.append(f"User: {h.get('question', '')}")
            history_parts.append(f"Assistant: {h.get('answer', '')}")

        return "\n\nPrevious conversation:\n" + "\n".join(history_parts)

    def _build_prompt(self, query: str, context: str, history_text: str) -> str:
        """
        Build the full prompt for the LLM.

        Args:
            query: User question
            context: Formatted context from chunks
            history_text: Chat history text

        Returns:
            Complete prompt string
        """
        prompt = f"""You are a helpful study assistant. Answer any question - both general and document-related.

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

        return prompt

    def _process_citations(self, answer: str, metadatas: List[Dict[str, Any]]) -> tuple:
        """
        Parse citations from answer, deduplicate sources, and remap citation numbers.

        Multiple chunks from same document (same name + page) get same citation number.

        Args:
            answer: LLM generated answer with citations like [1], [4], [7]
            metadatas: Chunk metadata list

        Returns:
            Tuple of (processed_answer, unique_sources_list)
        """
        if not answer:
            return answer, []

        if not metadatas:
            cleaned = re.sub(r'\[\d+\]', '', answer)
            return cleaned, []

        # Find all citation numbers in the answer
        all_cited = []
        for match in re.finditer(r'\[(\d+)\]', answer):
            num = int(match.group(1))
            if num not in all_cited:
                all_cited.append(num)

        if not all_cited:
            return answer, []

        # Filter to valid citations (within metadata range)
        valid_cited = [n for n in all_cited if 1 <= n <= len(metadatas)]

        if not valid_cited:
            cleaned = re.sub(r'\[\d+\]', '', answer)
            return cleaned, []

        # Build unique sources and map old citations to deduplicated numbers
        source_key_to_num = {}  # "doc_name|page" -> new_num
        unique_sources = []
        old_to_new = {}

        for old_num in sorted(valid_cited):
            meta = metadatas[old_num - 1]
            doc_name = meta.get("document_name", "Unknown")
            page = meta.get("page_number", 0)

            # Unique key for this source
            source_key = f"{doc_name}|{page}"

            if source_key not in source_key_to_num:
                # New unique source
                new_num = len(unique_sources) + 1
                source_key_to_num[source_key] = new_num

                if page:
                    unique_sources.append(f"{doc_name} - Page {page}")
                else:
                    unique_sources.append(doc_name)

            # Map old citation to deduplicated number
            old_to_new[old_num] = source_key_to_num[source_key]

        # Replace citations with deduplicated numbers
        def replace_citation(match):
            num = int(match.group(1))
            if num in old_to_new:
                return f"[{old_to_new[num]}]"
            return ""

        processed_answer = re.sub(r'\[(\d+)\]', replace_citation, answer)

        return processed_answer, unique_sources

    def generate_answer_stream(
        self,
        query: str,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]] = None,
        use_search: bool = False,
        use_thinking: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream an answer using the retrieved context.

        Yields:
            Dict with either:
                - {"type": "thinking", "content": "text"} for thinking chunks (Gemini only)
                - {"type": "chunk", "content": "text"} for text chunks
                - {"type": "done", "sources": [...], "thinking": "..."} when complete
                - {"type": "error", "message": "..."} on error
        """
        if not chunks:
            yield {
                "type": "done",
                "content": "I don't have any documents to search. Please upload some PDFs first.",
                "sources": [],
                "thinking": ""
            }
            return

        # Build context from chunks with source info
        context = self._build_context(chunks, metadatas)

        # Build chat history context
        history_text = self._build_history(chat_history)

        # Create the prompt
        prompt = self._build_prompt(query, context, history_text)

        # Stream response
        full_answer = ""
        full_thinking = ""
        try:
            for event in self.llm.generate_stream(
                prompt,
                use_search=use_search,
                use_thinking=use_thinking
            ):
                if event["type"] == "thinking":
                    full_thinking += event["content"]
                    yield {"type": "thinking", "content": event["content"]}
                elif event["type"] == "text":
                    full_answer += event["content"]
                    yield {"type": "chunk", "content": event["content"]}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        # Process citations after streaming complete
        processed_answer, sources = self._process_citations(full_answer, metadatas)

        # Yield final done event with sources and full thinking
        yield {
            "type": "done",
            "content": processed_answer,
            "sources": sources,
            "thinking": full_thinking
        }
