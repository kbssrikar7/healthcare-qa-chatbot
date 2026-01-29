"""
Context compression for improved LLM attention.

Based on rag-architecture skill pattern for solving the "lost-in-the-middle" problem.
Reorders and compresses context to maximize LLM attention on relevant content.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CompressedContext:
    """Result of context compression."""
    text: str
    num_docs_used: int
    original_length: int
    compressed_length: int
    compression_ratio: float


class ContextCompressor:
    """
    Compress and reorder retrieved context for better LLM performance.
    
    Addresses the "lost-in-the-middle" problem where LLMs pay more attention
    to content at the beginning and end of context.
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        sentences_per_doc: int = 5,
        llm=None
    ):
        """
        Initialize context compressor.
        
        Args:
            max_tokens: Maximum context length in tokens
            sentences_per_doc: Key sentences to extract per document
            llm: Optional LLM for summarization
        """
        self.max_tokens = max_tokens
        self.sentences_per_doc = sentences_per_doc
        self.llm = llm
    
    def compress(
        self,
        documents: List,
        query: str
    ) -> CompressedContext:
        """
        Compress and reorder documents for optimal LLM attention.
        
        Args:
            documents: Retrieved documents
            query: User query for relevance scoring
            
        Returns:
            CompressedContext with optimized text
        """
        if not documents:
            return CompressedContext(
                text="",
                num_docs_used=0,
                original_length=0,
                compressed_length=0,
                compression_ratio=1.0
            )
        
        # Extract content from documents
        doc_contents = []
        for doc in documents:
            if hasattr(doc, 'content'):
                content = doc.content
            elif isinstance(doc, dict):
                content = doc.get('content', str(doc))
            else:
                content = str(doc)
            doc_contents.append(content)
        
        original_text = "\n\n".join(doc_contents)
        original_length = len(original_text)
        
        # Extract key sentences from each document
        compressed_docs = []
        for i, content in enumerate(doc_contents):
            key_sentences = self._extract_key_sentences(content, query)
            if key_sentences:
                compressed_docs.append({
                    'index': i,
                    'content': key_sentences,
                    'score': documents[i].score if hasattr(documents[i], 'score') else 1.0
                })
        
        # Sort by score
        compressed_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # Reorder: most relevant at start AND end (lost-in-the-middle mitigation)
        reordered = self._reorder_for_attention(compressed_docs)
        
        # Format final context
        context_parts = []
        for doc in reordered:
            source_idx = doc['index'] + 1
            context_parts.append(f"[Source {source_idx}]\n{doc['content']}")
        
        compressed_text = "\n\n---\n\n".join(context_parts)
        
        # Truncate if still too long
        compressed_text = self._truncate_to_tokens(compressed_text)
        
        return CompressedContext(
            text=compressed_text,
            num_docs_used=len(reordered),
            original_length=original_length,
            compressed_length=len(compressed_text),
            compression_ratio=len(compressed_text) / max(original_length, 1)
        )
    
    def _extract_key_sentences(self, text: str, query: str) -> str:
        """Extract most relevant sentences from text."""
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= self.sentences_per_doc:
            return text
        
        # Score sentences by relevance to query
        query_terms = set(query.lower().split())
        scored = []
        
        for sent in sentences:
            sent_terms = set(sent.lower().split())
            overlap = len(query_terms & sent_terms)
            # Prefer longer sentences with more overlap
            score = overlap + (len(sent) / 200)
            scored.append((sent, score))
        
        # Get top sentences
        scored.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored[:self.sentences_per_doc]]
        
        # Return in original order
        ordered = [s for s in sentences if s in top_sentences]
        return ' '.join(ordered)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _reorder_for_attention(self, docs: List[Dict]) -> List[Dict]:
        """
        Reorder documents to put most relevant at beginning AND end.
        
        LLMs pay more attention to:
        1. Beginning of context
        2. End of context
        3. Less attention to middle
        """
        if len(docs) <= 2:
            return docs
        
        # Put best at start, second best at end, rest in middle
        reordered = []
        
        # First half (best docs)
        first_half = docs[:len(docs)//2]
        # Second half (remaining docs)
        second_half = docs[len(docs)//2:]
        
        # Interleave: best at start, second best at end
        reordered.append(first_half[0])  # Best at start
        
        # Add middle docs
        for i in range(1, len(first_half)):
            reordered.append(first_half[i])
        
        # Add second half
        for doc in second_half:
            reordered.append(doc)
        
        return reordered
    
    def _truncate_to_tokens(self, text: str) -> str:
        """Truncate text to max tokens (approximating 4 chars = 1 token)."""
        max_chars = self.max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
    
    def summarize_if_needed(
        self,
        context: str,
        query: str
    ) -> str:
        """
        Summarize context if it's very long.
        
        Only works if LLM is available.
        """
        if not self.llm or len(context) < self.max_tokens * 4:
            return context
        
        prompt = f"""Summarize the following context to help answer this question: "{query}"

Context:
{context[:8000]}

Provide a concise summary focusing on information relevant to the question."""
        
        try:
            response = self.llm.generate(prompt, max_new_tokens=500)
            return response.response.strip()
        except Exception:
            return context
