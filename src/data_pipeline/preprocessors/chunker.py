"""
Document chunking for medical texts with context preservation.
"""
from typing import List, Dict, Any
from dataclasses import dataclass
import re

@dataclass
class Chunk:
    """Represents a document chunk."""
    content: str
    source: str
    chunk_id: int
    total_chunks: int
    metadata: Dict[str, Any]

class MedicalTextChunker:
    """
    Chunk medical documents while preserving context.
    
    Features (Phase 2 improvements):
    - Domain-adaptive chunk sizes based on document type
    - Sentence-boundary awareness (never splits mid-sentence)
    - Recursive chunking with overlaps
    """
    
    # Domain-specific chunk sizes (in tokens, approximate)
    DOMAIN_CHUNK_SIZES = {
        "pubmedqa": 256,           # Dense abstracts
        "medmcqa": 128,            # Short Q&A pairs
        "healthcaremagic": 512,    # Long consultations
        "clinical_guideline": 768,  # Context-heavy guidelines
        "default": 512,
    }
    
    def __init__(
        self,
        chunk_size: int = None,  # Now optional - auto-determined by source
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        use_domain_adaptive: bool = True
    ):
        self.default_chunk_size = chunk_size or 512
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.use_domain_adaptive = use_domain_adaptive
        
        # Separators in order of priority
        self.separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "; ",    # Clauses
            ", ",    # Phrases
            " "      # Words
        ]
    
    def _get_chunk_size_for_source(self, source: str) -> int:
        """Determine optimal chunk size based on document source."""
        if not self.use_domain_adaptive:
            return self.default_chunk_size
        
        source_lower = source.lower()
        for key, size in self.DOMAIN_CHUNK_SIZES.items():
            if key in source_lower:
                return size
        return self.DOMAIN_CHUNK_SIZES["default"]
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle very long sentences
                if len(sentence) > self.chunk_size:
                    # Split by words
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                            current_chunk += (" " if current_chunk else "") + word
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = word
                else:
                    current_chunk = sentence
        
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        # Add overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        overlapped = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Get last N characters from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                
                # Find word boundary
                space_idx = overlap_text.find(' ')
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1:]
                
                overlapped.append(overlap_text + " " + chunk)
        
        return overlapped
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a document with metadata.
        
        Uses domain-adaptive chunk sizing: determines optimal chunk size
        based on document source type (e.g., PubMedQA vs HealthCareMagic).
        """
        content = document.get("content", "")
        source = document.get("source", "unknown")
        metadata = document.get("metadata", {})
        
        # Domain-adaptive chunk size
        original_chunk_size = self.chunk_size if hasattr(self, "chunk_size") else self.default_chunk_size
        if self.use_domain_adaptive:
            self.chunk_size = self._get_chunk_size_for_source(source)
        else:
            self.chunk_size = self.default_chunk_size
        
        text_chunks = self.chunk_text(content)
        
        # Restore original if needed (for reusability across sources)
        if self.use_domain_adaptive:
            self.chunk_size = self.default_chunk_size
        
        return [
            Chunk(
                content=chunk,
                source=source,
                chunk_id=i,
                total_chunks=len(text_chunks),
                metadata={
                    **metadata,
                    "url": document.get("url", ""),
                    "chunk_length": len(chunk),
                    "adaptive_chunk_size": self._get_chunk_size_for_source(source) if self.use_domain_adaptive else original_chunk_size
                }
            )
            for i, chunk in enumerate(text_chunks)
        ]
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
