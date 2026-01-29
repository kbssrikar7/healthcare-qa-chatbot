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
    """Chunk medical documents while preserving context."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Separators in order of priority
        self.separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "; ",    # Clauses
            ", ",    # Phrases
            " "      # Words
        ]
    
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
        """Chunk a document with metadata."""
        content = document.get("content", "")
        source = document.get("source", "unknown")
        metadata = document.get("metadata", {})
        
        text_chunks = self.chunk_text(content)
        
        return [
            Chunk(
                content=chunk,
                source=source,
                chunk_id=i,
                total_chunks=len(text_chunks),
                metadata={
                    **metadata,
                    "url": document.get("url", ""),
                    "chunk_length": len(chunk)
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
