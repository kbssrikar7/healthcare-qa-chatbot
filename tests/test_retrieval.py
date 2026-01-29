"""
Unit tests for retrieval components.
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.preprocessors.text_cleaner import MedicalTextCleaner
from src.data_pipeline.preprocessors.chunker import MedicalTextChunker


class TestMedicalTextCleaner:
    """Tests for MedicalTextCleaner."""
    
    @pytest.fixture
    def cleaner(self):
        return MedicalTextCleaner()
    
    def test_clean_empty_string(self, cleaner):
        """Test cleaning empty string."""
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""
    
    def test_remove_html_tags(self, cleaner):
        """Test HTML tag removal."""
        text = "<p>This is a <b>test</b></p>"
        cleaned = cleaner.clean(text)
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
    
    def test_remove_urls(self, cleaner):
        """Test URL removal."""
        text = "Visit https://example.com for more info."
        cleaned = cleaner.clean(text)
        assert "https://" not in cleaned
        assert "example.com" not in cleaned
    
    def test_remove_references(self, cleaner):
        """Test citation reference removal."""
        text = "Studies show [1] that diabetes [2] is common."
        cleaned = cleaner.clean(text)
        assert "[1]" not in cleaned
        assert "[2]" not in cleaned
    
    def test_normalize_whitespace(self, cleaner):
        """Test whitespace normalization."""
        text = "Too   many    spaces"
        cleaned = cleaner.clean(text)
        assert "   " not in cleaned
    
    def test_preserve_medical_abbreviations(self, cleaner):
        """Test that medical abbreviations are preserved."""
        text = "Patient has BP 120/80 mmHg and HR 70 bpm"
        cleaned = cleaner.clean(text)
        assert "BP" in cleaned or "bp" in cleaned.lower()
        assert "HR" in cleaned or "hr" in cleaned.lower()


class TestMedicalTextChunker:
    """Tests for MedicalTextChunker."""
    
    @pytest.fixture
    def chunker(self):
        return MedicalTextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=20)
    
    def test_short_text_no_chunking(self, chunker):
        """Test that short text is not chunked."""
        text = "This is a short medical text."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_long_text_chunked(self, chunker):
        """Test that long text is properly chunked."""
        text = "This is sentence one. " * 20  # Create long text
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
    
    def test_chunk_size_respected(self, chunker):
        """Test that chunk size limit is respected."""
        text = "This is a test sentence. " * 50
        chunks = chunker.chunk_text(text)
        for chunk in chunks:
            # Allow some flexibility due to word boundaries
            assert len(chunk) <= chunker.chunk_size + 50
    
    def test_chunk_document_with_metadata(self, chunker):
        """Test chunking with document metadata."""
        document = {
            "content": "Medical content. " * 20,
            "source": "PubMed",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345"
        }
        chunks = chunker.chunk_document(document)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.source == "PubMed"
            assert chunk.chunk_id >= 0
            assert chunk.total_chunks > 0
    
    def test_empty_text_returns_empty(self, chunker):
        """Test that empty text returns empty list."""
        chunks = chunker.chunk_text("")
        assert chunks == []


class TestHybridRetriever:
    """Tests for HybridRetriever (mock-based)."""
    
    def test_retriever_initialization(self):
        """Test retriever can be initialized."""
        # This test would require mocking embedder and vector store
        # For now, just test imports work
        from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedDocument
        assert HybridRetriever is not None
        assert RetrievedDocument is not None
    
    def test_retrieved_document_dataclass(self):
        """Test RetrievedDocument dataclass."""
        from src.retrieval.hybrid_retriever import RetrievedDocument
        
        doc = RetrievedDocument(
            content="Test content",
            source="Test Source",
            score=0.85,
            metadata={"key": "value"}
        )
        assert doc.content == "Test content"
        assert doc.score == 0.85
