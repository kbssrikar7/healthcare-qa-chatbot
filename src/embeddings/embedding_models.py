"""
Medical embedding models for semantic search.
"""
import torch
import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

class MedicalEmbedder:
    """
    Medical domain embedding model wrapper.
    Supports: MedCPT, PubMedBERT, BioBERT, BGE-M3
    """
    
    SUPPORTED_MODELS = {
        "medcpt-query": "ncbi/MedCPT-Query-Encoder",
        "medcpt-article": "ncbi/MedCPT-Article-Encoder", 
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "biobert": "dmis-lab/biobert-v1.1",
        "bge-small": "BAAI/bge-small-en-v1.5",
        "all-minilm": "sentence-transformers/all-MiniLM-L6-v2"  # Fallback, fast
    }
    
    def __init__(
        self,
        model_name: str = "all-minilm",  # Default to fast model for testing
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get model path
        if model_name in self.SUPPORTED_MODELS:
            model_path = self.SUPPORTED_MODELS[model_name]
        else:
            model_path = model_name
        
        self.model_name = model_name
        print(f"🔄 Loading embedding model: {model_path} on {self.device}")
        
        try:
            self.model = SentenceTransformer(
                model_path,
                device=self.device,
                cache_folder=cache_dir
            )
            print(f"✅ Model loaded. Dimension: {self.embedding_dimension}")
        except Exception as e:
            print(f"⚠️ Failed to load {model_path}, falling back to all-MiniLM")
            self.model = SentenceTransformer(
                self.SUPPORTED_MODELS["all-minilm"],
                device=self.device
            )
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate embeddings for texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embed(query, show_progress=False)[0]
    
    def embed_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple documents."""
        return self.embed(documents, batch_size=batch_size, show_progress=True)
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, query: str, documents: List[str]) -> np.ndarray:
        """Calculate similarity between query and documents."""
        query_emb = self.embed_query(query)
        doc_embs = self.embed_documents(documents, batch_size=32)
        
        # Cosine similarity (embeddings are normalized)
        similarities = np.dot(doc_embs, query_emb)
        return similarities
