"""
Medical embedding models for semantic search.
"""
from loguru import logger
import torch
import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
from collections import OrderedDict


class MedicalEmbedder:
    """
    Medical domain embedding model wrapper.
    Supports: MedCPT, PubMedBERT, BioBERT, BGE-M3

    Performance
    -----------
    embed_query() results are cached in a per-instance LRU dict (256 entries).
    This avoids redundant encoder forward passes for repeated queries, which
    is common during evaluation loops, BM25 re-init warmup, and cache hits.
    """

    SUPPORTED_MODELS = {
        "medcpt-query": "ncbi/MedCPT-Query-Encoder",
        "medcpt-article": "ncbi/MedCPT-Article-Encoder",
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "biobert": "dmis-lab/biobert-v1.1",
        "bge-small": "BAAI/bge-small-en-v1.5",
        "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",  # Fallback, fast
    }

    # Maximum number of query strings to keep in the in-memory embedding cache.
    # Each entry is ~768 * 4 bytes ≈ 3 KB → 256 entries ≈ 750 KB.
    _EMBED_CACHE_SIZE = 256

    def __init__(
        self,
        model_name: str = "all-minilm",  # Default to fast model for testing
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Get model path
        if model_name in self.SUPPORTED_MODELS:
            model_path = self.SUPPORTED_MODELS[model_name]
        else:
            model_path = model_name

        self.model_name = model_name
        self._model_path = model_path  # stored for cache-key purposes
        logger.info(f" Loading embedding model: {model_path} on {self.device}")

        try:
            self.model = SentenceTransformer(
                model_path,
                device=self.device,
                cache_folder=cache_dir,
            )
            logger.info(f" Model loaded. Dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.warning(f" Failed to load {model_path}, falling back to all-MiniLM: {e}")
            self.model = SentenceTransformer(
                self.SUPPORTED_MODELS["all-minilm"],
                device=self.device,
            )

        # Per-instance LRU cache: OrderedDict { query_str -> np.ndarray }
        # move_to_end on hit gives true LRU behaviour; popitem(last=False) for eviction.
        self._embed_cache: OrderedDict = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a string or list of strings."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query with LRU caching.

        Cached results are stored as immutable numpy arrays (read-only view).
        Callers that need to mutate the result should call `.copy()` themselves.
        """
        # Cache hit — move to end (most-recently-used) and return a copy
        if query in self._embed_cache:
            self._embed_cache.move_to_end(query)
            return self._embed_cache[query].copy()

        # Cache miss — encode and store
        embedding = self.embed(query, show_progress=False)[0]

        # Evict LRU entry if cache is full (O(1))
        if len(self._embed_cache) >= self._EMBED_CACHE_SIZE:
            self._embed_cache.popitem(last=False)

        self._embed_cache[query] = embedding

        return embedding.copy()

    def embed_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple documents (no caching — batch encoding is already efficient)."""
        return self.embed(documents, batch_size=batch_size, show_progress=True)

    def clear_cache(self) -> None:
        """Clear the query embedding cache (e.g., after knowledge-base rebuild)."""
        self._embed_cache.clear()
        logger.debug("Query embedding cache cleared.")

    def cache_stats(self) -> dict:
        """Return cache usage statistics."""
        return {
            "cached_queries": len(self._embed_cache),
            "max_cache_size": self._EMBED_CACHE_SIZE,
        }

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def similarity(self, query: str, documents: List[str]) -> np.ndarray:
        """Calculate cosine similarity between a query and a list of documents."""
        query_emb = self.embed_query(query)
        doc_embs = self.embed_documents(documents, batch_size=32)

        # Cosine similarity (embeddings are normalized)
        similarities = np.dot(doc_embs, query_emb)
        return similarities
