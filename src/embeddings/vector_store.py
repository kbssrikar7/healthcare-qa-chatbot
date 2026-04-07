"""
Vector store management for document retrieval.
"""
from loguru import logger
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import uuid
import hashlib
from pathlib import Path
import json


def generate_deterministic_id(
    source: str,
    content: str,
    chunk_id: int = 0,
) -> str:
    """
    Generate a deterministic document ID from source + chunk_id + content hash.
    
    This ensures:
    - Same document always gets the same ID across ingestion runs
    - Different chunks of the same source get distinct IDs
    - Collisions are extremely unlikely (SHA-256 prefix)
    
    Args:
        source: Document source name
        content: Document text content
        chunk_id: Chunk index within the source document
        
    Returns:
        Deterministic hex ID string (32 chars)
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    key = f"{source}::chunk{chunk_id}::{content_hash}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class VectorStore:
    """
    ChromaDB-based vector store for medical knowledge.
    """
    
    def __init__(
        self,
        collection_name: str = "medical_knowledge",
        persist_directory: str = "data/knowledge_base",
        embedding_function = None
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Version-agnostic chromadb client initialization
        try:
            # Try new API (chromadb >= 1.0)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )
        except (AttributeError, TypeError):
            try:
                # Try 0.4+ API with Settings
                self.client = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            except AttributeError:
                # Fall back to old API (chromadb < 0.4)
                self.client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(self.persist_directory),
                    anonymized_telemetry=False
                ))
        
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f" Vector store initialized: {collection_name}")
        logger.info(f" Documents in collection: {self.collection.count()}")

    def verify_embedding_compatibility(self, model_name: str, dimension: int) -> bool:
        """Check that the query model matches the indexed embedding model.

        Reads stored model metadata from collection. If no metadata is stored
        (older index), logs a warning but returns True (non-fatal).

        Args:
            model_name: Name of the embedding model being used for queries.
            dimension:  Expected embedding dimension.

        Returns:
            True if compatible (or no stored metadata to compare against).
        """
        try:
            meta = self.collection.metadata or {}
            stored_model = meta.get("embedding_model")
            stored_dim   = meta.get("embedding_dimension")
            if stored_model and stored_model != model_name:
                logger.warning(
                    f"Embedding model mismatch: index was built with '{stored_model}' "
                    f"but query model is '{model_name}'. Retrieval quality may be degraded."
                )
                return False
            if stored_dim and int(stored_dim) != dimension:
                logger.error(
                    f"Embedding dimension mismatch: index has {stored_dim}d, "
                    f"query model produces {dimension}d vectors. Queries will fail."
                )
                return False
            if not stored_model:
                logger.debug("No embedding model metadata in index — skipping compatibility check")
            return True
        except Exception as e:
            logger.warning(f"Could not verify embedding compatibility: {e}")
            return True  # non-fatal

    def record_embedding_model(self, model_name: str, dimension: int) -> None:
        """Persist embedding model info in collection metadata (call once at index build time)."""
        try:
            # ChromaDB does not support updating collection metadata after creation,
            # so we store it in a dedicated sentinel document instead.
            sentinel_id = "__embedding_meta__"
            existing = self.collection.get(ids=[sentinel_id])
            if not existing["ids"]:
                self.collection.add(
                    ids=[sentinel_id],
                    documents=["__embedding_metadata_sentinel__"],
                    metadatas=[{"embedding_model": model_name, "embedding_dimension": dimension}],
                    embeddings=[[0.0] * dimension],
                )
                logger.info(f"Recorded embedding model metadata: {model_name} ({dimension}d)")
        except Exception as e:
            logger.warning(f"Could not record embedding model metadata: {e}")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents with embeddings to the store."""
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        # Generate deterministic IDs if not provided
        if ids is None:
            ids = [
                generate_deterministic_id(
                    source=meta.get("source", ""),
                    content=doc,
                    chunk_id=meta.get("chunk_id", i),
                )
                for i, (doc, meta) in enumerate(zip(documents, metadatas))
            ]
        
        # Ensure metadata values are valid types (str, int, float, bool)
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif v is None:
                    clean_meta[k] = ""
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=clean_metadatas
        )
        
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """Search for similar documents."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        return results
    
    def mmr_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Maximal Marginal Relevance search for diverse results.
        
        MMR balances relevance to query with diversity among results.
        
        Args:
            query_embedding: Query vector
            k: Number of documents to return
            fetch_k: Number of documents to fetch before MMR
            lambda_mult: Balance between relevance and diversity
                        (0 = max diversity, 1 = max relevance)
            filter_metadata: Optional metadata filter
            
        Returns:
            Dict with documents, metadatas, and distances
        """
        import numpy as np
        
        # Fetch more documents than needed
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        if not results["documents"] or not results["documents"][0]:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        embeddings = results.get("embeddings", [[]])[0]
        
        if not embeddings or len(embeddings) == 0:
            # Fallback to regular search if embeddings not available
            return self.search(query_embedding, n_results=k, filter_metadata=filter_metadata)
        
        # Convert query to numpy
        query_vec = np.array(query_embedding)
        doc_embeddings = np.array(embeddings)
        
        # Calculate similarity to query (convert distance to similarity)
        query_similarities = 1 - np.array(distances)
        
        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = query_similarities[idx]
                
                # Max similarity to already selected docs
                if selected_indices:
                    selected_embeddings = doc_embeddings[selected_indices]
                    similarities = np.dot(selected_embeddings, doc_embeddings[idx])
                    max_sim_to_selected = np.max(similarities)
                else:
                    max_sim_to_selected = 0
                
                # MMR formula: λ * relevance - (1 - λ) * max_similarity
                mmr_score = (lambda_mult * relevance - 
                           (1 - lambda_mult) * max_sim_to_selected)
                mmr_scores.append((idx, mmr_score))
            
            # Select highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Build result
        return {
            "documents": [[documents[i] for i in selected_indices]],
            "metadatas": [[metadatas[i] for i in selected_indices]],
            "distances": [[distances[i] for i in selected_indices]]
        }
    
    def get_stats(self) -> Dict:
        """Get statistics about the collection."""
        return {
            "name": self.collection_name,
            "count": self.collection.count()
        }
    
    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
