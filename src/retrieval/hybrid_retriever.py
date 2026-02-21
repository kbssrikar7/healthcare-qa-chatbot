"""
Hybrid retrieval combining dense and sparse search with RRF fusion.

Enhanced with:
- Reciprocal Rank Fusion (RRF) for combining dense + sparse results
- Optional cross-encoder reranking for improved precision
- Configurable retrieval parameters
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class RetrievedDocument:
    """Represents a retrieved document."""
    content: str
    source: str
    score: float
    metadata: Dict


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    RRF score for document d = sum(weight_i / (k + rank_i(d))) for all lists i
    
    Args:
        result_lists: List of ranked lists, each containing (doc_id, score) tuples
        k: Ranking constant (default 60, reduces impact of high rankings)
        weights: Optional weights for each result list
        
    Returns:
        Dict mapping doc_id to RRF score
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    fused_scores: Dict[str, float] = {}
    
    for weight, results in zip(weights, result_lists):
        for rank, (doc_id, _) in enumerate(results, start=1):
            rrf_score = weight * (1.0 / (k + rank))
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score
    
    return fused_scores


class HybridRetriever:
    """
    Combine dense and sparse retrieval for better results.
    
    Features:
    - Dense retrieval via vector store
    - Sparse retrieval via BM25
    - Reciprocal Rank Fusion for score combination
    - Optional cross-encoder reranking
    """
    
    def __init__(
        self,
        embedder,
        vector_store,
        corpus: List[Dict] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        reranker = None,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            embedder: Embedding model for query encoding
            vector_store: Vector store for dense retrieval
            corpus: Document corpus for BM25 initialization
            dense_weight: Weight for dense retrieval in fusion
            sparse_weight: Weight for sparse retrieval in fusion
            reranker: Optional cross-encoder reranker
            rrf_k: RRF constant (default 60)
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.corpus = corpus
        self.corpus_map = {}  # Map content hash to corpus index
        self.bm25 = None
        
        # Initialize BM25 for sparse retrieval if corpus provided
        if corpus:
            self._init_bm25(corpus)
    
    def _init_bm25(self, corpus: List[Dict]):
        """Initialize BM25 index."""
        self.corpus = corpus
        tokenized_corpus = []
        
        for i, doc in enumerate(corpus):
            content = doc.get("content", "")
            tokenized_corpus.append(content.lower().split())
            # Create content hash mapping
            self.corpus_map[content[:100]] = i
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ BM25 index initialized with {len(corpus)} documents")
    
    def _dense_retrieve(
        self,
        query: str,
        k: int
    ) -> List[Tuple[str, float, Dict]]:
        """Perform dense vector retrieval."""
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding.tolist(), n_results=k)
        
        documents = []
        if results["documents"] and results["documents"][0]:
            for doc, distance, metadata in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0]
            ):
                # Convert distance to similarity score
                similarity = 1 - distance
                documents.append((doc, float(similarity), metadata))
        
        return documents
    
    def _sparse_retrieve(
        self,
        query: str,
        k: int
    ) -> List[Tuple[str, float, Dict]]:
        """Perform sparse BM25 retrieval."""
        if self.bm25 is None or self.corpus is None:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        documents = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.corpus[idx]
                documents.append((
                    doc.get("content", ""),
                    float(scores[idx]),
                    {
                        "source": doc.get("source", "unknown"),
                        "url": doc.get("url", ""),
                        **doc.get("metadata", {})
                    }
                ))
        
        return documents
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        use_hybrid: bool = True,
        use_reranking: bool = True
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents.
        
        Args:
            query: Search query
            k: Number of documents to return
            use_hybrid: Whether to use hybrid (dense + sparse) retrieval
            use_reranking: Whether to apply cross-encoder reranking
            
        Returns:
            List of retrieved documents sorted by relevance
        """
        fetch_k = k * 3 if use_reranking else k * 2
        
        # Dense retrieval
        dense_results = self._dense_retrieve(query, fetch_k)
        
        if use_hybrid and self.bm25 is not None:
            # Sparse retrieval
            sparse_results = self._sparse_retrieve(query, fetch_k)
            
            # Create ranked lists for RRF
            # Use content[:100] as document ID
            dense_ranked = [(doc[:100], score) for doc, score, _ in dense_results]
            sparse_ranked = [(doc[:100], score) for doc, score, _ in sparse_results]
            
            # Apply RRF
            fused_scores = reciprocal_rank_fusion(
                [dense_ranked, sparse_ranked],
                k=self.rrf_k,
                weights=[self.dense_weight, self.sparse_weight]
            )
            
            # Merge documents with fused scores
            doc_map = {}
            for doc, score, metadata in dense_results + sparse_results:
                doc_key = doc[:100]
                if doc_key not in doc_map:
                    doc_map[doc_key] = (doc, metadata)
            
            # Build final documents list
            documents = []
            for doc_key, fused_score in sorted(
                fused_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:fetch_k]:
                if doc_key in doc_map:
                    content, metadata = doc_map[doc_key]
                    documents.append(RetrievedDocument(
                        content=content,
                        source=metadata.get("source", "unknown"),
                        score=fused_score,
                        metadata=metadata
                    ))
        else:
            # Dense only
            documents = [
                RetrievedDocument(
                    content=doc,
                    source=metadata.get("source", "unknown"),
                    score=score,
                    metadata=metadata
                )
                for doc, score, metadata in dense_results
            ]
        
        # Apply reranking if available
        if use_reranking and self.reranker is not None:
            documents = self.reranker.rerank(query, documents, top_k=k)
            # Ensure we have RetrievedDocument objects
            if documents and not isinstance(documents[0], RetrievedDocument):
                # Reranker returned different format
                documents = [
                    RetrievedDocument(
                        content=d.text if hasattr(d, 'text') else d.content,
                        source=d.metadata.get("source", "unknown") if hasattr(d, 'metadata') else "unknown",
                        score=d.score if hasattr(d, 'score') else 0.0,
                        metadata=d.metadata if hasattr(d, 'metadata') else {}
                    )
                    for d in documents
                ]
        
        # Sort by score and return top k
        documents.sort(key=lambda x: x.score, reverse=True)
        return documents[:k]
    
    def retrieve_with_context(
        self,
        query: str,
        k: int = 5,
        max_context_length: int = 2000,
        use_reranking: bool = True
    ) -> Tuple[List[RetrievedDocument], str]:
        """
        Retrieve documents and build context string.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            max_context_length: Maximum context length in characters
            use_reranking: Whether to apply reranking
            
        Returns:
            Tuple of (retrieved documents, formatted context string)
        """
        documents = self.retrieve(query, k=k, use_reranking=use_reranking)
        
        # Build context from retrieved documents
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents, 1):
            if total_length + len(doc.content) > max_context_length:
                break
            context_parts.append(f"[{i}] Source: {doc.source}\n{doc.content}")
            total_length += len(doc.content)
        
        context = "\n\n".join(context_parts)
        
        return documents, context
