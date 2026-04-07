"""
Hybrid retrieval combining dense and sparse search with RRF fusion.

Enhanced with:
- Reciprocal Rank Fusion (RRF) for combining dense + sparse results
- Lazy batched BM25 initialization from vector store
- Stable document ID-based deduplication (not content prefix)
- Optional cross-encoder reranking for improved precision
- Configurable retrieval parameters
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi


@dataclass
class RetrievedDocument:
    """Represents a retrieved document."""

    content: str
    source: str
    score: float
    metadata: Dict
    doc_id: str = ""
    score_type: str = "cosine"  # "cosine" | "rrf" | "reranked"


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
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


def _stable_doc_id(content: str, source: str = "") -> str:
    """Generate a stable document ID from content + source for deduplication."""
    key = f"{source}::{content[:300]}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class HybridRetriever:
    """
    Combine dense and sparse retrieval for better results.

    Features:
    - Dense retrieval via vector store
    - Sparse retrieval via BM25 (lazy-loaded from vector store)
    - Reciprocal Rank Fusion with stable-ID deduplication
    - Optional cross-encoder reranking
    """

    # Batch size for loading documents from vector store for BM25
    _BM25_LOAD_BATCH = 5000

    def __init__(
        self,
        embedder,
        vector_store,
        corpus: List[Dict] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        reranker=None,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.

        Args:
            embedder: Embedding model for query encoding
            vector_store: Vector store for dense retrieval
            corpus: Document corpus for BM25 initialization (if None, auto-loads lazily)
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
        self.corpus_map: Dict[str, int] = {}  # doc_id -> corpus index
        self.bm25 = None
        self._bm25_init_attempted = False

        # Initialize BM25 for sparse retrieval if corpus provided
        if corpus:
            self._init_bm25(corpus)

    def _init_bm25(self, corpus: List[Dict]) -> None:
        """Initialize BM25 index from a corpus list."""
        self.corpus = corpus
        tokenized_corpus = []
        self.corpus_map = {}

        for i, doc in enumerate(corpus):
            content = doc.get("content", "")
            tokenized_corpus.append(self._medical_tokenize(content))
            doc_id = doc.get("id", _stable_doc_id(content, doc.get("source", "")))
            self.corpus_map[doc_id] = i

        self.bm25 = BM25Okapi(tokenized_corpus)
        self._bm25_init_attempted = True
        logger.info(f" BM25 index initialized with {len(corpus)} documents")

    def _lazy_init_bm25_from_store(self) -> None:
        """
        Lazy-load BM25 corpus from vector store in batches.
        Called on the first hybrid retrieval if no corpus was provided.
        """
        if self._bm25_init_attempted:
            return
        self._bm25_init_attempted = True

        try:
            total = self.vector_store.collection.count()
            if total == 0:
                logger.warning(" Vector store is empty — BM25 not initialized")
                return

            logger.info(
                f" Lazy-loading BM25 corpus from vector store ({total:,} docs)..."
            )
            corpus = []
            offset = 0

            while offset < total:
                batch = self.vector_store.collection.get(
                    limit=self._BM25_LOAD_BATCH,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                if not batch["documents"]:
                    break

                ids = batch.get("ids", [])
                for j, (doc_text, meta) in enumerate(
                    zip(batch["documents"], batch["metadatas"])
                ):
                    corpus.append(
                        {
                            "content": doc_text,
                            "source": meta.get("source", "unknown")
                            if meta
                            else "unknown",
                            "id": ids[j] if j < len(ids) else _stable_doc_id(doc_text),
                            "metadata": meta or {},
                        }
                    )
                offset += len(batch["documents"])

            if corpus:
                self._init_bm25(corpus)
            else:
                logger.warning(" No documents retrieved from vector store for BM25")
        except Exception as e:
            logger.warning(f" BM25 lazy-init failed: {e}")

    def initialize(self) -> None:
        """Pre-initialize BM25 index. Call at startup to avoid first-query delay."""
        self._lazy_init_bm25_from_store()

    @staticmethod
    def _medical_tokenize(text: str) -> list:
        """Tokenize text for BM25, handling medical terms and punctuation."""
        import re
        text = text.lower()
        # Keep hyphens in compounds (e.g. "non-insulin") but remove other punctuation
        text = re.sub(r'[^\w\s\-]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) >= 2]

    def _dense_retrieve(self, query: str, k: int) -> List[Tuple[str, float, Dict, str]]:
        """
        Perform dense vector retrieval.
        Returns list of (content, score, metadata, doc_id) tuples.
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding.tolist(), n_results=k)

        documents = []
        if results["documents"] and results["documents"][0]:
            ids = results.get("ids", [[]])[0]
            for idx, (doc, distance, metadata) in enumerate(
                zip(
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0],
                )
            ):
                similarity = 1 - distance
                doc_id = (
                    ids[idx]
                    if idx < len(ids)
                    else _stable_doc_id(doc, metadata.get("source", ""))
                )
                documents.append((doc, float(similarity), metadata, doc_id))

        return documents

    def _sparse_retrieve(
        self, query: str, k: int
    ) -> List[Tuple[str, float, Dict, str]]:
        """
        Perform sparse BM25 retrieval.
        Returns list of (content, score, metadata, doc_id) tuples.
        """
        if self.bm25 is None or self.corpus is None:
            return []

        tokenized_query = self._medical_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:k]

        documents = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.corpus[idx]
                content = doc.get("content", "")
                doc_id = doc.get("id", _stable_doc_id(content, doc.get("source", "")))
                metadata = {
                    "source": doc.get("source", "unknown"),
                    "url": doc.get("url", ""),
                    "id": doc_id,
                    **doc.get("metadata", {}),
                }
                documents.append((content, float(scores[idx]), metadata, doc_id))

        return documents

    def retrieve(
        self,
        query: str,
        k: int = 10,
        use_hybrid: bool = True,
        use_reranking: bool = True,
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
        # Lazy-init BM25 from vector store if needed
        if use_hybrid and self.bm25 is None and not self._bm25_init_attempted:
            self._lazy_init_bm25_from_store()

        fetch_k = k * 3 if use_reranking else k * 2
        _timings: Dict[str, float] = {}

        # Dense retrieval
        _t0 = time.perf_counter()
        dense_results = self._dense_retrieve(query, fetch_k)
        _timings["dense_ms"] = (time.perf_counter() - _t0) * 1000

        if use_hybrid and self.bm25 is not None:
            # Sparse retrieval
            _t0 = time.perf_counter()
            sparse_results = self._sparse_retrieve(query, fetch_k)
            _timings["sparse_ms"] = (time.perf_counter() - _t0) * 1000

            # Create ranked lists for RRF using STABLE doc IDs (not content prefix)
            dense_ranked = [(doc_id, score) for _, score, _, doc_id in dense_results]
            sparse_ranked = [(doc_id, score) for _, score, _, doc_id in sparse_results]

            # Apply RRF
            _t0 = time.perf_counter()
            fused_scores = reciprocal_rank_fusion(
                [dense_ranked, sparse_ranked],
                k=self.rrf_k,
                weights=[self.dense_weight, self.sparse_weight],
            )
            _timings["rrf_ms"] = (time.perf_counter() - _t0) * 1000

            # Merge documents by stable ID (dedup)
            doc_map: Dict[str, Tuple[str, Dict]] = {}
            for content, score, metadata, doc_id in dense_results + sparse_results:
                if doc_id not in doc_map:
                    doc_map[doc_id] = (content, metadata)

            # Build final documents list
            documents = []
            for doc_id, fused_score in sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )[:fetch_k]:
                if doc_id in doc_map:
                    content, metadata = doc_map[doc_id]
                    documents.append(
                        RetrievedDocument(
                            content=content,
                            source=metadata.get("source", "unknown"),
                            score=fused_score,
                            metadata=metadata,
                            doc_id=doc_id,
                            score_type="rrf",
                        )
                    )
        else:
            # Dense only
            documents = [
                RetrievedDocument(
                    content=doc,
                    source=metadata.get("source", "unknown"),
                    score=score,
                    metadata=metadata,
                    doc_id=doc_id,
                )
                for doc, score, metadata, doc_id in dense_results
            ]

        # Apply reranking if available
        if use_reranking and self.reranker is not None:
            _t0 = time.perf_counter()
            documents = self.reranker.rerank(query, documents, top_k=k)
            _timings["rerank_ms"] = (time.perf_counter() - _t0) * 1000
            # Ensure we have RetrievedDocument objects
            if documents and not isinstance(documents[0], RetrievedDocument):
                documents = [
                    RetrievedDocument(
                        content=d.text if hasattr(d, "text") else d.content,
                        source=d.metadata.get("source", "unknown")
                        if hasattr(d, "metadata")
                        else "unknown",
                        score=d.score if hasattr(d, "score") else 0.0,
                        metadata=d.metadata if hasattr(d, "metadata") else {},
                        doc_id=getattr(d, "doc_id", ""),
                        score_type="reranked",
                    )
                    for d in documents
                ]
            else:
                for d in documents:
                    d.score_type = "reranked"

        # Sort by score and return top k
        documents.sort(key=lambda x: x.score, reverse=True)

        # Log per-stage latency for diagnostics
        total = sum(_timings.values())
        parts = " | ".join(f"{k}={v:.1f}" for k, v in _timings.items())
        logger.debug(f"Retrieval latency: total={total:.1f}ms  [{parts}]")

        return documents[:k]

    def retrieve_with_context(
        self,
        query: str,
        k: int = 5,
        max_context_length: int = 2000,
        use_reranking: bool = True,
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
