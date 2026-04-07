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
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
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
    # Filename for the pickled BM25 index cache (stored next to ChromaDB dir)
    _BM25_CACHE_FILE = "bm25_index.pkl"

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
        self.bm25 = None
        self._bm25_init_attempted = False
        # Latency diagnostics — exposed for QA pipeline to include in stage_latencies
        self.last_timings: Dict[str, float] = {}
        # Emit INFO-level timing on first retrieval call; DEBUG thereafter
        self._first_query_logged = False

        # Initialize BM25 for sparse retrieval if corpus provided
        if corpus:
            self._init_bm25(corpus)
    
    def warm_up(self) -> None:
        """
        Pre-initialize BM25 index from vector store.
        
        Call this during API startup to avoid 20-60s first-query lag.
        If BM25 is already initialized (via corpus= at __init__), this is a no-op.
        """
        if self.bm25 is None and not self._bm25_init_attempted:
            logger.info("Warmup: pre-initializing BM25 from vector store...")
            self._lazy_init_bm25_from_store()
            logger.info("Warmup: BM25 initialization complete")

    def _init_bm25(self, corpus: List[Dict]) -> None:
        """Initialize BM25 index from a corpus list."""
        self.corpus = corpus
        tokenized_corpus = []

        for doc in corpus:
            content = doc.get("content", "")
            tokenized_corpus.append(self._medical_tokenize(content))

        self.bm25 = BM25Okapi(tokenized_corpus)
        self._bm25_init_attempted = True
        logger.info(f" BM25 index initialized with {len(corpus)} documents")

    def _bm25_cache_path(self) -> Optional[Path]:
        """Return path to the BM25 pickle cache, or None if vector store has no persist dir."""
        try:
            persist_dir = Path(self.vector_store.persist_directory)
            return persist_dir / self._BM25_CACHE_FILE
        except Exception:
            return None

    def _bm25_cache_key(self, doc_count: int) -> str:
        """Stable cache key = doc_count + embedding_meta mtime (or '0' if absent)."""
        try:
            meta_path = Path(self.vector_store.persist_directory) / "embedding_metadata.json"
            mtime = str(int(meta_path.stat().st_mtime)) if meta_path.exists() else "0"
        except Exception:
            mtime = "0"
        return f"{doc_count}:{mtime}"

    def _try_load_bm25_cache(self, doc_count: int) -> bool:
        """Try to load BM25 from pickle cache. Returns True on success."""
        cache_path = self._bm25_cache_path()
        if cache_path is None or not cache_path.exists():
            return False
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("key") != self._bm25_cache_key(doc_count):
                logger.debug("BM25 cache invalidated (doc count or KB metadata changed)")
                return False
            self.corpus = cached["corpus"]
            self.bm25 = cached["bm25"]
            logger.info(
                f" BM25 loaded from cache ({doc_count:,} docs, "
                f"{cache_path.stat().st_size // 1024:,} KB)"
            )
            return True
        except Exception as e:
            logger.warning(f" BM25 cache load failed (will rebuild): {e}")
            return False

    def _save_bm25_cache(self, doc_count: int) -> None:
        """Persist BM25 index and corpus to pickle cache for fast subsequent loads."""
        cache_path = self._bm25_cache_path()
        if cache_path is None:
            return
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "key": self._bm25_cache_key(doc_count),
                        "corpus": self.corpus,
                        "bm25": self.bm25,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info(
                f" BM25 index cached to {cache_path.name} "
                f"({cache_path.stat().st_size // (1024 * 1024):.1f} MB)"
            )
        except Exception as e:
            logger.warning(f" BM25 cache save failed (non-fatal): {e}")

    def _lazy_init_bm25_from_store(self) -> None:
        """
        Lazy-load BM25 corpus from vector store in batches.

        On first call: loads from pickle cache if valid, otherwise fetches from
        ChromaDB, tokenizes, builds BM25, and saves to cache for next restart.
        Subsequent API restarts load from cache in seconds instead of minutes.
        """
        if self._bm25_init_attempted:
            return
        self._bm25_init_attempted = True

        try:
            total = self.vector_store.collection.count()
            if total == 0:
                logger.warning(" Vector store is empty — BM25 not initialized")
                return

            # Try loading from pickle cache first
            if self._try_load_bm25_cache(total):
                return

            logger.info(
                f" Building BM25 index from vector store ({total:,} docs)..."
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
                self._save_bm25_cache(total)
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
        text = text.lower()
        # Keep hyphens in compounds (e.g. "non-insulin") but remove other punctuation
        text = re.sub(r'[^\w\s\-]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) >= 2]

    def _format_dense_results(self, results: Dict) -> List[Tuple[str, float, Dict, str]]:
        """Convert raw vector-store results into dense retrieval tuples."""
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

    def _dense_retrieve(
        self,
        query: str,
        k: int,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float, Dict, str]]:
        """
        Perform dense vector retrieval.
        Returns list of (content, score, metadata, doc_id) tuples.
        """
        if query_embedding is None:
            query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding.tolist(), n_results=k)
        return self._format_dense_results(results)

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
        query_embedding = self.embedder.embed_query(query)
        _timings["embedding_ms"] = (time.perf_counter() - _t0) * 1000

        _t0 = time.perf_counter()
        dense_results = self._dense_retrieve(query, fetch_k, query_embedding=query_embedding)
        _timings["chroma_query_ms"] = (time.perf_counter() - _t0) * 1000

        if use_hybrid and self.bm25 is not None:
            # Sparse retrieval
            _t0 = time.perf_counter()
            sparse_results = self._sparse_retrieve(query, fetch_k)
            _timings["bm25_query_ms"] = (time.perf_counter() - _t0) * 1000

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

        # Document-level safety filter: tag potentially harmful chunks
        _t0 = time.perf_counter()
        documents = self._safety_filter_documents(documents)
        _timings["safety_filter_ms"] = (time.perf_counter() - _t0) * 1000

        # Log per-stage latency for diagnostics
        # First query → INFO so it's visible without debug logging configured.
        # Subsequent queries → DEBUG to avoid log spam.
        _timings["total_ms"] = sum(v for k, v in _timings.items() if k != "total_ms")
        self.last_timings = dict(_timings)  # expose for pipeline stage_latencies
        parts = " | ".join(f"{k}={v:.1f}" for k, v in _timings.items())
        if not self._first_query_logged:
            logger.info(f"[Retrieval cold-start] {parts}")
            self._first_query_logged = True
        else:
            logger.debug(f"Retrieval latency: {parts}")

        return documents[:k]

    # ── Safety filter ─────────────────────────────────────────────────────────

    # Patterns for content that should not appear in source citations.
    # Pre-compiled once at class definition time — avoids re-compiling on every
    # document during the safety filter loop (up to 90 compilations per query).
    _HARMFUL_PATTERNS = [
        r"\bhow\s+to\s+(?:make|synthesize|produce|create)\s+\w+\s+(?:drug|poison|explosive)",
        r"\bself[- ]harm\b",
        r"\b(?:illegal|illicit)\s+drug\s+(?:synthesis|recipe|formula)",
    ]
    _HARMFUL_PATTERNS_RE = [re.compile(p) for p in _HARMFUL_PATTERNS]

    def _safety_filter_documents(
        self, documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Tag retrieved chunks that contain potentially harmful content.

        Does NOT remove documents — adds a ``safety_flagged`` key to metadata
        so the pipeline can decide whether to include them in the LLM context.
        Harmful chunks are moved to the end of the list so they are less likely
        to make the top-k cut.

        Uses pre-compiled patterns (_HARMFUL_PATTERNS_RE) — avoids regex
        recompilation on every call (was ~90 compile+search ops per query).
        """
        safe, flagged = [], []
        for doc in documents:
            content_lower = doc.content.lower()
            is_harmful = any(
                p.search(content_lower) for p in self._HARMFUL_PATTERNS_RE
            )
            if is_harmful:
                doc.metadata["safety_flagged"] = True
                logger.debug(f"Safety-flagged retrieved chunk from '{doc.source}'")
                flagged.append(doc)
            else:
                doc.metadata.setdefault("safety_flagged", False)
                safe.append(doc)
        return safe + flagged

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
