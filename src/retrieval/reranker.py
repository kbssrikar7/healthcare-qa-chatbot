"""
Cross-encoder reranking for improved retrieval precision.

Based on RAG skill patterns for production-grade reranking.
"""

from dataclasses import dataclass
from typing import List

from loguru import logger

# Optional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class RerankResult:
    """Result from reranking."""

    text: str
    score: float
    original_index: int
    metadata: dict


class CrossEncoderReranker:
    """
    Rerank documents using cross-encoder model.

    Cross-encoders jointly encode query-document pairs for more
    accurate relevance scoring than bi-encoder similarity.
    """

    SUPPORTED_MODELS = {
        "ms-marco-mini": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-base": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "bge-reranker": "BAAI/bge-reranker-base",
    }

    def __init__(self, model_name: str = "ms-marco-mini", device: str = None, batch_size: int = 32):
        """
        Initialize reranker.

        Args:
            model_name: Name of cross-encoder model or full HF path
            device: Device to use (cuda/cpu), auto-detected if None
            batch_size: Batch size for scoring
        """
        from sentence_transformers import CrossEncoder

        if device:
            self.device = device
        elif TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        self.batch_size = batch_size

        # Resolve model name
        if model_name in self.SUPPORTED_MODELS:
            model_path = self.SUPPORTED_MODELS[model_name]
        else:
            model_path = model_name

        self.model = CrossEncoder(model_path, device=self.device)
        self.model_name = model_name

        logger.info(f" Reranker initialized: {model_path} on {self.device}")

    def rerank(
        self, query: str, documents: List, top_k: int = 5, return_scores: bool = False
    ) -> List:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of documents (strings or objects with .content)
            top_k: Number of top documents to return
            return_scores: If True, return RerankResult objects

        Returns:
            Reranked documents (top_k)
        """
        if not documents:
            return []

        # Extract text content
        doc_texts = []
        for doc in documents:
            if hasattr(doc, "content"):
                doc_texts.append(doc.content)
            elif isinstance(doc, dict):
                doc_texts.append(doc.get("content", str(doc)))
            else:
                doc_texts.append(str(doc))

        # Create query-document pairs
        pairs = [[query, doc_text] for doc_text in doc_texts]

        # Get relevance scores
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Create indexed scores
        indexed_scores = list(enumerate(scores))

        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top k
        top_results = indexed_scores[:top_k]

        if return_scores:
            # Return RerankResult objects
            results = []
            for idx, score in top_results:
                doc = documents[idx]
                metadata = {}
                if hasattr(doc, "metadata"):
                    metadata = doc.metadata
                elif isinstance(doc, dict):
                    metadata = {k: v for k, v in doc.items() if k != "content"}

                results.append(
                    RerankResult(
                        text=doc_texts[idx],
                        score=float(score),
                        original_index=idx,
                        metadata=metadata,
                    )
                )
            return results
        else:
            # Return original documents in new order
            return [documents[idx] for idx, _ in top_results]

    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        return float(self.model.predict([[query, document]])[0])


class CohereReranker:
    """
    Reranker using Cohere API (optional, requires API key).
    """

    def __init__(self, api_key: str = None, model: str = "rerank-english-v3.0"):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key (or set COHERE_API_KEY env var)
            model: Cohere rerank model name
        """
        import os

        try:
            import cohere
        except ImportError:
            raise ImportError("Install cohere: pip install cohere")

        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key required")

        self.client = cohere.Client(api_key=self.api_key)
        self.model = model

    def rerank(self, query: str, documents: List, top_k: int = 5) -> List[RerankResult]:
        """Rerank using Cohere API."""
        # Extract text content
        doc_texts = []
        for doc in documents:
            if hasattr(doc, "content"):
                doc_texts.append(doc.content)
            elif isinstance(doc, dict):
                doc_texts.append(doc.get("content", str(doc)))
            else:
                doc_texts.append(str(doc))

        response = self.client.rerank(
            query=query, documents=doc_texts, top_n=top_k, model=self.model, return_documents=True
        )

        results = []
        for result in response.results:
            results.append(
                RerankResult(
                    text=result.document.text,
                    score=result.relevance_score,
                    original_index=result.index,
                    metadata={},
                )
            )

        return results
