"""
RAG evaluation metrics for measuring retrieval and generation quality.

Implements core metrics from RAG skill files:
- Retrieval: Precision@k, Recall@k, MRR, NDCG@k, Hit Rate
- Generation: Faithfulness, Answer Relevance (via LLM-as-judge)
"""
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    precision_at_k: float
    recall_at_k: float
    hit_rate: float
    mrr: float
    ndcg_at_k: float = 0.0
    
    def __str__(self) -> str:
        return (
            f"Precision@k: {self.precision_at_k:.3f}, "
            f"Recall@k: {self.recall_at_k:.3f}, "
            f"MRR: {self.mrr:.3f}, "
            f"NDCG@k: {self.ndcg_at_k:.3f}, "
            f"Hit Rate: {self.hit_rate:.3f}"
        )


@dataclass
class GenerationMetrics:
    """Container for generation evaluation metrics."""
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    
    def __str__(self) -> str:
        return (
            f"Faithfulness: {self.faithfulness:.3f}, "
            f"Answer Relevance: {self.answer_relevance:.3f}, "
            f"Context Relevance: {self.context_relevance:.3f}"
        )


def calculate_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Calculate Precision@k: proportion of retrieved docs that are relevant.
    
    Formula: |retrieved ∩ relevant| / k
    """
    top_k = set(retrieved_ids[:k])
    relevant_in_top_k = len(top_k & relevant_ids)
    return relevant_in_top_k / k if k > 0 else 0.0


def calculate_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Calculate Recall@k: proportion of relevant docs that were retrieved.
    
    Formula: |retrieved ∩ relevant| / |relevant|
    """
    top_k = set(retrieved_ids[:k])
    relevant_in_top_k = len(top_k & relevant_ids)
    return relevant_in_top_k / len(relevant_ids) if relevant_ids else 0.0


def calculate_hit_rate(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Calculate Hit Rate: 1 if any relevant doc in top-k, else 0.
    
    Binary success metric for retrieval.
    """
    top_k = set(retrieved_ids[:k])
    return 1.0 if (top_k & relevant_ids) else 0.0


def calculate_mrr(
    retrieved_ids: List[str],
    relevant_ids: Set[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Formula: 1 / rank of first relevant document
    """
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def _dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k.
    
    Formula: sum(rel_i / log2(i + 1)) for i in 1..k
    """
    scores = np.array(relevance_scores[:k])
    if len(scores) == 0:
        return 0.0
    
    discounts = np.log2(np.arange(2, len(scores) + 2))
    return float(np.sum(scores / discounts))


def calculate_ndcg_at_k(
    retrieved_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    Args:
        retrieved_ids: List of retrieved document IDs in order
        relevance_scores: Dict mapping doc_id to graded relevance (e.g., 0, 1, 2, 3)
        k: Cutoff position
        
    Returns:
        NDCG@k score (0.0 to 1.0)
    """
    # Get relevance scores for retrieved docs
    retrieved_relevance = [
        relevance_scores.get(doc_id, 0.0)
        for doc_id in retrieved_ids[:k]
    ]
    
    # Calculate DCG for retrieved order
    dcg = _dcg_at_k(retrieved_relevance, k)
    
    # Calculate ideal DCG (perfect ranking)
    ideal_relevance = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = _dcg_at_k(ideal_relevance, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
    relevance_scores: Optional[Dict[str, float]] = None
) -> RetrievalMetrics:
    """
    Calculate all retrieval metrics in one call.
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of ground truth relevant document IDs
        k: Cutoff for @k metrics
        relevance_scores: Optional graded relevance scores for NDCG
        
    Returns:
        RetrievalMetrics dataclass with all scores
    """
    precision = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
    recall = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
    hit_rate = calculate_hit_rate(retrieved_ids, relevant_ids, k)
    mrr = calculate_mrr(retrieved_ids, relevant_ids)
    
    # NDCG requires graded relevance
    if relevance_scores is not None:
        ndcg = calculate_ndcg_at_k(retrieved_ids, relevance_scores, k)
    else:
        # Use binary relevance (1 if relevant, 0 otherwise)
        binary_relevance = {doc_id: 1.0 for doc_id in relevant_ids}
        ndcg = calculate_ndcg_at_k(retrieved_ids, binary_relevance, k)
    
    return RetrievalMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
        hit_rate=hit_rate,
        mrr=mrr,
        ndcg_at_k=ndcg
    )


def aggregate_metrics(
    metrics_list: List[RetrievalMetrics]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple queries.
    
    Returns:
        Dict with mean, min, max, std for each metric
    """
    if not metrics_list:
        return {}
    
    result = {}
    metric_names = ['precision_at_k', 'recall_at_k', 'hit_rate', 'mrr', 'ndcg_at_k']
    
    for name in metric_names:
        values = [getattr(m, name) for m in metrics_list]
        result[name] = {
            'mean': float(np.mean(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'std': float(np.std(values))
        }
    
    return result


class RAGEvaluator:
    """
    Evaluate RAG system on test cases.
    
    Supports both retrieval-only and end-to-end evaluation.
    """
    
    def __init__(
        self,
        retriever=None,
        generator=None,
        llm_judge=None
    ):
        """
        Initialize evaluator.
        
        Args:
            retriever: Retrieval component for testing
            generator: Generation component for testing
            llm_judge: LLM for faithfulness/relevance scoring
        """
        self.retriever = retriever
        self.generator = generator
        self.llm_judge = llm_judge
    
    def evaluate_retrieval(
        self,
        test_cases: List[Dict],
        k: int = 5
    ) -> Tuple[List[RetrievalMetrics], Dict]:
        """
        Evaluate retrieval on test cases.
        
        Args:
            test_cases: List of dicts with 'query' and 'relevant_ids' keys
            k: Cutoff for @k metrics
            
        Returns:
            Tuple of (per-query metrics, aggregated metrics)
        """
        if self.retriever is None:
            raise ValueError("Retriever required for retrieval evaluation")
        
        per_query_metrics = []
        
        for case in test_cases:
            query = case['query']
            relevant_ids = set(case['relevant_ids'])
            relevance_scores = case.get('relevance_scores')
            
            # Retrieve documents
            results = self.retriever.retrieve(query, k=k)
            retrieved_ids = [
                r.metadata.get('id', r.content[:50])
                for r in results
            ]
            
            # Calculate metrics
            metrics = calculate_retrieval_metrics(
                retrieved_ids,
                relevant_ids,
                k,
                relevance_scores
            )
            per_query_metrics.append(metrics)
        
        aggregated = aggregate_metrics(per_query_metrics)
        
        return per_query_metrics, aggregated
    
    def evaluate_faithfulness(
        self,
        answer: str,
        context: str,
        question: str
    ) -> float:
        """
        Evaluate if answer is grounded in the provided context.
        
        Uses LLM-as-judge pattern.
        
        Returns:
            Faithfulness score (0.0 to 1.0)
        """
        if self.llm_judge is None:
            # Fallback: simple word overlap heuristic
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            if not answer_words:
                return 0.0
            overlap = len(answer_words & context_words)
            return min(1.0, overlap / len(answer_words))
        
        # LLM-based evaluation
        prompt = f"""Evaluate if the answer is fully supported by the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Score from 0 to 1 where:
- 1.0: Every claim in the answer is verifiable from context
- 0.5: Most claims are supported but some are not
- 0.0: Answer contains significant unsupported claims

Respond with only a number between 0 and 1."""

        try:
            response = self.llm_judge.generate(prompt, max_new_tokens=10)
            score = float(response.response.strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5  # Default fallback
    
    def print_summary(
        self,
        aggregated_metrics: Dict,
        title: str = "Retrieval Evaluation Summary"
    ):
        """Print formatted summary of aggregated metrics."""
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")
        
        for metric_name, values in aggregated_metrics.items():
            formatted_name = metric_name.replace('_', ' ').title()
            print(f"\n{formatted_name}:")
            print(f"  Mean: {values['mean']:.4f}")
            print(f"  Min:  {values['min']:.4f}")
            print(f"  Max:  {values['max']:.4f}")
            print(f"  Std:  {values['std']:.4f}")
        
        print(f"\n{'=' * 50}\n")


# =============================================================================
# Extended Evaluation Metrics (Improvement #8)
# =============================================================================

def compute_hallucination_rate(
    answers: List[str],
    contexts: List[str],
    min_term_length: int = 4,
    grounding_threshold: float = 0.3,
) -> float:
    """
    Estimate hallucination rate by checking if key claims in answers
    are grounded in the provided context.
    
    Args:
        answers: Generated answers
        contexts: Retrieved contexts used for generation
        min_term_length: Minimum word length to consider as key term
        grounding_threshold: Ratio below which a sentence is 'hallucinated'
        
    Returns:
        Hallucination rate (0.0 = fully grounded, 1.0 = fully hallucinated)
    """
    if not answers:
        return 0.0
    
    hallucinated_count = 0
    for answer, context in zip(answers, contexts):
        sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
        for sentence in sentences:
            key_terms = [w.lower() for w in sentence.split() if len(w) > min_term_length]
            if key_terms:
                grounded = sum(1 for t in key_terms if t in context.lower()) / len(key_terms)
                if grounded < grounding_threshold:
                    hallucinated_count += 1
                    break  # One hallucinated sentence = answer is hallucinated
    
    return hallucinated_count / len(answers)


def compute_calibration_error(
    predicted_confidences: List[float],
    actual_correctness: List[float],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    
    Measures how well confidence scores align with actual accuracy.
    A well-calibrated model has ECE close to 0.
    
    Args:
        predicted_confidences: Model's confidence for each prediction
        actual_correctness: Binary correctness (1=correct, 0=incorrect)
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score (lower is better, 0.0 = perfectly calibrated)
    """
    if not predicted_confidences:
        return 0.0
    
    confs = np.array(predicted_confidences)
    accs = np.array(actual_correctness)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc = accs[mask].mean()
        weight = mask.sum() / len(confs)
        ece += weight * abs(avg_acc - avg_conf)
    
    return float(ece)


@dataclass
class LatencyStats:
    """Latency statistics for pipeline performance tracking."""
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    count: int
    
    def to_dict(self) -> Dict:
        return {
            "avg_ms": round(self.avg_ms, 1),
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "p99_ms": round(self.p99_ms, 1),
            "min_ms": round(self.min_ms, 1),
            "max_ms": round(self.max_ms, 1),
            "count": self.count,
        }


class LatencyTracker:
    """Track and compute latency statistics for pipeline stages."""
    
    def __init__(self):
        self._latencies: List[float] = []
    
    def record(self, latency_ms: float) -> None:
        """Record a latency measurement in milliseconds."""
        self._latencies.append(latency_ms)
    
    def get_stats(self) -> Optional[LatencyStats]:
        """Compute and return latency statistics."""
        if not self._latencies:
            return None
        arr = np.array(self._latencies)
        return LatencyStats(
            avg_ms=float(np.mean(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            count=len(arr),
        )
    
    def reset(self) -> None:
        """Reset all recorded latencies."""
        self._latencies.clear()


@dataclass
class ComprehensiveMetrics:
    """Full evaluation metrics combining retrieval, generation, and operational."""
    # Retrieval quality
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    
    # Answer quality
    faithfulness: float = 0.0
    hallucination_rate: float = 0.0
    
    # Confidence calibration
    calibration_error: float = 0.0
    avg_confidence: float = 0.0
    
    # Safety
    emergency_detection_recall: float = 0.0
    
    # Performance
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}

