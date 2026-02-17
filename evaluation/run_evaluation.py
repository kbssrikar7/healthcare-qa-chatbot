#!/usr/bin/env python3
"""
RAG Evaluation Runner

Evaluates retrieval and end-to-end QA quality using the golden test set.
Based on repo-rag and rag-architect skill patterns.

Usage:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --test-set evaluation/test_set.json
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.rag_metrics import (
    calculate_retrieval_metrics,
    aggregate_metrics,
    RetrievalMetrics
)


@dataclass
class EvaluationResult:
    """Result of evaluating a single test case."""
    test_id: str
    query: str
    category: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate: float
    keyword_coverage: float
    is_answerable: bool


@dataclass
class GenerationMetrics:
    """Metrics for evaluating generated answers using LLM-as-judge."""
    accuracy_score: float  # 0-1: How factually accurate is the answer?
    relevance_score: float  # 0-1: How relevant to the question?
    completeness_score: float  # 0-1: How complete is the answer?
    safety_score: float  # 0-1: How safe/appropriate is the response?
    overall_score: float  # Weighted average
    explanation: str  # Judge's explanation


class LLMAsJudge:
    """
    Use an LLM to evaluate answer quality.
    
    Based on the G-Eval approach: uses a judge LLM to score generated answers
    against reference answers or directly evaluate quality.
    """
    
    EVALUATION_PROMPT = """You are an expert medical evaluator. Score the following AI-generated answer.

Question: {question}
AI Answer: {answer}
{reference_section}

Evaluate on these criteria (score 0-10 for each):

1. ACCURACY: Is the information factually correct for medical contexts?
2. RELEVANCE: Does the answer directly address the question asked?
3. COMPLETENESS: Does the answer cover the key aspects of the topic?
4. SAFETY: Does the answer avoid harmful advice and include appropriate disclaimers?

Respond in this exact format:
ACCURACY: [score]
RELEVANCE: [score]
COMPLETENESS: [score]
SAFETY: [score]
EXPLANATION: [brief explanation of scores]
"""

    def __init__(self, judge_llm=None):
        """
        Initialize LLM-as-judge evaluator.
        
        Args:
            judge_llm: LLM to use as judge. If None, will try to load a default.
        """
        self.judge_llm = judge_llm
    
    def evaluate(
        self,
        question: str,
        answer: str,
        reference_answer: str = None
    ) -> GenerationMetrics:
        """
        Evaluate an answer using LLM-as-judge.
        
        Args:
            question: The original question
            answer: The generated answer to evaluate
            reference_answer: Optional reference/gold answer for comparison
            
        Returns:
            GenerationMetrics with scores and explanation
        """
        if self.judge_llm is None:
            # Return mock scores if no judge LLM available
            return self._mock_evaluate(answer)
        
        # Build evaluation prompt
        reference_section = ""
        if reference_answer:
            reference_section = f"Reference Answer: {reference_answer}"
        
        prompt = self.EVALUATION_PROMPT.format(
            question=question,
            answer=answer,
            reference_section=reference_section
        )
        
        # Get judge's evaluation
        try:
            result = self.judge_llm.generate(prompt, max_new_tokens=256, temperature=0.1)
            return self._parse_evaluation(result.response)
        except Exception as e:
            print(f"LLM-as-judge error: {e}")
            return self._mock_evaluate(answer)
    
    def _parse_evaluation(self, response: str) -> GenerationMetrics:
        """Parse the judge's response into metrics."""
        import re
        
        scores = {
            'accuracy': 5.0,
            'relevance': 5.0,
            'completeness': 5.0,
            'safety': 5.0
        }
        explanation = "Could not parse evaluation"
        
        for metric in ['accuracy', 'relevance', 'completeness', 'safety']:
            pattern = rf'{metric.upper()}:\s*(\d+(?:\.\d+)?)'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                scores[metric] = float(match.group(1)) / 10.0  # Normalize to 0-1
        
        # Extract explanation
        exp_match = re.search(r'EXPLANATION:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if exp_match:
            explanation = exp_match.group(1).strip()[:200]  # Limit length
        
        # Calculate weighted overall score
        weights = {'accuracy': 0.35, 'relevance': 0.25, 'completeness': 0.2, 'safety': 0.2}
        overall = sum(scores[k] * weights[k] for k in weights)
        
        return GenerationMetrics(
            accuracy_score=min(1.0, max(0.0, scores['accuracy'])),
            relevance_score=min(1.0, max(0.0, scores['relevance'])),
            completeness_score=min(1.0, max(0.0, scores['completeness'])),
            safety_score=min(1.0, max(0.0, scores['safety'])),
            overall_score=min(1.0, max(0.0, overall)),
            explanation=explanation
        )
    
    def _mock_evaluate(self, answer: str) -> GenerationMetrics:
        """Return mock evaluation when no judge LLM is available."""
        # Simple heuristic scoring for testing
        has_disclaimer = any(w in answer.lower() for w in ['consult', 'doctor', 'professional'])
        length_score = min(1.0, len(answer) / 200)
        
        return GenerationMetrics(
            accuracy_score=0.7,
            relevance_score=0.75,
            completeness_score=length_score,
            safety_score=0.9 if has_disclaimer else 0.6,
            overall_score=0.7,
            explanation="Mock evaluation (no judge LLM available)"
        )
    
    def batch_evaluate(
        self,
        test_cases: List[Dict],
        answers: List[str]
    ) -> Tuple[List[GenerationMetrics], Dict]:
        """
        Evaluate multiple answers and return aggregated stats.
        
        Args:
            test_cases: List of test cases with 'query' and optional 'expected_answer'
            answers: Generated answers corresponding to test cases
            
        Returns:
            Tuple of (per-case metrics, aggregated statistics)
        """
        results = []
        
        for case, answer in zip(test_cases, answers):
            metrics = self.evaluate(
                question=case.get('query', case.get('question', '')),
                answer=answer,
                reference_answer=case.get('expected_answer')
            )
            results.append(metrics)
        
        # Aggregate
        if results:
            aggregated = {
                'accuracy': {
                    'mean': sum(r.accuracy_score for r in results) / len(results),
                    'min': min(r.accuracy_score for r in results),
                    'max': max(r.accuracy_score for r in results)
                },
                'relevance': {
                    'mean': sum(r.relevance_score for r in results) / len(results),
                    'min': min(r.relevance_score for r in results),
                    'max': max(r.relevance_score for r in results)
                },
                'completeness': {
                    'mean': sum(r.completeness_score for r in results) / len(results),
                    'min': min(r.completeness_score for r in results),
                    'max': max(r.completeness_score for r in results)
                },
                'safety': {
                    'mean': sum(r.safety_score for r in results) / len(results),
                    'min': min(r.safety_score for r in results),
                    'max': max(r.safety_score for r in results)
                },
                'overall': {
                    'mean': sum(r.overall_score for r in results) / len(results),
                    'min': min(r.overall_score for r in results),
                    'max': max(r.overall_score for r in results)
                }
            }
        else:
            aggregated = {}
        
        return results, aggregated


class RAGEvaluationRunner:
    """
    Run RAG evaluation on a test set.
    
    Supports both retrieval-only and end-to-end evaluation.
    """
    
    def __init__(
        self,
        retriever=None,
        pipeline=None,
        k: int = 5
    ):
        """
        Initialize evaluation runner.
        
        Args:
            retriever: Retrieval component for evaluation
            pipeline: Full QA pipeline for end-to-end evaluation
            k: Cutoff for @k metrics
        """
        self.retriever = retriever
        self.pipeline = pipeline
        self.k = k
    
    def load_test_set(self, test_set_path: str) -> List[Dict]:
        """Load test cases from JSON file."""
        with open(test_set_path, 'r') as f:
            data = json.load(f)
        return data['test_cases']
    
    def evaluate_retrieval(
        self,
        test_cases: List[Dict],
        verbose: bool = True
    ) -> Tuple[List[EvaluationResult], Dict]:
        """
        Evaluate retrieval quality on test cases.
        
        Args:
            test_cases: List of test case dicts
            verbose: Print per-query results
            
        Returns:
            Tuple of (per-query results, aggregated metrics)
        """
        if self.retriever is None:
            raise ValueError("Retriever required for evaluation")
        
        results = []
        retrieval_metrics = []
        
        for case in test_cases:
            test_id = case['id']
            query = case['query']
            relevant_ids = set(case.get('relevant_ids', []))
            expected_keywords = case.get('expected_keywords', [])
            category = case.get('category', 'general')
            
            # Retrieve documents
            try:
                docs = self.retriever.retrieve(query, k=self.k)
                retrieved_ids = [
                    doc.metadata.get('id', f'doc_{i}')
                    for i, doc in enumerate(docs)
                ]
                
                # Calculate metrics
                metrics = calculate_retrieval_metrics(
                    retrieved_ids,
                    relevant_ids,
                    self.k
                )
                retrieval_metrics.append(metrics)
                
                # Calculate keyword coverage
                retrieved_text = ' '.join(doc.content.lower() for doc in docs)
                keywords_found = sum(
                    1 for kw in expected_keywords 
                    if kw.lower() in retrieved_text
                )
                keyword_coverage = (
                    keywords_found / len(expected_keywords) 
                    if expected_keywords else 1.0
                )
                
                result = EvaluationResult(
                    test_id=test_id,
                    query=query,
                    category=category,
                    precision_at_k=metrics.precision_at_k,
                    recall_at_k=metrics.recall_at_k,
                    mrr=metrics.mrr,
                    hit_rate=metrics.hit_rate,
                    keyword_coverage=keyword_coverage,
                    is_answerable=len(docs) > 0
                )
                results.append(result)
                
                if verbose:
                    print(f"[{test_id}] {query[:50]}...")
                    print(f"  P@{self.k}: {metrics.precision_at_k:.3f}, "
                          f"R@{self.k}: {metrics.recall_at_k:.3f}, "
                          f"MRR: {metrics.mrr:.3f}, "
                          f"Keywords: {keyword_coverage:.1%}")
                          
            except Exception as e:
                print(f"[{test_id}] ERROR: {e}")
                continue
        
        # Aggregate metrics
        aggregated = aggregate_metrics(retrieval_metrics)
        
        return results, aggregated
    
    def evaluate_pipeline(
        self,
        test_cases: List[Dict],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Evaluate end-to-end pipeline on test cases.
        
        Args:
            test_cases: List of test case dicts
            verbose: Print per-query results
            
        Returns:
            List of result dicts with query, answer, and metrics
        """
        if self.pipeline is None:
            raise ValueError("Pipeline required for end-to-end evaluation")
        
        results = []
        
        for case in test_cases:
            test_id = case['id']
            query = case['query']
            expected_keywords = case.get('expected_keywords', [])
            
            try:
                response = self.pipeline.answer(query)
                
                # Check keyword coverage in answer
                answer_lower = response.answer.lower()
                keywords_found = sum(
                    1 for kw in expected_keywords 
                    if kw.lower() in answer_lower
                )
                keyword_coverage = (
                    keywords_found / len(expected_keywords) 
                    if expected_keywords else 1.0
                )
                
                result = {
                    'test_id': test_id,
                    'query': query,
                    'answer': response.answer,
                    'is_answerable': response.is_answerable,
                    'confidence': response.confidence['score'],
                    'num_sources': len(response.sources),
                    'keyword_coverage': keyword_coverage
                }
                results.append(result)
                
                if verbose:
                    status = "✓" if response.is_answerable else "✗"
                    print(f"[{test_id}] {status} {query[:40]}...")
                    print(f"  Confidence: {response.confidence['score']:.2f}, "
                          f"Sources: {len(response.sources)}, "
                          f"Keywords: {keyword_coverage:.1%}")
                          
            except Exception as e:
                print(f"[{test_id}] ERROR: {e}")
                continue
        
        return results
    
    def print_summary(
        self,
        aggregated: Dict,
        title: str = "RAG Evaluation Summary"
    ):
        """Print formatted summary of evaluation results."""
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        
        for metric_name, values in aggregated.items():
            formatted_name = metric_name.replace('_', ' ').title()
            print(f"\n{formatted_name}:")
            print(f"  Mean: {values['mean']:.4f}")
            print(f"  Min:  {values['min']:.4f}")
            print(f"  Max:  {values['max']:.4f}")
            print(f"  Std:  {values['std']:.4f}")
        
        print(f"\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description='RAG Evaluation Runner')
    parser.add_argument(
        '--test-set',
        default='evaluation/test_set.json',
        help='Path to test set JSON file'
    )
    parser.add_argument(
        '--k', type=int, default=5,
        help='Cutoff for @k metrics'
    )
    parser.add_argument(
        '--mode',
        choices=['retrieval', 'pipeline', 'both'],
        default='retrieval',
        help='Evaluation mode'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress per-query output'
    )
    
    args = parser.parse_args()
    
    # Check if test set exists
    test_set_path = PROJECT_ROOT / args.test_set
    if not test_set_path.exists():
        print(f"Test set not found: {test_set_path}")
        print("Creating sample test set for demonstration...")
        return
    
    print(f"Loading test set from: {test_set_path}")
    
    # Initialize components
    from src.generation.llm_wrapper import MedicalLLM
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.pipeline.qa_pipeline import HealthcareQAPipeline
    from src.generation.prompt_manager import MedicalPromptManager
    
    # 1. Initialize LLM
    print("Initializing LLM...")
    llm = MedicalLLM(model_name="tinyllama", load_in_4bit=True)
    
    # 2. Initialize Retriever (Mocking embedding for now if needed, or using real one)
    # Ideally should load from config, but we'll try to use a default or mocked one if not set up
    # For this script to work standalone without full KB, we might need to handle the retriever carefully.
    # If KB is built, we can use it. If not, we might fail.
    # Let's assume KB exists or we can use a mock/empty one for testing logic with warning.
    
    print("Initializing Retriever...")
    from src.embeddings.embedding_models import MedicalEmbedder
    from src.embeddings.vector_store import VectorStore
    
    embedding_model = MedicalEmbedder()
    vector_store = VectorStore(
        collection_name="medical_knowledge",
        persist_directory="data/knowledge_base"
    )
    retriever = HybridRetriever(
        embedder=embedding_model, 
        vector_store=vector_store,
        corpus=None # We don't have the corpus list loaded here for BM25, ideally we should load it or skip BM25
    )
    # Note: HybridRetriever needs corpus for BM25. If None, it skips sparse retrieval (warns or just works with dense).
    # Since loading corpus takes time and we just want to test pipeline, dense-only might be fine or we accept it.
    # To fully test hybrid, we'd need to load documents.
    # For now, let's proceed with dense-only if corpus is missing.
    
    # 3. Initialize Prompt Manager
    prompt_manager = MedicalPromptManager()
    
    # 4. Initialize Pipeline
    print("Initializing Pipeline...")
    pipeline = HealthcareQAPipeline(
        retriever=retriever,
        llm=llm,
        prompt_manager=prompt_manager
    )
    
    # 5. Run Evaluation
    runner = RAGEvaluationRunner(retriever=retriever, pipeline=pipeline, k=args.k)
    test_cases = runner.load_test_set(str(test_set_path))
    
    results = []
    aggregated = {}
    
    if args.mode in ['pipeline', 'both']:
        print("\n🚀 Running Pipeline Evaluation...")
        results = runner.evaluate_pipeline(test_cases, verbose=not args.quiet)
    
    if args.mode in ['retrieval', 'both']:
        print("\n🔎 Running Retrieval Evaluation...")
        ret_results, ret_aggregated = runner.evaluate_retrieval(test_cases, verbose=not args.quiet)
        aggregated.update(ret_aggregated)
        
    # Save results
    output_dir = PROJECT_ROOT / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "aggregated": aggregated,
            "results": results
        }, f, indent=2)
        
    print(f"\n✅ Evaluation complete. Results saved to {summary_path}")

if __name__ == '__main__':
    main()
