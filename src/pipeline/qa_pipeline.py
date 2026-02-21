"""
Main QA pipeline orchestrating all components.

Enhanced with grounding gate (answerability check) based on RAG skill patterns.
Includes response caching for performance optimization.
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Optional
from dataclasses import dataclass

# Import configuration
try:
    from config.settings import config
except ImportError:
    config = None

# Import cache manager
try:
    from src.utils.cache_manager import CacheManager
except ImportError:
    CacheManager = None

try:
    from src.xai.rationale_generator import RationaleGenerator
except ImportError:
    RationaleGenerator = None


@dataclass
class QAResponse:
    """Complete response from the QA pipeline."""
    question: str
    answer: str
    sources: List[Dict]
    confidence: Dict
    attributions: List[Dict]
    disclaimer: str
    rationale: Optional[str] = None # Added rationale field
    is_answerable: bool = True  # New field for grounding gate result
    from_cache: bool = False    # Indicates if response was cached

class HealthcareQAPipeline:
    """
    Main pipeline orchestrating retrieval, generation, and XAI.
    
    Includes grounding gate for answerability checking based on ai-rag skill patterns.
    Supports response caching for improved performance.
    """
    
    # Response when question is not answerable from context
    UNANSWERABLE_RESPONSE = (
        "I don't have enough information in my knowledge base to answer this question accurately. "
        "Please consult a healthcare professional for specific medical advice."
    )
    
    def __init__(
        self,
        retriever,
        llm,
        prompt_manager,
        confidence_scorer=None,
        source_attributor=None,
        rationale_generator=None,
        enable_grounding_gate: bool = None,
        cache_manager: Optional["CacheManager"] = None,
        # Configurable thresholds (override config if provided)
        min_retrieval_score: float = None,
        min_relevant_docs: int = None
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.confidence_scorer = confidence_scorer
        self.source_attributor = source_attributor
        self.rationale_generator = rationale_generator
        
        # Initialize rationale generator if not provided but class is available
        if self.rationale_generator is None and RationaleGenerator is not None and self.llm:
             self.rationale_generator = RationaleGenerator(self.llm)

        # Load from config with fallbacks
        
        # Load from config with fallbacks
        pipeline_config = getattr(config, 'pipeline', None) if config else None
        
        self.enable_grounding_gate = enable_grounding_gate if enable_grounding_gate is not None else (
            getattr(pipeline_config, 'enable_grounding_gate', True) if pipeline_config else True
        )
        self.min_retrieval_score = min_retrieval_score if min_retrieval_score is not None else (
            getattr(pipeline_config, 'min_retrieval_score', 0.3) if pipeline_config else 0.3
        )
        self.min_relevant_docs = min_relevant_docs if min_relevant_docs is not None else (
            getattr(pipeline_config, 'min_relevant_docs', 1) if pipeline_config else 1
        )
        
        # Initialize cache manager
        self.cache_manager = cache_manager
        if self.cache_manager is None and CacheManager is not None:
            if pipeline_config and getattr(pipeline_config, 'enable_response_cache', False):
                self.cache_manager = CacheManager(
                    cache_dir=getattr(pipeline_config, 'cache_dir', 'data/cache'),
                    ttl_seconds=getattr(pipeline_config, 'cache_ttl_seconds', 3600),
                    max_memory_items=getattr(pipeline_config, 'max_cache_items', 1000)
                )
    
    def answer(
        self,
        question: str,
        num_documents: int = 5,
        include_explanation: bool = True,
        template_name: str = "medical_qa"
    ) -> QAResponse:
        """
        Answer a medical question with explanations.
        
        Includes grounding gate to check if context is sufficient.
        Uses caching for improved performance when enabled.
        """
        # 0. Check cache first
        if self.cache_manager:
            cached = self.cache_manager.get_cached_response(question)
            if cached:
                return QAResponse(
                    question=cached.get('question', question),
                    answer=cached.get('answer', ''),
                    sources=cached.get('sources', []),
                    confidence=cached.get('confidence', {}),
                    attributions=cached.get('attributions', []),
                    disclaimer=cached.get('disclaimer', ''),
                    rationale=cached.get('rationale', None),
                    is_answerable=cached.get('is_answerable', True),
                    from_cache=True
                )
        
        # 1. Retrieve relevant documents
        documents, context = self.retriever.retrieve_with_context(
            question,
            k=num_documents
        )
        
        # 2. GROUNDING GATE: Check if context is sufficient to answer
        is_answerable = True
        if self.enable_grounding_gate:
            is_answerable = self._check_answerability(documents)
        
        if not is_answerable:
            # Return safe response when question cannot be answered from context
            disclaimer = self.prompt_manager.get_medical_disclaimer()
            return QAResponse(
                question=question,
                answer=self.UNANSWERABLE_RESPONSE,
                sources=[],
                confidence={
                    "score": 0.0,
                    "level": "low",
                    "explanation": "Insufficient relevant context found in knowledge base"
                },
                attributions=[],
                disclaimer=disclaimer,
                rationale=None,
                is_answerable=False
            )
        
        # 3. Build prompt
        prompt = self.prompt_manager.build_prompt(
            question=question,
            context=context,
            template_name=template_name
        )
        
        # 4. Generate answer
        generation_result = self.llm.generate(
            prompt,
            max_new_tokens=256,
            return_probabilities=include_explanation
        )
        
        answer = generation_result.response
        
        # 4a. Clean answer - remove any training data artifacts
        answer = self._clean_answer(answer)
        
        # 4. Calculate confidence
        if self.confidence_scorer and include_explanation:
            retrieval_scores = [doc.score for doc in documents]
            confidence_result = self.confidence_scorer.calculate_confidence(
                generation_probs=generation_result.probabilities,
                retrieval_scores=retrieval_scores,
                num_sources=len(documents)
            )
            confidence = {
                "score": confidence_result.calibrated_score,
                "level": confidence_result.level,
                "explanation": confidence_result.explanation
            }
        else:
            confidence = {
                "score": 0.7,
                "level": "medium",
                "explanation": "Confidence scoring not available"
            }
        
        # 5. Attribute sources
        if self.source_attributor and include_explanation:
            doc_dicts = [
                {"content": doc.content, "source": doc.source, "url": doc.metadata.get("url", "")}
                for doc in documents
            ]
            attributions_list = self.source_attributor.attribute_answer(answer, doc_dicts)
            attributions = [
                {
                    "claim": a.claim,
                    "source": a.source,
                    "evidence": a.evidence,
                    "similarity": a.similarity_score
                }
                for a in attributions_list
            ]
        else:
            attributions = []
        
        # 6. Generate Rationale
        rationale = None
        if self.rationale_generator and include_explanation:
             combined_context = "\n".join([d.content for d in documents])
             rationale = self.rationale_generator.generate_rationale(
                 question=question,
                 answer=answer,
                 context=combined_context
             )

        # 7. Build source list
        sources = [
            {
                "source": doc.source,
                "content": doc.content,
                "score": doc.score,
                "url": doc.metadata.get("url", "")
            }
            for doc in documents
        ]
        
        # 7. Get disclaimer
        disclaimer = self.prompt_manager.get_medical_disclaimer()
        
        response = QAResponse(
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            attributions=attributions,
            disclaimer=disclaimer,
            rationale=rationale,
            is_answerable=True,
            from_cache=False
        )
        
        # 8. Cache the response
        if self.cache_manager:
            self.cache_manager.cache_response(question, {
                'question': question,
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'attributions': attributions,
                'disclaimer': disclaimer,
                'rationale': rationale,
                'is_answerable': True
            })
        
        return response
    
    def _check_answerability(self, documents: List) -> bool:
        """
        Check if retrieved documents are sufficient to answer the question.
        
        Grounding gate based on ai-rag skill pattern.
        
        Args:
            documents: Retrieved documents with scores
            
        Returns:
            True if question appears answerable, False otherwise
        """
        if not documents:
            return False
        
        # Check if we have enough relevant documents (using configurable thresholds)
        relevant_docs = [
            doc for doc in documents 
            if doc.score >= self.min_retrieval_score
        ]
        
        if len(relevant_docs) < self.min_relevant_docs:
            return False
        
        # Check if top document has reasonable relevance
        top_score = max(doc.score for doc in documents)
        if top_score < self.min_retrieval_score:
            return False
        
        return True
    
    def _clean_answer(self, answer: str) -> str:
        """
        Pipeline-level answer cleaning to strip training data artifacts.
        
        This is a safety net in addition to MedicalLLM._clean_response().
        Catches patterns that may appear in cached or externally-generated responses.
        """
        if not answer:
            return answer
        
        # Strip leading answer prefixes the model may generate
        answer = answer.strip()
        for prefix in ['Answer:', 'Factual Answer:', 'Evidence-Based Answer:',
                       'Based on the reference text above, the answer is:',
                       'Based on the reference text above, the evidence-based answer is:',
                       'Based on the reference text,']:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Patterns to truncate at (everything after the first match is removed)
        truncate_patterns = [
            r'Best regards',
            r'Kind regards',
            r'Sincerely',
            r'\[Your Name\]',
            r'Chat Doctor',
            r'ChatDoctor',
            r'HealthCareMagic',
            r'Thank you for choosing',
            r'Thank you for using',
            r'If you have any further questions',
            r'please do not hesitate',
            r"don't hesitate to ask",
            r'I hope this helps',
            r'I hope this information',
            r'\nQuestion:',
            r'\nQ:',
            r'\nAnswer:',
            r'\[\d+\]\s*Source:',
        ]
        
        # Only match patterns AFTER first 50 chars to avoid truncating
        # legitimate content at the start of the answer
        min_match_pos = min(50, len(answer))
        earliest_pos = len(answer)
        for pattern in truncate_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match and match.start() >= min_match_pos and match.start() < earliest_pos:
                earliest_pos = match.start()
        
        if earliest_pos < len(answer):
            answer = answer[:earliest_pos]
        
        # Remove inline source refs like [1], [2], [3]
        answer = re.sub(r'\[\d+\]', '', answer)
        
        # Clean trailing incomplete sentence
        answer = answer.rstrip()
        if answer and answer[-1] not in '.!?:)"':
            last_period = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
            if last_period > len(answer) * 0.3:
                answer = answer[:last_period + 1]
        
        # Clean extra whitespace
        answer = re.sub(r'  +', ' ', answer)
        answer = re.sub(r'\n\n\n+', '\n\n', answer)
        
        return answer.strip()
    
    def batch_answer(
        self,
        questions: List[str],
        **kwargs
    ) -> List[QAResponse]:
        """Answer multiple questions."""
        return [self.answer(q, **kwargs) for q in questions]
