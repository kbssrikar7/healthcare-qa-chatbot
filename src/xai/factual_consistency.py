"""
Factual Consistency Checker using NLI for Hallucination Detection.

Uses Natural Language Inference (NLI) to verify that generated answers
are entailed by (supported by) the retrieved context. This helps detect
hallucinations where the model generates plausible but unsupported statements.
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import re
from loguru import logger


@dataclass
class ClaimResult:
    """Result of checking a single claim."""
    claim: str
    is_supported: bool
    confidence: float
    label: str  # ENTAILMENT, CONTRADICTION, NEUTRAL
    evidence_snippet: Optional[str] = None


@dataclass  
class ConsistencyResult:
    """Overall consistency check result."""
    is_consistent: bool
    consistency_score: float
    claim_results: List[ClaimResult]
    num_supported: int
    num_total: int
    summary: str


class FactualConsistencyChecker:
    """
    Check if generated answer is factually supported by retrieved context.
    
    Uses NLI (Natural Language Inference) to classify whether each claim
    in the answer is ENTAILED by, CONTRADICTS, or is NEUTRAL to the context.
    
    This helps detect:
    - Hallucinated facts not in the context
    - Contradictions with the source material
    - Unsupported generalizations
    """
    
    NLI_MODELS = {
        "deberta": "microsoft/deberta-base-mnli",
        "deberta-large": "microsoft/deberta-large-mnli", 
        "bart": "facebook/bart-large-mnli",
        "roberta": "roberta-large-mnli",
    }
    
    def __init__(
        self,
        model_name: str = "deberta",
        entailment_threshold: float = 0.7,
        consistency_threshold: float = 0.6,
        device: str = None
    ):
        """
        Initialize the factual consistency checker.
        
        Args:
            model_name: NLI model to use (key or HuggingFace path)
            entailment_threshold: Confidence threshold for entailment
            consistency_threshold: Minimum ratio of supported claims
            device: Device to run on (None for auto)
        """
        self.model_path = self.NLI_MODELS.get(model_name, model_name)
        self.entailment_threshold = entailment_threshold
        self.consistency_threshold = consistency_threshold
        self.device = device
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lazy load the NLI pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                logger.info(f"Loading NLI model: {self.model_path}")
                self._pipeline = pipeline(
                    "text-classification",
                    model=self.model_path,
                    device=self.device
                )
                logger.info("NLI model loaded successfully")
            except ImportError:
                logger.warning("transformers not installed")
                self._pipeline = None
            except Exception as e:
                logger.error(f"Failed to load NLI model: {e}")
                self._pipeline = None
        return self._pipeline
    
    def check_consistency(
        self,
        answer: str,
        context: str,
    ) -> ConsistencyResult:
        """
        Check if answer claims are supported by context.
        
        Args:
            answer: Generated answer text
            context: Retrieved context/evidence
            
        Returns:
            ConsistencyResult with detailed analysis
        """
        if not answer or not context:
            return ConsistencyResult(
                is_consistent=False,
                consistency_score=0.0,
                claim_results=[],
                num_supported=0,
                num_total=0,
                summary="Empty answer or context"
            )
        
        # Extract claims from answer
        claims = self._extract_claims(answer)
        
        if not claims:
            return ConsistencyResult(
                is_consistent=True,
                consistency_score=1.0,
                claim_results=[],
                num_supported=0,
                num_total=0,
                summary="No factual claims detected"
            )
        
        # Check each claim against context
        claim_results = []
        supported_count = 0
        
        for claim in claims:
            result = self._check_single_claim(claim, context)
            claim_results.append(result)
            if result.is_supported:
                supported_count += 1
        
        # Calculate overall consistency
        consistency_score = supported_count / len(claims)
        is_consistent = consistency_score >= self.consistency_threshold
        
        # Generate summary
        if is_consistent:
            summary = f"{supported_count}/{len(claims)} claims are supported by the context"
        else:
            unsupported = [r.claim[:50] + "..." for r in claim_results if not r.is_supported]
            summary = f"Low consistency: {supported_count}/{len(claims)} supported. Unsupported: {unsupported[:2]}"
        
        return ConsistencyResult(
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            claim_results=claim_results,
            num_supported=supported_count,
            num_total=len(claims),
            summary=summary
        )
    
    def _check_single_claim(self, claim: str, context: str) -> ClaimResult:
        """Check if a single claim is supported by context."""
        if self.pipeline is None:
            # Fallback: simple keyword matching
            return self._fallback_check(claim, context)
        
        try:
            # NLI format: Use proper pair encoding for MNLI models
            # Premise = context (what we know)
            # Hypothesis = claim (what we're checking)
            # Pass as dict to ensure tokenizer uses correct pair encoding
            raw = self.pipeline({"text": context[:1024], "text_pair": claim})
            result = raw[0] if isinstance(raw, list) else raw
            label = result['label'].upper()
            score = result['score']
            
            # Map labels (different models have different formats)
            if 'ENTAIL' in label:
                is_supported = score >= self.entailment_threshold
                label = 'ENTAILMENT'
            elif 'CONTRADICT' in label:
                is_supported = False
                label = 'CONTRADICTION'
            else:
                is_supported = False
                label = 'NEUTRAL'
            
            # Find supporting evidence snippet
            evidence = self._find_evidence(claim, context) if is_supported else None
            
            return ClaimResult(
                claim=claim,
                is_supported=is_supported,
                confidence=score,
                label=label,
                evidence_snippet=evidence
            )
            
        except Exception as e:
            logger.error(f"NLI check failed: {e}")
            return self._fallback_check(claim, context)
    
    def _fallback_check(self, claim: str, context: str) -> ClaimResult:
        """Fallback checking using keyword overlap."""
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'must', 'can', 'this',
                     'that', 'it', 'its', 'of', 'in', 'to', 'for', 'with', 'on',
                     'at', 'by', 'from', 'or', 'and', 'but', 'if', 'then'}
        
        claim_words = claim_words - stopwords
        context_words = context_words - stopwords
        
        if not claim_words:
            return ClaimResult(
                claim=claim,
                is_supported=True,
                confidence=1.0,
                label='NEUTRAL',
                evidence_snippet=None
            )
        
        overlap = len(claim_words & context_words) / len(claim_words)
        is_supported = overlap > 0.4
        
        return ClaimResult(
            claim=claim,
            is_supported=is_supported,
            confidence=overlap,
            label='ENTAILMENT' if is_supported else 'NEUTRAL',
            evidence_snippet=None
        )
    
    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.
        
        Focuses on declarative sentences that make factual assertions.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip too short or too long
            if len(sentence) < 20 or len(sentence) > 300:
                continue
            
            # Skip questions
            if sentence.endswith('?'):
                continue
            
            # Skip purely instructional/advisory
            skip_patterns = [
                r'^(please|you should|it is recommended)',
                r'^(consult|see|visit).*(doctor|physician|healthcare)',
                r'^(note|disclaimer|warning)',
            ]
            
            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, sentence.lower()):
                    skip = True
                    break
            
            if not skip:
                claims.append(sentence)
        
        return claims
    
    def _find_evidence(self, claim: str, context: str, max_length: int = 200) -> Optional[str]:
        """Find the most relevant snippet from context supporting the claim."""
        claim_words = set(claim.lower().split())
        
        # Split context into chunks
        context_sentences = re.split(r'(?<=[.!?])\s+', context)
        
        best_score = 0
        best_snippet = None
        
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(claim_words & sentence_words)
            
            if overlap > best_score:
                best_score = overlap
                best_snippet = sentence[:max_length]
        
        return best_snippet


class MedicalFactChecker(FactualConsistencyChecker):
    """
    Extended fact checker with medical-specific logic.
    
    Adds checks for:
    - Dosage claims (verify numbers)
    - Drug name accuracy
    - Condition statements
    """
    
    # Medical entity patterns
    DOSAGE_PATTERN = r'\b(\d+(?:\.\d+)?)\s*(mg|ml|mcg|g|units?|tablets?|capsules?)\b'
    DRUG_PATTERN = r'\b[A-Z][a-z]+(?:in|ol|ide|ate|ine|one)\b'
    
    def check_medical_claims(
        self,
        answer: str,
        context: str
    ) -> Dict:
        """Extended medical-specific fact checking."""
        # Get base consistency
        base_result = self.check_consistency(answer, context)
        
        # Check for dosage claims
        dosage_claims = re.findall(self.DOSAGE_PATTERN, answer)
        context_dosages = re.findall(self.DOSAGE_PATTERN, context)
        
        # Verify dosages mentioned in answer appear in context
        dosage_verified = all(
            d in context_dosages 
            for d in dosage_claims
        ) if dosage_claims else True
        
        return {
            'base_result': base_result,
            'dosage_claims': dosage_claims,
            'dosage_verified': dosage_verified,
            'medical_safe': base_result.is_consistent and dosage_verified
        }
