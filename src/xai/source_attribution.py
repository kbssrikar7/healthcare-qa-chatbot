"""
Source attribution for medical QA.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
import numpy as np

# Try to import spacy for better NLP, fall back to regex
try:
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    spacy = None

# Lazy load spacy model
_nlp = None

def _get_nlp():
    """Lazy load spacy model for sentence segmentation."""
    global _nlp
    if _nlp is None and NLP_AVAILABLE:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed, fallback to regex
            pass
    return _nlp


@dataclass
class Attribution:
    """A single source attribution."""
    claim: str
    source: str
    evidence: str
    similarity_score: float
    url: Optional[str] = None

class SourceAttributor:
    """
    Attribute generated claims to source documents.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.3,
        embedder = None,
        use_nlp: bool = True
    ):
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder
        self.use_nlp = use_nlp and NLP_AVAILABLE
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from generated text using NLP sentence segmentation.
        
        Uses spacy for proper sentence boundary detection when available,
        with regex fallback that handles medical abbreviations.
        """
        # Try NLP-based extraction first
        if self.use_nlp:
            nlp = _get_nlp()
            if nlp:
                return self._extract_claims_nlp(text, nlp)
        
        # Fallback to improved regex-based extraction
        return self._extract_claims_regex(text)
    
    def _extract_claims_nlp(self, text: str, nlp) -> List[str]:
        """Extract claims using spacy NLP pipeline."""
        doc = nlp(text)
        claims = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            # Filter criteria for factual claims
            if self._is_factual_claim(sent_text):
                claims.append(sent_text)
        
        return claims
    
    def _extract_claims_regex(self, text: str) -> List[str]:
        """
        Extract claims using improved regex that handles medical abbreviations.
        
        Handles cases like: "Dr.", "mg.", "ml.", "vs.", "i.e.", "e.g."
        """
        # Protect common medical abbreviations from sentence splitting
        protected = text
        abbrevs = [
            (r'\bDr\.', 'DR_ABBR'),
            (r'\bMr\.', 'MR_ABBR'),
            (r'\bMs\.', 'MS_ABBR'),
            (r'\bvs\.', 'VS_ABBR'),
            (r'\bi\.e\.', 'IE_ABBR'),
            (r'\be\.g\.', 'EG_ABBR'),
            (r'\betc\.', 'ETC_ABBR'),
            (r'(\d+)\s*mg\.', r'\1_MG_ABBR'),
            (r'(\d+)\s*ml\.', r'\1_ML_ABBR'),
        ]
        
        for pattern, replacement in abbrevs:
            protected = re.sub(pattern, replacement, protected, flags=re.IGNORECASE)
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', protected)
        
        # Restore abbreviations and filter
        claims = []
        for sent in sentences:
            # Restore abbreviations
            restored = sent
            for pattern, replacement in abbrevs:
                restored = restored.replace(replacement.replace(r'\1', ''), pattern.replace(r'\b', '').replace('\\', ''))
            
            restored = restored.strip()
            if self._is_factual_claim(restored):
                claims.append(restored)
        
        return claims
    
    def _is_factual_claim(self, text: str) -> bool:
        """Check if text is likely a factual claim worth attributing."""
        # Minimum length check
        if len(text) < 20:
            return False
        
        # Skip questions
        if text.endswith('?'):
            return False
        
        # Skip hedged/uncertain statements
        hedges = ('i think', 'maybe', 'perhaps', 'possibly', 'might be', 'could be')
        if text.lower().startswith(hedges):
            return False
        
        # Skip imperatives/instructions (common in medical text but not claims)
        imperatives = ('please', 'consult', 'see your doctor', 'always consult')
        if text.lower().startswith(imperatives):
            return False
        
        return True
    
    def find_evidence(
        self,
        claim: str,
        documents: List[Dict]
    ) -> Optional[Dict]:
        """Find best matching evidence for a claim."""
        best_match = None
        best_score = 0.0
        
        for doc in documents:
            content = doc.get("content", str(doc))
            source = doc.get("source", "Unknown")
            
            # Calculate similarity (using embedder if available)
            if self.embedder:
                try:
                    score = self._embedding_similarity(claim, content)
                except Exception:
                    score = self._text_overlap(claim, content)
            else:
                score = self._text_overlap(claim, content)
            
            if score > best_score:
                best_score = score
                best_match = {
                    "source": source,
                    "evidence": self._extract_relevant_snippet(claim, content),
                    "score": score,
                    "url": doc.get("url", "")
                }
        
        if best_score >= self.similarity_threshold:
            return best_match
        return None
    
    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate embedding-based similarity."""
        emb1 = self.embedder.embed_query(text1)
        emb2 = self.embedder.embed_query(text2)
        return float(np.dot(emb1, emb2))
    
    def _text_overlap(self, claim: str, content: str) -> float:
        """Calculate word overlap similarity."""
        claim_words = set(claim.lower().split())
        content_words = set(content.lower().split())
        
        if not claim_words:
            return 0.0
        
        overlap = claim_words.intersection(content_words)
        return len(overlap) / len(claim_words)
    
    def _extract_relevant_snippet(
        self,
        claim: str,
        content: str,
        max_length: int = 200
    ) -> str:
        """Extract the most relevant snippet from content."""
        # Find best matching window
        claim_words = claim.lower().split()
        content_lower = content.lower()
        
        best_start = 0
        best_score = 0
        
        words = content.split()
        for i in range(len(words) - 10):
            window = ' '.join(words[i:i+20]).lower()
            score = sum(1 for w in claim_words if w in window)
            if score > best_score:
                best_score = score
                best_start = i
        
        # Extract snippet
        snippet_words = words[best_start:best_start + 30]
        snippet = ' '.join(snippet_words)
        
        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "..."
        
        return snippet
    
    def attribute_answer(
        self,
        answer: str,
        documents: List[Dict]
    ) -> List[Attribution]:
        """Attribute an entire answer to sources."""
        claims = self.extract_claims(answer)
        attributions = []
        
        for claim in claims:
            evidence = self.find_evidence(claim, documents)
            
            if evidence:
                attributions.append(Attribution(
                    claim=claim,
                    source=evidence["source"],
                    evidence=evidence["evidence"],
                    similarity_score=evidence["score"],
                    url=evidence.get("url")
                ))
            else:
                # Mark as unsupported
                attributions.append(Attribution(
                    claim=claim,
                    source="Unsupported",
                    evidence="No matching evidence found",
                    similarity_score=0.0
                ))
        
        return attributions
    
    def calculate_attribution_coverage(
        self,
        attributions: List[Attribution]
    ) -> float:
        """Calculate what percentage of claims are attributed."""
        if not attributions:
            return 0.0
        
        supported = sum(
            1 for a in attributions 
            if a.source != "Unsupported"
        )
        return supported / len(attributions)
