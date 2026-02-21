"""
LangGraph Node Functions for Healthcare RAG Pipeline.

Each node receives the full state and returns partial updates.
Nodes are responsible for a single step in the RAG workflow.
"""
from typing import Dict, Any, List
from langchain_core.documents import Document

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.langgraph.langgraph_state import HealthcareRAGState


# Constants
MIN_RELEVANCE_SCORE = 0.3
MIN_RELEVANT_DOCS = 2
MAX_RETRY_COUNT = 2
LOW_CONFIDENCE_THRESHOLD = 0.5

UNANSWERABLE_RESPONSE = (
    "I don't have enough relevant information in my knowledge base to answer this question accurately. "
    "Please consult a healthcare professional for specific medical advice."
)

MEDICAL_DISCLAIMER = """
⚠️ MEDICAL DISCLAIMER: This information is for educational purposes only and is NOT a substitute 
for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
or other qualified health provider with any questions you may have regarding a medical condition.
"""


class HealthcareRAGNodes:
    """
    Node implementations for Healthcare RAG graph.
    
    Wraps existing components (retriever, LLM, XAI) and exposes
    them as LangGraph-compatible node functions.
    """
    
    def __init__(
        self,
        retriever,
        llm,
        confidence_scorer=None,
        source_attributor=None,
        rationale_generator=None,
        k: int = 5
    ):
        """
        Initialize nodes with existing components.
        
        Args:
            retriever: HybridRetriever instance
            llm: MedicalLLM instance
            confidence_scorer: Optional ConfidenceScorer
            source_attributor: Optional SourceAttributor
            rationale_generator: Optional RationaleGenerator
            k: Number of documents to retrieve
        """
        self.retriever = retriever
        self.llm = llm
        self.confidence_scorer = confidence_scorer
        self.source_attributor = source_attributor
        self.rationale_generator = rationale_generator
        self.k = k
    
    def retrieve_documents(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Retrieve documents from knowledge base.
        
        Uses the most recent query from query_history.
        """
        query = state.get("query_history", [state["question"]])[-1]
        
        try:
            # Use HybridRetriever
            docs = self.retriever.retrieve(query, k=self.k)
            
            # Convert to LangChain Documents
            lc_docs = []
            for doc in docs:
                lc_doc = Document(
                    page_content=doc.content if hasattr(doc, 'content') else str(doc),
                    metadata={
                        "source": doc.source if hasattr(doc, 'source') else "unknown",
                        "score": doc.score if hasattr(doc, 'score') else 0.5,
                        "url": doc.url if hasattr(doc, 'url') else ""
                    }
                )
                lc_docs.append(lc_doc)
            
            return {"documents": lc_docs}
        except Exception as e:
            return {
                "documents": [],
                "error": f"Retrieval error: {str(e)}"
            }
    
    def grade_relevance(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Grade document relevance to the question.
        
        Evaluates each document and counts relevant ones.
        """
        documents = state.get("documents", [])
        question = state["question"]
        
        grades = []
        relevant_count = 0
        
        for i, doc in enumerate(documents):
            score = doc.metadata.get("score", 0.5)
            content = doc.page_content.lower()
            query_terms = set(question.lower().split())
            content_terms = set(content.split())
            
            # Calculate term overlap
            term_overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            
            # Determine relevance
            if score >= 0.7 or (score >= MIN_RELEVANCE_SCORE and term_overlap > 0.4):
                relevance = "relevant"
                relevant_count += 1
            elif score >= MIN_RELEVANCE_SCORE:
                relevance = "ambiguous"
            else:
                relevance = "irrelevant"
            
            grades.append({
                "doc_index": i,
                "score": score,
                "term_overlap": term_overlap,
                "relevance": relevance
            })
        
        # Determine if answerable
        is_answerable = relevant_count >= MIN_RELEVANT_DOCS or (
            relevant_count >= 1 and len([g for g in grades if g["relevance"] == "ambiguous"]) >= 1
        )
        
        return {
            "doc_grades": grades,
            "is_answerable": is_answerable
        }
    
    def refine_query(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Refine query for better retrieval.
        
        Uses LLM to generate an improved query, or falls back to
        keyword extraction from partially relevant documents.
        """
        original_query = state["question"]
        documents = state.get("documents", [])
        retry_count = state.get("retry_count", 0)
        
        refined_query = None
        
        # Try LLM refinement
        if self.llm:
            try:
                prompt = f"""The search query "{original_query}" returned documents that weren't relevant enough for a medical question.

Suggest a more specific query that might find better medical information.
Return only the refined query, nothing else."""
                
                response = self.llm.generate(prompt, max_new_tokens=50)
                refined_query = response.response.strip() if hasattr(response, 'response') else str(response).strip()
            except Exception:
                pass
        
        # Fallback: extract keywords from documents
        if not refined_query and documents:
            # Get content from first document
            first_doc = documents[0]
            content = first_doc.page_content
            
            # Extract significant words
            words = [w for w in content.split()[:50] if len(w) > 4][:3]
            if words:
                refined_query = original_query + " " + " ".join(words)
        
        # If still no refinement, add generic medical terms
        if not refined_query:
            refined_query = original_query + " medical health treatment symptoms"
        
        return {
            "query_history": [refined_query],
            "retry_count": retry_count + 1
        }
    
    def generate_answer(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Generate answer using LLM with context.
        """
        question = state["question"]
        documents = state.get("documents", [])
        
        # Format context
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", f"Source {i+1}")
            context_parts.append(f"[{source}]: {doc.page_content}")
        
        context = "\n\n".join(context_parts[:5])  # Limit to top 5
        
        # Build prompt
        prompt = f"""You are a knowledgeable medical assistant. Answer the following question based on the provided context.

### Important Guidelines:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Use clear, patient-friendly language
4. NEVER provide diagnoses or prescriptions
5. Always recommend consulting a healthcare professional

### Context:
{context}

### Question:
{question}

### Answer:"""
        
        try:
            response = self.llm.generate(prompt, max_new_tokens=512)
            answer = response.response if hasattr(response, 'response') else str(response)
            
            return {
                "context": context,
                "answer": answer.strip()
            }
        except Exception as e:
            return {
                "context": context,
                "answer": UNANSWERABLE_RESPONSE,
                "error": f"Generation error: {str(e)}"
            }
    
    def verify_grounding(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Verify answer is grounded in context.
        
        Uses claim-based verification for more robust hallucination detection.
        Checks that factual claims in the answer are supported by context.
        """
        answer = state.get("answer", "")
        context = state.get("context", "")
        documents = state.get("documents", [])
        
        if not answer or not context:
            return {"is_grounded": False, "grounding_score": 0.0}
        
        # Method 1: Term overlap (fast baseline)
        term_overlap_score = self._calculate_term_overlap(answer, context)
        
        # Method 2: Claim-based verification (more accurate)
        claims = self._extract_claims(answer)
        claim_scores = []
        
        for claim in claims[:5]:  # Check top 5 claims
            best_match_score = 0.0
            for doc in documents:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                score = self._calculate_claim_support(claim, content)
                best_match_score = max(best_match_score, score)
            claim_scores.append(best_match_score)
        
        avg_claim_score = sum(claim_scores) / len(claim_scores) if claim_scores else 0.5
        
        # Combined grounding score (weighted average)
        grounding_score = 0.4 * term_overlap_score + 0.6 * avg_claim_score
        is_grounded = grounding_score > 0.35
        
        return {
            "is_grounded": is_grounded,
            "grounding_score": grounding_score
        }
    
    def _calculate_term_overlap(self, answer: str, context: str) -> float:
        """Calculate term overlap between answer and context."""
        answer_terms = self._extract_key_terms(answer)
        context_terms = self._extract_key_terms(context)
        
        if len(answer_terms) == 0:
            return 0.0
        
        overlap = len(answer_terms & context_terms)
        return overlap / len(answer_terms)
    
    def _extract_key_terms(self, text: str) -> set:
        """Extract meaningful terms, removing stopwords."""
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "can",
                     "this", "that", "these", "those", "it", "its", "of", "in",
                     "to", "for", "with", "on", "at", "by", "from", "or", "and",
                     "but", "if", "then", "so", "as", "what", "which", "who",
                     "when", "where", "why", "how", "all", "each", "every", "both",
                     "few", "more", "most", "other", "some", "such", "no", "not",
                     "only", "own", "same", "than", "too", "very", "just", "also"}
        words = set(text.lower().split())
        # Keep words longer than 3 characters
        return {w for w in words - stopwords if len(w) > 3}
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from answer text."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        for s in sentences:
            s = s.strip()
            # Include declarative sentences with enough content
            if len(s) > 20 and not s.strip().endswith('?'):
                # Skip disclaimers and recommendations
                skip_patterns = ['consult', 'recommend', 'important', 'disclaimer', 'please']
                if not any(p in s.lower() for p in skip_patterns):
                    claims.append(s)
        return claims
    
    def _calculate_claim_support(self, claim: str, context: str) -> float:
        """Calculate how well context supports a claim."""
        claim_terms = self._extract_key_terms(claim)
        context_lower = context.lower()
        
        if not claim_terms:
            return 0.5
        
        # Check for key term presence
        matches = sum(1 for word in claim_terms if word in context_lower)
        return matches / len(claim_terms)
    
    def enrich_xai(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Enrich response with XAI components.
        
        Adds confidence scoring, source attribution, and rationale.
        """
        question = state["question"]
        answer = state.get("answer", "")
        documents = state.get("documents", [])
        context = state.get("context", "")
        is_answerable = state.get("is_answerable", True)
        
        result = {}
        
        # Confidence scoring
        if self.confidence_scorer and is_answerable:
            try:
                retrieval_scores = [doc.metadata.get("score", 0.5) for doc in documents]
                confidence_result = self.confidence_scorer.calculate_confidence(
                    generation_probs=None,
                    retrieval_scores=retrieval_scores,
                    num_sources=len(documents)
                )
                result["confidence"] = {
                    "score": confidence_result.calibrated_score,
                    "level": confidence_result.level,
                    "explanation": confidence_result.explanation
                }
            except Exception:
                result["confidence"] = {
                    "score": 0.7,
                    "level": "medium",
                    "explanation": "Confidence scoring unavailable"
                }
        else:
            result["confidence"] = {
                "score": 0.0 if not is_answerable else 0.7,
                "level": "low" if not is_answerable else "medium",
                "explanation": "Insufficient context" if not is_answerable else "Default confidence"
            }
        
        # Source attribution
        if self.source_attributor and is_answerable:
            try:
                doc_dicts = [
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "url": doc.metadata.get("url", "")
                    }
                    for doc in documents
                ]
                attributions = self.source_attributor.attribute_answer(answer, doc_dicts)
                result["attributions"] = [
                    {
                        "claim": a.claim,
                        "source": a.source,
                        "evidence": a.evidence,
                        "similarity": a.similarity_score
                    }
                    for a in attributions
                ]
            except Exception:
                result["attributions"] = []
        else:
            result["attributions"] = []
        
        # Rationale generation
        if self.rationale_generator and is_answerable:
            try:
                rationale = self.rationale_generator.generate_rationale(
                    question=question,
                    answer=answer,
                    context=context
                )
                result["rationale"] = rationale
            except Exception:
                result["rationale"] = None
        else:
            result["rationale"] = None
        
        # Check if human review needed
        confidence_score = result["confidence"].get("score", 0.0)
        result["needs_review"] = confidence_score < LOW_CONFIDENCE_THRESHOLD
        
        return result
    
    def unanswerable_response(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Generate safe unanswerable response.
        
        Called when we can't find relevant information.
        """
        return {
            "answer": UNANSWERABLE_RESPONSE,
            "is_answerable": False,
            "confidence": {
                "score": 0.0,
                "level": "low",
                "explanation": "Insufficient relevant context found in knowledge base"
            },
            "attributions": [],
            "rationale": None,
            "needs_review": False
        }
    
    def handle_error(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Node: Handle errors gracefully.
        
        Called when other nodes fail, providing a safe fallback response.
        """
        error = state.get("error", "Unknown error occurred")
        
        return {
            "answer": f"I encountered an issue processing your question. {UNANSWERABLE_RESPONSE}",
            "is_answerable": False,
            "confidence": {
                "score": 0.0,
                "level": "low",
                "explanation": f"Error during processing: {str(error)[:100]}"
            },
            "attributions": [],
            "rationale": None,
            "needs_review": True
        }
    
    # Async node implementations for high-concurrency scenarios
    async def aretrieve_documents(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Async Node: Retrieve documents from knowledge base.
        
        For production use with high concurrency - runs sync retrieval
        in a thread pool to avoid blocking the event loop.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve_documents, state)
    
    async def agenerate_answer(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Async Node: Generate answer using LLM.
        
        Runs LLM generation in thread pool for async compatibility.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_answer, state)
    
    async def aenrich_xai(self, state: HealthcareRAGState) -> Dict[str, Any]:
        """
        Async Node: Enrich with XAI components.
        
        Runs XAI enrichment in thread pool.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.enrich_xai, state)
