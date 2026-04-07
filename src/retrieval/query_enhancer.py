"""
Query enhancement for improved retrieval recall.

Based on RAG skill patterns for query rewriting and expansion:
- Multi-query generation for broader coverage
- Medical terminology expansion
- HyDE (Hypothetical Document Embeddings) option
"""
from loguru import logger
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class EnhancedQuery:
    """Container for enhanced query results."""
    original: str
    expansions: List[str]
    all_queries: List[str]


class QueryEnhancer:
    """
    Enhance queries for better retrieval recall.
    
    Implements patterns from ai-rag and rag-pipeline-builder skills.
    """
    
    # Common medical abbreviation expansions
    # Covers: vitals, imaging, labs, conditions, cardiology, neurology,
    # GI, renal, pulmonary, endocrine, and common clinical shorthands.
    MEDICAL_EXPANSIONS = {
        # Vitals & basic measurements
        "bp": "blood pressure",
        "hr": "heart rate",
        "rr": "respiratory rate",
        "spo2": "oxygen saturation",
        "bmi": "body mass index",
        "temp": "temperature",
        # Imaging
        "ct": "computed tomography",
        "mri": "magnetic resonance imaging",
        "us": "ultrasound",
        "xr": "x-ray",
        "pet": "positron emission tomography",
        # Labs
        "cbc": "complete blood count",
        "bmp": "basic metabolic panel",
        "cmp": "comprehensive metabolic panel",
        "lfts": "liver function tests",
        "tsh": "thyroid stimulating hormone",
        "hba1c": "glycated hemoglobin",
        "pt": "prothrombin time",
        "inr": "international normalized ratio",
        "ptt": "partial thromboplastin time",
        # Cardiology
        "ecg": "electrocardiogram",
        "ekg": "electrocardiogram",
        "mi": "myocardial infarction",
        "cvd": "cardiovascular disease",
        "chf": "congestive heart failure",
        "afib": "atrial fibrillation",
        "af": "atrial fibrillation",
        "cabg": "coronary artery bypass grafting",
        "pci": "percutaneous coronary intervention",
        "htn": "hypertension",
        # Neurology
        "tia": "transient ischemic attack",
        "ms": "multiple sclerosis",
        "als": "amyotrophic lateral sclerosis",
        "cva": "cerebrovascular accident stroke",
        # Gastrointestinal
        "gerd": "gastroesophageal reflux disease",
        "ibs": "irritable bowel syndrome",
        "ibd": "inflammatory bowel disease",
        "gi": "gastrointestinal",
        # Renal
        "ckd": "chronic kidney disease",
        "esrd": "end stage renal disease",
        "aki": "acute kidney injury",
        "uti": "urinary tract infection",
        # Pulmonary
        "copd": "chronic obstructive pulmonary disease",
        "pe": "pulmonary embolism",
        "dvt": "deep vein thrombosis",
        "ards": "acute respiratory distress syndrome",
        # Endocrine / metabolic
        "dm": "diabetes mellitus",
        "t2dm": "type 2 diabetes mellitus",
        "t1dm": "type 1 diabetes mellitus",
        "pcos": "polycystic ovary syndrome",
        "dka": "diabetic ketoacidosis",
        # Dosing / timing shorthands (used in queries like "take ibuprofen prn")
        "prn": "as needed",
        "bid": "twice daily",
        "tid": "three times daily",
        "qid": "four times daily",
        "qd": "once daily",
        "npo": "nothing by mouth",
        "stat": "immediately",
        # Common oncology
        "nhl": "non hodgkin lymphoma",
        "cll": "chronic lymphocytic leukemia",
        "aml": "acute myeloid leukemia",
    }

    
    def __init__(
        self,
        llm=None,
        enable_llm_expansion: bool = False,
        max_expansions: int = 3
    ):
        """
        Initialize query enhancer.
        
        Args:
            llm: Optional LLM wrapper for generating query expansions
            enable_llm_expansion: Whether to use LLM for query expansion
            max_expansions: Maximum number of query expansions to generate
        """
        self.llm = llm
        self.enable_llm_expansion = enable_llm_expansion and llm is not None
        self.max_expansions = max_expansions
    
    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Remove potential prompt injection patterns from user query."""
        import re
        injection_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions",
            r"disregard\s+(previous|above|all)",
            r"you\s+are\s+now\s+",
            r"system\s*:\s*",
        ]
        sanitized = query
        for pattern in injection_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        return sanitized.strip()

    def enhance(self, query: str) -> EnhancedQuery:
        """
        Enhance a query with multiple formulations.

        Args:
            query: Original user query

        Returns:
            EnhancedQuery with original and expanded queries
        """
        query = self._sanitize_query(query)
        expansions = []
        
        # 1. Medical abbreviation expansion
        abbrev_expanded = self._expand_abbreviations(query)
        if abbrev_expanded != query:
            expansions.append(abbrev_expanded)
        
        # 2. LLM-based expansion (if enabled)
        if self.enable_llm_expansion:
            llm_expansions = self._llm_expand(query)
            expansions.extend(llm_expansions)
        
        # 3. Simple reformulations
        reformulations = self._simple_reformulations(query)
        expansions.extend(reformulations)
        
        # Deduplicate and limit
        seen = {query.lower()}
        unique_expansions = []
        for exp in expansions:
            if exp.lower() not in seen:
                seen.add(exp.lower())
                unique_expansions.append(exp)
                if len(unique_expansions) >= self.max_expansions:
                    break
        
        return EnhancedQuery(
            original=query,
            expansions=unique_expansions,
            all_queries=[query] + unique_expansions
        )
    
    def _expand_abbreviations(self, query: str) -> str:
        """Expand common medical abbreviations."""
        words = query.split()
        expanded_words = []
        
        for word in words:
            lower_word = word.lower().strip('?.,!')
            if lower_word in self.MEDICAL_EXPANSIONS:
                expanded_words.append(self.MEDICAL_EXPANSIONS[lower_word])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def _simple_reformulations(self, query: str) -> List[str]:
        """Generate simple query reformulations."""
        reformulations = []
        
        # Question to statement conversion
        query_lower = query.lower().strip()
        
        if query_lower.startswith("what is "):
            reformulations.append(query[8:].strip("?") + " definition and explanation")
        elif query_lower.startswith("how to "):
            reformulations.append(query[7:].strip("?") + " treatment and management")
        elif query_lower.startswith("what are the symptoms of "):
            condition = query[25:].strip("?")
            reformulations.append(f"{condition} clinical presentation signs")
        elif query_lower.startswith("what causes "):
            condition = query[12:].strip("?")
            reformulations.append(f"{condition} etiology pathophysiology")
        
        return reformulations
    
    def _llm_expand(self, query: str) -> List[str]:
        """Use LLM to generate query expansions."""
        if self.llm is None:
            return []
        
        prompt = f"""Generate {self.max_expansions} alternative phrasings of this medical question.
Focus on different medical terminology and perspectives.

Original question: {query}

Return only the alternative questions, one per line, without numbering."""
        
        try:
            response = self.llm.generate(prompt, max_new_tokens=150)
            lines = response.response.strip().split('\n')
            expansions = [line.strip() for line in lines if line.strip()]
            return expansions[:self.max_expansions]
        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}")
            return []
    
    def generate_hypothetical_answer(self, query: str) -> Optional[str]:
        """
        Generate a hypothetical answer for HyDE retrieval.
        
        HyDE: Use hypothetical document to find similar real documents.
        """
        if self.llm is None:
            return None
        
        prompt = f"""Write a brief, factual answer to this medical question as if you were a medical textbook.
Include specific medical terminology.

Question: {query}

Answer (2-3 sentences):"""
        
        try:
            response = self.llm.generate(prompt, max_new_tokens=100)
            return response.response.strip()
        except Exception as e:
            logger.warning(f"HyDE hypothetical answer generation failed: {e}")
            return None
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into simpler sub-queries.
        
        Based on rag-agent-builder skill pattern for handling
        multi-part questions.
        
        Args:
            query: Complex user query
            
        Returns:
            List of simpler sub-queries
        """
        # Check if query has multiple parts
        query_lower = query.lower()
        
        # Pattern: questions joined by "and"
        if " and " in query_lower:
            parts = query.split(" and ")
            if len(parts) >= 2:
                sub_queries = []
                for part in parts:
                    part = part.strip()
                    # Ensure it's a proper question
                    if not any(part.lower().startswith(q) for q in 
                              ["what", "how", "why", "when", "where", "which", "can", "is"]):
                        part = "What is " + part
                    sub_queries.append(part.strip("?") + "?")
                return sub_queries
        
        # Pattern: "compare X and Y" or "difference between X and Y"
        if "compare" in query_lower or "difference between" in query_lower:
            # Extract the items being compared
            import re
            match = re.search(r'(?:compare|difference between)\s+(.+)\s+and\s+(.+?)(?:\?|$)', 
                            query_lower)
            if match:
                item1, item2 = match.groups()
                return [
                    f"What is {item1.strip()}?",
                    f"What is {item2.strip()}?",
                    query  # Original comparison query
                ]
        
        # LLM-based decomposition for complex queries
        if self.llm and len(query.split()) > 10:
            prompt = f"""If this question has multiple parts, break it into 2-3 simpler questions.
If it's already simple, return just the original question.

Question: {query}

Return only the questions, one per line:"""
            
            try:
                response = self.llm.generate(prompt, max_new_tokens=150)
                lines = response.response.strip().split('\n')
                sub_queries = [line.strip() for line in lines if line.strip()]
                if len(sub_queries) > 1:
                    return sub_queries[:3]  # Max 3 sub-queries
            except Exception as e:
                logger.warning(f"Query decomposition failed: {e}")
        
        # Return original if no decomposition possible
        return [query]

