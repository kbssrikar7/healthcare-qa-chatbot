"""
Medical Query Expansion for improved retrieval.

Expands user queries with medical synonyms, abbreviations,
and related terms to improve hybrid retrieval recall.
"""
import re
from typing import List, Dict, Set


class MedicalQueryExpander:
    """Expand medical queries with synonyms and related terms.
    
    Improves retrieval recall by generating query variants
    that capture different ways to express the same medical concept.
    """
    
    # Medical synonym mappings (term -> list of synonyms/related terms)
    MEDICAL_SYNONYMS: Dict[str, List[str]] = {
        # Cardiovascular
        "heart attack": ["myocardial infarction", "MI", "cardiac event", "coronary event"],
        "high blood pressure": ["hypertension", "elevated BP", "elevated blood pressure"],
        "hypertension": ["high blood pressure", "elevated BP"],
        "stroke": ["cerebrovascular accident", "CVA", "brain attack"],
        "chest pain": ["angina", "thoracic pain", "angina pectoris"],
        
        # Metabolic
        "diabetes": ["diabetes mellitus", "DM", "hyperglycemia", "blood sugar disorder"],
        "type 2 diabetes": ["T2DM", "type II diabetes", "adult-onset diabetes", "non-insulin dependent"],
        "type 1 diabetes": ["T1DM", "type I diabetes", "juvenile diabetes", "insulin dependent"],
        "high cholesterol": ["hyperlipidemia", "hypercholesterolemia", "dyslipidemia"],
        
        # Respiratory
        "asthma": ["bronchial asthma", "reactive airway disease"],
        "pneumonia": ["lung infection", "pulmonary infection"],
        "copd": ["chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis"],
        "shortness of breath": ["dyspnea", "breathlessness", "difficulty breathing"],
        
        # Neurological
        "headache": ["cephalgia", "head pain"],
        "migraine": ["migraine headache", "severe headache"],
        "seizure": ["epileptic seizure", "convulsion", "fit"],
        "alzheimer": ["alzheimer's disease", "AD", "dementia"],
        
        # Gastrointestinal
        "stomach pain": ["abdominal pain", "gastralgia", "belly ache", "epigastric pain"],
        "acid reflux": ["GERD", "gastroesophageal reflux", "heartburn"],
        "ibs": ["irritable bowel syndrome", "spastic colon"],
        
        # Musculoskeletal
        "arthritis": ["joint inflammation", "joint pain", "osteoarthritis"],
        "back pain": ["lumbar pain", "lumbago", "dorsalgia"],
        
        # Mental Health
        "depression": ["major depressive disorder", "MDD", "clinical depression"],
        "anxiety": ["anxiety disorder", "generalized anxiety", "GAD"],
        
        # General
        "fever": ["pyrexia", "elevated temperature", "high temperature"],
        "infection": ["infectious disease", "pathogenic infection"],
        "cancer": ["malignancy", "neoplasm", "tumor", "carcinoma"],
        "allergy": ["allergic reaction", "hypersensitivity", "allergic response"],
        "pregnancy": ["gestation", "expecting", "prenatal"],
        "kidney disease": ["renal disease", "nephropathy", "kidney failure"],
        "liver disease": ["hepatic disease", "liver disorder", "hepatopathy"],
    }
    
    # Medical abbreviation expansions
    ABBREVIATIONS: Dict[str, str] = {
        "mi": "myocardial infarction",
        "cva": "cerebrovascular accident",
        "copd": "chronic obstructive pulmonary disease",
        "gerd": "gastroesophageal reflux disease",
        "ibs": "irritable bowel syndrome",
        "uti": "urinary tract infection",
        "bp": "blood pressure",
        "hr": "heart rate",
        "dm": "diabetes mellitus",
        "chf": "congestive heart failure",
        "cad": "coronary artery disease",
        "ckd": "chronic kidney disease",
        "dvt": "deep vein thrombosis",
        "pe": "pulmonary embolism",
        "ra": "rheumatoid arthritis",
        "ms": "multiple sclerosis",
        "tb": "tuberculosis",
        "hiv": "human immunodeficiency virus",
    }
    
    def __init__(self, max_expansions: int = 3):
        """
        Initialize query expander.
        
        Args:
            max_expansions: Maximum number of expanded queries to generate
        """
        self.max_expansions = max_expansions
    
    def expand(self, query: str) -> List[str]:
        """
        Expand query with medical synonyms.
        
        Args:
            query: Original user query
            
        Returns:
            List of queries (original + expanded variants)
        """
        expanded: List[str] = [query]
        query_lower = query.lower()
        
        # Check for synonym matches
        for term, synonyms in self.MEDICAL_SYNONYMS.items():
            if term in query_lower:
                for syn in synonyms[:self.max_expansions]:
                    variant = re.sub(
                        re.escape(term), syn, query_lower, flags=re.IGNORECASE
                    )
                    if variant not in expanded:
                        expanded.append(variant)
        
        # Check for abbreviation expansion
        words = query_lower.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.ABBREVIATIONS:
                full_term = self.ABBREVIATIONS[clean_word]
                variant = query_lower.replace(clean_word, full_term)
                if variant not in expanded:
                    expanded.append(variant)
        
        return expanded[:self.max_expansions + 1]  # original + max_expansions
    
    def get_expanded_terms(self, query: str) -> Set[str]:
        """
        Get just the expanded medical terms (without full query variants).
        
        Useful for boosting retrieval with additional search terms.
        
        Args:
            query: Original user query
            
        Returns:
            Set of related medical terms
        """
        terms: Set[str] = set()
        query_lower = query.lower()
        
        for term, synonyms in self.MEDICAL_SYNONYMS.items():
            if term in query_lower:
                terms.update(synonyms)
        
        words = query_lower.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.ABBREVIATIONS:
                terms.add(self.ABBREVIATIONS[clean_word])
        
        return terms
