"""
Text cleaning and preprocessing for medical documents.
"""
import re
from typing import List, Optional
import unicodedata

class MedicalTextCleaner:
    """Clean and normalize medical text."""
    
    # Common medical abbreviations to preserve
    MEDICAL_ABBREVIATIONS = {
        "mg", "ml", "kg", "lb", "oz", "bpm", "mmhg", "icd", "ecg", "ekg",
        "mri", "ct", "hiv", "aids", "copd", "bp", "hr", "rr", "temp", "spo2",
        "prn", "bid", "tid", "qid", "qd", "hs", "po", "iv", "im", "sc"
    }
    
    def __init__(self):
        self.reference_pattern = re.compile(r'\[\d+\]|\(\d+\)')
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.whitespace_pattern = re.compile(r'\s+')
        self.html_pattern = re.compile(r'<[^>]+>')
    
    def clean(self, text: str) -> str:
        """Apply all cleaning steps."""
        if not text:
            return ""
        
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_references(text)
        text = self.normalize_unicode(text)
        text = self.normalize_whitespace(text)
        text = self.fix_medical_terms(text)
        
        return text.strip()
    
    def remove_html(self, text: str) -> str:
        """Remove HTML tags."""
        return self.html_pattern.sub(' ', text)
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs."""
        return self.url_pattern.sub('', text)
    
    def remove_references(self, text: str) -> str:
        """Remove citation references like [1], (2)."""
        return self.reference_pattern.sub('', text)
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        return unicodedata.normalize('NFKC', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        return self.whitespace_pattern.sub(' ', text)
    
    def fix_medical_terms(self, text: str) -> str:
        """Preserve common medical abbreviations."""
        # Keep abbreviations uppercase
        words = text.split()
        fixed = []
        for word in words:
            lower = word.lower().rstrip('.,;:')
            if lower in self.MEDICAL_ABBREVIATIONS:
                fixed.append(word.upper() if word.isupper() else word)
            else:
                fixed.append(word)
        return ' '.join(fixed)
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean multiple texts."""
        return [self.clean(t) for t in texts]
