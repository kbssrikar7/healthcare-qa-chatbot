"""
Safety guardrails for medical QA chatbot.
"""
import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SafetyLevel(Enum):
    """Safety classification levels."""
    SAFE = "safe"
    CAUTION = "caution"
    BLOCKED = "blocked"
    EMERGENCY = "emergency"

@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    level: SafetyLevel
    passed: bool
    flags: List[str]
    message: str
    redirect_message: Optional[str] = None

class MedicalGuardrails:
    """
    Safety guardrails for medical QA to prevent harmful outputs.
    """
    
    # Emergency keywords that require immediate attention
    EMERGENCY_KEYWORDS = [
        "suicide", "suicidal", "kill myself", "end my life",
        "heart attack", "can't breathe", "chest pain", "stroke symptoms",
        "severe bleeding", "unconscious", "overdose", "poisoning",
        "anaphylaxis", "severe allergic reaction", "choking"
    ]
    
    # Dangerous advice patterns to block
    DANGEROUS_PATTERNS = [
        r"take\s+\d+\s*(mg|ml|pills|tablets)",  # Dosage recommendations
        r"you\s+(have|definitely\s+have)\s+\w+\s+(disease|disorder|syndrome|cancer)",  # Direct diagnosis
        r"stop\s+taking\s+your\s+(medication|medicine|prescription)",  # Stopping medication
        r"you\s+don'?t\s+need\s+to\s+see\s+a\s+doctor",  # Discouraging medical care
        r"instead\s+of\s+(seeing\s+a\s+doctor|going\s+to|medical\s+care)",  # Alternative to medical care
    ]
    
    # Topics requiring professional consultation
    SENSITIVE_TOPICS = [
        "pregnancy", "pregnant", "abortion",
        "mental health", "depression", "anxiety", "psychiatric",
        "cancer", "tumor", "malignant",
        "hiv", "aids", "std", "sexually transmitted",
        "surgery", "surgical", "operation",
        "prescription", "controlled substance"
    ]
    
    # Medical disclaimer
    DISCLAIMER = """
⚠️ **IMPORTANT MEDICAL DISCLAIMER**

This information is for educational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment. 

- Always consult a qualified healthcare provider for medical concerns
- Do not delay seeking medical care based on information provided here
- In case of emergency, call emergency services immediately
"""

    EMERGENCY_MESSAGE = """
🚨 **EMERGENCY DETECTED**

If you or someone else is experiencing a medical emergency, please:

1. **Call Emergency Services Immediately**: 
   - US: 911
   - UK: 999
   - EU: 112
   
2. **Do not wait** - get professional help now

3. **Stay calm** and follow dispatcher instructions

This chatbot cannot provide emergency medical assistance.
"""

    def __init__(
        self,
        enable_emergency_detection: bool = True,
        enable_content_filter: bool = True,
        enable_diagnosis_prevention: bool = True
    ):
        self.enable_emergency_detection = enable_emergency_detection
        self.enable_content_filter = enable_content_filter
        self.enable_diagnosis_prevention = enable_diagnosis_prevention
        
        # Compile patterns
        self.dangerous_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS
        ]
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text to prevent bypass attacks using Unicode lookalikes or spacing tricks.
        
        Args:
            text: Input text that may contain Unicode tricks
            
        Returns:
            Normalized text for reliable pattern matching
        """
        # Normalize Unicode (NFKD converts fancy characters to base forms)
        text = unicodedata.normalize('NFKD', text)
        # Normalize whitespace (collapse multiple spaces, handle special whitespace chars)
        text = re.sub(r'\s+', ' ', text)
        # Remove zero-width characters that could be used to bypass filters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        return text
    
    def check_input(self, text: str) -> SafetyCheckResult:
        """Check user input for safety concerns."""
        # Normalize text to prevent bypass attacks
        text = self._normalize_text(text)
        text_lower = text.lower()
        flags = []
        
        # Check for emergency keywords
        if self.enable_emergency_detection:
            for keyword in self.EMERGENCY_KEYWORDS:
                if keyword in text_lower:
                    return SafetyCheckResult(
                        level=SafetyLevel.EMERGENCY,
                        passed=False,
                        flags=["emergency_detected"],
                        message=f"Emergency keyword detected: {keyword}",
                        redirect_message=self.EMERGENCY_MESSAGE
                    )
        
        # Check for sensitive topics
        for topic in self.SENSITIVE_TOPICS:
            if topic in text_lower:
                flags.append(f"sensitive_topic:{topic}")
        
        if flags:
            return SafetyCheckResult(
                level=SafetyLevel.CAUTION,
                passed=True,
                flags=flags,
                message="Sensitive topic detected - include strong disclaimer"
            )
        
        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            passed=True,
            flags=[],
            message="Input passed safety checks"
        )
    
    def check_output(self, text: str) -> SafetyCheckResult:
        """Check generated output for dangerous content."""
        text_lower = text.lower()
        flags = []
        
        # Check for dangerous patterns
        if self.enable_content_filter:
            for pattern in self.dangerous_patterns:
                if pattern.search(text):
                    flags.append(f"dangerous_pattern:{pattern.pattern[:30]}")
        
        # Check for diagnosis-like statements
        if self.enable_diagnosis_prevention:
            diagnosis_patterns = [
                r"you\s+have\s+\w+",
                r"this\s+is\s+(definitely|clearly)\s+\w+",
                r"my\s+diagnosis\s+is"
            ]
            for pattern in diagnosis_patterns:
                if re.search(pattern, text_lower):
                    flags.append("potential_diagnosis")
        
        if "dangerous_pattern" in str(flags):
            return SafetyCheckResult(
                level=SafetyLevel.BLOCKED,
                passed=False,
                flags=flags,
                message="Output contains potentially dangerous content",
                redirect_message="I cannot provide specific medical advice. Please consult a healthcare professional."
            )
        
        if flags:
            return SafetyCheckResult(
                level=SafetyLevel.CAUTION,
                passed=True,
                flags=flags,
                message="Output requires additional disclaimers"
            )
        
        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            passed=True,
            flags=[],
            message="Output passed safety checks"
        )
    
    def add_disclaimer(self, text: str, level: SafetyLevel) -> str:
        """Add appropriate disclaimer based on safety level."""
        if level == SafetyLevel.EMERGENCY:
            return self.EMERGENCY_MESSAGE
        
        if level in [SafetyLevel.CAUTION, SafetyLevel.SAFE]:
            return f"{text}\n\n{self.DISCLAIMER}"
        
        return text
    
    def sanitize_output(self, text: str) -> Tuple[str, SafetyCheckResult]:
        """Sanitize output and add appropriate disclaimers."""
        check_result = self.check_output(text)
        
        if not check_result.passed:
            return check_result.redirect_message or "I cannot provide this information.", check_result
        
        sanitized = self.add_disclaimer(text, check_result.level)
        return sanitized, check_result


class ContentFilter:
    """Filter inappropriate or harmful content."""
    
    BLOCKED_CONTENT = [
        "how to make", "how to synthesize", "how to create drugs",
        "illegal drugs", "recreational drugs",
        "self-harm", "hurt myself", "harm myself"
    ]
    
    def __init__(self):
        self.blocked_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.BLOCKED_CONTENT
        ]
    
    def is_blocked(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if content should be blocked."""
        text_lower = text.lower()
        
        for pattern in self.blocked_patterns:
            if pattern.search(text_lower):
                return True, "This type of content is not supported by this medical chatbot."
        
        return False, None


class EmergencyDetector:
    """Detect medical emergencies requiring immediate attention."""
    
    EMERGENCY_SYMPTOMS = {
        "cardiac": ["chest pain", "heart attack", "can't breathe", "shortness of breath", "crushing chest"],
        "stroke": ["face drooping", "arm weakness", "speech difficulty", "sudden numbness", "sudden confusion"],
        "allergic": ["anaphylaxis", "throat swelling", "can't swallow", "severe allergic"],
        "mental": ["suicide", "suicidal", "want to die", "kill myself", "end my life"],
        "trauma": ["severe bleeding", "broken bone", "head injury", "unconscious", "not responding"]
    }
    
    def detect(self, text: str) -> Tuple[bool, Optional[str]]:
        """Detect if text indicates a medical emergency."""
        text_lower = text.lower()
        
        for category, symptoms in self.EMERGENCY_SYMPTOMS.items():
            for symptom in symptoms:
                if symptom in text_lower:
                    return True, category
        
        return False, None
    
    def get_emergency_response(self, category: str) -> str:
        """Get appropriate emergency response message."""
        responses = {
            "cardiac": "🚨 POSSIBLE CARDIAC EMERGENCY - Call 911 immediately! Symptoms like chest pain require immediate medical attention.",
            "stroke": "🚨 POSSIBLE STROKE - Call 911 immediately! Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call 911.",
            "allergic": "🚨 POSSIBLE ANAPHYLAXIS - Call 911 immediately! If you have an EpiPen, use it now.",
            "mental": "🚨 If you're having thoughts of suicide, please call the National Suicide Prevention Lifeline: 988 (US) or your local emergency services.",
            "trauma": "🚨 MEDICAL EMERGENCY - Call 911 immediately! Do not move the person unless necessary for safety."
        }
        
        return responses.get(category, "🚨 Please seek immediate medical attention. Call 911 or your local emergency services.")
