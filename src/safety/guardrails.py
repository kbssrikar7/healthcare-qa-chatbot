"""
Safety guardrails for medical QA chatbot.
"""

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


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

    # Pre-compiled regex to detect educational/informational queries (not emergencies)
    _EDUCATIONAL_RE = re.compile(
        r"\b(what\s+is|what\s+are|define|explain|tell\s+me\s+about|"
        r"symptoms\s+of|causes\s+of|treatment\s+for)\b",
        re.IGNORECASE,
    )

    # Emergency keywords that require immediate attention
    EMERGENCY_KEYWORDS = [
        "suicide",
        "suicidal",
        "kill myself",
        "end my life",
        "heart attack",
        "can't breathe",
        "chest pain",
        "stroke symptoms",
        "severe bleeding",
        "unconscious",
        "overdose",
        "poisoning",
        "anaphylaxis",
        "severe allergic reaction",
        "choking",
    ]

    # Dangerous advice patterns to block
    DANGEROUS_PATTERNS = [
        r"you\s+should\s+take\s+\d+\s*(mg|ml|pills|tablets)",  # Direct prescriptive dosage (not quoting sources)
        r"you\s+(have|definitely\s+have)\s+\w+\s+(disease|disorder|syndrome|cancer)",  # Direct diagnosis
        r"stop\s+taking\s+your\s+(medication|medicine|prescription)",  # Stopping medication
        r"you\s+don'?t\s+need\s+to\s+see\s+a\s+doctor",  # Discouraging medical care
        r"instead\s+of\s+(seeing\s+a\s+doctor|going\s+to|medical\s+care)",  # Alternative to medical care
    ]

    # Topics requiring professional consultation
    SENSITIVE_TOPICS = [
        "pregnancy",
        "pregnant",
        "abortion",
        "mental health",
        "depression",
        "anxiety",
        "psychiatric",
        "cancer",
        "tumor",
        "malignant",
        "hiv",
        "aids",
        "std",
        "sexually transmitted",
        "surgery",
        "surgical",
        "operation",
        "prescription",
        "controlled substance",
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
        enable_diagnosis_prevention: bool = True,
    ):
        self.enable_emergency_detection = enable_emergency_detection
        self.enable_content_filter = enable_content_filter
        self.enable_diagnosis_prevention = enable_diagnosis_prevention

        # Compile patterns
        self.dangerous_patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text to prevent bypass attacks using Unicode lookalikes or spacing tricks.

        Args:
            text: Input text that may contain Unicode tricks

        Returns:
            Normalized text for reliable pattern matching
        """
        # Normalize Unicode (NFKD converts fancy characters to base forms)
        text = unicodedata.normalize("NFKD", text)
        # Normalize whitespace (collapse multiple spaces, handle special whitespace chars)
        text = re.sub(r"\s+", " ", text)
        # Remove zero-width characters that could be used to bypass filters
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        return text

    def check_input(self, text: str) -> SafetyCheckResult:
        """Check user input for safety concerns."""
        # Normalize text to prevent bypass attacks
        text = self._normalize_text(text)
        text_lower = text.lower()
        flags = []

        # Check for emergency keywords (with negation awareness)
        if self.enable_emergency_detection:
            # Skip educational queries
            if not self._EDUCATIONAL_RE.search(text):
                negation_words = {
                    "not",
                    "no",
                    "don't",
                    "dont",
                    "doesn't",
                    "never",
                    "without",
                    "deny",
                    "denies",
                    "denied",
                }
                for keyword in self.EMERGENCY_KEYWORDS:
                    if keyword in text_lower:
                        # Check for negation in 40-char window before keyword
                        idx = text_lower.find(keyword)
                        prefix = text_lower[max(0, idx - 40) : idx]
                        if any(neg in prefix for neg in negation_words):
                            continue
                        return SafetyCheckResult(
                            level=SafetyLevel.EMERGENCY,
                            passed=False,
                            flags=["emergency_detected"],
                            message=f"Emergency keyword detected: {keyword}",
                            redirect_message=self.EMERGENCY_MESSAGE,
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
                message="Sensitive topic detected - include strong disclaimer",
            )

        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            passed=True,
            flags=[],
            message="Input passed safety checks",
        )

    def check_output(self, text: str) -> SafetyCheckResult:
        """Check generated output for dangerous content."""
        # Normalize text to prevent Unicode bypass attacks (same as check_input)
        text = self._normalize_text(text)
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
                r"you\s+have\s+(?!options|several|many|a\s+few|the\s+|some\s+|to\s+|been\s+asked|been\s+referred)\w+",
                r"this\s+is\s+(definitely|clearly)\s+\w+",
                r"my\s+diagnosis\s+is",
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
                redirect_message="I cannot provide specific medical advice. Please consult a healthcare professional.",
            )

        if flags:
            return SafetyCheckResult(
                level=SafetyLevel.CAUTION,
                passed=True,
                flags=flags,
                message="Output requires additional disclaimers",
            )

        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            passed=True,
            flags=[],
            message="Output passed safety checks",
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
            return (
                check_result.redirect_message or "I cannot provide this information.",
                check_result,
            )

        sanitized = self.add_disclaimer(text, check_result.level)
        return sanitized, check_result


def _normalize_filter_text(text: str) -> str:
    """Match MedicalGuardrails normalization so filters cannot be bypassed with lookalikes."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text.lower()


class ContentFilter:
    """Filter inappropriate or harmful content."""

    BLOCKED_CONTENT = [
        r"how\s+to\s+(?:make|synthesize|produce|manufacture)\s+(?:illegal\s+)?(?:drugs|narcotics|meth|cocaine|heroin|fentanyl)",
        r"how\s+to\s+(?:synthesize|create|cook)\s+\w+(?:\s+\w+)?\s+(?:at home|illegally)",
        r"\billegal\s+drugs\b",
        r"\brecreational\s+drugs\b",
        r"\bself-harm\b",
        r"\bhurt\s+myself\b",
        r"\bharm\s+myself\b",
    ]

    def __init__(self):
        self.blocked_patterns = [re.compile(p, re.IGNORECASE) for p in self.BLOCKED_CONTENT]

    def is_blocked(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if content should be blocked."""
        text_lower = _normalize_filter_text(text)

        for pattern in self.blocked_patterns:
            if pattern.search(text_lower):
                return (
                    True,
                    "This type of content is not supported by this medical chatbot.",
                )

        return False, None


class EmergencyDetector:
    """Detect medical emergencies requiring immediate attention.

    Uses negation-aware matching to reduce false positives
    (e.g. "I don't have chest pain" should NOT trigger emergency).
    """

    EMERGENCY_SYMPTOMS = {
        "cardiac": [
            "chest pain",
            "heart attack",
            "can't breathe",
            "shortness of breath",
            "crushing chest",
        ],
        "stroke": [
            "face drooping",
            "arm weakness",
            "speech difficulty",
            "sudden numbness",
            "sudden confusion",
        ],
        "allergic": [
            "anaphylaxis",
            "throat swelling",
            "can't swallow",
            "severe allergic",
        ],
        "mental": ["suicide", "suicidal", "want to die", "kill myself", "end my life"],
        "trauma": [
            "severe bleeding",
            "broken bone",
            "head injury",
            "unconscious",
            "not responding",
        ],
    }

    # Words that negate the symptom when they appear before it
    NEGATION_WORDS = {
        "not",
        "no",
        "don't",
        "dont",
        "doesn't",
        "doesnt",
        "never",
        "without",
        "deny",
        "denies",
        "denied",
        "absent",
        "negative",
        "ruled out",
        "no signs of",
    }

    # Patterns indicating educational/informational queries (not emergencies)
    EDUCATIONAL_PATTERNS = re.compile(
        r"\b(what\s+is|what\s+are|define|explain|tell\s+me\s+about|"
        r"symptoms\s+of|causes\s+of|treatment\s+for|how\s+to\s+treat|"
        r"difference\s+between|learned\s+about|studying|homework|"
        r"research\s+on|article\s+about|read\s+about)\b",
        re.IGNORECASE,
    )

    def _is_negated(self, text_lower: str, symptom: str) -> bool:
        """Check if a symptom mention is negated by preceding words."""
        idx = text_lower.find(symptom)
        if idx < 0:
            return False
        # Check a window of 40 chars before the symptom
        prefix = text_lower[max(0, idx - 40) : idx]
        for neg in self.NEGATION_WORDS:
            if neg in prefix:
                return True
        return False

    def _is_educational(self, text: str) -> bool:
        """Check if the query is educational/informational, not an emergency report."""
        return bool(self.EDUCATIONAL_PATTERNS.search(text))

    def detect(self, text: str) -> Tuple[bool, Optional[str]]:
        """Detect if text indicates a medical emergency.

        Returns (False, None) for negated symptoms and educational queries.
        """
        text_lower = text.lower()

        # Educational queries about symptoms are not emergencies
        if self._is_educational(text):
            return False, None

        for category, symptoms in self.EMERGENCY_SYMPTOMS.items():
            for symptom in symptoms:
                if symptom in text_lower:
                    # Skip if the symptom is negated
                    if self._is_negated(text_lower, symptom):
                        continue
                    return True, category

        return False, None

    def get_emergency_response(self, category: str) -> str:
        """Get appropriate emergency response message."""
        responses = {
            "cardiac": "🚨 POSSIBLE CARDIAC EMERGENCY - Call 911 immediately! Symptoms like chest pain require immediate medical attention.",
            "stroke": "🚨 POSSIBLE STROKE - Call 911 immediately! Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call 911.",
            "allergic": "🚨 POSSIBLE ANAPHYLAXIS - Call 911 immediately! If you have an EpiPen, use it now.",
            "mental": "🚨 If you're having thoughts of suicide, please call the National Suicide Prevention Lifeline: 988 (US) or your local emergency services.",
            "trauma": "🚨 MEDICAL EMERGENCY - Call 911 immediately! Do not move the person unless necessary for safety.",
        }

        return responses.get(
            category,
            "🚨 Please seek immediate medical attention. Call 911 or your local emergency services.",
        )


class DrugInteractionChecker:
    """
    Check for potential drug interaction mentions.

    Identifies when text mentions drug combinations that may have
    dangerous interactions, prompting a pharmacist consultation.
    """

    # Common high-risk drug interactions (drug_set_1, drug_set_2, severity, message)
    HIGH_RISK_INTERACTIONS = [
        (
            {"warfarin", "coumadin", "blood thinner"},
            {"aspirin", "ibuprofen", "nsaid", "advil", "motrin", "naproxen"},
            "severe",
            "Blood thinners + NSAIDs can increase bleeding risk significantly",
        ),
        (
            {"metformin", "glucophage"},
            {"alcohol", "contrast dye", "iodine contrast"},
            "moderate",
            "Metformin + alcohol/contrast can cause lactic acidosis",
        ),
        (
            {
                "ssri",
                "antidepressant",
                "prozac",
                "zoloft",
                "lexapro",
                "paxil",
                "celexa",
            },
            {"maoi", "tramadol", "fentanyl", "meperidine"},
            "severe",
            "SSRIs + MAOIs/opioids can cause serotonin syndrome",
        ),
        (
            {"ace inhibitor", "lisinopril", "enalapril", "ramipril"},
            {"potassium", "spironolactone", "salt substitute"},
            "moderate",
            "ACE inhibitors + potassium can cause dangerous hyperkalemia",
        ),
        (
            {"statin", "lipitor", "crestor", "simvastatin"},
            {"grapefruit", "erythromycin", "clarithromycin"},
            "moderate",
            "Statins + grapefruit/macrolides can increase muscle damage risk",
        ),
        (
            {"digoxin", "lanoxin"},
            {"amiodarone", "verapamil", "quinidine"},
            "severe",
            "Digoxin + antiarrhythmics can cause dangerous heart rhythm changes",
        ),
        (
            {"lithium"},
            {"nsaid", "ibuprofen", "diuretic", "ace inhibitor"},
            "severe",
            "Lithium levels can become toxic with NSAIDs/diuretics",
        ),
        (
            {"methotrexate"},
            {"nsaid", "ibuprofen", "bactrim", "trimethoprim"},
            "severe",
            "Methotrexate toxicity increases with NSAIDs/sulfas",
        ),
    ]

    # Negation words that negate a drug mention when they appear before it
    NEGATION_WORDS = {
        "not",
        "no",
        "never",
        "don't",
        "dont",
        "doesn't",
        "doesnt",
        "didn't",
        "didnt",
        "without",
        "stop",
        "stopped",
        "stopping",
        "discontinued",
        "quit",
    }

    def _is_negated(self, text_lower: str, drug: str, window: int = 30) -> bool:
        """Check if a drug mention is negated by preceding words."""
        idx = text_lower.find(drug)
        if idx < 0:
            return False
        prefix = text_lower[max(0, idx - window) : idx]
        return any(neg in prefix for neg in self.NEGATION_WORDS)

    def check_interaction_risk(self, text: str) -> List[Dict]:
        """
        Check if text mentions potentially dangerous drug combinations.

        Negation-aware: skips drugs preceded by negation words like
        "not", "stopped", "never" to avoid false positives.

        Args:
            text: Text to analyze

        Returns:
            List of warning dictionaries
        """
        text_lower = text.lower()
        warnings = []

        for (
            drug_set_1,
            drug_set_2,
            severity,
            description,
        ) in self.HIGH_RISK_INTERACTIONS:
            found_1 = [
                drug
                for drug in drug_set_1
                if drug in text_lower and not self._is_negated(text_lower, drug)
            ]
            found_2 = [
                drug
                for drug in drug_set_2
                if drug in text_lower and not self._is_negated(text_lower, drug)
            ]

            if found_1 and found_2:
                warnings.append(
                    {
                        "type": "drug_interaction",
                        "severity": severity,
                        "drug_1": found_1[0] if found_1 else "unknown",
                        "drug_2": found_2[0] if found_2 else "unknown",
                        "description": description,
                        "message": f"⚠️ POTENTIAL DRUG INTERACTION: {description}. Please consult a pharmacist before combining these medications.",
                    }
                )

        return warnings

    def get_interaction_warning(self, warnings: List[Dict]) -> Optional[str]:
        """Format warnings as a single message."""
        if not warnings:
            return None

        # Sort by severity
        severity_order = {"severe": 0, "moderate": 1, "mild": 2}
        warnings.sort(key=lambda w: severity_order.get(w.get("severity", "mild"), 3))

        messages = []
        for w in warnings:
            emoji = "🚨" if w["severity"] == "severe" else "⚠️"
            messages.append(f"{emoji} {w['message']}")

        return "\n\n".join(messages)


class PediatricSafetyChecker:
    """
    Check for pediatric-specific safety concerns.

    Identifies when text discusses children and mentions medications
    or substances that have different safety profiles for pediatric patients.
    """

    # Keywords indicating pediatric context
    PEDIATRIC_INDICATORS = [
        "child",
        "children",
        "kid",
        "infant",
        "baby",
        "toddler",
        "pediatric",
        "newborn",
        "neonate",
        "adolescent",
        "teen",
        "year old",
        "years old",
        "month old",
        "months old",
    ]

    # Substances with pediatric warnings
    PEDIATRIC_WARNINGS = {
        "aspirin": "Aspirin should NOT be given to children due to risk of Reye's syndrome",
        "honey": "Honey should NOT be given to infants under 12 months due to botulism risk",
        "codeine": "Codeine is contraindicated in children under 12 (FDA black box warning)",
        "tramadol": "Tramadol is contraindicated in children under 12",
        "diphenhydramine": "Use caution with diphenhydramine (Benadryl) in children under 2",
        "cough suppressant": "Cough/cold medicines may be dangerous for children under 4",
        "loperamide": "Loperamide (Imodium) should not be used in children under 2",
        "bismuth": "Bismuth (Pepto-Bismol) contains salicylates - avoid in children",
        "melatonin": "Melatonin safety not established in children - consult pediatrician",
        "antihistamine": "Many antihistamines not recommended for children under 2",
    }

    def __init__(self):
        self.pediatric_pattern = re.compile(
            r"\b(" + "|".join(self.PEDIATRIC_INDICATORS) + r")\b", re.IGNORECASE
        )

    def check_pediatric_safety(self, text: str) -> List[Dict]:
        """
        Check for pediatric-sensitive content.

        Args:
            text: Text to analyze

        Returns:
            List of warning dictionaries
        """
        text_lower = text.lower()
        warnings = []

        # Check if pediatric context is mentioned
        if not self.pediatric_pattern.search(text):
            return []

        # Check for substances with pediatric warnings
        for substance, warning in self.PEDIATRIC_WARNINGS.items():
            if substance in text_lower:
                warnings.append(
                    {
                        "type": "pediatric_safety",
                        "substance": substance,
                        "warning": warning,
                        "message": f"👶 PEDIATRIC WARNING: {warning}. Always consult a pediatrician before giving any medication to children.",
                    }
                )

        return warnings

    def get_pediatric_warning(self, warnings: List[Dict]) -> Optional[str]:
        """Format warnings as a single message."""
        if not warnings:
            return None

        messages = [w["message"] for w in warnings]
        return "\n\n".join(messages)


class ComprehensiveSafetyChecker:
    """
    Combined safety checker running all safety modules.
    """

    def __init__(self):
        self.guardrails = MedicalGuardrails()
        self.content_filter = ContentFilter()
        self.emergency_detector = EmergencyDetector()
        self.drug_checker = DrugInteractionChecker()
        self.pediatric_checker = PediatricSafetyChecker()

    def check_all(self, text: str, is_output: bool = False) -> Dict:
        """
        Run all safety checks on text.

        Args:
            text: Text to check
            is_output: Whether this is model output (vs user input)

        Returns:
            Comprehensive safety report
        """
        results = {
            "passed": True,
            "level": SafetyLevel.SAFE,
            "warnings": [],
            "blocked": False,
            "emergency": False,
        }

        # Content filter
        blocked, block_reason = self.content_filter.is_blocked(text)
        if blocked:
            results["passed"] = False
            results["blocked"] = True
            results["block_reason"] = block_reason
            results["level"] = SafetyLevel.BLOCKED
            return results

        # Emergency detection
        is_emergency, category = self.emergency_detector.detect(text)
        if is_emergency:
            results["emergency"] = True
            results["level"] = SafetyLevel.EMERGENCY
            results["emergency_category"] = category
            results["emergency_message"] = self.emergency_detector.get_emergency_response(category)
            if not is_output:
                results["passed"] = False

        # Guardrails
        if is_output:
            guardrail_result = self.guardrails.check_output(text)
        else:
            guardrail_result = self.guardrails.check_input(text)

        if not guardrail_result.passed:
            results["passed"] = False
            results["level"] = guardrail_result.level
        results["guardrail_flags"] = guardrail_result.flags

        # Drug interactions
        drug_warnings = self.drug_checker.check_interaction_risk(text)
        if drug_warnings:
            results["warnings"].extend(drug_warnings)
            if results["level"] == SafetyLevel.SAFE:
                results["level"] = SafetyLevel.CAUTION

        # Pediatric safety
        pediatric_warnings = self.pediatric_checker.check_pediatric_safety(text)
        if pediatric_warnings:
            results["warnings"].extend(pediatric_warnings)
            if results["level"] == SafetyLevel.SAFE:
                results["level"] = SafetyLevel.CAUTION

        return results
