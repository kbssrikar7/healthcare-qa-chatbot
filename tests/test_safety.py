"""
Unit tests for safety guardrails.
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.safety.guardrails import (
    MedicalGuardrails,
    SafetyLevel,
    SafetyCheckResult,
    ContentFilter,
    EmergencyDetector
)


class TestMedicalGuardrails:
    """Tests for MedicalGuardrails."""
    
    @pytest.fixture
    def guardrails(self):
        return MedicalGuardrails()
    
    def test_safe_input(self, guardrails):
        """Test that normal input passes."""
        result = guardrails.check_input("What are the symptoms of a cold?")
        
        assert result.passed == True
        assert result.level == SafetyLevel.SAFE
    
    def test_emergency_detection(self, guardrails):
        """Test emergency keyword detection."""
        result = guardrails.check_input("I'm having chest pain and can't breathe")
        
        assert result.passed == False
        assert result.level == SafetyLevel.EMERGENCY
        assert "emergency" in result.flags[0].lower()
    
    def test_suicide_emergency(self, guardrails):
        """Test suicide-related emergency detection."""
        result = guardrails.check_input("I want to kill myself")
        
        assert result.level == SafetyLevel.EMERGENCY
        assert result.redirect_message is not None
        assert "911" in result.redirect_message or "emergency" in result.redirect_message.lower()
    
    def test_sensitive_topic_caution(self, guardrails):
        """Test sensitive topic flagging."""
        result = guardrails.check_input("I am pregnant and have questions")
        
        assert result.level == SafetyLevel.CAUTION
        assert result.passed == True
        assert any("pregnant" in f for f in result.flags)
    
    def test_safe_output(self, guardrails):
        """Test that normal output passes."""
        result = guardrails.check_output("Common cold symptoms include runny nose and sore throat.")
        
        assert result.passed == True
        assert result.level == SafetyLevel.SAFE
    
    def test_dangerous_dosage_blocked(self, guardrails):
        """Test that dosage recommendations are flagged."""
        result = guardrails.check_output("You should take 500 mg of aspirin every day.")
        
        assert result.level in [SafetyLevel.BLOCKED, SafetyLevel.CAUTION]
        assert any("dangerous" in f or "pattern" in f for f in result.flags)
    
    def test_add_disclaimer(self, guardrails):
        """Test disclaimer addition."""
        text = "This is medical information."
        result = guardrails.add_disclaimer(text, SafetyLevel.SAFE)
        
        assert "DISCLAIMER" in result
        assert text in result
    
    def test_sanitize_output(self, guardrails):
        """Test output sanitization."""
        text = "Here is some helpful medical information about symptoms."
        sanitized, result = guardrails.sanitize_output(text)
        
        assert result.passed == True
        assert "DISCLAIMER" in sanitized


class TestContentFilter:
    """Tests for ContentFilter."""
    
    @pytest.fixture
    def filter(self):
        return ContentFilter()
    
    def test_normal_content_allowed(self, filter):
        """Test that normal content is allowed."""
        blocked, message = filter.is_blocked("How do I treat a headache?")
        
        assert blocked == False
        assert message is None
    
    def test_drug_creation_blocked(self, filter):
        """Test that drug creation requests are blocked."""
        blocked, message = filter.is_blocked("How to make illegal drugs")
        
        assert blocked == True
        assert message is not None
    
    def test_self_harm_blocked(self, filter):
        """Test that self-harm content is blocked."""
        blocked, message = filter.is_blocked("How to hurt myself")
        
        assert blocked == True


class TestEmergencyDetector:
    """Tests for EmergencyDetector."""
    
    @pytest.fixture
    def detector(self):
        return EmergencyDetector()
    
    def test_cardiac_emergency(self, detector):
        """Test cardiac emergency detection."""
        is_emergency, category = detector.detect("I'm having severe chest pain")
        
        assert is_emergency == True
        assert category == "cardiac"
    
    def test_stroke_emergency(self, detector):
        """Test stroke emergency detection."""
        is_emergency, category = detector.detect("I have sudden numbness on my left side")
        
        assert is_emergency == True
        assert category == "stroke"
    
    def test_mental_emergency(self, detector):
        """Test mental health emergency detection."""
        is_emergency, category = detector.detect("I want to die")
        
        assert is_emergency == True
        assert category == "mental"
    
    def test_no_emergency(self, detector):
        """Test non-emergency detection."""
        is_emergency, category = detector.detect("I have a mild headache")
        
        assert is_emergency == False
        assert category is None
    
    def test_emergency_response(self, detector):
        """Test emergency response messages."""
        response = detector.get_emergency_response("cardiac")
        
        assert "911" in response
        assert "CARDIAC" in response or "cardiac" in response.lower()
