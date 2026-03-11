"""
Conversation History Manager for Multi-turn QA.

Enables multi-turn conversations with context preservation,
follow-up question detection, and session management.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import re
from loguru import logger


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    question: str
    answer: str
    timestamp: datetime
    confidence: float
    sources: List[str]
    is_followup: bool = False
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'question': self.question,
            'answer': self.answer,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'sources': self.sources,
            'is_followup': self.is_followup,
            'metadata': self.metadata
        }


@dataclass
class Conversation:
    """
    A conversation session with history.
    
    Tracks all Q&A turns, enables context window for follow-ups,
    and provides session management.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: List[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def add_turn(
        self,
        question: str,
        answer: str,
        confidence: float,
        sources: List[str],
        is_followup: bool = False,
        **kwargs
    ) -> ConversationTurn:
        """Add a new turn to the conversation."""
        turn = ConversationTurn(
            question=question,
            answer=answer,
            timestamp=datetime.now(),
            confidence=confidence,
            sources=sources,
            is_followup=is_followup,
            metadata=kwargs
        )
        self.turns.append(turn)
        self.last_activity = datetime.now()
        return turn
    
    def get_context_window(self, max_turns: int = 3, max_chars: int = 2000) -> str:
        """
        Get recent conversation context for follow-up questions.
        
        Args:
            max_turns: Maximum number of turns to include
            max_chars: Maximum total characters
            
        Returns:
            Formatted context string
        """
        if not self.turns:
            return ""
        
        recent = self.turns[-max_turns:]
        context = []
        total_chars = 0
        
        for turn in recent:
            # Truncate answer if needed
            answer_preview = turn.answer[:200] + "..." if len(turn.answer) > 200 else turn.answer
            turn_text = f"User: {turn.question}\nAssistant: {answer_preview}"
            
            if total_chars + len(turn_text) > max_chars:
                break
            
            context.append(turn_text)
            total_chars += len(turn_text)
        
        return "\n\n".join(context)
    
    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent turn."""
        return self.turns[-1] if self.turns else None
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'num_turns': len(self.turns),
            'turns': [t.to_dict() for t in self.turns],
            'metadata': self.metadata
        }


class FollowUpDetector:
    """
    Detect if a question is a follow-up to previous context.
    
    Uses linguistic cues like pronouns and references to
    identify when user is continuing a topic.
    """
    
    # Pronouns that indicate reference to previous content
    REFERENCE_PRONOUNS = {
        'it', 'its', 'this', 'that', 'these', 'those',
        'they', 'them', 'their', 'he', 'she', 'his', 'her'
    }
    
    # Phrases indicating follow-up
    FOLLOWUP_PHRASES = [
        'what about', 'how about', 'and what',
        'can you also', 'tell me more', 'more about',
        'another question', 'one more', 'also',
        'regarding that', 'on that note', 'speaking of',
        'what else', 'anything else', 'is there',
        'why is that', 'how come', 'but why'
    ]
    
    # Short questions indicating follow-up
    SHORT_QUESTION_MAX_WORDS = 6
    
    def detect_followup(self, question: str, previous_context: Optional[str] = None) -> bool:
        """
        Detect if question is likely a follow-up.
        
        Args:
            question: Current question
            previous_context: Previous conversation context
            
        Returns:
            True if likely a follow-up
        """
        question_lower = question.lower().strip()
        words = question_lower.split()
        has_context = bool(previous_context and previous_context.strip())
        
        # Check for follow-up phrases
        for phrase in self.FOLLOWUP_PHRASES:
            if phrase in question_lower:
                return True
        
        # Check if starts with reference pronoun
        if words and words[0] in self.REFERENCE_PRONOUNS:
            return True
        
        # Check for pronouns in first few words
        first_words = words[:3]
        if any(w in self.REFERENCE_PRONOUNS for w in first_words):
            # Short questions with pronouns are likely follow-ups
            if len(words) <= self.SHORT_QUESTION_MAX_WORDS:
                return True
        
        # Very short questions are often follow-ups
        if len(words) <= 3 and has_context:
            return True
        
        # Missing-subject heuristic only applies when there is prior context.
        if not self._has_clear_subject(question):
            return has_context
        
        return False
    
    def _has_clear_subject(self, question: str) -> bool:
        """Check if question has a clear subject (not implied)."""
        question_lower = question.lower()
        
        # Questions starting with specific medical terms likely have clear subject
        medical_starters = ['what is', 'what are', 'how to treat', 'symptoms of',
                           'causes of', 'treatment for', 'medication for']
        
        for starter in medical_starters:
            if question_lower.startswith(starter):
                return True
        
        return False


class ConversationManager:
    """
    Manage multiple conversation sessions.
    
    Handles session creation, retrieval, and persistence.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize conversation manager.
        
        Args:
            storage_path: Optional path to persist conversations
        """
        self.conversations: Dict[str, Conversation] = {}
        self.storage_path = storage_path
        self.followup_detector = FollowUpDetector()
    
    def create_session(self, metadata: Optional[Dict] = None) -> Conversation:
        """Create a new conversation session."""
        conv = Conversation(metadata=metadata or {})
        self.conversations[conv.session_id] = conv
        logger.info(f"Created conversation session: {conv.session_id}")
        return conv
    
    def get_session(self, session_id: str) -> Optional[Conversation]:
        """Get an existing conversation by ID."""
        return self.conversations.get(session_id)
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> Conversation:
        """Get existing session or create new one."""
        if session_id and session_id in self.conversations:
            return self.conversations[session_id]
        return self.create_session()
    
    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        confidence: float,
        sources: List[str],
        **kwargs
    ) -> Optional[ConversationTurn]:
        """
        Add a turn to an existing session.
        
        Automatically detects if it's a follow-up question.
        """
        conv = self.get_session(session_id)
        if not conv:
            logger.warning(f"Session not found: {session_id}")
            return None
        
        # Detect if follow-up
        is_followup = self.followup_detector.detect_followup(
            question,
            conv.get_context_window() if conv.turns else None
        )
        
        return conv.add_turn(
            question=question,
            answer=answer,
            confidence=confidence,
            sources=sources,
            is_followup=is_followup,
            **kwargs
        )
    
    def get_context_for_query(
        self,
        session_id: str,
        question: str
    ) -> Optional[str]:
        """
        Get conversation context to append to query.
        
        Returns formatted context if question is a follow-up.
        """
        conv = self.get_session(session_id)
        if not conv or not conv.turns:
            return None
        
        if self.followup_detector.detect_followup(question):
            return conv.get_context_window()
        
        return None
    
    def save_sessions(self):
        """Save all sessions to storage."""
        if not self.storage_path:
            return
        
        data = {
            sid: conv.to_dict()
            for sid, conv in self.conversations.items()
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours."""
        cutoff = datetime.now()
        to_remove = []
        
        for sid, conv in self.conversations.items():
            age = (cutoff - conv.last_activity).total_seconds() / 3600
            if age > max_age_hours:
                to_remove.append(sid)
        
        for sid in to_remove:
            del self.conversations[sid]
            logger.info(f"Cleaned up old session: {sid}")
