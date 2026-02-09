"""
Conversation module for multi-turn QA support.
"""
from src.conversation.history import (
    Conversation,
    ConversationTurn,
    ConversationManager,
    FollowUpDetector
)

__all__ = [
    'Conversation',
    'ConversationTurn', 
    'ConversationManager',
    'FollowUpDetector'
]
