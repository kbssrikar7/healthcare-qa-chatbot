"""
User Feedback Collection for Continuous Improvement.

Collects and stores user feedback on QA responses for:
- Model improvement and fine-tuning
- Quality monitoring
- Issue identification
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class UserFeedback:
    """User feedback on a QA response."""
    feedback_id: str
    question_id: str
    session_id: Optional[str]
    rating: int  # 1-5
    was_helpful: bool
    was_accurate: bool
    was_safe: bool
    feedback_text: Optional[str]
    timestamp: datetime
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class FeedbackStats:
    """Aggregate feedback statistics."""
    total_feedback: int
    avg_rating: float
    helpful_rate: float
    accuracy_rate: float
    safety_rate: float
    low_rated_count: int


class FeedbackCollector:
    """
    Collect and store user feedback for model improvement.
    
    Features:
    - Persistent storage (JSONL format)
    - Retrieval of low-rated samples for review
    - Aggregate statistics
    """
    
    def __init__(
        self,
        storage_path: str = "data/feedback/user_feedback.jsonl",
        auto_save: bool = True
    ):
        """
        Initialize feedback collector.
        
        Args:
            storage_path: Path to save feedback
            auto_save: Automatically save after each submission
        """
        self.storage_path = Path(storage_path)
        self.auto_save = auto_save
        self._ensure_storage()
        self._cache: List[UserFeedback] = []
    
    def _ensure_storage(self):
        """Ensure storage directory exists."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    def submit_feedback(
        self,
        question_id: str,
        rating: int,
        was_helpful: bool,
        was_accurate: bool,
        was_safe: bool = True,
        feedback_text: Optional[str] = None,
        session_id: Optional[str] = None,
        **metadata
    ) -> UserFeedback:
        """
        Store user feedback.
        
        Args:
            question_id: ID of the question/response
            rating: 1-5 rating
            was_helpful: Whether response was helpful
            was_accurate: Whether response was accurate
            was_safe: Whether response was safe
            feedback_text: Optional text feedback
            session_id: Optional session ID
            **metadata: Additional metadata
            
        Returns:
            Created UserFeedback object
        """
        # Validate rating
        rating = max(1, min(5, rating))
        
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            question_id=question_id,
            session_id=session_id,
            rating=rating,
            was_helpful=was_helpful,
            was_accurate=was_accurate,
            was_safe=was_safe,
            feedback_text=feedback_text,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self._cache.append(feedback)
        
        if self.auto_save:
            self._save_feedback(feedback)
        
        logger.info(f"Feedback submitted: rating={rating}, helpful={was_helpful}")
        return feedback
    
    def _save_feedback(self, feedback: UserFeedback):
        """Append feedback to storage."""
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(feedback.to_dict()) + '\n')
    
    def load_all_feedback(self) -> List[UserFeedback]:
        """Load all feedback from storage."""
        feedback_list = []
        
        if not self.storage_path.exists():
            return feedback_list
        
        with open(self.storage_path) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    feedback_list.append(UserFeedback(**data))
                except Exception as e:
                    logger.warning(f"Error parsing feedback: {e}")
        
        return feedback_list
    
    def get_low_rated_samples(
        self,
        threshold: int = 2,
        limit: int = 100
    ) -> List[UserFeedback]:
        """
        Get samples rated poorly for review/retraining.
        
        Args:
            threshold: Maximum rating to include (inclusive)
            limit: Maximum samples to return
            
        Returns:
            List of low-rated feedback
        """
        all_feedback = self.load_all_feedback()
        
        low_rated = [f for f in all_feedback if f.rating <= threshold]
        
        # Sort by timestamp (most recent first)
        low_rated.sort(key=lambda f: f.timestamp, reverse=True)
        
        return low_rated[:limit]
    
    def get_inaccurate_samples(self, limit: int = 100) -> List[UserFeedback]:
        """Get samples marked as inaccurate."""
        all_feedback = self.load_all_feedback()
        
        inaccurate = [f for f in all_feedback if not f.was_accurate]
        inaccurate.sort(key=lambda f: f.timestamp, reverse=True)
        
        return inaccurate[:limit]
    
    def get_unsafe_samples(self, limit: int = 100) -> List[UserFeedback]:
        """Get samples marked as unsafe."""
        all_feedback = self.load_all_feedback()
        
        unsafe = [f for f in all_feedback if not f.was_safe]
        unsafe.sort(key=lambda f: f.timestamp, reverse=True)
        
        return unsafe[:limit]
    
    def get_statistics(self) -> FeedbackStats:
        """Calculate aggregate statistics."""
        all_feedback = self.load_all_feedback()
        
        if not all_feedback:
            return FeedbackStats(
                total_feedback=0,
                avg_rating=0,
                helpful_rate=0,
                accuracy_rate=0,
                safety_rate=0,
                low_rated_count=0
            )
        
        total = len(all_feedback)
        avg_rating = sum(f.rating for f in all_feedback) / total
        helpful_count = sum(1 for f in all_feedback if f.was_helpful)
        accurate_count = sum(1 for f in all_feedback if f.was_accurate)
        safe_count = sum(1 for f in all_feedback if f.was_safe)
        low_rated = sum(1 for f in all_feedback if f.rating <= 2)
        
        return FeedbackStats(
            total_feedback=total,
            avg_rating=avg_rating,
            helpful_rate=helpful_count / total,
            accuracy_rate=accurate_count / total,
            safety_rate=safe_count / total,
            low_rated_count=low_rated
        )
    
    def export_for_training(
        self,
        output_path: str,
        min_rating: int = 4
    ) -> int:
        """
        Export high-quality samples for fine-tuning.
        
        Args:
            output_path: Path to export JSON
            min_rating: Minimum rating to include
            
        Returns:
            Number of samples exported
        """
        all_feedback = self.load_all_feedback()
        
        high_quality = [
            f.to_dict() for f in all_feedback
            if f.rating >= min_rating and f.was_helpful and f.was_accurate
        ]
        
        with open(output_path, 'w') as f:
            json.dump(high_quality, f, indent=2, default=str)
        
        logger.info(f"Exported {len(high_quality)} samples to {output_path}")
        return len(high_quality)


# Simple thumbs up/down feedback API
class SimpleFeedback:
    """Simplified thumbs up/down feedback."""
    
    def __init__(self, collector: Optional[FeedbackCollector] = None):
        self.collector = collector or FeedbackCollector()
    
    def thumbs_up(self, question_id: str, **kwargs) -> UserFeedback:
        """Record positive feedback."""
        return self.collector.submit_feedback(
            question_id=question_id,
            rating=5,
            was_helpful=True,
            was_accurate=True,
            **kwargs
        )
    
    def thumbs_down(
        self,
        question_id: str,
        reason: Optional[str] = None,
        **kwargs
    ) -> UserFeedback:
        """Record negative feedback."""
        return self.collector.submit_feedback(
            question_id=question_id,
            rating=1,
            was_helpful=False,
            was_accurate=False,
            feedback_text=reason,
            **kwargs
        )
