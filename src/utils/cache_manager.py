"""
Cache manager for RAG system optimization.

Based on rag-service skill pattern for improved performance.
Caches embeddings and question-answer pairs with TTL.
"""

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclasses and other non-serializable objects."""

    def default(self, obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


class CacheManager:
    """
    Multi-level caching for RAG system components.

    Features:
    - Embedding cache (LRU in-memory)
    - Q&A response cache (with TTL)
    - Query cache for deduplication
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        ttl_seconds: int = 3600,  # 1 hour default
        max_memory_items: int = 1000,
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for persistent cache
            ttl_seconds: Time-to-live for cached items
            max_memory_items: Max items in memory cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.max_memory_items = max_memory_items

        # In-memory caches
        self._query_cache: Dict[str, Dict] = {}
        self._embedding_cache: Dict[str, Any] = {}

    def _hash_key(self, key: str, context_key: str = "") -> str:
        """Generate hash for cache key including optional context."""
        full_key = f"{key}::{context_key}"
        return hashlib.md5(full_key.encode()).hexdigest()

    def get_cached_response(self, query: str, context_key: str = "") -> Optional[Dict]:
        """
        Get cached Q&A response for query.

        Args:
            query: User query
            context_key: Optional context (model, pipeline, KB fingerprint)

        Returns:
            Cached response dict or None
        """
        key = self._hash_key(query.lower().strip(), context_key)

        # Check memory cache first
        if key in self._query_cache:
            entry = self._query_cache[key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["response"]
            else:
                # Expired
                del self._query_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"qa_{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    entry = json.load(f)
                if time.time() - entry["timestamp"] < self.ttl_seconds:
                    # Store in memory for fast access
                    self._query_cache[key] = entry
                    return entry["response"]
                else:
                    # Expired
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"Cache read failed for {cache_file}: {e}")

        return None

    def cache_response(self, query: str, response: Dict, context_key: str = "") -> None:
        """
        Cache Q&A response.

        Args:
            query: User query
            response: Response dict to cache
            context_key: Optional context (model, pipeline, KB fingerprint)
        """
        key = self._hash_key(query.lower().strip(), context_key)
        entry = {"query": query, "response": response, "timestamp": time.time()}

        # Store in memory
        if len(self._query_cache) >= self.max_memory_items:
            # Remove oldest entry
            oldest_key = min(
                self._query_cache.keys(),
                key=lambda k: self._query_cache[k]["timestamp"],
            )
            del self._query_cache[oldest_key]

        self._query_cache[key] = entry

        # Store on disk
        cache_file = self.cache_dir / f"qa_{key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(entry, f, cls=_SafeEncoder)
        except Exception as e:
            logger.warning(f"Cache write failed for {cache_file}: {e}")

    def get_cached_embedding(self, text: str) -> Optional[Any]:
        """
        Get cached embedding for text.

        Args:
            text: Text to get embedding for

        Returns:
            Cached embedding or None
        """
        key = self._hash_key(text)
        return self._embedding_cache.get(key)

    def cache_embedding(self, text: str, embedding: Any) -> None:
        """
        Cache text embedding.

        Args:
            text: Original text
            embedding: Generated embedding
        """
        key = self._hash_key(text)

        if len(self._embedding_cache) >= self.max_memory_items:
            # Clear half the cache (simple LRU approximation)
            keys_to_remove = list(self._embedding_cache.keys())[: self.max_memory_items // 2]
            for k in keys_to_remove:
                del self._embedding_cache[k]

        self._embedding_cache[key] = embedding

    def invalidate_cache(self) -> None:
        """Clear all caches (call when knowledge base updates)."""
        self._query_cache.clear()
        self._embedding_cache.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("qa_*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "query_cache_size": len(self._query_cache),
            "embedding_cache_size": len(self._embedding_cache),
            "cache_dir": str(self.cache_dir),
            "ttl_seconds": self.ttl_seconds,
        }
