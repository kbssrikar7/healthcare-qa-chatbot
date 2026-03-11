"""
Tests for CacheManager component.
Tests cache hit, miss, TTL expiration, and invalidation.
"""
import pytest
import time
from pathlib import Path
from src.utils.cache_manager import CacheManager


class TestCacheManager:
    """Test CacheManager operations."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a CacheManager with a temp directory and short TTL."""
        return CacheManager(cache_dir=str(tmp_path / "cache"), ttl_seconds=2, max_memory_items=5)

    def test_cache_miss_returns_none(self, cache):
        """Test that a cache miss returns None."""
        result = cache.get_cached_response("non-existent query")
        assert result is None

    def test_cache_hit_returns_response(self, cache):
        """Test that a cached response is returned correctly."""
        response = {"answer": "Diabetes is a metabolic disorder.", "confidence": 0.85}
        cache.cache_response("What is diabetes?", response)
        result = cache.get_cached_response("What is diabetes?")
        assert result is not None
        assert result["answer"] == "Diabetes is a metabolic disorder."
        assert result["confidence"] == 0.85

    def test_cache_ttl_expiration(self, cache):
        """Test that entries expire after TTL."""
        response = {"answer": "Some answer"}
        cache.cache_response("short lived query", response)
        # Should exist immediately
        assert cache.get_cached_response("short lived query") is not None
        # Wait for TTL to expire (2 seconds)
        time.sleep(2.1)
        assert cache.get_cached_response("short lived query") is None

    def test_cache_invalidation(self, cache):
        """Test that invalidation clears all caches."""
        cache.cache_response("q1", {"answer": "a1"})
        cache.cache_response("q2", {"answer": "a2"})
        cache.cache_embedding("text1", [0.1, 0.2, 0.3])
        # All should exist
        assert cache.get_cached_response("q1") is not None
        assert cache.get_cached_embedding("text1") is not None
        # Invalidate
        cache.invalidate_cache()
        assert cache.get_cached_response("q1") is None
        assert cache.get_cached_response("q2") is None
        assert cache.get_cached_embedding("text1") is None

    def test_cache_case_insensitive(self, cache):
        """Test that queries are normalized (lowered, stripped)."""
        cache.cache_response("What Is Diabetes?", {"answer": "test"})
        result = cache.get_cached_response("what is diabetes?")
        assert result is not None

    def test_embedding_cache(self, cache):
        """Test embedding caching."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        cache.cache_embedding("diabetes symptoms", embedding)
        result = cache.get_cached_embedding("diabetes symptoms")
        assert result == embedding

    def test_cache_stats(self, cache):
        """Test cache statistics reporting."""
        cache.cache_response("q1", {"answer": "a1"})
        cache.cache_embedding("t1", [1, 2, 3])
        stats = cache.get_cache_stats()
        assert stats["query_cache_size"] == 1
        assert stats["embedding_cache_size"] == 1
        assert stats["ttl_seconds"] == 2

    def test_cache_eviction_when_full(self, cache):
        """Test that old entries are evicted when max_memory_items is reached."""
        # max_memory_items=5, add 6 entries
        for i in range(6):
            cache.cache_response(f"query_{i}", {"answer": f"answer_{i}"})
        stats = cache.get_cache_stats()
        assert stats["query_cache_size"] <= 5

    def test_context_key_separation(self, cache):
        """Test that different context keys produce different cache entries."""
        cache.cache_response("same query", {"answer": "from model A"}, context_key="modelA")
        cache.cache_response("same query", {"answer": "from model B"}, context_key="modelB")
        a = cache.get_cached_response("same query", context_key="modelA")
        b = cache.get_cached_response("same query", context_key="modelB")
        assert a["answer"] == "from model A"
        assert b["answer"] == "from model B"

    def test_disk_persistence(self, cache, tmp_path):
        """Test that cached responses persist to disk."""
        cache.cache_response("persistent query", {"answer": "disk answer"})
        # Create new cache manager pointing to same directory
        cache2 = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_seconds=2)
        result = cache2.get_cached_response("persistent query")
        assert result is not None
        assert result["answer"] == "disk answer"
