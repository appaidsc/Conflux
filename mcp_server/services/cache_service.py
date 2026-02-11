"""
Services - Cache Service

TTL-based caching with separate query and content tiers.
"""

from typing import Any, Optional
from datetime import datetime, timedelta
from cachetools import TTLCache
import threading

from mcp_server.config import get_settings


class CacheService:
    """TTL-based cache with query and content tiers."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        
        # Query cache (short TTL for search results)
        self._query_cache = TTLCache(
            maxsize=1000,
            ttl=self.settings.cache.ttl_query,
        )
        
        # Content cache (longer TTL for page content)
        self._content_cache = TTLCache(
            maxsize=500,
            ttl=self.settings.cache.ttl_content,
        )
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key (prefix determines tier)
            
        Returns:
            Cached value or None
        """
        if not self.settings.cache.enabled:
            return None
        
        with self._lock:
            cache = self._get_cache_tier(key)
            return cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key (prefix determines tier)
            value: Value to cache
        """
        if not self.settings.cache.enabled:
            return
        
        with self._lock:
            cache = self._get_cache_tier(key)
            cache[key] = value
    
    def delete(self, key: str) -> None:
        """Delete a key from cache."""
        with self._lock:
            cache = self._get_cache_tier(key)
            if key in cache:
                del cache[key]
    
    def clear_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._query_cache.clear()
            self._content_cache.clear()
    
    def clear_tier(self, tier: str) -> None:
        """Clear a specific cache tier."""
        with self._lock:
            if tier == "query":
                self._query_cache.clear()
            elif tier == "content":
                self._content_cache.clear()
    
    def _get_cache_tier(self, key: str) -> TTLCache:
        """Determine cache tier based on key prefix."""
        if key.startswith("query:") or key.startswith("search:"):
            return self._query_cache
        else:
            return self._content_cache
    
    def get_versioned(self, key: str, current_version: int) -> Optional[Any]:
        """
        Version-aware cache get — returns cached data only if version matches.
        
        This is the key to "smart caching":
        - If cached version == current version → return cached (0ms, no fetch)
        - If versions differ → return None (triggers fresh fetch)
        
        Args:
            key: Cache key
            current_version: Current version from Confluence
            
        Returns:
            Cached value if version matches, None otherwise
        """
        if not self.settings.cache.enabled:
            return None
        
        with self._lock:
            cache = self._get_cache_tier(key)
            entry = cache.get(key)
            
            if entry is None:
                return None
            
            # Entry is a tuple: (version, data)
            if isinstance(entry, tuple) and len(entry) == 2:
                cached_version, data = entry
                if cached_version == current_version:
                    return data  # Version match — instant return!
                else:
                    # Stale — delete and return None
                    del cache[key]
                    return None
            
            return entry  # Fallback for non-versioned entries
    
    def set_versioned(self, key: str, value: Any, version: int) -> None:
        """
        Store value with its version number.
        
        Args:
            key: Cache key
            value: Data to cache
            version: Current Confluence page version
        """
        if not self.settings.cache.enabled:
            return
        
        with self._lock:
            cache = self._get_cache_tier(key)
            cache[key] = (version, value)  # Store as (version, data) tuple
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "query_cache": {
                "size": len(self._query_cache),
                "maxsize": self._query_cache.maxsize,
                "ttl": self._query_cache.ttl,
            },
            "content_cache": {
                "size": len(self._content_cache),
                "maxsize": self._content_cache.maxsize,
                "ttl": self._content_cache.ttl,
            },
            "enabled": self.settings.cache.enabled,
        }
