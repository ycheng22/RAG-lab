"""API-key authentication and a per-key rate limit.

Every call to /ask costs real money (OpenAI tokens), so the service never runs open to the internet:
clients must send the shared key in the X-API-Key header, and each key gets at most
RATE_LIMIT requests per WINDOW_SECONDS. The limiter lives in memory: fine for one instance
(the deploy guide caps Cloud Run at one); use Redis or your API gateway for more.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException

from raglab.config import get_settings

RATE_LIMIT = 30
WINDOW_SECONDS = 60.0


def dev_mode() -> bool:
    """RAGLAB_DEV=1 allows running without an API key (local development only)."""
    return os.environ.get("RAGLAB_DEV", "") == "1"


def check_startup_config() -> None:
    """Refuse to start without an API key, unless explicitly in dev mode."""
    if not get_settings().service_api_key and not dev_mode():
        raise RuntimeError(
            "RAGLAB_SERVICE_API_KEY is not set. Set it (clients send it as X-API-Key), "
            "or set RAGLAB_DEV=1 to run without a key on your own machine."
        )


class RateLimiter:
    """Sliding window: remembers the timestamps of each key's recent requests."""

    def __init__(self, limit: int = RATE_LIMIT, window: float = WINDOW_SECONDS) -> None:
        self.limit, self.window = limit, window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        """(allowed, seconds until the oldest request leaves the window)."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, int(self.window - (now - hits[0])) + 1
            hits.append(now)
            return True, 0


limiter = RateLimiter()


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency: 401 without a valid key, 429 when the key is over its limit."""
    expected = get_settings().service_api_key
    if expected:
        if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
        key = x_api_key
    else:  # dev mode (check_startup_config guarantees it)
        key = "dev"
    allowed, retry_after = limiter.allow(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: {RATE_LIMIT} requests per minute",
            headers={"Retry-After": str(retry_after)},
        )
    return key
