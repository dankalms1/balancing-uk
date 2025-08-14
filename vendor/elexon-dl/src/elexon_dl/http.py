import asyncio, random, time
from typing import Any, Dict, Optional
from collections import deque
import httpx
from .config import Settings

RETRIABLE = {429, 500, 502, 503, 504}

class RateLimiter:
    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None):
        self.rate = rate_per_sec
        self.capacity = capacity or max(1.0, rate_per_sec)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.updated
            self.updated = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0

class HTTPMetrics:
    def __init__(self, max_samples: int = 5000):
        self.count = 0
        self.latencies = deque(maxlen=max_samples)
        self._lock = asyncio.Lock()

    async def record(self, elapsed: float):
        async with self._lock:
            self.count += 1
            self.latencies.append(elapsed)

    def snapshot(self):
        n = self.count
        if self.latencies:
            avg = sum(self.latencies)/len(self.latencies)
            mx = max(self.latencies)
        else:
            avg = 0.0
            mx = 0.0
        return {"count": n, "avg_latency": avg, "max_latency": mx}

class AsyncHTTP:
    def __init__(self, settings: Settings):
        self.s = settings
        self._client: Optional[httpx.AsyncClient] = None
        self._limiter = RateLimiter(self.s.rate_per_sec)
        self.metrics = HTTPMetrics()

    async def __aenter__(self):
        headers = {"User-Agent": self.s.user_agent}
        if self.s.cache_enabled:
            headers["Cache-Control"] = "no-cache"
        self._client = httpx.AsyncClient(
            timeout=self.s.timeout_s,
            headers=headers,
            limits=httpx.Limits(max_keepalive_connections=self.s.max_concurrency,
                                max_connections=self.s.max_concurrency),
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client:
            await self._client.aclose()

    async def _sleep_backoff(self, attempt: int):
        base = self.s.backoff_base
        cap = self.s.backoff_cap
        sleep = min(cap, base * (2 ** attempt))
        sleep *= 0.5 + random.random()
        await asyncio.sleep(sleep)

    async def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> httpx.Response:
        assert self._client is not None, "Use within 'async with AsyncHTTP(settings)'"
        await self._limiter.acquire()
        params = params or {}
        for attempt in range(self.s.max_retries + 1):
            t0 = time.perf_counter()
            resp = await self._client.get(url, params=params)
            elapsed = time.perf_counter() - t0
            if resp.status_code in RETRIABLE:
                if attempt < self.s.max_retries:
                    await self._sleep_backoff(attempt)
                    continue
            await self.metrics.record(elapsed)
            return resp
