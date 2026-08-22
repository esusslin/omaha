"""HTTP fetching with conditional requests and backoff.

Two things this gets right that naive fetchers don't:

1. **Conditional requests.** We store ETag / Last-Modified per source and send
   If-None-Match / If-Modified-Since. A 304 costs the origin almost nothing and tells
   us definitively that nothing changed. Polling 32 club sites every hour without this
   is rude and gets you blocked.

2. **Failures are recorded, not raised.** A dead source should mark itself unhealthy and
   let the rest of the sweep continue. `/health` surfaces it. Nothing crashes because
   one team reorganised their site.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from omaha.config import get_settings

settings = get_settings()

# Last request time per host. Conditional requests make repeat polls of one URL cheap,
# but a backfill walks dozens of new article URLs per club and ETags do nothing there —
# so pace by host. Locked because the scheduler may run jobs on separate threads.
_last_request_at: dict[str, float] = {}
_throttle_lock = threading.Lock()


def _throttle(url: str) -> None:
    """Sleep just long enough to keep one host under the configured request rate."""
    interval = settings.min_request_interval_seconds
    if interval <= 0:
        return

    host = urlparse(url).netloc
    with _throttle_lock:
        elapsed = time.monotonic() - _last_request_at.get(host, 0.0)
        wait = interval - elapsed
        if wait > 0:
            time.sleep(wait)
        _last_request_at[host] = time.monotonic()


@dataclass(frozen=True)
class FetchResult:
    """Outcome of one fetch attempt."""

    url: str
    status_code: int
    fetched_at: dt.datetime
    content: bytes | None = None
    content_type: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    error: str | None = None

    @property
    def not_modified(self) -> bool:
        """Origin confirmed nothing changed. Not a failure."""
        return self.status_code == 304

    @property
    def ok(self) -> bool:
        return self.status_code == 200 and self.content is not None

    @property
    def content_hash(self) -> str | None:
        if self.content is None:
            return None
        return hashlib.sha256(self.content).hexdigest()


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _request(client: httpx.Client, url: str, headers: dict[str, str]) -> httpx.Response:
    return client.get(url, headers=headers, follow_redirects=True)


def fetch(
    url: str,
    *,
    etag: str | None = None,
    last_modified: str | None = None,
    client: httpx.Client | None = None,
) -> FetchResult:
    """Fetch a URL, honouring conditional-request caching.

    Never raises for HTTP or network problems — returns a FetchResult carrying the error
    so the caller can record it against the source and carry on.
    """
    now = dt.datetime.now(dt.UTC)
    headers = {"User-Agent": settings.user_agent}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified

    owned_client = client is None
    client = client or httpx.Client(timeout=settings.request_timeout_seconds)

    try:
        _throttle(url)
        response = _request(client, url, headers)
    except Exception as exc:
        return FetchResult(url=url, status_code=0, fetched_at=now, error=repr(exc))
    finally:
        if owned_client:
            client.close()

    if response.status_code == 304:
        return FetchResult(
            url=url,
            status_code=304,
            fetched_at=now,
            etag=etag,
            last_modified=last_modified,
        )

    if response.status_code != 200:
        return FetchResult(
            url=url,
            status_code=response.status_code,
            fetched_at=now,
            error=f"HTTP {response.status_code}",
        )

    return FetchResult(
        url=str(response.url),
        status_code=200,
        fetched_at=now,
        content=response.content,
        content_type=response.headers.get("content-type"),
        etag=response.headers.get("etag"),
        last_modified=response.headers.get("last-modified"),
    )
