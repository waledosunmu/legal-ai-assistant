"""NWLR Online crawler — authenticated, rate-limited, disk-cached, resumable.

Authentication flow:
  POST https://api.nwlronline.com/secure/v1/login
  Body: {"data": "<HS256 JWT>"}
  Header: signature: 2b72e1a-55d3-4afd-811f-63f24
    Response: {"success": {"access_token": "<bearer token>", "user_token": "<request signature>"}}

Case ID format: "{part}_1_{page_start}"  e.g. "2034_1_349"

Enumeration strategy (search endpoint is Cloudflare-protected):
  For each part, stride-probe pages 1, 50, 100, … until a case-details
  hit is found, then chain-follow via page_end + 1 to collect all cases
  in the part.

Usage::

    async with NWLRCrawler() as crawler:
        case_ids = await crawler.discover_part(2034)
        for cid in case_ids:
            meta = await crawler.fetch_case_metadata(cid)
            html = await crawler.fetch_case_html(cid)
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import jwt
import structlog

logger = structlog.get_logger(__name__)

_API_BASE = "https://api.nwlronline.com/secure/v1"
_JWT_SECRET = "abcdefghijklmnopqrst"  # noqa: S105 — site's own weak secret
_SIGNATURE_HEADER = "2b72e1a-55d3-4afd-811f-63f24"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
_DEFAULT_RATE_LIMIT = 3.0  # seconds between requests


@dataclass(frozen=True)
class NWLRCaseId:
    """Immutable identifier for one NWLR case."""

    part: int
    page_start: int

    def as_str(self) -> str:
        return f"{self.part}_1_{self.page_start}"

    @classmethod
    def from_str(cls, s: str) -> NWLRCaseId:
        parts = s.split("_")
        return cls(part=int(parts[0]), page_start=int(parts[2]))


class NWLRCrawler:
    """
    Authenticated crawler for NWLR Online.

    Design principles:
    1. Auth: HS256 JWT login; protected requests require both bearer token and session user_token.
    2. Rate-limited: 1 request per ``rate_limit_seconds`` (default 3s).
    3. Disk-cached: metadata JSON + HTML cached under ``raw_cache_dir``.
    4. Resumable: cache hits skip the network entirely.
    5. Async context manager: manages the shared httpx client lifecycle.
    """

    def __init__(
        self,
        email: str,
        password: str,
        raw_cache_dir: Path = Path("data/raw/nwlr"),
        rate_limit_seconds: float = _DEFAULT_RATE_LIMIT,
    ) -> None:
        self._email = email
        self._password = password
        self.raw_cache_dir = raw_cache_dir
        self.rate_limit_seconds = rate_limit_seconds
        self._current_rate_limit_seconds = rate_limit_seconds
        self._token: str | None = None
        self._user_token: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(1)  # serialise all outbound requests
        raw_cache_dir.mkdir(parents=True, exist_ok=True)
        (raw_cache_dir / "meta").mkdir(exist_ok=True)
        (raw_cache_dir / "html").mkdir(exist_ok=True)
        (raw_cache_dir / "probed").mkdir(exist_ok=True)  # null-probe cache per part

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def __aenter__(self) -> NWLRCrawler:
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": _USER_AGENT,
                "signature": _SIGNATURE_HEADER,
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://nwlronline.com",
                "Referer": "https://nwlronline.com/",
            },
            follow_redirects=True,
        )
        await self._authenticate()
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Authentication ────────────────────────────────────────────────────

    async def _authenticate(self) -> None:
        """Login and store the bearer token plus per-session request signature."""
        now = int(time.time())
        payload = {
            "email": self._email,
            "password": self._password,
            "jti": str(uuid.uuid4()),
            "iat": now,
            "exp": now + 3600,
        }
        token = jwt.encode(payload, _JWT_SECRET, algorithm="HS256")
        client = self._require_client()
        resp = await client.post(
            f"{_API_BASE}/login",
            json={"data": token},
        )
        if resp.status_code >= 400:
            logger.error(
                "nwlr.auth_failed",
                status=resp.status_code,
                message=self._response_message(resp),
            )
        resp.raise_for_status()
        data = resp.json()
        success = data.get("success") or {}
        access_token = success.get("access_token")
        user_token = success.get("user_token")
        if not access_token or not user_token:
            raise RuntimeError("NWLR login response missing access_token or user_token")

        self._token = access_token
        self._user_token = user_token
        logger.info("nwlr.authenticated", ipuser=bool(success.get("ipuser")))

    async def _ensure_auth(self) -> tuple[str, str]:
        """Return the current bearer token and request signature, re-authenticating if needed."""
        if not self._token or not self._user_token:
            await self._authenticate()
        assert self._token is not None
        assert self._user_token is not None
        return self._token, self._user_token

    # ── Public API ────────────────────────────────────────────────────────

    def _null_cache_path(self, part: int) -> Path:
        """Path to the per-part null-probe cache (JSON list of probed-null pages)."""
        return self.raw_cache_dir / "probed" / f"{part}.json"

    def _load_null_cache(self, part: int) -> set[int]:
        """Load the set of page numbers already probed (and confirmed null) for a part."""
        path = self._null_cache_path(part)
        if path.exists():
            return set(json.loads(path.read_text(encoding="utf-8")))
        return set()

    def _save_null_cache(self, part: int, probed_null: set[int]) -> None:
        """Persist the null-probe cache for a part."""
        path = self._null_cache_path(part)
        path.write_text(json.dumps(sorted(probed_null)), encoding="utf-8")

    async def fetch_case_metadata(self, case_id: NWLRCaseId) -> dict[str, Any] | None:
        """
        Fetch case metadata from ``/auth/case-details/{case_id}``.

        Returns the parsed ``data`` dict, or ``None`` if the case does not exist.
        Caches hits to ``data/raw/nwlr/meta/{case_id}.json``.
        Null responses are NOT cached here — the caller manages the null-probe cache.
        """
        cache_path = self.raw_cache_dir / "meta" / f"{case_id.as_str()}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        resp = await self._get(f"/auth/case-details/{case_id.as_str()}")
        if resp.status_code in (400, 404):
            return None
        if resp.status_code == 403:
            # Cloudflare block persisted through backoff — raise so caller can handle
            resp.raise_for_status()
        resp.raise_for_status()

        body = resp.json()
        meta: dict[str, Any] = body.get("data") or {}
        if not meta:
            return None

        cache_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return meta

    async def fetch_case_html(self, case_id: NWLRCaseId) -> str | None:
        """
        Fetch full judgment HTML from ``/auth/load-case/{case_id}``.

        Returns the HTML string, or ``None`` if unavailable.
        Caches to ``data/raw/nwlr/html/{case_id}.html``.
        """
        cache_path = self.raw_cache_dir / "html" / f"{case_id.as_str()}.html"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        resp = await self._get(f"/auth/load-case/{case_id.as_str()}")
        if resp.status_code in (400, 404):
            return None
        resp.raise_for_status()

        # Response is plain HTML (not JSON)
        html = resp.text
        if not html.strip():
            return None

        cache_path.write_text(html, encoding="utf-8")
        return html

    async def discover_part(
        self,
        part_num: int,
        max_scan_page: int = 700,
    ) -> list[NWLRCaseId]:
        """
        Discover all case IDs within ``part_num``.

        Strategy:
        1. Linear scan pages 1 → ``max_scan_page`` to find the FIRST case.
           Already-probed null pages are skipped via a per-part null-cache file
           (``data/raw/nwlr/probed/{part}.json``), making resumption fast.
        2. From the first hit, chain-follow via ``page_end + 1`` to collect
           all subsequent cases within the part.

        The null-cache persists between runs so incremental discovery works:
        run ``discover_part(2034)`` twice — the second run only probes pages
        not yet in the null-cache.

        Returns a list of NWLRCaseId sorted by page_start.
        """
        log = logger.bind(part=part_num)
        log.info("nwlr.discover_part_start", max_scan_page=max_scan_page)

        null_pages = self._load_null_cache(part_num)
        first_hit: dict[str, Any] | None = None

        # Step 1: linear scan to find first case
        for page in range(1, max_scan_page + 1):
            # Skip pages already confirmed null
            if page in null_pages:
                continue

            cid = NWLRCaseId(part=part_num, page_start=page)
            # If we have a cached metadata hit, use it directly
            meta_cache = self.raw_cache_dir / "meta" / f"{cid.as_str()}.json"
            if meta_cache.exists():
                first_hit = json.loads(meta_cache.read_text(encoding="utf-8"))
                log.info("nwlr.first_hit_cached", page_start=page)
                break

            meta = await self.fetch_case_metadata(cid)
            if meta is not None:
                first_hit = meta
                log.info("nwlr.first_hit", page_start=page)
                break
            else:
                null_pages.add(page)
                # Save null-cache every 50 probes to survive interruption
                if len(null_pages) % 50 == 0:
                    self._save_null_cache(part_num, null_pages)

        self._save_null_cache(part_num, null_pages)

        if first_hit is None:
            log.info("nwlr.no_cases_in_part")
            return []

        # Step 2: chain-follow via page_end + 1
        found: dict[int, NWLRCaseId] = {}
        cur_meta = first_hit
        while True:
            page_start = cur_meta["page_start"]
            page_end = cur_meta.get("page_end", page_start)
            found[page_start] = NWLRCaseId(part=part_num, page_start=page_start)

            next_page = page_end + 1
            if next_page in null_pages or next_page > max_scan_page:
                break

            next_cid = NWLRCaseId(part=part_num, page_start=next_page)
            next_meta = await self.fetch_case_metadata(next_cid)
            if next_meta is None:
                null_pages.add(next_page)
                break
            # Guard against API returning same page_start
            if next_meta["page_start"] == page_start:
                break
            cur_meta = next_meta

        self._save_null_cache(part_num, null_pages)
        result = sorted(found.values(), key=lambda c: c.page_start)
        log.info("nwlr.discover_part_done", cases=len(result))
        return result

    async def discover_all_parts(
        self,
        parts: list[int] | None = None,
        skip_existing: bool = True,
        manifest_path: Path = Path("data/nwlr_manifest.json"),
        max_scan_page: int = 700,
        on_part_error: Callable[[int, Exception], None] | None = None,
    ) -> dict[int, list[NWLRCaseId]]:
        """
        Discover cases across ``parts`` (default: all 1–2034).

        Loads/saves progress to ``manifest_path`` so runs are resumable.
        Returns mapping of part_num → list[NWLRCaseId].
        """
        manifest: dict[str, list[str]] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        all_parts = parts or list(range(1, 2035))
        result: dict[int, list[NWLRCaseId]] = {}

        for p in all_parts:
            key = str(p)
            if skip_existing and key in manifest:
                result[p] = [NWLRCaseId.from_str(s) for s in manifest[key]]
                continue

            try:
                ids = await self.discover_part(p, max_scan_page=max_scan_page)
            except Exception as exc:
                logger.error("nwlr.discover_part_failed", part=p, error=str(exc))
                if on_part_error is not None:
                    on_part_error(p, exc)
                result[p] = []
                continue
            result[p] = ids
            manifest[key] = [c.as_str() for c in ids]

            # Save manifest after each part so we can resume
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info("nwlr.manifest_saved", part=p, cases=len(ids))

        return result

    # ── Internal helpers ──────────────────────────────────────────────────

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("NWLRCrawler must be used as an async context manager")
        return self._client

    def _increase_rate_limit(
        self,
        *,
        multiplier: float = 1.5,
        floor: float | None = None,
        ceiling: float = 30.0,
    ) -> float:
        """Increase the live pacing after a block or throttling signal."""
        baseline = max(self._current_rate_limit_seconds, self.rate_limit_seconds)
        self._current_rate_limit_seconds = min(
            ceiling,
            max(floor or self.rate_limit_seconds, baseline * multiplier),
        )
        return self._current_rate_limit_seconds

    def _relax_rate_limit(self, *, decay_seconds: float = 0.25) -> float:
        """Gradually return to the configured base pacing after successful requests."""
        if self._current_rate_limit_seconds <= self.rate_limit_seconds:
            self._current_rate_limit_seconds = self.rate_limit_seconds
        else:
            self._current_rate_limit_seconds = max(
                self.rate_limit_seconds,
                self._current_rate_limit_seconds - decay_seconds,
            )
        return self._current_rate_limit_seconds

    @staticmethod
    def _response_message(resp: httpx.Response) -> str | None:
        """Extract a short diagnostic message without logging secrets or whole bodies."""
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = resp.json()
            except ValueError:
                return None
            if isinstance(body, dict):
                message = body.get("message")
                if isinstance(message, str):
                    return message[:200]
        return None

    async def _get(self, path: str) -> httpx.Response:
        """
        Rate-limited authenticated GET with:
        - Jitter (±0.5s) on top of base rate limit to avoid bot detection patterns
        - Automatic 401 re-auth (token expiry or invalid per-session signature)
        - 403 Cloudflare back-off: sleep 90s then retry once
        """
        resp = await self._get_once(path)

        if resp.status_code == 401:
            logger.warning(
                "nwlr.auth_rejected",
                path=path,
                status=resp.status_code,
                message=self._response_message(resp),
                retrying=True,
            )
            self._token = None
            self._user_token = None
            resp = await self._get_once(path)
            if resp.status_code == 401:
                logger.error(
                    "nwlr.auth_rejected_persisted",
                    path=path,
                    status=resp.status_code,
                    message=self._response_message(resp),
                    adaptive_rate_limit=round(self._increase_rate_limit(multiplier=1.25), 2),
                )

        if resp.status_code == 429:
            backoff_seconds = max(
                30.0,
                self._increase_rate_limit(multiplier=2.0, floor=10.0, ceiling=45.0) * 2,
            ) + random.uniform(0, 10)
            logger.warning(
                "nwlr.rate_limited",
                path=path,
                status=resp.status_code,
                message=self._response_message(resp),
                retrying=True,
                backoff_seconds=round(backoff_seconds, 2),
                adaptive_rate_limit=round(self._current_rate_limit_seconds, 2),
            )
            await asyncio.sleep(backoff_seconds)
            resp = await self._get_once(path)
            if resp.status_code == 429:
                logger.error(
                    "nwlr.rate_limited_persisted",
                    path=path,
                    status=resp.status_code,
                    message=self._response_message(resp),
                    adaptive_rate_limit=round(self._current_rate_limit_seconds, 2),
                )

        if resp.status_code == 403:
            backoff_seconds = 90 + random.uniform(0, 30)
            self._increase_rate_limit(multiplier=2.0, floor=10.0, ceiling=45.0)
            logger.warning(
                "nwlr.cloudflare_block",
                path=path,
                status=resp.status_code,
                message=self._response_message(resp),
                retrying=True,
                backoff_seconds=round(backoff_seconds, 2),
                adaptive_rate_limit=round(self._current_rate_limit_seconds, 2),
            )
            await asyncio.sleep(backoff_seconds)
            resp = await self._get_once(path)
            if resp.status_code == 403:
                logger.error(
                    "nwlr.cloudflare_block_persisted",
                    path=path,
                    status=resp.status_code,
                    message=self._response_message(resp),
                    adaptive_rate_limit=round(self._current_rate_limit_seconds, 2),
                )

        if resp.status_code not in (403, 429):
            self._relax_rate_limit()

        logger.debug("nwlr.get", path=path, status=resp.status_code)
        return resp

    async def _get_once(self, path: str) -> httpx.Response:
        """Single rate-limited GET (no retry logic)."""
        async with self._semaphore:
            # Jitter: add ±50% of rate_limit to break predictable timing patterns
            jitter = random.uniform(
                -0.5 * self._current_rate_limit_seconds,
                0.5 * self._current_rate_limit_seconds,
            )
            await asyncio.sleep(max(0.1, self._current_rate_limit_seconds + jitter))
            token, user_token = await self._ensure_auth()
            client = self._require_client()
            return await client.get(
                f"{_API_BASE}{path}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "signature": user_token,
                },
            )
