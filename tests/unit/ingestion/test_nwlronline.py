from __future__ import annotations

from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from ingestion.sources.nwlronline import NWLRCaseId, NWLRCrawler


def make_crawler(tmp_path: Path) -> NWLRCrawler:
    return NWLRCrawler(
        email="user@example.com",
        password="secret",
        raw_cache_dir=tmp_path / "nwlr",
        rate_limit_seconds=0.0,
    )


@pytest.mark.asyncio
async def test_fetch_case_metadata_uses_session_user_token_signature(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://api.nwlronline.com/secure/v1/login",
        json={
            "success": {
                "status": "success",
                "access_token": "access-1",
                "user_token": "user-token-1",
                "ipuser": False,
            }
        },
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.nwlronline.com/secure/v1/auth/case-details/2034_1_349",
        json={
            "status": "success",
            "message": "Result fetched successfully",
            "data": {
                "part": 2034,
                "page_start": 349,
                "page_end": 384,
                "case_name": "Ibrahim v. Kano State",
            },
        },
    )

    async with make_crawler(tmp_path) as crawler:
        meta = await crawler.fetch_case_metadata(NWLRCaseId(part=2034, page_start=349))

    requests = httpx_mock.get_requests()
    assert requests[1].headers["Authorization"] == "Bearer access-1"
    assert requests[1].headers["signature"] == "user-token-1"
    assert meta is not None
    assert meta["page_start"] == 349


@pytest.mark.asyncio
async def test_get_reauthenticates_and_refreshes_session_signature_on_401(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://api.nwlronline.com/secure/v1/login",
        json={
            "success": {
                "status": "success",
                "access_token": "access-1",
                "user_token": "user-token-1",
                "ipuser": False,
            }
        },
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.nwlronline.com/secure/v1/auth/case-details/2034_1_349",
        status_code=401,
        json={"status": "error", "message": "Invalid Signature"},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.nwlronline.com/secure/v1/login",
        json={
            "success": {
                "status": "success",
                "access_token": "access-2",
                "user_token": "user-token-2",
                "ipuser": False,
            }
        },
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.nwlronline.com/secure/v1/auth/case-details/2034_1_349",
        json={
            "status": "success",
            "message": "Result fetched successfully",
            "data": {
                "part": 2034,
                "page_start": 349,
                "page_end": 384,
                "case_name": "Ibrahim v. Kano State",
            },
        },
    )

    async with make_crawler(tmp_path) as crawler:
        meta = await crawler.fetch_case_metadata(NWLRCaseId(part=2034, page_start=349))

    requests = [request for request in httpx_mock.get_requests() if request.method == "GET"]
    assert requests[0].headers["Authorization"] == "Bearer access-1"
    assert requests[0].headers["signature"] == "user-token-1"
    assert requests[1].headers["Authorization"] == "Bearer access-2"
    assert requests[1].headers["signature"] == "user-token-2"
    assert meta is not None
    assert meta["case_name"] == "Ibrahim v. Kano State"
