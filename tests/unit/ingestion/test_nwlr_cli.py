from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from scripts import nwlr_crawl


class _FakeCrawler:
    def __init__(self, *_: object, **__: object) -> None:
        self.meta_calls: list[str] = []
        self.html_calls: list[str] = []

    async def __aenter__(self) -> _FakeCrawler:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def fetch_case_metadata(self, case_id):
        self.meta_calls.append(case_id.as_str())
        return {
            "case_name": "Probe v. Test",
            "page_start": case_id.page_start,
            "page_end": case_id.page_start + 10,
        }

    async def fetch_case_html(self, case_id):
        self.html_calls.append(case_id.as_str())
        return "<html>ok</html>"


def test_health_command_uses_explicit_case_id(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake = _FakeCrawler()

    monkeypatch.setattr(nwlr_crawl, "NWLRCrawler", lambda *args, **kwargs: fake)
    monkeypatch.setattr(nwlr_crawl.settings, "nwlr_email", "user@example.com")
    monkeypatch.setattr(nwlr_crawl.settings, "nwlr_password", "secret")
    monkeypatch.setattr(nwlr_crawl, "_RAW_DIR", tmp_path / "raw")

    result = runner.invoke(nwlr_crawl.cli, ["health", "--case-id", "2034_1_349"])

    assert result.exit_code == 0
    assert "Health check OK" in result.output
    assert "2034_1_349" in result.output
    assert fake.meta_calls == ["2034_1_349"]
    assert fake.html_calls == ["2034_1_349"]


def test_fetch_command_uses_explicit_case_id(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake = _FakeCrawler()
    raw_dir = tmp_path / "raw"
    (raw_dir / "meta").mkdir(parents=True)
    (raw_dir / "html").mkdir(parents=True)

    monkeypatch.setattr(nwlr_crawl, "NWLRCrawler", lambda *args, **kwargs: fake)
    monkeypatch.setattr(nwlr_crawl.settings, "nwlr_email", "user@example.com")
    monkeypatch.setattr(nwlr_crawl.settings, "nwlr_password", "secret")
    monkeypatch.setattr(nwlr_crawl, "_RAW_DIR", raw_dir)
    monkeypatch.setattr(nwlr_crawl, "_MANIFEST", tmp_path / "nwlr_manifest.json")

    result = runner.invoke(nwlr_crawl.cli, ["fetch", "--case-id", "2034_1_349"])

    assert result.exit_code == 0
    assert fake.meta_calls == ["2034_1_349"]
    assert fake.html_calls == ["2034_1_349"]


def test_fetch_command_filters_manifest_by_page_range(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    fake = _FakeCrawler()
    raw_dir = tmp_path / "raw"
    (raw_dir / "meta").mkdir(parents=True)
    (raw_dir / "html").mkdir(parents=True)
    manifest_path = tmp_path / "nwlr_manifest.json"
    manifest_path.write_text(
        '{"2034": ["2034_1_209", "2034_1_235", "2034_1_349"]}',
        encoding="utf-8",
    )

    monkeypatch.setattr(nwlr_crawl, "NWLRCrawler", lambda *args, **kwargs: fake)
    monkeypatch.setattr(nwlr_crawl.settings, "nwlr_email", "user@example.com")
    monkeypatch.setattr(nwlr_crawl.settings, "nwlr_password", "secret")
    monkeypatch.setattr(nwlr_crawl, "_RAW_DIR", raw_dir)
    monkeypatch.setattr(nwlr_crawl, "_MANIFEST", manifest_path)

    result = runner.invoke(
        nwlr_crawl.cli,
        ["fetch", "--part", "2034", "--page-from", "230", "--page-to", "300"],
    )

    assert result.exit_code == 0
    assert fake.meta_calls == ["2034_1_235"]
    assert fake.html_calls == ["2034_1_235"]


class _FailingCrawler(_FakeCrawler):
    async def fetch_case_metadata(self, case_id):
        raise RuntimeError(f"Boom for {case_id.as_str()}")


def test_fetch_command_writes_failure_log(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    (raw_dir / "meta").mkdir(parents=True)
    (raw_dir / "html").mkdir(parents=True)
    failures_log = raw_dir / "failures.jsonl"

    monkeypatch.setattr(nwlr_crawl, "NWLRCrawler", lambda *args, **kwargs: _FailingCrawler())
    monkeypatch.setattr(nwlr_crawl.settings, "nwlr_email", "user@example.com")
    monkeypatch.setattr(nwlr_crawl.settings, "nwlr_password", "secret")
    monkeypatch.setattr(nwlr_crawl, "_RAW_DIR", raw_dir)
    monkeypatch.setattr(nwlr_crawl, "_MANIFEST", tmp_path / "nwlr_manifest.json")
    monkeypatch.setattr(nwlr_crawl, "_FAILURES_LOG", failures_log)

    result = runner.invoke(nwlr_crawl.cli, ["fetch", "--case-id", "2034_1_349"])

    assert result.exit_code == 0
    lines = failures_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"stage": "fetch_exception"' in lines[0]
    assert '"identifier": "2034_1_349"' in lines[0]
