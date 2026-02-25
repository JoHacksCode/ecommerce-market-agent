"""Unit tests for WebScraperTool."""

import pytest
from market_agent.tools.web_scraper import WebScraperTool


@pytest.fixture
def scraper() -> WebScraperTool:
    return WebScraperTool()


def test_known_product_returns_listings(scraper):
    result = scraper.run(product_name="iphone 15")
    assert result.success is True
    assert result.tool_name == "web_scraper"
    assert len(result.data["listings"]) > 0


def test_price_summary_keys(scraper):
    result = scraper.run(product_name="iphone 15")
    ps = result.data["price_summary"]
    assert "min" in ps and "max" in ps and "avg" in ps


def test_min_lte_avg_lte_max(scraper):
    result = scraper.run(product_name="nike air max")
    ps = result.data["price_summary"]
    assert ps["min"] <= ps["avg"] <= ps["max"]


def test_unknown_product_uses_default(scraper):
    result = scraper.run(product_name="unknown gadget xyz")
    assert result.success is True
    assert result.data["platforms_scraped"] >= 1


def test_safe_run_never_raises(scraper, monkeypatch):
    monkeypatch.setattr(scraper, "run", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
    result = scraper.safe_run(product_name="anything")
    assert result.success is False
    assert "boom" in result.error
