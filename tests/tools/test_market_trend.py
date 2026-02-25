import pytest
from market_agent.tools.market_trend import MarketTrendAnalyzerTool


@pytest.fixture
def trend() -> MarketTrendAnalyzerTool:
    return MarketTrendAnalyzerTool()


def test_known_product_has_history(trend):
    result = trend.run(product_name="iphone 15")
    assert len(result.data["monthly_price_history"]) == 6


def test_price_change_computed(trend):
    result = trend.run(product_name="nike air max")
    assert "price_change_6m_pct" in result.data


def test_competitors_list(trend):
    result = trend.run(product_name="macbook pro")
    assert isinstance(result.data["competitors"], list)
    assert len(result.data["competitors"]) >= 1


def test_unknown_product_default(trend):
    result = trend.run(product_name="mystery item")
    assert result.success is True
    assert result.data["market_share_pct"] == 5.0
