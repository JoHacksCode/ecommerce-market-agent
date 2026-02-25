"""
Market Trend Analyzer Tool
Analyzes price trends, search volume, and competitive landscape.
Production variant: replace mock data with Google Trends API / proprietary data feeds.
"""

from market_agent.tools.base_tool import BaseTool, ToolResult

MOCK_TRENDS: dict[str, dict] = {
    "iphone 15": {
        "price_trend": "stable",
        "trend_direction": "slight_decrease",
        "monthly_price_history": [1099, 1079, 1049, 1029, 999, 989],
        "search_volume_trend": "increasing",
        "market_share_pct": 28.5,
        "competitors": ["Samsung Galaxy S24", "Google Pixel 8", "OnePlus 12"],
        "seasonal_demand": "high_q4",
        "yoy_growth": 5.2,
    },
    "nike air max": {
        "price_trend": "increasing",
        "trend_direction": "moderate_increase",
        "monthly_price_history": [105, 110, 115, 118, 120, 125],
        "search_volume_trend": "stable",
        "market_share_pct": 18.3,
        "competitors": ["Adidas Ultraboost", "New Balance 990", "ASICS Gel-Nimbus"],
        "seasonal_demand": "year_round",
        "yoy_growth": 8.7,
    },
    "macbook pro": {
        "price_trend": "stable",
        "trend_direction": "flat",
        "monthly_price_history": [2099, 2099, 1999, 1999, 1999, 1999],
        "search_volume_trend": "decreasing",
        "market_share_pct": 11.2,
        "competitors": [
            "Dell XPS 15",
            "Lenovo ThinkPad X1",
            "Microsoft Surface Laptop",
        ],
        "seasonal_demand": "high_q1_q4",
        "yoy_growth": 2.1,
    },
}

_DEFAULT_TREND = {
    "price_trend": "stable",
    "trend_direction": "flat",
    "monthly_price_history": [100, 102, 98, 101, 99, 100],
    "search_volume_trend": "stable",
    "market_share_pct": 5.0,
    "competitors": ["Competitor A", "Competitor B", "Competitor C"],
    "seasonal_demand": "year_round",
    "yoy_growth": 3.5,
}


class MarketTrendAnalyzerTool(BaseTool):
    """Analyzes market trends, price history, and competitive landscape."""

    name = "market_trend_analyzer"
    description = (
        "Returns price trend, 6-month price history, market share, competitors, "
        "YoY growth, and search-volume trend. Input: product_name (str)."
    )

    def run(self, product_name: str) -> ToolResult:  # type: ignore[override]
        data = MOCK_TRENDS.get(product_name.lower().strip(), _DEFAULT_TREND).copy()
        data["product"] = product_name
        history = data["monthly_price_history"]
        data["price_change_6m_pct"] = round((history[-1] - history[0]) / history[0] * 100, 2)
        return ToolResult(tool_name=self.name, success=True, data=data)
