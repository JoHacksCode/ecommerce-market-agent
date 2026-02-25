"""
Web Scraper Tool
Collects product prices and availability from mock e-commerce platforms.
In production, replace MOCK_DB with real HTTP scraping / SerpAPI calls.
"""

from typing import Any

from market_agent.tools.base_tool import BaseTool, ToolResult

MOCK_DB: dict[str, list[dict[str, Any]]] = {
    "iphone 15": [
        {
            "platform": "Amazon",
            "price": 999.99,
            "availability": "In Stock",
            "rating": 4.7,
            "reviews_count": 15_243,
        },
        {
            "platform": "eBay",
            "price": 879.00,
            "availability": "In Stock",
            "rating": 4.5,
            "reviews_count": 3_891,
        },
        {
            "platform": "BestBuy",
            "price": 1049.99,
            "availability": "In Stock",
            "rating": 4.8,
            "reviews_count": 8_721,
        },
        {
            "platform": "Walmart",
            "price": 949.00,
            "availability": "Limited",
            "rating": 4.6,
            "reviews_count": 5_234,
        },
    ],
    "nike air max": [
        {
            "platform": "Amazon",
            "price": 120.00,
            "availability": "In Stock",
            "rating": 4.4,
            "reviews_count": 9_832,
        },
        {
            "platform": "eBay",
            "price": 89.99,
            "availability": "In Stock",
            "rating": 4.2,
            "reviews_count": 4_521,
        },
        {
            "platform": "Nike.com",
            "price": 150.00,
            "availability": "In Stock",
            "rating": 4.7,
            "reviews_count": 21_043,
        },
        {
            "platform": "Foot Locker",
            "price": 145.00,
            "availability": "In Stock",
            "rating": 4.5,
            "reviews_count": 7_654,
        },
    ],
    "macbook pro": [
        {
            "platform": "Amazon",
            "price": 1999.99,
            "availability": "In Stock",
            "rating": 4.8,
            "reviews_count": 8_932,
        },
        {
            "platform": "Apple Store",
            "price": 1999.00,
            "availability": "In Stock",
            "rating": 4.9,
            "reviews_count": 34_521,
        },
        {
            "platform": "BestBuy",
            "price": 2099.99,
            "availability": "In Stock",
            "rating": 4.7,
            "reviews_count": 6_234,
        },
    ],
}

_DEFAULT_LISTINGS: list[dict[str, Any]] = [
    {
        "platform": "Amazon",
        "price": 199.99,
        "availability": "In Stock",
        "rating": 4.2,
        "reviews_count": 1_500,
    },
    {
        "platform": "eBay",
        "price": 174.99,
        "availability": "In Stock",
        "rating": 4.0,
        "reviews_count": 800,
    },
    {
        "platform": "Walmart",
        "price": 189.99,
        "availability": "Limited",
        "rating": 4.1,
        "reviews_count": 1_100,
    },
]


class WebScraperTool(BaseTool):
    """Scrapes product listings from multiple e-commerce platforms."""

    name = "web_scraper"
    description = (
        "Fetches product prices and availability from e-commerce platforms. "
        "Input: product_name (str). "
        "Output: platform listings with price, availability, rating, and review count."
    )

    def run(self, product_name: str) -> ToolResult:  # type: ignore[override]
        listings = MOCK_DB.get(product_name.lower().strip(), _DEFAULT_LISTINGS)
        prices = [item["price"] for item in listings]
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "product": product_name,
                "platforms_scraped": len(listings),
                "listings": listings,
                "price_summary": {
                    "min": min(prices),
                    "max": max(prices),
                    "avg": round(sum(prices) / len(prices), 2),
                },
            },
        )
