"""
Sentiment Analyzer Tool
Analyzes customer reviews and extracts sentiment insights.
Production variant: replace heuristics with an LLM-powered classifier.
"""

from market_agent.tools.base_tool import BaseTool, ToolResult

MOCK_REVIEWS: dict[str, list[str]] = {
    "iphone 15": [
        "Absolutely love the camera quality — best iPhone yet!",
        "Battery life is disappointing compared to the competition.",
        "The titanium design feels premium and durable.",
        "Too expensive for such minor upgrades from iPhone 14.",
        "Dynamic Island is genuinely useful now with more app support.",
        "USB-C is finally here — massive improvement over Lightning.",
        "Display is gorgeous but no ProMotion upgrade is a miss.",
        "Performance is blazing fast, zero complaints.",
    ],
    "nike air max": [
        "Super comfortable for all-day wear.",
        "Runs a bit small — order half a size up.",
        "The colorway is stunning, gets compliments everywhere.",
        "Sole started peeling after 3 months — quality control issue.",
        "Great cushioning for running and casual use.",
        "Worth every penny, this is my third pair!",
        "Not ideal for wide feet.",
    ],
    "macbook pro": [
        "M3 chip is absolutely incredible — handles everything.",
        "Keyboard is much improved over the butterfly days.",
        "Battery easily lasts a full workday.",
        "Gets noticeably warm during heavy video editing.",
        "Best laptop I have ever owned.",
        "Price is high, but you get what you pay for.",
        "Thunderbolt port selection is very convenient.",
    ],
}

_GENERIC_REVIEWS = [
    "Great product, highly recommend!",
    "Decent quality for the price.",
    "Shipping was fast and product was as described.",
    "Had some issues but customer service resolved them quickly.",
    "Would buy again without hesitation.",
]

_POSITIVE = {
    "love",
    "great",
    "best",
    "amazing",
    "excellent",
    "fantastic",
    "perfect",
    "incredible",
    "gorgeous",
    "worth",
    "comfortable",
    "improved",
    "useful",
}
_NEGATIVE = {
    "disappointing",
    "issue",
    "problem",
    "bad",
    "worst",
    "poor",
    "expensive",
    "peeling",
    "warm",
    "miss",
}


def _classify(text: str) -> str:
    words = set(text.lower().split())
    pos = len(words & _POSITIVE)
    neg = len(words & _NEGATIVE)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


class SentimentAnalyzerTool(BaseTool):
    """Analyzes customer reviews and returns a sentiment breakdown with key quotes."""

    name = "sentiment_analyzer"
    description = (
        "Analyzes customer reviews for a product and returns sentiment breakdown, "
        "satisfaction score, and representative quotes. "
        "Input: product_name (str)."
    )

    def run(self, product_name: str) -> ToolResult:  # type: ignore[override]
        reviews = MOCK_REVIEWS.get(product_name.lower().strip(), _GENERIC_REVIEWS)
        labels = [_classify(r) for r in reviews]
        total = len(labels)
        counts = {s: labels.count(s) for s in ("positive", "negative", "neutral")}
        score = round((counts["positive"] + counts["neutral"] * 0.5) / total * 10, 1)

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "product": product_name,
                "total_reviews_analyzed": total,
                "sentiment_breakdown": {
                    "positive_pct": round(counts["positive"] / total * 100, 1),
                    "negative_pct": round(counts["negative"] / total * 100, 1),
                    "neutral_pct": round(counts["neutral"] / total * 100, 1),
                },
                "overall_satisfaction_score": score,
                "top_positive_reviews": [review for review, label in zip(reviews, labels) if label == "positive"][:3],
                "top_negative_reviews": [review for review, label in zip(reviews, labels) if label == "negative"][:2],
            },
        )
