import pytest
from market_agent.tools.sentiment_analyzer import SentimentAnalyzerTool


@pytest.fixture
def sentiment() -> SentimentAnalyzerTool:
    return SentimentAnalyzerTool()


def test_sentiment_breakdown_sums_to_100(sentiment):
    result = sentiment.run(product_name="iphone 15")
    bd = result.data["sentiment_breakdown"]
    total = bd["positive_pct"] + bd["negative_pct"] + bd["neutral_pct"]
    assert abs(total - 100.0) < 0.5


def test_satisfaction_score_in_range(sentiment):
    result = sentiment.run(product_name="nike air max")
    score = result.data["overall_satisfaction_score"]
    assert 0.0 <= score <= 10.0


def test_reviews_lists_present(sentiment):
    result = sentiment.run(product_name="macbook pro")
    assert "top_positive_reviews" in result.data
    assert "top_negative_reviews" in result.data


def test_unknown_product_falls_back(sentiment):
    result = sentiment.run(product_name="nonexistent product 9999")
    assert result.success is True
    assert result.data["total_reviews_analyzed"] >= 1
