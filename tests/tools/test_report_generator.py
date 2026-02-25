import pytest
from market_agent.tools.report_generator import (
    ReportGeneratorTool,
    _assess_market_position,
    _assess_price_competitiveness,
    _build_recommendations,
    _summarize_sentiment,
)


@pytest.fixture
def reporter() -> ReportGeneratorTool:
    return ReportGeneratorTool()


@pytest.fixture
def sample_scraper_data() -> dict:
    return {
        "product": "iPhone 15",
        "platforms_scraped": 4,
        "listings": [
            {
                "platform": "Amazon",
                "price": 999.99,
                "availability": "In Stock",
                "rating": 4.7,
                "reviews_count": 15000,
            },
        ],
        "price_summary": {"min": 879.0, "max": 1049.99, "avg": 994.74},
    }


@pytest.fixture
def sample_sentiment_data() -> dict:
    return {
        "product": "iPhone 15",
        "total_reviews_analyzed": 8,
        "sentiment_breakdown": {
            "positive_pct": 62.5,
            "negative_pct": 25.0,
            "neutral_pct": 12.5,
        },
        "overall_satisfaction_score": 7.5,
        "top_positive_reviews": ["Great camera!"],
        "top_negative_reviews": ["Battery is weak."],
    }


@pytest.fixture
def sample_trend_data() -> dict:
    return {
        "product": "iPhone 15",
        "price_trend": "stable",
        "trend_direction": "slight_decrease",
        "monthly_price_history": [1099, 1079, 1049, 1029, 999, 989],
        "search_volume_trend": "increasing",
        "market_share_pct": 28.5,
        "competitors": ["Samsung Galaxy S24"],
        "seasonal_demand": "high_q4",
        "yoy_growth": 5.2,
        "price_change_6m_pct": -10.01,
    }


@pytest.fixture
def full_result(reporter, sample_scraper_data, sample_sentiment_data, sample_trend_data):
    return reporter.run(
        product_name="iPhone 15",
        scraper_data=sample_scraper_data,
        sentiment_data=sample_sentiment_data,
        trend_data=sample_trend_data,
    )


def test_report_has_required_keys(full_result):
    for key in (
        "report_id",
        "executive_summary",
        "pricing_analysis",
        "sentiment_analysis",
        "market_trends",
        "recommendations",
    ):
        assert key in full_result.data, f"Missing key: {key}"


def test_recommendations_is_non_empty_list(full_result):
    assert isinstance(full_result.data["recommendations"], list)
    assert len(full_result.data["recommendations"]) >= 1


def test_report_id_contains_product(full_result):
    assert "iphone_15" in full_result.data["report_id"]


def test_success_flag(full_result):
    assert full_result.success is True


# ── Visualizations ─────────────────────────────────────────────────────────────


def test_report_html_is_present(full_result):
    assert full_result.report_html is not None


def test_report_html_is_valid_html(full_result):
    assert "<html>" in full_result.report_html
    assert "</html>" in full_result.report_html


def test_report_html_contains_plotly(full_result):
    # CDN script tag injected by fig.to_html(include_plotlyjs="cdn")
    assert "plotly" in full_result.report_html.lower()


def test_report_html_contains_product_name(full_result):
    assert "iPhone 15" in full_result.report_html


def test_report_html_contains_all_chart_titles(full_result):
    html = full_result.report_html
    assert "Price by Platform" in html
    assert "Customer Sentiment" in html
    assert "6-Month Price History" in html


# ── Missing data robustness ────────────────────────────────────────────────────


def test_report_with_empty_scraper_data(reporter, sample_sentiment_data, sample_trend_data):
    """Tool must not raise when scraper_data is incomplete."""
    result = reporter.run(
        product_name="Test Product",
        scraper_data={},
        sentiment_data=sample_sentiment_data,
        trend_data=sample_trend_data,
    )
    assert result.success is True
    assert result.report_html is not None


def test_report_with_empty_sentiment_data(reporter, sample_scraper_data, sample_trend_data):
    result = reporter.run(
        product_name="Test Product",
        scraper_data=sample_scraper_data,
        sentiment_data={},
        trend_data=sample_trend_data,
    )
    assert result.success is True


def test_report_with_empty_trend_data(reporter, sample_scraper_data, sample_sentiment_data):
    result = reporter.run(
        product_name="Test Product",
        scraper_data=sample_scraper_data,
        sentiment_data=sample_sentiment_data,
        trend_data={},
    )
    assert result.success is True
    assert result.report_html is not None


# ── Pure helper functions ──────────────────────────────────────────────────────


class TestAssessMarketPosition:
    def test_strong_performer(self):
        assert "Strong" in _assess_market_position(yoy_growth=8.0, satisfaction_score=8.0)

    def test_moderate_growth(self):
        assert "Moderate" in _assess_market_position(yoy_growth=5.0, satisfaction_score=6.0)

    def test_mature_market(self):
        assert "Mature" in _assess_market_position(yoy_growth=1.0, satisfaction_score=5.0)


class TestAssessPriceCompetitiveness:
    def test_high_variance(self):
        ps = {"min": 80.0, "max": 150.0, "avg": 100.0}  # 70% spread
        assert "High" in _assess_price_competitiveness(ps)

    def test_moderate_variance(self):
        ps = {"min": 90.0, "max": 110.0, "avg": 100.0}  # 20% spread
        assert "Moderate" in _assess_price_competitiveness(ps)

    def test_low_variance(self):
        ps = {"min": 98.0, "max": 102.0, "avg": 100.0}  # 4% spread
        assert "Low" in _assess_price_competitiveness(ps)

    def test_empty_price_summary_does_not_raise(self):
        result = _assess_price_competitiveness({})
        assert isinstance(result, str)


class TestSummarizeSentiment:
    def test_positive(self):
        bd = {"positive_pct": 70.0, "neutral_pct": 20.0, "negative_pct": 10.0}
        assert "Positive" in _summarize_sentiment(bd, score=8.0)

    def test_mixed(self):
        bd = {"positive_pct": 45.0, "neutral_pct": 30.0, "negative_pct": 25.0}
        assert "Mixed" in _summarize_sentiment(bd, score=6.0)

    def test_negative(self):
        bd = {"positive_pct": 20.0, "neutral_pct": 30.0, "negative_pct": 50.0}
        assert "Negative" in _summarize_sentiment(bd, score=3.0)


class TestBuildRecommendations:
    def test_high_price_variance_recommendation(self):
        ps = {"min": 50.0, "max": 150.0, "avg": 100.0}
        recs = _build_recommendations(ps, {}, yoy_growth=3.0, direction="flat")
        assert any("Source" in r for r in recs)

    def test_negative_sentiment_recommendation(self):
        bd = {"positive_pct": 30.0}
        recs = _build_recommendations({"avg": 100, "min": 95, "max": 105}, bd, 3.0, "flat")
        assert any("pain-points" in r for r in recs)

    def test_growing_market_recommendation(self):
        recs = _build_recommendations({"avg": 100, "min": 95, "max": 105}, {"positive_pct": 65}, 6.0, "flat")
        assert any("inventory" in r for r in recs)

    def test_always_has_at_least_one_recommendation(self):
        recs = _build_recommendations({}, {}, yoy_growth=3.0, direction="flat")
        assert len(recs) >= 1
