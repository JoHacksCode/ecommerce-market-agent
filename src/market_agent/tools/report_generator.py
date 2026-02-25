"""
Report Generator Tool
Compiles data from all three analysis tools into a structured, actionable
report with embedded Plotly visualizations (HTML output).
"""

import logging
from datetime import datetime
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from market_agent.tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ReportGeneratorTool(BaseTool):
    """Compiles pricing, sentiment, and trend data into a full market report
    with a self-contained HTML file containing interactive charts."""

    name = "report_generator"
    description = (
        "Compiles scraper_data, sentiment_data, and trend_data into a structured "
        "market report with executive summary, business recommendations, and "
        "interactive visualizations (price comparison, sentiment breakdown, "
        "6-month price history). "
        "Inputs: product_name, scraper_data, sentiment_data, trend_data (all dicts)."
    )

    def run(  # type: ignore[override]
        self,
        product_name: str,
        scraper_data: dict[str, Any],
        sentiment_data: dict[str, Any],
        trend_data: dict[str, Any],
    ) -> ToolResult:
        logger.debug("ReportGeneratorTool.run called for product='%s'", product_name)
        report = self._build_report(product_name, scraper_data, sentiment_data, trend_data)
        report_html = self._generate_visualizations(report)
        logger.info("ReportGeneratorTool completed for product='%s'", product_name)
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=report,
            report_html=report_html,
        )

    # ── Report structure ───────────────────────────────────────────────────────

    def _build_report(
        self,
        product_name: str,
        scraper_data: dict[str, Any],
        sentiment_data: dict[str, Any],
        trend_data: dict[str, Any],
    ) -> dict[str, Any]:
        price_summary = scraper_data.get("price_summary", {})
        sentiment_bd = sentiment_data.get("sentiment_breakdown", {})
        satisfaction = sentiment_data.get("overall_satisfaction_score", 0.0)
        yoy_growth = trend_data.get("yoy_growth", 0.0)
        direction = trend_data.get("trend_direction", "unknown")

        if "price_summary" not in scraper_data:
            logger.warning("scraper_data missing 'price_summary' for product '%s'", product_name)
        if "sentiment_breakdown" not in sentiment_data:
            logger.warning("sentiment_data missing 'sentiment_breakdown' for product '%s'", product_name)

        return {
            "report_id": (
                f"report_{product_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ),
            "product": product_name,
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {
                "market_position": _assess_market_position(yoy_growth, satisfaction),
                "price_competitiveness": _assess_price_competitiveness(price_summary),
                "customer_sentiment": _summarize_sentiment(sentiment_bd, satisfaction),
                "market_momentum": direction.replace("_", " ").title(),
            },
            "pricing_analysis": {
                "price_range": (f"${price_summary.get('min', 0):.2f} – ${price_summary.get('max', 0):.2f}"),
                "average_price": f"${price_summary.get('avg', 0):.2f}",
                "platforms_analyzed": scraper_data.get("platforms_scraped", 0),
                "listings": scraper_data.get("listings", []),
            },
            "sentiment_analysis": sentiment_data,
            "market_trends": trend_data,
            "recommendations": _build_recommendations(price_summary, sentiment_bd, yoy_growth, direction),
            "metadata": {
                "data_sources": ["mock_scraper", "mock_reviews", "mock_market_db"],
                "confidence_score": 0.85,
            },
        }

    # ── Visualizations ─────────────────────────────────────────────────────────

    def _generate_visualizations(self, report: dict[str, Any]) -> str:
        """Build a self-contained HTML page with three interactive Plotly charts."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Price by Platform ($)",
                "Customer Sentiment Breakdown",
                "6-Month Price History ($)",
                "",
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "table"}],
            ],
        )

        # 1 — Price per platform (bar chart)
        listings = report["pricing_analysis"]["listings"]
        if listings:
            fig.add_trace(
                go.Bar(
                    x=[listing["platform"] for listing in listings],
                    y=[listing["price"] for listing in listings],
                    name="Price ($)",
                    text=[f"${p:.2f}" for p in [listing["price"] for listing in listings]],
                    textposition="outside",
                ),
                row=1,
                col=1,
            )

        # 2 — Sentiment breakdown (pie chart)
        bd = report["sentiment_analysis"].get("sentiment_breakdown", {})
        if bd:
            fig.add_trace(
                go.Pie(
                    labels=["Positive", "Neutral", "Negative"],
                    values=[
                        bd.get("positive_pct", 0),
                        bd.get("neutral_pct", 0),
                        bd.get("negative_pct", 0),
                    ],
                    name="Sentiment",
                    hole=0.35,
                    marker_colors=["#2ecc71", "#f39c12", "#e74c3c"],
                ),
                row=1,
                col=2,
            )

        # 3 — 6-month price history (line chart)
        history = report["market_trends"].get("monthly_price_history", [])
        if history:
            months = [f"Month {i}" for i in range(1, len(history) + 1)]
            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=history,
                    mode="lines+markers",
                    name="Avg Price ($)",
                    line={"color": "#3498db", "width": 2},
                    marker={"size": 8},
                ),
                row=2,
                col=1,
            )

        # 4 — Recommendations table (bottom-right)
        recs = report.get("recommendations", [])
        if recs:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["<b>Recommendations</b>"],
                        fill_color="#2c3e50",
                        font=dict(color="white", size=13),
                    ),
                    cells=dict(
                        values=[recs],
                        fill_color="#ecf0f1",
                        align="left",
                        font=dict(size=11),
                        height=30,
                    ),
                ),
                row=2,
                col=2,
            )

        product = report["product"]
        generated = report["generated_at"][:19].replace("T", " ")
        exec_summ = report["executive_summary"]

        fig.update_layout(
            title={
                "text": (
                    f"<b>Market Analysis Report — {product}</b>"
                    f"<br><sup>Generated: {generated} | "
                    f"Position: {exec_summ['market_position']}</sup>"
                ),
                "x": 0.5,
                "xanchor": "center",
            },
            height=900,
            showlegend=False,
            paper_bgcolor="#f8f9fa",
            plot_bgcolor="#ffffff",
        )

        logger.debug("Visualizations generated for product='%s'", product)
        return fig.to_html(full_html=True, include_plotlyjs="cdn")


# ── Pure helper functions (easily unit-tested without instantiating the tool) ──


def _assess_market_position(yoy_growth: float, satisfaction_score: float) -> str:
    if yoy_growth > 7 and satisfaction_score > 7:
        return "Strong market performer with high customer loyalty"
    if yoy_growth > 3:
        return "Moderate growth with stable market presence"
    return "Mature product in a competitive, low-growth market"


def _assess_price_competitiveness(price_summary: dict[str, Any]) -> str:
    avg = price_summary.get("avg", 1) or 1
    spread_pct = (price_summary.get("max", 0) - price_summary.get("min", 0)) / avg * 100
    if spread_pct > 20:
        return "High price variance — significant arbitrage opportunity"
    if spread_pct > 10:
        return "Moderate price variance — some optimisation possible"
    return "Low price variance — market is well-aligned across platforms"


def _summarize_sentiment(breakdown: dict[str, Any], score: float) -> str:
    pos = breakdown.get("positive_pct", 0)
    tag = f"{pos:.0f}% positive, score {score}/10"
    if pos >= 60:
        return f"Positive customer perception ({tag})"
    if pos >= 40:
        return f"Mixed customer reception ({tag})"
    return f"Negative customer perception — requires immediate attention ({tag})"


def _build_recommendations(
    price_summary: dict[str, Any],
    breakdown: dict[str, Any],
    yoy_growth: float,
    direction: str,
) -> list[str]:
    avg = price_summary.get("avg", 1) or 1
    recs: list[str] = []

    if (price_summary.get("max", 0) - price_summary.get("min", 0)) / avg * 100 > 15:
        recs.append("💰 Source from lowest-price platforms to maximise margin.")
    pos = breakdown.get("positive_pct", 0)
    if pos < 50:
        recs.append("⚠️  Resolve top pain-points — negative sentiment risks churn.")
    elif pos > 70:
        recs.append("✅ Leverage positive reviews in marketing and social proof.")
    if yoy_growth > 5:
        recs.append("📈 Growing market — increase inventory ahead of demand peaks.")
    elif yoy_growth < 2:
        recs.append("📉 Slow growth — differentiate via bundling or niche positioning.")
    if "increase" in direction:
        recs.append("🔼 Prices rising — lock in supplier contracts at current rates.")
    elif "decrease" in direction:
        recs.append("🔽 Prices declining — delay bulk purchasing if supply allows.")
    recs.append("🔍 Review competitor pricing weekly to maintain edge.")
    return recs
