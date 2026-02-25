"""Unit tests for the LangGraph agent orchestration."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from market_agent.agent.graph import run_analysis


@pytest.fixture
def mock_report_payload() -> dict:
    return {
        "report_id": "report_iphone_15_20260220_170000",
        "product": "iPhone 15",
        "executive_summary": {"market_position": "Strong"},
        "recommendations": ["Buy more stock"],
    }


async def test_run_analysis_returns_report(mock_report_payload):
    with patch("market_agent.agent.graph.build_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [],
            "product_name": "iPhone 15",
            "final_report": mock_report_payload,
            "report_html": "<html>...</html>",
        }
        mock_build.return_value = mock_graph

        response = await run_analysis("iPhone 15")
        report = response.get("final_report")
        html = response.get("report_html")
    assert report["product"] == "iPhone 15"
    assert html == "<html>...</html>"


async def test_run_analysis_fallback_to_text():
    with patch("market_agent.agent.graph.build_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="Here is the analysis text.")],
            "product_name": "test product",
            "final_report": None,
            "report_html": None,
        }
        mock_build.return_value = mock_graph

        response = await run_analysis("test product")

    assert "error" in response
    assert "last_raw_response" in response
    assert "report_html" not in response
    assert "final_report"
