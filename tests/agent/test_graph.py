"""Unit tests for the LangGraph agent orchestration."""

import json
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


@pytest.mark.asyncio
async def test_run_analysis_returns_report(mock_report_payload):
    """Agent should return the final_report when report_generator succeeds."""
    json.dumps(
        {
            "success": True,
            "data": mock_report_payload,
            "tool_name": "report_generator",
            "error": None,
        }
    )

    ai_with_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "report_generator",
                "args": {
                    "product_name": "iPhone 15",
                    "scraper_data": {},
                    "sentiment_data": {},
                    "trend_data": {},
                },
                "id": "call_001",
            }
        ],
    )
    ai_final = AIMessage(content="Done.")

    with patch("market_agent.agent.graph.build_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [ai_with_tool_call, ai_final],
            "product_name": "iPhone 15",
            "final_report": mock_report_payload,
        }
        mock_build.return_value = mock_graph

        result = await run_analysis("iPhone 15")

    assert result["product"] == "iPhone 15"
    assert "report_id" in result


@pytest.mark.asyncio
async def test_run_analysis_fallback_to_text():
    """If final_report is None, agent falls back to last AIMessage content."""
    with patch("market_agent.agent.graph.build_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="Here is the analysis text.")],
            "product_name": "test product",
            "final_report": None,
        }
        mock_build.return_value = mock_graph

        result = await run_analysis("test product")

    assert "raw_response" in result
