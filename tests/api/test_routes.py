"""Integration tests for the FastAPI REST layer."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from market_agent.main import app


MOCK_REPORT = {
    "report_id": "report_iphone_15_test",
    "product": "iPhone 15",
    "executive_summary": {"market_position": "Strong"},
    "recommendations": ["Increase inventory"],
}
MOCK_HTML = "<html><body><p>iPhone 15 Report</p></body></html>"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ── Default JSON route (/analyze) ──────────────────────────────────────────────


def test_analyze_json_success(client):
    with patch(
        "market_agent.api.routes.run_analysis",
        new=AsyncMock(return_value={"final_report": MOCK_REPORT, "report_html": None}),
    ):
        response = client.post("/api/v1/analyze", json={"product_name": "iPhone 15"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["product"] == "iPhone 15"
    assert body["report"]["report_id"] == "report_iphone_15_test"


def test_analyze_json_ignores_html(client):
    """The default route must not leak report_html into its response."""
    with patch(
        "market_agent.api.routes.run_analysis",
        new=AsyncMock(return_value={"final_report": MOCK_REPORT, "report_html": MOCK_HTML}),
    ):
        response = client.post("/api/v1/analyze", json={"product_name": "iPhone 15"})
    assert "report_html" not in response.json()


def test_analyze_json_agent_error_returns_500(client):
    with patch("market_agent.api.routes.run_analysis", new=AsyncMock(side_effect=RuntimeError("LLM timeout"))):
        response = client.post("/api/v1/analyze", json={"product_name": "Some Product"})
    assert response.status_code == 500


def test_analyze_json_short_product_name_rejected(client):
    response = client.post("/api/v1/analyze", json={"product_name": "x"})
    assert response.status_code == 422


# ── HTML route (/analyze/html) ─────────────────────────────────────────────────


def test_analyze_html_success(client):
    with patch(
        "market_agent.api.routes.run_analysis",
        new=AsyncMock(return_value={"final_report": MOCK_REPORT, "report_html": MOCK_HTML}),
    ):
        response = client.post("/api/v1/analyze/html", json={"product_name": "iPhone 15"})
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "iPhone 15 Report" in response.text


def test_analyze_html_contains_no_json_wrapper(client):
    """HTML route must return raw HTML, not a JSON envelope."""
    with patch(
        "market_agent.api.routes.run_analysis",
        new=AsyncMock(return_value={"final_report": MOCK_REPORT, "report_html": MOCK_HTML}),
    ):
        response = client.post("/api/v1/analyze/html", json={"product_name": "iPhone 15"})
    assert response.text.strip().startswith("<html>")


def test_analyze_html_none_returns_500(client):
    """If the agent produces no HTML, the route must return 500."""
    with patch("market_agent.api.routes.run_analysis", new=AsyncMock(return_value=(MOCK_REPORT, None))):
        response = client.post("/api/v1/analyze/html", json={"product_name": "iPhone 15"})
    assert response.status_code == 500


def test_analyze_html_agent_error_returns_500(client):
    with patch("market_agent.api.routes.run_analysis", new=AsyncMock(side_effect=RuntimeError("boom"))):
        response = client.post("/api/v1/analyze/html", json={"product_name": "Some Product"})
    assert response.status_code == 500


def test_analyze_html_short_product_name_rejected(client):
    response = client.post("/api/v1/analyze/html", json={"product_name": "x"})
    assert response.status_code == 422
