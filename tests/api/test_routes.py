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


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze_post_success(client):
    with patch("market_agent.api.routes.run_analysis", new=AsyncMock(return_value=MOCK_REPORT)):
        response = client.post("/api/v1/analyze", json={"product_name": "iPhone 15"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["product"] == "iPhone 15"
    assert body["report"]["report_id"] == "report_iphone_15_test"


def test_analyze_get_success(client):
    with patch("market_agent.api.routes.run_analysis", new=AsyncMock(return_value=MOCK_REPORT)):
        response = client.get("/api/v1/analyze/iPhone%2015")
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_analyze_empty_product_name(client):
    response = client.post("/api/v1/analyze", json={"product_name": "x"})
    # min_length is 2, "x" is 1 char → validation error
    assert response.status_code == 422


def test_analyze_agent_exception_returns_500(client):
    with patch(
        "market_agent.api.routes.run_analysis",
        new=AsyncMock(side_effect=RuntimeError("LLM timeout")),
    ):
        response = client.post("/api/v1/analyze", json={"product_name": "Some Product"})
    assert response.status_code == 500
