"""Pydantic request/response models for the REST API."""

from typing import Any
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    product_name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        examples=["iPhone 15", "Nike Air Max", "MacBook Pro"],
        description="The product or market to analyze.",
    )


class AnalysisResponse(BaseModel):
    success: bool
    product: str
    report: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
