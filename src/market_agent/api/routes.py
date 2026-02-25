"""API route definitions."""

import logging

from fastapi import APIRouter, HTTPException, status

from market_agent.agent.graph import run_analysis
from market_agent.api.models import AnalysisRequest, AnalysisResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a full market analysis for a product",
)
async def analyze_product(request: AnalysisRequest) -> AnalysisResponse:
    """
    Triggers the LangGraph agent to:
    1. Scrape product prices across platforms
    2. Analyze customer sentiment
    3. Assess market trends
    4. Compile and return a structured report
    """
    logger.info("Analysis requested for: %s", request.product_name)
    try:
        report = await run_analysis(request.product_name)
        return AnalysisResponse(success=True, product=request.product_name, report=report)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent failed for product: %s", request.product_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get(
    "/analyze/{product_name}",
    response_model=AnalysisResponse,
    summary="Run analysis via GET (convenience endpoint)",
)
async def analyze_product_get(product_name: str) -> AnalysisResponse:
    """GET variant — useful for quick browser/curl tests."""
    return await analyze_product(AnalysisRequest(product_name=product_name))
