"""API route definitions."""

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse

from market_agent.agent.graph import run_analysis
from market_agent.api.models import AnalysisRequest, AnalysisResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a market analysis and return a structured JSON report",
)
async def analyze_product_json(request: AnalysisRequest) -> AnalysisResponse:
    """Returns the full structured report as JSON (default)."""
    logger.info("JSON analysis requested for: %s", request.product_name)
    try:
        analysis = await run_analysis(request.product_name)
        return AnalysisResponse(success=True, product=request.product_name, report=analysis.get("final_report"))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent failed for product: %s", request.product_name)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post(
    "/analyze/html",
    response_class=HTMLResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a market analysis and return an interactive HTML report",
)
async def analyze_product_html(request: AnalysisRequest) -> HTMLResponse:
    """Returns a self-contained HTML page with interactive Plotly visualizations."""
    logger.info("HTML analysis requested for: %s", request.product_name)
    try:
        analysis = await run_analysis(request.product_name)
        if report_html := analysis.get("report_html"):
            return HTMLResponse(content=report_html)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report generator did not produce an HTML output.",
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent failed for product: %s", request.product_name)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
