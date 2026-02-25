"""API route definitions."""

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse

from market_agent.agent.graph import AgentError, run_analysis
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
        result = await run_analysis(request.product_name)
        return AnalysisResponse(
            success=True,
            product=request.product_name,
            report=result["final_report"],
        )
    except AgentError as exc:
        logger.error("AgentError for '%s': %s", request.product_name, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"message": str(exc), "tool_errors": exc.tool_errors},
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error for product: %s", request.product_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


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
        result = await run_analysis(request.product_name)
        if not result.get("report_html"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Report generator did not produce an HTML output.",
            )
        return HTMLResponse(content=result["report_html"])
    except HTTPException:
        raise
    except AgentError as exc:
        logger.error("AgentError for '%s': %s", request.product_name, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"message": str(exc), "tool_errors": exc.tool_errors},
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error for product: %s", request.product_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
