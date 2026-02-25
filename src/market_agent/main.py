"""FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from market_agent.api.routes import router
from market_agent.config import settings

logging.basicConfig(level=logging.DEBUG if settings.debug else logging.INFO)

app = FastAPI(
    title="Market Analysis Agent",
    description="E-commerce market intelligence powered by LangGraph + LangChain",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["Meta"])
async def health_check() -> dict:
    """Health probe used by Docker and load balancers."""
    return {"status": "healthy", "version": "0.1.0"}
