FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml .
COPY README.md .

# Install production dependencies only
RUN uv sync --no-dev --frozen

# Copy source
COPY src/ src/

# Expose API port
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "market_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
