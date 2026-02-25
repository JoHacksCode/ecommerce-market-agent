# Ecommerce Market Analysis Agent

![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)
![Python](https://img.shields.io/badge/python-3.13-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)


An e-commerce market intelligence agent built with **LangGraph**, **LangChain**,
**FastAPI**, and **uv**. It orchestrates four specialised tools to produce structured
market reports for any product.

---

## Architecture Decision: LangGraph

LangGraph was chosen over CrewAI or AutoGen because:
- **Graph-based state** makes the tool-calling loop explicit and debuggable.
- **Streaming support** enables real-time progress for long analyses.
- **First-class LangChain integration** means every LangChain-compatible LLM and
  tool works without adapters.
- **Thin abstraction** — the graph is plain Python, easy to extend.

The agent uses a **ReAct loop**: `agent → tools → agent → … → END`.
The LLM decides the order; the system prompt enforces the four-step workflow.

---

## Installation & Usage

### 1. Prerequisites

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup

```bash
git clone <your-repo-url>
cd market-analysis-agent

cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
# Free key at https://openrouter.ai — DeepSeek models cost ~$0.001/request

uv sync          # installs all dependencies into .venv
```

### 3a. Run locally

```bash
uv run uvicorn market_agent.main:app --reload --port 8000
```

Open [**http://localhost:8000/docs**](http://localhost:8000/docs) for the interactive Swagger UI.

```bash
# Quick test with curl
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"product_name": "iPhone 15"}'
```

### 3b. Run with Docker

```bash
docker compose up --build
```

### 3c. Run tests

```bash
uv run pytest           # all tests with coverage
uv run pytest -v        # verbose output
```

---

## Step 4 — Data Architecture

| Data type | Recommended store | Rationale |
|---|---|---|
| Analysis results | **PostgreSQL** (JSONB) | Flexible schema, queryable, battle-tested |
| Request history | PostgreSQL | Relational links to results, audit trail |
| Scraped product cache | **Redis** (TTL 1h) | Sub-millisecond reads, automatic expiry |
| Agent config / prompts | PostgreSQL or YAML files | Version-controlled, easy rollback |

Schema sketch:
```sql
CREATE TABLE analyses (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  product     TEXT NOT NULL,
  report      JSONB NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON analyses(product, created_at DESC);

CREATE TABLE request_log (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id  UUID REFERENCES analyses(id),
  requested_at TIMESTAMPTZ DEFAULT now(),
  duration_ms  INTEGER,
  model_used   TEXT
);
```

---

## Step 5 — Monitoring & Observability

- **Tracing**: LangSmith (native LangChain integration) or OpenTelemetry with Jaeger. Every agent invocation gets a `run_id` propagated through all tool calls.
- **Metrics** (Prometheus + Grafana): `p95_agent_latency`, `tool_error_rate`, `llm_tokens_per_request`, `reports_per_minute`.
- **Alerting**: PagerDuty/Opsgenie rules on `tool_error_rate > 5%` and `p95_latency > 30s`.
- **Output quality**: LLM-as-Judge prompt scores each report on completeness, accuracy, and actionability (0–10). Score stored in `analyses` table and alerted when rolling average drops below 7.

---

## Step 6 — Scaling & Optimisation

- **Horizontal scaling**: Stateless FastAPI workers behind a load balancer (Kubernetes HPA); each worker runs its own LangGraph instance.
- **Parallelism**: Tools 1–3 (scraper, sentiment, trend) are **independent** — in production, fan them out with `asyncio.gather` before calling the report generator, cutting latency by ~60%.
- **LLM cost optimisation**: Cache identical `(product, date)` requests in Redis for 1 hour; use a cheap model (DeepSeek) for tool orchestration, GPT-4o only for final report synthesis.
- **Queue for spikes**: Celery + Redis queue absorbs bursts; 100+ simultaneous requests are processed at a controlled rate without dropping.

---

## Step 7 — Continuous Improvement & A/B Testing

- **LLM-as-Judge**: A secondary prompt evaluates each report on 4 axes (data completeness, recommendation quality, clarity, actionability). Scores are stored and trigger alerts.
- **Prompt A/B testing**: Each prompt variant gets a `variant_id`. Requests are randomly assigned 50/50; Judge scores are compared with a t-test after 100 samples.
- **User feedback loop**: A `/feedback` endpoint accepts thumbs-up/down + optional comment. Negative feedback triggers an automatic re-analysis with a fallback model and flags the prompt variant.
- **Agent evolution**: New tools are added as LangChain `StructuredTool` objects and registered in `graph.py`; the LLM's system prompt is updated to describe the new capability. No graph restructuring required.
