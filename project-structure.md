market-analysis-agent/
├── pyproject.toml
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── README.md
├── src/
│   └── market_agent/
│       ├── __init__.py
│       ├── config.py
│       ├── main.py
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── graph.py
│       │   └── prompts.py
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── web_scraper.py
│       │   ├── sentiment_analyzer.py
│       │   ├── market_trend.py
│       │   └── report_generator.py
│       └── api/
│           ├── __init__.py
│           ├── models.py
│           └── routes.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_tools/
│   │   ├── __init__.py
│   │   ├── test_web_scraper.py
│   │   ├── test_sentiment_analyzer.py
│   │   ├── test_market_trend.py
│   │   └── test_report_generator.py
│   ├── test_agent/
│   │   ├── __init__.py
│   │   └── test_graph.py
│   └── test_api/
│       ├── __init__.py
│       └── test_routes.py
└── examples/
    ├── sample_report.json
    └── api_requests.http
