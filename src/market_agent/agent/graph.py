"""
LangGraph ReAct agent graph.
  START → agent_node → (tool_node | END)
"""

import json
import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from market_agent.agent.prompts import SYSTEM_PROMPT
from market_agent.config import settings
from market_agent.tools.market_trend import MarketTrendAnalyzerTool
from market_agent.tools.report_generator import ReportGeneratorTool
from market_agent.tools.sentiment_analyzer import SentimentAnalyzerTool
from market_agent.tools.web_scraper import WebScraperTool

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    product_name: str
    final_report: dict[str, Any] | None
    report_html: str | None


# ── Tool wrappers ──────────────────────────────────────────────────────────────


def _make_lc_tools() -> list[StructuredTool]:
    scraper = WebScraperTool()
    sentiment = SentimentAnalyzerTool()
    trend = MarketTrendAnalyzerTool()
    report = ReportGeneratorTool()

    def web_scraper(product_name: str) -> str:
        """Fetch prices and availability across e-commerce platforms."""
        return scraper.safe_run(product_name=product_name).model_dump_json()

    def sentiment_analyzer(product_name: str) -> str:
        """Analyze customer reviews and return sentiment breakdown."""
        return sentiment.safe_run(product_name=product_name).model_dump_json()

    def market_trend_analyzer(product_name: str) -> str:
        """Analyze price trend, market share, and competitive landscape."""
        return trend.safe_run(product_name=product_name).model_dump_json()

    def report_generator(
        product_name: str,
        scraper_data: dict,
        sentiment_data: dict,
        trend_data: dict,
    ) -> str:
        """Compile all analysis data into a final structured market report."""
        return report.safe_run(
            product_name=product_name,
            scraper_data=scraper_data,
            sentiment_data=sentiment_data,
            trend_data=trend_data,
        ).model_dump_json()

    return [
        StructuredTool.from_function(web_scraper),
        StructuredTool.from_function(sentiment_analyzer),
        StructuredTool.from_function(market_trend_analyzer),
        StructuredTool.from_function(report_generator),
    ]


# ── Graph builder ──────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    lc_tools = _make_lc_tools()
    tool_map = {t.name: t for t in lc_tools}

    llm = ChatOpenAI(
        model=settings.model_name,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0,
    ).bind_tools(lc_tools)

    def agent_node(state: AgentState) -> dict:
        logger.debug("agent_node invoked, messages=%d", len(state["messages"]))
        response: AIMessage = llm.invoke(state["messages"])

        if not response.tool_calls and not response.content:
            logger.warning("LLM returned an empty response with no tool calls")

        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        last: AIMessage = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        update: dict[str, Any] = {}

        for call in last.tool_calls:
            tool_name = call["name"]
            logger.debug("Invoking tool '%s' with args: %s", tool_name, call["args"])

            try:
                result = tool_map[tool_name].invoke(call["args"])
            except KeyError:
                logger.error("Unknown tool requested by LLM: '%s'", tool_name)
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                        tool_call_id=call["id"],
                    )
                )
                continue
            except Exception as exc:  # noqa: BLE001
                logger.exception("Tool '%s' raised an unexpected exception", tool_name)
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps({"error": str(exc)}),
                        tool_call_id=call["id"],
                    )
                )
                continue

            tool_messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

            if tool_name == "report_generator":
                try:
                    payload = json.loads(result)
                except json.JSONDecodeError as exc:
                    logger.error("Failed to parse report_generator output as JSON: %s", exc)
                    continue

                if not payload.get("success"):
                    logger.error(
                        "report_generator returned success=False: %s",
                        payload.get("error"),
                    )
                    continue

                if "data" not in payload:
                    logger.error("report_generator payload missing 'data' key")
                    continue

                update = {
                    "final_report": payload["data"],
                    "report_html": payload.get("report_html"),
                }
                logger.info("report_generator completed successfully")

        return {"messages": tool_messages, **update}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Public runner ──────────────────────────────────────────────────────────────


async def run_analysis(product_name: str) -> dict[str, Any]:
    """
    Runs the ReAct agent and returns:
      {
        "final_report": dict | None,
        "report_html":  str  | None,
      }
    Raises AgentError on unrecoverable failures so the API layer
    can return a meaningful 500 instead of a silent empty report.
    """
    logger.info("run_analysis started for product='%s'", product_name)
    graph = build_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=(f"{SYSTEM_PROMPT}\n\nPerform a full market analysis for: {product_name}"))],
        "product_name": product_name,
        "final_report": None,
        "report_html": None,
    }

    final_state = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": settings.agent_recursion_limit},
    )

    final_report = final_state.get("final_report")
    report_html = final_state.get("report_html")

    if not final_report:
        # Collect tool errors from message history to surface in the exception
        tool_errors = [
            msg.content
            for msg in final_state["messages"]
            if isinstance(msg, ToolMessage) and "error" in msg.content.lower()
        ]
        logger.error(
            "Agent finished without a report for '%s'. Tool errors: %s",
            product_name,
            tool_errors or "none found",
        )
        raise AgentError(
            f"Agent did not produce a report for '{product_name}'.",
            tool_errors=tool_errors,
        )

    logger.info("run_analysis completed for product='%s'", product_name)
    return {"final_report": final_report, "report_html": report_html}


# ── Custom exception ───────────────────────────────────────────────────────────


class AgentError(Exception):
    """Raised when the agent completes without producing a valid report."""

    def __init__(self, message: str, tool_errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.tool_errors = tool_errors or []
