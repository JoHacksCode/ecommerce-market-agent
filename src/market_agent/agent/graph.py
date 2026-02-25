"""
LangGraph agent graph.
Uses a ReAct-style tool-calling loop:
  START → agent_node → (tool_node | END)
The LLM decides which tools to call and in what order.
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

# ── Tool wrappers (convert BaseTool → LangChain StructuredTool) ────────────────


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

    # ── Nodes ──────────────────────────────────────────────────────────────────

    def agent_node(state: AgentState) -> dict:
        logger.debug("agent_node invoked, messages=%d", len(state["messages"]))
        response: AIMessage = llm.invoke(state["messages"])
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        last: AIMessage = state["messages"][-1]
        tool_messages: list[ToolMessage] = []

        for call in last.tool_calls:
            tool = tool_map[call["name"]]
            logger.debug("Calling tool %s with args %s", call["name"], call["args"])
            result = tool.invoke(call["args"])
            tool_messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

            # Capture the final report when the report_generator tool finishes
            if call["name"] == "report_generator":
                try:
                    payload = json.loads(result)
                    if payload.get("success") and "data" in payload:
                        return {
                            "messages": tool_messages,
                            "final_report": payload["data"],
                            "report_html": payload.get("report_html"),
                        }
                except json.JSONDecodeError:
                    pass

        return {"messages": tool_messages}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    # ── Assemble ───────────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Public runner ──────────────────────────────────────────────────────────────


async def run_analysis(product_name: str) -> dict[str, Any]:
    """Entry-point called by the API layer."""
    graph = build_graph()
    initial_state: AgentState = AgentState(
        messages=[HumanMessage(content=(f"{SYSTEM_PROMPT}\n\nPerform a full market analysis for: {product_name}"))],
        product_name=product_name,
        final_report=None,
        report_html=None,
    )
    final_state = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": settings.agent_recursion_limit},
    )

    if final_state.get("final_report"):
        return final_state["final_report"]

    # Fallback: extract last AI text message
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return {"error": "No final report generated", "last_raw_response": msg.content, "product": product_name}

    return {"error": "Agent did not produce a report.", "product": product_name}
