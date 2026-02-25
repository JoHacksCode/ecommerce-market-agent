"""Base abstractions shared by all market-analysis tools."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Standardised envelope returned by every tool."""

    tool_name: str
    success: bool
    data: dict[str, Any]
    report_html: str | None = None  # populated only by ReportGeneratorTool
    error: str | None = None


class BaseTool(ABC):
    """Abstract base class that enforces a consistent tool contract."""

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute tool logic — subclasses must implement this."""
        ...

    def safe_run(self, **kwargs: Any) -> ToolResult:
        """Execute with error handling; never raises."""
        try:
            return self.run(**kwargs)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error=str(exc),
            )
