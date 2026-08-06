"""日报 Agent 类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class StepResult:
    """单步流水线执行结果。"""

    name: str
    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0


@dataclass
class EventItem:
    """识别出的新闻/事件条目。"""

    title: str
    entity_name: str
    category: str
    sentiment: Literal["利好", "利空", "风险", "业绩变化", "重大事件", "中性"]
    publish_time: str
    url: str
    summary: str
    score: float = 0.0
    source: str = ""


@dataclass
class DailyAnalysis:
    """日报分析结果。"""

    summary: str = ""
    major_events: list[EventItem] = field(default_factory=list)
    bullish: list[EventItem] = field(default_factory=list)
    bearish: list[EventItem] = field(default_factory=list)
    risks: list[EventItem] = field(default_factory=list)
    performance_changes: list[EventItem] = field(default_factory=list)
    llm_used: bool = False


@dataclass
class DailyContext:
    """日报 Agent 运行时上下文。"""

    report_date: str
    news_days: int = 1
    enable_llm: bool = True
    skip_fetch: bool = False
    skip_process: bool = False
    skip_index: bool = False
    stats: dict[str, Any] = field(default_factory=dict)
    step_results: list[StepResult] = field(default_factory=list)
    news_records: list[dict[str, Any]] = field(default_factory=list)
    analysis: DailyAnalysis = field(default_factory=DailyAnalysis)
    report_path: Path | None = None
    errors: list[str] = field(default_factory=list)

    def add_step(self, result: StepResult) -> None:
        self.step_results.append(result)
        self.stats[result.name] = result.data
        if not result.success and result.message:
            self.errors.append(f"{result.name}: {result.message}")

    @property
    def success(self) -> bool:
        return not self.errors
