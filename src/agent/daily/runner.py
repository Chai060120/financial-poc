"""Daily Agent 编排器。"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence, Type

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import AGENT_ENABLE_LLM, AGENT_NEWS_DAYS, ensure_dirs, setup_logging
from src.agent.daily.base import DailyStep
from src.agent.daily.steps import (
    AnalyzeStep,
    BuildIndexStep,
    DEFAULT_DAILY_STEPS,
    FetchNewsStep,
    GenerateReportStep,
    ProcessDocumentsStep,
)
from src.agent.daily.types import DailyContext, StepResult
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine, create_retrieval_engine

logger = setup_logging(__name__)


class DailyAgent:
    """
    金融信息处理 Daily Agent。

    默认流水线:
        抓新闻 → 解析 PDF/News → 建向量 → 分析 → 生成 Markdown 日报
    """

    def __init__(
        self,
        steps: Sequence[DailyStep] | None = None,
        *,
        store: ChromaStore | None = None,
        engine: UnifiedRetrievalEngine | None = None,
    ) -> None:
        if steps is None:
            store = store or ChromaStore()
            engine = engine or create_retrieval_engine()
            self.steps: list[DailyStep] = [
                FetchNewsStep(),
                ProcessDocumentsStep(),
                BuildIndexStep(store=store),
                AnalyzeStep(engine=engine),
                GenerateReportStep(),
            ]
        else:
            self.steps = list(steps)

    @classmethod
    def with_step_classes(
        cls,
        step_classes: Sequence[Type[DailyStep]] = DEFAULT_DAILY_STEPS,
        **kwargs: Any,
    ) -> DailyAgent:
        """通过步骤类列表构建 Agent（便于扩展/替换）。"""
        store = kwargs.get("store") or ChromaStore()
        engine = kwargs.get("engine") or create_retrieval_engine()
        instances: list[DailyStep] = []
        for step_cls in step_classes:
            if step_cls is BuildIndexStep:
                instances.append(BuildIndexStep(store=store))
            elif step_cls is AnalyzeStep:
                instances.append(AnalyzeStep(engine=engine))
            else:
                instances.append(step_cls())
        return cls(steps=instances)

    def run(
        self,
        *,
        report_date: str | None = None,
        news_days: int = AGENT_NEWS_DAYS,
        enable_llm: bool = AGENT_ENABLE_LLM,
        skip_fetch: bool = False,
        skip_process: bool = False,
        skip_index: bool = False,
    ) -> DailyContext:
        ensure_dirs()
        ctx = DailyContext(
            report_date=report_date or date.today().isoformat(),
            news_days=news_days,
            enable_llm=enable_llm,
            skip_fetch=skip_fetch,
            skip_process=skip_process,
            skip_index=skip_index,
        )

        logger.info(
            "Daily Agent 启动: date=%s, steps=%d, llm=%s",
            ctx.report_date,
            len(self.steps),
            enable_llm,
        )

        for step in self.steps:
            logger.info(">>> 执行步骤: %s", step.name)
            result: StepResult = step.run(ctx)
            ctx.add_step(result)

        if ctx.report_path:
            logger.info("Daily Agent 完成: report=%s, success=%s", ctx.report_path, ctx.success)
        else:
            logger.warning("Daily Agent 完成但未生成报告, success=%s, errors=%s", ctx.success, ctx.errors)

        return ctx


def run_daily_agent(**kwargs: Any) -> DailyContext:
    """便捷入口：运行默认 Daily Agent。"""
    return DailyAgent().run(**kwargs)
