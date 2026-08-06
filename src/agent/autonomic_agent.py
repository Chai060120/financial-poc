"""
Autonomic Financial Agent：自主运行 + LLM 分析与简报。

无需用户逐条提问，Agent 自动完成：
1. 抓取新闻、处理 PDF/News、更新向量索引
2. 规则 + LLM 事件分析
3. 对监控列表公司自动生成 RAG 研究简报（LLM）
4. 输出 Markdown 报告到 docs/daily/
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import schedule

from config import (
    AGENT_ENABLE_LLM,
    AGENT_NEWS_DAYS,
    AGENT_SCHEDULE_TIME,
    AGENT_WATCHLIST,
    DAILY_REPORT_DIR,
    DEFAULT_ENCODING,
    setup_logging,
)
from src.agent.daily.types import DailyContext
from src.agent.financial_agent import FinancialAgent, create_financial_agent
from src.llm.llm_client import create_llm_client
from src.utils.stock_registry import get_stock_registry

logger = setup_logging(__name__)

@dataclass
class CompanyBriefing:
    """单家公司自主生成的 LLM 简报。"""

    entity_id: str
    entity_name: str
    question: str
    answer: str
    result_count: int
    llm_provider: str = ""
    llm_model: str = ""
    preview_mode: bool = False


@dataclass
class AutonomicRunResult:
    """一次自主运行周期结果。"""

    report_date: str
    success: bool
    summary: str = ""
    report_path: Path | None = None
    briefings_path: Path | None = None
    company_briefings: list[CompanyBriefing] = field(default_factory=list)
    daily_context: DailyContext | None = None
    llm_used: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class AutonomicAgent:
    """
    自主金融 Agent（PDF 财报 + 财经新闻 + LLM）。

    示例:
        agent = AutonomicAgent()
        result = agent.run_once()          # 立即跑一轮
        agent.serve()                      # 每天定时自主运行（阻塞）
    """

    financial: FinancialAgent | None = None
    provider: str | None = None
    enable_llm: bool = AGENT_ENABLE_LLM
    news_days: int = AGENT_NEWS_DAYS
    top_k: int = 5
    skip_briefings: bool = False

    def __post_init__(self) -> None:
        client = create_llm_client(provider=self.provider, allow_preview=True)
        self.financial = self.financial or create_financial_agent(
            llm=client,
            top_k=self.top_k,
        )
        if self.financial.llm is None:
            self.financial.llm = client

    def _resolve_entity_name(self, entity_id: str) -> str:
        registry = get_stock_registry()
        found = registry.lookup_by_id(entity_id)
        return found["entity_name"] if found else entity_id

    def _generate_watchlist_briefings(self, report_date: str) -> list[CompanyBriefing]:
        """对监控列表每家公司自主发起 RAG + LLM 分析。"""
        briefings: list[CompanyBriefing] = []
        if not self.enable_llm:
            logger.info("LLM 已禁用，跳过公司简报")
            return briefings

        # 各公司简报独立，不累积对话历史
        self.financial.history = None  # type: ignore[union-attr]

        for entity_id in AGENT_WATCHLIST:
            entity_name = self._resolve_entity_name(entity_id)
            question = (
                f"{entity_name}（{entity_id}）最新业绩要点、主要财务指标与近期新闻，"
                f"截至{report_date}，需关注的风险有哪些？"
            )
            logger.info("Autonomic: 生成简报 entity=%s", entity_name)
            try:
                payload = self.financial.ask(  # type: ignore[union-attr]
                    question,
                    top_k=self.top_k,
                    provider=self.provider,
                )
                # 本轮问答不进入下一轮的对话历史
                self.financial.history = None  # type: ignore[union-attr]
            except Exception as exc:
                logger.exception("简报生成失败: %s", entity_name)
                briefings.append(
                    CompanyBriefing(
                        entity_id=entity_id,
                        entity_name=entity_name,
                        question=question,
                        answer=f"（生成失败: {exc}）",
                        result_count=0,
                    )
                )
                continue

            llm_meta = payload.get("llm") or {}
            briefings.append(
                CompanyBriefing(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    question=question,
                    answer=str(payload.get("answer") or "").strip(),
                    result_count=int(payload.get("result_count") or 0),
                    llm_provider=str(llm_meta.get("provider") or ""),
                    llm_model=str(llm_meta.get("model") or ""),
                    preview_mode=bool(llm_meta.get("preview_mode")),
                )
            )
        return briefings

    def _save_briefings(
        self,
        report_date: str,
        briefings: list[CompanyBriefing],
    ) -> Path:
        target = DAILY_REPORT_DIR / f"{report_date}-briefings.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "entity_id": item.entity_id,
                "entity_name": item.entity_name,
                "question": item.question,
                "answer": item.answer,
                "result_count": item.result_count,
                "llm_provider": item.llm_provider,
                "llm_model": item.llm_model,
                "preview_mode": item.preview_mode,
            }
            for item in briefings
        ]
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding=DEFAULT_ENCODING,
        )
        return target

    def _append_autonomic_section(
        self,
        ctx: DailyContext,
        briefings: list[CompanyBriefing],
    ) -> Path | None:
        """将 LLM 公司简报追加到日报 Markdown。"""
        if ctx.report_path is None or not ctx.report_path.exists():
            return None

        lines = [
            "",
            "---",
            "",
            "## 自主研究简报（LLM + RAG）",
            "",
            "> 由 Autonomic Agent 自动生成，无需人工提问。",
            "",
        ]
        for index, item in enumerate(briefings, start=1):
            lines.append(f"### {index}. {item.entity_name} ({item.entity_id})")
            if item.preview_mode:
                lines.append("")
                lines.append("*（LLM 未配置 API Key，以下为 Prompt 预览）*")
            lines.append("")
            lines.append(item.answer or "（无输出）")
            lines.append("")

        content = ctx.report_path.read_text(encoding=DEFAULT_ENCODING)
        autonomic_path = DAILY_REPORT_DIR / f"{ctx.report_date}-autonomic.md"
        autonomic_path.write_text(content + "\n".join(lines), encoding=DEFAULT_ENCODING)
        return autonomic_path

    def run_once(
        self,
        *,
        report_date: str | None = None,
        skip_fetch: bool = False,
        skip_process: bool = False,
        skip_index: bool = False,
        skip_briefings: bool | None = None,
    ) -> AutonomicRunResult:
        """
        执行一轮完整自主周期：
        同步数据 → 事件分析 → LLM 日报 → 监控列表公司简报。
        """
        day = report_date or date.today().isoformat()
        logger.info("Autonomic Agent 启动: date=%s", day)

        result = AutonomicRunResult(report_date=day, success=True)

        try:
            ctx = self.financial.daily(  # type: ignore[union-attr]
                report_date=day,
                news_days=self.news_days,
                enable_llm=self.enable_llm,
                skip_fetch=skip_fetch,
                skip_process=skip_process,
                skip_index=skip_index,
            )
            result.daily_context = ctx
            result.summary = ctx.analysis.summary if ctx.analysis else ""
            result.llm_used = bool(ctx.analysis and ctx.analysis.llm_used)
            result.report_path = ctx.report_path
            if ctx.errors:
                result.errors.extend(ctx.errors)
                result.success = False
        except Exception as exc:
            logger.exception("Daily 流水线失败")
            result.success = False
            result.errors.append(str(exc))
            return result

        skip_brief = self.skip_briefings if skip_briefings is None else skip_briefings

        if not skip_brief and self.enable_llm:
            briefings = self._generate_watchlist_briefings(day)
            result.company_briefings = briefings
            result.briefings_path = self._save_briefings(day, briefings)
            autonomic_md = self._append_autonomic_section(ctx, briefings)
            if autonomic_md:
                result.report_path = autonomic_md
            if any(item.preview_mode for item in briefings):
                result.errors.append(
                    "LLM 处于 Preview 模式：请在 .env 配置 API Key 以生成真实简报"
                )
            elif briefings:
                result.llm_used = True

        logger.info(
            "Autonomic Agent 完成: success=%s, report=%s",
            result.success,
            result.report_path,
        )
        return result

    def serve(
        self,
        *,
        schedule_time: str = AGENT_SCHEDULE_TIME,
        run_immediately: bool = True,
    ) -> None:
        """
        启动自主 Agent 服务（阻塞）：每天在 schedule_time 自动运行一轮。

        进程需保持运行；适合服务器或本地常驻。
        """
        logger.info("Autonomic Agent 服务启动: 每天 %s 自动运行", schedule_time)

        def _job() -> None:
            outcome = self.run_once()
            if outcome.success:
                logger.info("定时自主运行成功: %s", outcome.report_path)
            else:
                logger.error("定时自主运行失败: %s", outcome.errors)

        schedule.every().day.at(schedule_time).do(_job)
        if run_immediately:
            _job()

        logger.info("等待下次定时任务…（Ctrl+C 退出）")
        while True:
            schedule.run_pending()
            time.sleep(30)


def create_autonomic_agent(**kwargs: Any) -> AutonomicAgent:
    return AutonomicAgent(**kwargs)
