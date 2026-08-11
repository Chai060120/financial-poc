"""
财报分析 Agent：一次输入 → 五段式完整报告，静默运行。

不依赖 LLM API。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import setup_logging
from src.agent.financial_agent import FinancialAgent
from src.agent.intent_router import AgentIntent, RoutedIntent, route_intent
from src.agent.report_card import build_peer_rows, format_report_card
from src.analysis.full_report import FullAnalysisResult
from src.analysis.market_compare import ComparisonResult
from src.analysis.valuation import ValuationResult
from src.utils.quiet_mode import quiet_analysis

logger = setup_logging(__name__)

AGENT_WELCOME = """
════════════════════════════════════════════════════════════
  Financial Agent · 财报一键分析
════════════════════════════════════════════════════════════
  输入公司名或 PDF 文件名，自动输出完整五段报告：
    1. 估值结论（高估/合理/低估）
    2. 财报核心指标
    3. 关键词
    4. 网络行情与横向对比
    5. 近期资讯与研判依据

  示例: 贵州茅台   |   贵州茅台2024年报.pdf

  命令: 退出
════════════════════════════════════════════════════════════
"""


@dataclass
class AgentSession:
    last_entity_name: str = ""
    last_entity_id: str = ""
    last_analysis: FullAnalysisResult | None = None
    last_valuation: ValuationResult | None = None
    last_comparison: ComparisonResult | None = None
    turns: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class AgentTurnResult:
    intent: AgentIntent
    message: str
    payload: Any = None


class AnalysisAgent:
    """规则驱动财报分析 Agent。"""

    def __init__(self, financial_agent: FinancialAgent) -> None:
        self.financial = financial_agent
        self.session = AgentSession()

    def _fundamentals_for(self, entity_name: str, entity_id: str) -> dict:
        if (
            self.session.last_analysis
            and self.session.last_entity_id == entity_id
            and self.session.last_analysis.fundamentals
        ):
            return self.session.last_analysis.fundamentals
        if (
            self.session.last_valuation
            and self.session.last_valuation.entity_id == entity_id
            and self.session.last_valuation.fundamentals
        ):
            return self.session.last_valuation.fundamentals
        engine = self.financial.engine
        if engine is None:
            return {}
        try:
            from src.analysis.valuation import _detect_report_context, _extract_fundamentals

            ctx = _detect_report_context(engine, entity_name, entity_id)  # type: ignore[arg-type]
            return _extract_fundamentals(engine, entity_name, entity_id, ctx)  # type: ignore[arg-type]
        except Exception as exc:
            logger.debug("财报指标加载跳过: %s", exc)
            return {}

    def _build_report_card(self, result: FullAnalysisResult) -> str:
        peer_rows = build_peer_rows(result, self._fundamentals_for)
        return format_report_card(result, peer_rows)

    def _handle_full_analyze(self, route: RoutedIntent) -> AgentTurnResult:
        target: str | Path | None
        if route.pdf_path:
            target = route.pdf_path
        elif route.entity_name:
            target = route.entity_name
        else:
            target = None

        with quiet_analysis():
            results = self.financial.analyze(target, save_report=True)

        if not results:
            return AgentTurnResult(AgentIntent.FULL_ANALYZE, "未生成分析结果，请检查 PDF 或公司名。")

        blocks: list[str] = []
        for primary in results:
            self.session.last_analysis = primary
            self.session.last_entity_name = primary.entity_name
            self.session.last_entity_id = primary.entity_id
            if primary.valuation:
                self.session.last_valuation = primary.valuation
            if primary.comparison:
                self.session.last_comparison = primary.comparison
            blocks.append(self._build_report_card(primary))

        message = blocks[0] if len(blocks) == 1 else "\n\n".join(blocks)
        return AgentTurnResult(AgentIntent.FULL_ANALYZE, message, results[0] if len(results) == 1 else results)

    def handle(self, user_input: str) -> AgentTurnResult:
        route = route_intent(
            user_input,
            last_entity_name=self.session.last_entity_name,
            last_entity_id=self.session.last_entity_id,
            has_last_analysis=self.session.last_analysis is not None,
        )

        if route.intent == AgentIntent.EXIT:
            return AgentTurnResult(AgentIntent.EXIT, "再见。")
        if route.intent == AgentIntent.RESET:
            self.session = AgentSession()
            return AgentTurnResult(AgentIntent.RESET, "已清空。")
        if route.intent == AgentIntent.HELP:
            return AgentTurnResult(AgentIntent.HELP, AGENT_WELCOME.strip())

        # 默认：任何公司名 / PDF / 分析类输入 → 一次出完整五段报告
        auto_full = route.intent in {
            AgentIntent.FULL_ANALYZE,
            AgentIntent.INGEST_PDF,
            AgentIntent.VALUATE,
            AgentIntent.COMPARE,
            AgentIntent.COMPARE_WITH,
            AgentIntent.QUERY,
            AgentIntent.EXPLAIN,
        }
        if auto_full or route.intent == AgentIntent.UNKNOWN:
            if route.intent == AgentIntent.UNKNOWN and not route.entity_name and not route.pdf_path:
                if not user_input.strip() in {"分析", "开始分析", "analyze"}:
                    return AgentTurnResult(
                        AgentIntent.UNKNOWN,
                        "请输入公司名或 PDF 文件名，例如：贵州茅台",
                    )
            # 统一走全量分析（含五段报告 + 自动横向对比）
            if route.entity_name or route.pdf_path or route.intent == AgentIntent.FULL_ANALYZE:
                merged = RoutedIntent(
                    AgentIntent.FULL_ANALYZE,
                    route.raw_input,
                    entity_name=route.entity_name,
                    entity_id=route.entity_id,
                    pdf_path=route.pdf_path,
                )
                result = self._handle_full_analyze(merged)
                self.session.turns.append((user_input, result.message))
                return result
            if self.session.last_entity_name:
                merged = RoutedIntent(
                    AgentIntent.FULL_ANALYZE,
                    route.raw_input,
                    entity_name=self.session.last_entity_name,
                    entity_id=self.session.last_entity_id,
                )
                result = self._handle_full_analyze(merged)
                self.session.turns.append((user_input, result.message))
                return result

        return AgentTurnResult(AgentIntent.UNKNOWN, "请输入公司名或 PDF 文件名。")

    def run_interactive(self) -> None:
        print(AGENT_WELCOME.strip())
        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q", "退出", "再见"}:
                print("再见。")
                break

            print("\n分析中，请稍候…")
            result = self.handle(user_input)
            if result.intent == AgentIntent.EXIT:
                print(result.message)
                break
            print("\n" + result.message)


def run_analysis_agent(financial_agent: FinancialAgent) -> None:
    AnalysisAgent(financial_agent).run_interactive()


def format_results_for_display(results: list[FullAnalysisResult], fundamentals_lookup) -> str:
    """供 analyze 子命令使用的五段式输出。"""
    blocks = [
        format_report_card(r, build_peer_rows(r, fundamentals_lookup)) for r in results
    ]
    return blocks[0] if len(blocks) == 1 else "\n\n".join(blocks)
