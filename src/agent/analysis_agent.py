"""
财报分析 Agent：全量分析 + 多轮追问。

规则驱动；可选 Cursor 增强解释。不依赖 OpenAI 类 LLM API。
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
from src.agent.cursor_narrator import (
    answer_followup_question,
    append_cursor_narrative,
)
from src.agent.report_card import build_peer_rows, format_report_card
from src.analysis.full_report import FullAnalysisResult
from src.analysis.market_compare import ComparisonResult
from src.analysis.market_data import fetch_market_snapshot_enriched
from src.analysis.valuation import ValuationResult
from src.utils.quiet_mode import quiet_analysis
from src.utils.stock_registry import get_stock_registry

logger = setup_logging(__name__)

AGENT_WELCOME = """
════════════════════════════════════════════════════════════
  Financial Agent · 财报对话分析
════════════════════════════════════════════════════════════
  直接输入「公司名」或「公司名 + 年份」，例如：
    · 美的集团
    · 美的集团 2024
    · 贵州茅台2024年报
  Agent 会自动检索巨潮年报、抓取网络新闻，并输出五段报告。
  也可继续手动上传 PDF。

  之后可继续追问，例如：
    · 为什么低估？ / 依据是什么？
    · 和同业比呢？ / 和某某公司比呢？
    · 净利润多少？ / PE 多少？

  命令: 帮助 | 清空 | 退出
════════════════════════════════════════════════════════════
"""

_SLOW_INTENTS = frozenset(
    {
        AgentIntent.FULL_ANALYZE,
        AgentIntent.INGEST_PDF,
        AgentIntent.VALUATE,
        AgentIntent.COMPARE,
        AgentIntent.COMPARE_WITH,
    }
)


@dataclass
class AgentSession:
    last_entity_name: str = ""
    last_entity_id: str = ""
    last_analysis: FullAnalysisResult | None = None
    last_valuation: ValuationResult | None = None
    last_comparison: ComparisonResult | None = None
    last_report_card: str = ""
    turns: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class AgentTurnResult:
    intent: AgentIntent
    message: str
    payload: Any = None


class AnalysisAgent:
    """规则驱动财报分析 Agent（支持多轮追问）。"""

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

    def _remember_analysis(self, primary: FullAnalysisResult, report_card: str = "") -> None:
        self.session.last_analysis = primary
        self.session.last_entity_name = primary.entity_name
        self.session.last_entity_id = primary.entity_id
        if primary.valuation:
            self.session.last_valuation = primary.valuation
        if primary.comparison:
            self.session.last_comparison = primary.comparison
        if report_card:
            self.session.last_report_card = report_card

    def _build_report_card(self, result: FullAnalysisResult) -> str:
        peer_rows = build_peer_rows(result, self._fundamentals_for)
        card = format_report_card(result, peer_rows)
        card = append_cursor_narrative(card, result.entity_name)
        try:
            from src.agent.llm_research import append_llm_research_report

            card = append_llm_research_report(card, result)
        except Exception as exc:
            logger.debug("LLM 研报增强跳过: %s", exc)
        return card

    def _handle_full_analyze(self, route: RoutedIntent) -> AgentTurnResult:
        target: str | Path | None
        if route.pdf_path:
            target = route.pdf_path
        elif route.entity_name:
            target = route.entity_name
        else:
            target = None

        with quiet_analysis():
            results = self.financial.analyze(
                target,
                save_report=True,
                report_year=route.report_year or None,
                report_type=route.report_type or "年报",
                auto_fetch_report=True,
            )

        if not results:
            return AgentTurnResult(AgentIntent.FULL_ANALYZE, "未生成分析结果，请检查 PDF 或公司名。")

        blocks: list[str] = []
        for primary in results:
            card = self._build_report_card(primary)
            self._remember_analysis(primary, card)
            blocks.append(card)

        tip = (
            "\n\n可继续追问：为什么？ / 和同业比呢？ / 和某某公司比呢？ / 净利润多少？"
            if len(blocks) == 1
            else ""
        )
        message = (blocks[0] if len(blocks) == 1 else "\n\n".join(blocks)) + tip
        return AgentTurnResult(
            AgentIntent.FULL_ANALYZE,
            message,
            results[0] if len(results) == 1 else results,
        )

    def _handle_explain(self, route: RoutedIntent) -> AgentTurnResult:
        analysis = self.session.last_analysis
        if analysis is None:
            return AgentTurnResult(
                AgentIntent.EXPLAIN,
                "还没有分析记录。请先输入公司名，例如：中国平安",
            )

        # 优先用 Cursor 基于上次报告回答
        context = self.session.last_report_card or ""
        if not context:
            peer_rows = build_peer_rows(analysis, self._fundamentals_for)
            context = format_report_card(analysis, peer_rows)

        cursor_answer = answer_followup_question(
            route.raw_input or "为什么得出这个估值结论？",
            entity_name=analysis.entity_name,
            context=context,
        )
        if cursor_answer:
            return AgentTurnResult(AgentIntent.EXPLAIN, cursor_answer)

        # 规则兜底：列出研判依据
        lines = [
            f"关于 {analysis.entity_name} 的估值说明",
            "",
            f"结论: {analysis.final_verdict}（置信度 {analysis.final_confidence}）",
            "",
            "主要依据:",
        ]
        reasons = analysis.synthesis_reasons or []
        if analysis.valuation and analysis.valuation.reasons:
            for reason in analysis.valuation.reasons:
                if reason not in reasons:
                    reasons.append(reason)
        if not reasons:
            lines.append("  · 暂无详细依据，可重新输入公司名做全量分析。")
        else:
            for reason in reasons[:10]:
                lines.append(f"  · {reason}")

        if analysis.data_warnings:
            lines.append("")
            lines.append("数据质量提示:")
            for warn in analysis.data_warnings[:4]:
                lines.append(f"  · {warn}")

        lines.extend(
            [
                "",
                "如需重新分析，直接输入公司名；要比价可问：和同业比呢？ / 和某某公司比呢？",
            ]
        )
        return AgentTurnResult(AgentIntent.EXPLAIN, "\n".join(lines))

    def _handle_query(self, route: RoutedIntent) -> AgentTurnResult:
        entity_name = route.entity_name or self.session.last_entity_name
        entity_id = route.entity_id or self.session.last_entity_id
        question = route.question or route.raw_input

        if not entity_name:
            return AgentTurnResult(AgentIntent.QUERY, "请先指定公司，例如：贵州茅台净利润")

        # 先用会话里的 fundamentals 快速回答
        fund = self._fundamentals_for(entity_name, entity_id)
        quick = self._answer_from_fundamentals(question, fund, entity_name)
        if quick:
            return AgentTurnResult(AgentIntent.QUERY, quick)

        # Cursor 追问（有上下文时）
        if self.session.last_report_card:
            cursor_answer = answer_followup_question(
                question,
                entity_name=entity_name,
                context=self.session.last_report_card,
            )
            if cursor_answer:
                return AgentTurnResult(AgentIntent.QUERY, cursor_answer)

        # 回退检索
        try:
            payload = self.financial.query(question)
            results = payload.get("results") or []
            if not results:
                return AgentTurnResult(
                    AgentIntent.QUERY,
                    f"未在 {entity_name} 财报中检索到相关内容。可换个问法，或先做全量分析。",
                )
            lines = [f"检索：{question}", f"命中 {len(results)} 条：", ""]
            for index, item in enumerate(results[:3], start=1):
                meta = item.get("metadata") or {}
                text = str(item.get("text") or "").replace("\n", " ")[:180]
                title = meta.get("file_name") or meta.get("title") or ""
                lines.append(f"[{index}] {title}")
                lines.append(f"    {text}…")
                lines.append("")
            return AgentTurnResult(AgentIntent.QUERY, "\n".join(lines).rstrip())
        except Exception as exc:
            return AgentTurnResult(AgentIntent.QUERY, f"检索失败: {exc}")

    @staticmethod
    def _answer_from_fundamentals(question: str, fund: dict, entity_name: str) -> str:
        if not fund:
            return ""
        q = question.lower()
        mapping = (
            (("净利润", "净利"), "net_profit", "净利润"),
            (("营收", "营业收入", "收入"), "revenue", "营业收入"),
            (("每股收益", "eps"), "eps", "每股收益"),
            (("每股净资产", "bvps", "净资产/股"), "bvps", "每股净资产"),
            (("roe", "净资产收益率"), "roe", "ROE"),
        )
        hits: list[str] = []
        for keys, field, label in mapping:
            if any(k.lower() in q or k in question for k in keys):
                item = fund.get(field)
                if isinstance(item, dict) and item.get("display"):
                    hits.append(f"{label}: {item['display']}")
        if "pe" in q or "市盈" in question:
            # 由会话估值行情补充
            hits.append("PE: 请看上次报告【4】网络行情；或问「为什么」查看依据")
        if not hits:
            return ""
        return f"{entity_name}（来自已分析财报）\n  " + "\n  ".join(hits)

    def _handle_compare(self, route: RoutedIntent) -> AgentTurnResult:
        entity_name = route.entity_name or self.session.last_entity_name
        if not entity_name:
            return AgentTurnResult(AgentIntent.COMPARE, "请先分析一家公司，再问同业对比。")

        # 有上次全量结果时，直接用报告里的监控列表对比段落
        if (
            self.session.last_analysis
            and self.session.last_entity_name == entity_name
            and self.session.last_report_card
        ):
            card = self.session.last_report_card
            start = card.find("【4】网络行情与横向对比")
            end = card.find("【5】")
            if start >= 0:
                snippet = card[start:end].strip() if end > start else card[start:].strip()
                return AgentTurnResult(
                    AgentIntent.COMPARE,
                    f"{entity_name} 横向对比（来自上次分析）\n\n{snippet}\n\n"
                    "要比某一家，可以说：和某某公司比呢？",
                )

        with quiet_analysis():
            try:
                result = self.financial.compare(entity_name, watchlist=True)
            except Exception as exc:
                return AgentTurnResult(AgentIntent.COMPARE, f"对比失败: {exc}")

        self.session.last_comparison = result
        self.session.last_entity_name = result.entity_name
        self.session.last_entity_id = result.entity_id
        lines = [
            f"实时对比 · {result.entity_name}",
            f"相对同业: {result.relative_verdict}",
            f"新闻情绪: {result.news_sentiment}",
            "",
            f"摘要: {result.summary}",
        ]
        if result.watchlist:
            lines.append("")
            lines.append("监控列表:")
            for row in result.watchlist[:6]:
                pe = f"{row.pe_ttm:.2f}" if row.pe_ttm else "—"
                pb = f"{row.pb:.2f}" if row.pb else "—"
                lines.append(f"  {row.entity_name}: PE={pe}  PB={pb}")
        return AgentTurnResult(AgentIntent.COMPARE, "\n".join(lines))

    def _handle_compare_with(self, route: RoutedIntent) -> AgentTurnResult:
        base_name = route.entity_name or self.session.last_entity_name
        base_id = route.entity_id or self.session.last_entity_id
        other_name = route.compare_target
        if not base_name:
            return AgentTurnResult(
                AgentIntent.COMPARE_WITH,
                "请先分析一家公司，再问：和同业比呢？ / 和某某公司比呢？",
            )
        if not other_name:
            return AgentTurnResult(AgentIntent.COMPARE_WITH, "请说明要对比的公司，例如：和某某公司比呢？")

        registry = get_stock_registry()
        other = registry.lookup_by_name(other_name) or {}
        other_id = str(other.get("entity_id") or "")
        other_name = str(other.get("entity_name") or other_name)

        fund_a = self._fundamentals_for(base_name, base_id)
        fund_b = self._fundamentals_for(other_name, other_id)
        snap_a = fetch_market_snapshot_enriched(base_id, base_name, fund_a)
        snap_b = fetch_market_snapshot_enriched(other_id, other_name, fund_b)

        def _disp(fund: dict, key: str) -> str:
            item = fund.get(key)
            if isinstance(item, dict) and item.get("display"):
                return str(item["display"])
            return "—"

        def _num(v: float | None) -> str:
            return f"{v:.2f}" if v is not None else "—"

        lines = [
            f"{base_name} vs {other_name}",
            "",
            f"{'指标':<12} {base_name:<16} {other_name:<16}",
            f"{'现价':<12} {_num(snap_a.price):<16} {_num(snap_b.price):<16}",
            f"{'PE':<12} {_num(snap_a.pe_ttm):<16} {_num(snap_b.pe_ttm):<16}",
            f"{'PB':<12} {_num(snap_a.pb):<16} {_num(snap_b.pb):<16}",
            f"{'净利润':<12} {_disp(fund_a, 'net_profit'):<16} {_disp(fund_b, 'net_profit'):<16}",
            f"{'营收':<12} {_disp(fund_a, 'revenue'):<16} {_disp(fund_b, 'revenue'):<16}",
            f"{'EPS':<12} {_disp(fund_a, 'eps'):<16} {_disp(fund_b, 'eps'):<16}",
            f"{'ROE':<12} {_disp(fund_a, 'roe'):<16} {_disp(fund_b, 'roe'):<16}",
        ]

        note = ""
        if snap_a.pe_ttm and snap_b.pe_ttm:
            if snap_a.pe_ttm > snap_b.pe_ttm * 1.1:
                note = f"{base_name} 的 PE 更高（相对更贵）"
            elif snap_a.pe_ttm < snap_b.pe_ttm * 0.9:
                note = f"{base_name} 的 PE 更低（相对更便宜）"
            else:
                note = "两家 PE 接近"
        if note:
            lines.extend(["", f"简评: {note}"])

        if not fund_b.get("eps") and not fund_b.get("net_profit"):
            lines.append(
                f"提示: {other_name} 本地可能尚未入库财报，指标可能不完整。"
                "可将 PDF 放入 data/raw/pdf/ 后执行 python scripts/agent.py pdf"
            )

        # 可选 Cursor 润色
        table = "\n".join(lines)
        cursor_answer = answer_followup_question(
            f"请简要比较 {base_name} 和 {other_name}",
            entity_name=base_name,
            context=table,
        )
        if cursor_answer:
            return AgentTurnResult(
                AgentIntent.COMPARE_WITH,
                table + "\n\n自然语言解读:\n" + cursor_answer,
            )
        return AgentTurnResult(AgentIntent.COMPARE_WITH, table)

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
            return AgentTurnResult(AgentIntent.RESET, "已清空会话，可重新输入公司名。")
        if route.intent == AgentIntent.HELP:
            return AgentTurnResult(AgentIntent.HELP, AGENT_WELCOME.strip())

        handlers = {
            AgentIntent.EXPLAIN: self._handle_explain,
            AgentIntent.QUERY: self._handle_query,
            AgentIntent.COMPARE: self._handle_compare,
            AgentIntent.COMPARE_WITH: self._handle_compare_with,
            AgentIntent.FULL_ANALYZE: self._handle_full_analyze,
            AgentIntent.INGEST_PDF: self._handle_full_analyze,
            AgentIntent.VALUATE: self._handle_full_analyze,
        }

        handler = handlers.get(route.intent)
        if handler is not None:
            # VALUATE / INGEST 统一走全量时，补齐 entity
            if route.intent in {AgentIntent.VALUATE, AgentIntent.INGEST_PDF}:
                route = RoutedIntent(
                    AgentIntent.FULL_ANALYZE,
                    route.raw_input,
                    entity_name=route.entity_name,
                    entity_id=route.entity_id,
                    pdf_path=route.pdf_path,
                )
            result = handler(route)
            self.session.turns.append((user_input, result.message))
            return result

        # UNKNOWN：有公司名则分析；有上次公司且像追问则解释/查询
        if route.entity_name or route.pdf_path:
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

        if self.session.last_entity_name and user_input.strip() in {
            "分析",
            "开始分析",
            "analyze",
            "再分析",
            "重新分析",
        }:
            merged = RoutedIntent(
                AgentIntent.FULL_ANALYZE,
                user_input,
                entity_name=self.session.last_entity_name,
                entity_id=self.session.last_entity_id,
            )
            result = self._handle_full_analyze(merged)
            self.session.turns.append((user_input, result.message))
            return result

        return AgentTurnResult(
            AgentIntent.UNKNOWN,
            "请输入公司名或 PDF 文件名。分析完成后可追问：为什么？ / 和同业比呢？ / 净利润多少？",
        )

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

            # 先路由，避免追问也显示「分析中」
            preview = route_intent(
                user_input,
                last_entity_name=self.session.last_entity_name,
                last_entity_id=self.session.last_entity_id,
                has_last_analysis=self.session.last_analysis is not None,
            )
            if preview.intent in _SLOW_INTENTS or (
                preview.intent == AgentIntent.UNKNOWN
                and (preview.entity_name or preview.pdf_path)
            ):
                print("\n分析中，请稍候…")
            elif preview.intent in {
                AgentIntent.EXPLAIN,
                AgentIntent.QUERY,
                AgentIntent.COMPARE,
                AgentIntent.COMPARE_WITH,
            }:
                print("\n处理中…")

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
        append_cursor_narrative(
            format_report_card(r, build_peer_rows(r, fundamentals_lookup)),
            r.entity_name,
        )
        for r in results
    ]
    return blocks[0] if len(blocks) == 1 else "\n\n".join(blocks)
