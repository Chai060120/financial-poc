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
  Financial Research Agent · 财报对话分析
════════════════════════════════════════════════════════════
  直接输入「公司名」或「公司名 + 年份」，例如：
    · 美的集团
    · 美的集团 2024
    · 贵州茅台2024年报
  Agent 会自动检索巨潮年报、抓取网络新闻，并输出带引用溯源的报告。
  也可继续手动上传 PDF。

  公司对比：
    · 茅台 vs 五粮液
    · 对比美的集团和格力电器
    · 和某某公司比呢？

  之后可继续追问：
    · 为什么低估？ / 依据是什么？ / 出处在哪？
    · 净利润多少？ / PE 多少？
    · 导出报告

  命令: 帮助 | 清空 | 导出报告 | 退出
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
        from src.agent.compare_card import (
            ensure_peer_fundamentals,
            format_company_compare_card,
        )

        base_name = route.entity_name or self.session.last_entity_name
        base_id = route.entity_id or self.session.last_entity_id
        other_name = route.compare_target
        if not base_name:
            return AgentTurnResult(
                AgentIntent.COMPARE_WITH,
                "请先分析一家公司，或直接输入：茅台 vs 五粮液",
            )
        if not other_name:
            return AgentTurnResult(
                AgentIntent.COMPARE_WITH,
                "请说明要对比的公司，例如：茅台 vs 五粮液 / 和某某公司比呢？",
            )

        registry = get_stock_registry()
        base = registry.lookup_by_name(base_name) or detect_fallback(base_name, base_id)
        other = registry.lookup_by_name(other_name) or {}
        if not base_id:
            base_id = str(base.get("entity_id") or "")
        base_name = str(base.get("entity_name") or base_name)
        other_id = str(other.get("entity_id") or "")
        other_name = str(other.get("entity_name") or other_name)

        notes: list[str] = []
        fund_a = self._fundamentals_for(base_name, base_id)
        if not (fund_a.get("eps") or fund_a.get("net_profit")):
            fund_a, note_a = ensure_peer_fundamentals(
                name=base_name,
                entity_id=base_id,
                fundamentals_lookup=self._fundamentals_for,
                financial_agent=self.financial,
                report_year=route.report_year or None,
            )
            if note_a:
                notes.append(note_a)

        fund_b, note_b = ensure_peer_fundamentals(
            name=other_name,
            entity_id=other_id,
            fundamentals_lookup=self._fundamentals_for,
            financial_agent=self.financial,
            report_year=route.report_year or None,
        )
        if note_b:
            notes.append(note_b)

        snap_a = fetch_market_snapshot_enriched(base_id, base_name, fund_a)
        snap_b = fetch_market_snapshot_enriched(other_id, other_name, fund_b)

        table = format_company_compare_card(
            left_name=base_name,
            right_name=other_name,
            left_id=base_id,
            right_id=other_id,
            fund_a=fund_a,
            fund_b=fund_b,
            snap_a=snap_a,
            snap_b=snap_b,
            notes=notes,
        )
        self.session.last_entity_name = base_name
        self.session.last_entity_id = base_id
        self.session.last_report_card = table

        cursor_answer = answer_followup_question(
            f"请简要比较 {base_name} 和 {other_name}",
            entity_name=base_name,
            context=table,
        )
        if cursor_answer:
            message = table + "\n\n自然语言解读:\n" + cursor_answer
            self.session.last_report_card = message
            return AgentTurnResult(AgentIntent.COMPARE_WITH, message)
        return AgentTurnResult(AgentIntent.COMPARE_WITH, table)

    def _handle_export(self, route: RoutedIntent) -> AgentTurnResult:
        from src.agent.report_export import export_report

        report = self.session.last_report_card.strip()
        if not report and self.session.last_analysis is not None:
            report = self._build_report_card(self.session.last_analysis)
            self.session.last_report_card = report
        if not report:
            return AgentTurnResult(
                AgentIntent.EXPORT,
                "暂无可导出报告。请先分析一家公司，或执行「茅台 vs 五粮液」对比。",
            )

        try:
            md_path = export_report(
                report,
                entity_name=self.session.last_entity_name or "report",
                fmt="md",
            )
            html_path = export_report(
                report,
                entity_name=self.session.last_entity_name or "report",
                fmt="html",
            )
        except Exception as exc:
            return AgentTurnResult(AgentIntent.EXPORT, f"导出失败: {exc}")

        return AgentTurnResult(
            AgentIntent.EXPORT,
            "报告已导出：\n"
            f"  Markdown: {md_path}\n"
            f"  HTML:     {html_path}\n"
            "可用浏览器打开 HTML，再「打印 → 另存为 PDF」。\n"
            "网页端也可点击「导出报告」直接下载。",
        )

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
            AgentIntent.EXPORT: self._handle_export,
            AgentIntent.FULL_ANALYZE: self._handle_full_analyze,
            AgentIntent.INGEST_PDF: self._handle_full_analyze,
            AgentIntent.VALUATE: self._handle_full_analyze,
        }

        handler = handlers.get(route.intent)
        if handler is not None:
            if route.intent in {AgentIntent.VALUATE, AgentIntent.INGEST_PDF}:
                route = RoutedIntent(
                    AgentIntent.FULL_ANALYZE,
                    route.raw_input,
                    entity_name=route.entity_name,
                    entity_id=route.entity_id,
                    pdf_path=route.pdf_path,
                    report_year=route.report_year,
                    report_type=route.report_type,
                )
            result = handler(route)
            self.session.turns.append((user_input, result.message))
            return result

        if route.entity_name or route.pdf_path:
            merged = RoutedIntent(
                AgentIntent.FULL_ANALYZE,
                route.raw_input,
                entity_name=route.entity_name,
                entity_id=route.entity_id,
                pdf_path=route.pdf_path,
                report_year=route.report_year,
                report_type=route.report_type,
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
            "请输入公司名或 PDF。可试：茅台 vs 五粮液 / 导出报告 / 为什么？",
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
                AgentIntent.EXPORT,
            }:
                print("\n处理中…")

            result = self.handle(user_input)
            if result.intent == AgentIntent.EXIT:
                print(result.message)
                break
            print("\n" + result.message)


def detect_fallback(name: str, entity_id: str) -> dict[str, str]:
    return {"entity_name": name, "entity_id": entity_id}


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
