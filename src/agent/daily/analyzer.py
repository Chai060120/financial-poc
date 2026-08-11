"""LLM 日报分析：自动总结与结构化提取。"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import AGENT_RETRIEVAL_TOP_K, AGENT_WATCHLIST, setup_logging
from src.agent.daily.event_detector import detect_events
from src.agent.daily.types import DailyAnalysis, DailyContext, EventItem
from src.agent.prompts.news_prompt import build_daily_summary_prompt
from src.agent.prompts.system_prompt import FINANCIAL_LLM_TEMPERATURE, SYSTEM_PROMPT
from src.agent.workflow import merge_retrieval_context
from src.llm.llm_client import LLMClientError, create_llm_client
from src.utils.stock_registry import get_stock_registry
from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine

logger = setup_logging(__name__)


def _filter_news_by_date(records: list[dict[str, Any]], report_date: str) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for record in records:
        publish_time = str(record.get("publish_time") or record.get("date") or "")
        if publish_time.startswith(report_date):
            matched.append(record)
    return matched


def _rule_based_summary(ctx: DailyContext, analysis: DailyAnalysis) -> str:
    parts = [
        f"{ctx.report_date} 共监测新闻 {len(ctx.news_records)} 条。",
        f"识别重大事件 {len(analysis.major_events)} 条、"
        f"利好 {len(analysis.bullish)} 条、"
        f"利空 {len(analysis.bearish)} 条、"
        f"风险 {len(analysis.risks)} 条、"
        f"业绩相关 {len(analysis.performance_changes)} 条。",
    ]
    if analysis.bullish:
        parts.append(f"关注利好: {analysis.bullish[0].title[:40]}")
    if analysis.bearish:
        parts.append(f"关注利空: {analysis.bearish[0].title[:40]}")
    return " ".join(parts)


def _retrieve_watchlist_context(engine: UnifiedRetrievalEngine, report_date: str) -> str:
    registry = get_stock_registry()
    results = []
    per_entity_k = max(2, AGENT_RETRIEVAL_TOP_K // max(len(AGENT_WATCHLIST), 1))

    for entity_id in AGENT_WATCHLIST:
        found = registry.lookup_by_id(entity_id)
        entity_name = found["entity_name"] if found else entity_id
        query = f"{entity_name} {report_date} 业绩 风险 重大事件"
        try:
            batch = engine.retrieve(
                query,
                top_k=per_entity_k,
                where={"entity_id": entity_id},
            )
            results.extend(batch)
        except Exception as exc:
            logger.warning("检索失败: entity=%s | %s", entity_id, exc)

    if not results:
        try:
            results = engine.retrieve(f"金融市场 {report_date}", top_k=AGENT_RETRIEVAL_TOP_K)
        except Exception as exc:
            logger.warning("通用检索失败: %s", exc)
            return ""

    return merge_retrieval_context(results[:AGENT_RETRIEVAL_TOP_K])


def _parse_llm_json(text: str) -> dict[str, Any]:
    content = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
    if fence:
        content = fence.group(1).strip()
    return json.loads(content)


def _merge_llm_analysis(base: DailyAnalysis, payload: dict[str, Any]) -> DailyAnalysis:
    base.summary = str(payload.get("summary") or base.summary)
    base.llm_used = True
    return base


def analyze_daily(
    ctx: DailyContext,
    engine: UnifiedRetrievalEngine,
    *,
    provider: str | None = None,
) -> DailyAnalysis:
    """执行日报分析：规则检测 + 可选 LLM 总结。"""
    ctx.news_records = _filter_news_by_date(ctx.news_records, ctx.report_date)
    if not ctx.news_records:
        logger.warning("报告日 %s 无匹配新闻，使用全部新闻做分析", ctx.report_date)
        # keep all records loaded by fetch step

    analysis = detect_events(ctx.news_records)
    analysis.summary = _rule_based_summary(ctx, analysis)

    if not ctx.enable_llm:
        return analysis

    context = _retrieve_watchlist_context(engine, ctx.report_date)
    event_lines = []
    for item in analysis.major_events[:5] + analysis.bullish[:3] + analysis.bearish[:3]:
        event_lines.append(f"- [{item.sentiment}] {item.title}")

    user_prompt = build_daily_summary_prompt(
        report_date=ctx.report_date,
        event_lines=event_lines,
        context=context,
    )

    try:
        client = create_llm_client(provider=provider or os.getenv("LLM_PROVIDER"), allow_preview=True)
        if client.preview_mode:
            logger.info("LLM 未配置，使用规则引擎摘要")
            return analysis

        result = client.generate_with_metadata(
            SYSTEM_PROMPT,
            user_prompt,
            temperature=FINANCIAL_LLM_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        try:
            payload = _parse_llm_json(result.answer)
            analysis = _merge_llm_analysis(analysis, payload)
        except json.JSONDecodeError:
            analysis.summary = result.answer.strip() or analysis.summary
            analysis.llm_used = True
    except LLMClientError as exc:
        logger.warning("LLM 分析失败，回退规则摘要: %s", exc)

    return analysis
