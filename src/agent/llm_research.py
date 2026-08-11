"""
可选 LLM 投研报告编排：基于已有指标/估值/新闻/RAG 上下文生成专业报告。

不改动规则估值与检索核心逻辑；无 API Key 时直接跳过。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from config import AGENT_ENABLE_LLM, setup_logging
from src.agent.prompts.financial_prompt import build_financial_analysis_prompt
from src.agent.prompts.news_prompt import build_news_analysis_prompt
from src.agent.prompts.report_prompt import build_final_report_prompt
from src.agent.prompts.system_prompt import FINANCIAL_LLM_TEMPERATURE, SYSTEM_PROMPT
from src.agent.prompts.valuation_prompt import build_valuation_analysis_prompt
from src.llm.llm_client import LLMClient, LLMClientError, create_llm_client

logger = setup_logging(__name__)

JSON_RESPONSE_FORMAT = {"type": "json_object"}


@dataclass
class LLMResearchBundle:
    """LLM 增强投研结果（可选）。"""

    enabled: bool = False
    financial_analysis: dict[str, Any] = field(default_factory=dict)
    valuation_analysis: dict[str, Any] = field(default_factory=dict)
    news_analysis: dict[str, Any] = field(default_factory=dict)
    final_report: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "financial_analysis": self.financial_analysis,
            "valuation_analysis": self.valuation_analysis,
            "news_analysis": self.news_analysis,
            "final_report": self.final_report,
            "error": self.error,
        }


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}


def _generate_json(
    client: LLMClient,
    user_prompt: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    result = client.generate_with_metadata(
        system_prompt,
        user_prompt,
        temperature=FINANCIAL_LLM_TEMPERATURE,
        response_format=JSON_RESPONSE_FORMAT,
    )
    return _parse_json_object(result.answer)


def generate_llm_research_report(
    *,
    company: str,
    financial_metrics: str | dict[str, Any] = "",
    financial_documents: str = "",
    pe: Any = None,
    pb: Any = None,
    industry_data: str | dict[str, Any] = "",
    growth_data: str | dict[str, Any] = "",
    news: str | list[Any] = "",
    context: str = "",
    company_info: str | dict[str, Any] | None = None,
    client: LLMClient | None = None,
    provider: str | None = None,
) -> LLMResearchBundle:
    """
    依次调用财务 / 估值 / 新闻 / 终稿 Prompt，生成专业研究报告。

    无 Key、Preview 模式或调用失败时：enabled=False，不抛出到上层。
    """
    if not AGENT_ENABLE_LLM:
        return LLMResearchBundle(error="LLM 已禁用（FINANCIAL_POC_AGENT_ENABLE_LLM=false）")

    try:
        llm = client or create_llm_client(provider=provider, allow_preview=True)
    except LLMClientError as exc:
        return LLMResearchBundle(error=str(exc))

    if llm.preview_mode:
        return LLMResearchBundle(error="未配置有效 API Key，跳过 LLM 研报增强")

    bundle = LLMResearchBundle(enabled=True)
    info = company_info or {"company": company}

    try:
        bundle.financial_analysis = _generate_json(
            llm,
            build_financial_analysis_prompt(
                company=company,
                financial_metrics=financial_metrics,
                financial_documents=financial_documents,
            ),
        )
        bundle.valuation_analysis = _generate_json(
            llm,
            build_valuation_analysis_prompt(
                company=company,
                pe=pe,
                pb=pb,
                industry_data=industry_data,
                growth_data=growth_data,
            ),
        )
        bundle.news_analysis = _generate_json(
            llm,
            build_news_analysis_prompt(news=news),
        )
        report_result = llm.generate_with_metadata(
            SYSTEM_PROMPT,
            build_final_report_prompt(
                company_info=info,
                financial_analysis=bundle.financial_analysis,
                valuation_analysis=bundle.valuation_analysis,
                news_analysis=bundle.news_analysis,
                context=context,
            ),
            temperature=FINANCIAL_LLM_TEMPERATURE,
        )
        bundle.final_report = (report_result.answer or "").strip()
    except Exception as exc:
        logger.warning("LLM 投研报告生成失败: %s", exc)
        bundle.error = str(exc)
        bundle.enabled = bool(bundle.final_report or bundle.financial_analysis)

    return bundle


def generate_llm_research_from_analysis(
    analysis: Any,
    *,
    context: str = "",
    client: LLMClient | None = None,
    provider: str | None = None,
) -> LLMResearchBundle:
    """
    基于规则分析结果（FullAnalysisResult）调用 Prompt 体系生成 LLM 研报。

    不修改 analysis 本身；仅消费其已有字段。
    """
    company = str(getattr(analysis, "entity_name", "") or "")
    entity_id = str(getattr(analysis, "entity_id", "") or "")
    fund = getattr(analysis, "fundamentals", None) or {}
    valuation = getattr(analysis, "valuation", None)
    comparison = getattr(analysis, "comparison", None)

    pe = None
    pb = None
    if valuation is not None:
        market = getattr(valuation, "market", None) or {}
        if isinstance(market, dict):
            pe = market.get("pe_ttm")
            pb = market.get("pb")
    if comparison is not None and getattr(comparison, "target", None) is not None:
        pe = pe if pe is not None else comparison.target.pe_ttm
        pb = pb if pb is not None else comparison.target.pb

    industry_data: dict[str, Any] = {}
    if comparison is not None:
        industry_data = {
            "industry": getattr(comparison, "industry", ""),
            "relative_verdict": getattr(comparison, "relative_verdict", ""),
            "news_sentiment": getattr(comparison, "news_sentiment", ""),
        }
        stats = getattr(comparison, "industry_stats", None)
        if stats is not None:
            industry_data["avg_pe"] = getattr(stats, "avg_pe", None)
            industry_data["avg_pb"] = getattr(stats, "avg_pb", None)
            industry_data["peer_count"] = getattr(stats, "peer_count", None)

    growth_data = {
        "profit_growth_pct": fund.get("profit_growth_pct"),
        "revenue_growth_pct": fund.get("revenue_growth_pct"),
        "rule_verdict": getattr(analysis, "final_verdict", ""),
        "rule_score": getattr(analysis, "final_score", None),
        "rule_reasons": list(getattr(analysis, "synthesis_reasons", None) or [])[:8],
    }

    news_items: list[Any] = []
    if comparison is not None:
        news_items = list(getattr(comparison, "news_items", None) or [])[:10]

    docs = context or ""
    if not docs and getattr(analysis, "executive_summary", ""):
        docs = str(analysis.executive_summary)

    company_info = {
        "company": company,
        "entity_id": entity_id,
        "period_label": getattr(analysis, "period_label", ""),
        "keywords": list(getattr(analysis, "keywords", None) or [])[:12],
    }

    return generate_llm_research_report(
        company=company,
        financial_metrics=fund if isinstance(fund, dict) else {},
        financial_documents=docs,
        pe=pe,
        pb=pb,
        industry_data=industry_data,
        growth_data=growth_data,
        news=news_items,
        context=docs,
        company_info=company_info,
        client=client,
        provider=provider,
    )


def append_llm_research_report(report_text: str, analysis: Any) -> str:
    """若 LLM 可用则追加专业研报段落；否则原样返回。"""
    bundle = generate_llm_research_from_analysis(analysis)
    if not bundle.enabled or not bundle.final_report:
        if bundle.error:
            logger.info("跳过 LLM 研报增强: %s", bundle.error)
        return report_text
    return (
        f"{report_text}\n\n"
        f"{'─' * 58}\n"
        f"【LLM 投研报告】\n\n"
        f"{bundle.final_report}\n\n"
        f"免责声明: 由可选 LLM 基于已提供数据生成，不构成投资建议"
    )
