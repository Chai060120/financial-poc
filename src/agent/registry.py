"""
Financial Agent 注册表：仅 PDF 财报 + 财经新闻。
"""

from __future__ import annotations

from src.agent.types import IntentSpec, SourceSpec

SOURCE_REGISTRY: dict[str, SourceSpec] = {
    "pdf": SourceSpec(
        source_id="pdf",
        label="财报",
        chroma_source="pdf",
        enabled=True,
        description="PDF 财报、定期报告",
    ),
    "news": SourceSpec(
        source_id="news",
        label="新闻",
        chroma_source="news",
        enabled=True,
        description="财经新闻、资讯",
    ),
}

INTENT_REGISTRY: dict[str, IntentSpec] = {
    "financial_report": IntentSpec(
        intent_id="financial_report",
        label="财报",
        source_ids=("pdf",),
        keywords=(
            "财报", "年报", "半年报", "季报", "三季报", "一季报",
            "利润", "净利润", "营收", "收入", "毛利率", "净资产",
            "资产负债表", "现金流量", "roe", "eps", "业绩",
            "财务", "报表", "审计", "合并报表",
        ),
        description="财务报表、经营指标类问题",
    ),
    "news": IntentSpec(
        intent_id="news",
        label="新闻",
        source_ids=("news",),
        keywords=(
            "新闻", "报道", "资讯", "消息", "媒体", "快讯",
            "头条", "传闻", "动态", "热点", "舆情",
        ),
        description="新闻资讯、市场动态类问题",
    ),
    "comprehensive": IntentSpec(
        intent_id="comprehensive",
        label="综合分析",
        source_ids=("pdf", "news"),
        keywords=(
            "综合", "全面", "对比", "比较", "分析", "研判",
            "怎么样", "如何看", "整体", "概况", "总结", "评价",
        ),
        description="财报 + 新闻综合分析",
    ),
}


def register_source(spec: SourceSpec) -> None:
    SOURCE_REGISTRY[spec.source_id] = spec


def register_intent(spec: IntentSpec) -> None:
    INTENT_REGISTRY[spec.intent_id] = spec


def get_enabled_sources() -> list[SourceSpec]:
    return [spec for spec in SOURCE_REGISTRY.values() if spec.enabled]


def resolve_chroma_sources(source_ids: list[str]) -> list[str]:
    values: list[str] = []
    for source_id in source_ids:
        spec = SOURCE_REGISTRY.get(source_id)
        if spec is None or not spec.enabled:
            continue
        if spec.chroma_source not in values:
            values.append(spec.chroma_source)
    return values


def intent_source_ids(intent_id: str) -> list[str]:
    spec = INTENT_REGISTRY.get(intent_id)
    if spec is None:
        return []
    return [
        source_id
        for source_id in spec.source_ids
        if source_id in SOURCE_REGISTRY and SOURCE_REGISTRY[source_id].enabled
    ]
