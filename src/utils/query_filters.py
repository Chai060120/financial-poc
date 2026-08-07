"""
检索过滤与查询增强：实体自动识别、财务指标问句改写。
"""

from __future__ import annotations

import re

from src.utils.entity_parser import detect_query_entity

_FINANCIAL_METRIC_KEYWORDS: tuple[str, ...] = (
    "净利润",
    "营业收入",
    "营业总收入",
    "营收",
    "毛利",
    "每股收益",
    "净资产",
    "总资产",
    "资产负债",
    "现金流",
    "分红",
    "roe",
    "ROE",
)

_TABLE_HINT_KEYWORDS: tuple[str, ...] = (
    "主要会计数据",
    "财务指标",
    "利润表",
    "资产负债表",
    "现金流量表",
)

_QUARTERLY_HINTS: tuple[str, ...] = (
    "分季度",
    "第一季度",
    "第二季度",
    "第三季度",
    "第四季度",
    "(1-3",
    "(4-6",
    "(7-9",
    "(10-12",
)

_ANNUAL_HINTS: tuple[str, ...] = (
    "主要会计数据",
    "近三年主要",
    "主要财务指标",
    "主要会计数据和财务指标",
    "利润表及现金流量表",
    "合并利润表",
    "1—12 月",
    "1-12 月",
)

_SECTION_QUARTERLY = "分季度财务"
_SECTION_ANNUAL = "主要会计数据"


def asks_full_year_period(question: str) -> bool:
    """问句是否在要全年/年报口径（而非单季度或半年报）。"""
    text = str(question or "")
    if any(
        token in text
        for token in (
            "季度",
            "一季度",
            "二季度",
            "三季度",
            "四季度",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "半年报",
            "1-6月",
            "1—6月",
            "上半年",
            "下半年",
        )
    ):
        return False
    return bool(re.search(r"20\d{2}年", text) or "年报" in text or "全年" in text)


def asks_financial_metric(question: str) -> bool:
    return any(keyword in str(question or "") for keyword in _FINANCIAL_METRIC_KEYWORDS)


def is_quarterly_chunk(text: str, metadata: dict | None = None) -> bool:
    meta = metadata or {}
    section = str(meta.get("section") or "")
    if section == _SECTION_QUARTERLY:
        return True
    body = str(text or "")
    return any(hint in body for hint in _QUARTERLY_HINTS)


def is_annual_summary_chunk(text: str, metadata: dict | None = None) -> bool:
    meta = metadata or {}
    section = str(meta.get("section") or "")
    if section == _SECTION_ANNUAL:
        return True
    body = str(text or "")
    if is_quarterly_chunk(body, meta):
        return False
    return any(hint in body for hint in _ANNUAL_HINTS)


def resolve_entity_filters(
    question: str,
    *,
    entity_name: str | None = None,
    entity_id: str | None = None,
    auto_entity: bool = True,
) -> tuple[str | None, str | None, bool]:
    """
    合并显式过滤与问题中的自动实体识别。

    Returns:
        (entity_name, entity_id, auto_applied)
    """
    explicit_name = str(entity_name or "").strip()
    explicit_id = str(entity_id or "").strip()

    if explicit_name or explicit_id:
        return explicit_name or None, explicit_id or None, False

    if not auto_entity:
        return None, None, False

    detected_name, detected_id = detect_query_entity(question)
    if detected_name or detected_id:
        return detected_name or None, detected_id or None, True

    return None, None, False


def enhance_retrieval_query(question: str) -> str:
    """
    在不改变用户可见问题的前提下，为检索追加财务表相关关键词。

    例如「贵州茅台2024年净利润」→ 追加「主要会计数据 财务指标」，
    提高 BM25 / Rerank 对摘要表与利润表的命中率。
    """
    query = str(question or "").strip()
    if not query:
        return query

    extras: list[str] = []
    lower_query = query.lower()

    asks_metric = asks_financial_metric(query)
    has_table_hint = any(keyword in query for keyword in _TABLE_HINT_KEYWORDS)
    full_year = asks_full_year_period(query)

    if asks_metric and not has_table_hint:
        extras.extend(["主要会计数据", "财务指标"])

    if full_year and asks_metric:
        extras.extend(
            [
                "近三年主要会计数据",
                "合并利润表",
                "全年",
                "1-12月",
            ]
        )

    if re.search(r"20\d{2}", query) and "年报" not in query and "半年报" not in query:
        if asks_metric:
            extras.append("年度报告")

    if "归属于上市公司" in query and "股东" in query and "净利润" in query:
        extras.append("归属于上市公司股东的净利润")

    if full_year and "营业收入" in query:
        extras.append("营业总收入")

    if not extras:
        return query

    merged = f"{query} {' '.join(dict.fromkeys(extras))}"
    return merged.strip()


def describe_retrieval_mode(*, hybrid: bool, rerank: bool) -> str:
    """返回可读检索模式标签。"""
    if hybrid and rerank:
        return "hybrid+rerank"
    if hybrid:
        return "hybrid"
    if rerank:
        return "vector+rerank"
    return "vector"
