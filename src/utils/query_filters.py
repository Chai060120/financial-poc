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

    asks_metric = any(keyword in query for keyword in _FINANCIAL_METRIC_KEYWORDS)
    has_table_hint = any(keyword in query for keyword in _TABLE_HINT_KEYWORDS)

    if asks_metric and not has_table_hint:
        extras.extend(["主要会计数据", "财务指标"])

    if re.search(r"20\d{2}", query) and "年报" not in query and "半年报" not in query:
        if asks_metric:
            extras.append("年度报告")

    if "归属于上市公司" in query and "股东" in query and "净利润" in query:
        extras.append("归属于上市公司股东的净利润")

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
