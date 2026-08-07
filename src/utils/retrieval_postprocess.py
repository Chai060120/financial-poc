"""
检索结果后处理：全年口径问句时，优先全年摘要片段、降低分季度片段排序。
"""

from __future__ import annotations

from typing import Any, TypeVar

from src.utils.query_filters import (
    asks_full_year_period,
    is_annual_summary_chunk,
    is_quarterly_chunk,
)

T = TypeVar("T", bound=dict[str, Any])


def _chunk_boost(item: dict[str, Any]) -> float:
    text = str(item.get("text") or "")
    meta = item.get("metadata") or {}
    score = 0.0

    if is_annual_summary_chunk(text, meta):
        score += 100.0
    if is_quarterly_chunk(text, meta):
        score -= 90.0

    rerank = item.get("rerank_score")
    if rerank is not None:
        score += float(rerank) * 10.0

    base = item.get("score", item.get("similarity"))
    if base is not None:
        score += float(base) * 100.0

    return score


def postprocess_retrieval_results(
    question: str,
    results: list[T],
    *,
    top_k: int | None = None,
) -> list[T]:
    """
    按问句时间口径重排检索结果。

    全年/年报问句：主要会计数据、利润表等优先；分季度表沉底。
    """
    if not results:
        return results

    if not asks_full_year_period(question):
        if top_k is not None:
            return results[:top_k]
        return results

    ranked = sorted(
        enumerate(results),
        key=lambda pair: (_chunk_boost(pair[1]), -pair[0]),
        reverse=True,
    )
    reordered = [item for _, item in ranked]

    annual: list[T] = []
    other: list[T] = []
    quarterly: list[T] = []

    for item in reordered:
        text = str(item.get("text") or "")
        meta = item.get("metadata") or {}
        if is_annual_summary_chunk(text, meta):
            annual.append(item)
        elif is_quarterly_chunk(text, meta):
            quarterly.append(item)
        else:
            other.append(item)

    merged = annual + other + quarterly
    if top_k is not None:
        return merged[:top_k]
    return merged


def full_year_fetch_size(top_k: int) -> int:
    """全年问句多拉一些候选，供后处理筛选。"""
    return max(top_k * 4, top_k + 10, 15)
