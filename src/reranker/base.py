"""
Reranker 抽象接口与通用工具。

Retrieval 层只依赖本模块的 Protocol，不直接耦合具体 CrossEncoder 实现。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """重排序器协议：输入 query + 文档列表，返回 (原始下标, 分数) 排序结果。"""

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """对候选文档重排序。"""
        ...


def apply_rerank_to_results(
    reranker: Reranker,
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    text_key: str = "text",
    score_key: str = "rerank_score",
) -> list[dict[str, Any]]:
    """
    对检索候选应用 Reranker，返回带 rerank_score 的新结果列表。

    Args:
        reranker: 任意实现 Reranker 协议的对象。
        query: 用户问题。
        candidates: 检索候选，每项需包含 text_key 字段。
        top_k: 最终返回条数。
        text_key: 正文字段名。
        score_key: 写入 rerank 分数的字段名。

    Returns:
        重排后的候选列表（保留原始 score/similarity，附加 score_key）。
    """
    if not candidates:
        return []

    documents = [str(item.get(text_key) or "") for item in candidates]
    ranked = reranker.rerank(query, documents, top_k=top_k)

    reranked: list[dict[str, Any]] = []
    for source_index, rerank_score in ranked:
        item = dict(candidates[source_index])
        item[score_key] = rerank_score
        reranked.append(item)
    return reranked
