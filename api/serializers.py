"""API 响应序列化工具。"""

from __future__ import annotations

from typing import Any

from src.agent.types import RetrievalPlan
from src.vectorstore.retrieval import RetrievalResult


def serialize_retrieval_result(item: RetrievalResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, dict):
        meta = dict(item.get("metadata") or {})
        return {
            "id": str(item.get("id") or ""),
            "text": str(item.get("text") or ""),
            "score": float(item.get("score") or item.get("similarity") or 0.0),
            "source": str(meta.get("source") or ""),
            "metadata": meta,
            "rerank_score": item.get("rerank_score"),
            "rrf_score": item.get("rrf_score"),
        }

    meta = dict(item.get("metadata") or {})
    return {
        "id": item["id"],
        "text": item["text"],
        "score": float(item.get("score") or item.get("similarity") or 0.0),
        "source": str(meta.get("source") or ""),
        "metadata": meta,
        "rerank_score": item.get("rerank_score"),
        "rrf_score": item.get("rrf_score"),
    }


def serialize_plan(plan: RetrievalPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "intent_id": plan.primary_intent.intent_id,
        "intent_label": plan.primary_intent.label,
        "source_filters": list(plan.source_filters),
        "reasoning": plan.reasoning,
    }
