"""
Planner：根据用户问题规划检索意图与数据源。

采用可插拔 IntentClassifier，默认基于注册表关键词打分，不写死业务分支。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import setup_logging
from src.agent.registry import (
    INTENT_REGISTRY,
    get_enabled_sources,
    intent_source_ids,
    resolve_chroma_sources,
)
from src.agent.types import QueryIntent, RetrievalPlan
from src.chat.history import ConversationHistory
from src.chat.memory import ConversationMemory

logger = setup_logging(__name__)

COMPREHENSIVE_INTENT_ID = "comprehensive"
COMPREHENSIVE_THRESHOLD = 0.15


class IntentClassifier(Protocol):
    """意图分类器协议，可替换为 LLM / 模型分类器。"""

    def classify(self, question: str) -> list[QueryIntent]:
        """返回按 confidence 降序排列的意图列表。"""


class RuleBasedIntentClassifier:
    """基于 INTENT_REGISTRY 关键词的规则分类器。"""

    def __init__(self, *, comprehensive_threshold: float = COMPREHENSIVE_THRESHOLD) -> None:
        self.comprehensive_threshold = comprehensive_threshold

    def classify(self, question: str) -> list[QueryIntent]:
        text = question.strip().lower()
        if not text:
            return []

        scored: list[QueryIntent] = []
        for intent_id, spec in INTENT_REGISTRY.items():
            if not intent_source_ids(intent_id) and intent_id != COMPREHENSIVE_INTENT_ID:
                continue

            hits = sum(1 for keyword in spec.keywords if keyword.lower() in text)
            if hits == 0 and intent_id != COMPREHENSIVE_INTENT_ID:
                continue

            keyword_total = max(len(spec.keywords), 1)
            confidence = hits / keyword_total
            if intent_id == COMPREHENSIVE_INTENT_ID and hits > 0:
                confidence = min(1.0, confidence + 0.2)

            scored.append(
                QueryIntent(
                    intent_id=intent_id,
                    label=spec.label,
                    confidence=confidence,
                    source_ids=intent_source_ids(intent_id)
                    or [item.source_id for item in get_enabled_sources()],
                )
            )

        scored.sort(key=lambda item: item.confidence, reverse=True)
        return scored


class Planner:
    """
    检索 Planner：Question → RetrievalPlan。

    不直接调用 Retriever，仅输出计划供 Workflow 执行。
    """

    def __init__(self, classifier: IntentClassifier | None = None) -> None:
        self.classifier = classifier or RuleBasedIntentClassifier()

    def plan(
        self,
        question: str,
        *,
        top_k: int,
        memory: ConversationHistory | ConversationMemory | None = None,
    ) -> RetrievalPlan:
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        query = question.strip()
        retrieval_query = (
            memory.build_retrieval_query(query) if memory is not None else query
        )

        intents = self.classifier.classify(query)
        primary, source_filters, reasoning = self._select_strategy(query, intents)

        plan = RetrievalPlan(
            question=query,
            retrieval_query=retrieval_query,
            primary_intent=primary,
            intents=intents,
            source_filters=source_filters,
            top_k=top_k,
            reasoning=reasoning,
        )

        logger.info(
            "Planner: intent=%s (%s), sources=%s, query=%r",
            primary.intent_id,
            primary.label,
            source_filters,
            retrieval_query[:80],
        )
        logger.info("Planner reasoning: %s", reasoning)
        return plan

    def _select_strategy(
        self,
        question: str,
        intents: list[QueryIntent],
    ) -> tuple[QueryIntent, list[str], str]:
        enabled_sources = resolve_chroma_sources(
            [item.source_id for item in get_enabled_sources()]
        )

        if not intents:
            fallback = QueryIntent(
                intent_id=COMPREHENSIVE_INTENT_ID,
                label=INTENT_REGISTRY[COMPREHENSIVE_INTENT_ID].label,
                confidence=0.0,
                source_ids=[item.source_id for item in get_enabled_sources()],
            )
            return (
                fallback,
                enabled_sources,
                "未匹配到明确意图，默认综合分析全部已启用数据源",
            )

        top = intents[0]
        second = intents[1] if len(intents) > 1 else None

        if top.intent_id == COMPREHENSIVE_INTENT_ID:
            return (
                top,
                enabled_sources,
                f"识别为「{top.label}」，检索全部已启用数据源",
            )

        if second and abs(top.confidence - second.confidence) <= COMPREHENSIVE_THRESHOLD:
            merged_sources = resolve_chroma_sources(
                list(dict.fromkeys(top.source_ids + second.source_ids))
            )
            primary = QueryIntent(
                intent_id=COMPREHENSIVE_INTENT_ID,
                label=INTENT_REGISTRY[COMPREHENSIVE_INTENT_ID].label,
                confidence=max(top.confidence, second.confidence),
                source_ids=[item.source_id for item in get_enabled_sources()],
            )
            return (
                primary,
                merged_sources or enabled_sources,
                f"「{top.label}」与「{second.label}」置信度接近，升级为综合分析",
            )

        sources = resolve_chroma_sources(top.source_ids)
        if not sources:
            sources = enabled_sources

        return (
            top,
            sources,
            f"识别为「{top.label}」，检索数据源: {', '.join(sources)}",
        )


def main() -> None:
    planner = Planner()
    samples = [
        "招商银行净利润怎么样？",
        "最近有什么新闻？",
        "请综合分析一下贵州茅台的基本面和近期新闻",
    ]
    for question in samples:
        plan = planner.plan(question, top_k=5)
        print("---")
        print("Q:", question)
        print("Intent:", plan.primary_intent.label, plan.primary_intent.intent_id)
        print("Sources:", plan.source_filters)
        print("Reason:", plan.reasoning)


if __name__ == "__main__":
    main()
