"""
Agent 类型定义：Pipeline 上下文、检索计划、步骤协议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.llm.llm_client import LLMGenerateResult
from src.vectorstore.retrieval import RetrievalResult


@dataclass(frozen=True)
class SourceSpec:
    """数据源规格，便于后续注册 Wind / 同花顺 / 雪球等。"""

    source_id: str
    label: str
    chroma_source: str
    enabled: bool = True
    description: str = ""


@dataclass(frozen=True)
class IntentSpec:
    """问题意图规格。"""

    intent_id: str
    label: str
    source_ids: tuple[str, ...]
    keywords: tuple[str, ...] = ()
    description: str = ""


@dataclass
class QueryIntent:
    """Planner 识别出的单条意图。"""

    intent_id: str
    label: str
    confidence: float
    source_ids: list[str]


@dataclass
class RetrievalPlan:
    """Planner 输出的检索计划。"""

    question: str
    retrieval_query: str
    primary_intent: QueryIntent
    intents: list[QueryIntent]
    source_filters: list[str]
    top_k: int
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_comprehensive(self) -> bool:
        return self.primary_intent.intent_id == "comprehensive" or len(self.source_filters) > 1


@dataclass
class AgentContext:
    """Agent Pipeline 运行时上下文。"""

    question: str
    retrieval_query: str = ""
    plan: RetrievalPlan | None = None
    results: list[RetrievalResult] = field(default_factory=list)
    results_by_source: dict[str, list[RetrievalResult]] = field(default_factory=dict)
    merged_context: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    answer: str = ""
    llm_result: LLMGenerateResult | None = None
    top_k: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """转为与 05_chat 兼容的 payload 结构。"""
        llm_result = self.llm_result
        return {
            "question": self.question,
            "retrieval_query": self.retrieval_query or self.question,
            "plan": self.plan,
            "results": self.results,
            "results_by_source": self.results_by_source,
            "context": self.merged_context,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "full_prompt": f"【System】\n{self.system_prompt}\n\n{self.user_prompt}",
            "answer": self.answer,
            "llm_result": llm_result,
            "provider": llm_result.provider if llm_result else "",
            "model": llm_result.model if llm_result else "",
            "top_k": self.top_k,
            "metadata": self.metadata,
        }


class PipelineStep(Protocol):
    """Pipeline 步骤协议。"""

    name: str

    def run(self, ctx: AgentContext) -> AgentContext:
        """执行步骤并返回更新后的上下文。"""
