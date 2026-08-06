"""
Agent Workflow：Question → Planner → Retriever → Merge Context → LLM → Final Answer

Pipeline 步骤可组合、可替换，不写死业务逻辑。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import RERANK_TOP_K, setup_logging
from src.agent.planner import Planner
from src.agent.registry import SOURCE_REGISTRY
from src.agent.types import AgentContext, PipelineStep, RetrievalPlan
from src.chat.history import ConversationHistory
from src.chat.memory import ConversationMemory
from src.llm.llm_client import LLMClient, LLMGenerateResult, create_llm_client
from src.utils.source_display import format_reference_meta, source_type_label
from src.vectorstore.retrieval import RetrievalError, RetrievalResult
from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine, create_retrieval_engine

logger = setup_logging(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "你是金融信息处理 Agent，只使用两类资料回答用户问题："
    "（1）PDF 财报、定期报告；（2）财经新闻资讯。"
    "请严格基于提供的参考资料作答，不要编造资料中不存在的数据。"
    "引用时请区分来源是「财报」还是「新闻」。"
    "若用户问题省略了公司或指标主体，请结合对话历史理解其指代。"
    "若资料中出现分季度财务数据而用户询问全年指标，请将各季度数值加总后作答。"
    "若参考资料不足以回答，请明确说明。"
)


class PlanStep:
    """Step 1: Planner 生成检索计划。"""

    name = "plan"

    def __init__(self, planner: Planner | None = None) -> None:
        self.planner = planner or Planner()

    def run(self, ctx: AgentContext) -> AgentContext:
        memory = ctx.metadata.get("memory")
        plan = self.planner.plan(
            ctx.question,
            top_k=ctx.top_k,
            memory=memory if isinstance(memory, ConversationHistory) else None,
        )
        return replace(
            ctx,
            plan=plan,
            retrieval_query=plan.retrieval_query,
        )


class RetrieveStep:
    """Step 2: 按 Plan 执行 Retriever（支持多源并行检索）。"""

    name = "retrieve"

    def __init__(self, engine: UnifiedRetrievalEngine | None = None) -> None:
        self.engine = engine

    def run(self, ctx: AgentContext) -> AgentContext:
        if ctx.plan is None:
            raise RetrievalError("RetrieveStep 需要 Planner 输出的 RetrievalPlan")

        engine = self.engine or ctx.metadata.get("retrieval_engine")
        if engine is None:
            engine = create_retrieval_engine(top_k=ctx.top_k)
        elif not hasattr(engine, "retrieve"):
            engine = create_retrieval_engine(top_k=ctx.top_k)

        results, results_by_source = retrieve_by_plan(engine, ctx.plan)
        return replace(ctx, results=results, results_by_source=results_by_source)


def retrieve_by_plan(
    engine: UnifiedRetrievalEngine,
    plan: RetrievalPlan,
) -> tuple[list[RetrievalResult], dict[str, list[RetrievalResult]]]:
    """根据 RetrievalPlan 执行检索并合并结果。"""
    query = plan.retrieval_query
    top_k = plan.top_k
    sources = plan.source_filters

    if not sources:
        batch = engine.retrieve(query, top_k=top_k)
        grouped = _group_results_by_source(batch)
        return batch, grouped

    if len(sources) == 1 and not plan.is_comprehensive:
        batch = engine.retrieve(query, top_k=top_k, where={"source": sources[0]})
        return batch, {sources[0]: batch}

    per_source_k = max(1, top_k // len(sources))
    grouped: dict[str, list[RetrievalResult]] = {}
    merged: list[RetrievalResult] = []

    for source in sources:
        try:
            batch = engine.retrieve(
                query,
                top_k=per_source_k,
                where={"source": source},
            )
        except RetrievalError:
            logger.warning("数据源检索失败，已跳过: source=%s", source)
            batch = []
        grouped[source] = batch
        merged.extend(batch)

    merged = _dedupe_and_rank(merged)[:top_k]
    return merged, grouped


def _group_results_by_source(results: list[RetrievalResult]) -> dict[str, list[RetrievalResult]]:
    grouped: dict[str, list[RetrievalResult]] = {}
    for item in results:
        source = str(item["metadata"].get("source") or "unknown")
        grouped.setdefault(source, []).append(item)
    return grouped


def _dedupe_and_rank(results: list[RetrievalResult]) -> list[RetrievalResult]:
    seen: set[str] = set()
    unique: list[RetrievalResult] = []
    for item in sorted(
        results,
        key=lambda row: (
            row.get("rerank_score") if row.get("rerank_score") is not None else row["score"]
        ),
        reverse=True,
    ):
        token_id = item["id"]
        if token_id in seen:
            continue
        seen.add(token_id)
        unique.append(item)
    return unique


class MergeContextStep:
    """Step 3: 合并多源检索结果为 LLM 上下文。"""

    name = "merge_context"

    def run(self, ctx: AgentContext) -> AgentContext:
        merged = merge_retrieval_context(
            ctx.results,
            results_by_source=ctx.results_by_source,
            plan=ctx.plan,
        )
        return replace(ctx, merged_context=merged)


def merge_retrieval_context(
    results: list[RetrievalResult],
    *,
    results_by_source: dict[str, list[RetrievalResult]] | None = None,
    plan: RetrievalPlan | None = None,
) -> str:
    """将检索结果按来源分组并格式化为 Prompt 上下文。"""
    if not results:
        return "（未检索到相关参考资料）"

    sections: list[str] = []
    intent_label = plan.primary_intent.label if plan else "参考资料"
    sections.append(f"【检索意图: {intent_label}】")

    if results_by_source and len(results_by_source) > 1:
        global_index = 1
        for source, batch in results_by_source.items():
            if not batch:
                continue
            label = _source_section_label(source)
            blocks: list[str] = [f"## {label}"]
            for item in batch:
                blocks.append(_format_reference_block(global_index, item))
                global_index += 1
            sections.append("\n".join(blocks))
    else:
        blocks = ["## 参考资料"]
        for index, item in enumerate(results, start=1):
            blocks.append(_format_reference_block(index, item))
        sections.append("\n".join(blocks))

    return "\n\n".join(sections)


def _source_section_label(source: str) -> str:
    spec = SOURCE_REGISTRY.get(source)
    if spec:
        return f"{spec.label}（{source}）"
    return source_type_label(source)


def _format_reference_block(index: int, item: RetrievalResult) -> str:
    meta = item["metadata"]
    score = item.get("score", item["similarity"])
    rerank_score = item.get("rerank_score")
    header_parts = [f"[{index}]"] + format_reference_meta(meta)
    header_parts.append(f"score: {score:.4f}")
    if rerank_score is not None:
        header_parts.append(f"rerank_score: {rerank_score:.4f}")
    return f"{', '.join(header_parts)}\n{item['text']}"


class GenerateStep:
    """Step 4: 调用 LLM 生成最终回答。"""

    name = "generate"

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def run(self, ctx: AgentContext) -> AgentContext:
        client = self.llm or ctx.metadata.get("llm_client")
        if not isinstance(client, LLMClient):
            provider = ctx.metadata.get("provider")
            client = create_llm_client(provider=provider, allow_preview=True)

        user_prompt = build_agent_user_prompt(
            ctx.question,
            ctx.merged_context,
            memory=ctx.metadata.get("memory"),
            plan=ctx.plan,
        )
        llm_result = client.generate_with_metadata(self.system_prompt, user_prompt)

        memory = ctx.metadata.get("memory")
        if isinstance(memory, (ConversationMemory, ConversationHistory)):
            memory.add(ctx.question, llm_result.answer)

        return replace(
            ctx,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            answer=llm_result.answer,
            llm_result=llm_result,
        )


def build_agent_user_prompt(
    question: str,
    context: str,
    *,
    memory: ConversationMemory | ConversationHistory | None = None,
    plan: RetrievalPlan | None = None,
) -> str:
    plan_block = ""
    if plan is not None:
        plan_block = (
            f"【Agent 计划】\n"
            f"意图: {plan.primary_intent.label} ({plan.primary_intent.intent_id})\n"
            f"数据源: {', '.join(plan.source_filters) or '全部'}\n"
            f"说明: {plan.reasoning}"
        )

    if memory is not None and hasattr(memory, "build_prompt_context"):
        return memory.build_prompt_context(
            question,
            references=context,
            plan_block=plan_block,
        )

    parts: list[str] = []
    if plan_block:
        parts.append(plan_block)

    if memory is not None and len(memory) > 0:
        parts.append(
            "【对话历史】\n"
            f"{memory.format_history()}\n\n"
            "请结合对话历史理解当前问题的指代。"
        )

    parts.extend(
        [
            f"【参考资料】\n{context}\n",
            f"【用户问题】\n{question.strip()}\n",
            "【回答要求】\n请用中文回答，引用资料时标注序号（如 [1]）。",
        ]
    )
    return "\n".join(parts)


class MemoryUpdateStep:
    """可选步骤：显式标记 Memory 更新（GenerateStep 已写入时可跳过）。"""

    name = "memory"

    def run(self, ctx: AgentContext) -> AgentContext:
        return ctx


class AgentWorkflow:
    """
    Information Processing Agent 主工作流。

    默认 Pipeline:
        Plan → Retrieve → Merge Context → Generate
    """

    def __init__(
        self,
        steps: list[PipelineStep] | None = None,
        *,
        planner: Planner | None = None,
        engine: UnifiedRetrievalEngine | None = None,
        llm: LLMClient | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.steps = steps or [
            PlanStep(planner),
            RetrieveStep(engine),
            MergeContextStep(),
            GenerateStep(llm, system_prompt=system_prompt),
        ]

    def run(
        self,
        question: str,
        *,
        top_k: int = RERANK_TOP_K,
        memory: ConversationHistory | ConversationMemory | None = None,
        engine: UnifiedRetrievalEngine | None = None,
        llm: LLMClient | None = None,
        provider: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentContext:
        ctx = AgentContext(
            question=question,
            top_k=top_k,
            metadata=dict(metadata or {}),
        )
        if memory is not None:
            ctx.metadata["memory"] = memory
        if engine is not None:
            ctx.metadata["retrieval_engine"] = engine
        if llm is not None:
            ctx.metadata["llm_client"] = llm
        if provider is not None:
            ctx.metadata["provider"] = provider

        for step in self.steps:
            logger.info("Agent Pipeline → %s", step.name)
            ctx = step.run(ctx)

        logger.info("Agent Pipeline 完成: answer_length=%d", len(ctx.answer))
        return ctx


def run_agent(
    question: str,
    *,
    top_k: int = RERANK_TOP_K,
    memory: ConversationHistory | ConversationMemory | None = None,
    engine: UnifiedRetrievalEngine | None = None,
    llm: LLMClient | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """便捷入口：执行 Agent 并返回 payload dict。"""
    workflow = AgentWorkflow(engine=engine, llm=llm)
    ctx = workflow.run(
        question,
        top_k=top_k,
        memory=memory,
        engine=engine,
        llm=llm,
        provider=provider,
    )
    return ctx.to_payload()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Information Processing Agent 调试")
    parser.add_argument("question", nargs="?", default="招商银行净利润怎么样？")
    parser.add_argument("-k", "--top-k", type=int, default=3)
    args = parser.parse_args()

    ctx = AgentWorkflow().run(args.question, top_k=args.top_k)
    print("Intent:", ctx.plan.primary_intent.label if ctx.plan else "-")
    print("Sources:", ctx.plan.source_filters if ctx.plan else "-")
    print("Results:", len(ctx.results))
    print("Answer:", ctx.answer[:300])


if __name__ == "__main__":
    main()
