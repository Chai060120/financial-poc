"""
Financial Agent：仅处理 PDF 财报 + 财经新闻。

统一入口，覆盖：
- 数据同步（抓新闻 → 处理 → 建索引）
- PDF 增量入库
- 智能检索
- RAG 问答
- 日报生成（可选）
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    AGENT_ENABLE_LLM,
    AGENT_NEWS_DAYS,
    NEWS_FETCH_LIMIT,
    RERANK_TOP_K,
    setup_logging,
)
from api.serializers import serialize_plan, serialize_retrieval_result
from src.agent.daily.runner import DailyAgent, run_daily_agent
from src.agent.daily.types import DailyContext
from src.agent.workflow import AgentWorkflow
from src.chat.history import ConversationHistory
from src.collectors.news_collector import DEFAULT_RSS_SOURCES, collect_and_update_news
from src.llm.llm_client import LLMClient, create_llm_client
from src.pipelines.document_pipeline import run_document_processing, run_incremental_pdfs
from src.pipelines.index_pipeline import run_index_build
from src.utils.query_filters import describe_retrieval_mode
from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine, create_retrieval_engine

logger = setup_logging(__name__)

# Agent 能力范围：仅 PDF 财报 + 财经新闻
AGENT_DATA_SOURCES: tuple[str, ...] = ("pdf", "news")

FINANCIAL_AGENT_PROMPT = (
    "你是金融信息处理 Agent，只使用两类资料回答用户问题："
    "（1）PDF 财报、定期报告；（2）财经新闻资讯。"
    "请严格基于提供的参考资料作答，不要编造资料中不存在的数据。"
    "引用时请区分来源是「财报」还是「新闻」。"
    "若用户问题省略了公司或指标主体，请结合对话历史理解其指代。"
    "若资料中出现分季度财务数据而用户询问全年指标，请将各季度数值加总后作答。"
    "若参考资料不足以回答，请明确说明。"
)


@dataclass
class FinancialAgent:
    """
    金融信息处理 Agent（PDF 财报 + 财经新闻）。

    示例:
        agent = FinancialAgent()
        agent.sync()                              # 抓新闻 + 处理 + 建索引
        agent.process_pdfs()                      # 增量处理 PDF
        agent.query("贵州茅台2024年净利润")      # 检索
        agent.ask("贵州茅台2024年净利润是多少")  # RAG 问答
        agent.daily()                             # 生成日报
    """

    engine: UnifiedRetrievalEngine | None = None
    llm: LLMClient | None = None
    history: ConversationHistory | None = None
    top_k: int = RERANK_TOP_K

    _workflow: AgentWorkflow | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.engine = self.engine or create_retrieval_engine(top_k=self.top_k)

    @property
    def workflow(self) -> AgentWorkflow:
        if self._workflow is None:
            self._workflow = AgentWorkflow(
                engine=self.engine,
                llm=self.llm,
                system_prompt=FINANCIAL_AGENT_PROMPT,
            )
        return self._workflow

    def sync(
        self,
        *,
        news_days: int = AGENT_NEWS_DAYS,
        skip_fetch: bool = False,
        rebuild_index: bool = True,
    ) -> dict[str, Any]:
        """
        同步数据：抓新闻 → PDF/News 处理 → 建向量索引。

        适合「把 data/raw/ 下新材料全部入库」。
        """
        payload: dict[str, Any] = {"steps": []}

        if not skip_fetch:
            logger.info("Agent.sync: 抓取财经新闻")
            records, stats = collect_and_update_news(
                sources=list(DEFAULT_RSS_SOURCES),
                limit=NEWS_FETCH_LIMIT,
                days=news_days,
                incremental=True,
            )
            fetch_info = {
                "step": "fetch_news",
                "success": True,
                "fetched": stats.fetched,
                "added": stats.added,
                "total": stats.total,
            }
            payload["steps"].append(fetch_info)
            payload["news_records"] = len(records)
        else:
            payload["steps"].append({"step": "fetch_news", "success": True, "skipped": True})

        logger.info("Agent.sync: 处理 PDF + 新闻")
        process_stats = run_document_processing()
        payload["steps"].append({"step": "process", **process_stats})
        if not process_stats.get("success"):
            payload["success"] = False
            payload["message"] = process_stats.get("message", "文档处理失败")
            return payload

        logger.info("Agent.sync: 构建向量索引")
        index_stats = run_index_build(rebuild=rebuild_index)
        payload["steps"].append({"step": "index", **index_stats})
        payload["success"] = index_stats.get("success", True)
        payload["message"] = (
            f"同步完成: {process_stats.get('total_tokens', 0)} Token, "
            f"索引 {index_stats.get('after_count', 0)} 条"
        )
        return payload

    def process_pdfs(
        self,
        pdf_paths: list[Path] | None = None,
        *,
        build_index: bool = True,
    ) -> dict[str, Any]:
        """增量处理 PDF 并可选更新索引。"""
        stats = run_incremental_pdfs(pdf_paths, build_index=build_index)
        stats["step"] = "process_pdf"
        return stats

    def query(self, question: str, *, top_k: int | None = None) -> dict[str, Any]:
        """智能检索（Hybrid + Rerank + 实体识别）。"""
        if not question.strip():
            raise ValueError("问题不能为空")

        final_k = top_k if top_k is not None else self.top_k
        results = self.engine.retrieve(question, top_k=final_k)  # type: ignore[union-attr]
        mode = describe_retrieval_mode(
            hybrid=self.engine.enable_hybrid,  # type: ignore[union-attr]
            rerank=self.engine.enable_rerank,  # type: ignore[union-attr]
        )
        return {
            "question": question.strip(),
            "mode": mode,
            "count": len(results),
            "results": [serialize_retrieval_result(item) for item in results],
        }

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        provider: str | None = None,
        show_prompt: bool = False,
    ) -> dict[str, Any]:
        """RAG 问答：Planner → 检索 → LLM 生成回答。"""
        if not question.strip():
            raise ValueError("问题不能为空")

        final_k = top_k if top_k is not None else self.top_k
        client = self.llm or create_llm_client(provider=provider, allow_preview=True)
        history = self.history or ConversationHistory()

        ctx = self.workflow.run(
            question.strip(),
            top_k=final_k,
            memory=history,
            engine=self.engine,
            llm=client,
            provider=provider,
        )
        payload = ctx.to_payload()

        response: dict[str, Any] = {
            "question": payload["question"],
            "retrieval_query": payload.get("retrieval_query") or payload["question"],
            "answer": payload.get("answer") or "",
            "result_count": len(payload.get("results") or []),
            "plan": serialize_plan(payload.get("plan")),
            "results": [
                serialize_retrieval_result(item) for item in (payload.get("results") or [])
            ],
        }

        llm_result = payload.get("llm_result")
        if llm_result is not None:
            response["llm"] = {
                "provider": llm_result.provider,
                "model": llm_result.model,
                "total_tokens": llm_result.total_tokens,
                "duration_ms": llm_result.duration_ms,
                "preview_mode": getattr(llm_result, "preview_mode", False),
            }
        if show_prompt:
            response["full_prompt"] = payload.get("full_prompt")

        self.history = history
        return response

    def daily(
        self,
        *,
        report_date: str | None = None,
        news_days: int = AGENT_NEWS_DAYS,
        enable_llm: bool = AGENT_ENABLE_LLM,
        skip_fetch: bool = False,
        skip_process: bool = False,
        skip_index: bool = False,
    ) -> DailyContext:
        """运行日报流水线：同步数据 + 分析 + 生成 Markdown 报告。"""
        daily_agent = DailyAgent(engine=self.engine)
        return daily_agent.run(
            report_date=report_date or date.today().isoformat(),
            news_days=news_days,
            enable_llm=enable_llm,
            skip_fetch=skip_fetch,
            skip_process=skip_process,
            skip_index=skip_index,
        )

    def valuate(
        self,
        query: str,
        *,
        report_year: str | None = None,
        save_report: bool = True,
        compare: bool = False,
    ):
        """基于财报 + 实时行情，输出高估/合理/低估结论。"""
        from src.analysis.valuation import analyze_valuation

        result = analyze_valuation(
            query,
            self.engine,  # type: ignore[arg-type]
            report_year=report_year,
            save_report=save_report,
        )
        if compare:
            from src.analysis.market_compare import analyze_market_comparison

            result.comparison = analyze_market_comparison(  # type: ignore[attr-defined]
                query,
                self.engine,  # type: ignore[arg-type]
                include_valuation=False,
                save_report=True,
            )
        return result

    def compare(
        self,
        query: str = "",
        *,
        watchlist: bool = False,
        include_valuation: bool = True,
        save_report: bool = True,
    ):
        """爬取网络资源，实时对比分析（同业 + 监控列表 + 新闻）。"""
        from src.analysis.market_compare import (
            analyze_market_comparison,
            analyze_watchlist_comparison,
        )

        if watchlist:
            return analyze_watchlist_comparison(
                self.engine,  # type: ignore[arg-type]
                save_report=save_report,
            )
        if not query.strip():
            raise ValueError("请指定公司名/代码，或使用 --watchlist")
        return analyze_market_comparison(
            query,
            self.engine,  # type: ignore[arg-type]
            include_valuation=include_valuation,
            save_report=save_report,
        )

    def analyze(
        self,
        target: str | Path | None = None,
        *,
        save_report: bool = True,
    ):
        """
        一键全量分析 Agent：
        导入财报 → 抽取指标/关键词 → 爬取网络 → 对比股价 → 高估/低估结论。

        target: PDF 路径 / 公司名 / None=处理 data/raw/pdf/ 下全部 PDF
        """
        from src.analysis.full_report import run_full_analysis

        return run_full_analysis(
            self.engine,  # type: ignore[arg-type]
            self.process_pdfs,
            target,
            save_report=save_report,
        )

    def chat(self) -> None:
        """进入对话式分析 Agent（意图识别 + 多轮追问）。"""
        from src.agent.analysis_agent import run_analysis_agent

        run_analysis_agent(self)


def create_financial_agent(**kwargs: Any) -> FinancialAgent:
    """创建 Financial Agent 实例。"""
    return FinancialAgent(**kwargs)
