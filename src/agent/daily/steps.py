"""日报 Agent 流水线步骤。"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    AGENT_NEWS_DAYS,
    DAILY_REPORT_DIR,
    DEFAULT_ENCODING,
    NEWS_FETCH_LIMIT,
    NEWS_JSON,
    setup_logging,
)
from src.agent.daily.analyzer import analyze_daily
from src.agent.daily.report_generator import save_daily_report
from src.agent.daily.types import DailyContext, StepResult
from src.collectors.news_collector import DEFAULT_RSS_SOURCES, collect_and_update_news, load_news_json
from src.pipelines.document_pipeline import run_document_processing
from src.pipelines.index_pipeline import run_index_build
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine, create_retrieval_engine

logger = setup_logging(__name__)


def _timed_step(name: str, func, ctx: DailyContext) -> StepResult:
    started = time.perf_counter()
    try:
        data = func(ctx)
        duration_ms = int((time.perf_counter() - started) * 1000)
        message = str(data.get("message") or "完成")
        success = bool(data.get("success", True))
        logger.info("步骤 [%s] %s (%dms)", name, "成功" if success else "失败", duration_ms)
        return StepResult(name=name, success=success, message=message, data=data, duration_ms=duration_ms)
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.exception("步骤 [%s] 异常: %s", name, exc)
        return StepResult(
            name=name,
            success=False,
            message=str(exc),
            data={},
            duration_ms=duration_ms,
        )


class FetchNewsStep:
    name = "fetch_news"

    def run(self, ctx: DailyContext) -> StepResult:
        if ctx.skip_fetch:
            logger.info("跳过新闻抓取")
            try:
                ctx.news_records = load_news_json(NEWS_JSON)
            except Exception:
                ctx.news_records = []
            return StepResult(
                name=self.name,
                success=True,
                message="已跳过抓取，使用现有 news.json",
                data={"skipped": True, "loaded": len(ctx.news_records)},
            )

        def _fetch(context: DailyContext) -> dict:
            records, stats = collect_and_update_news(
                sources=list(DEFAULT_RSS_SOURCES),
                limit=NEWS_FETCH_LIMIT,
                days=max(context.news_days, AGENT_NEWS_DAYS),
                incremental=True,
            )
            context.news_records = records
            return {
                "success": True,
                "message": f"新闻抓取完成: 新增 {stats.added}, 合计 {stats.total}",
                "fetched": stats.fetched,
                "added": stats.added,
                "total": stats.total,
                "failed_sources": list(stats.failed_sources),
            }

        return _timed_step(self.name, _fetch, ctx)


class ProcessDocumentsStep:
    name = "process_documents"

    def run(self, ctx: DailyContext) -> StepResult:
        if ctx.skip_process:
            return StepResult(
                name=self.name,
                success=True,
                message="已跳过文档处理",
                data={"skipped": True},
            )

        def _process(_: DailyContext) -> dict:
            stats = run_document_processing()
            if not ctx.news_records:
                try:
                    ctx.news_records = load_news_json(NEWS_JSON)
                except Exception:
                    ctx.news_records = []
            return stats

        return _timed_step(self.name, _process, ctx)


class BuildIndexStep:
    name = "build_index"

    def __init__(self, store: ChromaStore | None = None) -> None:
        self._store = store

    def run(self, ctx: DailyContext) -> StepResult:
        if ctx.skip_index:
            return StepResult(
                name=self.name,
                success=True,
                message="已跳过向量索引",
                data={"skipped": True},
            )

        def _index(_: DailyContext) -> dict:
            rebuild = not ctx.skip_process
            return run_index_build(store=self._store, rebuild=rebuild)

        return _timed_step(self.name, _index, ctx)


class AnalyzeStep:
    name = "analyze"

    def __init__(self, engine: UnifiedRetrievalEngine | None = None) -> None:
        self._engine = engine

    def run(self, ctx: DailyContext) -> StepResult:
        def _analyze(context: DailyContext) -> dict:
            if not context.news_records:
                try:
                    context.news_records = load_news_json(NEWS_JSON)
                except Exception:
                    context.news_records = []

            engine = self._engine or create_retrieval_engine()
            analysis = analyze_daily(context, engine)
            context.analysis = analysis

            return {
                "success": True,
                "message": (
                    f"分析完成: 重大 {len(analysis.major_events)} / "
                    f"利好 {len(analysis.bullish)} / 利空 {len(analysis.bearish)} / "
                    f"风险 {len(analysis.risks)} / 业绩 {len(analysis.performance_changes)}"
                ),
                "major_events": len(analysis.major_events),
                "bullish": len(analysis.bullish),
                "bearish": len(analysis.bearish),
                "risks": len(analysis.risks),
                "performance_changes": len(analysis.performance_changes),
                "llm_used": analysis.llm_used,
            }

        return _timed_step(self.name, _analyze, ctx)


class GenerateReportStep:
    name = "generate_report"

    def run(self, ctx: DailyContext) -> StepResult:
        def _generate(context: DailyContext) -> dict:
            report_path = save_daily_report(context)
            json_path = _save_analysis_json(context)
            return {
                "success": True,
                "message": f"日报已生成: {report_path.name}",
                "report_path": str(report_path),
                "analysis_json": str(json_path),
            }

        return _timed_step(self.name, _generate, ctx)


def _save_analysis_json(ctx: DailyContext) -> Path:
    """保存结构化分析 JSON，便于后续扩展/API 消费。"""
    DAILY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DAILY_REPORT_DIR / f"{ctx.report_date}.json"
    analysis = ctx.analysis

    def _event_to_dict(item) -> dict:
        return {
            "title": item.title,
            "entity_name": item.entity_name,
            "category": item.category,
            "sentiment": item.sentiment,
            "publish_time": item.publish_time,
            "url": item.url,
            "summary": item.summary,
            "score": item.score,
            "source": item.source,
        }

    payload = {
        "report_date": ctx.report_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": analysis.summary,
        "llm_used": analysis.llm_used,
        "major_events": [_event_to_dict(i) for i in analysis.major_events],
        "bullish": [_event_to_dict(i) for i in analysis.bullish],
        "bearish": [_event_to_dict(i) for i in analysis.bearish],
        "risks": [_event_to_dict(i) for i in analysis.risks],
        "performance_changes": [_event_to_dict(i) for i in analysis.performance_changes],
        "pipeline_stats": ctx.stats,
        "errors": ctx.errors,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding=DEFAULT_ENCODING)
    logger.info("分析 JSON 已保存: %s", json_path)
    return json_path


DEFAULT_DAILY_STEPS = (
    FetchNewsStep,
    ProcessDocumentsStep,
    BuildIndexStep,
    AnalyzeStep,
    GenerateReportStep,
)
