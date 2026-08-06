"""
Financial Agent 统一入口（PDF 财报 + 财经新闻）。

用法:
    python scripts/agent.py run                     # 自主 Agent：同步+分析+LLM简报（推荐）
    python scripts/agent.py serve --time 08:00      # 常驻自主 Agent，每天自动运行
    python scripts/agent.py sync                    # 抓新闻 + 处理 + 建索引
    python scripts/agent.py pdf                     # 增量处理 PDF
    python scripts/agent.py query "贵州茅台2024年净利润"
    python scripts/agent.py ask "贵州茅台2024年净利润是多少"
    python scripts/agent.py ask -i                  # 连续对话
    python scripts/agent.py daily                   # 生成日报（无公司简报）
    python scripts/agent.py schedule --time 08:00   # 定时日报（旧版）
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    AGENT_ENABLE_LLM,
    AGENT_NEWS_DAYS,
    AGENT_SCHEDULE_TIME,
    DAILY_REPORT_DIR,
    add_project_root_to_path,
    ensure_dirs,
    setup_logging,
)
from src.agent.autonomic_agent import AutonomicAgent, create_autonomic_agent
from src.agent.daily.scheduler import start_scheduler
from src.agent.financial_agent import FinancialAgent, create_financial_agent
from src.chat.history import INTERACTIVE_COMMAND_HELP, ConversationHistory
from src.utils.source_display import format_reference_meta, source_type_label

add_project_root_to_path()
logger = setup_logging(__name__)


def _print_query(payload: dict) -> None:
    print(f"\n问题: {payload['question']}")
    print(f"模式: {payload['mode']}")
    print(f"命中 {payload['count']} 条:\n")
    for index, item in enumerate(payload["results"], start=1):
        meta = item.get("metadata") or {}
        source = str(meta.get("source") or "")
        print(f"--- [{index}] {source_type_label(source)} ---")
        for part in format_reference_meta(meta):
            print(f"  {part}")
        print(f"  score: {item.get('score', 0):.4f}")
        if item.get("rerank_score") is not None:
            print(f"  rerank_score: {item['rerank_score']:.4f}")
        print(f"  text: {item.get('text', '')[:300]}")
        print()


def _print_ask(payload: dict, *, show_prompt: bool = False) -> None:
    plan = payload.get("plan")
    print(f"\n问题: {payload['question']}")
    if payload.get("retrieval_query") != payload["question"]:
        print(f"检索 query: {payload['retrieval_query']}")
    if plan:
        print(f"意图: {plan.get('intent_label')} | 数据源: {', '.join(plan.get('source_filters') or [])}")
    print(f"检索命中: {payload.get('result_count', 0)} 条")
    llm = payload.get("llm") or {}
    preview = llm.get("preview_mode", False)
    print("\n" + "=" * 60)
    print("Prompt Preview" if preview else "回答")
    print("=" * 60)
    print(payload.get("answer") or "")
    if show_prompt and payload.get("full_prompt"):
        print("\n--- Prompt ---\n")
        print(payload["full_prompt"])


def _print_autonomic(result) -> None:
    print("\n" + "=" * 60)
    print(f"Autonomic Agent · {result.report_date}")
    print("=" * 60)
    ctx = result.daily_context
    if ctx:
        for step in ctx.step_results:
            status = "OK" if step.success else "FAIL"
            print(f"  [{status}] {step.name}: {step.message} ({step.duration_ms}ms)")
        if result.summary:
            print(f"\n事件分析: {result.summary[:300]}...")
    if result.company_briefings:
        print(f"\n公司简报 ({len(result.company_briefings)} 家):")
        for item in result.company_briefings:
            tag = " [Preview]" if item.preview_mode else ""
            print(f"\n--- {item.entity_name}{tag} ---")
            print((item.answer or "")[:500])
            if len(item.answer or "") > 500:
                print("  …")
    if result.report_path:
        print(f"\n完整报告: {result.report_path}")
    if result.briefings_path:
        print(f"简报 JSON: {result.briefings_path}")
    if result.errors:
        print("\n提示:")
        for error in result.errors:
            print(f"  - {error}")
    print(f"\nLLM 已启用: {'是' if result.llm_used else '否（请配置 .env API Key）'}")


def _print_daily(ctx) -> None:
    print("\n" + "=" * 60)
    print(f"Daily Agent · {ctx.report_date}")
    print("=" * 60)
    for step in ctx.step_results:
        status = "OK" if step.success else "FAIL"
        print(f"  [{status}] {step.name}: {step.message} ({step.duration_ms}ms)")
    analysis = ctx.analysis
    if analysis:
        print(f"\n分析摘要: {analysis.summary[:200]}...")
    if ctx.report_path:
        print(f"\nMarkdown 报告: {ctx.report_path}")
        print(f"JSON 分析:     {DAILY_REPORT_DIR / f'{ctx.report_date}.json'}")
    if ctx.errors:
        print("\n告警:")
        for error in ctx.errors:
            print(f"  - {error}")


def _print_sync(payload: dict) -> None:
    print("\n=== Agent 同步完成 ===" if payload.get("success") else "\n=== Agent 同步失败 ===")
    print(payload.get("message", ""))
    for step in payload.get("steps", []):
        name = step.get("step", "?")
        if step.get("skipped"):
            print(f"  [{name}] 已跳过")
        elif name == "process":
            print(
                f"  [process] PDF {step.get('pdf_tokens', 0)} + "
                f"News {step.get('news_tokens', 0)} = {step.get('total_tokens', 0)} Token"
            )
        elif name == "index":
            print(f"  [index] 索引合计 {step.get('after_count', 0)} 条")
        elif name == "fetch_news":
            print(f"  [fetch_news] 新增 {step.get('added', 0)} 条, 合计 {step.get('total', 0)} 条")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Financial Agent：PDF 财报 + 财经新闻（采集/入库/检索/问答/日报）"
    )
    sub = parser.add_subparsers(dest="command")

    sync_p = sub.add_parser("sync", help="抓新闻 + 处理 PDF/News + 建索引")
    sync_p.add_argument("--skip-fetch", action="store_true", help="跳过新闻抓取")
    sync_p.add_argument("--news-days", type=int, default=AGENT_NEWS_DAYS)
    sync_p.add_argument("--no-rebuild", action="store_true", help="增量建索引（默认全量重建）")

    pdf_p = sub.add_parser("pdf", help="增量处理 PDF")
    pdf_p.add_argument("files", nargs="*", help="PDF 路径；省略则处理 data/raw/pdf/ 全部")
    pdf_p.add_argument("--no-index", action="store_true", help="不更新向量索引")

    query_p = sub.add_parser("query", help="智能检索")
    query_p.add_argument("question", help="检索问题")
    query_p.add_argument("-k", "--top-k", type=int, default=0)

    ask_p = sub.add_parser("ask", help="RAG 问答")
    ask_p.add_argument("question", nargs="?", help="问题；省略则进入连续对话")
    ask_p.add_argument("-i", "--interactive", action="store_true", help="连续对话")
    ask_p.add_argument("-k", "--top-k", type=int, default=0)
    ask_p.add_argument("--provider", choices=("openai", "deepseek", "qwen"), default=None)
    ask_p.add_argument("--show-prompt", action="store_true")

    daily_p = sub.add_parser("daily", help="运行日报流水线")
    daily_p.add_argument("--date", default="", help="报告日期 YYYY-MM-DD")
    daily_p.add_argument("--skip-fetch", action="store_true")
    daily_p.add_argument("--skip-process", action="store_true")
    daily_p.add_argument("--skip-index", action="store_true")
    daily_p.add_argument("--no-llm", action="store_true")
    daily_p.add_argument("--news-days", type=int, default=AGENT_NEWS_DAYS)

    sched_p = sub.add_parser("schedule", help="定时运行日报 Agent")
    sched_p.add_argument("--time", default=AGENT_SCHEDULE_TIME, help="HH:MM")
    sched_p.add_argument("--now", action="store_true", help="启动前先跑一次")
    sched_p.add_argument("--news-days", type=int, default=AGENT_NEWS_DAYS)
    sched_p.add_argument("--no-llm", action="store_true")

    run_p = sub.add_parser("run", help="自主 Agent：同步+分析+LLM 公司简报（一次）")
    run_p.add_argument("--date", default="", help="报告日期 YYYY-MM-DD")
    run_p.add_argument("--skip-fetch", action="store_true")
    run_p.add_argument("--skip-process", action="store_true")
    run_p.add_argument("--skip-index", action="store_true")
    run_p.add_argument("--skip-briefings", action="store_true", help="跳过监控列表 LLM 简报")
    run_p.add_argument("--no-llm", action="store_true")
    run_p.add_argument("--news-days", type=int, default=AGENT_NEWS_DAYS)
    run_p.add_argument("--provider", choices=("openai", "deepseek", "qwen"), default=None)

    serve_p = sub.add_parser("serve", help="常驻自主 Agent，每天定时自动运行")
    serve_p.add_argument("--time", default=AGENT_SCHEDULE_TIME, help="HH:MM")
    serve_p.add_argument("--now", action="store_true", help="启动前先跑一次")
    serve_p.add_argument("--skip-briefings", action="store_true")
    serve_p.add_argument("--no-llm", action="store_true")
    serve_p.add_argument("--news-days", type=int, default=AGENT_NEWS_DAYS)
    serve_p.add_argument("--provider", choices=("openai", "deepseek", "qwen"), default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    agent = create_financial_agent()
    command = args.command

    if command is None:
        parser = argparse.ArgumentParser(
            description="Financial Agent：PDF 财报 + 财经新闻"
        )
        sub = parser.add_subparsers(dest="command")
        sub.add_parser("sync")
        sub.add_parser("pdf")
        sub.add_parser("query")
        sub.add_parser("ask")
        sub.add_parser("daily")
        sub.add_parser("run")
        sub.add_parser("serve")
        sub.add_parser("schedule")
        parser.print_help()
        sys.exit(1)

    if command == "sync":
        payload = agent.sync(
            news_days=args.news_days,
            skip_fetch=args.skip_fetch,
            rebuild_index=not args.no_rebuild,
        )
        _print_sync(payload)
        sys.exit(0 if payload.get("success") else 1)

    if command == "pdf":
        paths = [Path(f) for f in args.files] if args.files else None
        payload = agent.process_pdfs(paths, build_index=not args.no_index)
        print(payload.get("message", ""))
        if payload.get("failed"):
            for item in payload["failed"]:
                print(f"  失败: {item.get('file')} - {item.get('reason')}")
        sys.exit(0 if payload.get("success") else 1)

    if command == "query":
        top_k = args.top_k or None
        payload = agent.query(args.question, top_k=top_k)
        _print_query(payload)
        return

    if command == "ask":
        top_k = args.top_k or None
        provider = args.provider or os.getenv("LLM_PROVIDER", "openai")
        interactive = args.interactive or not args.question

        if not interactive:
            payload = agent.ask(
                args.question,
                top_k=top_k,
                provider=provider,
                show_prompt=args.show_prompt,
            )
            _print_ask(payload, show_prompt=args.show_prompt)
            return

        history = ConversationHistory()
        agent.history = history
        print("Financial Agent 连续对话（PDF 财报 + 财经新闻）")
        print(INTERACTIVE_COMMAND_HELP.strip())
        while True:
            try:
                question = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q", "退出"}:
                print("再见。")
                break
            if question.lower() in {"reset", "清空", "clear"}:
                history.reset()
                print("对话历史已清空。")
                continue
            payload = agent.ask(
                question,
                top_k=top_k,
                provider=provider,
                show_prompt=args.show_prompt,
            )
            _print_ask(payload, show_prompt=args.show_prompt)
        return

    if command == "run":
        autonomic = create_autonomic_agent(
            provider=args.provider,
            enable_llm=AGENT_ENABLE_LLM and not args.no_llm,
            news_days=args.news_days,
        )
        result = autonomic.run_once(
            report_date=args.date.strip() or None,
            skip_fetch=args.skip_fetch,
            skip_process=args.skip_process,
            skip_index=args.skip_index,
            skip_briefings=args.skip_briefings,
        )
        _print_autonomic(result)
        sys.exit(0 if result.success else 1)

    if command == "serve":
        autonomic = create_autonomic_agent(
            provider=args.provider,
            enable_llm=AGENT_ENABLE_LLM and not args.no_llm,
            news_days=args.news_days,
            skip_briefings=args.skip_briefings,
        )
        autonomic.serve(
            schedule_time=args.time,
            run_immediately=args.now,
        )
        return

    if command == "daily":
        ctx = agent.daily(
            report_date=args.date.strip() or None,
            news_days=args.news_days,
            enable_llm=AGENT_ENABLE_LLM and not args.no_llm,
            skip_fetch=args.skip_fetch,
            skip_process=args.skip_process,
            skip_index=args.skip_index,
        )
        _print_daily(ctx)
        sys.exit(0 if ctx.success else 1)

    if command == "schedule":
        if args.now:
            ctx = agent.daily(
                news_days=args.news_days,
                enable_llm=AGENT_ENABLE_LLM and not args.no_llm,
            )
            _print_daily(ctx)
        logger.info("定时调度 %s，Ctrl+C 退出", args.time)
        start_scheduler(schedule_time=args.time, run_immediately=False)
        return


if __name__ == "__main__":
    main()
