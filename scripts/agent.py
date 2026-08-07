"""
Financial Agent 统一入口（PDF 财报 + 财经新闻）。

用法:
    python scripts/agent.py analyze                    # 一键全分析（推荐）
    python scripts/agent.py analyze 贵州茅台.pdf       # 指定财报 PDF
    python scripts/agent.py analyze 贵州茅台            # 指定公司（已入库）
    python scripts/agent.py                            # 交互检索演示
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
from src.analysis.valuation import ValuationResult
from src.analysis.market_compare import ComparisonResult
from src.analysis.full_report import FullAnalysisResult
from src.agent.autonomic_agent import create_autonomic_agent
from src.agent.daily.scheduler import start_scheduler
from src.agent.financial_agent import FinancialAgent, create_financial_agent
from src.chat.history import INTERACTIVE_COMMAND_HELP, ConversationHistory
from src.utils.query_insights import build_query_insight, format_query_insight, mark_insight_source
from src.utils.source_display import format_reference_meta, source_type_label

add_project_root_to_path()
logger = setup_logging(__name__)


def _print_query(payload: dict) -> None:
    print(f"\n问题: {payload['question']}")
    print(f"模式: {payload['mode']}")

    results = payload.get("results") or []
    insight = build_query_insight(payload["question"], results)
    if insight:
        print(format_query_insight(insight))

    print(f"命中 {payload['count']} 条原文:\n")
    for index, item in enumerate(results, start=1):
        meta = item.get("metadata") or {}
        source = str(meta.get("source") or "")
        tag = mark_insight_source(index, insight)
        print(f"--- [{index}] {source_type_label(source)}{tag} ---")
        for part in format_reference_meta(meta):
            print(f"  {part}")
        print(f"  score: {item.get('score', 0):.4f}")
        if item.get("rerank_score") is not None:
            print(f"  rerank_score: {item['rerank_score']:.4f}")

        text = str(item.get("text") or "")
        if insight and index in {m.source_rank for m in insight.metrics}:
            snippet = next(
                (m.snippet for m in insight.metrics if m.source_rank == index and m.snippet),
                "",
            )
            if snippet:
                print("  关键片段:")
                for line in snippet.splitlines():
                    print(f"  {line}")
            else:
                print(f"  text: {text[:200]}")
        else:
            print(f"  text: {text[:200]}")
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


def _print_valuation(result: ValuationResult) -> None:
    print("\n" + "=" * 60)
    print(f"  估值分析 · {result.entity_name} ({result.entity_id})")
    print("=" * 60)
    print(f"  财报年份: {result.report_year}")
    print(f"  结论: {result.verdict}  （评分 {result.score:+.1f}，置信度 {result.confidence}）")
    print()

    market = result.market
    print("  【估值指标】")
    if market.get("price") is not None:
        src = market.get("price_source") or market.get("source") or ""
        suffix = f" ({src})" if src else ""
        print(f"    现价: {market['price']}{suffix}")
    if market.get("pe_ttm") is not None:
        pe_src = market.get("pe_source") or ""
        tag = "推算" if str(pe_src).startswith("computed") else "动态"
        print(f"    PE({tag}): {market['pe_ttm']:.2f}")
    if market.get("pb") is not None:
        pb_src = market.get("pb_source") or ""
        tag = "推算" if str(pb_src).startswith("computed") else "实时"
        print(f"    PB({tag}): {market['pb']:.2f}")
    if market.get("market_cap") is not None:
        print(f"    总市值: {market['market_cap']}")
    if market.get("industry"):
        print(f"    行业: {market['industry']}")

    print("\n  【财报基本面】")
    period = result.fundamentals.get("period_label") or result.report_year
    print(f"    报告期: {period}")
    for key, label in (
        ("net_profit", "净利润"),
        ("revenue", "营业收入"),
        ("eps", "每股收益"),
        ("bvps", "每股净资产"),
        ("roe", "ROE"),
    ):
        item = result.fundamentals.get(key)
        if isinstance(item, dict) and item.get("display"):
            print(f"    {label}: {item['display']}")
    if result.fundamentals.get("revenue_growth_pct") is not None:
        print(f"    营收同比: {result.fundamentals['revenue_growth_pct']}%")
    if result.fundamentals.get("profit_growth_pct") is not None:
        print(f"    净利润同比: {result.fundamentals['profit_growth_pct']}%")
    if result.fundamentals.get("peg") is not None:
        print(f"    PEG: {result.fundamentals['peg']}")

    print("\n  【分析依据】")
    for reason in result.reasons:
        print(f"    · {reason}")

    if result.report_path:
        print(f"\n  报告已保存: {result.report_path}")
    print("=" * 60)
    print("  免责声明: PoC 规则分析，不构成投资建议")
    print("=" * 60)


def _print_comparison(result: ComparisonResult) -> None:
    print("\n" + "=" * 60)
    print(f"  实时对比分析 · {result.entity_name} ({result.entity_id})")
    print("=" * 60)
    print(f"  行业: {result.industry or '—'}")
    print(f"  相对同业: {result.relative_verdict}")
    print(f"  新闻情绪: {result.news_sentiment}")
    print(f"\n  摘要: {result.summary}")

    t = result.target
    print("\n  【目标公司 · 实时行情】")
    if t.price is not None:
        print(f"    现价: {t.price}")
    if t.pe_ttm is not None:
        print(f"    PE: {t.pe_ttm:.2f}")
    if t.pb is not None:
        print(f"    PB: {t.pb:.2f}")
    if t.change_pct is not None:
        print(f"    涨跌幅: {t.change_pct:.2f}%")

    s = result.industry_stats
    if s.peer_count:
        print("\n  【行业对比】")
        print(f"    同业样本: {s.peer_count} 只")
        if s.avg_pe is not None:
            print(f"    行业平均 PE: {s.avg_pe:.1f}（目标排名 {s.target_pe_rank}）")
        if s.avg_pb is not None:
            print(f"    行业平均 PB: {s.avg_pb:.2f}（目标排名 {s.target_pb_rank}）")
        print("\n    同业 PE/PB 一览:")
        for row in [result.target, *result.peers[:6]]:
            pe = f"{row.pe_ttm:.1f}" if row.pe_ttm else "—"
            pb = f"{row.pb:.2f}" if row.pb else "—"
            mark = " ← 目标" if row.entity_id == result.entity_id else ""
            print(f"      {row.entity_name}: PE={pe}, PB={pb}{mark}")

    if result.watchlist:
        print("\n  【监控列表对比】")
        for row in result.watchlist:
            pe = f"{row.pe_ttm:.1f}" if row.pe_ttm else "—"
            pb = f"{row.pb:.2f}" if row.pb else "—"
            print(f"    {row.entity_name}: PE={pe}, PB={pb}")

    if result.news:
        print("\n  【近期网络资讯】")
        for item in result.news[:5]:
            print(f"    · [{item.source}] {item.title[:60]}")

    if result.valuation:
        v = result.valuation
        print(f"\n  【财报估值】{v.verdict}（置信度 {v.confidence}）")

    if result.report_path:
        print(f"\n  报告已保存: {result.report_path}")
    print("=" * 60)


def _print_full_analysis(results: list[FullAnalysisResult]) -> None:
    for result in results:
        print("\n" + "=" * 60)
        print(f"  全量分析 · {result.entity_name} ({result.entity_id})")
        print("=" * 60)
        print(f"  报告期: {result.period_label}")
        print(f"  最终结论: {result.final_verdict}  （评分 {result.final_score:+.1f}，置信度 {result.final_confidence}）")
        print(f"\n  摘要: {result.executive_summary}")

        if result.keywords:
            print(f"\n  【关键词】{', '.join(result.keywords[:12])}")

        v = result.valuation
        if v:
            print("\n  【财报基本面】")
            for key, label in (
                ("net_profit", "净利润"),
                ("revenue", "营业收入"),
                ("eps", "每股收益"),
                ("roe", "ROE"),
            ):
                item = v.fundamentals.get(key)
                if isinstance(item, dict) and item.get("display"):
                    print(f"    {label}: {item['display']}")

        c = result.comparison
        if c:
            print("\n  【网络实时】")
            if c.target.price is not None:
                print(f"    现价: {c.target.price}")
            if c.target.pe_ttm is not None:
                print(f"    PE: {c.target.pe_ttm:.2f}")
            if c.target.pb is not None:
                print(f"    PB: {c.target.pb:.2f}")
            print(f"    相对同业: {c.relative_verdict}")
            print(f"    新闻情绪: {c.news_sentiment}")
            if c.news:
                print("    近期资讯:")
                for item in c.news[:3]:
                    print(f"      · [{item.source}] {item.title[:50]}")

        print("\n  【综合依据】")
        for reason in result.synthesis_reasons[:6]:
            print(f"    · {reason}")

        if result.report_path:
            print(f"\n  报告已保存: {result.report_path}")
        print("=" * 60)
        print("  免责声明: PoC 自动分析，不构成投资建议")
        print("=" * 60)


def _run_interactive_query(agent: FinancialAgent, *, top_k: int | None = None) -> None:
    """交互式检索演示：无需 LLM，适合汇报现场。"""
    print("\n" + "=" * 60)
    print("  Financial Agent · 智能检索")
    print("=" * 60)
    print("  输入问题，从 PDF 财报 / 财经新闻中检索相关内容")
    print()
    print("  示例问题：")
    print("    贵州茅台2024年归属于上市公司股东的净利润")
    print("    招商银行2024年营业收入")
    print()
    print("  输入 q 或 退出 结束演示")
    print("  （首次检索需加载模型，约 1～2 分钟）")
    print("=" * 60)

    while True:
        try:
            question = input("\n请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n演示结束。")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q", "退出"}:
            print("演示结束。")
            break
        try:
            payload = agent.query(question, top_k=top_k)
            _print_query(payload)
        except Exception as exc:
            print(f"\n检索失败: {exc}")


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
    pdf_p.add_argument(
        "--valuate",
        action="store_true",
        help="处理完成后自动输出估值结论（高估/合理/低估）",
    )

    query_p = sub.add_parser("query", help="智能检索")
    query_p.add_argument("question", nargs="?", help="检索问题；省略则进入交互模式")
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

    val_p = sub.add_parser("valuate", help="财报估值分析：高估/合理/低估")
    val_p.add_argument("target", help="公司名或代码，如 贵州茅台 / 600519.SH")
    val_p.add_argument("--year", default="", help="财报年份，默认自动识别最新年报")
    val_p.add_argument("--no-save", action="store_true", help="不写入 docs/valuation/")
    val_p.add_argument(
        "--compare",
        action="store_true",
        help="附加实时网络对比分析（同业/新闻）",
    )

    cmp_p = sub.add_parser("compare", help="爬取网络资源，实时对比分析")
    cmp_p.add_argument("target", nargs="?", default="", help="公司名或代码")
    cmp_p.add_argument(
        "--watchlist",
        action="store_true",
        help="对比监控列表（FINANCIAL_POC_AGENT_WATCHLIST）",
    )
    cmp_p.add_argument(
        "--no-valuation",
        action="store_true",
        help="不做财报估值，仅网络实时对比",
    )
    cmp_p.add_argument("--no-save", action="store_true", help="不写入 docs/comparison/")

    ana_p = sub.add_parser(
        "analyze",
        help="一键全分析：导入财报→关键词→网络对比→高估/低估结论",
    )
    ana_p.add_argument(
        "target",
        nargs="?",
        default="",
        help="PDF 路径或公司名；省略则分析 data/raw/pdf/ 下全部 PDF",
    )
    ana_p.add_argument("--no-save", action="store_true", help="不写入 docs/analysis/")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    agent = create_financial_agent()
    command = args.command

    if command is None:
        _run_interactive_query(agent)
        return

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
        if payload.get("success") and args.valuate:
            entities = payload.get("entities") or []
            if not entities:
                print("\n未识别到实体，跳过估值分析。")
            else:
                print(f"\n开始估值分析（共 {len(entities)} 家公司）...")
                for ent in entities:
                    try:
                        result = agent.valuate(ent["entity_name"])
                        _print_valuation(result)
                    except Exception as exc:
                        print(f"\n  {ent.get('entity_name')} 估值失败: {exc}")
        sys.exit(0 if payload.get("success") else 1)

    if command == "query":
        top_k = args.top_k or None
        if not args.question:
            _run_interactive_query(agent, top_k=top_k)
            return
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

    if command == "analyze":
        try:
            target = args.target.strip() or None
            results = agent.analyze(target, save_report=not args.no_save)
            _print_full_analysis(results)
        except Exception as exc:
            print(f"\n全量分析失败: {exc}")
            sys.exit(1)
        return

    if command == "valuate":
        try:
            result = agent.valuate(
                args.target,
                report_year=args.year.strip() or None,
                save_report=not args.no_save,
                compare=getattr(args, "compare", False),
            )
            _print_valuation(result)
            comparison = getattr(result, "comparison", None)
            if comparison is not None:
                _print_comparison(comparison)
        except Exception as exc:
            print(f"\n估值分析失败: {exc}")
            sys.exit(1)
        return

    if command == "compare":
        try:
            result = agent.compare(
                args.target,
                watchlist=args.watchlist,
                include_valuation=not args.no_valuation,
                save_report=not args.no_save,
            )
            _print_comparison(result)
        except Exception as exc:
            print(f"\n实时对比分析失败: {exc}")
            sys.exit(1)
        return

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
