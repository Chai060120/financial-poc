"""
步骤1：新闻 RSS 采集。

支持 Google News RSS、东方财富 RSS、新浪财经 RSS 及自定义 RSS。
统一输出 data/raw/news/news.json，支持去重、时间过滤与增量更新。

用法:
    python scripts/01_fetch_news.py
    python scripts/01_fetch_news.py --days 3
    python scripts/01_fetch_news.py --since 2025-07-01
    python scripts/01_fetch_news.py --sources google_news,sina_finance
    python scripts/01_fetch_news.py --full
    python scripts/01_fetch_news.py --process --index
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    NEWS_DEFAULT_DAYS,
    NEWS_FETCH_LIMIT,
    NEWS_JSON,
    add_project_root_to_path,
    ensure_dirs,
    setup_logging,
)
from src.collectors.news_collector import (
    DEFAULT_RSS_SOURCES,
    RSS_SOURCE_REGISTRY,
    NewsCollectorError,
    collect_and_update_news,
    parse_publish_datetime,
)

add_project_root_to_path()
logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSS 新闻采集并写入 data/raw/news/news.json")
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_RSS_SOURCES),
        help=f"RSS 源 ID，逗号分隔，可选: {', '.join(RSS_SOURCE_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=NEWS_FETCH_LIMIT,
        help=f"每个 RSS 源最多抓取条数，默认 {NEWS_FETCH_LIMIT}",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=NEWS_DEFAULT_DAYS,
        help=f"仅保留最近 N 天新闻，默认 {NEWS_DEFAULT_DAYS}；0 表示不过滤",
    )
    parser.add_argument(
        "--since",
        default="",
        help="起始日期过滤，格式 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="全量覆盖 news.json（不做增量合并）",
    )
    parser.add_argument(
        "--no-extra-rss",
        action="store_true",
        help="不抓取 config.NEWS_RSS_FEEDS 中的自定义 RSS",
    )
    parser.add_argument(
        "--output",
        default=str(NEWS_JSON),
        help=f"输出文件路径，默认 {NEWS_JSON}",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="采集后运行 02_process --news-only（清洗 → 分块 → Token）",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="采集并处理后运行 03_build_index（写入 Chroma）",
    )
    return parser.parse_args()


def parse_since(value: str) -> datetime | None:
    """解析 --since 参数。"""
    text = value.strip()
    if not text:
        return None

    dt = parse_publish_datetime(text)
    if dt is None:
        raise ValueError(f"无法解析 --since 日期: {text}")
    return dt


def run_script(script_name: str, *script_args: str) -> int:
    script_path = _SCRIPT_DIR / script_name
    logger.info("运行子脚本: %s %s", script_path.name, " ".join(script_args))
    result = subprocess.run(
        [sys.executable, str(script_path), *script_args],
        cwd=str(_PROJECT_ROOT),
        check=False,
    )
    return result.returncode


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if args.limit <= 0:
        print(f"limit 必须大于 0，当前为 {args.limit}")
        sys.exit(1)

    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    output_path = Path(args.output)

    try:
        since = parse_since(args.since) if args.since else None
    except ValueError as exc:
        logger.error("参数错误: %s", exc)
        print(exc)
        sys.exit(1)

    days = None if since is not None else (args.days if args.days > 0 else None)

    logger.info(
        "开始新闻采集: sources=%s, limit=%d, days=%s, since=%s, incremental=%s, output=%s",
        sources,
        args.limit,
        days,
        since.strftime("%Y-%m-%d %H:%M:%S") if since else None,
        not args.full,
        output_path,
    )

    try:
        records, stats = collect_and_update_news(
            sources=sources,
            limit=args.limit,
            days=days,
            since=since,
            incremental=not args.full,
            output_path=output_path,
            include_extra_rss=not args.no_extra_rss,
        )
    except NewsCollectorError as exc:
        logger.error("新闻采集失败: %s", exc)
        print(f"新闻采集失败: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("新闻采集异常: %s", exc)
        print(f"新闻采集失败: {exc}")
        sys.exit(1)

    print("新闻采集完成")
    print(f"  输出文件:     {output_path}")
    print(f"  本次抓取:     {stats.fetched} 条")
    print(f"  时间过滤后:   {stats.after_time_filter} 条")
    print(f"  原有记录:     {stats.existing} 条")
    print(f"  新增记录:     {stats.added} 条")
    print(f"  重复跳过:     {stats.skipped_duplicate} 条")
    print(f"  合计:         {stats.total} 条")
    if stats.failed_sources:
        print(f"  失败源:       {', '.join(stats.failed_sources)}")

    if not records and stats.fetched == 0:
        print("\n未抓取到任何新闻，请检查网络或 RSS 源是否可用。")
        sys.exit(1)

    if args.process or args.index:
        code = run_script("02_process.py", "--news-only")
        if code != 0:
            print(f"02_process --news-only 失败，退出码 {code}")
            sys.exit(code)
        print("\n02_process --news-only 完成")

    if args.index:
        code = run_script("03_build_index.py")
        if code != 0:
            print(f"03_build_index 失败，退出码 {code}")
            sys.exit(code)
        print("\n03_build_index 完成")
    elif not args.process:
        print("\n下一步:")
        print("  python scripts/02_process.py --news-only")
        print("  python scripts/03_build_index.py")


if __name__ == "__main__":
    main()
