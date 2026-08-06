"""
步骤2：文档处理（PDF + 新闻 → tokens.json）。

    python scripts/02_process.py           # 全量处理
    python scripts/02_process.py --news-only  # 仅追加新闻 Token
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import NEWS_JSON, add_project_root_to_path, ensure_dirs
from src.pipelines.document_pipeline import run_document_processing, run_news_append

add_project_root_to_path()


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF + 新闻 → Token 化")
    parser.add_argument(
        "--news-only",
        action="store_true",
        help="仅处理 news.json 并追加到 tokens（不重建 PDF Token）",
    )
    args = parser.parse_args()
    ensure_dirs()

    if args.news_only:
        stats = run_news_append()
        if not stats.get("success"):
            print(stats.get("message", "未生成任何新闻 Token"))
            print(f"请检查 {NEWS_JSON}")
            sys.exit(1)
        print("新闻处理完成")
        print(f"  新增: {stats.get('added', 0)} 条")
        print(f"  跳过重复: {stats.get('skipped', 0)} 条")
        print(f"  合计: {stats.get('total', 0)} 条")
        return

    stats = run_document_processing()
    if not stats.get("success"):
        print(stats.get("message", "未生成任何 Token"))
        print("  PDF:  data/raw/pdf/")
        print(f"  News: {NEWS_JSON}")
        sys.exit(1)

    print(f"处理完成，共 {stats['total_tokens']} 个 Token")
    print(f"  PDF:  {stats['pdf_tokens']}")
    print(f"  News: {stats['news_tokens']}")
    print(f"  去重: {stats.get('dedup_removed', 0)}")
    print(f"  JSON: {stats['tokens_json']}")
    print(f"  CSV:  {stats['unified_csv']}")
    print("\n下一步: python scripts/03_build_index.py")


if __name__ == "__main__":
    main()
