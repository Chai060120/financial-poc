"""
步骤4：交互式检索。

默认 Hybrid + Rerank + 实体自动识别 + 查询增强，无需手动加参数。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    ENABLE_HYBRID,
    ENABLE_RERANK,
    RERANK_TOP_K,
    TOP_K,
    add_project_root_to_path,
    setup_logging,
)
from src.utils.source_display import format_reference_meta, source_type_label
from src.vectorstore.retrieval import RetrievalError, RetrievalResult
from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine, create_retrieval_engine

add_project_root_to_path()
logger = setup_logging(__name__)

_DEFAULT_TOP_K = RERANK_TOP_K if ENABLE_RERANK else TOP_K


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 Chroma 检索与用户问题相关的 Token")
    parser.add_argument("question", nargs="?", help="用户问题；省略则在运行时提示输入")
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help=f"返回结果数量，默认 {_DEFAULT_TOP_K}",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="关闭 Hybrid，仅使用向量检索",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="关闭 CrossEncoder 重排序",
    )
    parser.add_argument("--entity-name", default="", help="按公司简称过滤（如 招商银行）")
    parser.add_argument("--entity-id", default="", help="按股票代码过滤（如 600036.SH）")
    parser.add_argument(
        "--source",
        choices=["pdf", "news"],
        default="",
        help="按来源过滤：pdf 或 news",
    )
    parser.add_argument(
        "--section",
        default="",
        help="按财报章节过滤（如 利润表、资产负债表、管理层讨论）",
    )
    parser.add_argument("--date-from", default="", help="按日期下限过滤（YYYY-MM-DD）")
    parser.add_argument("--date-to", default="", help="按日期上限过滤（YYYY-MM-DD）")
    parser.add_argument(
        "--no-auto-entity",
        action="store_true",
        help="禁用从问题中自动识别公司实体过滤",
    )
    return parser.parse_args()


def read_question(args: argparse.Namespace) -> str:
    if args.question and args.question.strip():
        return args.question.strip()

    try:
        question = input("请输入问题: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        sys.exit(0)

    if not question:
        print("问题不能为空。")
        sys.exit(1)

    return question


def print_results(
    question: str,
    results: list[RetrievalResult],
    *,
    mode_label: str,
    show_hybrid_detail: bool,
) -> None:
    print(f"\n问题: {question}")
    print(f"模式: {mode_label}")
    print(f"命中 {len(results)} 条结果（按 score 降序）:\n")

    if not results:
        print("未检索到相关 Token，请先运行: python scripts/03_build_index.py")
        return

    for index, item in enumerate(results, start=1):
        meta = item["metadata"]
        chunk_index = meta.get("chunk_index", "")
        score = item.get("score", item["similarity"])
        rerank_score = item.get("rerank_score")
        text = item["text"]
        source = str(meta.get("source") or "")

        print(f"--- [{index}] {source_type_label(source)} ---")
        for part in format_reference_meta(meta):
            print(f"  {part}")
        print(f"  chunk_index: {chunk_index}")
        print(f"  score: {score:.4f}")

        if show_hybrid_detail:
            embedding_score = item.get("embedding_score")
            bm25_score = item.get("bm25_score")
            embedding_rank = item.get("embedding_rank")
            bm25_rank = item.get("bm25_rank")
            if embedding_rank is not None or embedding_score is not None:
                print(
                    f"  embedding: rank={embedding_rank}, score={embedding_score}"
                    if embedding_score is not None
                    else f"  embedding: rank={embedding_rank}"
                )
            if bm25_rank is not None or bm25_score is not None:
                print(
                    f"  bm25: rank={bm25_rank}, score={bm25_score:.4f}"
                    if bm25_score is not None
                    else f"  bm25: rank={bm25_rank}"
                )

        if rerank_score is not None:
            print(f"  rerank_score: {rerank_score:.4f}")

        print(f"  text: {text}")
        print()


def main() -> None:
    args = parse_args()
    question = read_question(args)

    if args.top_k <= 0:
        print(f"top_k 必须大于 0，当前为 {args.top_k}")
        sys.exit(1)

    use_hybrid = ENABLE_HYBRID and not args.no_hybrid
    use_rerank = ENABLE_RERANK and not args.no_rerank

    logger.info(
        "用户问题: %s | top_k=%d | hybrid=%s | rerank=%s",
        question[:80],
        args.top_k,
        use_hybrid,
        use_rerank,
    )

    engine = create_retrieval_engine(top_k=args.top_k)

    try:
        results = engine.retrieve(
            question,
            top_k=args.top_k,
            hybrid=use_hybrid,
            rerank=use_rerank,
            auto_entity=not args.no_auto_entity,
            entity_name=args.entity_name or None,
            entity_id=args.entity_id or None,
            source=args.source or None,
            section=args.section or None,
            date_from=args.date_from or None,
            date_to=args.date_to or None,
        )
    except ValueError as exc:
        logger.error("参数错误: %s", exc)
        print(f"检索失败: {exc}")
        sys.exit(1)
    except RetrievalError as exc:
        logger.error("检索失败: %s", exc)
        print(f"检索失败: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("检索异常: %s", exc)
        print(f"检索失败: {exc}")
        sys.exit(1)

    print_results(
        question,
        results,
        mode_label=engine.retrieval_mode_label(hybrid=use_hybrid, rerank=use_rerank),
        show_hybrid_detail=use_hybrid,
    )


if __name__ == "__main__":
    main()
