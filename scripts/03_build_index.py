"""
步骤3：构建向量索引。

读取 tokens.json → 生成 embedding → 批量写入 ChromaDB。

用法:
    python scripts/03_build_index.py
    python scripts/03_build_index.py --reset   # 清空后全量重建，避免重复索引
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
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOKENS_JSON,
    add_project_root_to_path,
    ensure_dirs,
    setup_logging,
)
from src.embeddings.text_embedding import TextEmbedder
from src.vectorstore.chroma_store import ChromaStore

add_project_root_to_path()
logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Chroma 向量索引")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重建前清空 collection，避免重复运行导致文档翻倍",
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=TOKENS_JSON,
        help=f"Token 文件路径，默认 {TOKENS_JSON}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if not args.tokens.exists():
        logger.error("Token 文件不存在: %s", args.tokens)
        print(f"索引构建失败: Token 文件不存在: {args.tokens}")
        print("请先运行: python scripts/02_process.py")
        sys.exit(1)

    logger.info("加载 Embedding 模型: %s", EMBEDDING_MODEL)

    try:
        embedder = TextEmbedder.get_instance(model_name=EMBEDDING_MODEL)
        store = ChromaStore(persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME)
    except Exception as exc:
        logger.exception("初始化 Embedding 或 Chroma 失败: %s", exc)
        print(f"索引构建失败: {exc}")
        sys.exit(1)

    from api.services.index_service import IndexServiceError, run_index

    try:
        payload = run_index(
            store,
            embedder,
            tokens_path=args.tokens,
            rebuild=args.reset,
        )
    except IndexServiceError as exc:
        logger.error("索引构建失败: %s", exc)
        print(f"索引构建失败: {exc}")
        sys.exit(1)

    print("\n索引构建完成")
    print(f"  模式:           {'全量重建' if args.reset else '增量追加'}")
    print(f"  本次成功索引: {payload['indexed']} 条")
    print(f"  跳过（已存在）: {payload['skipped_existing']} 条")
    print(f"  跳过（无效）:   {payload['skipped_invalid']} 条")
    print(f"  失败:           {payload['failed']} 条")
    print(f"  Collection:     {COLLECTION_NAME}")
    print(f"  存储路径:       {CHROMA_DIR}")
    print(f"  索引前文档数:   {payload['before_count']}")
    print(f"  索引后文档数:   {payload['after_count']}")
    print(f"  模型:           {payload['embedding_model']} ({embedder.embedding_dim} 维)")

    if payload["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
