"""
向量索引构建流水线。

供 scripts/03_build_index.py 与 Daily Agent 共用。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DEFAULT_ENCODING, EMBEDDING_MODEL, TOKENS_JSON, setup_logging
from src.embeddings.text_embedding import TextEmbedder
from src.vectorstore.chroma_store import ChromaStore

logger = setup_logging(__name__)


def run_index_build(
    *,
    store: ChromaStore | None = None,
    embedder: TextEmbedder | None = None,
    tokens_path: Path | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """构建向量索引，返回统计信息。"""
    from api.services.index_service import run_index

    chroma_store = store or ChromaStore()
    text_embedder = embedder or TextEmbedder.get_instance(model_name=EMBEDDING_MODEL)
    path = tokens_path or TOKENS_JSON

    if not path.exists():
        return {
            "success": False,
            "message": f"Token 文件不存在: {path}",
            "indexed": 0,
            "after_count": chroma_store.count(),
        }

    try:
        payload = run_index(
            chroma_store,
            text_embedder,
            tokens_path=str(path),
            rebuild=rebuild,
        )
    except Exception as exc:
        logger.exception("索引构建失败: %s", exc)
        return {
            "success": False,
            "message": str(exc),
            "indexed": 0,
            "after_count": chroma_store.count(),
        }

    payload["success"] = payload.get("status") == "success"
    payload["message"] = (
        f"索引完成: 新增 {payload.get('indexed', 0)} 条, "
        f"合计 {payload.get('after_count', 0)} 条"
    )
    logger.info(payload["message"])
    return payload
