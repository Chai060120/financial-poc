"""索引服务：读取 tokens.json 并写入 ChromaDB。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    COLLECTION_NAME,
    DEFAULT_ENCODING,
    EMBEDDING_MODEL,
    TOKENS_JSON,
    add_project_root_to_path,
    setup_logging,
)
from src.embeddings.text_embedding import TextEmbedder
from src.vectorstore.chroma_store import ChromaStore, ChromaStoreError

add_project_root_to_path()
logger = setup_logging(__name__)

DEFAULT_INDEX_BATCH_SIZE = int(os.getenv("FINANCIAL_POC_INDEX_BATCH_SIZE", "64"))
DEFAULT_EMBED_BATCH_SIZE = int(os.getenv("FINANCIAL_POC_EMBED_BATCH_SIZE", "32"))


class IndexServiceError(Exception):
    """索引服务错误。"""


def _load_tokens(tokens_path: Path) -> list[dict[str, Any]]:
    if not tokens_path.exists():
        raise IndexServiceError(f"Token 文件不存在: {tokens_path}")

    with open(tokens_path, encoding=DEFAULT_ENCODING) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise IndexServiceError(f"Token 文件格式错误，期望 JSON 数组: {tokens_path}")
    return data


def _filter_valid_tokens(tokens: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    valid: list[dict[str, Any]] = []
    skipped = 0
    for token in tokens:
        if not token.get("id") or not str(token.get("text") or "").strip():
            skipped += 1
            continue
        valid.append(token)
    return valid, skipped


def _index_tokens(
    tokens: list[dict[str, Any]],
    embedder: TextEmbedder,
    store: ChromaStore,
) -> dict[str, int]:
    from src.vectorstore.chroma_store import token_to_chroma_metadata

    valid_tokens, skipped_invalid = _filter_valid_tokens(tokens)
    stats = {
        "indexed": 0,
        "skipped_existing": 0,
        "skipped_invalid": skipped_invalid,
        "failed": 0,
    }

    if not valid_tokens:
        return stats

    for batch_index in range(0, len(valid_tokens), DEFAULT_INDEX_BATCH_SIZE):
        batch = valid_tokens[batch_index : batch_index + DEFAULT_INDEX_BATCH_SIZE]
        batch_ids = [str(token["id"]) for token in batch]

        try:
            existing_ids = store.get_existing_ids(batch_ids)
        except ChromaStoreError as exc:
            logger.error("查询已有 id 失败: %s", exc)
            stats["failed"] += len(batch)
            continue

        new_tokens = [token for token in batch if str(token["id"]) not in existing_ids]
        stats["skipped_existing"] += len(batch) - len(new_tokens)
        if not new_tokens:
            continue

        new_ids = [str(token["id"]) for token in new_tokens]
        texts = [str(token["text"]) for token in new_tokens]
        metadatas = [token_to_chroma_metadata(token) for token in new_tokens]

        try:
            embeddings = embedder.embed_batch(texts, batch_size=DEFAULT_EMBED_BATCH_SIZE)
            stats["indexed"] += store.add_batch(new_ids, embeddings, texts, metadatas)
        except (ValueError, ChromaStoreError) as exc:
            logger.error("索引批次失败: %s", exc)
            stats["failed"] += len(new_tokens)

    return stats


def run_index(
    store: ChromaStore,
    embedder: TextEmbedder,
    *,
    tokens_path: str | Path | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """构建向量索引并返回统计信息。"""
    path = Path(tokens_path) if tokens_path else TOKENS_JSON

    try:
        tokens = _load_tokens(path)
    except json.JSONDecodeError as exc:
        raise IndexServiceError(f"Token JSON 解析失败: {exc}") from exc

    if not tokens:
        raise IndexServiceError("Token 文件为空，请先运行 scripts/02_process.py")

    before_count = store.count()
    if rebuild and before_count > 0:
        store.delete_all()
        before_count = 0

    stats = _index_tokens(tokens, embedder, store)
    after_count = store.count()

    status = "failed" if stats["failed"] > 0 and stats["indexed"] == 0 else "success"
    logger.info(
        "IndexService: status=%s, indexed=%d, after=%d",
        status,
        stats["indexed"],
        after_count,
    )

    return {
        "status": status,
        "collection": COLLECTION_NAME,
        "indexed": stats["indexed"],
        "skipped_existing": stats["skipped_existing"],
        "skipped_invalid": stats["skipped_invalid"],
        "failed": stats["failed"],
        "before_count": before_count,
        "after_count": after_count,
        "embedding_model": embedder.model_name,
    }
