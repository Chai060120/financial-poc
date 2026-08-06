"""
ChromaDB 向量存储封装。

所有 Chroma API 调用必须集中在本模块，其他模块通过 ChromaStore 访问。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chromadb
from chromadb.api.models.Collection import Collection

from config import CHROMA_DIR, COLLECTION_NAME, TOP_K, ensure_dirs, setup_logging

logger = setup_logging(__name__)


class ChromaStoreError(Exception):
    """ChromaDB 操作失败时抛出。"""


class SearchResult(TypedDict):
    """向量检索单条结果。"""

    id: str
    document: str
    metadata: dict[str, Any]
    distance: float


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """
    将 metadata 转为 Chroma 支持的标量类型。

    Chroma 仅接受 str / int / float / bool，其余类型转为字符串。
    """
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


class ChromaStore:
    """ChromaDB 持久化存储封装。"""

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        collection_name: str = COLLECTION_NAME,
        *,
        auto_create: bool = True,
    ) -> None:
        self.persist_directory = Path(persist_directory or CHROMA_DIR)
        self.collection_name = collection_name
        self._client: chromadb.ClientAPI | None = None
        self._collection: Collection | None = None

        ensure_dirs()
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._init_client()

        if auto_create:
            self.create_collection()

    def _init_client(self) -> None:
        """初始化 ChromaDB 持久化客户端。"""
        logger.info("初始化 ChromaDB: path=%s", self.persist_directory)

        try:
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        except Exception as exc:
            raise ChromaStoreError(
                f"无法初始化 ChromaDB: {self.persist_directory}"
            ) from exc

    def _ensure_client(self) -> chromadb.ClientAPI:
        if self._client is None:
            raise ChromaStoreError("ChromaDB 客户端未初始化")
        return self._client

    def _ensure_collection(self) -> Collection:
        if self._collection is None:
            raise ChromaStoreError(
                f"Collection 未创建: {self.collection_name}，请先调用 create_collection()"
            )
        return self._collection

    def create_collection(
        self,
        metadata: dict[str, Any] | None = None,
        *,
        reset: bool = False,
    ) -> None:
        """
        创建或获取 Collection。

        Args:
            metadata: 可选的 collection 元数据。
            reset: 为 True 时先删除已有 collection 再重建。
        """
        client = self._ensure_client()

        if reset and self.collection_exists():
            logger.info("删除已有 collection: %s", self.collection_name)
            client.delete_collection(name=self.collection_name)
            self._collection = None

        try:
            kwargs: dict[str, Any] = {"name": self.collection_name}
            if metadata is not None:
                kwargs["metadata"] = _sanitize_metadata(metadata)

            self._collection = client.get_or_create_collection(**kwargs)
            logger.info(
                "Collection 就绪: name=%s, count=%d",
                self.collection_name,
                self._collection.count(),
            )
        except Exception as exc:
            raise ChromaStoreError(
                f"无法创建 Collection: {self.collection_name}"
            ) from exc

    def collection_exists(self) -> bool:
        """判断 collection 是否已存在。"""
        client = self._ensure_client()

        try:
            collections = client.list_collections()
            return any(col.name == self.collection_name for col in collections)
        except Exception as exc:
            raise ChromaStoreError("查询 Collection 列表失败") from exc

    def count(self) -> int:
        """返回 collection 中已有文档数量。"""
        if not self.collection_exists() or self._collection is None:
            return 0

        try:
            return self._ensure_collection().count()
        except Exception as exc:
            raise ChromaStoreError("统计 Collection 文档数量失败") from exc

    def get_existing_ids(self, ids: list[str]) -> set[str]:
        """
        查询给定 id 列表中已存在于 collection 的 id 集合。

        Args:
            ids: 待检查的 Token id 列表。

        Returns:
            已存在的 id 集合。
        """
        if not ids:
            return set()

        if not self.collection_exists():
            return set()

        try:
            result = self._ensure_collection().get(ids=ids, include=[])
            return set(result.get("ids", []))
        except Exception as exc:
            raise ChromaStoreError("查询已有 id 失败") from exc

    def filter_new_ids(self, ids: list[str]) -> list[str]:
        """
        过滤已存在的 id，返回尚未写入 collection 的 id 列表（保持原顺序）。
        """
        existing_ids = self.get_existing_ids(ids)
        return [item_id for item_id in ids if item_id not in existing_ids]

    def add_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """
        批量写入向量与文档。

        调用方需确保 ids 在 collection 中不存在，否则 Chroma 会报错。
        写入前可用 get_existing_ids / filter_new_ids 去重。

        Args:
            ids: Token id 列表。
            embeddings: 与 ids 等长的向量列表。
            documents: 与 ids 等长的文本列表。
            metadatas: 与 ids 等长的元数据列表。

        Returns:
            成功写入条数。

        Raises:
            ChromaStoreError: 写入失败。
            ValueError: 输入为空或长度不匹配。
        """
        if not ids:
            return 0

        length = len(ids)
        if not (len(embeddings) == len(documents) == len(metadatas) == length):
            raise ValueError(
                "ids、embeddings、documents、metadatas 长度必须一致: "
                f"ids={length}, embeddings={len(embeddings)}, "
                f"documents={len(documents)}, metadatas={len(metadatas)}"
            )

        sanitized_metadatas = [_sanitize_metadata(meta) for meta in metadatas]

        try:
            self._ensure_collection().add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=sanitized_metadatas,
            )
        except Exception as exc:
            raise ChromaStoreError(f"批量写入失败，批次大小 {length}") from exc

        logger.debug("批量写入 %d 条到 collection=%s", length, self.collection_name)
        return length

    def delete_all(self) -> int:
        """
        删除 collection 中的全部文档。

        通过删除并重建 collection 实现，保留 collection 名称不变。

        Returns:
            删除前的文档数量。
        """
        if not self.collection_exists():
            logger.info("Collection 不存在，无需清空: %s", self.collection_name)
            return 0

        before_count = self.count()

        try:
            self._ensure_client().delete_collection(name=self.collection_name)
            self._collection = None
            self.create_collection()
        except Exception as exc:
            raise ChromaStoreError(
                f"清空 Collection 失败: {self.collection_name}"
            ) from exc

        logger.info(
            "已清空 collection=%s, 删除 %d 条",
            self.collection_name,
            before_count,
        )
        return before_count

    def query(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        按向量检索 Top K 文档。

        Args:
            query_embedding: 查询向量。
            top_k: 返回结果数量，默认使用 config.TOP_K。
            where: 可选 metadata 过滤条件。

        Returns:
            按相似度排序的检索结果列表。
        """
        if not query_embedding:
            raise ValueError("query_embedding 不能为空")

        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，当前为 {top_k}")

        if not self.collection_exists() or self.count() == 0:
            logger.warning("Collection 为空，返回空结果: %s", self.collection_name)
            return []

        n_results = min(top_k, self.count())

        try:
            raw = self._ensure_collection().query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise ChromaStoreError("向量检索失败") from exc

        return _parse_query_results(raw)

    def get_documents(
        self,
        *,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """
        读取 collection 中的文档（用于 BM25 等稀疏检索）。

        Args:
            where: 可选 metadata 过滤条件。
            limit: 最多返回条数；默认返回全部匹配文档。
        """
        if not self.collection_exists() or self.count() == 0:
            logger.warning("Collection 为空，返回空结果: %s", self.collection_name)
            return []

        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if where is not None:
            kwargs["where"] = where
        if limit is not None and limit > 0:
            kwargs["limit"] = limit

        try:
            raw = self._ensure_collection().get(**kwargs)
        except Exception as exc:
            raise ChromaStoreError("读取文档失败") from exc

        ids = raw.get("ids", []) or []
        documents = raw.get("documents", []) or []
        metadatas = raw.get("metadatas", []) or []

        results: list[SearchResult] = []
        for index, item_id in enumerate(ids):
            results.append(
                {
                    "id": item_id,
                    "document": documents[index] if index < len(documents) else "",
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": 0.0,
                }
            )
        return results


def _parse_query_results(raw: dict[str, Any]) -> list[SearchResult]:
    """将 Chroma query 原始结果转为 SearchResult 列表。"""
    ids = raw.get("ids", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    results: list[SearchResult] = []
    for index, item_id in enumerate(ids):
        results.append(
            {
                "id": item_id,
                "document": documents[index] if index < len(documents) else "",
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "distance": float(distances[index]) if index < len(distances) else 0.0,
            }
        )
    return results


def token_to_chroma_metadata(token: dict[str, Any]) -> dict[str, Any]:
    """
    将 Token 转为 Chroma metadata 字典。

    合并顶层 type/source 与 metadata 字段。
    """
    meta = dict(token.get("metadata", {}))
    meta["type"] = token.get("type", "")
    meta["source"] = token.get("source", "")
    return meta


def main() -> None:
    """命令行调试入口。"""
    store = ChromaStore()
    print(f"Collection: {store.collection_name}")
    print(f"Persist path: {store.persist_directory}")
    print(f"Exists:       {store.collection_exists()}")
    print(f"Document count: {store.count()}")


if __name__ == "__main__":
    main()
