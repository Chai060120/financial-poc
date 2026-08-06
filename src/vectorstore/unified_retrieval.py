"""
统一检索引擎：默认 Hybrid + Rerank + 实体自动过滤 + 查询增强。

对外提供与 RetrievalEngine 兼容的 retrieve() 接口，供 API、CLI、RAG Agent 共用。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    ENABLE_HYBRID,
    ENABLE_RERANK,
    HYBRID_BM25_WEIGHT,
    HYBRID_EMBEDDING_WEIGHT,
    RERANK_MODEL,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    TOP_K,
    setup_logging,
)
from src.embeddings.text_embedding import TextEmbedder
from src.utils.query_filters import (
    describe_retrieval_mode,
    enhance_retrieval_query,
    resolve_entity_filters,
)
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.hybrid_retrieval import (
    HybridRetrievalEngine,
    HybridRetrievalResult,
    RetrievalFilters,
    build_where_filter,
)
from src.vectorstore.retrieval import RetrievalEngine, RetrievalError, RetrievalResult

logger = setup_logging(__name__)


def merge_where_filters(
    where: dict[str, Any] | None,
    filters: RetrievalFilters | None,
) -> dict[str, Any] | None:
    """合并显式 where 与结构化 filters（扁平化 $and，避免 Chroma 嵌套过滤失效）。"""
    parts: list[dict[str, Any]] = []

    def _collect(clause: dict[str, Any] | None) -> None:
        if not clause:
            return
        nested = clause.get("$and")
        if isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    _collect(item)
            return
        parts.append(clause)

    _collect(where)
    _collect(build_where_filter(filters))

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


class UnifiedRetrievalEngine:
    """
    项目默认检索入口。

    - Hybrid（Embedding + BM25 + RRF）与 Rerank 按 config 默认开启
    - 自动从问题识别公司实体并过滤
    - 财务指标问句自动追加表头关键词
    """

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        store: ChromaStore | None = None,
        *,
        top_k: int = TOP_K,
        enable_hybrid: bool | None = None,
        enable_rerank: bool | None = None,
        embedding_weight: float = HYBRID_EMBEDDING_WEIGHT,
        bm25_weight: float = HYBRID_BM25_WEIGHT,
        rerank_candidate_k: int = RETRIEVAL_TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
        model_name: str = EMBEDDING_MODEL,
        rerank_model_name: str = RERANK_MODEL,
        persist_directory: Path | str | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.top_k = top_k
        self.enable_hybrid = ENABLE_HYBRID if enable_hybrid is None else enable_hybrid
        self.enable_rerank = ENABLE_RERANK if enable_rerank is None else enable_rerank
        self.rerank_top_k = rerank_top_k
        self.rerank_candidate_k = rerank_candidate_k

        self.embedder = embedder or TextEmbedder.get_instance(model_name=model_name)
        self.store = store or ChromaStore(
            persist_directory=persist_directory or CHROMA_DIR,
            collection_name=collection_name,
        )

        self._hybrid = HybridRetrievalEngine(
            embedder=self.embedder,
            store=self.store,
            top_k=top_k,
            embedding_weight=embedding_weight,
            bm25_weight=bm25_weight,
            enable_rerank=self.enable_rerank,
            rerank_candidate_k=rerank_candidate_k,
            rerank_model_name=rerank_model_name,
        )
        self._vector = RetrievalEngine(
            embedder=self.embedder,
            store=self.store,
            top_k=top_k,
            enable_rerank=self.enable_rerank,
            retrieval_top_k=rerank_candidate_k,
            rerank_top_k=rerank_top_k,
            rerank_model_name=rerank_model_name,
        )

        logger.info(
            "UnifiedRetrievalEngine 就绪: hybrid=%s, rerank=%s, top_k=%d",
            self.enable_hybrid,
            self.enable_rerank,
            self.top_k,
        )

    @property
    def collection_name(self) -> str:
        return self.store.collection_name

    def resolve_filters(
        self,
        question: str,
        *,
        entity_name: str | None = None,
        entity_id: str | None = None,
        source: str | None = None,
        section: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        auto_entity: bool = True,
    ) -> tuple[RetrievalFilters, str, bool]:
        """解析实体过滤与增强后的检索问句。"""
        resolved_name, resolved_id, auto_applied = resolve_entity_filters(
            question,
            entity_name=entity_name,
            entity_id=entity_id,
            auto_entity=auto_entity,
        )
        retrieval_query = enhance_retrieval_query(question)
        filters = RetrievalFilters(
            entity_name=resolved_name or "",
            entity_id=resolved_id or "",
            source=str(source or "").strip(),
            section=str(section or "").strip(),
            date_from=str(date_from or "").strip(),
            date_to=str(date_to or "").strip(),
        )
        return filters, retrieval_query, auto_applied

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        hybrid: bool | None = None,
        rerank: bool | None = None,
        where: dict[str, Any] | None = None,
        auto_entity: bool = True,
        entity_name: str | None = None,
        entity_id: str | None = None,
        source: str | None = None,
        section: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[RetrievalResult | HybridRetrievalResult]:
        """
        执行统一检索。

        与 RetrievalEngine.retrieve 参数兼容，并额外支持结构化过滤字段。
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        final_k = top_k if top_k is not None else self.top_k
        if final_k <= 0:
            raise ValueError(f"top_k 必须大于 0，当前为 {final_k}")

        use_hybrid = self.enable_hybrid if hybrid is None else hybrid
        use_rerank = self.enable_rerank if rerank is None else rerank

        filters, retrieval_query, auto_applied = self.resolve_filters(
            question,
            entity_name=entity_name,
            entity_id=entity_id,
            source=source,
            section=section,
            date_from=date_from,
            date_to=date_to,
            auto_entity=auto_entity,
        )
        chroma_where = merge_where_filters(where, filters)

        mode = describe_retrieval_mode(hybrid=use_hybrid, rerank=use_rerank)
        logger.info(
            "Unified 检索: mode=%s, question=%r, retrieval_query=%r, auto_entity=%s",
            mode,
            question.strip()[:80],
            retrieval_query[:120],
            auto_applied and bool(filters.entity_name),
        )
        if auto_applied and filters.entity_name:
            logger.info("自动识别实体过滤: %s", filters.entity_name)

        try:
            if use_hybrid:
                results = self._hybrid.retrieve(
                    retrieval_query,
                    top_k=final_k,
                    where=chroma_where,
                    rerank=use_rerank,
                )
            else:
                results = self._vector.retrieve(
                    retrieval_query,
                    top_k=final_k,
                    rerank=use_rerank,
                    where=chroma_where,
                )
        except ValueError:
            raise
        except Exception as exc:
            raise RetrievalError(f"统一检索失败: {exc}") from exc

        logger.info("Unified 检索完成: mode=%s, count=%d", mode, len(results))
        return results

    def retrieval_mode_label(
        self,
        *,
        hybrid: bool | None = None,
        rerank: bool | None = None,
    ) -> str:
        use_hybrid = self.enable_hybrid if hybrid is None else hybrid
        use_rerank = self.enable_rerank if rerank is None else rerank
        label = describe_retrieval_mode(hybrid=use_hybrid, rerank=use_rerank)
        mapping = {
            "hybrid+rerank": "Hybrid + Rerank",
            "hybrid": "Hybrid (RRF)",
            "vector+rerank": "向量 + Rerank",
            "vector": "向量",
        }
        return mapping.get(label, label)


def create_retrieval_engine(**kwargs: Any) -> UnifiedRetrievalEngine:
    """工厂方法：创建项目默认检索引擎。"""
    return UnifiedRetrievalEngine(**kwargs)
