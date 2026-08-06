"""
Hybrid Retrieval：Embedding Search + BM25 Search + RRF Fusion。

将稠密向量检索与 BM25 稀疏检索结果通过 Reciprocal Rank Fusion (RRF) 融合，
支持 top_k、权重、公司/时间/来源过滤。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    ENABLE_RERANK,
    HYBRID_BM25_WEIGHT,
    HYBRID_CANDIDATE_K,
    HYBRID_EMBEDDING_WEIGHT,
    HYBRID_RRF_K,
    RERANK_MODEL,
    RETRIEVAL_TOP_K,
    TOP_K,
    setup_logging,
)
from src.embeddings.text_embedding import TextEmbedder
from src.reranker.base import Reranker, apply_rerank_to_results
from src.reranker.cross_encoder import CrossEncoderReranker
from src.vectorstore.chroma_store import ChromaStore, ChromaStoreError, SearchResult
from src.vectorstore.retrieval import (
    RetrievalError,
    RetrievalResult,
    _to_retrieval_result,
)

logger = setup_logging(__name__)

_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+")


class HybridRetrievalResult(RetrievalResult):
    """Hybrid 检索结果，在 RetrievalResult 基础上附加融合细节。"""

    rrf_score: NotRequired[float]
    embedding_score: NotRequired[float | None]
    bm25_score: NotRequired[float | None]
    embedding_rank: NotRequired[int | None]
    bm25_rank: NotRequired[int | None]


@dataclass(frozen=True)
class RetrievalFilters:
    """检索过滤条件。"""

    entity_name: str = ""
    entity_id: str = ""
    source: str = ""
    section: str = ""
    date_from: str = ""
    date_to: str = ""


def _build_filters(
    filters: RetrievalFilters | None,
    *,
    entity_name: str | None = None,
    entity_id: str | None = None,
    source: str | None = None,
    section: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> RetrievalFilters:
    base = filters or RetrievalFilters()
    return RetrievalFilters(
        entity_name=entity_name if entity_name is not None else base.entity_name,
        entity_id=entity_id if entity_id is not None else base.entity_id,
        source=source if source is not None else base.source,
        section=section if section is not None else base.section,
        date_from=date_from if date_from is not None else base.date_from,
        date_to=date_to if date_to is not None else base.date_to,
    )


def build_where_filter(filters: RetrievalFilters | None = None) -> dict[str, Any] | None:
    """将 RetrievalFilters 转为 Chroma where 条件。"""
    if filters is None:
        return None

    conditions: list[dict[str, Any]] = []

    entity_id = str(filters.entity_id or "").strip()
    entity_name = str(filters.entity_name or "").strip()
    source = str(filters.source or "").strip()
    section = str(filters.section or "").strip()
    date_from = str(filters.date_from or "").strip()
    date_to = str(filters.date_to or "").strip()

    if entity_id:
        conditions.append({"entity_id": entity_id})
    if entity_name:
        conditions.append({"entity_name": entity_name})
    if source:
        conditions.append({"source": source})
    if section:
        conditions.append({"section": section})
    if date_from:
        conditions.append({"date": {"$gte": date_from}})
    if date_to:
        conditions.append({"date": {"$lte": date_to}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _tokenize(text: str) -> list[str]:
    """中文友好的 BM25 分词：汉字单字 + 英文数字词。"""
    tokens = [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]
    return tokens or [text.strip().lower()] if text.strip() else []


def _normalize_weights(embedding_weight: float, bm25_weight: float) -> tuple[float, float]:
    embed_w = max(0.0, float(embedding_weight))
    bm25_w = max(0.0, float(bm25_weight))
    total = embed_w + bm25_w
    if total <= 0:
        return 0.5, 0.5
    return embed_w / total, bm25_w / total


def _rrf_fusion(
    ranked_lists: list[list[str]],
    weights: list[float],
    *,
    rrf_k: int,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion。

    score(doc) = sum(weight_i / (k + rank_i))
    """
    if not ranked_lists:
        return {}

    scores: dict[str, float] = {}
    safe_k = max(1, int(rrf_k))

    for ranked_ids, weight in zip(ranked_lists, weights):
        if weight <= 0 or not ranked_ids:
            continue
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (safe_k + rank)

    return scores


def _match_filters(metadata: dict[str, Any], filters: RetrievalFilters | None) -> bool:
    if filters is None:
        return True

    entity_id = str(filters.entity_id or "").strip()
    entity_name = str(filters.entity_name or "").strip()
    source = str(filters.source or "").strip()
    section = str(filters.section or "").strip()
    date_from = str(filters.date_from or "").strip()
    date_to = str(filters.date_to or "").strip()

    if entity_id and str(metadata.get("entity_id") or "").strip() != entity_id:
        return False
    if entity_name and str(metadata.get("entity_name") or "").strip() != entity_name:
        return False
    if source and str(metadata.get("source") or "").strip() != source:
        return False
    if section and str(metadata.get("section") or "").strip() != section:
        return False

    doc_date = str(metadata.get("date") or metadata.get("report_date") or "").strip()
    if date_from and (not doc_date or doc_date < date_from):
        return False
    if date_to and (not doc_date or doc_date > date_to):
        return False

    return True


class _Bm25Index:
    """基于 rank_bm25 的简易索引，按 filter 缓存。"""

    def __init__(self) -> None:
        self._cache_key: tuple[Any, ...] | None = None
        self._ids: list[str] = []
        self._documents: dict[str, str] = {}
        self._metadatas: dict[str, dict[str, Any]] = {}
        self._bm25: Any | None = None

    def build(self, documents: list[SearchResult]) -> None:
        from rank_bm25 import BM25Okapi

        self._ids = [item["id"] for item in documents]
        self._documents = {item["id"]: item["document"] for item in documents}
        self._metadatas = {item["id"]: dict(item.get("metadata") or {}) for item in documents}

        tokenized = [_tokenize(self._documents[doc_id]) for doc_id in self._ids]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._ids:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(self._ids, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(doc_id, float(score)) for doc_id, score in ranked[:top_k] if score > 0]


class HybridRetrievalEngine:
    """
    Hybrid 检索引擎：Embedding + BM25 + RRF。

    与 RetrievalEngine 接口兼容，额外支持 weight 与结构化过滤。
    """

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        store: ChromaStore | None = None,
        *,
        top_k: int = TOP_K,
        embedding_weight: float = HYBRID_EMBEDDING_WEIGHT,
        bm25_weight: float = HYBRID_BM25_WEIGHT,
        rrf_k: int = HYBRID_RRF_K,
        candidate_k: int = HYBRID_CANDIDATE_K,
        model_name: str = EMBEDDING_MODEL,
        persist_directory: Path | str | None = None,
        collection_name: str = COLLECTION_NAME,
        enable_rerank: bool | None = None,
        rerank_candidate_k: int = RETRIEVAL_TOP_K,
        rerank_model_name: str = RERANK_MODEL,
    ) -> None:
        self.top_k = top_k
        self.embedding_weight = embedding_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.enable_rerank = ENABLE_RERANK if enable_rerank is None else enable_rerank
        self.rerank_candidate_k = rerank_candidate_k
        self.rerank_model_name = rerank_model_name

        self.embedder = embedder or TextEmbedder.get_instance(model_name=model_name)
        self.store = store or ChromaStore(
            persist_directory=persist_directory or CHROMA_DIR,
            collection_name=collection_name,
        )
        self._bm25_index = _Bm25Index()
        self._bm25_cache_key: tuple[Any, ...] | None = None
        self._reranker: Reranker | None = None

        embed_w, bm25_w = _normalize_weights(self.embedding_weight, self.bm25_weight)
        logger.info(
            "HybridRetrievalEngine 就绪: collection=%s, top_k=%d, "
            "weights=(embedding=%.2f, bm25=%.2f), rrf_k=%d, candidate_k=%d, rerank=%s",
            self.store.collection_name,
            self.top_k,
            embed_w,
            bm25_w,
            self.rrf_k,
            self.candidate_k,
            self.enable_rerank,
        )
        if self.enable_rerank:
            logger.info(
                "Hybrid Rerank 流程: RRF Top%d → CrossEncoder → Top%d (model=%s)",
                self.rerank_candidate_k,
                self.top_k,
                self.rerank_model_name,
            )

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker.get_instance(
                model_name=self.rerank_model_name
            )
        return self._reranker

    def _should_rerank(self, rerank: bool | None) -> bool:
        if rerank is not None:
            return rerank
        return self.enable_rerank

    def _resolve_candidate_k(self, top_k: int) -> int:
        return max(top_k * 3, self.candidate_k, top_k)

    def _cache_key(self, where: dict[str, Any] | None) -> tuple[Any, ...]:
        return (self.store.collection_name, self.store.count(), repr(where))

    def _ensure_bm25_index(self, where: dict[str, Any] | None) -> None:
        cache_key = self._cache_key(where)
        if cache_key == self._bm25_cache_key and self._bm25_index._bm25 is not None:
            return

        documents = self.store.get_documents(where=where)
        self._bm25_index.build(documents)
        self._bm25_cache_key = cache_key
        logger.info("BM25 索引已构建: %d 篇文档", len(documents))

    def _embedding_search(
        self,
        query_text: str,
        candidate_k: int,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed(query_text)
        raw_results = self.store.query(
            query_embedding,
            top_k=candidate_k,
            where=where,
        )
        results = [_to_retrieval_result(item) for item in raw_results]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def _bm25_search(
        self,
        query_text: str,
        candidate_k: int,
        *,
        where: dict[str, Any] | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[tuple[str, float]]:
        self._ensure_bm25_index(where)
        ranked = self._bm25_index.search(query_text, top_k=candidate_k)

        if filters is None:
            return ranked

        filtered: list[tuple[str, float]] = []
        for doc_id, score in ranked:
            metadata = self._bm25_index._metadatas.get(doc_id, {})
            if _match_filters(metadata, filters):
                filtered.append((doc_id, score))
        return filtered

    def _build_hybrid_result(
        self,
        doc_id: str,
        *,
        rrf_score: float,
        embedding_result: RetrievalResult | None,
        bm25_score: float | None,
        embedding_rank: int | None,
        bm25_rank: int | None,
    ) -> HybridRetrievalResult:
        if embedding_result is not None:
            text = embedding_result["text"]
            metadata = dict(embedding_result["metadata"])
            distance = embedding_result["distance"]
            embedding_score = embedding_result["score"]
        else:
            text = self._bm25_index._documents.get(doc_id, "")
            metadata = dict(self._bm25_index._metadatas.get(doc_id, {}))
            distance = 0.0
            embedding_score = None

        result: HybridRetrievalResult = {
            "id": doc_id,
            "text": text,
            "score": rrf_score,
            "similarity": rrf_score,
            "metadata": metadata,
            "distance": distance,
            "rrf_score": rrf_score,
            "embedding_score": embedding_score,
            "bm25_score": bm25_score,
            "embedding_rank": embedding_rank,
            "bm25_rank": bm25_rank,
        }
        return result

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        embedding_weight: float | None = None,
        bm25_weight: float | None = None,
        entity_name: str | None = None,
        entity_id: str | None = None,
        source: str | None = None,
        section: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        where: dict[str, Any] | None = None,
        filters: RetrievalFilters | None = None,
        rerank: bool | None = None,
    ) -> list[HybridRetrievalResult]:
        """
        Hybrid 检索：Embedding + BM25 + RRF 融合。

        Args:
            question: 用户问题。
            top_k: 返回条数，默认 self.top_k。
            embedding_weight: Embedding 路权重；默认 self.embedding_weight。
            bm25_weight: BM25 路权重；默认 self.bm25_weight。
            entity_name / entity_id / source / date_from / date_to: 结构化过滤。
            where: 直接传给 Chroma 的 where；若与 filters 同时存在则合并。
            filters: 结构化过滤对象；优先级低于显式参数时会补全。

        Returns:
            按 RRF 分数降序排列的结果列表。
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        query_text = question.strip()
        final_k = top_k if top_k is not None else self.top_k
        if final_k <= 0:
            raise ValueError(f"top_k 必须大于 0，当前为 {final_k}")

        use_rerank = self._should_rerank(rerank)
        rrf_top_k = max(final_k, self.rerank_candidate_k) if use_rerank else final_k

        embed_w, bm25_w = _normalize_weights(
            embedding_weight if embedding_weight is not None else self.embedding_weight,
            bm25_weight if bm25_weight is not None else self.bm25_weight,
        )

        active_filters = _build_filters(
            filters,
            entity_name=entity_name,
            entity_id=entity_id,
            source=source,
            section=section,
            date_from=date_from,
            date_to=date_to,
        )

        filter_where = build_where_filter(active_filters)
        if where and filter_where:
            chroma_where: dict[str, Any] = {"$and": [where, filter_where]}
        else:
            chroma_where = where or filter_where

        candidate_k = self._resolve_candidate_k(max(final_k, rrf_top_k))

        logger.info(
            "Hybrid 检索: question=%r, top_k=%d, rrf_top_k=%d, candidate_k=%d, "
            "weights=(%.2f, %.2f), rerank=%s, filters=%s",
            query_text[:80],
            final_k,
            rrf_top_k,
            candidate_k,
            embed_w,
            bm25_w,
            use_rerank,
            active_filters,
        )

        try:
            embedding_results = self._embedding_search(
                query_text,
                candidate_k,
                where=chroma_where,
            )
            bm25_ranked = self._bm25_search(
                query_text,
                candidate_k,
                where=chroma_where,
                filters=active_filters,
            )
        except ValueError:
            raise
        except ChromaStoreError as exc:
            raise RetrievalError(f"Hybrid 检索失败: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(f"Hybrid 检索过程异常: {exc}") from exc

        embedding_ids = [item["id"] for item in embedding_results]
        bm25_ids = [doc_id for doc_id, _ in bm25_ranked]

        rrf_scores = _rrf_fusion(
            [embedding_ids, bm25_ids],
            [embed_w, bm25_w],
            rrf_k=self.rrf_k,
        )

        if not rrf_scores:
            logger.info("Hybrid 检索完成: 无命中")
            return []

        embedding_map = {item["id"]: item for item in embedding_results}
        embedding_rank_map = {doc_id: rank for rank, doc_id in enumerate(embedding_ids, start=1)}
        bm25_rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_ranked, start=1)}
        bm25_score_map = {doc_id: score for doc_id, score in bm25_ranked}

        fused_ids = sorted(rrf_scores.keys(), key=lambda doc_id: rrf_scores[doc_id], reverse=True)
        results: list[HybridRetrievalResult] = []

        for doc_id in fused_ids[:rrf_top_k]:
            results.append(
                self._build_hybrid_result(
                    doc_id,
                    rrf_score=rrf_scores[doc_id],
                    embedding_result=embedding_map.get(doc_id),
                    bm25_score=bm25_score_map.get(doc_id),
                    embedding_rank=embedding_rank_map.get(doc_id),
                    bm25_rank=bm25_rank_map.get(doc_id),
                )
            )

        logger.info("Hybrid RRF 融合: %d 条候选", len(results))

        if use_rerank and results:
            reranked = apply_rerank_to_results(
                self.reranker,
                query_text,
                results,  # type: ignore[arg-type]
                top_k=final_k,
            )
            results = [item for item in reranked]  # type: ignore[misc]
            logger.info("Hybrid Rerank 完成: 返回 %d 条", len(results))
        else:
            results = results[:final_k]
            logger.info("Hybrid 检索完成: 返回 %d 条结果", len(results))

        return results


def main() -> None:
    """命令行调试入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid 检索调试")
    parser.add_argument("question", nargs="?", default="招商银行净利润情况如何？")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K)
    parser.add_argument("--embedding-weight", type=float, default=HYBRID_EMBEDDING_WEIGHT)
    parser.add_argument("--bm25-weight", type=float, default=HYBRID_BM25_WEIGHT)
    parser.add_argument("--entity-name", default="")
    parser.add_argument("--entity-id", default="")
    parser.add_argument("--source", choices=["pdf", "news"], default="")
    parser.add_argument("--date-from", default="")
    parser.add_argument("--date-to", default="")
    args = parser.parse_args()

    from src.utils.query_filters import resolve_entity_filters

    resolved_name, resolved_id, auto_applied = resolve_entity_filters(
        args.question,
        entity_name=args.entity_name or None,
        entity_id=args.entity_id or None,
        auto_entity=True,
    )
    if auto_applied and resolved_name:
        print(f"自动识别实体过滤: {resolved_name}")

    engine = HybridRetrievalEngine(
        top_k=args.top_k,
        embedding_weight=args.embedding_weight,
        bm25_weight=args.bm25_weight,
    )
    results = engine.retrieve(
        args.question,
        top_k=args.top_k,
        entity_name=resolved_name,
        entity_id=resolved_id,
        source=args.source or None,
        date_from=args.date_from or None,
        date_to=args.date_to or None,
    )

    print(f"问题: {args.question}")
    print(f"Collection: {engine.store.collection_name} ({engine.store.count()} 条)")
    print(f"返回 {len(results)} 条结果:\n")

    for index, item in enumerate(results, start=1):
        meta = item["metadata"]
        print(f"--- [{index}] rrf_score={item['rrf_score']:.4f} ---")
        print(f"  embedding_rank/score: {item.get('embedding_rank')} / {item.get('embedding_score')}")
        print(f"  bm25_rank/score:      {item.get('bm25_rank')} / {item.get('bm25_score')}")
        print(f"  entity: {meta.get('entity_name')} ({meta.get('entity_id')})")
        print(f"  source: {meta.get('source')} | date: {meta.get('date')}")
        print(f"  text: {item['text'][:120]}...\n")


if __name__ == "__main__":
    main()
