"""
检索引擎：将用户问题向量化，从 ChromaDB 检索最相关 Token。

可选两阶段检索：Embedding Top-N → CrossEncoder Rerank → Top-M。
不直接调用 Chroma API，统一通过 TextEmbedder 与 ChromaStore 协作。
"""

from __future__ import annotations

import sys
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
    RERANK_MODEL,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    TOP_K,
    setup_logging,
)
from src.embeddings.text_embedding import TextEmbedder
from src.reranker.base import Reranker, apply_rerank_to_results
from src.reranker.cross_encoder import CrossEncoderReranker
from src.vectorstore.chroma_store import ChromaStore, ChromaStoreError, SearchResult

logger = setup_logging(__name__)


class RetrievalError(Exception):
    """检索流程失败时抛出。"""


class RetrievalResult(TypedDict):
    """单条检索结果。"""

    id: str
    text: str
    score: float
    similarity: float
    metadata: dict[str, Any]
    distance: float
    rerank_score: NotRequired[float | None]


def _distance_to_similarity(distance: float) -> float:
    """
    将 Chroma 距离转为 [0, 1] 相似度，越大表示越相关。

    使用 1 / (1 + distance)，与距离单调递减，且对 L2 / 余弦距离均适用。
    """
    return max(0.0, 1.0 / (1.0 + distance))


def _to_retrieval_result(item: SearchResult) -> RetrievalResult:
    distance = float(item["distance"])
    score = _distance_to_similarity(distance)
    return {
        "id": item["id"],
        "text": item["document"],
        "score": score,
        "similarity": score,
        "metadata": dict(item.get("metadata") or {}),
        "distance": distance,
    }


def _format_result_line(index: int, item: RetrievalResult, *, show_rerank: bool) -> str:
    meta = item["metadata"]
    title = meta.get("title", "")
    chunk_index = meta.get("chunk_index", "")
    parts = [
        f"  [{index}] score={item['score']:.4f}",
        f"id={item['id']}",
    ]
    if title:
        parts.append(f"title={title}")
    if chunk_index != "":
        parts.append(f"chunk={chunk_index}")
    if show_rerank and item.get("rerank_score") is not None:
        parts.insert(1, f"rerank_score={item['rerank_score']:.4f}")
    return " | ".join(parts)


def _log_ranking(title: str, results: list[RetrievalResult], *, show_rerank: bool = False) -> None:
    lines = [title]
    for index, item in enumerate(results, start=1):
        lines.append(_format_result_line(index, item, show_rerank=show_rerank))
    logger.info("\n".join(lines))


class RetrievalEngine:
    """基于向量相似度的 Token 检索引擎，可选 CrossEncoder 重排序。"""

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        store: ChromaStore | None = None,
        reranker: Reranker | None = None,
        *,
        top_k: int = TOP_K,
        model_name: str = EMBEDDING_MODEL,
        persist_directory: Path | str | None = None,
        collection_name: str = COLLECTION_NAME,
        enable_rerank: bool | None = None,
        retrieval_top_k: int = RETRIEVAL_TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
        rerank_model_name: str = RERANK_MODEL,
    ) -> None:
        self.top_k = top_k
        self.enable_rerank = ENABLE_RERANK if enable_rerank is None else enable_rerank
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.rerank_model_name = rerank_model_name

        self.embedder = embedder or TextEmbedder.get_instance(model_name=model_name)
        self.store = store or ChromaStore(
            persist_directory=persist_directory or CHROMA_DIR,
            collection_name=collection_name,
        )
        self._reranker = reranker

        logger.info(
            "RetrievalEngine 就绪: collection=%s, top_k=%d, model=%s, rerank=%s",
            self.store.collection_name,
            self.top_k,
            self.embedder.model_name,
            self.enable_rerank,
        )
        if self.enable_rerank:
            logger.info(
                "Rerank 流程: Embedding Top%d → CrossEncoder → Top%d (model=%s)",
                self.retrieval_top_k,
                self.rerank_top_k,
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
        """解析本次检索是否启用 Rerank（实例配置 vs 调用参数）。"""
        if rerank is not None:
            return rerank
        return self.enable_rerank

    def _embedding_retrieve(
        self,
        query_text: str,
        k: int,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed(query_text)
        raw_results = self.store.query(
            query_embedding,
            top_k=k,
            where=where,
        )
        results = [_to_retrieval_result(item) for item in raw_results]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def _apply_rerank(
        self,
        query_text: str,
        candidates: list[RetrievalResult],
        final_k: int,
    ) -> list[RetrievalResult]:
        _log_ranking("原始排序", candidates)

        reranked = apply_rerank_to_results(
            self.reranker,
            query_text,
            candidates,
            top_k=final_k,
        )
        logger.info("rerank后数量: %d", len(reranked))

        logger.info("↓")
        typed_results: list[RetrievalResult] = [
            item  # type: ignore[misc]
            for item in reranked
        ]
        _log_ranking("重排序", typed_results, show_rerank=True)
        return typed_results

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        rerank: bool | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        检索与用户问题最相关的 Token。

        启用 Rerank 时：Embedding 召回 Top retrieval_top_k，CrossEncoder 精排后返回 Top final_k。
        关闭 Rerank 时：直接向量化检索 Top final_k（与原有行为一致）。

        Args:
            question: 用户问题或查询文本。
            top_k: 最终返回条数；Rerank 关闭时默认 self.top_k，开启时默认 self.rerank_top_k。
            rerank: 是否启用 CrossEncoder 重排序。
                None 时使用实例配置 enable_rerank；True/False 覆盖实例配置。
            where: 可选 metadata 过滤条件，传给 ChromaStore.query。

        Returns:
            检索结果列表。每项含 score（Embedding 分数）、similarity（同 score，兼容旧字段）、
            以及可选 rerank_score（启用 Rerank 时）。

        Raises:
            ValueError: 问题为空。
            RetrievalError: 向量化或检索失败。
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        query_text = question.strip()
        use_rerank = self._should_rerank(rerank)

        if use_rerank:
            candidate_k = self.retrieval_top_k
            final_k = top_k if top_k is not None else self.rerank_top_k
        else:
            candidate_k = top_k if top_k is not None else self.top_k
            final_k = candidate_k

        if candidate_k <= 0 or final_k <= 0:
            raise ValueError(
                f"top_k 必须大于 0，当前 candidate_k={candidate_k}, final_k={final_k}"
            )

        if use_rerank:
            logger.info(
                "开始检索(Rerank): question=%r, candidate_k=%d, final_k=%d",
                query_text[:80],
                candidate_k,
                final_k,
            )
        else:
            logger.info("开始检索: question=%r, top_k=%d", query_text[:80], final_k)

        try:
            candidates = self._embedding_retrieve(query_text, candidate_k, where=where)
            logger.info("向量召回数量: %d", len(candidates))

            if use_rerank and candidates:
                results = self._apply_rerank(query_text, candidates, final_k)
            elif use_rerank and not candidates:
                logger.warning("向量召回为 0，跳过 rerank（请检查过滤条件或索引覆盖）")
                results = []
            else:
                results = candidates[:final_k]
                logger.info("rerank后数量: %d (未启用 rerank)", len(results))
        except ValueError:
            raise
        except ChromaStoreError as exc:
            raise RetrievalError(f"向量检索失败: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(f"检索过程异常: {exc}") from exc

        logger.info("检索完成: 返回 %d 条结果 (rerank=%s)", len(results), use_rerank)
        if results:
            pdf_count = sum(
                1 for item in results if str(item["metadata"].get("source")) == "pdf"
            )
            news_count = sum(
                1 for item in results if str(item["metadata"].get("source")) == "news"
            )
            if pdf_count or news_count:
                logger.info(
                    "混合检索: 财报 %d 条, 新闻 %d 条",
                    pdf_count,
                    news_count,
                )
        return results


def main() -> None:
    """命令行调试入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="检索与用户问题相关的 Token")
    parser.add_argument("question", nargs="?", default="招商银行净利润情况如何？")
    parser.add_argument("-k", "--top-k", type=int, default=None)
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="启用 CrossEncoder 重排序 Top20→Top5（也可 FINANCIAL_POC_ENABLE_RERANK=true）",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="关闭 CrossEncoder 重排序",
    )
    args = parser.parse_args()

    use_rerank: bool | None = None
    if args.rerank:
        use_rerank = True
    elif args.no_rerank:
        use_rerank = False

    engine = RetrievalEngine(
        top_k=args.top_k or TOP_K,
        enable_rerank=use_rerank if use_rerank is not None else ENABLE_RERANK,
    )
    results = engine.retrieve(args.question, top_k=args.top_k, rerank=use_rerank)

    print(f"问题: {args.question}")
    print(f"Collection: {engine.store.collection_name} ({engine.store.count()} 条)")
    print(f"Rerank: {engine.enable_rerank}")
    print(f"返回 {len(results)} 条结果:\n")

    for index, item in enumerate(results, start=1):
        meta = item["metadata"]
        title = meta.get("title", "")
        line = f"--- [{index}] score={item['score']:.4f}"
        if item.get("rerank_score") is not None:
            line += f" rerank_score={item['rerank_score']:.4f}"
        print(line + " ---")
        print(f"id:       {item['id']}")
        if title:
            print(f"title:    {title}")
        print(f"text:     {item['text'][:200]}...")
        print(f"metadata: {meta}\n")


if __name__ == "__main__":
    main()
