"""
CrossEncoder 重排序：对 Embedding 召回候选进行精排。

模型默认使用 config.RERANK_MODEL，GPU 可用时自动使用 CUDA。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import RERANK_MODEL, setup_logging

logger = setup_logging(__name__)

DEFAULT_MODEL_NAME = RERANK_MODEL


def _resolve_device() -> str:
    """检测可用设备，优先 GPU。"""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        logger.debug("未安装 torch，使用 CPU")
    return "cpu"


class CrossEncoderReranker:
    """CrossEncoder 重排序封装（单例，按 model_name 复用），实现 Reranker 协议。"""

    _instances: dict[str, CrossEncoderReranker] = {}
    _lock = threading.Lock()

    def __new__(cls, model_name: str = DEFAULT_MODEL_NAME) -> CrossEncoderReranker:
        normalized_name = model_name or DEFAULT_MODEL_NAME
        with cls._lock:
            cached = cls._instances.get(normalized_name)
            if cached is not None:
                logger.info("Reranker model loaded from cache.")
                return cached

            instance = super().__new__(cls)
            cls._instances[normalized_name] = instance
            instance._initialized = False
            return instance

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        if getattr(self, "_initialized", False):
            return

        from sentence_transformers import CrossEncoder

        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.device = _resolve_device()

        logger.info("加载 Reranker 模型: %s | device=%s", self.model_name, self.device)
        self._model = CrossEncoder(self.model_name, device=self.device)
        self._initialized = True

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL_NAME) -> CrossEncoderReranker:
        """获取 CrossEncoderReranker 单例。"""
        return cls(model_name=model_name)

    @classmethod
    def clear_cache(cls) -> None:
        """清除已缓存实例（主要用于测试）。"""
        with cls._lock:
            cls._instances.clear()

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        对 query-document 对进行重排序。

        Args:
            query: 用户问题。
            documents: 候选文档列表，与 Embedding 召回顺序一致。
            top_k: 返回前 K 条；None 表示返回全部（按分数降序）。

        Returns:
            (原始下标, rerank_score) 列表，按 rerank_score 降序。
        """
        if not query or not query.strip():
            raise ValueError("query 不能为空")
        if not documents:
            return []

        pairs = [[query.strip(), doc if doc else ""] for doc in documents]
        raw_scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            enumerate(float(score) for score in raw_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        if top_k is not None and top_k > 0:
            ranked = ranked[:top_k]
        return ranked


def main() -> None:
    """命令行调试入口。"""
    reranker = CrossEncoderReranker.get_instance()
    query = "招商银行净利润情况如何？"
    docs = [
        "招商银行2024年净利润同比增长12%。",
        "贵州茅台白酒销量稳步提升。",
        "中国平安保险业务保持增长。",
    ]

    ranked = reranker.rerank(query, docs)
    print(f"模型: {reranker.model_name}")
    print(f"设备: {reranker.device}")
    for rank, (index, score) in enumerate(ranked, start=1):
        print(f"[{rank}] doc_index={index} rerank_score={score:.4f} | {docs[index]}")


if __name__ == "__main__":
    main()
