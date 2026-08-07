"""
文本向量化：使用 sentence-transformers 将文本转为 embedding 向量。

模型默认使用 config.EMBEDDING_MODEL，GPU 可用时自动使用 CUDA，无需 OpenAI API。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import EMBEDDING_MODEL, HF_OFFLINE, setup_logging

logger = setup_logging(__name__)

DEFAULT_MODEL_NAME = EMBEDDING_MODEL


def _resolve_device() -> str:
    """检测可用设备，优先 GPU。"""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        logger.debug("未安装 torch，使用 CPU")
    return "cpu"


class TextEmbedder:
    """文本 Embedding 封装，基于 sentence-transformers（单例，按 model_name 复用）。"""

    _instances: dict[str, TextEmbedder] = {}
    _lock = threading.Lock()

    def __new__(cls, model_name: str = DEFAULT_MODEL_NAME) -> TextEmbedder:
        normalized_name = model_name or DEFAULT_MODEL_NAME
        with cls._lock:
            cached = cls._instances.get(normalized_name)
            if cached is not None:
                logger.info("Embedding model loaded from cache.")
                return cached

            instance = super().__new__(cls)
            cls._instances[normalized_name] = instance
            instance._initialized = False
            return instance

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        if getattr(self, "_initialized", False):
            return

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.device = _resolve_device()

        logger.info("加载 Embedding 模型: %s | device=%s", self.model_name, self.device)
        load_kwargs: dict = {"device": self.device}
        if HF_OFFLINE:
            load_kwargs["local_files_only"] = True
        try:
            self._model = SentenceTransformer(self.model_name, **load_kwargs)
        except Exception as exc:
            if not HF_OFFLINE:
                raise
            logger.warning("离线加载失败 (%s)，尝试联网下载...", exc)
            self._model = SentenceTransformer(self.model_name, device=self.device)

        self.embedding_dim = (
            self._model.get_embedding_dimension()
            if hasattr(self._model, "get_embedding_dimension")
            else self._model.get_sentence_embedding_dimension()
        )
        logger.info("Embedding 向量维度: %d", self.embedding_dim)
        self._initialized = True

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL_NAME) -> TextEmbedder:
        """
        获取 TextEmbedder 单例。

        同一 model_name 在整个程序生命周期内只加载一次模型。
        """
        return cls(model_name=model_name)

    @classmethod
    def clear_cache(cls) -> None:
        """清除已缓存实例（主要用于测试）。"""
        with cls._lock:
            cls._instances.clear()

    def embed(self, text: str) -> list[float]:
        """
        将单条文本转为 embedding 向量。

        Args:
            text: 待向量化文本。

        Returns:
            浮点向量列表。
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")

        vector = self._model.encode(
            text.strip(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        批量将文本转为 embedding 向量。

        Args:
            texts: 待向量化文本列表。
            batch_size: 批大小，默认 32。

        Returns:
            与输入等长的向量列表。
        """
        if not texts:
            return []

        if batch_size <= 0:
            raise ValueError(f"batch_size 必须大于 0，当前为 {batch_size}")

        normalized = [text.strip() if text else "" for text in texts]

        vectors = self._model.encode(
            normalized,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]


def main() -> None:
    """命令行调试入口。"""
    embedder = TextEmbedder()

    samples = [
        "贵州茅台2024年第三季度业绩超预期。",
        "招商银行净利润同比增长，资产质量保持稳定。",
    ]

    single = embedder.embed(samples[0])
    batch = embedder.embed_batch(samples)

    print(f"模型: {embedder.model_name}")
    print(f"设备: {embedder.device}")
    print(f"向量维度: {embedder.embedding_dim}")
    print(f"单条向量长度: {len(single)}")
    print(f"批量向量数量: {len(batch)}")
    print(f"单条向量前 5 维: {single[:5]}")


if __name__ == "__main__":
    main()
