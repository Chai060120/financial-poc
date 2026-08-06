"""
文本分块器：按固定字符长度切分文本，支持重叠窗口。

为 RAG 检索准备 chunk 列表，不负责向量化或存储。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import CHUNK_OVERLAP, CHUNK_SIZE, setup_logging

logger = setup_logging(__name__)


def _validate_params(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须大于 0，当前为 {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap 不能为负数，当前为 {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap 必须小于 chunk_size（当前 overlap={overlap}, chunk_size={chunk_size}）"
        )


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    将文本按固定字符长度切分，相邻块之间保留重叠区域。

    采用滑动窗口策略：每次前进 ``chunk_size - overlap`` 个字符。
    空文本返回空列表；短于 chunk_size 的文本返回仅含一块的列表。

    Args:
        text: 待切分文本（通常为清洗后的正文）。
        chunk_size: 每块最大字符数，默认 500。
        overlap: 相邻块重叠字符数，默认 100。

    Returns:
        分块后的字符串列表。
    """
    _validate_params(chunk_size, overlap)

    if not text or not text.strip():
        return []

    content = text.strip()
    if len(content) <= chunk_size:
        logger.debug("文本长度 %d <= chunk_size，返回单块", len(content))
        return [content]

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]
        if chunk:
            chunks.append(chunk)

        if end >= len(content):
            break
        start += step

    logger.debug(
        "文本分块完成: 原文 %d 字符 -> %d 块 (size=%d, overlap=%d)",
        len(content),
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks


def main() -> None:
    """命令行调试入口。"""
    sample = "贵州茅台2024年业绩公告。" * 80  # 约 1120 字符

    chunks = chunk_text(sample)
    print(f"原文长度: {len(sample.strip())}")
    print(f"分块数量: {len(chunks)}")
    print(f"参数: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} | 长度 {len(chunk)} ---")
        print(chunk[:80], "..." if len(chunk) > 80 else "")

    if len(chunks) >= 2:
        overlap_len = len(set(chunks[0][-CHUNK_OVERLAP:]) & set(chunks[1][:CHUNK_OVERLAP]))
        print(f"\n第 1、2 块边界重叠区域约 {CHUNK_OVERLAP} 字符（设计值）")


if __name__ == "__main__":
    main()
