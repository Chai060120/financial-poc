"""
文本清洗器：去除多余空白、连续换行与不可见字符。

仅处理纯文本字符串，不负责分块或结构化。
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import setup_logging

logger = setup_logging(__name__)

# 零宽字符、BOM、软连字符等不可见符号
INVISIBLE_CHARS = re.compile(
    r"[\u200b-\u200d\ufeff\u00ad\u2060\ufffe\u180e]"
)

# 控制字符（保留 \n、\t）
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# 行内连续空白
INLINE_WHITESPACE = re.compile(r"[ \t\f\v]+")

# 连续换行
MULTI_NEWLINES = re.compile(r"\n{2,}")


def clean_text(text: str) -> str:
    """
    清洗文本并返回字符串。

    处理步骤：
    1. Unicode 规范化（NFKC）
    2. 统一换行符，替换不间断空格
    3. 移除不可见字符与控制字符
    4. 压缩行内多余空白
    5. 合并连续换行为单个换行
    6. 去除首尾空白

    Args:
        text: 原始文本。

    Returns:
        清洗后的文本；输入为空或仅空白时返回空字符串。
    """
    if not text:
        return ""

    original_len = len(text)
    cleaned = unicodedata.normalize("NFKC", text)

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = cleaned.replace("\u2028", "\n").replace("\u2029", "\n")

    cleaned = INVISIBLE_CHARS.sub("", cleaned)
    cleaned = CONTROL_CHARS.sub("", cleaned)

    lines = []
    for line in cleaned.split("\n"):
        line = INLINE_WHITESPACE.sub(" ", line).strip()
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = MULTI_NEWLINES.sub("\n", cleaned)
    cleaned = cleaned.strip()

    logger.debug(
        "文本清洗完成: 原始 %d 字符 -> 清洗后 %d 字符",
        original_len,
        len(cleaned),
    )
    return cleaned


def main() -> None:
    """命令行调试入口。"""
    samples = [
        "贵州  茅台\r\n\r\n\r\n2024年\r\n\t业绩  公告",
        "标题\u200b内容\u00a0\u00a0段落",
        "",
    ]

    for sample in samples:
        result = clean_text(sample)
        print("---")
        print("原始:", repr(sample))
        print("清洗:", repr(result))


if __name__ == "__main__":
    main()
