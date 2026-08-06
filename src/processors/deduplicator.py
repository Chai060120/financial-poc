"""
Token 去重器：基于 id、文本 hash、标题组合检测并去除重复 Token。

适用于 PDF 分块、新闻、公告等多源数据的增量合并与去重。
"""

from __future__ import annotations

import hashlib
import re
import sys
from enum import Flag, auto
from pathlib import Path
from typing import Any, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import setup_logging
from src.processors.text_cleaner import clean_text
from src.processors.tokenizer import Token

logger = setup_logging(__name__)

TITLE_NORMALIZE_PATTERN = re.compile(r"\s+")


class DedupReason(Flag):
    """去重命中原因（可组合）。"""

    NONE = 0
    ID = auto()
    TEXT_HASH = auto()
    TITLE_TEXT = auto()


class DedupStats(TypedDict):
    """去重统计信息。"""

    input_count: int
    output_count: int
    removed_count: int
    removed_by_id: int
    removed_by_text_hash: int
    removed_by_title_text: int


class DuplicateInfo(TypedDict):
    """单条重复 Token 的判定信息。"""

    token_id: str
    reasons: list[str]
    duplicate_of: str


def normalize_title(title: str) -> str:
    """规范化标题，便于比较。"""
    if not title:
        return ""
    cleaned = clean_text(title)
    return TITLE_NORMALIZE_PATTERN.sub(" ", cleaned).strip().casefold()


def compute_text_hash(text: str) -> str:
    """
    计算文本 hash（清洗后 SHA256 前 16 位）。

    对清洗后的正文做 hash，避免空白差异导致误判。
    """
    normalized = clean_text(text)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def get_token_title(token: Token | dict[str, Any]) -> str:
    """从 Token 提取 title。"""
    metadata = token.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("title") or "")


def build_title_text_key(token: Token | dict[str, Any], text_hash: str | None = None) -> tuple[str, str]:
    """
    构造 (规范化标题, 文本 hash) 组合键。

    用于同一标题下重复段落的检测（新闻、公告场景）。
    """
    title_key = normalize_title(get_token_title(token))
    hash_value = text_hash or compute_text_hash(str(token.get("text", "")))
    return title_key, hash_value


class TokenDeduplicator:
    """
    Token 去重器。

    维护已见 id、文本 hash、标题+文本 hash 三类索引，支持增量注册与批量去重。
    """

    def __init__(
        self,
        *,
        check_id: bool = True,
        check_text_hash: bool = True,
        check_title_text: bool = True,
    ) -> None:
        self.check_id = check_id
        self.check_text_hash = check_text_hash
        self.check_title_text = check_title_text

        self._seen_ids: set[str] = set()
        self._seen_text_hashes: dict[str, str] = {}
        self._seen_title_text: dict[tuple[str, str], str] = {}

    def clear(self) -> None:
        """清空全部去重索引。"""
        self._seen_ids.clear()
        self._seen_text_hashes.clear()
        self._seen_title_text.clear()

    def register_existing(self, tokens: list[Token | dict[str, Any]]) -> None:
        """将已有 Token 注册进索引（不判定为重复）。"""
        for token in tokens:
            self._register(token, allow_duplicate=True)

    def check_duplicate(self, token: Token | dict[str, Any]) -> DedupReason:
        """
        检测 Token 是否重复，不修改内部索引。

        Returns:
            DedupReason 标志位，0 表示不重复。
        """
        reasons = DedupReason.NONE
        token_id = str(token.get("id", ""))
        text_hash = compute_text_hash(str(token.get("text", "")))

        if self.check_id and token_id and token_id in self._seen_ids:
            reasons |= DedupReason.ID

        if self.check_text_hash and text_hash and text_hash in self._seen_text_hashes:
            reasons |= DedupReason.TEXT_HASH

        if self.check_title_text:
            title_key, title_text_hash = build_title_text_key(token, text_hash)
            if title_key and title_text_hash:
                composite = (title_key, title_text_hash)
                if composite in self._seen_title_text:
                    reasons |= DedupReason.TITLE_TEXT

        return reasons

    def _register(
        self,
        token: Token | dict[str, Any],
        *,
        allow_duplicate: bool = False,
    ) -> None:
        token_id = str(token.get("id", ""))
        text = str(token.get("text", ""))
        text_hash = compute_text_hash(text)

        if token_id:
            self._seen_ids.add(token_id)

        if text_hash:
            if allow_duplicate or text_hash not in self._seen_text_hashes:
                self._seen_text_hashes[text_hash] = token_id

        title_key, title_text_hash = build_title_text_key(token, text_hash)
        if title_key and title_text_hash:
            composite = (title_key, title_text_hash)
            if allow_duplicate or composite not in self._seen_title_text:
                self._seen_title_text[composite] = token_id

    def add(self, token: Token | dict[str, Any]) -> tuple[bool, DedupReason]:
        """
        尝试添加 Token。

        Returns:
            (是否保留, 重复原因)。保留时同步更新索引。
        """
        reasons = self.check_duplicate(token)
        if reasons != DedupReason.NONE:
            return False, reasons

        self._register(token)
        return True, DedupReason.NONE

    def deduplicate(
        self,
        tokens: list[Token | dict[str, Any]],
    ) -> tuple[list[Token | dict[str, Any]], DedupStats]:
        """
        对 Token 列表去重，保留首次出现的条目。

        Returns:
            (去重后列表, 统计信息)
        """
        stats: DedupStats = {
            "input_count": len(tokens),
            "output_count": 0,
            "removed_count": 0,
            "removed_by_id": 0,
            "removed_by_text_hash": 0,
            "removed_by_title_text": 0,
        }

        unique: list[Token | dict[str, Any]] = []

        for token in tokens:
            kept, reasons = self.add(token)
            if kept:
                unique.append(token)
                continue

            stats["removed_count"] += 1
            if reasons & DedupReason.ID:
                stats["removed_by_id"] += 1
            if reasons & DedupReason.TEXT_HASH:
                stats["removed_by_text_hash"] += 1
            if reasons & DedupReason.TITLE_TEXT:
                stats["removed_by_title_text"] += 1

            logger.debug(
                "去重跳过: id=%s, reasons=%s",
                token.get("id", ""),
                reasons,
            )

        stats["output_count"] = len(unique)
        logger.info(
            "去重完成: 输入 %d, 输出 %d, 移除 %d (id=%d, hash=%d, title+text=%d)",
            stats["input_count"],
            stats["output_count"],
            stats["removed_count"],
            stats["removed_by_id"],
            stats["removed_by_text_hash"],
            stats["removed_by_title_text"],
        )
        return unique, stats


def deduplicate_tokens(
    tokens: list[Token | dict[str, Any]],
    existing: list[Token | dict[str, Any]] | None = None,
    *,
    check_id: bool = True,
    check_text_hash: bool = True,
    check_title_text: bool = True,
) -> tuple[list[Token | dict[str, Any]], DedupStats]:
    """
    便捷接口：对 Token 列表去重，可选与已有 Token 合并判重。

    Args:
        tokens: 待去重的 Token 列表。
        existing: 已存在 Token，参与判重但不出现在返回结果中。
        check_id: 是否按 Token id 去重。
        check_text_hash: 是否按文本 hash 去重。
        check_title_text: 是否按标题+文本 hash 去重。

    Returns:
        (去重后列表, 统计信息)
    """
    deduplicator = TokenDeduplicator(
        check_id=check_id,
        check_text_hash=check_text_hash,
        check_title_text=check_title_text,
    )

    if existing:
        deduplicator.register_existing(existing)

    return deduplicator.deduplicate(tokens)


def find_duplicates(
    tokens: list[Token | dict[str, Any]],
) -> list[DuplicateInfo]:
    """
    扫描列表内部重复项，返回重复 Token 及原因（不修改输入）。

    用于调试与质量检查。
    """
    deduplicator = TokenDeduplicator()
    duplicates: list[DuplicateInfo] = []

    for token in tokens:
        reasons = deduplicator.check_duplicate(token)
        if reasons != DedupReason.NONE:
            reason_names = [
                name
                for name, flag in (
                    ("id", DedupReason.ID),
                    ("text_hash", DedupReason.TEXT_HASH),
                    ("title_text", DedupReason.TITLE_TEXT),
                )
                if reasons & flag
            ]
            duplicate_of = ""
            token_id = str(token.get("id", ""))
            text_hash = compute_text_hash(str(token.get("text", "")))

            if reasons & DedupReason.ID:
                duplicate_of = token_id
            elif reasons & DedupReason.TEXT_HASH:
                duplicate_of = deduplicator._seen_text_hashes.get(text_hash, "")
            elif reasons & DedupReason.TITLE_TEXT:
                key = build_title_text_key(token, text_hash)
                duplicate_of = deduplicator._seen_title_text.get(key, "")

            duplicates.append(
                {
                    "token_id": token_id,
                    "reasons": reason_names,
                    "duplicate_of": duplicate_of,
                }
            )
        else:
            deduplicator._register(token)

    return duplicates


def main() -> None:
    """命令行调试入口。"""
    from src.processors.tokenizer import build_news_tokens, build_pdf_tokens
    from src.processors.chunker import chunk_text
    from src.processors.text_cleaner import clean_text as clean

    pdf_chunks = chunk_text(clean("贵州茅台2024年业绩公告。" * 10))
    pdf_tokens = build_pdf_tokens(
        pdf_chunks,
        "report.pdf",
        metadata={"title": "2024年业绩公告"},
    )

    news_chunks = chunk_text(clean("茅台三季度业绩超预期。" * 10))
    news_tokens = build_news_tokens(
        news_chunks,
        {"title": "茅台三季度业绩超预期", "date": "2024-10-29"},
    )

    # 构造重复：相同 id、相同文本、相同标题+文本
    duplicated = [
        pdf_tokens[0],
        dict(pdf_tokens[0]),
        dict(pdf_tokens[1], text=pdf_tokens[0]["text"]),
        dict(news_tokens[0], id="duplicate_news_id"),
    ]

    all_tokens = pdf_tokens + news_tokens + duplicated
    unique, stats = deduplicate_tokens(all_tokens)

    print(f"输入: {stats['input_count']}, 输出: {stats['output_count']}")
    print(f"移除: id={stats['removed_by_id']}, hash={stats['removed_by_text_hash']}, "
          f"title+text={stats['removed_by_title_text']}")

    internal_dups = find_duplicates(all_tokens)
    print(f"内部重复项: {len(internal_dups)}")


if __name__ == "__main__":
    main()
