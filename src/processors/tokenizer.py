"""
Token 构建器：将文本块转换为统一 Token 格式。
统一表示层的数据单元，供存储、向量化和 RAG 检索使用。
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import UNKNOWN_ENTITY_ID, UNKNOWN_ENTITY_NAME, setup_logging
from src.processors.report_parser import empty_report_metadata
from src.utils.entity_parser import normalize_entity_fields

logger = setup_logging(__name__)

TokenType = Literal["pdf", "news"]
SourceType = Literal["pdf", "news"]

TOKEN_TYPES: tuple[TokenType, ...] = ("pdf", "news")


class Token(TypedDict):
    """统一 Token 结构。"""

    id: str
    type: TokenType
    source: SourceType
    text: str
    metadata: dict[str, Any]


def _sanitize_key(value: str, max_length: int = 40) -> str:
    cleaned = re.sub(r"[^\w\-]", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:max_length] or "unknown"


def generate_token_id(
    entity_id: str,
    token_type: TokenType,
    source_key: str,
    chunk_index: int,
) -> str:
    """
    生成稳定的 Token ID。

    格式: {entity_id}_{type}_{source_hash}_{chunk_index}
    """
    safe_key = _sanitize_key(source_key)
    digest = hashlib.md5(source_key.encode("utf-8")).hexdigest()[:8]
    return f"{entity_id}_{token_type}_{safe_key}_{digest}_chunk_{chunk_index}"


def _resolve_entity_id(meta: dict[str, Any]) -> str:
    """从 metadata 解析用于 Token ID 的 entity 标识。"""
    entity_name, entity_id = normalize_entity_fields(
        str(meta.get("entity_name") or ""),
        str(meta.get("entity_id") or ""),
    )
    meta["entity_name"] = entity_name
    meta["entity_id"] = entity_id

    if entity_id and entity_id != UNKNOWN_ENTITY_ID:
        return _sanitize_key(entity_id.replace(".", "_"))
    if entity_name and entity_name != UNKNOWN_ENTITY_NAME:
        return _sanitize_key(entity_name)
    return _sanitize_key(UNKNOWN_ENTITY_ID)


def build_token(
    text: str,
    token_type: TokenType,
    source: SourceType,
    metadata: dict[str, Any],
    *,
    chunk_index: int = 0,
    source_key: str | None = None,
) -> Token:
    """
    由单个文本块构建 Token。

    Args:
        text: 分块后的正文。
        token_type: Token 类型，``pdf`` 或 ``news``。
        source: 数据来源，``pdf`` 或 ``news``。
        metadata: 附加元数据（entity_id、date、title 等）。
        chunk_index: 当前块序号。
        source_key: 用于生成 ID 的来源标识；默认取 metadata 中的 title 或 file_name。

    Returns:
        统一 Token 字典。
    """
    if token_type not in TOKEN_TYPES:
        raise ValueError(f"不支持的 token type: {token_type}")
    if source not in TOKEN_TYPES:
        raise ValueError(f"不支持的 source: {source}")
    if not text or not text.strip():
        raise ValueError("Token 文本不能为空")

    meta = dict(metadata)
    id_entity_key = _resolve_entity_id(meta)
    key = source_key or str(
        meta.get("file_name") or meta.get("title") or meta.get("record_id") or "document"
    )

    meta["chunk_index"] = chunk_index

    token: Token = {
        "id": generate_token_id(id_entity_key, token_type, key, chunk_index),
        "type": token_type,
        "source": source,
        "text": text.strip(),
        "metadata": meta,
    }
    return token


def chunks_to_tokens(
    chunks: list[str],
    token_type: TokenType,
    source: SourceType,
    metadata: dict[str, Any],
    *,
    source_key: str | None = None,
    chunk_metadatas: list[dict[str, Any]] | None = None,
) -> list[Token]:
    """将多个文本块批量转换为 Token 列表。"""
    if not chunks:
        return []

    total_chunks = len(chunks)
    tokens: list[Token] = []

    for index, chunk in enumerate(chunks):
        if not chunk or not chunk.strip():
            logger.debug("跳过空 chunk: index=%d", index)
            continue

        chunk_meta = dict(metadata)
        chunk_meta["total_chunks"] = total_chunks
        if chunk_metadatas and index < len(chunk_metadatas):
            chunk_meta.update(chunk_metadatas[index])

        token = build_token(
            chunk,
            token_type=token_type,
            source=source,
            metadata=chunk_meta,
            chunk_index=index,
            source_key=source_key,
        )
        tokens.append(token)

    logger.info(
        "Token 构建完成: type=%s, source=%s, 输入 %d 块 -> 输出 %d 个 Token",
        token_type,
        source,
        total_chunks,
        len(tokens),
    )
    return tokens


def build_pdf_tokens(
    chunks: list[str],
    pdf_path: str | Path,
    metadata: dict[str, Any] | None = None,
    *,
    chunk_metadatas: list[dict[str, Any]] | None = None,
) -> list[Token]:
    """
    从 PDF 分块构建 Token 列表。

    metadata 可包含: entity_id, entity_name, date, title 等。
    若未提供 entity 字段，优先从 PDF 文件名自动识别。
    会自动补充 file_name、file_path。
    """
    from src.utils.entity_parser import parse_filename, to_token_metadata

    path = Path(pdf_path).resolve()
    meta = dict(metadata or {})

    has_entity = bool(str(meta.get("entity_name") or "").strip()) or bool(
        str(meta.get("entity_id") or "").strip()
    )
    if not has_entity:
        parsed = parse_filename(path)
        meta.update(to_token_metadata(parsed, file_name=path.name, file_path=str(path)))

    meta.setdefault("file_name", path.name)
    meta.setdefault("file_path", str(path))
    meta.setdefault("title", meta.get("title", path.stem))
    entity_name, entity_id = normalize_entity_fields(
        str(meta.get("entity_name") or ""),
        str(meta.get("entity_id") or ""),
    )
    meta["entity_name"] = entity_name
    meta["entity_id"] = entity_id
    meta.setdefault("news_source", "")
    meta.setdefault("url", "")
    meta.setdefault("publish_time", "")
    meta.update(empty_report_metadata())

    return chunks_to_tokens(
        chunks,
        token_type="pdf",
        source="pdf",
        metadata=meta,
        source_key=path.name,
        chunk_metadatas=chunk_metadatas,
    )


def build_news_tokens(
    chunks: list[str],
    news_record: dict[str, Any],
) -> list[Token]:
    """
    从新闻记录及其分块构建 Token 列表。

    news_record 建议包含:
    entity_id, entity_name, date, title, content（可选）, url（可选）
    """
    from src.utils.entity_parser import enrich_record_metadata

    required_fields = ("title",)
    for field in required_fields:
        if field not in news_record:
            raise ValueError(f"新闻记录缺少必要字段: {field}")

    enriched = enrich_record_metadata(news_record)
    entity_name, entity_id = normalize_entity_fields(
        str(enriched.get("entity_name") or enriched.get("entity") or ""),
        str(enriched.get("entity_id") or ""),
    )
    enriched["entity_name"] = entity_name
    enriched["entity_id"] = entity_id

    publish_time = str(
        enriched.get("publish_time") or enriched.get("date") or enriched.get("report_date") or ""
    )
    news_source = str(
        enriched.get("news_source")
        or enriched.get("source")
        or enriched.get("fetch_source")
        or enriched.get("publisher")
        or ""
    )
    meta = {
        "entity_id": str(enriched.get("entity_id") or ""),
        "entity_name": str(enriched.get("entity_name") or ""),
        "date": str(enriched.get("date") or publish_time[:10] if publish_time else ""),
        "title": enriched["title"],
        "url": str(enriched.get("url") or ""),
        "news_source": news_source,
        "publisher": str(enriched.get("publisher") or news_source),
        "publish_time": publish_time,
        "record_id": enriched.get("record_id", enriched["title"]),
        "fetch_source": str(enriched.get("fetch_source") or news_source),
        "report_year": str(enriched.get("report_year") or ""),
        "report_type": str(enriched.get("report_type") or ""),
        "report_date": str(enriched.get("report_date") or ""),
    }

    source_key = str(meta["url"] or meta["record_id"])

    tokens = chunks_to_tokens(
        chunks,
        token_type="news",
        source="news",
        metadata=meta,
        source_key=source_key,
    )
    return tokens


def tokens_to_dicts(tokens: list[Token]) -> list[dict[str, Any]]:
    """将 Token 列表转为可 JSON 序列化的普通字典列表。"""
    return [dict(token) for token in tokens]


def main() -> None:
    """命令行调试入口。"""
    from src.processors.chunker import chunk_text
    from src.processors.text_cleaner import clean_text

    pdf_text = clean_text("贵州茅台2024年第三季度报告。" * 30)
    pdf_chunks = chunk_text(pdf_text)
    pdf_tokens = build_pdf_tokens(
        pdf_chunks,
        "data/raw/pdf/2024Q3_report.pdf",
        metadata={
            "entity_id": "600519.SH",
            "entity_name": "贵州茅台",
            "date": "2024-10-28",
            "title": "2024年第三季度报告",
            "report_year": "2024",
            "report_type": "Q3",
        },
    )

    news_text = clean_text(
        "茅台三季度业绩超预期，机构维持买入评级。" * 20
    )
    news_chunks = chunk_text(news_text)
    news_tokens = build_news_tokens(
        news_chunks,
        {
            "entity_id": "600519.SH",
            "entity_name": "贵州茅台",
            "date": "2024-10-29",
            "title": "茅台三季度业绩超预期",
            "url": "https://example.com/news/1",
        },
    )

    print(f"PDF Token 数量: {len(pdf_tokens)}")
    if pdf_tokens:
        sample = pdf_tokens[0]
        print("PDF 样例:", sample["id"], "|", sample["type"], "|", sample["source"])

    print(f"\n新闻 Token 数量: {len(news_tokens)}")
    if news_tokens:
        sample = news_tokens[0]
        print("新闻样例:", sample["id"], "|", sample["type"], "|", sample["source"])
        print("metadata:", sample["metadata"])


if __name__ == "__main__":
    main()
