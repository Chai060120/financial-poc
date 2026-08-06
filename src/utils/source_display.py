"""检索结果展示辅助：区分财报与新闻 metadata。"""

from __future__ import annotations

from typing import Any


def source_type_label(source: str) -> str:
    """将 source 字段转为可读类型标签。"""
    mapping = {
        "pdf": "财报",
        "news": "新闻",
        "announcement": "公告",
        "wind": "Wind",
        "ths": "同花顺",
        "xueqiu": "雪球",
        "research": "研报",
        "social": "社交媒体",
    }
    return mapping.get(source.strip().lower(), source or "未知")


def format_reference_meta(meta: dict[str, Any]) -> list[str]:
    """
    从 Chroma metadata 提取展示字段。

    财报: 类型、title、entity、report_type、file_name
    新闻: 类型、publisher、publish_time、url、title、entity
    """
    source = str(meta.get("source") or meta.get("type") or "")
    parts: list[str] = [f"类型:{source_type_label(source)}"]

    title = meta.get("title", "")
    entity = meta.get("entity_name", "")
    if entity:
        parts.append(f"entity: {entity}")
    if title:
        parts.append(f"title: {title}")

    if source == "news":
        publisher = meta.get("publisher", "")
        publish_time = meta.get("publish_time") or meta.get("date", "")
        url = meta.get("url", "")
        if publisher:
            parts.append(f"publisher: {publisher}")
        if publish_time:
            parts.append(f"publish_time: {publish_time}")
        if url:
            parts.append(f"url: {url}")
    elif source == "pdf":
        report_type = meta.get("report_type", "")
        report_year = meta.get("report_year", "")
        file_name = meta.get("file_name", "")
        if report_year:
            parts.append(f"report_year: {report_year}")
        if report_type:
            parts.append(f"report_type: {report_type}")
        if file_name:
            parts.append(f"file: {file_name}")
        section = meta.get("section", "")
        table_name = meta.get("table_name", "")
        page = meta.get("page", 0)
        if section:
            parts.append(f"section: {section}")
        if table_name:
            parts.append(f"table: {table_name}")
        if page:
            parts.append(f"page: {page}")
        date = meta.get("date") or meta.get("report_date", "")
        if date:
            parts.append(f"date: {date}")
    elif source == "announcement":
        date = meta.get("date") or meta.get("publish_time", "")
        url = meta.get("url", "")
        if date:
            parts.append(f"date: {date}")
        if url:
            parts.append(f"url: {url}")

    return parts
