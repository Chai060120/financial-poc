"""
新闻采集器：RSS 抓取、去重、时间过滤、增量写入 news.json。

支持 Google News RSS、东方财富 RSS（可用时）、新浪财经 RSS 及自定义 RSS。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import feedparser

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    DEFAULT_ENCODING,
    NEWS_DEFAULT_DAYS,
    NEWS_FETCH_LIMIT,
    NEWS_JSON,
    NEWS_RSS_FEEDS,
    RAW_NEWS_DIR,
    ensure_dirs,
    setup_logging,
)

logger = setup_logging(__name__)

JSON_SUFFIX = ".json"
CONTENT_FIELDS = ("content", "body", "text", "summary")
TIME_FIELDS = ("publish_time", "published", "pubDate", "date")
OUTPUT_FIELDS = ("title", "content", "url", "publish_time", "source", "entity")

# RSS 源注册表（source_id -> feed_url）
RSS_SOURCE_REGISTRY: dict[str, str] = {
    "google_news": (
        "https://news.google.com/rss/search?"
        "q=%E8%B4%A2%E7%BB%8F&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
    ),
    "sina_finance": "https://rss.sina.com.cn/finance/stock.xml",
    "sina_finance_news": "https://rss.sina.com.cn/finance/finance.xml",
    "eastmoney": "https://rss.eastmoney.com/roll_news.xml",
}

DEFAULT_RSS_SOURCES: tuple[str, ...] = (
    "google_news",
    "sina_finance",
    "sina_finance_news",
    "eastmoney",
)


class NewsCollectorError(Exception):
    """新闻采集失败。"""


@dataclass(frozen=True)
class CollectStats:
    """采集统计信息。"""

    fetched: int
    after_time_filter: int
    existing: int
    added: int
    skipped_duplicate: int
    total: int
    failed_sources: tuple[str, ...]


def is_news_json_file(path: Path) -> bool:
    """判断路径是否为 JSON 新闻文件。"""
    return path.is_file() and path.suffix.lower() == JSON_SUFFIX


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_publish_datetime(value: Any) -> datetime | None:
    """解析 publish_time 为 datetime。"""
    if value is None:
        return None

    if isinstance(value, (list, tuple)) and len(value) >= 6:
        try:
            return datetime(*[int(value[i]) for i in range(6)])
        except (TypeError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return parsedate_to_datetime(text).replace(tzinfo=None)
    except (TypeError, ValueError, IndexError):
        pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None


def format_publish_time(value: Any) -> str:
    """将时间规范化为 YYYY-MM-DD HH:MM:SS 字符串。"""
    dt = parse_publish_datetime(value)
    if dt is not None:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    text = str(value or "").strip()
    return text[:19] if len(text) >= 19 else text


def _dedup_key(record: dict[str, Any]) -> str:
    url = str(record.get("url") or "").strip()
    if url:
        return hashlib.md5(url.encode("utf-8")).hexdigest()
    title = str(record.get("title") or "").strip()
    publish_time = str(record.get("publish_time") or "").strip()
    raw = f"{title}|{publish_time}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def deduplicate_news(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """
    新闻去重，保留先出现的记录。

    Returns:
        (去重后列表, 跳过重复条数)
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    skipped = 0

    for record in records:
        key = _dedup_key(record)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        unique.append(record)

    logger.info("去重完成: 输入 %d 条, 保留 %d 条, 跳过 %d 条", len(records), len(unique), skipped)
    return unique, skipped


def filter_news_by_time(
    records: list[dict[str, Any]],
    *,
    since: datetime | None = None,
    days: int | None = None,
) -> list[dict[str, Any]]:
    """按发布时间过滤新闻。"""
    cutoff = since
    if days is not None and days > 0:
        cutoff = datetime.now() - timedelta(days=days)

    if cutoff is None:
        return records

    filtered: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        publish_time = record.get("publish_time")
        dt = parse_publish_datetime(publish_time)
        if dt is None:
            filtered.append(record)
            continue
        if dt >= cutoff:
            filtered.append(record)
        else:
            skipped += 1

    logger.info(
        "时间过滤: cutoff=%s, 保留 %d 条, 过滤 %d 条",
        cutoff.strftime("%Y-%m-%d %H:%M:%S"),
        len(filtered),
        skipped,
    )
    return filtered


def infer_entity(title: str, content: str) -> str:
    """从标题与正文推断 entity（公司名），仅返回注册表中的已知实体。"""
    from src.utils.entity_parser import detect_entity_in_text

    text = f"{title} {content[:500]}".strip()
    if not text:
        return ""
    found = detect_entity_in_text(text)
    if not found:
        return ""
    return str(found.get("entity_name") or "")


def normalize_news_record(
    raw: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any] | None:
    """
    规范化为统一输出 schema。

    字段: title, content, url, publish_time, source, entity
    """
    title = str(raw.get("title") or "").strip()
    if not title:
        return None

    content = ""
    for field in CONTENT_FIELDS:
        value = raw.get(field)
        if value and str(value).strip():
            content = _strip_html(str(value))
            break
    if not content:
        content = _strip_html(str(raw.get("summary") or raw.get("description") or title))

    url = str(raw.get("url") or raw.get("link") or "").strip()
    publish_time = format_publish_time(
        raw.get("publish_time") or raw.get("published") or raw.get("date")
    )
    entity = str(raw.get("entity") or raw.get("entity_name") or "").strip()
    if entity:
        from src.utils.entity_parser import normalize_entity_fields
        from config import UNKNOWN_ENTITY_NAME

        validated_name, _ = normalize_entity_fields(entity_name=entity, entity_id="")
        entity = validated_name if validated_name != UNKNOWN_ENTITY_NAME else ""
    if not entity:
        entity = infer_entity(title, content)

    return {
        "title": title,
        "content": content,
        "url": url,
        "publish_time": publish_time,
        "source": source,
        "entity": entity,
    }


def _parse_rss_entry(entry: Any, source: str) -> dict[str, Any] | None:
    published = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    publish_raw = published
    if published is None:
        publish_raw = getattr(entry, "published", "") or getattr(entry, "updated", "")

    summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
    content_value = summary
    if hasattr(entry, "content") and entry.content:
        try:
            content_value = entry.content[0].get("value", summary)
        except (AttributeError, IndexError, KeyError, TypeError):
            content_value = summary

    raw = {
        "title": getattr(entry, "title", ""),
        "content": content_value,
        "url": getattr(entry, "link", ""),
        "publish_time": publish_raw,
    }
    return normalize_news_record(raw, source=source)


def fetch_rss_source(
    source_id: str,
    feed_url: str,
    *,
    limit: int = NEWS_FETCH_LIMIT,
) -> list[dict[str, Any]]:
    """从单个 RSS 源抓取新闻。"""
    logger.info("RSS 抓取开始: source=%s, url=%s", source_id, feed_url)
    records: list[dict[str, Any]] = []

    try:
        parsed = feedparser.parse(feed_url)
    except Exception as exc:
        logger.error("RSS 解析异常: source=%s | %s", source_id, exc)
        raise NewsCollectorError(f"RSS 解析失败: {source_id}") from exc

    if getattr(parsed, "bozo", False):
        bozo_exc = getattr(parsed, "bozo_exception", None)
        logger.warning("RSS 源可能不可用: source=%s | %s", source_id, bozo_exc)

    entries = list(getattr(parsed, "entries", []))[:limit]
    if not entries:
        logger.warning("RSS 源无条目: source=%s", source_id)
        return records

    for entry in entries:
        try:
            record = _parse_rss_entry(entry, source=source_id)
        except Exception as exc:
            logger.warning("RSS 条目解析失败，已跳过: source=%s | %s", source_id, exc)
            continue
        if record:
            records.append(record)

    logger.info("RSS 抓取完成: source=%s, 条数=%d", source_id, len(records))
    return records


def fetch_rss_sources(
    source_ids: list[str] | None = None,
    *,
    limit: int = NEWS_FETCH_LIMIT,
    extra_feeds: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    批量抓取 RSS 源。

    Returns:
        (新闻列表, 失败源 ID 列表)
    """
    selected = source_ids or list(DEFAULT_RSS_SOURCES)
    all_records: list[dict[str, Any]] = []
    failed_sources: list[str] = []

    for source_id in selected:
        feed_url = RSS_SOURCE_REGISTRY.get(source_id, "")
        if not feed_url:
            logger.warning("未知 RSS 源，已跳过: %s", source_id)
            failed_sources.append(source_id)
            continue
        try:
            batch = fetch_rss_source(source_id, feed_url, limit=limit)
            all_records.extend(batch)
            if not batch and source_id == "eastmoney":
                logger.warning("东方财富 RSS 当前不可用或无数据，已跳过")
        except NewsCollectorError as exc:
            logger.error("RSS 源抓取失败: source=%s | %s", source_id, exc)
            failed_sources.append(source_id)
        except Exception as exc:
            logger.exception("RSS 源抓取异常: source=%s | %s", source_id, exc)
            failed_sources.append(source_id)

    if extra_feeds:
        for index, feed_url in enumerate(extra_feeds, start=1):
            source_id = f"rss_{index}"
            try:
                batch = fetch_rss_source(source_id, feed_url, limit=limit)
                all_records.extend(batch)
            except Exception as exc:
                logger.error("自定义 RSS 抓取失败: %s | %s", feed_url, exc)
                failed_sources.append(source_id)

    return all_records, failed_sources


def sort_news_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 publish_time 降序排序；无时间条目排在末尾。"""

    def sort_key(record: dict[str, Any]) -> tuple[int, float]:
        dt = parse_publish_datetime(record.get("publish_time"))
        if dt is None:
            return (1, 0.0)
        return (0, -dt.timestamp())

    return sorted(records, key=sort_key)


def load_news_json(path: Path | None = None) -> list[dict[str, Any]]:
    """加载 news.json；文件不存在时返回空列表。"""
    file_path = path or NEWS_JSON
    if not file_path.exists():
        logger.info("新闻文件不存在，将创建新文件: %s", file_path)
        return []

    try:
        with open(file_path, encoding=DEFAULT_ENCODING) as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise NewsCollectorError(f"新闻 JSON 格式错误: {file_path}") from exc

    if not isinstance(payload, list):
        raise NewsCollectorError(f"新闻 JSON 应为数组: {file_path}")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            logger.warning("跳过非对象条目: %s [%d]", file_path.name, index)
            continue
        source = str(item.get("source") or "unknown")
        normalized = normalize_news_record(item, source=source)
        if normalized:
            records.append(normalized)

    logger.info("已加载现有新闻 %d 条: %s", len(records), file_path)
    return records


def repair_news_json(path: Path | None = None) -> int:
    """
    重新规范化 news.json 中的 entity 字段并写回磁盘。

    Returns:
        修复的条目数量。
    """
    file_path = path or NEWS_JSON
    if not file_path.exists():
        return 0

    try:
        with open(file_path, encoding=DEFAULT_ENCODING) as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise NewsCollectorError(f"新闻 JSON 格式错误: {file_path}") from exc

    if not isinstance(payload, list):
        raise NewsCollectorError(f"新闻 JSON 应为数组: {file_path}")

    repaired: list[dict[str, Any]] = []
    fixed = 0
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "unknown")
        normalized = normalize_news_record(item, source=source)
        if not normalized:
            continue
        old_entity = str(item.get("entity") or item.get("entity_name") or "").strip()
        new_entity = str(normalized.get("entity") or "").strip()
        if old_entity != new_entity:
            fixed += 1
        repaired.append(normalized)

    if fixed:
        save_news_json(repaired, file_path)
        logger.info("已修复 news.json 中 %d 条 entity 字段: %s", fixed, file_path)

    return fixed


def save_news_json(
    records: list[dict[str, Any]],
    path: Path | None = None,
) -> Path:
    """保存新闻到 news.json（仅输出标准字段）。"""
    file_path = path or NEWS_JSON
    file_path.parent.mkdir(parents=True, exist_ok=True)

    output = [{field: record.get(field, "") for field in OUTPUT_FIELDS} for record in records]
    with open(file_path, "w", encoding=DEFAULT_ENCODING) as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    logger.info("新闻已保存: %s (%d 条)", file_path, len(output))
    return file_path


def merge_incremental(
    existing: list[dict[str, Any]],
    fetched: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """
    增量合并：existing + fetched，去重后排序。

    Returns:
        (合并结果, 新增条数, 跳过重复条数)
    """
    existing_count = len(existing)
    merged, skipped = deduplicate_news(existing + fetched)
    added = len(merged) - existing_count
    merged = sort_news_records(merged)
    logger.info(
        "增量合并: 原有 %d 条, 抓取 %d 条, 新增 %d 条, 重复跳过 %d 条, 合计 %d 条",
        existing_count,
        len(fetched),
        max(added, 0),
        skipped,
        len(merged),
    )
    return merged, max(added, 0), skipped


def collect_and_update_news(
    *,
    sources: list[str] | None = None,
    limit: int = NEWS_FETCH_LIMIT,
    days: int | None = NEWS_DEFAULT_DAYS,
    since: datetime | None = None,
    incremental: bool = True,
    output_path: Path | None = None,
    include_extra_rss: bool = True,
) -> tuple[list[dict[str, Any]], CollectStats]:
    """
    抓取 RSS 新闻并增量更新 news.json。

    Args:
        sources: RSS 源 ID 列表，默认全部内置源。
        limit: 每个源最多抓取条数。
        days: 仅保留最近 N 天（与 since 二选一，since 优先）。
        since: 起始时间过滤。
        incremental: True 则与已有 news.json 合并；False 则全量覆盖。
        output_path: 输出路径，默认 NEWS_JSON。
        include_extra_rss: 是否包含 config.NEWS_RSS_FEEDS 自定义源。
    """
    ensure_dirs()
    target = output_path or NEWS_JSON
    extra = list(NEWS_RSS_FEEDS) if include_extra_rss else None

    fetched, failed_sources = fetch_rss_sources(
        sources,
        limit=limit,
        extra_feeds=extra,
    )
    fetched_count = len(fetched)

    filtered = filter_news_by_time(fetched, since=since, days=days if since is None else None)
    filtered, _ = deduplicate_news(filtered)

    if incremental and target.exists():
        existing = load_news_json(target)
    else:
        existing = []

    merged, added, skipped_duplicate = merge_incremental(existing, filtered)
    save_news_json(merged, target)

    stats = CollectStats(
        fetched=fetched_count,
        after_time_filter=len(filtered),
        existing=len(existing),
        added=added,
        skipped_duplicate=skipped_duplicate,
        total=len(merged),
        failed_sources=tuple(failed_sources),
    )
    return merged, stats


# ---------------------------------------------------------------------------
# 兼容旧接口：扫描目录加载 JSON
# ---------------------------------------------------------------------------

def collect_news_json_paths(
    directory: Path | None = None,
    *,
    recursive: bool = False,
) -> list[Path]:
    """扫描目录下的 JSON 新闻文件。"""
    scan_dir = directory or RAW_NEWS_DIR

    if not scan_dir.exists():
        logger.warning("新闻目录不存在: %s", scan_dir)
        return []

    if not scan_dir.is_dir():
        logger.error("新闻路径不是目录: %s", scan_dir)
        return []

    candidates = scan_dir.rglob("*") if recursive else scan_dir.iterdir()
    paths = sorted(
        (path.resolve() for path in candidates if is_news_json_file(path)),
        key=lambda item: str(item).lower(),
    )

    logger.info("在 %s 发现 %d 个新闻 JSON 文件", scan_dir, len(paths))
    return paths


def _normalize_record(raw: dict[str, Any], source_file: Path) -> dict[str, Any]:
    """兼容旧 pipeline：加载 JSON 时规范化记录。"""
    source = str(raw.get("source") or "unknown")
    normalized = normalize_news_record(raw, source=source)
    if normalized is None:
        raise ValueError(f"新闻记录缺少 title: {source_file.name}")

    record = dict(normalized)
    record["entity_name"] = record.get("entity", "")
    record["entity_id"] = str(raw.get("entity_id") or "")
    record["date"] = record.get("publish_time", "")[:10]
    record["publisher"] = str(raw.get("publisher") or record["source"])
    record["record_id"] = _dedup_key(record)
    record["source_file"] = source_file.name
    return record


def _parse_news_payload(payload: Any, source_file: Path) -> list[dict[str, Any]]:
    """解析 JSON 载荷为新闻记录列表。"""
    if isinstance(payload, dict):
        items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"不支持的新闻 JSON 结构: {source_file}")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            logger.warning("跳过非对象新闻条目: %s [%d]", source_file.name, index)
            continue
        try:
            records.append(_normalize_record(item, source_file))
        except ValueError as exc:
            logger.warning("跳过无效新闻条目: %s | %s", source_file.name, exc)

    return records


def load_news_from_file(path: Path | str) -> list[dict[str, Any]]:
    """从单个 JSON 文件加载新闻记录。"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"新闻文件不存在: {file_path}")

    with open(file_path, encoding=DEFAULT_ENCODING) as file:
        payload = json.load(file)

    records = _parse_news_payload(payload, file_path)
    logger.info("从 %s 加载 %d 条新闻", file_path.name, len(records))
    return records


def collect_news_records(
    directory: Path | None = None,
    *,
    recursive: bool = False,
) -> list[dict[str, Any]]:
    """扫描并加载全部新闻记录（兼容 news_pipeline）。"""
    paths = collect_news_json_paths(directory, recursive=recursive)
    if not paths:
        return []

    all_records: list[dict[str, Any]] = []
    failed_files: list[str] = []

    for path in paths:
        try:
            records = load_news_from_file(path)
            all_records.extend(records)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.error("加载新闻文件失败，已跳过: %s | %s", path.name, exc)
            failed_files.append(path.name)

    logger.info(
        "新闻加载完成: 文件 %d 个, 记录 %d 条, 失败 %d 个",
        len(paths),
        len(all_records),
        len(failed_files),
    )
    return all_records


def main() -> None:
    """命令行调试入口。"""
    ensure_dirs()
    records, stats = collect_and_update_news()
    print(f"采集完成: total={stats.total}, added={stats.added}, fetched={stats.fetched}")
    for record in records[:5]:
        print(f"  - [{record.get('source')}] {record.get('title')}")


if __name__ == "__main__":
    main()
