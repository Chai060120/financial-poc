"""
文档处理流水线：PDF + News → tokens.json。

供 scripts/02_process.py 与 Daily Agent 共用。
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    DEFAULT_ENCODING,
    NEWS_JSON,
    TOKENS_DIR,
    TOKENS_JSON,
    UNIFIED_CSV,
    ensure_dirs,
    setup_logging,
)
from src.collectors.news_collector import load_news_json, repair_news_json
from src.collectors.pdf_collector import collect_pdf_paths
from src.pipelines.index_pipeline import run_index_build
from src.processors.chunker import chunk_text
from src.processors.deduplicator import deduplicate_tokens
from src.processors.pdf_parser import PdfParseError, extract_pages_from_pdf
from src.processors.report_parser import parse_and_chunk_report
from src.processors.text_cleaner import clean_text
from src.processors.tokenizer import Token, build_news_tokens, build_pdf_tokens, tokens_to_dicts
from src.utils.entity_parser import parse_filename, to_token_metadata

logger = setup_logging(__name__)


def _atomic_write_text(path: Path, text: str, *, encoding: str = DEFAULT_ENCODING) -> None:
    """原子写入文本文件，规避 Windows 上短暂锁文件导致的 Errno 22。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    last_error: Exception | None = None
    try:
        tmp.write_text(text, encoding=encoding)
        for attempt in range(8):
            try:
                os.replace(tmp, path)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        # 回退：直接写目标
        for attempt in range(5):
            try:
                path.write_text(text, encoding=encoding)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.08 * (attempt + 1))
        raise OSError(f"写入失败: {path} ({last_error})")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _sanitize_csv_cell(value: Any) -> Any:
    """去掉 NUL 等 Windows 写入时易触发 Errno 22 的字符。"""
    if not isinstance(value, str):
        return value
    return value.replace("\x00", "")


def _write_unified_csv(df: pd.DataFrame, target: Path = UNIFIED_CSV) -> Path:
    """
    原子写入 unified.csv，并在 Windows 文件被短暂占用时重试。

    Desktop 路径下 Defender/索引器常短暂锁文件，直接 open('w') 可能报 Errno 22。
    tokens.json 才是索引主数据源；CSV 写失败时降级为警告，不阻断入库。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_df = df.copy()
    for col in safe_df.select_dtypes(include=["object"]).columns:
        safe_df[col] = safe_df[col].map(_sanitize_csv_cell)

    tmp = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp")
    last_error: Exception | None = None
    try:
        safe_df.to_csv(tmp, index=False, encoding="utf-8-sig")
        for attempt in range(8):
            try:
                os.replace(tmp, target)
                return target
            except OSError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        # replace 持续失败时尝试直接写入目标
        for attempt in range(5):
            try:
                safe_df.to_csv(target, index=False, encoding="utf-8-sig")
                return target
            except OSError as exc:
                last_error = exc
                time.sleep(0.08 * (attempt + 1))
        logger.warning(
            "unified.csv 写入失败（已跳过，不影响 tokens.json）: %s",
            last_error,
        )
        return target
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def tokens_to_rows(tokens: list[Token]) -> list[dict[str, Any]]:
    """将 Token 列表展平为 CSV 行。"""
    rows: list[dict[str, Any]] = []
    for token in tokens:
        meta = token["metadata"]
        rows.append(
            {
                "id": token["id"],
                "type": token["type"],
                "source": token["source"],
                "text": _sanitize_csv_cell(token["text"]),
                "entity_id": meta.get("entity_id", ""),
                "entity_name": meta.get("entity_name", ""),
                "date": meta.get("date", ""),
                "report_year": meta.get("report_year", ""),
                "report_type": meta.get("report_type", ""),
                "title": meta.get("title", ""),
                "file_name": meta.get("file_name", ""),
                "file_path": meta.get("file_path", ""),
                "news_source": meta.get("news_source", ""),
                "url": meta.get("url", ""),
                "publish_time": meta.get("publish_time", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 0),
                "section": meta.get("section", ""),
                "page": meta.get("page", 0),
                "page_start": meta.get("page_start", 0),
                "page_end": meta.get("page_end", 0),
                "table_name": meta.get("table_name", ""),
            }
        )
    return rows


def process_pdf(pdf_path: Path, metadata: dict[str, Any] | None = None) -> list[Token]:
    pages = extract_pages_from_pdf(pdf_path)
    chunks, chunk_metadatas = parse_and_chunk_report(pages)
    if not chunks:
        return []

    parsed = parse_filename(pdf_path)
    pdf_metadata = to_token_metadata(
        parsed,
        file_name=pdf_path.name,
        file_path=str(pdf_path.resolve()),
        extra=metadata,
    )
    return build_pdf_tokens(
        chunks,
        pdf_path,
        metadata=pdf_metadata,
        chunk_metadatas=chunk_metadatas,
    )


def process_all_pdfs() -> tuple[list[Token], dict[str, int]]:
    pdf_paths = collect_pdf_paths()
    all_tokens: list[Token] = []
    failed = 0

    for pdf_path in pdf_paths:
        try:
            all_tokens.extend(process_pdf(pdf_path))
        except (PdfParseError, Exception) as exc:
            logger.error("PDF 处理失败，已跳过: %s | %s", pdf_path.name, exc)
            failed += 1

    return all_tokens, {"pdf_files": len(pdf_paths), "pdf_failed": failed}


def process_news_record(news_record: dict[str, Any]) -> list[Token]:
    raw_content = str(
        news_record.get("content")
        or news_record.get("body")
        or news_record.get("text")
        or ""
    )
    cleaned_text = clean_text(raw_content)
    if not cleaned_text:
        return []

    chunks = chunk_text(cleaned_text)
    if not chunks:
        return []

    return build_news_tokens(chunks, news_record)


def process_news_from_json(news_path: Path | None = None) -> tuple[list[Token], dict[str, int]]:
    path = news_path or NEWS_JSON
    if not path.exists():
        logger.warning("新闻文件不存在: %s", path)
        return [], {"news_records": 0, "news_failed": 0}

    try:
        repaired = repair_news_json(path)
        if repaired:
            logger.info("新闻 entity 修复: %d 条", repaired)
        records = load_news_json(path)
    except Exception as exc:
        logger.error("加载 news.json 失败: %s", exc)
        return [], {"news_records": 0, "news_failed": 0}

    all_tokens: list[Token] = []
    failed = 0
    for record in records:
        try:
            all_tokens.extend(process_news_record(record))
        except Exception as exc:
            logger.error("新闻处理失败: %s | %s", record.get("title"), exc)
            failed += 1

    return all_tokens, {"news_records": len(records), "news_failed": failed}


def load_existing_token_dicts(tokens_path: Path = TOKENS_JSON) -> list[dict[str, Any]]:
    if not tokens_path.exists():
        return []
    with open(tokens_path, encoding=DEFAULT_ENCODING) as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def run_news_append() -> dict[str, Any]:
    """仅处理新闻 Token 并追加到现有 tokens.json。"""
    news_tokens, news_stats = process_news_from_json()
    if not news_tokens:
        return {
            "success": False,
            "message": "未生成任何新闻 Token",
            **news_stats,
        }

    existing = load_existing_token_dicts()
    unique_new, dedup_stats = deduplicate_tokens(news_tokens, existing=existing)
    merged_dicts = existing + tokens_to_dicts(list(unique_new))
    save_tokens_from_dicts(merged_dicts)

    return {
        "success": True,
        "message": f"新闻追加完成，新增 {dedup_stats['output_count']} 条",
        "added": dedup_stats["output_count"],
        "skipped": dedup_stats["removed_count"],
        "existing": len(existing),
        "total": len(merged_dicts),
        **news_stats,
    }


def save_tokens_from_dicts(token_dicts: list[dict[str, Any]]) -> tuple[Path, Path]:
    ensure_dirs()
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        TOKENS_JSON,
        json.dumps(token_dicts, ensure_ascii=False, indent=2),
    )

    as_tokens: list[Token] = [
        {
            "id": str(item["id"]),
            "type": item["type"],
            "source": item["source"],
            "text": item["text"],
            "metadata": dict(item.get("metadata") or {}),
        }
        for item in token_dicts
    ]
    df = pd.DataFrame(tokens_to_rows(as_tokens))
    _write_unified_csv(df, UNIFIED_CSV)
    return TOKENS_JSON, UNIFIED_CSV


def save_tokens(tokens: list[Token]) -> tuple[Path, Path]:
    ensure_dirs()
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)

    token_dicts = tokens_to_dicts(tokens)
    _atomic_write_text(
        TOKENS_JSON,
        json.dumps(token_dicts, ensure_ascii=False, indent=2),
    )

    rows = tokens_to_rows(tokens)
    df = pd.DataFrame(rows)
    _write_unified_csv(df, UNIFIED_CSV)
    return TOKENS_JSON, UNIFIED_CSV


def run_document_processing() -> dict[str, Any]:
    """执行 PDF + News 全量处理并保存 tokens。"""
    logger.info("开始文档处理流水线")

    pdf_tokens, pdf_stats = process_all_pdfs()
    news_tokens, news_stats = process_news_from_json()

    combined = pdf_tokens + news_tokens
    if not combined:
        return {
            "success": False,
            "message": "未生成任何 Token",
            "pdf_tokens": 0,
            "news_tokens": 0,
            "total_tokens": 0,
            **pdf_stats,
            **news_stats,
        }

    unique_tokens, dedup_stats = deduplicate_tokens(combined)
    tokens = list(unique_tokens)
    json_path, csv_path = save_tokens(tokens)

    stats = {
        "success": True,
        "message": f"处理完成，共 {len(tokens)} 个 Token",
        "pdf_tokens": len(pdf_tokens),
        "news_tokens": len(news_tokens),
        "total_tokens": len(tokens),
        "dedup_removed": dedup_stats.get("removed_count", 0),
        "tokens_json": str(json_path),
        "unified_csv": str(csv_path),
        **pdf_stats,
        **news_stats,
    }
    logger.info("文档处理完成: %s", stats["message"])
    return stats


def merge_pdf_tokens(processed_files: list[Path], new_tokens: list[Token]) -> list[dict[str, Any]]:
    """将新 PDF Token 合并进 tokens.json（同名文件覆盖）。"""
    file_names = {path.name for path in processed_files}
    existing = load_existing_token_dicts()
    kept = [
        item
        for item in existing
        if str(item.get("metadata", {}).get("file_name") or "") not in file_names
    ]
    new_dicts = tokens_to_dicts(new_tokens)
    as_tokens: list[Token] = [
        {
            "id": str(item["id"]),
            "type": item["type"],
            "source": item["source"],
            "text": item["text"],
            "metadata": dict(item.get("metadata") or {}),
        }
        for item in kept + new_dicts
    ]
    unique, stats = deduplicate_tokens(as_tokens)
    logger.info(
        "合并 PDF Token: 移除旧 %d 条, 新增 %d 条, 合计 %d 条",
        len(existing) - len(kept),
        len(new_dicts),
        stats["output_count"],
    )
    return tokens_to_dicts(list(unique))


def run_incremental_pdfs(
    pdf_paths: list[Path] | None = None,
    *,
    build_index: bool = True,
) -> dict[str, Any]:
    """增量处理指定 PDF（或 raw/pdf 下全部），并可选更新索引。"""
    from src.collectors.pdf_collector import collect_pdf_paths
    from config import RAW_PDF_DIR

    paths = pdf_paths if pdf_paths is not None else collect_pdf_paths(RAW_PDF_DIR)
    if not paths:
        return {"success": False, "message": f"未找到 PDF，请放入 {RAW_PDF_DIR}"}

    all_tokens: list[Token] = []
    succeeded: list[Path] = []
    failed: list[dict[str, str]] = []
    entities: dict[str, str] = {}

    for pdf_path in paths:
        try:
            tokens = process_pdf(pdf_path)
            if not tokens:
                failed.append({"file": pdf_path.name, "reason": "未生成 Token"})
                continue
            all_tokens.extend(tokens)
            succeeded.append(pdf_path)
            for token in tokens:
                meta = token.get("metadata") or {}
                entity_id = str(meta.get("entity_id") or "").strip()
                entity_name = str(meta.get("entity_name") or "").strip()
                if entity_id and entity_name and entity_id != "UNKNOWN":
                    entities[entity_id] = entity_name
        except Exception as exc:
            logger.exception("PDF 处理失败: %s", pdf_path.name)
            failed.append({"file": pdf_path.name, "reason": str(exc)})

    if not all_tokens:
        return {
            "success": False,
            "message": "所有 PDF 均未成功",
            "failed": failed,
        }

    merged = merge_pdf_tokens(succeeded, all_tokens)
    json_path, csv_path = save_tokens_from_dicts(merged)

    result: dict[str, Any] = {
        "success": True,
        "message": f"PDF 处理完成: {len(succeeded)} 个文件, {len(all_tokens)} Token",
        "files_ok": len(succeeded),
        "new_tokens": len(all_tokens),
        "tokens_json": str(json_path),
        "unified_csv": str(csv_path),
        "failed": failed,
        "entities": [
            {"entity_id": entity_id, "entity_name": entity_name}
            for entity_id, entity_name in entities.items()
        ],
    }

    if build_index:
        index_stats = run_index_build(rebuild=False)
        result["index"] = index_stats
        result["message"] += f", 索引 {index_stats.get('after_count', 0)} 条"

    return result
