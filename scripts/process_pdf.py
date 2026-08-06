"""
通用 PDF 处理：任意 PDF → Token → 增量索引。

    python scripts/process_pdf.py
    python scripts/process_pdf.py "data/raw/pdf/某某年报.pdf"
    python scripts/process_pdf.py --inspect
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    DEFAULT_ENCODING,
    RAW_PDF_DIR,
    TOKENS_JSON,
    add_project_root_to_path,
    ensure_dirs,
    setup_logging,
)
from src.collectors.pdf_collector import collect_pdf_paths
from src.pipelines.document_pipeline import process_pdf, save_tokens_from_dicts
from src.pipelines.index_pipeline import run_index_build
from src.processors.deduplicator import deduplicate_tokens
from src.processors.pdf_parser import extract_pages_from_pdf
from src.processors.report_parser import parse_and_chunk_report
from src.processors.tokenizer import Token, tokens_to_dicts
from src.utils.entity_parser import parse_filename
from src.vectorstore.unified_retrieval import create_retrieval_engine

add_project_root_to_path()
logger = setup_logging(__name__)


def resolve_pdf_paths(
    paths: list[str],
    *,
    directory: str | None = None,
    recursive: bool = False,
) -> list[Path]:
    resolved: list[Path] = []
    if directory:
        dir_path = Path(directory).resolve()
        if not dir_path.is_dir():
            raise FileNotFoundError(f"目录不存在: {dir_path}")
        resolved.extend(collect_pdf_paths(dir_path, recursive=recursive))

    for raw in paths:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        if path.is_dir():
            resolved.extend(collect_pdf_paths(path, recursive=recursive))
        elif path.suffix.lower() == ".pdf":
            resolved.append(path)
        else:
            raise ValueError(f"不是 PDF 文件: {path}")

    if not resolved and not paths and not directory:
        resolved = collect_pdf_paths(RAW_PDF_DIR, recursive=recursive)

    seen: set[str] = set()
    unique: list[Path] = []
    for path in sorted(resolved, key=lambda p: str(p).lower()):
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path.resolve())
    return unique


def inspect_pdf(pdf_path: Path) -> dict:
    parsed = parse_filename(pdf_path)
    pages = extract_pages_from_pdf(pdf_path)
    chunks, chunk_metadatas = parse_and_chunk_report(pages)
    sections = Counter(str(m.get("section") or "其他") for m in chunk_metadatas)
    return {
        "file": pdf_path.name,
        "path": str(pdf_path),
        "pages": len(pages),
        "chunks": len(chunks),
        "entity_name": parsed.get("entity_name"),
        "entity_id": parsed.get("entity_id"),
        "report_year": parsed.get("report_year"),
        "report_type": parsed.get("report_type"),
        "sections": dict(sections),
        "sample_chunk": (chunks[0][:200] if chunks else ""),
    }


def load_existing_tokens() -> list[dict]:
    if not TOKENS_JSON.exists():
        return []
    with open(TOKENS_JSON, encoding=DEFAULT_ENCODING) as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def merge_pdf_tokens(processed_files: list[Path], new_tokens: list[Token]) -> list[dict]:
    file_names = {path.name for path in processed_files}
    existing = load_existing_tokens()
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
        "合并 Token: 移除旧文件 %d 条, 新增 %d 条, 合计 %d 条",
        len(existing) - len(kept),
        len(new_dicts),
        stats["output_count"],
    )
    return tokens_to_dicts(list(unique))


def run_query(question: str) -> None:
    engine = create_retrieval_engine()
    results = engine.retrieve(question, top_k=3)
    print(f"\n检索: {question}")
    print(f"命中 {len(results)} 条:\n")
    for index, item in enumerate(results, start=1):
        meta = item["metadata"]
        print(
            f"[{index}] score={item['score']:.4f} | {meta.get('entity_name')} | "
            f"{meta.get('section')} | p.{meta.get('page')} | {meta.get('file_name')}"
        )
        print(f"    {item['text'][:150]}...\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="通用 PDF 处理")
    parser.add_argument("pdfs", nargs="*", help="PDF 路径；省略则处理 data/raw/pdf/ 全部")
    parser.add_argument("--dir", default="", help="指定 PDF 目录")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument("--inspect", action="store_true", help="仅预览，不写库")
    parser.add_argument("--no-index", action="store_true", help="只更新 tokens，不写 Chroma")
    parser.add_argument("--query", default="", help="处理完成后检索验证")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        pdf_paths = resolve_pdf_paths(
            args.pdfs, directory=args.dir or None, recursive=args.recursive
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}")
        sys.exit(1)

    if not pdf_paths:
        print(f"未找到 PDF。请将文件放入: {RAW_PDF_DIR}")
        sys.exit(1)

    if args.inspect:
        for pdf_path in pdf_paths:
            info = inspect_pdf(pdf_path)
            print("\n=== PDF 预览 ===")
            for key, value in info.items():
                if key == "sample_chunk":
                    print(f"{key}:\n  {value}...")
                else:
                    print(f"{key}: {value}")
        return

    all_tokens: list[Token] = []
    succeeded: list[Path] = []
    failed: list[tuple[str, str]] = []

    for pdf_path in pdf_paths:
        logger.info("开始处理: %s", pdf_path.name)
        try:
            tokens = process_pdf(pdf_path)
            if not tokens:
                failed.append((pdf_path.name, "未生成 Token"))
                continue
            all_tokens.extend(tokens)
            succeeded.append(pdf_path)
        except Exception as exc:
            logger.exception("处理失败: %s", pdf_path.name)
            failed.append((pdf_path.name, str(exc)))

    if not all_tokens:
        print("所有 PDF 均未成功生成 Token。")
        for name, reason in failed:
            print(f"  - {name}: {reason}")
        sys.exit(1)

    merged = merge_pdf_tokens(succeeded, all_tokens)
    json_path, csv_path = save_tokens_from_dicts(merged)
    logger.info("已写入 %s, %s", json_path, csv_path)

    if not args.no_index:
        payload = run_index_build(rebuild=False)
        print(
            f"\n索引更新: +{payload.get('indexed', 0)} 条, "
            f"合计 {payload.get('after_count', 0)} 条"
        )

    print("\n=== 处理完成 ===")
    print(f"成功: {len(succeeded)} 个文件, {len(all_tokens)} 个新 Token")
    for path in succeeded:
        parsed = parse_filename(path)
        print(
            f"  - {path.name} | {parsed.get('entity_name')} | "
            f"{parsed.get('report_year')} {parsed.get('report_type')}"
        )
    if failed:
        print(f"失败: {len(failed)} 个")
        for name, reason in failed:
            print(f"  - {name}: {reason}")

    if args.query:
        run_query(args.query)


if __name__ == "__main__":
    main()
