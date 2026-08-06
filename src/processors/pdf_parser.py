"""
PDF 解析器：使用 PyMuPDF 提取 PDF 全文文本。

仅负责文本提取，不负责文件发现或后续清洗。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import fitz

from config import setup_logging

logger = setup_logging(__name__)

PAGE_SEPARATOR = "\n\n"


class PageContent(TypedDict):
    """单页 PDF 文本。"""

    page_number: int
    text: str


class PdfParseError(Exception):
    """PDF 解析失败时抛出。"""


def _normalize_pdf_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path).expanduser().resolve()

    if not path.exists():
        raise PdfParseError(f"PDF 文件不存在: {path}")

    if not path.is_file():
        raise PdfParseError(f"路径不是文件: {path}")

    if path.suffix.lower() != ".pdf":
        raise PdfParseError(f"文件后缀不是 .pdf: {path}")

    return path


def extract_pages_from_pdf(pdf_path: str | Path) -> list[PageContent]:
    """
    按页提取 PDF 文本，保留页码信息供财报结构化解析使用。

    Returns:
        PageContent 列表，page_number 从 1 开始；空页仍保留条目但 text 为空。
    """
    path = _normalize_pdf_path(pdf_path)
    pages: list[PageContent] = []

    try:
        document = fitz.open(path)
    except fitz.FileDataError as exc:
        raise PdfParseError(f"PDF 文件损坏或格式无效: {path}") from exc
    except Exception as exc:
        raise PdfParseError(f"无法打开 PDF: {path}") from exc

    total_pages = document.page_count
    try:
        for page_index in range(total_pages):
            page_number = page_index + 1
            try:
                page = document.load_page(page_index)
                text = page.get_text("text").strip()
            except Exception as exc:
                logger.warning(
                    "第 %d 页提取失败: %s | %s",
                    page_number,
                    path.name,
                    exc,
                )
                text = ""

            pages.append({"page_number": page_number, "text": text})
    finally:
        document.close()

    non_empty = sum(1 for item in pages if item["text"])
    logger.info(
        "PDF 按页解析完成: %s | 有效页数 %d / 总页数 %d",
        path.name,
        non_empty,
        total_pages,
    )
    return pages


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    读取 PDF 全部页面文本并合并为字符串。

    空页（无文本或仅空白）会被跳过；单页提取失败时记录警告并继续。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        非空页面文本，以双换行符连接。

    Raises:
        PdfParseError: 文件无效或 PDF 无法打开。
    """
    path = _normalize_pdf_path(pdf_path)
    pages = extract_pages_from_pdf(path)
    page_texts = [item["text"] for item in pages if item["text"]]

    if not page_texts:
        logger.warning("PDF 未提取到任何文本: %s", path)

    return PAGE_SEPARATOR.join(page_texts)


def main() -> None:
    """命令行调试入口。"""
    from src.collectors.pdf_collector import collect_pdf_paths

    pdf_paths = collect_pdf_paths()
    if not pdf_paths:
        print("未发现 PDF 文件。")
        return

    for path in pdf_paths:
        try:
            text = extract_text_from_pdf(path)
            preview = text[:200].replace("\n", " ")
            print(f"\n[{path.name}] 提取字符数: {len(text)}")
            print(f"预览: {preview}...")
        except PdfParseError as exc:
            print(f"\n[{path.name}] 解析失败: {exc}")


if __name__ == "__main__":
    main()
