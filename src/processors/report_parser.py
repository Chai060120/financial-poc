"""
财报结构化解析：识别主要章节与报表类型，生成 section / page / table_name 元数据。

支持章节：
- 利润表
- 资产负债表
- 现金流量表
- 管理层讨论
- 风险提示

输出 ReportSegment，供分块与 Token metadata 写入，便于后续按章节检索。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import CHUNK_OVERLAP, CHUNK_SIZE, setup_logging
from src.processors.chunker import chunk_text
from src.processors.pdf_parser import PageContent
from src.processors.text_cleaner import clean_text

from src.financial.metric_dictionary import TABLE_CONTEXT_PATTERNS

logger = setup_logging(__name__)

# 标准章节名称（用于 metadata.section）
SECTION_INCOME_STATEMENT = "利润表"
SECTION_BALANCE_SHEET = "资产负债表"
SECTION_CASH_FLOW = "现金流量表"
SECTION_MANAGEMENT_DISCUSSION = "管理层讨论"
SECTION_RISK = "风险提示"
SECTION_QUARTERLY = "分季度财务"
SECTION_ACCOUNTING_SUMMARY = "主要会计数据"
SECTION_OTHER = "其他"

KNOWN_SECTIONS: tuple[str, ...] = (
    SECTION_INCOME_STATEMENT,
    SECTION_BALANCE_SHEET,
    SECTION_CASH_FLOW,
    SECTION_ACCOUNTING_SUMMARY,
    SECTION_QUARTERLY,
    SECTION_MANAGEMENT_DISCUSSION,
    SECTION_RISK,
)

FINANCIAL_STATEMENT_SECTIONS: frozenset[str] = frozenset(
    {SECTION_INCOME_STATEMENT, SECTION_BALANCE_SHEET, SECTION_CASH_FLOW}
)

# 章节识别规则（按优先级排序，先匹配更具体的）
SECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        SECTION_QUARTERLY,
        re.compile(r"分季度主要财务数据|分季度.*?主要财务|季度主要财务数据"),
    ),
    (
        SECTION_ACCOUNTING_SUMMARY,
        re.compile(
            r"主要会计数据和财务指标|近三年主要会计数据|"
            r"主要会计数据|主要财务指标摘要"
        ),
    ),
    (
        SECTION_RISK,
        re.compile(
            r"可能面对的风险|重大风险(?:因素)?提示|风险因素|"
            r"(?:公司|本集团)?(?:面临|面对)的(?:主要)?风险"
        ),
    ),
    (
        SECTION_MANAGEMENT_DISCUSSION,
        re.compile(
            r"管理层讨论与分析|经营情况讨论与分析|"
            r"董事会报告(?:及有关情况)?|管理层讨论"
        ),
    ),
    (
        SECTION_CASH_FLOW,
        re.compile(r"(?:合并|母公司|本公司)?现金流量表"),
    ),
    (
        SECTION_BALANCE_SHEET,
        re.compile(r"(?:合并|母公司|本公司)?资产负债表"),
    ),
    (
        SECTION_INCOME_STATEMENT,
        re.compile(r"(?:合并|母公司|本公司)?(?:利润表|损益表)"),
    ),
)

TABLE_NAME_PATTERN = re.compile(
    r"(?:合并|母公司|本公司)?(?:利润表|损益表|资产负债表|现金流量表)"
)


class ReportSegment(TypedDict):
    """财报结构化片段。"""

    section: str
    page: int
    page_start: int
    page_end: int
    table_name: str
    table_context: str
    text: str


class ChunkWithMetadata(TypedDict):
    """分块文本及其章节 metadata。"""

    text: str
    metadata: dict[str, Any]


def empty_report_metadata() -> dict[str, Any]:
    """返回空的章节 metadata 默认值（兼容 news / 未识别 PDF）。"""
    return {
        "section": "",
        "page": 0,
        "page_start": 0,
        "page_end": 0,
        "table_name": "",
        "table_context": "",
    }


def segment_to_metadata(segment: ReportSegment) -> dict[str, Any]:
    """将 ReportSegment 转为 Token metadata 字段。"""
    return {
        "section": segment["section"],
        "page": segment["page"],
        "page_start": segment["page_start"],
        "page_end": segment["page_end"],
        "table_name": segment["table_name"],
        "table_context": segment.get("table_context") or "",
    }


def detect_table_context(line: str) -> str:
    """识别表格上下文标题（如主要会计数据和财务指标）。"""
    content = line.strip()
    if not content or len(content) > 80:
        return ""
    for pattern, label in TABLE_CONTEXT_PATTERNS:
        if pattern in content:
            return label
    return ""


def detect_section(line: str) -> str:
    """从单行文本识别章节类型，未识别返回空字符串。"""
    content = line.strip()
    if not content or len(content) > 120:
        return ""

    for section_name, pattern in SECTION_RULES:
        if pattern.search(content):
            return section_name
    return ""


def detect_table_name(line: str) -> str:
    """从单行文本识别报表名称（如 合并利润表）。"""
    content = line.strip()
    if not content or len(content) > 80:
        return ""

    match = TABLE_NAME_PATTERN.search(content)
    if match is None:
        return ""

    table_name = match.group(0).strip()
    if table_name.endswith("项目") or table_name.endswith("科目"):
        return ""
    return table_name


def _normalize_table_name(table_name: str, section: str) -> str:
    """补全或规范化 table_name。"""
    name = table_name.strip()
    if name:
        return name

    if section in FINANCIAL_STATEMENT_SECTIONS:
        return section
    return ""


def parse_report(pages: list[PageContent]) -> list[ReportSegment]:
    """
    解析财报页面，按章节切分为结构化片段。

    Args:
        pages: 带页码的 PDF 页面列表（page_number 从 1 开始）。

    Returns:
        ReportSegment 列表；若无有效文本则返回空列表。
    """
    if not pages:
        return []

    segments: list[ReportSegment] = []
    current_section = ""
    current_table = ""
    current_table_context = ""
    current_lines: list[str] = []
    current_page_start = 0
    current_page_end = 0

    def flush() -> None:
        nonlocal current_lines, current_section, current_table, current_table_context
        nonlocal current_page_start, current_page_end

        text = clean_text("\n".join(current_lines))
        if not text:
            current_lines = []
            return

        section = current_section or SECTION_OTHER
        table_name = _normalize_table_name(current_table, section)
        page = current_page_start or current_page_end or 1

        segments.append(
            {
                "section": section,
                "page": page,
                "page_start": current_page_start or page,
                "page_end": current_page_end or page,
                "table_name": table_name,
                "table_context": current_table_context,
                "text": text,
            }
        )
        current_lines = []

    for page in pages:
        page_number = int(page["page_number"])
        raw_text = str(page.get("text") or "")
        if not raw_text.strip():
            continue

        for line in raw_text.split("\n"):
            stripped = line.strip()
            detected_section = detect_section(stripped)
            detected_table = detect_table_name(stripped)
            detected_context = detect_table_context(stripped)

            if detected_context:
                current_table_context = detected_context

            if detected_section and detected_section != current_section:
                flush()
                current_section = detected_section
                current_table = detected_table or _normalize_table_name("", detected_section)
                current_page_start = page_number
                current_page_end = page_number
            elif (
                detected_table
                and detected_table != current_table
                and current_section in FINANCIAL_STATEMENT_SECTIONS
            ):
                flush()
                current_table = detected_table
                current_page_start = page_number
                current_page_end = page_number
            elif detected_table and not current_table:
                current_table = detected_table

            current_lines.append(line)
            current_page_end = page_number
            if not current_page_start:
                current_page_start = page_number

    flush()

    if not segments:
        merged_text = clean_text(
            "\n\n".join(str(page.get("text") or "") for page in pages if page.get("text"))
        )
        if merged_text:
            first_page = int(pages[0]["page_number"])
            last_page = int(pages[-1]["page_number"])
            segments.append(
                {
                    "section": SECTION_OTHER,
                    "page": first_page,
                    "page_start": first_page,
                    "page_end": last_page,
                    "table_name": "",
                    "table_context": "",
                    "text": merged_text,
                }
            )

    logger.info(
        "财报结构化解析完成: %d 页 -> %d 个片段 | 章节分布=%s",
        len(pages),
        len(segments),
        _summarize_sections(segments),
    )
    return segments


def chunk_report_segments(
    segments: list[ReportSegment],
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[ChunkWithMetadata]:
    """
    将结构化片段分块，并为每块附加 section / page / table_name metadata。
    """
    results: list[ChunkWithMetadata] = []

    for segment in segments:
        text = segment["text"]
        if not text:
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        base_meta = segment_to_metadata(segment)

        for chunk in chunks:
            results.append({"text": chunk, "metadata": dict(base_meta)})

    logger.info(
        "财报分块完成: %d 个片段 -> %d 个 chunk",
        len(segments),
        len(results),
    )
    return results


def parse_and_chunk_report(
    pages: list[PageContent],
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    一站式：解析财报并分块。

    Returns:
        (chunks, chunk_metadatas) 等长列表，供 build_pdf_tokens 使用。
    """
    segments = parse_report(pages)
    if not segments:
        return [], []

    chunked = chunk_report_segments(segments, chunk_size=chunk_size, overlap=overlap)
    if not chunked:
        return [], []

    return (
        [item["text"] for item in chunked],
        [item["metadata"] for item in chunked],
    )


def _summarize_sections(segments: list[ReportSegment]) -> dict[str, int]:
    counter: dict[str, int] = {}
    for segment in segments:
        section = segment["section"] or SECTION_OTHER
        counter[section] = counter.get(section, 0) + 1
    return counter


def main() -> None:
    """命令行调试入口。"""
    from src.collectors.pdf_collector import collect_pdf_paths
    from src.processors.pdf_parser import extract_pages_from_pdf

    pdf_paths = collect_pdf_paths()
    if not pdf_paths:
        print("未发现 PDF 文件。")
        return

    path = pdf_paths[0]
    pages = extract_pages_from_pdf(path)
    segments = parse_report(pages)

    print(f"文件: {path.name}")
    print(f"页数: {len(pages)} | 片段: {len(segments)}")
    print(f"章节分布: {_summarize_sections(segments)}\n")

    for index, segment in enumerate(segments[:12], start=1):
        preview = segment["text"][:80].replace("\n", " ")
        print(
            f"[{index}] section={segment['section']} | page={segment['page']} | "
            f"table={segment['table_name'] or '-'} | {preview}..."
        )


if __name__ == "__main__":
    main()
