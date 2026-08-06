"""
实体解析器：从文件名或文本中识别上市公司、股票代码、年份与报告类型。

优先从 PDF 文件名提取公司简称，并通过 StockRegistry（CSV / AkShare / Tushare）映射股票代码。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import UNKNOWN_ENTITY_ID, UNKNOWN_ENTITY_NAME, setup_logging
from src.utils.stock_registry import get_stock_registry

logger = setup_logging(__name__)

MAX_ENTITY_CANDIDATE_LEN = 24
_HTML_ENTITY_PATTERN = re.compile(r"&(?:nbsp|amp|lt|gt|quot|#\d+);?", re.I)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _strip_html_noise(text: str) -> str:
    """清理 HTML 实体与多余空白，便于实体扫描。"""
    cleaned = _HTML_ENTITY_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()

# 向后兼容：旧代码仍可 import ENTITY_REGISTRY
DEFAULT_STOCK_SEED = {
    "600519.SH": {
        "entity_id": "600519.SH",
        "entity_name": "贵州茅台",
        "aliases": ("贵州茅台", "茅台"),
    },
    "600036.SH": {
        "entity_id": "600036.SH",
        "entity_name": "招商银行",
        "aliases": ("招商银行", "招行"),
    },
    "601318.SH": {
        "entity_id": "601318.SH",
        "entity_name": "中国平安",
        "aliases": ("中国平安", "平安"),
    },
    "600000.SH": {
        "entity_id": "600000.SH",
        "entity_name": "浦发银行",
        "aliases": ("浦发银行", "浦发"),
    },
    "000001.SZ": {
        "entity_id": "000001.SZ",
        "entity_name": "平安银行",
        "aliases": ("平安银行",),
    },
    "000858.SZ": {
        "entity_id": "000858.SZ",
        "entity_name": "五粮液",
        "aliases": ("五粮液",),
    },
}
ENTITY_REGISTRY = DEFAULT_STOCK_SEED

ENTITY_CODE_PATTERN = re.compile(
    r"(?<!\d)(?P<code>\d{6})\.(?P<market>SH|SZ|sh|sz)(?!\d)|"
    r"(?<!\d)(?P<code_bare>\d{6})(?!\d)"
)
YEAR_PATTERN = re.compile(r"(20\d{2})")
ISO_DATE_PATTERN = re.compile(
    r"(20\d{2})[-_/年\.](\d{1,2})[-_/月\.](\d{1,2})"
)

REPORT_TYPE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"半年度?报告?|中期报告?|半年报"), "半年报"),
    (re.compile(r"(?<![一二三四])年度?报告?|(?<![一二三四])年报"), "年报"),
    (re.compile(r"20\d{2}Q1", re.I), "Q1"),
    (re.compile(r"20\d{2}Q2", re.I), "Q2"),
    (re.compile(r"20\d{2}Q3", re.I), "Q3"),
    (re.compile(r"20\d{2}Q4", re.I), "Q4"),
    (re.compile(r"第?[一1]季度报告?|一季报|Q1", re.I), "Q1"),
    (re.compile(r"第?[二2]季度报告?|二季报|Q2", re.I), "Q2"),
    (re.compile(r"第?[三3]季度报告?|三季报|Q3", re.I), "Q3"),
    (re.compile(r"第?[四4]季度报告?|四季报|Q4", re.I), "Q4"),
)

_REPORT_PERIOD_END: dict[str, str] = {
    "Q1": "03-31",
    "Q2": "06-30",
    "半年报": "06-30",
    "Q3": "09-30",
    "Q4": "12-31",
    "年报": "12-31",
}


class EntityParseResult(TypedDict, total=False):
    """实体解析结果。"""

    entity_name: str
    entity_id: str
    report_year: str
    report_type: str
    report_date: str
    title: str


def _normalize_stem(path_or_name: str | Path) -> tuple[str, str]:
    name = Path(path_or_name).name
    stem = Path(name).stem.strip()
    return name, stem


def normalize_entity_fields(
    entity_name: str | None = None,
    entity_id: str | None = None,
) -> tuple[str, str]:
    """
    规范化 entity 字段，补全映射；无法识别时使用 UNKNOWN。

    Returns:
        (entity_name, entity_id)
    """
    registry = get_stock_registry()
    resolved = registry.resolve(
        entity_name=str(entity_name or "").strip(),
        entity_id=str(entity_id or "").strip(),
    )
    name = str(resolved.get("entity_name") or UNKNOWN_ENTITY_NAME).strip() or UNKNOWN_ENTITY_NAME
    eid = str(resolved.get("entity_id") or UNKNOWN_ENTITY_ID).strip() or UNKNOWN_ENTITY_ID
    return name, eid


def extract_entity_name_candidate(text: str) -> str:
    """
    从文件名 stem 中提取公司简称候选。

    示例: 招商银行2025半年报 -> 招商银行
    """
    content = text.strip()
    if not content:
        return ""

    cleaned = YEAR_PATTERN.sub("", content)
    for pattern, _ in REPORT_TYPE_RULES:
        cleaned = pattern.sub("", cleaned)

    cleaned = re.sub(r"[_\-—–\s]+", "", cleaned)
    cleaned = re.sub(r"(报告|财报|业绩|全文|更新|公告|pdf|PDF)+", "", cleaned)
    cleaned = re.sub(r"^[0-9._\-]+|[0-9._\-]+$", "", cleaned)
    return cleaned.strip()


def _match_entity_code(text: str) -> dict[str, str] | None:
    registry = get_stock_registry()
    match = ENTITY_CODE_PATTERN.search(text)
    if not match:
        return None

    code = match.group("code") or match.group("code_bare")
    market = match.group("market")
    if market:
        entity_id = f"{code}.{market.upper()}"
    else:
        entity_id = f"{code}.SH" if code.startswith("6") else f"{code}.SZ"

    found = registry.lookup_by_id(entity_id)
    if found:
        return found
    return {"entity_id": entity_id, "entity_name": ""}


def detect_entity_in_text(text: str) -> dict[str, str] | None:
    """
    在文本中扫描已知股票代码或别名（最长优先）。

    未识别到注册表中的实体时返回 None，避免将新闻标题误判为公司名。
    """
    content = _strip_html_noise(text)
    if not content:
        return None

    by_code = _match_entity_code(content)
    if by_code and by_code.get("entity_name") and by_code["entity_name"] != UNKNOWN_ENTITY_NAME:
        return by_code

    registry = get_stock_registry()
    by_alias = registry.lookup_by_alias(content)
    if by_alias:
        return by_alias

    return None


def detect_query_entity(question: str) -> tuple[str, str]:
    """从检索问题中识别公司过滤条件，返回 (entity_name, entity_id)。"""
    found = detect_entity_in_text(question)
    if not found:
        return "", ""
    return str(found.get("entity_name") or ""), str(found.get("entity_id") or "")


def parse_entity(text: str) -> dict[str, str]:
    """
    从文本中解析 entity_name 与 entity_id。

    优先级:
    1. 文本中的标准代码（600036.SH / 600036）
    2. 注册表别名最长匹配
    3. 文件名简称候选 + 注册表映射（仅短候选，避免新闻标题误判）
    """
    registry = get_stock_registry()
    content = _strip_html_noise(text)

    detected = detect_entity_in_text(content)
    if detected:
        return detected

    candidate = extract_entity_name_candidate(content)
    if candidate and len(candidate) <= MAX_ENTITY_CANDIDATE_LEN:
        by_name = registry.lookup_by_name(candidate)
        if by_name:
            return by_name

    by_code = _match_entity_code(content)
    if by_code:
        name, eid = normalize_entity_fields(
            entity_name=by_code.get("entity_name", ""),
            entity_id=by_code.get("entity_id", ""),
        )
        if name != UNKNOWN_ENTITY_NAME or eid != UNKNOWN_ENTITY_ID:
            return {"entity_name": name, "entity_id": eid}

    return {
        "entity_id": UNKNOWN_ENTITY_ID,
        "entity_name": UNKNOWN_ENTITY_NAME,
    }


def parse_report_year(text: str) -> str:
    """解析报告年份，返回字符串如 '2025'。"""
    match = YEAR_PATTERN.search(text)
    if not match:
        return ""
    return match.group(1)


def parse_report_type(text: str) -> str:
    """解析报告类型。"""
    for pattern, report_type in REPORT_TYPE_RULES:
        if pattern.search(text):
            return report_type
    return ""


def _parse_iso_date(text: str) -> str:
    match = ISO_DATE_PATTERN.search(text)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def build_report_date(report_year: str, report_type: str, *, iso_date: str = "") -> str:
    """根据年份与报告类型构造 report_date。"""
    if iso_date:
        return iso_date
    if not report_year:
        return ""
    period_end = _REPORT_PERIOD_END.get(report_type)
    if period_end:
        return f"{report_year}-{period_end}"
    return report_year


def build_title(entity_name: str, report_year: str, report_type: str, fallback: str) -> str:
    parts: list[str] = []
    if entity_name and entity_name != UNKNOWN_ENTITY_NAME:
        parts.append(entity_name)
    if report_year:
        parts.append(report_year)
    if report_type:
        parts.append(report_type)
    return "".join(parts) if parts else fallback


def parse_filename(path_or_name: str | Path) -> EntityParseResult:
    """
    从 PDF 文件名解析实体与报告信息。

    示例: 招商银行2025半年报.pdf
    -> entity_name=招商银行, entity_id=600036.SH, report_year=2025, report_type=半年报
    """
    file_name, stem = _normalize_stem(path_or_name)
    return parse_from_text(stem, fallback_title=stem, file_name=file_name)


def parse_from_text(
    text: str,
    *,
    fallback_title: str = "",
    file_name: str = "",
) -> EntityParseResult:
    """从任意文本（文件名 stem、新闻标题等）解析实体与报告信息。"""
    content = text.strip()
    entity = parse_entity(content)
    report_year = parse_report_year(content)
    report_type = parse_report_type(content)
    iso_date = _parse_iso_date(content)
    report_date = build_report_date(report_year, report_type, iso_date=iso_date)

    entity_name, entity_id = normalize_entity_fields(
        entity.get("entity_name", ""),
        entity.get("entity_id", ""),
    )
    title = build_title(entity_name, report_year, report_type, fallback_title or content)

    result: EntityParseResult = {
        "entity_name": entity_name,
        "entity_id": entity_id,
        "report_year": report_year,
        "report_type": report_type,
        "report_date": report_date,
        "title": title,
    }

    logger.info(
        "实体解析: input=%r -> entity=%s(%s), year=%s, type=%s",
        (file_name or content)[:80],
        entity_name,
        entity_id,
        report_year or "-",
        report_type or "-",
    )
    return result


def to_token_metadata(
    parsed: EntityParseResult,
    *,
    file_name: str = "",
    file_path: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将解析结果转为 Token metadata 字典。"""
    entity_name, entity_id = normalize_entity_fields(
        parsed.get("entity_name", ""),
        parsed.get("entity_id", ""),
    )

    meta: dict[str, Any] = {
        "entity_name": entity_name,
        "entity_id": entity_id,
        "report_year": parsed.get("report_year", ""),
        "report_type": parsed.get("report_type", ""),
        "report_date": parsed.get("report_date", ""),
        "date": parsed.get("report_date", ""),
        "title": parsed.get("title", ""),
    }

    if file_name:
        meta["file_name"] = file_name
    if file_path:
        meta["file_path"] = file_path

    if parsed.get("report_year"):
        meta["year"] = parsed["report_year"]

    if extra:
        meta.update(extra)

    return meta


def enrich_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """
    补全新闻/公告记录中的实体字段（仅填充缺失项）。

    优先使用 record 已有字段，否则从 title 等文本解析。
    """
    meta = dict(record)
    if not str(meta.get("entity_name") or "").strip() and str(meta.get("entity") or "").strip():
        raw_entity = str(meta["entity"]).strip()
        name, eid = normalize_entity_fields(entity_name=raw_entity, entity_id="")
        if name != UNKNOWN_ENTITY_NAME:
            meta["entity_name"] = name
            if eid != UNKNOWN_ENTITY_ID:
                meta["entity_id"] = eid

    has_entity = bool(str(meta.get("entity_name") or "").strip()) and str(
        meta.get("entity_name")
    ) not in {UNKNOWN_ENTITY_NAME, ""}

    if not has_entity or not str(meta.get("entity_id") or "").strip():
        source_text = " ".join(
            filter(
                None,
                [
                    str(meta.get("title") or ""),
                    str(meta.get("content") or "")[:500],
                ],
            )
        )
        detected = detect_entity_in_text(source_text)
        if detected:
            for key in ("entity_name", "entity_id"):
                if detected.get(key):
                    meta[key] = detected[key]
        parsed = parse_from_text(source_text, fallback_title=str(meta.get("title") or ""))
        for key in ("entity_name", "entity_id", "report_year", "report_type", "report_date"):
            current = str(meta.get(key) or "").strip()
            if (not current or current == UNKNOWN_ENTITY_NAME) and parsed.get(key):
                meta[key] = parsed[key]
        if not str(meta.get("date") or "").strip() and parsed.get("report_date"):
            meta["date"] = parsed["report_date"]

    entity_name, entity_id = normalize_entity_fields(
        meta.get("entity_name"),
        meta.get("entity_id"),
    )
    meta["entity_name"] = entity_name
    meta["entity_id"] = entity_id
    return meta


def main() -> None:
    """命令行调试入口。"""
    registry = get_stock_registry()
    print(f"StockRegistry: {registry.count()} 条")

    samples = [
        "招商银行2025半年报.pdf",
        "贵州茅台2024年第三季度报告.pdf",
        "600519.SH_2024Q3_report.pdf",
        "unknown_document.pdf",
        "茅台三季度业绩超预期",
    ]

    for sample in samples:
        if sample.endswith(".pdf"):
            result = parse_filename(sample)
        else:
            result = parse_from_text(sample)
        print("---")
        print("input:", sample)
        print("result:", result)


if __name__ == "__main__":
    main()
