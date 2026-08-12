"""
规则化意图识别：PDF / 公司名 / 自然语言问题 → 分析动作。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from config import RAW_PDF_DIR
from src.utils.entity_parser import detect_entity_in_text
from src.utils.stock_registry import get_stock_registry

_PDF_PATTERN = re.compile(r"\.pdf\b", re.I)
_YEAR_PATTERN = re.compile(r"(20\d{2})")
_COMPARE_WITH = re.compile(
    r"(?:和|跟|与|同)(?P<target>[\u4e00-\u9fffA-Za-z0-9\.]{2,20})(?:比|对比|比较)"
)
_VS_COMPARE = re.compile(
    r"(?P<left>[\u4e00-\u9fffA-Za-z0-9\.]{2,20})"
    r"\s*(?:vs\.?|VS\.?|对比|比)\s*"
    r"(?P<right>[\u4e00-\u9fffA-Za-z0-9\.]{2,20})"
)
_COMPARE_PAIR = re.compile(
    r"(?:对比|比较)\s*(?P<left>[\u4e00-\u9fffA-Za-z0-9\.]{2,20})"
    r"\s*(?:和|与|跟)\s*(?P<right>[\u4e00-\u9fffA-Za-z0-9\.]{2,20})"
)
_EXPLAIN_MARKERS = ("为什么", "为何", "依据", "原因", "怎么判断", "凭什么", "溯源", "引用来源")
_EXPORT_MARKERS = ("导出报告", "导出", "下载报告", "export")
_FULL_ANALYZE_MARKERS = (
    "分析",
    "评估",
    "研判",
    "诊断",
    "高估",
    "低估",
    "合理",
    "值不值得",
    "能不能买",
)
_VALUATE_MARKERS = ("估值", "pe", "pb", "市盈率", "市净率")
_COMPARE_MARKERS = ("对比", "同业", "比较", "横向", "排名")
_QUERY_METRICS = (
    "净利润",
    "营收",
    "营业收入",
    "每股收益",
    "roe",
    "净资产收益率",
    "不良贷款",
    "现金流",
    "分红",
    "多少",
    "是多少",
)


class AgentIntent(str, Enum):
    FULL_ANALYZE = "full_analyze"
    VALUATE = "valuate"
    COMPARE = "compare"
    COMPARE_WITH = "compare_with"
    QUERY = "query"
    INGEST_PDF = "ingest_pdf"
    EXPLAIN = "explain"
    EXPORT = "export"
    HELP = "help"
    RESET = "reset"
    EXIT = "exit"
    UNKNOWN = "unknown"


@dataclass
class RoutedIntent:
    intent: AgentIntent
    raw_input: str
    entity_name: str = ""
    entity_id: str = ""
    pdf_path: Path | None = None
    compare_target: str = ""
    question: str = ""
    report_year: str = ""
    report_type: str = "年报"
    notes: list[str] = field(default_factory=list)


def _extract_report_meta(text: str) -> tuple[str, str]:
    """从用户输入抽取报告年份与类型，默认年报。"""
    report_type = "年报"
    if re.search(r"半年|中期", text):
        report_type = "半年报"
    elif re.search(r"一季|Q1", text, re.I):
        report_type = "一季报"
    elif re.search(r"三季|Q3", text, re.I):
        report_type = "三季报"
    year = ""
    match = _YEAR_PATTERN.search(text)
    if match:
        year = match.group(1)
    return year, report_type


def _with_report_meta(route: RoutedIntent, text: str) -> RoutedIntent:
    year, report_type = _extract_report_meta(text)
    if year:
        route.report_year = year
    route.report_type = report_type
    return route


def _resolve_entity(text: str) -> tuple[str, str]:
    registry = get_stock_registry()
    found = detect_entity_in_text(text) or registry.lookup_by_name(text)
    if not found:
        return "", ""
    return str(found.get("entity_name") or ""), str(found.get("entity_id") or "")


def _extract_pdf_path(text: str) -> Path | None:
    candidate = text.strip().strip('"').strip("'")
    path = Path(candidate)
    if path.suffix.lower() == ".pdf" and path.is_file():
        return path
    under_raw = RAW_PDF_DIR / path.name
    if under_raw.is_file():
        return under_raw
    if _PDF_PATTERN.search(candidate):
        name = Path(candidate).name
        if (RAW_PDF_DIR / name).is_file():
            return RAW_PDF_DIR / name
    return None


def route_intent(
    user_input: str,
    *,
    last_entity_name: str = "",
    last_entity_id: str = "",
    has_last_analysis: bool = False,
) -> RoutedIntent:
    """将用户输入路由到 Agent 动作。"""
    text = user_input.strip()
    if not text:
        return RoutedIntent(AgentIntent.UNKNOWN, text)

    lowered = text.lower()
    if lowered in {"exit", "quit", "q", "退出", "再见"}:
        return RoutedIntent(AgentIntent.EXIT, text)
    if lowered in {"help", "帮助", "?"}:
        return RoutedIntent(AgentIntent.HELP, text)
    if lowered in {"reset", "clear", "清空", "重新开始"}:
        return RoutedIntent(AgentIntent.RESET, text)

    pdf_path = _extract_pdf_path(text)
    if pdf_path is not None:
        return _with_report_meta(
            RoutedIntent(
                AgentIntent.FULL_ANALYZE,
                text,
                pdf_path=pdf_path,
                notes=["检测到 PDF 路径，将导入并全量分析"],
            ),
            text,
        )

    if has_last_analysis and any(marker in text for marker in _EXPLAIN_MARKERS):
        return RoutedIntent(
            AgentIntent.EXPLAIN,
            text,
            entity_name=last_entity_name,
            entity_id=last_entity_id,
        )

    if any(marker in lowered for marker in _EXPORT_MARKERS) or text.strip() in {
        "导出报告",
        "下载报告",
    }:
        return RoutedIntent(
            AgentIntent.EXPORT,
            text,
            entity_name=last_entity_name,
            entity_id=last_entity_id,
        )

    vs_match = _VS_COMPARE.search(text) or _COMPARE_PAIR.search(text)
    if vs_match:
        left_raw = vs_match.group("left").strip()
        right_raw = vs_match.group("right").strip()
        left_name, left_id = _resolve_entity(left_raw)
        right_name, right_id = _resolve_entity(right_raw)
        return _with_report_meta(
            RoutedIntent(
                AgentIntent.COMPARE_WITH,
                text,
                entity_name=left_name or last_entity_name or left_raw,
                entity_id=left_id or last_entity_id,
                compare_target=right_name or right_raw,
                notes=["双公司对比"],
            ),
            text,
        )

    match = _COMPARE_WITH.search(text)
    if match:
        compare_name = match.group("target").strip()
        name, eid = _resolve_entity(compare_name)
        return _with_report_meta(
            RoutedIntent(
                AgentIntent.COMPARE_WITH,
                text,
                entity_name=last_entity_name or name,
                entity_id=last_entity_id or eid,
                compare_target=name or compare_name,
            ),
            text,
        )

    name, eid = _resolve_entity(text)

    if any(marker in text for marker in _COMPARE_MARKERS):
        if name:
            return RoutedIntent(
                AgentIntent.COMPARE,
                text,
                entity_name=name,
                entity_id=eid,
            )
        if last_entity_name:
            return RoutedIntent(
                AgentIntent.COMPARE,
                text,
                entity_name=last_entity_name,
                entity_id=last_entity_id,
                notes=["未识别新公司，沿用上一家公司"],
            )

    if any(marker in lowered for marker in _VALUATE_MARKERS) or any(
        marker in text for marker in _FULL_ANALYZE_MARKERS
    ):
        if name:
            intent = (
                AgentIntent.VALUATE
                if any(m in lowered for m in _VALUATE_MARKERS)
                and not any(m in text for m in ("分析", "评估", "研判"))
                else AgentIntent.FULL_ANALYZE
            )
            return _with_report_meta(
                RoutedIntent(
                    intent,
                    text,
                    entity_name=name,
                    entity_id=eid,
                ),
                text,
            )
        if last_entity_name and any(m in text for m in ("高估", "低估", "估值", "分析")):
            return _with_report_meta(
                RoutedIntent(
                    AgentIntent.FULL_ANALYZE,
                    text,
                    entity_name=last_entity_name,
                    entity_id=last_entity_id,
                    notes=["沿用上一家公司"],
                ),
                text,
            )

    if name and any(metric in text for metric in _QUERY_METRICS):
        return RoutedIntent(
            AgentIntent.QUERY,
            text,
            entity_name=name,
            entity_id=eid,
            question=text,
        )

    if name:
        return _with_report_meta(
            RoutedIntent(
                AgentIntent.FULL_ANALYZE,
                text,
                entity_name=name,
                entity_id=eid,
                notes=["识别到公司，将自动检索财报并分析"],
            ),
            text,
        )

    if last_entity_name and (
        any(metric in text for metric in _QUERY_METRICS) or len(text) < 30
    ):
        return RoutedIntent(
            AgentIntent.QUERY,
            text,
            entity_name=last_entity_name,
            entity_id=last_entity_id,
            question=text,
            notes=["追问模式，沿用上一家公司"],
        )

    if text in {"分析", "开始分析", "全部分析", "analyze"}:
        return RoutedIntent(AgentIntent.FULL_ANALYZE, text)

    return RoutedIntent(AgentIntent.UNKNOWN, text, question=text)
