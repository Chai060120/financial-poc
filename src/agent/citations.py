"""引用溯源：从财务指标、新闻、财报路径收集可核对来源。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Citation:
    kind: str  # metric | news | document | reason
    label: str
    detail: str
    locator: str = ""  # 页码 / URL / 文件名


def collect_citations(result: Any) -> list[Citation]:
    """从 FullAnalysisResult 收集引用，供报告展示与导出。"""
    citations: list[Citation] = []
    seen: set[str] = set()

    def _add(kind: str, label: str, detail: str, locator: str = "") -> None:
        key = f"{kind}|{label}|{locator}|{detail[:40]}"
        if key in seen:
            return
        seen.add(key)
        citations.append(Citation(kind=kind, label=label, detail=detail, locator=locator))

    fund = getattr(result, "fundamentals", None) or {}
    metric_labels = {
        "net_profit": "净利润",
        "attributable_profit": "归母净利润",
        "revenue": "营业收入",
        "eps": "每股收益",
        "bvps": "每股净资产",
        "roe": "ROE",
    }
    for key, label in metric_labels.items():
        item = fund.get(key)
        if not isinstance(item, dict):
            continue
        display = str(item.get("display") or item.get("value") or "").strip()
        if not display:
            continue
        page = item.get("source_page") or 0
        section = str(item.get("source_section") or "").strip()
        locator_parts = []
        if page:
            locator_parts.append(f"P{page}")
        if section:
            locator_parts.append(section)
        locator = " · ".join(locator_parts)
        _add("metric", label, display, locator)

    pdf_source = str(getattr(result, "pdf_source", "") or "").strip()
    if pdf_source:
        _add("document", "财报文件", Path(pdf_source).name, pdf_source)

    comparison = getattr(result, "comparison", None)
    if comparison is not None:
        news_items = list(getattr(comparison, "news", None) or [])[:8]
        for item in news_items:
            title = str(getattr(item, "title", "") or "").strip()
            if not title:
                continue
            url = str(getattr(item, "url", "") or "").strip()
            source = str(getattr(item, "source", "") or "").strip()
            detail = title if len(title) <= 80 else title[:79] + "…"
            locator = url or source
            _add("news", "新闻", detail, locator)

    for reason in list(getattr(result, "synthesis_reasons", None) or [])[:6]:
        text = str(reason).strip()
        if text:
            _add("reason", "研判依据", text, "")

    return citations


def format_citations_section(citations: list[Citation], *, max_items: int = 12) -> str:
    """渲染报告中的引用溯源段落。"""
    if not citations:
        return (
            "【引用溯源】\n"
            "  未收集到可核对来源（可能缺少已入库财报或新闻）。"
        )

    lines = ["【引用溯源】", "  下列条目用于核对结论，非投资建议："]
    for idx, item in enumerate(citations[:max_items], start=1):
        loc = f" ｜ {item.locator}" if item.locator else ""
        if item.kind == "metric":
            lines.append(f"  [{idx}] 指标·{item.label}: {item.detail}{loc}")
        elif item.kind == "news":
            lines.append(f"  [{idx}] 新闻: {item.detail}{loc}")
        elif item.kind == "document":
            lines.append(f"  [{idx}] 文档: {item.detail}{loc}")
        else:
            lines.append(f"  [{idx}] 依据: {item.detail}{loc}")
    return "\n".join(lines)
