"""双公司对比报告卡片。"""

from __future__ import annotations

from typing import Any, Callable


def _disp(fund: dict[str, Any], key: str) -> str:
    item = fund.get(key)
    if isinstance(item, dict) and item.get("display"):
        return str(item["display"])
    return "—"


def _num(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "—"


def _growth(fund: dict[str, Any], key: str) -> str:
    value = fund.get(key)
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def format_company_compare_card(
    *,
    left_name: str,
    right_name: str,
    left_id: str,
    right_id: str,
    fund_a: dict[str, Any],
    fund_b: dict[str, Any],
    snap_a: Any,
    snap_b: Any,
    notes: list[str] | None = None,
) -> str:
    """生成并排财务/估值对比报告。"""
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        f"  公司对比 · {left_name} vs {right_name}",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        "【对比摘要】",
        f"  {left_name}（{left_id or '—'}）  vs  {right_name}（{right_id or '—'}）",
        "",
        "【估值与行情】",
        f"  {'指标':<10} {left_name:<18} {right_name:<18}",
        f"  {'现价':<10} {_num(getattr(snap_a, 'price', None)):<18} {_num(getattr(snap_b, 'price', None)):<18}",
        f"  {'PE':<10} {_num(getattr(snap_a, 'pe_ttm', None)):<18} {_num(getattr(snap_b, 'pe_ttm', None)):<18}",
        f"  {'PB':<10} {_num(getattr(snap_a, 'pb', None)):<18} {_num(getattr(snap_b, 'pb', None)):<18}",
        "",
        "【财务指标】",
        f"  {'指标':<10} {left_name:<18} {right_name:<18}",
        f"  {'净利润':<10} {_disp(fund_a, 'net_profit'):<18} {_disp(fund_b, 'net_profit'):<18}",
        f"  {'营收':<10} {_disp(fund_a, 'revenue'):<18} {_disp(fund_b, 'revenue'):<18}",
        f"  {'EPS':<10} {_disp(fund_a, 'eps'):<18} {_disp(fund_b, 'eps'):<18}",
        f"  {'ROE':<10} {_disp(fund_a, 'roe'):<18} {_disp(fund_b, 'roe'):<18}",
        f"  {'净利同比':<10} {_growth(fund_a, 'profit_growth_pct'):<18} {_growth(fund_b, 'profit_growth_pct'):<18}",
        f"  {'营收同比':<10} {_growth(fund_a, 'revenue_growth_pct'):<18} {_growth(fund_b, 'revenue_growth_pct'):<18}",
    ]

    pe_a = getattr(snap_a, "pe_ttm", None)
    pe_b = getattr(snap_b, "pe_ttm", None)
    verdict = ""
    if pe_a and pe_b:
        if pe_a > pe_b * 1.1:
            verdict = f"{left_name} 的 PE 更高（相对估值更贵）"
        elif pe_a < pe_b * 0.9:
            verdict = f"{left_name} 的 PE 更低（相对估值更便宜）"
        else:
            verdict = "两家 PE 接近"
    lines.extend(["", "【差异结论】"])
    if verdict:
        lines.append(f"  · {verdict}")
    else:
        lines.append("  · PE 数据不足，暂无法给出相对估值结论")

    for note in notes or []:
        if note:
            lines.append(f"  · {note}")

    # 简易溯源：指标是否有页码
    cite_lines: list[str] = []
    for name, fund in ((left_name, fund_a), (right_name, fund_b)):
        for key, label in (("eps", "EPS"), ("roe", "ROE"), ("net_profit", "净利润")):
            item = fund.get(key)
            if isinstance(item, dict) and item.get("source_page"):
                cite_lines.append(
                    f"  · {name}·{label}: P{item.get('source_page')}"
                    + (f" · {item.get('source_section')}" if item.get("source_section") else "")
                )
    lines.extend(["", "【引用溯源】"])
    if cite_lines:
        lines.extend(cite_lines[:8])
    else:
        lines.append("  · 对比主要基于本地已抽取指标与实时行情；缺少页码时请先入库年报 PDF")

    lines.extend(
        [
            "",
            "──────────────────────────────────────────────────────────",
            "  免责声明: PoC 自动对比，不构成投资建议",
            "──────────────────────────────────────────────────────────",
        ]
    )
    return "\n".join(lines)


def ensure_peer_fundamentals(
    *,
    name: str,
    entity_id: str,
    fundamentals_lookup: Callable[[str, str], dict],
    financial_agent: Any | None = None,
    report_year: str | None = None,
) -> tuple[dict[str, Any], str]:
    """
    获取对比方财务指标；若缺失且可自动下载年报，则尝试补齐。
    返回 (fundamentals, note)。
    """
    fund = fundamentals_lookup(name, entity_id) or {}
    has_core = bool(fund.get("eps") or fund.get("net_profit") or fund.get("revenue"))
    if has_core or financial_agent is None or not entity_id:
        return fund, ""

    note = ""
    try:
        from src.collectors.report_downloader import ensure_report_pdf

        downloaded = ensure_report_pdf(name, entity_id, report_year)
        financial_agent.process_pdfs([downloaded.path], build_index=True)
        fund = fundamentals_lookup(name, entity_id) or {}
        tag = "本地缓存" if downloaded.from_cache else "巨潮下载"
        note = f"已为 {name} 自动获取{downloaded.report_year}年报（{tag}）"
    except Exception as exc:
        note = f"{name} 财报可能未入库，指标可能不完整（{exc}）"
    return fund, note
