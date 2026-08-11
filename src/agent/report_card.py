"""
五段式分析报告：一次输出全部结论，无需多轮提问。
"""

from __future__ import annotations

from dataclasses import dataclass

from config import AGENT_WATCHLIST
from src.analysis.full_report import FullAnalysisResult
from src.analysis.market_data import fetch_market_snapshot_enriched


@dataclass
class PeerRow:
    entity_name: str
    price: str
    pe: str
    pb: str
    note: str = ""


def _metric_display(fundamentals: dict, key: str) -> str:
    item = fundamentals.get(key)
    if isinstance(item, dict) and item.get("display"):
        conf = item.get("confidence")
        text = str(item["display"])
        if conf is not None and float(conf) >= 0.8:
            return text
        if conf is not None and float(conf) > 0:
            return f"{text} (置信{float(conf):.0%})"
        return text
    return "—"


def _valuation_verdict_line(result: FullAnalysisResult) -> str:
    verdict = result.final_verdict
    if not result.valuation_reliable:
        return "无法可靠判断估值"
    method = "基于PE/PB估值"
    if verdict in {"高估", "合理", "低估"}:
        return f"{verdict}（{method}）"
    return verdict


def _format_val(value: float | None, source: str = "") -> str:
    if value is None:
        return "—"
    text = f"{value:.2f}"
    if source.startswith("computed"):
        text += "(推算)"
    return text


def build_peer_rows(
    primary: FullAnalysisResult,
    fundamentals_lookup,
) -> list[PeerRow]:
    """与监控列表其他公司自动对比（无需用户提问）。"""
    rows: list[PeerRow] = []
    base_id = primary.entity_id
    fund_a = primary.fundamentals
    market = (primary.valuation.market if primary.valuation else {}) or {}
    snap_a = fetch_market_snapshot_enriched(base_id, primary.entity_name, fund_a)
    pe_val = market.get("pe_ttm") or snap_a.pe_ttm
    pb_val = market.get("pb") or snap_a.pb
    pe_src = str(market.get("pe_source") or snap_a.pe_source or "")
    pb_src = str(market.get("pb_source") or snap_a.pb_source or "")
    price_val = market.get("price") or snap_a.price
    rows.append(
        PeerRow(
            entity_name=f"{primary.entity_name} ← 分析对象",
            price=str(price_val or "—"),
            pe=_format_val(pe_val, pe_src),
            pb=_format_val(pb_val, pb_src),
            note="",
        )
    )

    for other_id in AGENT_WATCHLIST:
        if other_id == base_id:
            continue
        from src.utils.stock_registry import get_stock_registry

        registry = get_stock_registry()
        found = registry.lookup_by_id(other_id) or {}
        other_name = str(found.get("entity_name") or other_id)
        fund_b = fundamentals_lookup(other_name, other_id)
        snap_b = fetch_market_snapshot_enriched(other_id, other_name, fund_b)
        note = ""
        if snap_a.pe_ttm and snap_b.pe_ttm:
            if snap_a.pe_ttm > snap_b.pe_ttm * 1.1:
                note = "对象 PE 更高"
            elif snap_a.pe_ttm < snap_b.pe_ttm * 0.9:
                note = "对象 PE 更低"
            else:
                note = "PE 接近"
        rows.append(
            PeerRow(
                entity_name=other_name,
                price=str(snap_b.price or "—"),
                pe=_format_val(snap_b.pe_ttm, snap_b.pe_source),
                pb=_format_val(snap_b.pb, snap_b.pb_source),
                note=note,
            )
        )
    return rows


def format_report_card(
    result: FullAnalysisResult,
    peer_rows: list[PeerRow],
) -> str:
    """生成五段式纯结果报告（无日志、无技术细节）。"""
    v = result.valuation
    c = result.comparison
    fund = result.fundamentals

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        f"  {result.entity_name}（{result.entity_id}）· {result.period_label}",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        "【1】估值结论",
        f"  估值结论: {_valuation_verdict_line(result)}",
        f"  估值可信度: {result.final_confidence}    综合评分: {result.final_score:+.1f}",
    ]
    if result.data_warnings:
        lines.append("  数据质量提示:")
        for warn in result.data_warnings[:4]:
            lines.append(f"    · {warn}")
    elif not result.valuation_reliable:
        lines.append("  数据质量提示: 财务指标置信度不足，估值仅供参考")

    lines.extend(
        [
            "",
            "【2】财报核心指标",
            f"  净利润:     {_metric_display(fund, 'net_profit')}",
            f"  营业收入:   {_metric_display(fund, 'revenue')}",
            f"  每股收益:   {_metric_display(fund, 'eps')}",
            f"  每股净资产: {_metric_display(fund, 'bvps')}",
            f"  ROE:        {_metric_display(fund, 'roe')}",
        ]
    )

    if fund.get("profit_growth_pct") is not None:
        lines.append(f"  净利润同比: {fund['profit_growth_pct']}%")
    if fund.get("revenue_growth_pct") is not None:
        lines.append(f"  营收同比:   {fund['revenue_growth_pct']}%")

    lines.extend(
        [
            "",
            "【3】关键词",
            f"  {', '.join(result.keywords) if result.keywords else '—'}",
            "",
            "【4】网络行情与横向对比",
        ]
    )

    primary_row = peer_rows[0] if peer_rows else None
    if primary_row:
        lines.append(
            f"  现价: {primary_row.price}  PE: {primary_row.pe}  PB: {primary_row.pb}"
        )
    elif c and c.target.price is not None:
        pe = _format_val(c.target.pe_ttm, "")
        pb = _format_val(c.target.pb, "")
        lines.append(f"  现价: {c.target.price}  PE: {pe}  PB: {pb}")
    if c:
        lines.append(f"  相对同业: {c.relative_verdict}")
        if c.industry_stats.avg_pe is not None:
            lines.append(
                f"  行业平均 PE: {c.industry_stats.avg_pe}（排名 {c.industry_stats.target_pe_rank}）"
            )
        lines.append(f"  新闻情绪: {c.news_sentiment}")

    if peer_rows:
        lines.append("")
        lines.append("  与监控列表对比:")
        lines.append(f"  {'公司':<16} {'现价':<10} {'PE':<12} {'PB':<12} 备注")
        for row in peer_rows:
            lines.append(
                f"  {row.entity_name:<16} {row.price:<10} {row.pe:<12} {row.pb:<12} {row.note}"
            )
        missing_peers = [
            row.entity_name.replace(" ← 分析对象", "")
            for row in peer_rows
            if row.pe == "—" and row.pb == "—" and row.price != "—"
        ]
        if missing_peers:
            names = "、".join(missing_peers[:3])
            lines.append(
                f"  说明: {names} 暂无 PE/PB（网络接口未返回且本地无财报可推算）"
            )

    lines.extend(["", "【5】近期资讯与研判依据"])

    if c and c.news:
        lines.append("  近期资讯:")
        for item in c.news[:5]:
            title = item.title[:55] + ("…" if len(item.title) > 55 else "")
            lines.append(f"    · {title}")
    else:
        lines.append("  近期资讯: —")

    lines.append("")
    lines.append("  研判依据:")
    for reason in result.synthesis_reasons[:8]:
        lines.append(f"    · {reason}")

    lines.extend(
        [
            "",
            "──────────────────────────────────────────────────────────",
            "  免责声明: PoC 自动分析，不构成投资建议",
        ]
    )
    if result.report_path:
        lines.append(f"  完整报告: {result.report_path}")
    lines.append("──────────────────────────────────────────────────────────")

    return "\n".join(lines)
