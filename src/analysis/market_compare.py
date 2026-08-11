"""
实时对比分析：爬取网络行情/新闻，与同行业及监控列表横向比较。

不依赖 LLM；输出 Markdown/JSON 报告。
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import AGENT_WATCHLIST, DEFAULT_ENCODING, DOCS_DIR, setup_logging
from src.analysis.market_data import fetch_market_snapshot_enriched
from src.analysis.valuation import ValuationResult, analyze_valuation
from src.collectors.market_collector import (
    NewsItem,
    PeerQuote,
    fetch_industry_peers,
    fetch_peer_quotes,
    fetch_stock_news,
    fetch_watchlist_quotes,
    snapshot_to_peer,
)
from src.utils.entity_parser import detect_entity_in_text
from src.utils.stock_registry import get_stock_registry
from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine

logger = setup_logging(__name__)

COMPARISON_REPORT_DIR = DOCS_DIR / "comparison"

_POSITIVE_NEWS = ("增长", "超预期", "上调", "买入", "盈利", "创新高", "回购", "分红")
_NEGATIVE_NEWS = ("下滑", "亏损", "下调", "减持", "处罚", "诉讼", "违规", "暴跌", "警示")


@dataclass
class IndustryStats:
    peer_count: int = 0
    avg_pe: float | None = None
    avg_pb: float | None = None
    avg_change_pct: float | None = None
    target_pe_rank: str = ""
    target_pb_rank: str = ""


@dataclass
class ComparisonResult:
    entity_name: str
    entity_id: str
    industry: str
    target: PeerQuote
    peers: list[PeerQuote] = field(default_factory=list)
    watchlist: list[PeerQuote] = field(default_factory=list)
    industry_stats: IndustryStats = field(default_factory=IndustryStats)
    news: list[NewsItem] = field(default_factory=list)
    news_sentiment: str = "中性"
    relative_verdict: str = ""
    summary: str = ""
    valuation: ValuationResult | None = None
    report_path: Path | None = None


def _resolve_entity(query: str) -> tuple[str, str]:
    registry = get_stock_registry()
    found = detect_entity_in_text(query) or registry.lookup_by_name(query)
    if not found:
        raise ValueError(f"无法识别公司: {query}")
    return str(found.get("entity_name") or ""), str(found.get("entity_id") or "")


def _rank_label(value: float | None, values: list[float], *, lower_is_cheaper: bool) -> str:
    clean = [v for v in values if v is not None and v > 0]
    if value is None or value <= 0 or not clean:
        return "—"
    sorted_vals = sorted(clean, reverse=not lower_is_cheaper)
    try:
        rank = sorted_vals.index(value) + 1
    except ValueError:
        return "—"
    return f"{rank}/{len(clean)}"


def _analyze_news_sentiment(news: list[NewsItem]) -> str:
    if not news:
        return "无近期新闻"
    pos = neg = 0
    for item in news:
        title = item.title
        if any(k in title for k in _POSITIVE_NEWS):
            pos += 1
        if any(k in title for k in _NEGATIVE_NEWS):
            neg += 1
    if pos > neg:
        return "偏正面"
    if neg > pos:
        return "偏负面"
    return "中性"


def _build_industry_stats(target: PeerQuote, peers: list[PeerQuote]) -> IndustryStats:
    stats = IndustryStats(peer_count=len(peers))
    pe_vals = [p.pe_ttm for p in peers if p.pe_ttm and p.pe_ttm > 0]
    pb_vals = [p.pb for p in peers if p.pb and p.pb > 0]
    chg_vals = [p.change_pct for p in peers if p.change_pct is not None]

    if pe_vals:
        stats.avg_pe = round(mean(pe_vals), 2)
        stats.target_pe_rank = _rank_label(target.pe_ttm, pe_vals, lower_is_cheaper=True)
    if pb_vals:
        stats.avg_pb = round(mean(pb_vals), 2)
        stats.target_pb_rank = _rank_label(target.pb, pb_vals, lower_is_cheaper=True)
    if chg_vals:
        stats.avg_change_pct = round(mean(chg_vals), 2)
    return stats


def _relative_verdict(
    target: PeerQuote,
    stats: IndustryStats,
    sentiment: str,
) -> str:
    """基于同业 PE/PB 排名 + 新闻情绪的相对结论。"""
    score = 0.0
    if target.pe_ttm and stats.avg_pe:
        if target.pe_ttm < stats.avg_pe * 0.85:
            score -= 1.0
        elif target.pe_ttm > stats.avg_pe * 1.15:
            score += 1.0
    if target.pb and stats.avg_pb:
        if target.pb < stats.avg_pb * 0.85:
            score -= 0.5
        elif target.pb > stats.avg_pb * 1.15:
            score += 0.5
    if sentiment == "偏正面":
        score -= 0.5
    elif sentiment == "偏负面":
        score += 0.5

    if score >= 1.0:
        return "相对同业偏贵"
    if score <= -1.0:
        return "相对同业偏便宜"
    return "相对同业合理"


def _build_summary(
    result: ComparisonResult,
) -> str:
    t = result.target
    s = result.industry_stats
    parts = [f"{result.entity_name}（{result.entity_id}）"]
    if t.pe_ttm is not None and s.avg_pe is not None:
        parts.append(f"PE {t.pe_ttm:.1f} vs 行业均值 {s.avg_pe:.1f}（排名 {s.target_pe_rank}）")
    if t.pb is not None and s.avg_pb is not None:
        parts.append(f"PB {t.pb:.2f} vs 行业均值 {s.avg_pb:.2f}（排名 {s.target_pb_rank}）")
    parts.append(f"新闻情绪 {result.news_sentiment}")
    parts.append(f"相对判断 {result.relative_verdict}")
    if result.valuation:
        parts.append(f"财报估值 {result.valuation.verdict}")
    return "；".join(parts)


def _render_report(result: ComparisonResult) -> str:
    lines = [
        f"# 实时对比分析 · {result.entity_name} ({result.entity_id})",
        "",
        f"- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 行业: {result.industry or '—'}",
        f"- **相对同业: {result.relative_verdict}**",
        "",
        "## 一句话摘要",
        "",
        result.summary,
        "",
        "## 目标公司（实时）",
        "",
    ]
    t = result.target
    for label, val in (
        ("现价", t.price),
        ("PE", t.pe_ttm),
        ("PB", t.pb),
        ("涨跌幅%", t.change_pct),
        ("总市值", t.market_cap),
    ):
        if val is not None:
            lines.append(f"- {label}: {val}")

    if result.industry_stats.peer_count:
        s = result.industry_stats
        lines.extend(
            [
                "",
                "## 行业对比",
                "",
                f"- 样本数: {s.peer_count} 只同业",
                f"- 行业平均 PE: {s.avg_pe or '—'}（目标排名 {s.target_pe_rank}）",
                f"- 行业平均 PB: {s.avg_pb or '—'}（目标排名 {s.target_pb_rank}）",
                f"- 行业平均涨跌幅: {s.avg_change_pct or '—'}%",
                "",
                "| 公司 | 代码 | PE | PB | 涨跌幅% |",
                "|------|------|-----|-----|---------|",
            ]
        )
        all_rows = [result.target, *result.peers]
        for row in all_rows[:12]:
            pe = f"{row.pe_ttm:.1f}" if row.pe_ttm else "—"
            pb = f"{row.pb:.2f}" if row.pb else "—"
            chg = f"{row.change_pct:.2f}" if row.change_pct is not None else "—"
            mark = "**" if row.entity_id == result.entity_id else ""
            lines.append(
                f"| {mark}{row.entity_name}{mark} | {row.entity_id} | {pe} | {pb} | {chg} |"
            )

    if result.watchlist:
        lines.extend(
            [
                "",
                "## 监控列表对比",
                "",
                "| 公司 | PE | PB | 涨跌幅% |",
                "|------|-----|-----|---------|",
            ]
        )
        for row in result.watchlist:
            pe = f"{row.pe_ttm:.1f}" if row.pe_ttm else "—"
            pb = f"{row.pb:.2f}" if row.pb else "—"
            chg = f"{row.change_pct:.2f}" if row.change_pct is not None else "—"
            lines.append(f"| {row.entity_name} | {pe} | {pb} | {chg} |")

    if result.news:
        lines.extend(["", "## 近期网络资讯", ""])
        for item in result.news:
            time_text = f" ({item.publish_time})" if item.publish_time else ""
            lines.append(f"- [{item.source}] {item.title}{time_text}")

    if result.valuation:
        v = result.valuation
        lines.extend(
            [
                "",
                "## 财报估值（规则引擎）",
                "",
                f"- 报告期: {v.fundamentals.get('period_label') or v.report_year}",
                f"- 结论: **{v.verdict}**（评分 {v.score:+.1f}，置信度 {v.confidence}）",
            ]
        )
        for reason in v.reasons[:6]:
            lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "> 免责声明: 数据来自公开网络接口，仅供 PoC 演示，不构成投资建议。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_market_comparison(
    query: str,
    engine: UnifiedRetrievalEngine | None = None,
    *,
    include_valuation: bool = True,
    include_watchlist: bool = True,
    peer_limit: int = 8,
    news_limit: int = 5,
    save_report: bool = True,
) -> ComparisonResult:
    """
    爬取网络资源并做实时对比分析。

    Args:
        query: 公司名或代码
        engine: 检索引擎（include_valuation=True 时需要）
        include_valuation: 是否叠加财报估值结论
        include_watchlist: 是否对比 AGENT_WATCHLIST
        peer_limit: 同业样本数量
        news_limit: 新闻条数
        save_report: 是否写入 docs/comparison/
    """
    entity_name, entity_id = _resolve_entity(query)
    code = entity_id.split(".")[0]

    logger.info("实时对比分析: %s (%s)", entity_name, entity_id)

    valuation: ValuationResult | None = None
    fundamentals: dict = {}
    if include_valuation and engine is not None:
        try:
            valuation = analyze_valuation(
                query,
                engine,
                save_report=False,
            )
            fundamentals = valuation.fundamentals or {}
        except Exception as exc:
            logger.warning("财报估值跳过: %s", exc)

    snapshot = fetch_market_snapshot_enriched(entity_id, entity_name, fundamentals)
    target = snapshot_to_peer(snapshot)
    industry = snapshot.industry

    # 爬取同业 + 报价
    peer_pairs = fetch_industry_peers(industry, exclude_code=code, limit=peer_limit)
    peers = fetch_peer_quotes(peer_pairs, max_peers=peer_limit)

    # 同业接口失败时，用监控列表其余标的作参考对比
    watchlist: list[PeerQuote] = []
    wl_ids = list(AGENT_WATCHLIST)
    if include_watchlist and wl_ids:
        watchlist = fetch_watchlist_quotes(wl_ids)

    if not peers and watchlist:
        peers = [q for q in watchlist if q.entity_id != entity_id][:peer_limit]
        logger.info("同业数据不可用，改用监控列表 %d 只做参考对比", len(peers))

    # 网络新闻
    news = fetch_stock_news(entity_name, entity_id, limit=news_limit)
    sentiment = _analyze_news_sentiment(news)

    stats = _build_industry_stats(target, peers)
    relative = _relative_verdict(target, stats, sentiment)

    result = ComparisonResult(
        entity_name=entity_name,
        entity_id=entity_id,
        industry=industry,
        target=target,
        peers=peers,
        watchlist=watchlist,
        industry_stats=stats,
        news=news,
        news_sentiment=sentiment,
        relative_verdict=relative,
        valuation=valuation,
    )
    result.summary = _build_summary(result)

    if save_report:
        COMPARISON_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = date.today().isoformat()
        report_path = COMPARISON_REPORT_DIR / f"{entity_id.replace('.', '_')}_{stamp}.md"
        report_path.write_text(_render_report(result), encoding=DEFAULT_ENCODING)
        json_path = report_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {
                    "entity_name": result.entity_name,
                    "entity_id": result.entity_id,
                    "industry": result.industry,
                    "summary": result.summary,
                    "relative_verdict": result.relative_verdict,
                    "news_sentiment": result.news_sentiment,
                    "target": asdict(result.target),
                    "peers": [asdict(p) for p in result.peers],
                    "watchlist": [asdict(w) for w in result.watchlist],
                    "industry_stats": asdict(result.industry_stats),
                    "news": [asdict(n) for n in result.news],
                    "valuation_verdict": (
                        result.valuation.verdict if result.valuation else None
                    ),
                    "report_path": str(report_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding=DEFAULT_ENCODING,
        )
        result.report_path = report_path

    return result


def analyze_watchlist_comparison(
    engine: UnifiedRetrievalEngine | None = None,
    *,
    save_report: bool = True,
) -> ComparisonResult:
    """监控列表批量实时对比（以列表第一只为展示主体）。"""
    if not AGENT_WATCHLIST:
        raise ValueError("监控列表为空，请配置 FINANCIAL_POC_AGENT_WATCHLIST")

    primary = AGENT_WATCHLIST[0]
    registry = get_stock_registry()
    found = registry.lookup_by_id(primary) or {"entity_id": primary, "entity_name": primary}
    name = str(found.get("entity_name") or primary)

    result = analyze_market_comparison(
        name,
        engine,
        include_valuation=False,
        include_watchlist=True,
        save_report=False,
    )
    result.watchlist = fetch_watchlist_quotes(list(AGENT_WATCHLIST))
    result.summary = (
        f"监控列表 {len(AGENT_WATCHLIST)} 只实时对比；"
        + "；".join(
            f"{q.entity_name} PE={q.pe_ttm or '—'}"
            for q in result.watchlist[:5]
        )
    )

    if save_report:
        COMPARISON_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = date.today().isoformat()
        report_path = COMPARISON_REPORT_DIR / f"watchlist_{stamp}.md"
        report_path.write_text(_render_report(result), encoding=DEFAULT_ENCODING)
        result.report_path = report_path

    return result
