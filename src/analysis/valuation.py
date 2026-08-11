"""
基于 PDF 财报 + 实时行情的规则化估值分析。

输出：高估 / 合理 / 低估，并给出依据（不依赖 LLM API）。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DEFAULT_ENCODING, DOCS_DIR, setup_logging
from src.analysis.market_data import (
    MarketSnapshot,
    enrich_market_from_fundamentals,
    fetch_market_snapshot,
)
from src.utils.entity_parser import detect_entity_in_text
from src.financial.metric_extractor import FinancialMetricExtractor
from src.financial.valuation import ValuationCalculator
from src.financial.validator import FinancialDataValidator
from src.utils.stock_registry import get_stock_registry
from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine

logger = setup_logging(__name__)

VALUATION_REPORT_DIR = DOCS_DIR / "valuation"

# 行业合理估值区间（PoC 默认，可按行业扩展）
_INDUSTRY_BENCHMARKS: dict[str, dict[str, tuple[float, float]]] = {
    "白酒": {"pe": (20.0, 35.0), "pb": (5.0, 12.0), "peg_fair": (1.0, 1.8)},
    "银行": {"pe": (4.0, 8.0), "pb": (0.5, 1.0), "peg_fair": (0.8, 1.5)},
    "保险": {"pe": (6.0, 12.0), "pb": (0.8, 1.5), "peg_fair": (0.8, 1.5)},
    "default": {"pe": (8.0, 25.0), "pb": (1.0, 4.0), "peg_fair": (0.8, 1.8)},
}

@dataclass
class ReportContext:
    report_year: str
    report_type: str
    period_label: str


def _period_label(year: str, report_type: str) -> str:
    if "半年" in report_type:
        return f"{year}年上半年"
    if "一季" in report_type:
        return f"{year}年一季度"
    if "三季" in report_type:
        return f"{year}年前三季度"
    return f"{year}年"


@dataclass
class ValuationResult:
    entity_name: str
    entity_id: str
    report_year: str
    verdict: str
    score: float
    confidence: str
    market: dict[str, Any] = field(default_factory=dict)
    fundamentals: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    valuation_metrics: dict[str, Any] = field(default_factory=dict)
    data_reliable: bool = True
    report_path: Path | None = None


def _resolve_entity(query: str) -> tuple[str, str]:
    registry = get_stock_registry()
    found = detect_entity_in_text(query) or registry.lookup_by_name(query)
    if not found:
        raise ValueError(f"无法识别公司: {query}")
    return str(found.get("entity_name") or ""), str(found.get("entity_id") or "")


def _benchmarks_for(industry: str, entity_name: str) -> dict[str, tuple[float, float]]:
    if any(keyword in industry for keyword in ("白酒", "酿酒")):
        return _INDUSTRY_BENCHMARKS["白酒"]
    if any(keyword in industry or keyword in entity_name for keyword in _FINANCIAL_KEYWORDS):
        return _INDUSTRY_BENCHMARKS["银行"]
    if "保险" in industry or "保险" in entity_name:
        return _INDUSTRY_BENCHMARKS["保险"]
    return _INDUSTRY_BENCHMARKS["default"]


def _parse_number(raw: str) -> float | None:
    cleaned = str(raw or "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_yoy_growth(text: str, label: str) -> float | None:
    """从主要会计数据表抽取同比增减（%）。"""
    pattern = rf"{label}[\s\n]+[\d,\.]+[\s\n]+[\d,\.]+[\s\n]+(\-?\d+\.?\d*)"
    match = re.search(pattern, text)
    if match:
        return _parse_number(match.group(1))
    return None


def _retrieve_text(engine: UnifiedRetrievalEngine, question: str, *, top_k: int = 5) -> str:
    results = engine.retrieve(question, top_k=top_k)
    return "\n".join(str(item.get("text") or "") for item in results)


_FINANCIAL_KEYWORDS = ("银行", "保险", "证券", "信托")


def _detect_report_context(
    engine: UnifiedRetrievalEngine,
    entity_name: str,
    entity_id: str,
) -> ReportContext:
    """从索引元数据推断最新财报周期（年报/半年报/季报均适用）。"""
    results = engine.retrieve(
        f"{entity_name} 主要会计数据 财务指标",
        top_k=12,
        entity_id=entity_id,
    )
    best_meta: dict[str, Any] = {}
    best_date = ""
    for item in results:
        meta = item.get("metadata") or {}
        report_date = str(meta.get("report_date") or meta.get("date") or "")
        if report_date >= best_date:
            best_date = report_date
            best_meta = meta

    year = str(best_meta.get("report_year") or "")
    report_type = str(best_meta.get("report_type") or "年报")
    if not year:
        year = str(date.today().year - 1)
        report_type = "年报"

    return ReportContext(
        report_year=year,
        report_type=report_type,
        period_label=_period_label(year, report_type),
    )


def _infer_industry_from_report(
    engine: UnifiedRetrievalEngine,
    entity_name: str,
    entity_id: str,
) -> str:
    """从财报正文推断行业（行情接口失败时的通用备用）。"""
    text = _retrieve_text(
        engine,
        f"{entity_name} 所属行业 行业分类 主营业务",
        top_k=4,
    )
    patterns = (
        r"所属(?:证监会)?行业[\s:：]*([^\n,，；;]{2,24})",
        r"行业分类[\s:：]*([^\n,，；;]{2,24})",
        r"主要(?:从事|业务)[^\n]{0,20}([^\n,，；;]{2,24})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value and value not in {"-", "--", "无"}:
                return value
    return ""


def _extract_fundamentals(
    engine: UnifiedRetrievalEngine,
    entity_name: str,
    entity_id: str,
    context: ReportContext,
) -> dict[str, Any]:
    """使用 FinancialMetricExtractor 从已索引财报抽取核心指标。"""
    extractor = FinancialMetricExtractor(engine)
    return extractor.extract_as_fundamentals(
        entity_name,
        entity_id,
        report_year=context.report_year,
        report_type=context.report_type,
        period_label=context.period_label,
    )


def _pe_label(market: MarketSnapshot) -> str:
    if (market.pe_source or "").startswith("computed"):
        return "PE(推算)"
    return "PE(动态)"


def _pb_label(market: MarketSnapshot) -> str:
    if (market.pb_source or "").startswith("computed"):
        return "PB(推算)"
    return "PB"


def _score_valuation(
    *,
    market: MarketSnapshot,
    fundamentals: dict[str, Any],
) -> tuple[str, float, str, list[str]]:
    """规则打分 → 高估 / 合理 / 低估。"""
    reasons: list[str] = []
    score = 0.0

    bench = _benchmarks_for(market.industry, market.entity_name)
    pe_low, pe_high = bench["pe"]
    pb_low, pb_high = bench["pb"]
    peg_low, peg_high = bench["peg_fair"]

    pe = market.pe_ttm
    pb = market.pb
    profit_growth = fundamentals.get("profit_growth_pct")
    roe_item = fundamentals.get("roe") or {}
    roe = roe_item.get("value")
    if roe is None:
        roe_text = str(roe_item.get("raw") or roe_item.get("display") or "")
        roe = _parse_number(roe_text.replace("%", ""))

    is_financial = any(
        keyword in (market.industry + market.entity_name)
        for keyword in _FINANCIAL_KEYWORDS
    )

    if pe is not None:
        pe_tag = _pe_label(market)
        if pe > pe_high * 1.15:
            score += 2.0
            reasons.append(
                f"市盈率 {pe_tag}≈{pe:.1f}，高于行业合理区间 {pe_low:.0f}~{pe_high:.0f}"
            )
        elif pe > pe_high:
            score += 1.0
            reasons.append(f"市盈率 {pe_tag}≈{pe:.1f}，处于行业合理区间上沿")
        elif pe < pe_low * 0.85:
            score -= 2.0
            reasons.append(
                f"市盈率 {pe_tag}≈{pe:.1f}，低于行业合理区间 {pe_low:.0f}~{pe_high:.0f}"
            )
        elif pe < pe_low:
            score -= 1.0
            reasons.append(f"市盈率 {pe_tag}≈{pe:.1f}，处于行业合理区间下沿")
        else:
            reasons.append(
                f"市盈率 {pe_tag}≈{pe:.1f}，处于行业合理区间 {pe_low:.0f}~{pe_high:.0f}"
            )
        if (market.pe_source or "").startswith("computed"):
            reasons.append("PE 由现价 ÷ 财报每股收益推算")

    if pb is not None:
        pb_tag = _pb_label(market)
        if is_financial:
            if pb > pb_high * 1.1:
                score += 1.5
                reasons.append(f"市净率 {pb_tag}≈{pb:.2f}，偏高（银行常用 PB 估值）")
            elif pb < pb_low * 0.9:
                score -= 1.5
                reasons.append(f"市净率 {pb_tag}≈{pb:.2f}，偏低（银行常用 PB 估值）")
            else:
                reasons.append(
                    f"市净率 {pb_tag}≈{pb:.2f}，处于常见区间 {pb_low:.1f}~{pb_high:.1f}"
                )
        elif pb > pb_high * 1.15:
            score += 1.0
            reasons.append(f"市净率 {pb_tag}≈{pb:.2f}，高于行业常见水平")
        elif pb < pb_low * 0.85:
            score -= 1.0
            reasons.append(f"市净率 {pb_tag}≈{pb:.2f}，低于行业常见水平")
        if (market.pb_source or "").startswith("computed"):
            reasons.append("PB 由现价 ÷ 财报每股净资产推算")

    if pe is not None and profit_growth is not None and profit_growth > 0:
        peg = pe / profit_growth
        fundamentals["peg"] = round(peg, 2)
        if peg > peg_high * 1.2:
            score += 1.5
            reasons.append(
                f"PEG≈{peg:.2f}（PE/利润增速），成长性不足以支撑当前估值"
            )
        elif peg < peg_low * 0.8:
            score -= 1.5
            reasons.append(
                f"PEG≈{peg:.2f}（PE/利润增速），盈利增速相对估值更有吸引力"
            )
        else:
            reasons.append(f"PEG≈{peg:.2f}，成长性与估值匹配度尚可")

    if roe is not None:
        if roe >= 20 and score > 0:
            score -= 0.5
            reasons.append(f"ROE≈{roe:.1f}%，盈利质量较高，对高估值有一定支撑")
        elif roe >= 20 and score < 0:
            score += 0.5
        elif roe < 8 and not is_financial:
            score += 0.5
            reasons.append(f"ROE≈{roe:.1f}%，盈利质量偏弱")

    # 无实时 PE/PB 时，仅依据财报基本面做弱信号判断
    if pe is None and pb is None:
        profit_growth_val = profit_growth if profit_growth is not None else 0.0
        if roe is not None and roe >= 25 and profit_growth_val >= 10:
            score -= 1.0
            reasons.append(
                f"财报显示 ROE≈{roe:.1f}%、净利润同比 {profit_growth_val:.1f}%，基本面较强"
            )
        elif roe is not None and roe < 10 and profit_growth_val < 0:
            score += 1.0
            reasons.append(
                f"财报显示 ROE≈{roe:.1f}%、净利润同比下滑，基本面承压"
            )

    if score >= 2.0:
        verdict = "高估"
    elif score <= -2.0:
        verdict = "低估"
    else:
        verdict = "合理"

    confidence = "中"
    has_live_pe_pb = (
        market.pe_source
        and not market.pe_source.startswith("computed")
        and market.pe_ttm is not None
    ) or (
        market.pb_source
        and not market.pb_source.startswith("computed")
        and market.pb is not None
    )
    has_computed_pe_pb = market.pe_ttm is not None or market.pb is not None

    if not has_computed_pe_pb:
        confidence = "低"
        reasons.append("缺少 PE/PB（行情与财报推算均不可用），结论仅供参考")
    elif not has_live_pe_pb:
        confidence = "中"
        if not any("推算" in reason for reason in reasons):
            reasons.append("PE/PB 部分由现价与财报数据推算")
    elif len(reasons) >= 4:
        confidence = "较高"

    return verdict, score, confidence, reasons


def _render_report(result: ValuationResult) -> str:
    lines = [
        f"# 估值分析 · {result.entity_name} ({result.entity_id})",
        "",
        f"- 报告年份: {result.report_year}",
        f"- 分析日期: {date.today().isoformat()}",
        f"- **结论: {result.verdict}**（评分 {result.score:+.1f}，置信度 {result.confidence}）",
        "",
        "## 实时行情",
        "",
    ]
    market = result.market
    for key, label in (
        ("price", "现价"),
        ("pe_ttm", "市盈率"),
        ("pb", "市净率"),
        ("market_cap", "总市值"),
        ("industry", "行业"),
        ("price_source", "价格来源"),
        ("pe_source", "PE来源"),
        ("pb_source", "PB来源"),
    ):
        value = market.get(key)
        if value is not None and value != "":
            lines.append(f"- {label}: {value}")

    lines.extend(["", "## 财报基本面", ""])
    for key, label in (
        ("net_profit", "净利润"),
        ("revenue", "营业收入"),
        ("eps", "每股收益"),
        ("roe", "净资产收益率"),
        ("bvps", "每股净资产"),
    ):
        item = result.fundamentals.get(key)
        if isinstance(item, dict) and item.get("display"):
            lines.append(f"- {label}: {item['display']}")

    growth_parts = []
    if result.fundamentals.get("revenue_growth_pct") is not None:
        growth_parts.append(f"营收同比 {result.fundamentals['revenue_growth_pct']}%")
    if result.fundamentals.get("profit_growth_pct") is not None:
        growth_parts.append(f"净利润同比 {result.fundamentals['profit_growth_pct']}%")
    if growth_parts:
        lines.append(f"- 成长性: {', '.join(growth_parts)}")
    if result.fundamentals.get("peg") is not None:
        lines.append(f"- PEG: {result.fundamentals['peg']}")

    lines.extend(["", "## 分析依据", ""])
    for reason in result.reasons:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "> 免责声明: 本结论为 PoC 规则引擎输出，基于财报摘要与实时 PE/PB/PEG，"
            "不构成投资建议。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_valuation(
    query: str,
    engine: UnifiedRetrievalEngine,
    *,
    report_year: str | None = None,
    save_report: bool = True,
) -> ValuationResult:
    """
    对单只股票做估值分析。

    Args:
        query: 公司名或代码，如「贵州茅台」「600519.SH」
        engine: 检索引擎（需已索引该公司财报 PDF）
        report_year: 财报年份，默认当前年-1（最近年报）
        save_report: 是否写入 docs/valuation/
    """
    entity_name, entity_id = _resolve_entity(query)
    context = (
        ReportContext(
            report_year=report_year,
            report_type="年报",
            period_label=_period_label(report_year, "年报"),
        )
        if report_year
        else _detect_report_context(engine, entity_name, entity_id)
    )

    logger.info(
        "估值分析: %s (%s), period=%s",
        entity_name,
        entity_id,
        context.period_label,
    )

    fundamentals = _extract_fundamentals(engine, entity_name, entity_id, context)
    market = fetch_market_snapshot(entity_id, entity_name)
    if not market.industry:
        market.industry = _infer_industry_from_report(engine, entity_name, entity_id)

    calc = ValuationCalculator()
    val_metrics = calc.calculate(
        market.price,
        fundamentals,
        live_pe=market.pe_ttm,
        live_pb=market.pb,
    )

    if val_metrics.pe is not None:
        market.pe_ttm = val_metrics.pe
        market.pe_source = val_metrics.pe_source or market.pe_source
    if val_metrics.pb is not None:
        market.pb = val_metrics.pb
        market.pb_source = val_metrics.pb_source or market.pb_source

    validator = FinancialDataValidator()
    validation = validator.validate(
        fundamentals,
        pe=val_metrics.pe,
        pb=val_metrics.pb,
        entity_name=entity_name,
    )

    if not val_metrics.usable or not validation.reliable:
        verdict = "无法可靠判断估值"
        score = 0.0
        confidence = "低"
        reasons = ["财务指标或估值数据质量不足，无法给出可靠估值结论"]
        if validation.warnings:
            reasons.extend(validation.warnings[:4])
    else:
        verdict, score, confidence, reasons = _score_valuation(
            market=market,
            fundamentals=fundamentals,
        )
        if validation.warnings:
            reasons.append("部分指标存在数据质量提示，结论仅供参考")
            for warn in validation.warnings[:3]:
                if warn not in reasons:
                    reasons.append(warn)

    result = ValuationResult(
        entity_name=entity_name,
        entity_id=entity_id,
        report_year=context.report_year,
        verdict=verdict,
        score=score,
        confidence=confidence,
        market={
            "price": market.price,
            "pe_ttm": market.pe_ttm,
            "pb": market.pb,
            "market_cap": market.market_cap,
            "change_pct": market.change_pct,
            "industry": market.industry,
            "source": market.source,
            "price_source": market.price_source,
            "pe_source": market.pe_source,
            "pb_source": market.pb_source,
        },
        fundamentals=fundamentals,
        reasons=reasons,
        warnings=validation.warnings,
        valuation_metrics=val_metrics.to_dict(),
        data_reliable=validation.reliable and val_metrics.usable,
    )

    if save_report:
        VALUATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = (
            VALUATION_REPORT_DIR
            / f"{entity_id.replace('.', '_')}_{context.report_year}.md"
        )
        report_path.write_text(_render_report(result), encoding=DEFAULT_ENCODING)
        json_path = report_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {
                    **asdict(result),
                    "report_path": str(report_path),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding=DEFAULT_ENCODING,
        )
        result.report_path = report_path

    return result
