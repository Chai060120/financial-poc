"""
一键全量分析：导入财报 → 抽取指标/关键词 → 爬取网络 → 对比股价 → 高估/低估结论。

不依赖 LLM API。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DEFAULT_ENCODING, DOCS_DIR, RAW_PDF_DIR, setup_logging
from src.analysis.market_compare import ComparisonResult, analyze_market_comparison
from src.analysis.valuation import ValuationResult, analyze_valuation
from src.collectors.pdf_collector import collect_pdf_paths
from src.utils.entity_parser import detect_entity_in_text, parse_filename
from src.utils.stock_registry import get_stock_registry
from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine

logger = setup_logging(__name__)

ANALYSIS_REPORT_DIR = DOCS_DIR / "analysis"

_KEYWORD_SEEDS: tuple[str, ...] = (
    "营业收入",
    "净利润",
    "毛利率",
    "净资产收益率",
    "ROE",
    "每股收益",
    "现金流",
    "分红",
    "主营业务",
    "市场份额",
    "产能",
    "研发",
    "数字化",
    "不良贷款",
    "拨备覆盖率",
    "增长",
    "下降",
    "风险",
    "竞争",
    "创新",
)


@dataclass
class FullAnalysisResult:
    entity_name: str
    entity_id: str
    period_label: str
    keywords: list[str] = field(default_factory=list)
    fundamentals: dict[str, Any] = field(default_factory=dict)
    valuation: ValuationResult | None = None
    comparison: ComparisonResult | None = None
    final_verdict: str = "合理"
    final_score: float = 0.0
    final_confidence: str = "中"
    data_warnings: list[str] = field(default_factory=list)
    valuation_reliable: bool = True
    executive_summary: str = ""
    synthesis_reasons: list[str] = field(default_factory=list)
    pdf_source: str = ""
    report_path: Path | None = None


def _retrieve_text(
    engine: UnifiedRetrievalEngine,
    question: str,
    *,
    entity_id: str = "",
    top_k: int = 6,
) -> str:
    kwargs: dict[str, Any] = {"top_k": top_k}
    if entity_id:
        kwargs["entity_id"] = entity_id
    results = engine.retrieve(question, **kwargs)
    return "\n".join(str(item.get("text") or "") for item in results)


def extract_keywords(
    engine: UnifiedRetrievalEngine,
    entity_name: str,
    entity_id: str,
    period_label: str,
    fundamentals: dict[str, Any],
    news_titles: list[str],
) -> list[str]:
    """从财报检索结果与网络新闻中提取关键词。"""
    found: set[str] = set()

    _SKIP_KEYWORD_KEYS = frozenset({
        "report_year", "report_type", "period_label",
        "revenue_growth_pct", "profit_growth_pct", "peg",
        "revenue", "net_profit", "attributable_profit", "eps", "bvps", "roe",
        "operating_profit", "total_assets", "total_equity", "cash_flow_operating",
    })

    for key, item in fundamentals.items():
        if key in _SKIP_KEYWORD_KEYS:
            continue
        if isinstance(item, dict):
            label = str(item.get("label") or "").strip()
            if label:
                found.add(label)

    report_text = _retrieve_text(
        engine,
        f"{entity_name}{period_label} 经营情况讨论与分析 主营业务 核心竞争力 行业格局",
        entity_id=entity_id,
    )
    for seed in _KEYWORD_SEEDS:
        if seed in report_text:
            found.add(seed)

    for match in re.finditer(r"([\u4e00-\u9fff]{2,10})(?:同比|环比)?(?:增长|下降|提升|改善|扩大)", report_text):
        phrase = match.group(1).strip()
        if len(phrase) >= 2:
            found.add(phrase)

    for title in news_titles:
        for seed in _KEYWORD_SEEDS:
            if seed in title:
                found.add(seed)

    return sorted(found)[:20]


def synthesize_final_verdict(
    valuation: ValuationResult,
    comparison: ComparisonResult,
) -> tuple[str, float, str, list[str]]:
    """综合财报估值 + 网络对比 + 新闻情绪 → 最终高估/合理/低估。"""
    reasons: list[str] = []
    score = valuation.score

    if not getattr(valuation, "data_reliable", True):
        return (
            "无法可靠判断估值",
            0.0,
            "低",
            ["财务指标或估值数据质量不足，无法给出可靠结论"]
            + list(getattr(valuation, "warnings", []) or [])[:4],
        )

    rel = comparison.relative_verdict or ""
    if "偏贵" in rel:
        score += 1.0
        reasons.append(f"网络对比: {rel}")
    elif "偏便宜" in rel:
        score -= 1.0
        reasons.append(f"网络对比: {rel}")

    sentiment = comparison.news_sentiment or ""
    if sentiment == "偏负面":
        score += 0.5
        reasons.append("近期网络资讯情绪偏负面")
    elif sentiment == "偏正面":
        score -= 0.5
        reasons.append("近期网络资讯情绪偏正面")

    for reason in valuation.reasons[:5]:
        if reason not in reasons:
            reasons.append(reason)

    if score >= 2.0:
        verdict = "高估"
    elif score <= -2.0:
        verdict = "低估"
    else:
        verdict = valuation.verdict or "合理"

    confidence = valuation.confidence
    if comparison.industry_stats.peer_count >= 2 and valuation.confidence != "低":
        confidence = "较高" if len(reasons) >= 5 else "中"
    if not comparison.target.pe_ttm and not comparison.target.price:
        confidence = "低"

    return verdict, score, confidence, reasons


def _build_executive_summary(result: FullAnalysisResult) -> str:
    parts = [
        f"{result.entity_name}（{result.entity_id}，{result.period_label}）",
        f"最终结论 **{result.final_verdict}**（置信度 {result.final_confidence}）",
    ]
    v = result.valuation
    c = result.comparison
    if v and v.fundamentals.get("net_profit"):
        parts.append(f"财报净利润 {v.fundamentals['net_profit'].get('display', '')}")
    if c and c.target.pe_ttm is not None:
        parts.append(f"实时 PE≈{c.target.pe_ttm:.1f}")
    if c and c.industry_stats.avg_pe is not None:
        parts.append(f"同业平均 PE≈{c.industry_stats.avg_pe:.1f}")
    if result.keywords:
        parts.append(f"关键词: {', '.join(result.keywords[:8])}")
    return "；".join(parts)


def _render_full_report(result: FullAnalysisResult) -> str:
    lines = [
        f"# 全量分析报告 · {result.entity_name} ({result.entity_id})",
        "",
        f"- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 报告期: {result.period_label}",
        f"- **最终结论: {result.final_verdict}**（评分 {result.final_score:+.1f}，置信度 {result.final_confidence}）",
        "",
        "## 一句话摘要",
        "",
        result.executive_summary,
        "",
        "## 关键词",
        "",
    ]
    if result.keywords:
        lines.append(", ".join(result.keywords))
    else:
        lines.append("（未提取到显著关键词）")

    if result.pdf_source:
        lines.extend(["", "## 财报来源", "", f"- {result.pdf_source}"])

    v = result.valuation
    if v:
        lines.extend(["", "## 财报基本面", ""])
        for key, label in (
            ("net_profit", "净利润"),
            ("revenue", "营业收入"),
            ("eps", "每股收益"),
            ("bvps", "每股净资产"),
            ("roe", "ROE"),
        ):
            item = v.fundamentals.get(key)
            if isinstance(item, dict) and item.get("display"):
                lines.append(f"- {label}: {item['display']}")
        if v.fundamentals.get("profit_growth_pct") is not None:
            lines.append(f"- 净利润同比: {v.fundamentals['profit_growth_pct']}%")
        lines.append(f"- 财报估值（单维度）: {v.verdict}（{v.confidence}）")

    c = result.comparison
    if c:
        lines.extend(["", "## 网络实时对比", ""])
        t = c.target
        if t.price is not None:
            lines.append(f"- 现价: {t.price}")
        if t.pe_ttm is not None:
            lines.append(f"- PE: {t.pe_ttm}")
        if t.pb is not None:
            lines.append(f"- PB: {t.pb}")
        lines.append(f"- 相对同业: {c.relative_verdict}")
        lines.append(f"- 新闻情绪: {c.news_sentiment}")
        if c.industry_stats.avg_pe is not None:
            lines.append(
                f"- 同业平均 PE: {c.industry_stats.avg_pe}（排名 {c.industry_stats.target_pe_rank}）"
            )
        if c.news:
            lines.extend(["", "### 近期网络资讯", ""])
            for item in c.news[:6]:
                lines.append(f"- [{item.source}] {item.title}")

    lines.extend(["", "## 综合研判依据", ""])
    for reason in result.synthesis_reasons:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "> 免责声明: 本报告由 PoC 规则引擎自动生成，不构成投资建议。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_entity_full(
    entity_name: str,
    entity_id: str,
    engine: UnifiedRetrievalEngine,
    *,
    pdf_source: str = "",
    save_report: bool = True,
) -> FullAnalysisResult:
    """对单家公司执行全量分析（假设财报已入库）。"""
    logger.info("全量分析: %s (%s)", entity_name, entity_id)

    valuation = analyze_valuation(entity_name, engine, save_report=False)
    comparison = analyze_market_comparison(
        entity_name,
        engine,
        include_valuation=False,
        save_report=False,
    )

    period = str(
        valuation.fundamentals.get("period_label")
        or valuation.report_year
    )
    news_titles = [n.title for n in comparison.news]
    keywords = extract_keywords(
        engine,
        entity_name,
        entity_id,
        period,
        valuation.fundamentals,
        news_titles,
    )

    verdict, score, confidence, reasons = synthesize_final_verdict(valuation, comparison)

    result = FullAnalysisResult(
        entity_name=entity_name,
        entity_id=entity_id,
        period_label=period,
        keywords=keywords,
        fundamentals=valuation.fundamentals,
        valuation=valuation,
        comparison=comparison,
        final_verdict=verdict,
        final_score=score,
        final_confidence=confidence,
        synthesis_reasons=reasons,
        pdf_source=pdf_source,
        data_warnings=list(getattr(valuation, "warnings", []) or []),
        valuation_reliable=getattr(valuation, "data_reliable", True),
    )
    result.executive_summary = _build_executive_summary(result)

    if save_report:
        ANALYSIS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = date.today().isoformat()
        report_path = ANALYSIS_REPORT_DIR / f"{entity_id.replace('.', '_')}_{stamp}.md"
        report_path.write_text(_render_full_report(result), encoding=DEFAULT_ENCODING)
        json_path = report_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {
                    "entity_name": result.entity_name,
                    "entity_id": result.entity_id,
                    "period_label": result.period_label,
                    "final_verdict": result.final_verdict,
                    "final_score": result.final_score,
                    "final_confidence": result.final_confidence,
                    "executive_summary": result.executive_summary,
                    "keywords": result.keywords,
                    "synthesis_reasons": result.synthesis_reasons,
                    "report_path": str(report_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding=DEFAULT_ENCODING,
        )
        result.report_path = report_path

    return result


def _resolve_target(target: str | Path | None) -> tuple[str, Path | None, str | None]:
    """
    解析用户输入：PDF 路径 / 公司名 / None（处理 raw/pdf 全部）。

    Returns:
        (mode, pdf_path, company_query)
        mode: "pdf" | "company" | "all_pdfs"
    """
    if target is None or (isinstance(target, str) and not target.strip()):
        pdfs = collect_pdf_paths(RAW_PDF_DIR)
        if not pdfs:
            raise ValueError(f"请在 {RAW_PDF_DIR} 放入财报 PDF，或指定公司名/文件路径")
        return "all_pdfs", None, None

    path = Path(str(target))
    if path.suffix.lower() == ".pdf" and path.exists():
        return "pdf", path, None

    if path.suffix.lower() == ".pdf":
        candidate = RAW_PDF_DIR / path.name
        if candidate.exists():
            return "pdf", candidate, None
        raise FileNotFoundError(f"找不到 PDF: {target}")

    return "company", None, str(target).strip()


def run_full_analysis(
    engine: UnifiedRetrievalEngine,
    process_pdfs_fn,
    target: str | Path | None = None,
    *,
    save_report: bool = True,
) -> list[FullAnalysisResult]:
    """
    一键全量分析入口。

    Args:
        engine: 检索引擎
        process_pdfs_fn: callable(paths, build_index=True) -> dict
        target: PDF 路径 / 公司名 / None=处理 data/raw/pdf/ 全部
    """
    mode, pdf_path, company_query = _resolve_target(target)
    results: list[FullAnalysisResult] = []

    if mode == "company":
        registry = get_stock_registry()
        found = detect_entity_in_text(company_query or "") or registry.lookup_by_name(
            company_query or ""
        )
        if not found:
            raise ValueError(f"无法识别公司: {company_query}")
        name = str(found.get("entity_name") or "")
        eid = str(found.get("entity_id") or "")
        results.append(
            analyze_entity_full(name, eid, engine, save_report=save_report)
        )
        return results

    if mode == "pdf":
        assert pdf_path is not None
        payload = process_pdfs_fn([pdf_path], build_index=True)
        if not payload.get("success"):
            raise RuntimeError(payload.get("message", "PDF 处理失败"))
        entities = payload.get("entities") or []
        if not entities:
            parsed = parse_filename(pdf_path.name)
            entities = [
                {
                    "entity_name": parsed.get("entity_name", ""),
                    "entity_id": parsed.get("entity_id", ""),
                }
            ]
        for ent in entities:
            name = str(ent.get("entity_name") or "")
            eid = str(ent.get("entity_id") or "")
            if not name or not eid or eid == "UNKNOWN":
                continue
            results.append(
                analyze_entity_full(
                    name,
                    eid,
                    engine,
                    pdf_source=str(pdf_path),
                    save_report=save_report,
                )
            )
        if not results:
            raise ValueError(f"未能从 PDF 识别公司: {pdf_path.name}")
        return results

    # all_pdfs
    pdfs = collect_pdf_paths(RAW_PDF_DIR)
    payload = process_pdfs_fn(pdfs, build_index=True)
    if not payload.get("success"):
        raise RuntimeError(payload.get("message", "PDF 处理失败"))
    entities = payload.get("entities") or []
    if not entities:
        raise ValueError("PDF 已处理但未识别到任何公司实体")
    for ent in entities:
        name = str(ent.get("entity_name") or "")
        eid = str(ent.get("entity_id") or "")
        if not name or not eid or eid == "UNKNOWN":
            continue
        results.append(
            analyze_entity_full(name, eid, engine, save_report=save_report)
        )
    return results
