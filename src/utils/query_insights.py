"""
检索结果洞察：从 Top 片段中规则抽取财务指标，生成演示友好的摘要。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.utils.entity_parser import detect_query_entity
from src.utils.query_filters import (
    asks_full_year_period,
    is_annual_summary_chunk,
    is_quarterly_chunk,
)

# 问句关键词 → 抽取指标名
_QUESTION_METRICS: tuple[tuple[str, str], ...] = (
    ("归属于上市公司股东的净利润", "净利润"),
    ("归属于本行股东的净利润", "净利润"),
    ("扣非净利润", "扣非净利润"),
    ("净利润", "净利润"),
    ("营业总收入", "营业收入"),
    ("营业收入", "营业收入"),
    ("营收", "营业收入"),
    ("不良贷款率", "不良贷款率"),
    ("拨备覆盖率", "拨备覆盖率"),
    ("总资产", "总资产"),
    ("净资产", "净资产"),
    ("基本每股收益", "每股收益"),
    ("每股收益", "每股收益"),
    ("净资产收益率", "净资产收益率"),
    ("ROE", "净资产收益率"),
    ("roe", "净资产收益率"),
    ("每股净资产", "每股净资产"),
    ("经营活动产生的现金流量净额", "经营现金流"),
    ("现金流", "经营现金流"),
    ("每10股派息", "分红"),
    ("现金红利", "分红"),
    ("分红", "分红"),
)

# 指标 → 文本抽取正则（按优先级）
_METRIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "净利润": (
        r"归属于上市公司股东的[\s\n]*(?:净[\s\n]*)?利润[\s\n]*([\d,]+\.?\d*)",
        r"归属于本行(?:普通股)?股东的(?:净[\s\n]*)?利润[\s\n]*([\d,]+\.?\d*)",
        r"归属于公司普通股股东的(?:净[\s\n]*)?利润[\s\n]*([\d,]+\.?\d*)",
        r"净利润[\s\n]*([\d,]+\.?\d*)",
    ),
    "扣非净利润": (
        r"扣除非经常性损益后归属于(?:公司|本行)?(?:普通股)?股东的(?:净[\s\n]*)?利润"
        r"[\s\n]*([\d,]+\.?\d*)",
    ),
    "营业收入": (
        r"营业总收入[\s\n]*([\d,]+\.?\d*)",
        r"营业收入[\s\n]*([\d,]+\.?\d*)",
    ),
    "不良贷款率": (r"不良贷款率[\s\n]*([\d.]+)",),
    "拨备覆盖率": (r"拨备覆盖率[\s\n]*([\d.]+)",),
    "总资产": (r"(?:资产总额|总资产)[\s\n]*([\d,]+\.?\d*)",),
    "净资产": (r"归属于上市公司股东的(?:净[\s\n]*)?资产[\s\n]*([\d,]+\.?\d*)",),
    "每股收益": (
        r"基本每股收益(?:\(元/股\))?[\s\n]*([\d.]+)",
        r"基本每股收益[\s\n]*([\d.]+)",
    ),
    "净资产收益率": (
        r"加权平均净资产收益率[\s\n]*(?:\(%?\))?[\s\n]*([\d.]+)",
        r"加权平均净资产收益率[\s\n]*([\d.]+)",
        r"净资产收益率[\s\n]*(?:\(%?\))?[\s\n]*([\d.]+)",
        r"净资产收益率[\s\n]*([\d.]+)",
    ),
    "每股净资产": (
        r"每股净资产(?:\(元/股\))?[\s\n]*([\d.]+)",
        r"归属于上市公司股东的每股净资产[\s\n]*([\d.]+)",
        r"归属于(?:公司|本行)?(?:普通股)?股东的每股净资产[\s\n]*([\d.]+)",
        r"每股净资产[\s\n]*(?:\(元\))?[\s\n]*([\d.]+)",
    ),
    "经营现金流": (
        r"经营活动产生的现金流量(?:净额)?[\s\n]*([\d,]+\.?\d*)",
    ),
    "分红": (
        r"每10[\s]*股派(?:发现金)?(?:红利|息)[数]?[\s\n]*(?:\(元\)[^\n]*)?[\s\n]*([\d.]+)",
        r"每10股派发现金红利([\d.]+)元",
    ),
}

_PERCENT_METRICS = frozenset({"不良贷款率", "拨备覆盖率", "净资产收益率"})
_PER_SHARE_METRICS = frozenset({"每股收益", "每股净资产"})
_AMOUNT_METRICS = frozenset({"净利润", "营业收入", "总资产", "净资产", "经营现金流"})
_QUARTERLY_SUM_LABELS: dict[str, str] = {
    "营业收入": r"营业收入",
    "净利润": r"归属于上市公司股东[\s\n]*的(?:净[\s\n]*)?利润",
}


def _score_metric_candidate(
    *,
    metric: str,
    raw: str,
    text: str,
    question: str,
    rank: int,
    rerank_score: float | None,
) -> float:
    """为抽取候选打分，优先全年摘要表，降低分季度误命中。"""
    score = max(0.0, 12.0 - rank)
    if rerank_score is not None:
        score += rerank_score * 3.0

    if is_annual_summary_chunk(text):
        score += 45.0
    if is_quarterly_chunk(text):
        score -= 55.0
    if asks_full_year_period(question) and is_quarterly_chunk(text):
        score -= 40.0

    num = _parse_number(raw)
    if num is None:
        return score

    if metric in _AMOUNT_METRICS:
        if asks_full_year_period(question) and num >= 1e10:
            score += 30.0
        if asks_full_year_period(question):
            score += min(num / 1e9, 25.0)
        if asks_full_year_period(question) and is_quarterly_chunk(text) and num < 8e10:
            score -= 45.0

    return score


def _format_raw_amount(value: float) -> str:
    return f"{value:,.2f}"


def _try_sum_quarterly_metric(text: str, metric: str) -> tuple[str, str, str] | None:
    """从分季度表将四个季度数值加总，推算全年口径。"""
    if not is_quarterly_chunk(text):
        return None
    label = _QUARTERLY_SUM_LABELS.get(metric)
    if not label:
        return None

    pattern = rf"{label}[\s\n]+((?:[\d,]+\.?\d*[\s\n]+){{3,4}})"
    match = re.search(pattern, text, re.I)
    if not match:
        return None

    nums = [_parse_number(item) for item in re.findall(r"[\d,]+\.?\d*", match.group(1))]
    nums = [item for item in nums if item is not None]
    if len(nums) < 4:
        return None

    total = sum(nums[:4])
    raw = _format_raw_amount(total)
    snippet = (
        f">> {label}（四个季度合计）\n"
        f">> {raw} 元"
    )
    return raw, snippet, text


@dataclass
class _MetricCandidate:
    metric: str
    raw: str
    context: str
    snippet: str
    full_text: str
    rank: int
    meta: dict
    score: float
    summed_quarters: bool = False


@dataclass
class MetricInsight:
    """单条抽取到的指标。"""

    metric: str
    raw_value: str
    display: str
    detail: str
    snippet: str
    source_rank: int


@dataclass
class QueryInsight:
    """一次检索的结构化摘要。"""

    entity_name: str = ""
    entity_id: str = ""
    report_label: str = ""
    file_name: str = ""
    page: int = 0
    metrics: list[MetricInsight] = field(default_factory=list)


def detect_question_metrics(question: str) -> list[str]:
    """从问句识别用户关心的指标（去重、保序）。"""
    text = str(question or "")
    found: list[str] = []
    seen: set[str] = set()
    for keyword, metric in _QUESTION_METRICS:
        if keyword in text and metric not in seen:
            found.append(metric)
            seen.add(metric)
    return found


def _parse_number(raw: str) -> float | None:
    cleaned = str(raw or "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _detect_amount_unit(context: str, full_text: str = "") -> str:
    """根据上下文判断金额单位。"""
    probe = f"{context}\n{full_text[:800]}"
    if "人民币百万元" in probe or "(人民币百万元" in probe:
        return "百万"
    if re.search(r"单位:\s*元|单位：元|币种:人民币", probe):
        return "元"
    if "亿元" in probe:
        return "亿"
    return "元"


def _format_amount(value: float, unit: str) -> tuple[str, str]:
    """返回 (主显示, 补充说明)。"""
    if unit == "百万":
        yuan = value * 1_000_000
        return f"{yuan / 1e8:.2f} 亿元", f"{value:,.0f} 百万元"
    if unit == "亿":
        yuan = value * 1e8
        return f"{value:.2f} 亿元", f"{yuan:,.2f} 元"
    if value >= 1e8:
        return f"{value / 1e8:.2f} 亿元", f"{value:,.2f} 元"
    if value >= 1e4:
        return f"{value / 1e4:.2f} 万元", f"{value:,.2f} 元"
    return f"{value:,.2f} 元", ""


def _format_metric_display(metric: str, raw: str, context: str, full_text: str = "") -> tuple[str, str]:
    value = _parse_number(raw)
    if value is None:
        return raw, ""

    if metric in _PERCENT_METRICS:
        return f"{value:.2f}%", raw

    if metric in _PER_SHARE_METRICS:
        return f"{value:.2f} 元/股", raw

    if metric == "分红":
        return f"{value:.2f} 元/10股", raw

    unit = _detect_amount_unit(context, full_text)
    main, detail = _format_amount(value, unit)
    return main, detail


def _extract_snippet(text: str, raw_value: str, *, width: int = 160) -> str:
    """抽取含数值的原文短片段，并标注关键行。"""
    if not text or not raw_value:
        return ""

    idx = text.find(raw_value.replace(",", ""))
    if idx < 0:
        idx = text.find(raw_value)
    if idx < 0:
        return ""

    start = max(0, idx - width // 2)
    chunk = text[start : start + width].strip()
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    if not lines:
        return chunk

    highlighted: list[str] = []
    plain_digits = raw_value.replace(",", "")
    for line in lines[:6]:
        if raw_value in line or plain_digits in line.replace(",", ""):
            highlighted.append(f">> {line}")
        elif any(key in line for key in ("利润", "收入", "资产", "负债", "比率", "收益")):
            highlighted.append(f"   {line}")
    return "\n".join(highlighted) if highlighted else chunk[:120]


def _extract_metric_candidates(text: str, metric: str) -> list[tuple[str, str, str, str]]:
    """从单段文本抽取某指标的全部候选值。"""
    patterns = _METRIC_PATTERNS.get(metric, ())
    hits: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            raw = match.group(1).strip()
            if not raw or raw in seen:
                continue
            num = _parse_number(raw)
            if num is None:
                continue
            if (
                metric not in _PERCENT_METRICS
                and num < 100
                and metric not in _PER_SHARE_METRICS
                and metric in {"净利润", "营业收入", "总资产", "净资产", "经营现金流"}
            ):
                continue
            seen.add(raw)
            context = text[max(0, match.start() - 120) : match.end() + 40]
            hits.append((raw, context, _extract_snippet(text, raw), text))
    return hits


def _select_best_candidate(
    candidates: list[_MetricCandidate],
    metric: str,
    question: str,
) -> _MetricCandidate:
    """挑选最佳候选：全年问句优先摘要表；否则用四季度合计。"""
    if not candidates:
        raise ValueError("candidates 不能为空")

    if asks_full_year_period(question) and metric in _AMOUNT_METRICS:
        direct_annual = [
            item
            for item in candidates
            if not item.summed_quarters and not is_quarterly_chunk(item.full_text)
        ]
        if direct_annual:
            return max(
                direct_annual,
                key=lambda item: (item.score, _parse_number(item.raw) or 0.0),
            )

        summed = [item for item in candidates if item.summed_quarters]
        if summed:
            return max(summed, key=lambda item: item.score)

    return max(
        candidates,
        key=lambda item: (item.score, _parse_number(item.raw) or 0.0),
    )


def build_query_insight(
    question: str,
    results: list[dict],
) -> QueryInsight | None:
    """
    从检索结果构建结构化摘要。

    results: serialize_retrieval_result 格式的 dict 列表。
    """
    if not results:
        return None

    metrics_needed = detect_question_metrics(question)
    if not metrics_needed:
        return None

    entity_name, entity_id = detect_query_entity(question)
    insight = QueryInsight(entity_name=entity_name, entity_id=entity_id)

    for metric in metrics_needed:
        candidates: list[_MetricCandidate] = []

        for rank, item in enumerate(results, start=1):
            text = str(item.get("text") or "")
            meta = item.get("metadata") or {}
            rerank = item.get("rerank_score")
            rerank_score = float(rerank) if rerank is not None else None

            for raw, context, snippet, full_text in _extract_metric_candidates(text, metric):
                if (
                    asks_full_year_period(question)
                    and metric in _AMOUNT_METRICS
                    and is_quarterly_chunk(full_text, meta)
                ):
                    continue

                score = _score_metric_candidate(
                    metric=metric,
                    raw=raw,
                    text=full_text,
                    question=question,
                    rank=rank,
                    rerank_score=rerank_score,
                )
                candidates.append(
                    _MetricCandidate(
                        metric=metric,
                        raw=raw,
                        context=context,
                        snippet=snippet,
                        full_text=full_text,
                        rank=rank,
                        meta=meta,
                        score=score,
                    )
                )

            if asks_full_year_period(question) and metric in _QUARTERLY_SUM_LABELS:
                summed = _try_sum_quarterly_metric(text, metric)
                if summed:
                    raw, snippet, full_text = summed
                    candidates.append(
                        _MetricCandidate(
                            metric=metric,
                            raw=raw,
                            context="分季度表四季合计",
                            snippet=snippet,
                            full_text=full_text,
                            rank=rank,
                            meta=meta,
                            score=85.0 + max(0.0, 12.0 - rank),
                            summed_quarters=True,
                        )
                    )

        if not candidates:
            continue

        best = _select_best_candidate(candidates, metric, question)
        display, detail = _format_metric_display(
            metric, best.raw, best.context, best.full_text
        )
        if best.summed_quarters and detail:
            detail = f"{detail}；分季度合计"
        elif best.summed_quarters:
            detail = "分季度合计推算"
        insight.metrics.append(
            MetricInsight(
                metric=metric,
                raw_value=best.raw,
                display=display,
                detail=detail,
                snippet=best.snippet,
                source_rank=best.rank,
            )
        )

        if not insight.file_name:
            insight.file_name = str(best.meta.get("file_name") or "")
            page = best.meta.get("page") or 0
            insight.page = int(page) if page else 0
            year = str(best.meta.get("report_year") or "")
            rtype = str(best.meta.get("report_type") or "")
            insight.report_label = f"{year} {rtype}".strip()
            if not insight.entity_name:
                insight.entity_name = str(best.meta.get("entity_name") or "")

    return insight if insight.metrics else None


def extract_metric_from_text(
    text: str,
    metric: str,
    *,
    question: str = "",
) -> dict[str, str] | None:
    """从一段文本规则抽取单个指标，返回与 valuation fundamentals 兼容的结构。"""
    if not text or not metric:
        return None

    candidates: list[_MetricCandidate] = []
    q = question or metric
    for raw, context, snippet, full_text in _extract_metric_candidates(text, metric):
        score = _score_metric_candidate(
            metric=metric,
            raw=raw,
            text=full_text,
            question=q,
            rank=1,
            rerank_score=None,
        )
        candidates.append(
            _MetricCandidate(
                metric=metric,
                raw=raw,
                context=context,
                snippet=snippet,
                full_text=full_text,
                rank=1,
                meta={},
                score=score,
            )
        )

    if not candidates:
        return None

    best = _select_best_candidate(candidates, metric, q)
    display, _detail = _format_metric_display(metric, best.raw, best.context, best.full_text)
    return {
        "label": metric,
        "display": display,
        "raw": best.raw,
        "source": "text_extract",
    }


def format_query_insight(insight: QueryInsight) -> str:
    """格式化为终端友好的摘要块。"""
    lines = [
        "",
        "=" * 60,
        "检索摘要（自动从命中片段抽取）",
        "-" * 60,
    ]

    if insight.entity_name:
        entity_line = insight.entity_name
        if insight.entity_id:
            entity_line += f" ({insight.entity_id})"
        lines.append(f"  公司   {entity_line}")
    if insight.report_label:
        lines.append(f"  报告   {insight.report_label}")

    for item in insight.metrics:
        line = f"  {item.metric:<6} {item.display}"
        if item.detail:
            line += f"  ({item.detail})"
        lines.append(line)

    if insight.file_name:
        source = insight.file_name
        if insight.page:
            source += f" · 第 {insight.page} 页"
        lines.append(f"  出处   {source}")

    lines.append("=" * 60)
    return "\n".join(lines)


def mark_insight_source(rank: int, insight: QueryInsight | None) -> str:
    """标记哪条原文提供了摘要数据。"""
    if insight is None:
        return ""
    ranks = {item.source_rank for item in insight.metrics}
    return " ★ 摘要来源" if rank in ranks else ""
