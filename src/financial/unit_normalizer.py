"""
财报数值单位检测与归一化。

支持：人民币百万元、亿元、万元、元、元/股、%。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class UnitKind(str, Enum):
    YI_YUAN = "亿元"
    WAN_YUAN = "万元"
    YUAN = "元"
    YUAN_PER_SHARE = "元/股"
    PERCENT = "%"
    BAIWAN_YUAN = "百万元"  # 原始单位，待转换


@dataclass
class NormalizedValue:
    value: float
    unit: str
    display: str
    raw: str
    source_unit: str


_UNIT_PATTERNS: tuple[tuple[re.Pattern[str], UnitKind], ...] = (
    (re.compile(r"人民币百万元|（人民币百万元）|\(人民币百万元\)"), UnitKind.BAIWAN_YUAN),
    (re.compile(r"单位:\s*元|单位：元|币种:\s*人民币|币种：人民币"), UnitKind.YUAN),
    (re.compile(r"人民币亿元|（亿元）|\(亿元\)"), UnitKind.YI_YUAN),
    (re.compile(r"人民币万元|（万元）|\(万元\)"), UnitKind.WAN_YUAN),
    (re.compile(r"元/股|\(元/股\)"), UnitKind.YUAN_PER_SHARE),
    (re.compile(r"%|百分点"), UnitKind.PERCENT),
)


def detect_unit_context(text: str, *, default: UnitKind = UnitKind.YUAN) -> UnitKind:
    """从段落上下文检测金额/每股/百分比单位。"""
    probe = text[:800]
    for pattern, kind in _UNIT_PATTERNS:
        if pattern.search(probe):
            return kind
    if "每股" in probe and "收益率" not in probe:
        return UnitKind.YUAN_PER_SHARE
    if "收益率" in probe or "(%)" in probe:
        return UnitKind.PERCENT
    return default


def parse_number(raw: str) -> float | None:
    cleaned = str(raw or "").replace(",", "").replace("，", "").strip()
    if not cleaned or cleaned in {"-", "--", "—", "nan"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_year_like(value: float) -> bool:
    return 1990 <= value <= 2035 and value == int(value)


def is_likely_date_fragment(raw: str, value: float) -> bool:
    """排除表头日期/月份误识别（如 1-6月、2025年）。"""
    text = str(raw or "").strip()
    if is_year_like(value):
        return True
    if text in {"1", "6", "12"} and value in {1, 6, 12}:
        return True
    if re.fullmatch(r"1-6|1–6|7-9|10-12", text):
        return True
    return False


def normalize_value(
    raw: str,
    *,
    metric_kind: str,
    unit_context: UnitKind,
) -> NormalizedValue | None:
    """
    将原始数值归一化为标准单位。

    - 百万元 → 亿元（÷100）
    - 元（大金额）→ 亿元（÷1e8）
    - 元/股、% 保持
    """
    num = parse_number(raw)
    if num is None:
        return None

    kind = unit_context
    if metric_kind in {"eps", "bvps"}:
        kind = UnitKind.YUAN_PER_SHARE
    elif metric_kind == "roe":
        kind = UnitKind.PERCENT

    if kind == UnitKind.BAIWAN_YUAN:
        yi = round(num / 100.0, 4)
        return NormalizedValue(
            value=yi,
            unit="亿元",
            display=f"{yi:.2f} 亿元",
            raw=raw,
            source_unit="百万元",
        )

    if kind == UnitKind.YI_YUAN:
        return NormalizedValue(
            value=num,
            unit="亿元",
            display=f"{num:.2f} 亿元",
            raw=raw,
            source_unit="亿元",
        )

    if kind == UnitKind.WAN_YUAN:
        yi = round(num / 10000.0, 4)
        return NormalizedValue(
            value=yi,
            unit="亿元",
            display=f"{yi:.2f} 亿元",
            raw=raw,
            source_unit="万元",
        )

    if kind == UnitKind.YUAN_PER_SHARE:
        val = round(num, 4)
        return NormalizedValue(
            value=val,
            unit="元/股",
            display=f"{val:.2f} 元/股",
            raw=raw,
            source_unit="元/股",
        )

    if kind == UnitKind.PERCENT:
        # 禁止 1250% 类错误：若 >100 且像是小数误放大，修正
        val = num
        if val > 100 and val <= 10000:
            val = round(val / 100.0, 4)
        val = round(val, 2)
        return NormalizedValue(
            value=val,
            unit="%",
            display=f"{val:.2f}%",
            raw=raw,
            source_unit="%",
        )

    # 默认「元」大金额
    if metric_kind in {
        "revenue",
        "operating_profit",
        "net_profit",
        "attributable_profit",
        "total_assets",
        "total_equity",
        "cash_flow_operating",
    }:
        if num >= 1e8:
            yi = round(num / 1e8, 4)
            return NormalizedValue(
                value=yi,
                unit="亿元",
                display=f"{yi:.2f} 亿元",
                raw=raw,
                source_unit="元",
            )
        if num >= 1e4:
            yi = round(num / 1e8, 4) if num >= 1e8 else round(num / 10000.0 / 10000.0, 4)
            # 1e4~1e8 可能是万元级
            if num >= 1e6:
                yi = round(num / 1e8, 4)
                return NormalizedValue(
                    value=yi,
                    unit="亿元",
                    display=f"{yi:.2f} 亿元",
                    raw=raw,
                    source_unit="元",
                )

    return NormalizedValue(
        value=round(num, 4),
        unit="元",
        display=f"{num:,.2f} 元",
        raw=raw,
        source_unit="元",
    )
