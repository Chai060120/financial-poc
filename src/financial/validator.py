"""FinancialDataValidator：报告生成前的财务数据质量检查。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationReport:
    warnings: list[str] = field(default_factory=list)
    reliable: bool = True
    quality_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "warnings": self.warnings,
            "reliable": self.reliable,
            "quality_score": round(self.quality_score, 2),
        }


class FinancialDataValidator:
    def validate(
        self,
        fundamentals: dict[str, Any],
        *,
        pe: float | None = None,
        pb: float | None = None,
        entity_name: str = "",
    ) -> ValidationReport:
        report = ValidationReport()
        is_financial = any(k in entity_name for k in ("银行", "保险", "平安", "招行"))

        roe = self._scalar(fundamentals.get("roe"))
        eps = self._scalar(fundamentals.get("eps"))
        bvps = self._scalar(fundamentals.get("bvps"))
        net_profit = self._scalar(
            fundamentals.get("net_profit") or fundamentals.get("attributable_profit")
        )
        revenue = self._scalar(fundamentals.get("revenue"))

        if roe is not None and roe > 100:
            report.warnings.append("ROE异常（>100%），可能单位错误或误抽年份")
        if eps is not None and eps > 50 and not is_financial:
            report.warnings.append("EPS异常（>50元/股），普通公司需核对")
        if bvps is not None and bvps > 500:
            report.warnings.append("每股净资产异常偏高，可能单位错误")
        if pe is not None:
            if pe < 1:
                report.warnings.append("PE异常（<1），可能EPS单位错误")
            elif pe > 200:
                report.warnings.append("PE异常（>200），可能EPS过小或单位错误")
        if pb is not None and pb < 0.1:
            report.warnings.append("PB异常（<0.1），可能每股净资产单位错误")
        if net_profit is not None and revenue is not None and net_profit > revenue:
            report.warnings.append("净利润大于营业收入，逻辑异常")
        if net_profit is not None and net_profit < 1 and revenue and revenue > 50:
            report.warnings.append("净利润规模异常偏小，可能单位错误")

        for key, label in (("eps", "每股收益"), ("bvps", "每股净资产"), ("roe", "ROE")):
            item = fundamentals.get(key)
            if isinstance(item, dict):
                conf = float(item.get("confidence") or 0)
                if 0 < conf < 0.8:
                    report.warnings.append(f"{label}抽取置信度偏低（{conf:.0%}）")

        if report.warnings:
            report.reliable = len(report.warnings) <= 1 and not any(
                "单位错误" in w or "异常" in w for w in report.warnings
            )
            report.quality_score = max(0.2, 1.0 - 0.15 * len(report.warnings))
        return report

    @staticmethod
    def _scalar(item: Any) -> float | None:
        if not isinstance(item, dict):
            return None
        if item.get("value") is not None:
            try:
                return float(item["value"])
            except (TypeError, ValueError):
                pass
        raw = str(item.get("raw") or item.get("display") or "")
        raw = raw.replace("元/股", "").replace("%", "").replace("亿元", "").replace(",", "")
        try:
            return float(raw.strip())
        except ValueError:
            return None
