"""ValuationCalculator：基于标准化指标计算 PE/PB。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ValuationMetrics:
    pe: float | None = None
    pb: float | None = None
    confidence: float = 0.0
    pe_note: str = ""
    pb_note: str = ""
    pe_source: str = ""
    pb_source: str = ""
    usable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "PE": self.pe,
            "PB": self.pb,
            "confidence": round(self.confidence, 2),
            "pe_note": self.pe_note,
            "pb_note": self.pb_note,
            "usable": self.usable,
        }


class ValuationCalculator:
    MIN_METRIC_CONFIDENCE = 0.8

    def calculate(
        self,
        current_price: float | None,
        fundamentals: dict[str, Any],
        *,
        live_pe: float | None = None,
        live_pb: float | None = None,
    ) -> ValuationMetrics:
        if current_price is None or current_price <= 0:
            return ValuationMetrics(pe_note="缺少有效现价", pb_note="缺少有效现价")

        eps_item = fundamentals.get("eps") or {}
        bvps_item = fundamentals.get("bvps") or {}
        eps_conf = float(eps_item.get("confidence") or 0)
        bvps_conf = float(bvps_item.get("confidence") or 0)
        eps = self._scalar(eps_item)
        bvps = self._scalar(bvps_item)

        result = ValuationMetrics()
        confidences: list[float] = []

        if live_pe is not None and live_pe > 0:
            result.pe = round(live_pe, 2)
            result.pe_source = "market"
            result.pe_note = "PE(动态)"
            confidences.append(0.9)
        elif eps is not None and eps > 0 and eps_conf >= self.MIN_METRIC_CONFIDENCE:
            result.pe = round(current_price / eps, 2)
            result.pe_source = "computed:price/eps"
            result.pe_note = "PE(推算)"
            confidences.append(eps_conf)
        elif eps is not None and eps <= 0:
            result.pe_note = "无法计算PE（EPS≤0）"
        elif eps_conf < self.MIN_METRIC_CONFIDENCE:
            result.pe_note = "无法计算PE（EPS置信度不足）"
        else:
            result.pe_note = "无法计算PE"

        if live_pb is not None and live_pb > 0:
            result.pb = round(live_pb, 2)
            result.pb_source = "market"
            result.pb_note = "PB(动态)"
            confidences.append(0.9)
        elif bvps is not None and bvps > 0 and bvps_conf >= self.MIN_METRIC_CONFIDENCE:
            result.pb = round(current_price / bvps, 2)
            result.pb_source = "computed:price/bvps"
            result.pb_note = "PB(推算)"
            confidences.append(bvps_conf)
        elif bvps is not None and bvps <= 0:
            result.pb_note = "无法计算PB（每股净资产≤0）"
        elif bvps_conf < self.MIN_METRIC_CONFIDENCE:
            result.pb_note = "无法计算PB（BVPS置信度不足）"
        else:
            result.pb_note = "无法计算PB"

        if confidences:
            result.confidence = min(confidences)
            result.usable = result.confidence >= self.MIN_METRIC_CONFIDENCE and (
                result.pe is not None or result.pb is not None
            )
        return result

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
