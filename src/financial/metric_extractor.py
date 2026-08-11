"""
FinancialMetricExtractor：从已索引财报结构化抽取财务指标。

优先解析「主要会计数据和财务指标」章节，结合单位上下文与指标词典，
避免简单关键词误匹配（如把年份 2024 当成 ROE）。
"""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import TOKENS_JSON, setup_logging
from src.financial.metric_dictionary import METRIC_DEFINITIONS, PRIORITY_SECTIONS
from src.financial.unit_normalizer import (
    UnitKind,
    detect_unit_context,
    is_likely_date_fragment,
    is_year_like,
    normalize_value,
    parse_number,
)
from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine

logger = setup_logging(__name__)

_NUMBER_PATTERN = re.compile(r"-?[\d,]+\.?\d*")


@dataclass
class ExtractedMetric:
    metric: str
    value: float
    unit: str
    year: int
    source_page: int
    confidence: float
    display: str = ""
    raw_value: str = ""
    source_section: str = ""
    table_context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_legacy_fundamental(self) -> dict[str, Any]:
        return {
            "label": self.metric,
            "display": self.display,
            "raw": self.raw_value,
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "source_page": self.source_page,
            "source_section": self.source_section,
        }


class FinancialMetricExtractor:
    MIN_CONFIDENCE = 0.8

    def __init__(self, engine: UnifiedRetrievalEngine) -> None:
        self.engine = engine

    def extract(
        self,
        entity_name: str,
        entity_id: str,
        *,
        report_year: str,
        report_type: str = "年报",
    ) -> list[ExtractedMetric]:
        target_year = int(report_year)
        chunks = self._collect_chunks(entity_name, entity_id)
        if not chunks:
            logger.warning("未找到 %s 财报片段", entity_name)
            return []

        results: dict[str, ExtractedMetric] = {}

        for metric_key, definition in METRIC_DEFINITIONS.items():
            metric = self._extract_metric(
                metric_key, definition, chunks, target_year=target_year
            )
            if metric:
                results[metric_key] = metric

        if "attributable_profit" in results:
            results["net_profit"] = results["attributable_profit"]

        if "bvps" not in results or results["bvps"].confidence < self.MIN_CONFIDENCE:
            computed = self._compute_bvps(chunks, target_year)
            if computed and (
                "bvps" not in results
                or computed.confidence > results["bvps"].confidence
            ):
                results["bvps"] = computed

        return list(results.values())

    def extract_as_fundamentals(
        self,
        entity_name: str,
        entity_id: str,
        *,
        report_year: str,
        report_type: str = "年报",
        period_label: str = "",
    ) -> dict[str, Any]:
        metrics = self.extract(
            entity_name, entity_id, report_year=report_year, report_type=report_type
        )
        payload: dict[str, Any] = {
            "report_year": report_year,
            "report_type": report_type,
            "period_label": period_label or f"{report_year}年",
        }
        for item in metrics:
            payload[item.metric] = item.to_legacy_fundamental()
        if payload.get("attributable_profit") and not payload.get("net_profit"):
            payload["net_profit"] = dict(payload["attributable_profit"])
            payload["net_profit"]["label"] = "net_profit"

        summary_text = self._merge_section_text(
            self._collect_chunks(entity_name, entity_id), "主要会计数据"
        )
        payload["profit_growth_pct"] = self._extract_yoy(summary_text, "净利润")
        if payload["profit_growth_pct"] is None:
            payload["profit_growth_pct"] = self._extract_yoy(
                summary_text, "归属于母公司股东"
            )
        payload["revenue_growth_pct"] = self._extract_yoy(summary_text, "营业收入")
        return payload

    def _collect_chunks(
        self, entity_name: str, entity_id: str
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        seen: set[str] = set()

        # 优先：直接从 tokens.json 按 entity + section 读取（比检索更稳定）
        if TOKENS_JSON.is_file():
            try:
                import json

                tokens = json.loads(TOKENS_JSON.read_text(encoding="utf-8"))
                for token in tokens:
                    meta = token.get("metadata") or {}
                    if str(meta.get("entity_id") or "") != entity_id:
                        continue
                    text = str(token.get("text") or "").strip()
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    chunks.append(
                        {
                            "text": text,
                            "section": str(meta.get("section") or ""),
                            "page": int(meta.get("page") or 0),
                            "table_context": str(meta.get("table_context") or ""),
                        }
                    )
            except Exception as exc:
                logger.debug("tokens.json 读取跳过: %s", exc)

        if not chunks:
            queries = [
                f"{entity_name} 主要会计数据和财务指标",
                f"{entity_name} 主要会计数据及财务指标",
                f"{entity_name} 主要财务指标",
                f"{entity_name} 合并资产负债表",
                f"{entity_name} 合并利润表",
            ]
            for query in queries:
                results = self.engine.retrieve(query, top_k=8, entity_id=entity_id)
                for item in results:
                    text = str(item.get("text") or "").strip()
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    meta = item.get("metadata") or {}
                    chunks.append(
                        {
                            "text": text,
                            "section": str(meta.get("section") or ""),
                            "page": int(meta.get("page") or 0),
                            "table_context": str(meta.get("table_context") or ""),
                        }
                    )

        chunks.sort(
            key=lambda c: (
                PRIORITY_SECTIONS.index(c["section"])
                if c["section"] in PRIORITY_SECTIONS
                else 99,
                -c["page"],
            )
        )
        return chunks

    def _extract_metric(
        self,
        metric_key: str,
        definition: dict,
        chunks: list[dict[str, Any]],
        *,
        target_year: int,
    ) -> ExtractedMetric | None:
        aliases: tuple[str, ...] = definition["aliases"]
        preferred_sections: tuple[str, ...] = definition["sections"]
        exclude: tuple[str, ...] = definition.get("exclude", ())

        ordered = sorted(
            chunks,
            key=lambda c: (
                0 if c["section"] in preferred_sections else 1,
                PRIORITY_SECTIONS.index(c["section"])
                if c["section"] in PRIORITY_SECTIONS
                else 99,
            ),
        )

        best: ExtractedMetric | None = None
        for chunk in ordered:
            text = chunk["text"]
            unit_context = detect_unit_context(text)
            if metric_key in {"eps", "bvps"}:
                unit_context = UnitKind.YUAN_PER_SHARE
            elif metric_key == "roe":
                unit_context = UnitKind.PERCENT

            raw_value = self._find_row_value(text, aliases, exclude, metric_key)
            if raw_value is None:
                continue

            normalized = normalize_value(
                raw_value, metric_kind=metric_key, unit_context=unit_context
            )
            if normalized is None:
                continue

            confidence = self._score_confidence(
                metric_key, normalized.value, chunk["section"], preferred_sections
            )
            if confidence < 0.5:
                continue

            candidate = ExtractedMetric(
                metric=metric_key,
                value=normalized.value,
                unit=normalized.unit,
                year=target_year,
                source_page=chunk["page"],
                confidence=confidence,
                display=normalized.display,
                raw_value=normalized.raw,
                source_section=chunk["section"],
                table_context=chunk.get("table_context") or "",
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate
        return best

    def _find_row_value(
        self,
        text: str,
        aliases: tuple[str, ...],
        exclude: tuple[str, ...],
        metric_key: str,
    ) -> str | None:
        # 每股类指标优先在「每股/人民币元」子段落中查找
        search_text = text
        if metric_key in {"eps", "bvps", "roe"}:
            for marker in ("每股", "主要财务指标", "(人民币元)", "（人民币元）"):
                pos = text.find(marker)
                if pos >= 0:
                    search_text = text[pos:]
                    break

        lines = [line.strip() for line in search_text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if self._is_header_line(line):
                continue
            if not any(alias in line for alias in aliases):
                continue
            if any(ex in line for ex in exclude):
                continue
            inline = self._pick_value_from_line(line, aliases, metric_key)
            if inline:
                return inline
            following = self._pick_following_values(lines, index + 1, metric_key)
            if following:
                return following
        return None

    @staticmethod
    def _is_header_line(line: str) -> bool:
        if re.search(r"20\d{2}\s*年", line) and len(line) < 20:
            return True
        if re.fullmatch(r"1-6月|1–6月|7-9月|10-12月|本期比.*?变动", line):
            return True
        if "同比" in line and "变动" in line and len(line) < 30:
            return True
        return False

    def _pick_value_from_line(
        self, line: str, aliases: tuple[str, ...], metric_key: str
    ) -> str | None:
        for alias in aliases:
            pos = line.find(alias)
            if pos < 0:
                continue
            tail = line[pos + len(alias) :]
            picked = self._select_best_number(_NUMBER_PATTERN.findall(tail), metric_key)
            if picked:
                return picked
        return None

    def _pick_following_values(
        self, lines: list[str], start: int, metric_key: str
    ) -> str | None:
        collected: list[str] = []
        label_keywords = (
            "营业收入",
            "营业总收入",
            "净利润",
            "利润总额",
            "每股收益",
            "净资产",
            "现金流",
            "股东权益",
            "资产总计",
            "负债",
            "营业利润",
            "基本每股",
            "稀释每股",
            "加权平均",
            "主要财务",
            "注:",
            "注：",
        )
        for line in lines[start : start + 6]:
            stripped = line.strip()
            if not stripped:
                continue
            # 遇到下一行指标标签则停止
            if collected and any(kw in stripped for kw in label_keywords):
                if not _NUMBER_PATTERN.fullmatch(stripped.replace(",", "")):
                    break
            nums = _NUMBER_PATTERN.findall(stripped)
            if not nums:
                continue
            collected.extend(nums)
            picked = self._select_best_number(collected, metric_key)
            if picked:
                return picked
        return self._select_best_number(collected, metric_key)

    def _select_best_number(self, numbers: list[str], metric_key: str) -> str | None:
        parsed: list[tuple[str, float]] = []
        for raw in numbers:
            val = parse_number(raw)
            if val is None or is_year_like(val) or is_likely_date_fragment(raw, val):
                continue
            parsed.append((raw, val))
        if not parsed:
            return None

        if metric_key == "roe":
            for raw, val in parsed:
                if 0 < val <= 100:
                    return raw
            return None

        if metric_key in {"eps", "bvps"}:
            decimals = [(raw, val) for raw, val in parsed if "." in raw and 0 < val <= 500]
            if decimals:
                return decimals[0][0]
            for raw, val in parsed:
                if 0 < val <= 500:
                    return raw
            return None

        amount_keys = {
            "revenue",
            "net_profit",
            "attributable_profit",
            "total_assets",
            "total_equity",
            "cash_flow_operating",
        }
        if metric_key in amount_keys:
            for raw, val in parsed:
                if val >= 100:
                    return raw
            return None
        if metric_key == "operating_profit":
            for raw, val in parsed:
                if val >= 10:
                    return raw
            return None
        return parsed[0][0]

    def _score_confidence(
        self,
        metric_key: str,
        value: float,
        section: str,
        preferred_sections: tuple[str, ...],
    ) -> float:
        score = 0.7
        if section in preferred_sections:
            score += 0.2
        if section == "主要会计数据":
            score += 0.05
        if metric_key == "roe" and (value <= 0 or value > 100):
            return 0.3
        if metric_key in {"eps", "bvps"} and (value <= 0 or value > 500):
            return 0.3
        return min(score, 0.98)

    def _compute_bvps(
        self, chunks: list[dict[str, Any]], target_year: int
    ) -> ExtractedMetric | None:
        balance_text = self._merge_section_text(chunks, "资产负债表") or self._merge_section_text(
            chunks, "主要会计数据"
        )
        equity = None
        for pattern in (
            r"归属于母公司(?:所有者)?(?:股东)?权益[\s\S]{0,40}?合计[\s\n|]*([\d,\.]+)",
            r"归属于母公司(?:股东|所有者)权益[\s\n|]*([\d,\.]+)",
        ):
            match = re.search(pattern, balance_text)
            if match:
                equity = parse_number(match.group(1))
                if equity:
                    break
        shares = None
        for pattern in (
            r"实收资本\(或股本\)[\s\n|]*[\d,\.]+[\s\n|]*([\d,\.]+)",
            r"股本[\s\n|]*([\d,\.]+)",
        ):
            match = re.search(pattern, balance_text, re.M)
            if match:
                shares = parse_number(match.group(1))
                if shares and shares > 1e6:
                    break
        if not equity or not shares:
            return None
        unit_context = detect_unit_context(balance_text)
        equity_norm = normalize_value(
            str(equity), metric_kind="total_equity", unit_context=unit_context
        )
        if equity_norm is None:
            return None
        equity_yuan = equity_norm.value * 1e8
        bvps = round(equity_yuan / shares, 2)
        if bvps <= 0 or bvps > 500:
            return None
        page = next((c["page"] for c in chunks if c["section"] == "资产负债表"), 0)
        return ExtractedMetric(
            metric="bvps",
            value=bvps,
            unit="元/股",
            year=target_year,
            source_page=page,
            confidence=0.82,
            display=f"{bvps:.2f} 元/股",
            raw_value=str(bvps),
            source_section="资产负债表",
            table_context="computed",
        )

    @staticmethod
    def _merge_section_text(chunks: list[dict[str, Any]], section: str) -> str:
        return "\n".join(c["text"] for c in chunks if c["section"] == section)

    @staticmethod
    def _extract_yoy(text: str, label: str) -> float | None:
        pattern = rf"{label}[\s\S]{{0,120}}?[\d,\.]+[\s\n]+[\d,\.]+[\s\n]+(\-?\d+\.?\d*)"
        match = re.search(pattern, text)
        return parse_number(match.group(1)) if match else None
