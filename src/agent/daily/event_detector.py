"""
事件检测：从新闻中识别重大事件、利好、利空、风险、业绩变化。

规则引擎为主，可扩展为 LLM 增强。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import setup_logging
from src.agent.daily.types import DailyAnalysis, EventItem
from src.collectors.news_collector import infer_entity

logger = setup_logging(__name__)

_RULES: tuple[tuple[str, str, float, re.Pattern[str]], ...] = (
    ("重大事件", "重大事件", 1.0, re.compile(r"并购|重组|收购|停牌|立案|调查|政策|央行|证监会|国务院|重大合同")),
    ("利好", "利好", 0.9, re.compile(r"超预期|增持|回购|分红|买入评级|上调评级|涨停|大涨|创新高|盈利增长|净利润增")),
    ("利空", "利空", 0.9, re.compile(r"减持|下调评级|卖出评级|跌停|大跌|亏损|预亏|暴雷|立案|处罚|违规")),
    ("风险", "风险", 0.85, re.compile(r"风险|诉讼|警示|ST\b|退市|债务|违约|流动性|监管")),
    (
        "业绩变化",
        "业绩变化",
        0.8,
        re.compile(r"业绩|净利润|营收|季报|年报|半年报|预增|预减|同比|环比|EPS|ROE"),
    ),
)


def _build_event(record: dict[str, Any], category: str, sentiment: str, score: float) -> EventItem:
    title = str(record.get("title") or "").strip()
    content = str(record.get("content") or record.get("body") or "")
    entity = str(record.get("entity_name") or record.get("entity") or "").strip()
    if not entity:
        entity = infer_entity(title, content)

    summary = content[:200].strip() if content else title
    return EventItem(
        title=title,
        entity_name=entity,
        category=category,
        sentiment=sentiment,  # type: ignore[arg-type]
        publish_time=str(record.get("publish_time") or record.get("date") or ""),
        url=str(record.get("url") or ""),
        summary=summary,
        score=score,
        source=str(record.get("source") or record.get("news_source") or ""),
    )


def detect_events(records: list[dict[str, Any]], *, max_per_category: int = 10) -> DailyAnalysis:
    """从新闻列表中检测并分类事件。"""
    analysis = DailyAnalysis()
    buckets: dict[str, list[EventItem]] = {
        "major_events": [],
        "bullish": [],
        "bearish": [],
        "risks": [],
        "performance_changes": [],
    }

    for record in records:
        title = str(record.get("title") or "")
        content = str(record.get("content") or record.get("body") or "")
        text = f"{title} {content}"
        if not text.strip():
            continue

        matched_categories: list[tuple[str, str, float]] = []
        for category, sentiment, score, pattern in _RULES:
            if pattern.search(text):
                matched_categories.append((category, sentiment, score))

        if not matched_categories:
            continue

        best = max(matched_categories, key=lambda item: item[2])
        event = _build_event(record, best[0], best[1], best[2])

        if best[0] == "重大事件":
            buckets["major_events"].append(event)
        elif best[1] == "利好":
            buckets["bullish"].append(event)
        elif best[1] == "利空":
            buckets["bearish"].append(event)
        elif best[1] == "风险":
            buckets["risks"].append(event)
        elif best[0] == "业绩变化":
            buckets["performance_changes"].append(event)

    for key in buckets:
        buckets[key].sort(key=lambda item: item.score, reverse=True)
        buckets[key] = buckets[key][:max_per_category]

    analysis.major_events = buckets["major_events"]
    analysis.bullish = buckets["bullish"]
    analysis.bearish = buckets["bearish"]
    analysis.risks = buckets["risks"]
    analysis.performance_changes = buckets["performance_changes"]

    logger.info(
        "事件检测完成: 重大=%d 利好=%d 利空=%d 风险=%d 业绩=%d",
        len(analysis.major_events),
        len(analysis.bullish),
        len(analysis.bearish),
        len(analysis.risks),
        len(analysis.performance_changes),
    )
    return analysis
