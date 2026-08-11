"""
新闻与市场情绪分析 Prompt。
"""

from __future__ import annotations

from typing import Any, Iterable


NEWS_ANALYSIS_INSTRUCTION = """任务：分析新闻和市场情绪。

请严格基于下方新闻内容，禁止编造未出现的事件。
若新闻为空，sentiment 设为 neutral，并说明「未提供相关数据」。

要求：
- 区分短期影响与长期影响
- important_events 只列出材料中明确出现的事件
- sentiment 只能是：positive / neutral / negative
- sentiment_score 取值范围 -1 到 1（负数偏空，正数偏多）

请只输出 JSON（不要 Markdown 代码块），格式如下：
{
  "sentiment": "positive/neutral/negative",
  "sentiment_score": 0.0,
  "important_events": [],
  "impact_analysis": ""
}

impact_analysis 中请分别简述：短期影响、长期影响。
"""


def build_news_analysis_prompt(*, news: str | Iterable[Any]) -> str:
    """构造新闻分析 user prompt。"""
    if isinstance(news, str):
        news_text = news.strip()
    else:
        lines: list[str] = []
        for idx, item in enumerate(news, start=1):
            if isinstance(item, dict):
                title = str(item.get("title") or item.get("headline") or "").strip()
                source = str(item.get("source") or "").strip()
                time_str = str(item.get("publish_time") or item.get("time") or "").strip()
                parts = [p for p in (title, source, time_str) if p]
                if parts:
                    lines.append(f"{idx}. " + " | ".join(parts))
            else:
                title = str(getattr(item, "title", "") or "").strip()
                source = str(getattr(item, "source", "") or "").strip()
                time_str = str(getattr(item, "publish_time", "") or "").strip()
                if title or source or time_str:
                    parts = [p for p in (title, source, time_str) if p]
                    lines.append(f"{idx}. " + " | ".join(parts))
                else:
                    text = str(item).strip()
                    if text:
                        lines.append(f"{idx}. {text}")
        news_text = "\n".join(lines)

    return (
        f"{NEWS_ANALYSIS_INSTRUCTION}\n\n"
        f"新闻：\n{news_text or '未提供相关数据'}\n"
    )


def build_daily_summary_prompt(
    *,
    report_date: str,
    event_lines: list[str],
    context: str,
) -> str:
    """构造日报市场摘要 user prompt（可选 LLM 增强）。"""
    events = "\n".join(event_lines) if event_lines else "- 无"
    return (
        f"报告日期: {report_date}\n\n"
        f"【今日新闻事件】\n{events}\n\n"
        f"【参考资料】\n{(context or '').strip() or '（无向量检索结果）'}\n\n"
        "请严格基于上述材料输出 JSON（不要 Markdown 代码块）：\n"
        '{"summary": "200字以内市场摘要", "highlights": ["要点1", "要点2"]}\n'
        "禁止编造未出现的事件；缺失时写「未提供相关数据」。"
    )
