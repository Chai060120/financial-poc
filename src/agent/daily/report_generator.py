"""Markdown 研究报告生成器。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DAILY_REPORT_DIR, DEFAULT_ENCODING, setup_logging
from src.agent.daily.types import DailyAnalysis, DailyContext, EventItem

logger = setup_logging(__name__)


def _format_events(title: str, events: list[EventItem]) -> str:
    if not events:
        return f"## {title}\n\n暂无相关条目。\n"

    lines = [f"## {title}", ""]
    for index, item in enumerate(events, start=1):
        lines.append(f"### {index}. {item.title}")
        if item.entity_name:
            lines.append(f"- **公司**: {item.entity_name}")
        if item.publish_time:
            lines.append(f"- **时间**: {item.publish_time}")
        if item.source:
            lines.append(f"- **来源**: {item.source}")
        if item.url:
            lines.append(f"- **链接**: {item.url}")
        lines.append(f"- **摘要**: {item.summary}")
        lines.append("")
    return "\n".join(lines)


def _format_pipeline_stats(ctx: DailyContext) -> str:
    lines = ["## 流水线统计", ""]
    for step in ctx.step_results:
        status = "OK" if step.success else "FAIL"
        lines.append(f"- **{step.name}** [{status}] {step.message or ''} ({step.duration_ms}ms)")
        if step.data:
            compact = json.dumps(step.data, ensure_ascii=False)
            if len(compact) > 200:
                compact = compact[:200] + "..."
            lines.append(f"  - 数据: `{compact}`")
    lines.append("")
    return "\n".join(lines)


def generate_markdown_report(ctx: DailyContext) -> str:
    """生成 Markdown 研究报告正文。"""
    analysis = ctx.analysis
    sections = [
        f"# 金融信息日报 · {ctx.report_date}",
        "",
        f"> 自动生成 | LLM={'是' if analysis.llm_used else '否（规则引擎）'}",
        "",
        "## 市场摘要",
        "",
        analysis.summary or "（暂无摘要）",
        "",
        _format_events("重大事件", analysis.major_events),
        _format_events("利好", analysis.bullish),
        _format_events("利空", analysis.bearish),
        _format_events("风险提示", analysis.risks),
        _format_events("业绩变化", analysis.performance_changes),
        _format_pipeline_stats(ctx),
    ]

    if ctx.errors:
        sections.extend(["## 执行告警", ""])
        for error in ctx.errors:
            sections.append(f"- {error}")
        sections.append("")

    return "\n".join(sections).strip() + "\n"


def save_daily_report(ctx: DailyContext, *, output_dir: Path | None = None) -> Path:
    """写入 Markdown 文件并更新 ctx.report_path。"""
    target_dir = output_dir or DAILY_REPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    report_path = target_dir / f"{ctx.report_date}.md"
    content = generate_markdown_report(ctx)
    report_path.write_text(content, encoding=DEFAULT_ENCODING)

    ctx.report_path = report_path
    logger.info("日报已保存: %s", report_path)
    return report_path
