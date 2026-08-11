"""
财务分析 Prompt：基于已提供指标与财报文本做结构化研判。
"""

from __future__ import annotations

from typing import Any


FINANCIAL_ANALYSIS_INSTRUCTION = """任务：分析上市公司财务情况。

请严格基于下方「财务指标」与「财报文本」作答。
不要创造未提供的指标；缺失时写「未提供相关数据」。

分析维度：
1. 营收变化
2. 净利润变化
3. 盈利能力
4. ROE 情况
5. 现金流情况
6. 成长性

请只输出 JSON（不要 Markdown 代码块），格式如下：
{
  "financial_summary": "",
  "growth": "",
  "profitability": "",
  "cash_flow": "",
  "financial_score": 0
}

其中 financial_score 为 0-100 的整数，表示财务质量综合评分（仅基于已给信息）。
"""


def build_financial_analysis_prompt(
    *,
    company: str,
    financial_metrics: str | dict[str, Any],
    financial_documents: str,
) -> str:
    """构造财务分析 user prompt。"""
    metrics_text = (
        financial_metrics
        if isinstance(financial_metrics, str)
        else _format_mapping(financial_metrics)
    )
    return (
        f"{FINANCIAL_ANALYSIS_INSTRUCTION}\n\n"
        f"公司名称：\n{company or '未提供相关数据'}\n\n"
        f"财务指标：\n{metrics_text.strip() or '未提供相关数据'}\n\n"
        f"财报文本：\n{(financial_documents or '').strip() or '未提供相关数据'}\n"
    )


def _format_mapping(data: dict[str, Any]) -> str:
    if not data:
        return "未提供相关数据"
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            display = value.get("display") or value.get("raw") or value.get("value")
            lines.append(f"- {key}: {display}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)
