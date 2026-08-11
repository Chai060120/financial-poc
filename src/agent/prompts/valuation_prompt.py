"""
估值分析 Prompt：基于已有 PE/PB 等指标判断估值水平。
"""

from __future__ import annotations

from typing import Any


VALUATION_ANALYSIS_INSTRUCTION = """任务：根据已有估值指标判断股票估值水平。

请严格基于下方数据作答，禁止编造 PE/PB 或行业数据。
缺失字段请写「未提供相关数据」。
必须解释判断依据。

请考虑：
1. 当前 PE 水平
2. 当前 PB 水平
3. 历史估值位置（若未提供则说明未提供）
4. 行业比较
5. 公司成长能力

请只输出 JSON（不要 Markdown 代码块），格式如下：
{
  "valuation_level": "低估/合理/高估",
  "valuation_reason": "",
  "valuation_score": 0
}

其中 valuation_score 为 0-100 的整数（分数越高表示估值吸引力越高 / 越偏向低估）。
valuation_level 只能是：低估、合理、高估 三者之一。
"""


def build_valuation_analysis_prompt(
    *,
    company: str,
    pe: Any = None,
    pb: Any = None,
    industry_data: str | dict[str, Any] = "",
    growth_data: str | dict[str, Any] = "",
) -> str:
    """构造估值分析 user prompt。"""
    industry_text = (
        industry_data
        if isinstance(industry_data, str)
        else _format_mapping(industry_data)
    )
    growth_text = (
        growth_data if isinstance(growth_data, str) else _format_mapping(growth_data)
    )
    pe_text = "未提供相关数据" if pe is None or pe == "" else str(pe)
    pb_text = "未提供相关数据" if pb is None or pb == "" else str(pb)
    return (
        f"{VALUATION_ANALYSIS_INSTRUCTION}\n\n"
        f"股票：\n{company or '未提供相关数据'}\n\n"
        f"估值数据：\nPE: {pe_text}\nPB: {pb_text}\n\n"
        f"行业数据：\n{industry_text.strip() or '未提供相关数据'}\n\n"
        f"成长数据：\n{growth_text.strip() or '未提供相关数据'}\n"
    )


def _format_mapping(data: dict[str, Any]) -> str:
    if not data:
        return "未提供相关数据"
    lines: list[str] = []
    for key, value in data.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)
