"""
最终投研报告 Prompt：汇总财务/估值/新闻/RAG 上下文。
"""

from __future__ import annotations

from typing import Any


REPORT_INSTRUCTION = """请基于下方已整理好的分析结果与检索材料，生成五段式投资研究报告。

硬性要求：
1. 只能使用提供的信息，禁止编造。
2. 缺失处写「未提供相关数据」。
3. 不给出绝对买入/卖出建议，不预测确定收益。
4. 估值判断必须引用已有估值分析中的依据。
5. 使用中文 Markdown 输出，不要包在代码块中。

输出格式必须严格如下：

# 公司分析报告

## 1. 公司概况

介绍公司业务和行业位置。

## 2. 财务表现

分析：
- 营收
- 利润
- 盈利能力
- 现金流

## 3. 估值分析

输出：
估值判断：低估 / 合理 / 高估

说明：
为什么。

## 4. 市场信息

分析：
新闻事件
市场情绪

## 5. 风险因素

## 综合评分

总分：
xx/100

最终判断：
低估 / 合理 / 高估
"""


def build_final_report_prompt(
    *,
    company_info: str | dict[str, Any],
    financial_analysis: str | dict[str, Any],
    valuation_analysis: str | dict[str, Any],
    news_analysis: str | dict[str, Any],
    context: str,
) -> str:
    """构造最终报告 user prompt。"""
    return (
        f"{REPORT_INSTRUCTION}\n\n"
        f"公司信息：\n{_as_text(company_info)}\n\n"
        f"财务分析：\n{_as_text(financial_analysis)}\n\n"
        f"估值分析：\n{_as_text(valuation_analysis)}\n\n"
        f"新闻分析：\n{_as_text(news_analysis)}\n\n"
        f"RAG检索结果：\n{(context or '').strip() or '未提供相关数据'}\n"
    )


def _as_text(value: str | dict[str, Any]) -> str:
    if isinstance(value, str):
        return value.strip() or "未提供相关数据"
    if not value:
        return "未提供相关数据"
    try:
        import json

        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)
