"""
Financial Research Agent 的统一 Prompt 模块。

仅供可选 LLM 增强使用；无 API Key 时规则分析仍可独立运行。
"""

from __future__ import annotations

from src.agent.prompts.financial_prompt import (
    FINANCIAL_ANALYSIS_INSTRUCTION,
    build_financial_analysis_prompt,
)
from src.agent.prompts.news_prompt import (
    NEWS_ANALYSIS_INSTRUCTION,
    build_daily_summary_prompt,
    build_news_analysis_prompt,
)
from src.agent.prompts.report_prompt import (
    REPORT_INSTRUCTION,
    build_final_report_prompt,
)
from src.agent.prompts.system_prompt import (
    FINANCIAL_LLM_TEMPERATURE,
    RAG_QA_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from src.agent.prompts.valuation_prompt import (
    VALUATION_ANALYSIS_INSTRUCTION,
    build_valuation_analysis_prompt,
)

__all__ = [
    "SYSTEM_PROMPT",
    "RAG_QA_SYSTEM_PROMPT",
    "FINANCIAL_LLM_TEMPERATURE",
    "FINANCIAL_ANALYSIS_INSTRUCTION",
    "VALUATION_ANALYSIS_INSTRUCTION",
    "NEWS_ANALYSIS_INSTRUCTION",
    "REPORT_INSTRUCTION",
    "build_financial_analysis_prompt",
    "build_valuation_analysis_prompt",
    "build_news_analysis_prompt",
    "build_daily_summary_prompt",
    "build_final_report_prompt",
]
