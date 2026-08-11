"""
用 Cursor SDK 将五段式分析报告改写成自然语言解读。

未配置 CURSOR_API_KEY / 包不可用 / 调用失败时返回空字符串，不中断主分析。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CURSOR_API_KEY,
    CURSOR_MODEL,
    CURSOR_NARRATIVE_ENABLED,
    PROJECT_ROOT,
    setup_logging,
)

logger = setup_logging(__name__)

_NARRATIVE_PROMPT = """你是金融分析助手。下面是一份已由规则引擎生成的财报分析报告。
请基于报告中的数字与结论，用中文写 4～8 段自然语言解读。

硬性要求：
1. 只输出解读正文，不要修改任何文件、不要执行工具、不要调用 shell。
2. 不要编造报告中未出现的数字或事实。
3. 覆盖：估值结论、核心财务指标、PE/PB 含义、横向对比要点、数据质量/风险提示。
4. 语气专业、简洁，面向非技术读者。
5. 结尾用一句话注明：本解读基于 PoC 规则结果，不构成投资建议。

公司：{entity_name}

===== 分析报告开始 =====
{report_text}
===== 分析报告结束 =====
"""


def cursor_narrative_available() -> bool:
    """是否具备调用 Cursor 解读的条件。"""
    return bool(CURSOR_NARRATIVE_ENABLED and CURSOR_API_KEY)


def narrate_report_card(report_text: str, entity_name: str = "") -> str:
    """
    调用 Cursor Agent 生成自然语言解读。

    Returns:
        解读文本；不可用或失败时返回空字符串。
    """
    if not report_text.strip():
        return ""
    if not CURSOR_NARRATIVE_ENABLED:
        return ""
    if not CURSOR_API_KEY:
        logger.debug("未配置 CURSOR_API_KEY，跳过自然语言解读")
        return ""

    try:
        from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
    except ImportError:
        logger.warning("未安装 cursor-sdk，跳过自然语言解读。请运行: pip install cursor-sdk")
        return ""

    prompt = _NARRATIVE_PROMPT.format(
        entity_name=entity_name or "目标公司",
        report_text=report_text.strip(),
    )

    try:
        result = Agent.prompt(
            prompt,
            AgentOptions(
                api_key=CURSOR_API_KEY,
                model=CURSOR_MODEL or "composer-2.5",
                local=LocalAgentOptions(cwd=str(PROJECT_ROOT)),
                # 解读场景禁止改仓库 / 跑命令
                disallowed_tools=(
                    "Shell",
                    "Write",
                    "StrReplace",
                    "Delete",
                    "EditNotebook",
                    "ApplyPatch",
                ),
            ),
        )
    except Exception as exc:
        logger.warning("Cursor 自然语言解读失败（启动）: %s", exc)
        return ""

    status = getattr(result, "status", None)
    if status == "error":
        logger.warning("Cursor 自然语言解读失败（运行）: id=%s", getattr(result, "id", ""))
        return ""

    text = _extract_result_text(result)
    if not text:
        logger.warning("Cursor 自然语言解读返回空文本")
        return ""
    return text.strip()


def append_cursor_narrative(report_text: str, entity_name: str = "") -> str:
    """在五段报告后追加【6】自然语言解读（有结果时）。"""
    narrative = narrate_report_card(report_text, entity_name)
    if not narrative:
        return report_text

    section = (
        "\n\n【6】自然语言解读（Cursor）\n"
        + "\n".join(f"  {line}" if line.strip() else "" for line in narrative.splitlines())
        + "\n"
    )
    # 插在免责声明分隔线之前（若存在）
    marker = "──────────────────────────────────────────────────────────"
    idx = report_text.rfind(marker)
    if idx >= 0:
        return report_text[:idx].rstrip() + section + "\n" + report_text[idx:]
    return report_text.rstrip() + section


def answer_followup_question(
    question: str,
    *,
    entity_name: str,
    context: str,
) -> str:
    """用 Cursor 回答多轮追问；不可用时返回空字符串。"""
    if not question.strip() or not context.strip():
        return ""
    if not CURSOR_NARRATIVE_ENABLED or not CURSOR_API_KEY:
        return ""

    try:
        from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
    except ImportError:
        return ""

    prompt = (
        "你是金融分析助手。请仅基于下列上下文回答用户追问。\n"
        "要求：只输出中文回答；不要修改文件；不要编造上下文没有的数字。\n\n"
        f"公司：{entity_name or '目标公司'}\n"
        f"用户问题：{question.strip()}\n\n"
        f"===== 上下文 =====\n{context.strip()}\n===== 结束 =====\n"
    )
    try:
        result = Agent.prompt(
            prompt,
            AgentOptions(
                api_key=CURSOR_API_KEY,
                model=CURSOR_MODEL or "composer-2.5",
                local=LocalAgentOptions(cwd=str(PROJECT_ROOT)),
                disallowed_tools=(
                    "Shell",
                    "Write",
                    "StrReplace",
                    "Delete",
                    "EditNotebook",
                    "ApplyPatch",
                ),
            ),
        )
    except Exception as exc:
        logger.warning("Cursor 追问回答失败: %s", exc)
        return ""

    if getattr(result, "status", None) == "error":
        return ""
    return _extract_result_text(result).strip()


def _extract_result_text(result: object) -> str:
    """兼容不同 SDK 返回字段。"""
    for attr in ("result", "text", "output", "message"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if callable(value):
            try:
                called = value()
                if isinstance(called, str) and called.strip():
                    return called.strip()
            except Exception:
                pass

    if isinstance(result, dict):
        for key in ("result", "text", "output"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    # 部分 SDK 把最终文本放在 status/result 组合对象里
    nested = getattr(result, "result", None)
    if nested is not None and nested is not result:
        return _extract_result_text(nested)
    return ""
