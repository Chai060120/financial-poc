"""
对话记忆（兼容层）：委托给 ConversationHistory 实现。

保留 ConversationMemory / ChatTurn 名称，供 Planner 与 Agent Workflow 继续使用。
"""

from __future__ import annotations

from src.chat.history import (
    ConversationHistory,
    HistoryTurn,
    estimate_tokens,
)

ChatTurn = HistoryTurn


class ConversationMemory(ConversationHistory):
    """ConversationHistory 的兼容别名。"""


__all__ = [
    "ChatTurn",
    "ConversationMemory",
    "HistoryTurn",
    "ConversationHistory",
    "estimate_tokens",
]
