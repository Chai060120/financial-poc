"""日报 Agent 步骤协议。"""

from __future__ import annotations

from typing import Protocol

from src.agent.daily.types import DailyContext, StepResult


class DailyStep(Protocol):
    """可插拔日报流水线步骤。"""

    name: str

    def run(self, ctx: DailyContext) -> StepResult:
        """执行步骤并返回结果。"""
