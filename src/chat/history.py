"""
对话历史：保存最近 N 轮问答，自动拼接 Prompt，并限制 Token 用量。

供连续聊天使用，支持 reset / history 命令。
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import CHAT_HISTORY_MAX_TOKENS, CHAT_MEMORY_MAX_TURNS, setup_logging

logger = setup_logging(__name__)

_FOLLOW_UP_MARKERS = (
    "呢",
    "吗",
    "那",
    "还",
    "另外",
    "其次",
    "同样",
    "同比",
    "环比",
    "如何",
    "怎么样",
    "多少",
    "哪些",
)

_HISTORY_COMMANDS = frozenset(
    {
        "history",
        "/history",
        "/memory",
        "memory",
    }
)
_RESET_COMMANDS = frozenset(
    {
        "reset",
        "/reset",
        "/clear",
        "clear",
    }
)
_EXIT_COMMANDS = frozenset({"/exit", "/quit", "exit", "quit"})
_HELP_COMMANDS = frozenset({"/help", "help"})


@dataclass(frozen=True)
class HistoryTurn:
    """单轮对话。"""

    question: str
    answer: str


def estimate_tokens(text: str) -> int:
    """
    估算文本 Token 数（无需 tiktoken 依赖）。

    中文为主时按约 1.5 字符/token 估算。
    """
    content = text.strip()
    if not content:
        return 0
    return max(1, int(len(content) / 1.5))


def _looks_like_follow_up(question: str) -> bool:
    text = question.strip()
    if not text:
        return False
    if len(text) <= 15:
        return True
    if len(text) >= 30:
        return False
    return any(marker in text for marker in _FOLLOW_UP_MARKERS)


class ConversationHistory:
    """
    连续对话历史管理。

    - 保留最近 max_turns 轮（默认 10）
    - format_history / build_prompt_context 自动裁剪以不超过 max_tokens
    """

    def __init__(
        self,
        max_turns: int = CHAT_MEMORY_MAX_TURNS,
        max_tokens: int = CHAT_HISTORY_MAX_TOKENS,
    ) -> None:
        if max_turns <= 0:
            raise ValueError(f"max_turns 必须大于 0，当前为 {max_turns}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens 必须大于 0，当前为 {max_tokens}")

        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._turns: deque[HistoryTurn] = deque(maxlen=max_turns)

    def __len__(self) -> int:
        return len(self._turns)

    def add(self, question: str, answer: str) -> None:
        """追加一轮对话；超出 max_turns 时丢弃最早一轮。"""
        q = question.strip()
        a = answer.strip()
        if not q:
            raise ValueError("question 不能为空")
        if not a:
            raise ValueError("answer 不能为空")

        self._turns.append(HistoryTurn(question=q, answer=a))
        logger.info(
            "History 已保存第 %d/%d 轮 | 估算 tokens=%d/%d",
            len(self._turns),
            self.max_turns,
            self.history_token_count(),
            self.max_tokens,
        )

    def reset(self) -> None:
        """重置对话历史（同 clear）。"""
        count = len(self._turns)
        self._turns.clear()
        logger.info("History 已 reset (%d 轮)", count)

    def clear(self) -> None:
        """清空对话历史（reset 别名）。"""
        self.reset()

    def turns(self) -> list[HistoryTurn]:
        return list(self._turns)

    def history_token_count(self, turns: list[HistoryTurn] | None = None) -> int:
        selected = turns if turns is not None else list(self._turns)
        total = 0
        for turn in selected:
            total += estimate_tokens(turn.question)
            total += estimate_tokens(turn.answer)
            total += 12
        return total

    def _select_turns_for_prompt(self, max_tokens: int | None = None) -> list[HistoryTurn]:
        """按 Token 预算从最近轮次向前选取历史。"""
        budget = max_tokens if max_tokens is not None else self.max_tokens
        if budget <= 0 or not self._turns:
            return []

        selected: list[HistoryTurn] = []
        used = 0
        for turn in reversed(self._turns):
            turn_tokens = estimate_tokens(turn.question) + estimate_tokens(turn.answer) + 12
            if selected and used + turn_tokens > budget:
                break
            selected.append(turn)
            used += turn_tokens

        selected.reverse()
        return selected

    def format_history(self, *, max_tokens: int | None = None) -> str:
        """格式化为 Prompt 可用的对话历史文本（受 Token 限制）。"""
        selected = self._select_turns_for_prompt(max_tokens=max_tokens)
        if not selected:
            return ""

        blocks: list[str] = []
        start_index = len(self._turns) - len(selected) + 1
        for offset, turn in enumerate(selected):
            index = start_index + offset
            blocks.append(f"第{index}轮")
            blocks.append(f"用户: {turn.question}")
            blocks.append(f"助手: {turn.answer}")
            blocks.append("")

        return "\n".join(blocks).strip()

    def build_prompt_context(
        self,
        question: str,
        *,
        references: str = "",
        plan_block: str = "",
        max_tokens: int | None = None,
    ) -> str:
        """
        自动拼接完整 User Prompt：计划 + 历史 + 参考资料 + 当前问题。
        """
        parts: list[str] = []

        if plan_block.strip():
            parts.append(plan_block.strip())

        history = self.format_history(max_tokens=max_tokens)
        if history:
            parts.append(
                "【对话历史】\n"
                f"{history}\n\n"
                "请结合对话历史理解当前问题的指代。"
            )

        if references.strip():
            parts.append(f"【参考资料】\n{references.strip()}")

        parts.append(f"【用户问题】\n{question.strip()}")
        parts.append("【回答要求】\n请用中文回答，引用资料时标注序号（如 [1]）。")
        return "\n".join(parts)

    def view(self, *, max_answer_chars: int = 300) -> str:
        """供 history 命令展示的可读摘要。"""
        if not self._turns:
            return "（对话历史为空）"

        selected = self._select_turns_for_prompt()
        lines = [
            f"对话历史: {len(self._turns)}/{self.max_turns} 轮",
            f"Prompt 历史预算: {self.history_token_count(selected)}/{self.max_tokens} tokens",
            "",
        ]

        start_index = len(self._turns) - len(selected) + 1
        for offset, turn in enumerate(selected):
            index = start_index + offset
            answer = turn.answer
            if len(answer) > max_answer_chars:
                answer = answer[:max_answer_chars] + "..."
            lines.append(f"--- 第 {index} 轮 ---")
            lines.append(f"用户: {turn.question}")
            lines.append(f"助手: {answer}")
            lines.append("")

        if len(selected) < len(self._turns):
            omitted = len(self._turns) - len(selected)
            lines.append(f"（另有 {omitted} 轮较早历史未纳入 Prompt，受 Token 限制）")

        return "\n".join(lines).strip()

    def build_retrieval_query(self, question: str) -> str:
        """基于对话历史扩展检索 query，便于理解省略主语的追问。"""
        current = question.strip()
        if not current or not self._turns:
            return current

        if not _looks_like_follow_up(current):
            return current

        previous = self._turns[-1].question
        expanded = f"{previous} {current}"
        logger.info("History 扩展检索 query: %r -> %r", current, expanded)
        return expanded

    def handle_command(self, text: str) -> tuple[bool, str | None]:
        """
        处理内置命令。

        Returns:
            (handled, message)
            - handled=True 表示已消费输入，不应继续当作问题处理
            - message 为需要打印给用户的文本；None 表示无需输出
        """
        command = text.strip().lower()
        if not command:
            return False, None

        if command in _RESET_COMMANDS:
            self.reset()
            return True, "对话历史已重置。"

        if command in _HISTORY_COMMANDS:
            return True, self.view()

        if command in _HELP_COMMANDS:
            return True, INTERACTIVE_COMMAND_HELP.strip()

        if command in _EXIT_COMMANDS:
            return True, "__EXIT__"

        return False, None


INTERACTIVE_COMMAND_HELP = """
连续对话命令:
  history   查看对话历史
  reset     重置对话历史
  /help     显示帮助
  exit      退出
"""


def main() -> None:
    history = ConversationHistory(max_turns=3, max_tokens=200)
    history.add("招商银行利润怎么样？", "招商银行净利润同比增长。")
    history.add("手续费呢？", "手续费及佣金净收入略有上升。")
    print(history.view())
    print()
    print(history.build_prompt_context("那同比呢？", references="[1] 参考资料示例"))
    print()
    handled, msg = history.handle_command("reset")
    print("reset:", handled, msg)


if __name__ == "__main__":
    main()
