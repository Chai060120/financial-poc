"""
步骤5：RAG 对话（Information Processing Agent）。

Pipeline: Question → Planner → Retriever → Merge Context → LLM → Final Answer
支持连续对话与 Conversation History。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CHAT_HISTORY_MAX_TOKENS,
    CHAT_MEMORY_MAX_TURNS,
    ENABLE_RERANK,
    RERANK_TOP_K,
    TOP_K,
    add_project_root_to_path,
    setup_logging,
)
from src.agent.workflow import AgentWorkflow, DEFAULT_SYSTEM_PROMPT
from src.chat.history import INTERACTIVE_COMMAND_HELP, ConversationHistory
from src.llm.llm_client import (
    LLMClient,
    LLMClientError,
    LLMConfigError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMTimeoutError,
    create_llm_client,
)
from src.vectorstore.retrieval import RetrievalError
from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine, create_retrieval_engine

add_project_root_to_path()
logger = setup_logging(__name__)

_DEFAULT_TOP_K = RERANK_TOP_K if ENABLE_RERANK else TOP_K


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Information Processing Agent：Planner + RAG 连续对话"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="用户问题；省略则进入连续对话模式",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help=f"检索 Top K，默认 {_DEFAULT_TOP_K}",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "deepseek", "qwen"),
        default=None,
        help="LLM 提供商，默认读取 .env 中 LLM_PROVIDER",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="打印完整 Prompt（默认仅输出回答）",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="连续对话模式（无 question 参数时默认开启）",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="仅回答一次后退出（即使未提供 question）",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=CHAT_MEMORY_MAX_TURNS,
        help=f"对话历史轮数上限，默认 {CHAT_MEMORY_MAX_TURNS}",
    )
    parser.add_argument(
        "--history-max-tokens",
        type=int,
        default=CHAT_HISTORY_MAX_TOKENS,
        help=f"对话历史 Prompt Token 上限，默认 {CHAT_HISTORY_MAX_TOKENS}",
    )
    return parser.parse_args()


def read_question(prompt: str = "请输入问题: ") -> str | None:
    try:
        question = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")
        return None

    return question or None


def handle_history_command(question: str, history: ConversationHistory) -> bool:
    """处理 history / reset / help / exit 命令。返回 True 表示已消费输入。"""
    handled, message = history.handle_command(question)
    if not handled:
        return False

    if message == "__EXIT__":
        print("再见。")
        raise SystemExit(0)

    if message:
        print(message)
    return True


def run_chat(
    question: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
    provider: str | None = None,
    engine: UnifiedRetrievalEngine | None = None,
    llm: LLMClient | None = None,
    history: ConversationHistory | None = None,
    workflow: AgentWorkflow | None = None,
) -> dict[str, Any]:
    """执行 Agent Pipeline 并返回 payload。"""
    agent = workflow or AgentWorkflow(
        engine=engine,
        llm=llm,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    ctx = agent.run(
        question,
        top_k=top_k,
        memory=history,
        engine=engine,
        llm=llm,
        provider=provider,
    )
    return ctx.to_payload()


def print_chat_output(payload: dict[str, Any], *, show_prompt: bool = False) -> None:
    results = payload["results"]
    llm_result = payload["llm_result"]
    plan = payload.get("plan")

    print(f"\n问题: {payload['question']}")
    if payload.get("retrieval_query") != payload["question"]:
        print(f"检索 query: {payload['retrieval_query']}")
    if plan is not None:
        print(
            f"Agent 意图: {plan.primary_intent.label} ({plan.primary_intent.intent_id})"
        )
        print(f"数据源:     {', '.join(plan.source_filters) or '全部'}")
        print(f"计划说明:   {plan.reasoning}")
    print(f"检索命中: {len(results)} 条 (Top K={payload['top_k']})")

    if show_prompt:
        print("\n" + "=" * 60)
        print("完整 Prompt")
        print("=" * 60)
        print(payload["full_prompt"])

    preview_mode = llm_result is not None and getattr(llm_result, "preview_mode", False)
    print("\n" + "=" * 60)
    print("Prompt Preview" if preview_mode else "回答")
    print("=" * 60)
    print(payload["answer"])

    if llm_result is not None and not preview_mode:
        mode = "Prompt Preview" if getattr(llm_result, "preview_mode", False) else "LLM"
        print(
            f"\n[mode={mode}, provider={llm_result.provider}, model={llm_result.model}, "
            f"tokens={llm_result.total_tokens}, duration={llm_result.duration_ms}ms]"
        )


def execute_chat_turn(
    question: str,
    *,
    top_k: int,
    provider: str,
    show_prompt: bool,
    engine: UnifiedRetrievalEngine,
    llm: LLMClient,
    history: ConversationHistory,
    workflow: AgentWorkflow,
) -> bool:
    logger.info(
        "Agent 对话: question=%r, top_k=%d, provider=%s, history=%d/%d, token_budget=%d",
        question[:80],
        top_k,
        provider,
        len(history),
        history.max_turns,
        history.max_tokens,
    )

    try:
        payload = run_chat(
            question,
            top_k=top_k,
            provider=provider,
            engine=engine,
            llm=llm,
            history=history,
            workflow=workflow,
        )
    except ValueError as exc:
        logger.error("参数错误: %s", exc)
        print(f"对话失败: {exc}")
        return False
    except RetrievalError as exc:
        logger.error("检索失败: %s", exc)
        print(f"检索失败: {exc}")
        return False
    except LLMConfigError as exc:
        logger.error("LLM 配置错误: %s", exc)
        print(f"LLM 配置错误: {exc}")
        print("请检查项目根目录 .env 文件中的 API Key 与 Provider 配置。")
        return False
    except LLMRateLimitError as exc:
        logger.error("LLM 速率限制: %s", exc)
        print(f"LLM 速率限制: {exc}")
        return False
    except LLMTimeoutError as exc:
        logger.error("LLM 超时: %s", exc)
        print(f"LLM 请求超时: {exc}")
        return False
    except LLMNetworkError as exc:
        logger.error("LLM 网络错误: %s", exc)
        print(f"LLM 网络错误: {exc}")
        return False
    except LLMClientError as exc:
        logger.error("LLM 调用失败: %s", exc)
        print(f"LLM 调用失败: {exc}")
        return False
    except Exception as exc:
        logger.exception("对话异常: %s", exc)
        print(f"对话失败: {exc}")
        return False

    if not payload["results"]:
        print(
            "提示: 未检索到相关 Token，回答可能不完整。"
            "可先运行: python scripts/02_process.py / process_pdf.py / 03_build_index.py"
        )

    print_chat_output(payload, show_prompt=show_prompt)
    return True


def resolve_llm_client(provider: str | None) -> LLMClient:
    """创建 LLMClient；无 API Key 时自动进入 Prompt Preview 模式。"""
    client = create_llm_client(provider=provider, allow_preview=True)
    if client.preview_mode:
        print(
            f"提示: Provider={provider or 'default'} 未配置 API Key，"
            "已进入 Prompt Preview 模式（仅展示 Prompt，不调用 LLM）。"
        )
        print("请在项目根目录 .env 中配置对应的 API Key。")
    return client


def run_interactive_session(args: argparse.Namespace) -> None:
    if args.memory_limit <= 0:
        print(f"memory_limit 必须大于 0，当前为 {args.memory_limit}")
        sys.exit(1)
    if args.history_max_tokens <= 0:
        print(f"history_max_tokens 必须大于 0，当前为 {args.history_max_tokens}")
        sys.exit(1)

    provider = args.provider or os.getenv("LLM_PROVIDER", "openai")
    history = ConversationHistory(
        max_turns=args.memory_limit,
        max_tokens=args.history_max_tokens,
    )
    engine = create_retrieval_engine(top_k=args.top_k)
    llm = resolve_llm_client(provider)
    workflow = AgentWorkflow(engine=engine, llm=llm, system_prompt=DEFAULT_SYSTEM_PROMPT)

    print("=" * 60)
    print("Financial PoC · Information Processing Agent")
    print("=" * 60)
    print(
        f"历史: 最近 {history.max_turns} 轮 | "
        f"Prompt Token 上限 {history.max_tokens} | 输入 help 查看命令"
    )
    print(INTERACTIVE_COMMAND_HELP.strip())

    first_question = args.question.strip() if args.question else None
    if first_question:
        if not handle_history_command(first_question, history):
            execute_chat_turn(
                first_question,
                top_k=args.top_k,
                provider=provider,
                show_prompt=args.show_prompt,
                engine=engine,
                llm=llm,
                history=history,
                workflow=workflow,
            )

    while True:
        question = read_question("\n请输入问题: ")
        if question is None:
            break
        if handle_history_command(question, history):
            continue
        execute_chat_turn(
            question,
            top_k=args.top_k,
            provider=provider,
            show_prompt=args.show_prompt,
            engine=engine,
            llm=llm,
            history=history,
            workflow=workflow,
        )


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        print(f"top_k 必须大于 0，当前为 {args.top_k}")
        sys.exit(1)

    interactive = args.interactive or (not args.question and not args.once)
    if interactive:
        run_interactive_session(args)
        return

    question = args.question
    if not question:
        question = read_question()
        if not question:
            sys.exit(1)

    if args.memory_limit <= 0:
        print(f"memory_limit 必须大于 0，当前为 {args.memory_limit}")
        sys.exit(1)
    if args.history_max_tokens <= 0:
        print(f"history_max_tokens 必须大于 0，当前为 {args.history_max_tokens}")
        sys.exit(1)

    provider = args.provider or os.getenv("LLM_PROVIDER", "openai")
    history = ConversationHistory(
        max_turns=args.memory_limit,
        max_tokens=args.history_max_tokens,
    )

    if handle_history_command(question, history):
        return

    llm = resolve_llm_client(provider)
    engine = create_retrieval_engine(top_k=args.top_k)
    workflow = AgentWorkflow(engine=engine, llm=llm, system_prompt=DEFAULT_SYSTEM_PROMPT)

    success = execute_chat_turn(
        question,
        top_k=args.top_k,
        provider=provider,
        show_prompt=args.show_prompt,
        engine=engine,
        llm=llm,
        history=history,
        workflow=workflow,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
