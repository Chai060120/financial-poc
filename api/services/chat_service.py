"""对话服务：封装 Financial Agent RAG 对话。"""



from __future__ import annotations



import os

import sys

from pathlib import Path

from typing import Any



_PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(_PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(_PROJECT_ROOT))



from config import add_project_root_to_path, setup_logging

from src.agent.financial_agent import FinancialAgent

from src.chat.history import ConversationHistory

from src.llm.llm_client import LLMClient, LLMClientError, LLMConfigError, create_llm_client

from src.vectorstore.retrieval import RetrievalError

from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine



add_project_root_to_path()

logger = setup_logging(__name__)





class ChatServiceError(Exception):

    """对话服务错误。"""





def run_chat(

    engine: UnifiedRetrievalEngine,

    history: ConversationHistory,

    *,

    question: str,

    top_k: int = 5,

    provider: str | None = None,

    llm: LLMClient | None = None,

    show_prompt: bool = False,

) -> dict[str, Any]:

    """执行 Financial Agent 对话并返回 JSON 可序列化结果。"""

    if not question.strip():

        raise ChatServiceError("question 不能为空")



    resolved_provider = provider or os.getenv("LLM_PROVIDER", "openai")

    client = llm or create_llm_client(provider=resolved_provider, allow_preview=True)

    agent = FinancialAgent(engine=engine, llm=client, history=history, top_k=top_k)



    try:

        payload = agent.ask(

            question.strip(),

            top_k=top_k,

            provider=resolved_provider,

            show_prompt=show_prompt,

        )

    except ValueError as exc:

        raise ChatServiceError(str(exc)) from exc

    except RetrievalError as exc:

        raise ChatServiceError(f"检索失败: {exc}") from exc

    except LLMConfigError as exc:

        raise ChatServiceError(f"LLM 配置错误: {exc}") from exc

    except LLMClientError as exc:

        raise ChatServiceError(f"LLM 调用失败: {exc}") from exc



    payload["history_turns"] = len(history)

    return payload

