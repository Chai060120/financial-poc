"""
统一 LLM 客户端：封装 Provider 模块，保持与 Agent / scripts 的兼容。

底层实现已迁移至 src.llm.base / factory / *_provider.py。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import setup_logging
from src.llm.base import (
    LLMConfigError,
    LLMNetworkError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    PromptPreviewProvider,
    normalize_provider_name,
)
from src.llm.factory import create_provider, try_create_provider

logger = setup_logging(__name__)

# 向后兼容：保留旧名称导出
LLMClientError = LLMProviderError
SUPPORTED_PROVIDERS = ("openai", "deepseek", "qwen")


@dataclass(frozen=True)
class LLMGenerateResult:
    """LLM 生成结果（兼容旧接口）。"""

    answer: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: int
    answer_length: int
    preview_mode: bool = False


class LLMClient:
    """
    统一 LLM 客户端（Adapter）。

    委托底层 LLMProvider 执行实际 API 调用。
    """

    def __init__(
        self,
        provider: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        allow_preview: bool = True,
        _provider_instance: LLMProvider | None = None,
    ) -> None:
        self.requested_provider = normalize_provider_name(provider)

        if _provider_instance is not None:
            self._provider = _provider_instance
        elif api_key or base_url or model or timeout is not None:
            self._provider = self._build_legacy_provider(
                provider=self.requested_provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=timeout,
            )
        else:
            self._provider = (
                try_create_provider(self.requested_provider)
                if allow_preview
                else create_provider(self.requested_provider, allow_preview=False)
            )

        self.provider = self._provider.provider_name
        self.model = self._provider.model
        self.preview_mode = isinstance(self._provider, PromptPreviewProvider)

        logger.info(
            "LLMClient 就绪: provider=%s, model=%s, preview_mode=%s",
            self.provider,
            self.model,
            self.preview_mode,
        )

    @staticmethod
    def _build_legacy_provider(
        *,
        provider: str,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        timeout: float | None,
    ) -> LLMProvider:
        """支持旧版显式传参（主要用于测试）。"""
        import os

        name = normalize_provider_name(provider)
        env_map = {
            "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "gpt-4o-mini", None),
            "deepseek": (
                "DEEPSEEK_API_KEY",
                "DEEPSEEK_BASE_URL",
                "DEEPSEEK_MODEL",
                "deepseek-chat",
                "https://api.deepseek.com",
            ),
            "qwen": (
                "QWEN_API_KEY",
                "QWEN_BASE_URL",
                "QWEN_MODEL",
                "qwen-plus",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        }
        key_env, url_env, model_env, default_model, default_url = env_map[name]

        if api_key:
            os.environ[key_env] = api_key
        if base_url:
            os.environ[url_env] = base_url
        if model:
            os.environ[model_env] = model
        if timeout is not None:
            os.environ["LLM_TIMEOUT"] = str(timeout)

        return create_provider(name, allow_preview=False)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """调用 LLM 生成回答。"""
        return self.generate_with_metadata(
            system_prompt,
            user_prompt,
            temperature=temperature,
            response_format=response_format,
        ).answer

    def generate_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> LLMGenerateResult:
        """调用 LLM 并返回带用量与耗时的结果。"""
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt 不能为空")
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt 不能为空")

        response = self._provider.generate_chat(
            system_prompt.strip(),
            user_prompt.strip(),
            temperature=temperature,
            response_format=response_format,
        )

        return LLMGenerateResult(
            answer=response.text,
            provider=response.provider,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            duration_ms=response.duration_ms,
            answer_length=len(response.text),
            preview_mode=self.preview_mode,
        )


def create_llm_client(
    provider: str | None = None,
    *,
    allow_preview: bool = True,
) -> LLMClient:
    """根据环境变量或指定 provider 创建 LLMClient。"""
    return LLMClient(provider=provider, allow_preview=allow_preview)


def main() -> None:
    """命令行调试入口。"""
    client = create_llm_client()
    answer = client.generate(
        "你是简洁的助手。",
        "用一句话介绍 RAG。",
    )
    print(f"Provider:    {client.provider}")
    print(f"Model:       {client.model}")
    print(f"PreviewMode: {client.preview_mode}")
    print(f"Answer:      {answer[:500]}")


if __name__ == "__main__":
    main()
