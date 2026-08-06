"""
LLM Provider 工厂：根据 LLM_PROVIDER / --provider 创建对应 Provider。

无 API Key 时可返回 PromptPreviewProvider。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import setup_logging
from src.llm.base import (
    LLMConfigError,
    LLMProvider,
    PromptPreviewProvider,
    normalize_provider_name,
)
from src.llm.deepseek_provider import DeepSeekProvider
from src.llm.openai_provider import OpenAIProvider
from src.llm.qwen_provider import QwenProvider

logger = setup_logging(__name__)

ProviderFactory = Callable[[], LLMProvider]

_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """注册自定义 Provider 工厂。"""
    _PROVIDER_FACTORIES[normalize_provider_name(name)] = factory
    logger.info("已注册 LLM Provider: %s", name)


def create_provider(
    provider: str | None = None,
    *,
    allow_preview: bool = False,
) -> LLMProvider:
    """
    创建 LLM Provider。

    Args:
        provider: openai / deepseek / qwen；默认读取 LLM_PROVIDER。
        allow_preview: True 时 API Key 缺失则返回 PromptPreviewProvider。

    Raises:
        LLMConfigError: Provider 无效且不允许 preview。
    """
    name = normalize_provider_name(provider)
    factory = _PROVIDER_FACTORIES.get(name)
    if factory is None:
        raise LLMConfigError(f"未注册的 Provider: {name}")

    try:
        instance = factory()
        logger.info("Provider 创建成功: %s", name)
        return instance
    except LLMConfigError as exc:
        if allow_preview:
            logger.warning("Provider %s 不可用，回退 Prompt Preview: %s", name, exc)
            return PromptPreviewProvider(requested_provider=name)
        raise


def try_create_provider(provider: str | None = None) -> LLMProvider:
    """尝试创建 Provider；无 API Key 时自动回退 Prompt Preview。"""
    return create_provider(provider, allow_preview=True)


def is_provider_configured(provider: str | None = None) -> bool:
    """检查指定 Provider 是否已配置 API Key。"""
    name = normalize_provider_name(provider)
    try:
        instance = create_provider(name, allow_preview=False)
    except LLMConfigError:
        return False
    return instance.is_available()


def list_providers() -> list[str]:
    """返回已注册的 Provider 名称列表。"""
    return sorted(_PROVIDER_FACTORIES.keys())
