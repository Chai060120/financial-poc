"""
LLM Provider 基础定义：统一接口、异常、OpenAI Compatible 实现。
"""

from __future__ import annotations

import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

from config import PROJECT_ROOT, setup_logging

logger = setup_logging(__name__)

load_dotenv(PROJECT_ROOT / ".env", override=False)

SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai", "deepseek", "qwen")

PROVIDER_ALIASES: dict[str, str] = {
    "dashscope": "qwen",
    "tongyi": "qwen",
    "通义千问": "qwen",
}


class LLMProviderError(Exception):
    """LLM Provider 基础异常。"""


class LLMConfigError(LLMProviderError):
    """配置错误（如 API Key 缺失）。"""


class LLMNetworkError(LLMProviderError):
    """网络连接失败。"""


class LLMRateLimitError(LLMProviderError):
    """触发 API 速率限制。"""


class LLMTimeoutError(LLMProviderError):
    """请求超时。"""


@dataclass(frozen=True)
class LLMResponse:
    """LLM 生成结果。"""

    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0


def normalize_provider_name(provider: str | None) -> str:
    """规范化 Provider 名称。"""
    name = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    name = PROVIDER_ALIASES.get(name, name)
    if name not in SUPPORTED_PROVIDERS:
        raise LLMConfigError(
            f"不支持的 LLM Provider: {provider}，可选: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return name


def resolve_api_key(*env_names: str) -> str:
    """按顺序读取第一个非空 API Key。"""
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


class LLMProvider(ABC):
    """LLM Provider 统一接口。"""

    provider_name: str

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """根据单条 prompt 生成文本。"""

    @abstractmethod
    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """根据 system + user prompt 生成文本并返回元数据。"""

    @abstractmethod
    def is_available(self) -> bool:
        """当前 Provider 是否已配置可用（如 API Key 存在）。"""

    @property
    @abstractmethod
    def model(self) -> str:
        """当前使用的模型名称。"""


class OpenAICompatibleProvider(LLMProvider):
    """
    基于 OpenAI SDK 的 Compatible API Provider。

    OpenAI / DeepSeek / Qwen 共用此实现，子类仅配置环境变量与默认值。
    """

    provider_name: str = "openai"

    def __init__(
        self,
        *,
        api_key_env: str,
        base_url_env: str,
        model_env: str,
        default_model: str,
        default_base_url: str | None = None,
        fallback_api_key_envs: tuple[str, ...] = (),
        timeout_env: str = "LLM_TIMEOUT",
        require_api_key: bool = True,
    ) -> None:
        self._api_key_env = api_key_env
        self._base_url_env = base_url_env
        self._model_env = model_env
        self._default_model = default_model
        self._default_base_url = default_base_url
        self._fallback_api_key_envs = fallback_api_key_envs
        self._timeout = float(os.getenv(timeout_env, "60"))
        self._require_api_key = require_api_key

        self._api_key = resolve_api_key(api_key_env, *fallback_api_key_envs)
        self._base_url = (
            os.getenv(base_url_env, "").strip() or default_base_url or ""
        ).strip() or None
        self._model = (
            os.getenv(model_env, "").strip() or default_model
        ).strip()
        self._client: Any | None = None

        if self._require_api_key and not self._api_key:
            raise LLMConfigError(self._missing_key_message())

        if not self._model:
            raise LLMConfigError(f"未配置模型名称，请设置 {model_env}")

        if self.is_available():
            self._client = self._build_client()
            logger.info(
                "%s Provider 就绪: model=%s, base_url=%s, timeout=%.1fs",
                self.provider_name,
                self._model,
                self._base_url or "(default)",
                self._timeout,
            )

    def _missing_key_message(self) -> str:
        keys = ", ".join((self._api_key_env, *self._fallback_api_key_envs))
        return f"未配置 API Key，请设置环境变量: {keys}"

    def is_available(self) -> bool:
        return bool(self._api_key) and bool(self._model)

    @property
    def model(self) -> str:
        return self._model

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigError("未安装 openai 包，请运行: pip install openai") from exc

        kwargs: dict[str, Any] = {
            "api_key": self._api_key,
            "timeout": self._timeout,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return OpenAI(**kwargs)

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt 不能为空")
        return self.generate_chat("", prompt.strip()).text

    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt 不能为空")
        if not self.is_available() or self._client is None:
            raise LLMConfigError(self._missing_key_message())

        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt.strip()})

        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            request["response_format"] = response_format

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            if response_format is not None:
                logger.debug("response_format 不被支持，回退普通调用: %s", exc)
                request.pop("response_format", None)
                try:
                    response = self._client.chat.completions.create(**request)
                except Exception as retry_exc:
                    raise self._translate_error(retry_exc) from retry_exc
            else:
                raise self._translate_error(exc) from exc

        duration_ms = int((time.perf_counter() - started) * 1000)
        text = self._extract_text(response)
        usage = getattr(response, "usage", None)

        result = LLMResponse(
            text=text,
            provider=self.provider_name,
            model=self._model,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            duration_ms=duration_ms,
        )
        logger.info(
            "LLM 调用完成: provider=%s, model=%s, tokens=%d, duration=%dms, temperature=%s",
            result.provider,
            result.model,
            result.total_tokens,
            result.duration_ms,
            temperature,
        )
        return result

    def _extract_text(self, response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMProviderError("LLM 返回格式异常，无法解析回答内容") from exc

        if not content or not str(content).strip():
            raise LLMProviderError("LLM 返回空回答")
        return str(content).strip()

    def _translate_error(self, exc: Exception) -> LLMProviderError:
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                AuthenticationError,
                RateLimitError,
            )
        except ImportError:
            return LLMProviderError(f"LLM 调用失败: {exc}")

        if isinstance(exc, AuthenticationError):
            return LLMConfigError(
                f"API Key 无效或未授权 (provider={self.provider_name})，请检查 .env 配置"
            )
        if isinstance(exc, RateLimitError):
            return LLMRateLimitError(
                f"触发 API 速率限制 (provider={self.provider_name})，请稍后重试"
            )
        if isinstance(exc, APITimeoutError):
            return LLMTimeoutError(
                f"LLM 请求超时 (provider={self.provider_name}, timeout={self._timeout}s)"
            )
        if isinstance(exc, APIConnectionError):
            return LLMNetworkError(
                f"LLM 网络连接失败 (provider={self.provider_name})，请检查网络与 BASE_URL"
            )

        message = str(exc).strip() or exc.__class__.__name__
        return LLMProviderError(
            f"LLM 调用失败 (provider={self.provider_name}): {message}"
        )


class PromptPreviewProvider(LLMProvider):
    """无 API Key 时的 Prompt Preview 模式。"""

    provider_name = "preview"

    def __init__(self, requested_provider: str | None = None) -> None:
        self._requested_provider = normalize_provider_name(requested_provider)
        self._model = "prompt-preview"
        logger.warning(
            "进入 Prompt Preview 模式: provider=%s 未配置 API Key",
            self._requested_provider,
        )

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str) -> str:
        return self.generate_chat("", prompt).text

    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        _ = temperature, response_format
        sections: list[str] = [
            "【Prompt Preview 模式】",
            "未检测到有效 API Key，未调用大语言模型。",
            f"目标 Provider: {self._requested_provider}",
            "",
        ]
        if system_prompt.strip():
            sections.extend(["--- System Prompt ---", system_prompt.strip(), ""])
        sections.extend(["--- User Prompt ---", user_prompt.strip()])

        preview_text = "\n".join(sections)
        return LLMResponse(
            text=preview_text,
            provider="preview",
            model=self._model,
            duration_ms=0,
        )
