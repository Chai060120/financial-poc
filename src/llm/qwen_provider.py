"""通义千问 Provider（OpenAI Compatible API）。"""

from __future__ import annotations

from src.llm.base import OpenAICompatibleProvider


class QwenProvider(OpenAICompatibleProvider):
    """通义千问（DashScope Compatible Mode）Provider。"""

    provider_name = "qwen"

    def __init__(self) -> None:
        super().__init__(
            api_key_env="QWEN_API_KEY",
            base_url_env="QWEN_BASE_URL",
            model_env="QWEN_MODEL",
            default_model="qwen-plus",
            default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            fallback_api_key_envs=("DASHSCOPE_API_KEY",),
        )
