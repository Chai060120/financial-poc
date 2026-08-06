"""DeepSeek Provider（OpenAI Compatible API）。"""

from __future__ import annotations

from src.llm.base import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API Provider。"""

    provider_name = "deepseek"

    def __init__(self) -> None:
        super().__init__(
            api_key_env="DEEPSEEK_API_KEY",
            base_url_env="DEEPSEEK_BASE_URL",
            model_env="DEEPSEEK_MODEL",
            default_model="deepseek-chat",
            default_base_url="https://api.deepseek.com",
        )
