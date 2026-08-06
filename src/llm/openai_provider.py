"""OpenAI Provider。"""

from __future__ import annotations

from src.llm.base import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI API Provider。"""

    provider_name = "openai"

    def __init__(self) -> None:
        super().__init__(
            api_key_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL",
            model_env="OPENAI_MODEL",
            default_model="gpt-4o-mini",
            default_base_url=None,
        )
