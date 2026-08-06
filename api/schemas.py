"""API 请求/响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from config import ENABLE_HYBRID, ENABLE_RERANK, RERANK_TOP_K, TOP_K

_DEFAULT_QUERY_TOP_K = RERANK_TOP_K if ENABLE_RERANK else TOP_K
_DEFAULT_CHAT_TOP_K = RERANK_TOP_K if ENABLE_RERANK else TOP_K


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    collection: str
    document_count: int
    embedding_model: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="检索问题")
    top_k: int = Field(
        default=_DEFAULT_QUERY_TOP_K,
        ge=1,
        le=50,
        description="返回条数",
    )
    hybrid: bool | None = Field(
        default=None,
        description="启用 Hybrid 检索；省略则使用系统默认（通常为 true）",
    )
    rerank: bool | None = Field(
        default=None,
        description="CrossEncoder 重排序；省略则使用系统默认（通常为 true）",
    )
    entity_name: str | None = Field(default=None, description="按公司简称过滤")
    entity_id: str | None = Field(default=None, description="按股票代码过滤")
    source: Literal["pdf", "news"] | None = Field(default=None, description="按来源过滤")
    section: str | None = Field(default=None, description="按财报章节过滤")
    date_from: str | None = Field(default=None, description="日期下限 YYYY-MM-DD")
    date_to: str | None = Field(default=None, description="日期上限 YYYY-MM-DD")


class RetrievalItem(BaseModel):
    id: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any]
    rerank_score: float | None = None
    rrf_score: float | None = None


class QueryResponse(BaseModel):
    question: str
    mode: str
    count: int
    results: list[RetrievalItem]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(
        default=_DEFAULT_CHAT_TOP_K,
        ge=1,
        le=50,
        description="检索 Top K",
    )
    provider: Literal["openai", "deepseek", "qwen"] | None = Field(
        default=None, description="LLM 提供商"
    )
    session_id: str | None = Field(default=None, description="连续对话会话 ID")
    reset: bool = Field(default=False, description="重置会话历史")
    show_prompt: bool = Field(default=False, description="响应中返回完整 Prompt")


class ChatPlanInfo(BaseModel):
    intent_id: str
    intent_label: str
    source_filters: list[str]
    reasoning: str


class ChatLLMInfo(BaseModel):
    provider: str
    model: str
    total_tokens: int
    duration_ms: int
    preview_mode: bool


class ChatResponse(BaseModel):
    question: str
    retrieval_query: str
    answer: str
    session_id: str
    history_turns: int
    result_count: int
    plan: ChatPlanInfo | None = None
    llm: ChatLLMInfo | None = None
    full_prompt: str | None = None
    results: list[RetrievalItem] = Field(default_factory=list)


class IndexRequest(BaseModel):
    tokens_path: str | None = Field(default=None, description="Token JSON 路径，默认 config.TOKENS_JSON")
    rebuild: bool = Field(default=False, description="清空后重建索引")


class IndexResponse(BaseModel):
    status: Literal["success", "failed"] = "success"
    collection: str
    indexed: int
    skipped_existing: int
    skipped_invalid: int
    failed: int
    before_count: int
    after_count: int
    embedding_model: str


class ErrorResponse(BaseModel):
    detail: str
