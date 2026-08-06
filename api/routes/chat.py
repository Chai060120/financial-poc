"""对话路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import AppState, get_app_state
from api.schemas import ChatLLMInfo, ChatPlanInfo, ChatRequest, ChatResponse, RetrievalItem
from api.services.chat_service import ChatServiceError, run_chat

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="RAG 连续对话")
def post_chat(
    body: ChatRequest,
    state: AppState = Depends(get_app_state),
) -> ChatResponse:
    session_id, history = state.get_or_create_session(
        body.session_id,
        reset=body.reset,
    )
    engine = state.get_retrieval_engine()

    try:
        payload = run_chat(
            engine,
            history,
            question=body.question,
            top_k=body.top_k,
            provider=body.provider,
            show_prompt=body.show_prompt,
        )
    except ChatServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    plan_data = payload.get("plan")
    llm_data = payload.get("llm")

    return ChatResponse(
        question=payload["question"],
        retrieval_query=payload["retrieval_query"],
        answer=payload["answer"],
        session_id=session_id,
        history_turns=payload["history_turns"],
        result_count=payload["result_count"],
        plan=ChatPlanInfo(**plan_data) if plan_data else None,
        llm=ChatLLMInfo(**llm_data) if llm_data else None,
        full_prompt=payload.get("full_prompt"),
        results=[RetrievalItem(**item) for item in payload.get("results", [])],
    )
