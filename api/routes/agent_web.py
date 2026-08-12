"""网页 Agent REST 接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.dependencies import AppState, get_app_state

router = APIRouter(prefix="/api/agent", tags=["web-agent"])


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入：公司名 / 追问 / PDF 名")
    session_id: str | None = Field(default=None, description="会话 ID，连续对话时回传")


class AgentChatResponse(BaseModel):
    session_id: str
    intent: str
    reply: str
    entity_name: str = ""
    entity_id: str = ""
    filename: str | None = None


class AgentResetRequest(BaseModel):
    session_id: str | None = None


@router.get("/watchlist", summary="监控列表快捷公司")
def get_watchlist(state: AppState = Depends(get_app_state)) -> dict:
    hub = state.get_web_agent_hub()
    return {"items": hub.watchlist()}


@router.post("/chat", response_model=AgentChatResponse, summary="网页 Agent 对话")
def post_agent_chat(
    body: AgentChatRequest,
    state: AppState = Depends(get_app_state),
) -> AgentChatResponse:
    hub = state.get_web_agent_hub()
    try:
        payload = hub.chat(body.message, body.session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AgentChatResponse(**payload)


@router.post("/reset", response_model=AgentChatResponse, summary="清空网页 Agent 会话")
def post_agent_reset(
    body: AgentResetRequest,
    state: AppState = Depends(get_app_state),
) -> AgentChatResponse:
    hub = state.get_web_agent_hub()
    sid, _agent = hub.reset(body.session_id)
    return AgentChatResponse(
        session_id=sid,
        intent="reset",
        reply="已清空会话。请输入公司名或上传 PDF 开始分析。",
    )


@router.post("/upload", response_model=AgentChatResponse, summary="上传 PDF 并分析")
async def post_agent_upload(
    file: UploadFile = File(...),
    session_id: str | None = Query(default=None),
    state: AppState = Depends(get_app_state),
) -> AgentChatResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    hub = state.get_web_agent_hub()
    try:
        payload = hub.upload_and_analyze(file.filename, data, session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AgentChatResponse(**payload)


@router.get("/export", summary="导出最近一次分析/对比报告")
def export_agent_report(
    session_id: str | None = Query(default=None),
    format: str = Query(default="md", pattern="^(md|html|markdown)$"),
    state: AppState = Depends(get_app_state),
):
    hub = state.get_web_agent_hub()
    try:
        path = hub.export_report(session_id=session_id, fmt=format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"导出失败: {exc}") from exc

    media = "text/markdown; charset=utf-8" if path.suffix == ".md" else "text/html; charset=utf-8"
    return FileResponse(path, media_type=media, filename=path.name)
