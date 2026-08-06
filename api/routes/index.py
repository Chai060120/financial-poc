"""索引路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import AppState, get_app_state
from api.schemas import IndexRequest, IndexResponse
from api.services.index_service import IndexServiceError, run_index

router = APIRouter(tags=["index"])


@router.post("/index", response_model=IndexResponse, summary="构建向量索引")
def post_index(
    body: IndexRequest,
    state: AppState = Depends(get_app_state),
) -> IndexResponse:
    store = state.get_chroma_store()
    embedder = state.get_embedder()

    try:
        payload = run_index(
            store,
            embedder,
            tokens_path=body.tokens_path,
            rebuild=body.rebuild,
        )
    except IndexServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if payload["status"] == "failed":
        raise HTTPException(status_code=500, detail="索引构建失败，请查看服务日志")

    return IndexResponse(**payload)
