"""健康检查路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import AppState, get_app_state
from api.schemas import HealthResponse
from api.services.health_service import get_health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="服务健康检查")
def get_health_status(state: AppState = Depends(get_app_state)) -> HealthResponse:
    store = state.get_chroma_store()
    payload = get_health(store)
    return HealthResponse(**payload)
