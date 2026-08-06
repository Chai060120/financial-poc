"""检索路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import AppState, get_app_state
from api.schemas import QueryRequest, QueryResponse, RetrievalItem
from api.services.query_service import QueryServiceError, run_query

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse, summary="智能检索（默认 Hybrid + Rerank）")
def post_query(
    body: QueryRequest,
    state: AppState = Depends(get_app_state),
) -> QueryResponse:
    engine = state.get_retrieval_engine()
    try:
        payload = run_query(
            engine,
            question=body.question,
            top_k=body.top_k,
            hybrid=body.hybrid,
            rerank=body.rerank,
            entity_name=body.entity_name,
            entity_id=body.entity_id,
            source=body.source,
            section=body.section,
            date_from=body.date_from,
            date_to=body.date_to,
        )
    except QueryServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return QueryResponse(
        question=payload["question"],
        mode=payload["mode"],
        count=payload["count"],
        results=[RetrievalItem(**item) for item in payload["results"]],
    )
