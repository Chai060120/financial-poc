"""检索服务：封装 Financial Agent 检索。"""



from __future__ import annotations



import sys

from pathlib import Path

from typing import Any



_PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(_PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(_PROJECT_ROOT))



from config import RERANK_TOP_K, add_project_root_to_path, setup_logging

from src.agent.financial_agent import FinancialAgent

from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine



add_project_root_to_path()

logger = setup_logging(__name__)





class QueryServiceError(Exception):

    """检索服务错误。"""





def run_query(

    engine: UnifiedRetrievalEngine,

    *,

    question: str,

    top_k: int | None = None,

    hybrid: bool | None = None,

    rerank: bool | None = None,

    entity_name: str | None = None,

    entity_id: str | None = None,

    source: str | None = None,

    section: str | None = None,

    date_from: str | None = None,

    date_to: str | None = None,

) -> dict[str, Any]:

    """执行检索并返回 JSON 可序列化结果。"""

    if not question.strip():

        raise QueryServiceError("question 不能为空")



    agent = FinancialAgent(engine=engine, top_k=top_k or RERANK_TOP_K)

    if hybrid is not None:

        agent.engine.enable_hybrid = hybrid  # type: ignore[union-attr]

    if rerank is not None:

        agent.engine.enable_rerank = rerank  # type: ignore[union-attr]



    try:

        if any([entity_name, entity_id, source, section, date_from, date_to]):

            final_k = top_k or RERANK_TOP_K

            results = engine.retrieve(

                question,

                top_k=final_k,

                hybrid=hybrid,

                rerank=rerank,

                entity_name=entity_name,

                entity_id=entity_id,

                source=source,

                section=section,

                date_from=date_from,

                date_to=date_to,

            )

            from src.utils.query_filters import describe_retrieval_mode



            use_hybrid = engine.enable_hybrid if hybrid is None else hybrid

            use_rerank = engine.enable_rerank if rerank is None else rerank

            from api.serializers import serialize_retrieval_result



            serialized = [serialize_retrieval_result(item) for item in results]

            mode = describe_retrieval_mode(hybrid=use_hybrid, rerank=use_rerank)

            return {

                "question": question.strip(),

                "mode": mode,

                "count": len(serialized),

                "results": serialized,

            }

        return agent.query(question, top_k=top_k)

    except ValueError as exc:

        raise QueryServiceError(str(exc)) from exc

    except Exception as exc:

        raise QueryServiceError(str(exc)) from exc

