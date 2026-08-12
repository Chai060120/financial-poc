"""网页 Agent：会话状态与 AnalysisAgent 封装。"""

from __future__ import annotations

import uuid
from pathlib import Path

from config import AGENT_WATCHLIST, RAW_PDF_DIR, setup_logging
from src.agent.analysis_agent import AnalysisAgent
from src.agent.financial_agent import FinancialAgent, create_financial_agent
from src.agent.intent_router import AgentIntent
from src.agent.report_export import export_report
from src.utils.stock_registry import get_stock_registry

logger = setup_logging(__name__)


class WebAgentHub:
    """共享 FinancialAgent，按 session 隔离 AnalysisAgent 会话。"""

    def __init__(self) -> None:
        self._financial: FinancialAgent | None = None
        self._sessions: dict[str, AnalysisAgent] = {}

    def get_financial_agent(self) -> FinancialAgent:
        if self._financial is None:
            logger.info("初始化 FinancialAgent（网页 Agent）…")
            self._financial = create_financial_agent()
        return self._financial

    def get_or_create(self, session_id: str | None = None) -> tuple[str, AnalysisAgent]:
        sid = session_id or uuid.uuid4().hex
        agent = self._sessions.get(sid)
        if agent is None:
            agent = AnalysisAgent(self.get_financial_agent())
            self._sessions[sid] = agent
        return sid, agent

    def reset(self, session_id: str | None = None) -> tuple[str, AnalysisAgent]:
        sid, agent = self.get_or_create(session_id)
        from src.agent.analysis_agent import AgentSession

        agent.session = AgentSession()
        return sid, agent

    def chat(self, message: str, session_id: str | None = None) -> dict:
        sid, agent = self.get_or_create(session_id)
        result = agent.handle(message.strip())
        return {
            "session_id": sid,
            "intent": result.intent.value if isinstance(result.intent, AgentIntent) else str(result.intent),
            "reply": result.message,
            "entity_name": agent.session.last_entity_name,
            "entity_id": agent.session.last_entity_id,
        }

    def save_pdf(self, filename: str, data: bytes) -> Path:
        RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        dest = RAW_PDF_DIR / safe_name
        dest.write_bytes(data)
        return dest

    def upload_and_analyze(
        self, filename: str, data: bytes, session_id: str | None = None
    ) -> dict:
        path = self.save_pdf(filename, data)
        sid, agent = self.get_or_create(session_id)
        try:
            agent.financial.process_pdfs([path], build_index=True)
        except Exception as exc:
            logger.warning("PDF 入库提示: %s", exc)
        result = agent.handle(str(path))
        return {
            "session_id": sid,
            "intent": result.intent.value if isinstance(result.intent, AgentIntent) else str(result.intent),
            "reply": result.message,
            "filename": path.name,
            "entity_name": agent.session.last_entity_name,
            "entity_id": agent.session.last_entity_id,
        }

    def export_report(self, session_id: str | None = None, fmt: str = "md") -> Path:
        _sid, agent = self.get_or_create(session_id)
        report = (agent.session.last_report_card or "").strip()
        if not report and agent.session.last_analysis is not None:
            report = agent._build_report_card(agent.session.last_analysis)
            agent.session.last_report_card = report
        if not report:
            raise ValueError("暂无可导出报告，请先完成一次分析或公司对比")
        return export_report(
            report,
            entity_name=agent.session.last_entity_name or "report",
            fmt=fmt,
        )

    def watchlist(self) -> list[dict[str, str]]:
        registry = get_stock_registry()
        rows: list[dict[str, str]] = []
        for eid in AGENT_WATCHLIST:
            found = registry.lookup_by_id(eid) or {}
            rows.append(
                {
                    "entity_id": eid,
                    "entity_name": str(found.get("entity_name") or eid),
                }
            )
        return rows
