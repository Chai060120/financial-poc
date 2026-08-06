"""Information Processing Agent：PDF 财报 + 财经新闻。"""

from src.agent.autonomic_agent import (
    AutonomicAgent,
    AutonomicRunResult,
    create_autonomic_agent,
)
from src.agent.daily import DailyAgent, run_daily_agent
from src.agent.financial_agent import (
    AGENT_DATA_SOURCES,
    FINANCIAL_AGENT_PROMPT,
    FinancialAgent,
    create_financial_agent,
)
from src.agent.planner import Planner, RuleBasedIntentClassifier
from src.agent.registry import (
    INTENT_REGISTRY,
    SOURCE_REGISTRY,
    register_intent,
    register_source,
)
from src.agent.types import AgentContext, IntentSpec, PipelineStep, RetrievalPlan, SourceSpec
from src.agent.workflow import (
    AgentWorkflow,
    DEFAULT_SYSTEM_PROMPT,
    merge_retrieval_context,
    run_agent,
)

__all__ = [
    "AGENT_DATA_SOURCES",
    "AgentContext",
    "AgentWorkflow",
    "AutonomicAgent",
    "AutonomicRunResult",
    "DailyAgent",
    "DEFAULT_SYSTEM_PROMPT",
    "FINANCIAL_AGENT_PROMPT",
    "FinancialAgent",
    "INTENT_REGISTRY",
    "IntentSpec",
    "PipelineStep",
    "Planner",
    "RetrievalPlan",
    "RuleBasedIntentClassifier",
    "SOURCE_REGISTRY",
    "SourceSpec",
    "create_autonomic_agent",
    "create_financial_agent",
    "merge_retrieval_context",
    "register_intent",
    "register_source",
    "run_agent",
    "run_daily_agent",
]
