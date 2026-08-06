"""可复用数据处理流水线（供 scripts 与 Agent 共用）。"""

from src.pipelines.document_pipeline import run_document_processing
from src.pipelines.index_pipeline import run_index_build

__all__ = ["run_document_processing", "run_index_build"]
