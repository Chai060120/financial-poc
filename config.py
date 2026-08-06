"""
项目全局配置。

统一管理路径、分块参数、日志等级、向量模型与业务常量。
所有路径使用 pathlib.Path，便于跨平台运行。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 项目根目录
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 数据目录（Medallion 分层）
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"

# Bronze：原始数据
RAW_DIR: Path = DATA_DIR / "raw"
RAW_PDF_DIR: Path = RAW_DIR / "pdf"
RAW_NEWS_DIR: Path = RAW_DIR / "news"

# Silver / Gold：清洗与 token 层
PROCESSED_DIR: Path = DATA_DIR / "processed"
CLEANED_DIR: Path = PROCESSED_DIR / "cleaned"
CHUNKS_DIR: Path = PROCESSED_DIR / "chunks"
TOKENS_DIR: Path = PROCESSED_DIR / "tokens"

# 向量索引
CHROMA_DIR: Path = DATA_DIR / "chroma"

# 常用输出文件
CLEANED_JSON: Path = CLEANED_DIR / "cleaned.json"
CHUNKS_JSON: Path = CHUNKS_DIR / "chunks.json"
TOKENS_JSON: Path = TOKENS_DIR / "tokens.json"
UNIFIED_CSV: Path = TOKENS_DIR / "unified.csv"

# ---------------------------------------------------------------------------
# 源码、脚本、文档
# ---------------------------------------------------------------------------
SRC_DIR: Path = PROJECT_ROOT / "src"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

COLLECTORS_DIR: Path = SRC_DIR / "collectors"
PROCESSORS_DIR: Path = SRC_DIR / "processors"
EMBEDDINGS_DIR: Path = SRC_DIR / "embeddings"
VECTORSTORE_DIR: Path = SRC_DIR / "vectorstore"
UTILS_DIR: Path = SRC_DIR / "utils"

# ---------------------------------------------------------------------------
# 文本处理
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("FINANCIAL_POC_CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("FINANCIAL_POC_CHUNK_OVERLAP", "100"))

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("FINANCIAL_POC_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Embedding / 向量检索
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "FINANCIAL_POC_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
)
COLLECTION_NAME: str = os.getenv("FINANCIAL_POC_COLLECTION", "financial_poc")
TOP_K: int = int(os.getenv("FINANCIAL_POC_TOP_K", "5"))

# Hybrid Retrieval（Embedding + BM25 + RRF，默认开启）
ENABLE_HYBRID: bool = os.getenv("FINANCIAL_POC_ENABLE_HYBRID", "true").lower() in (
    "true",
    "1",
    "yes",
)

# CrossEncoder 重排序（Hybrid/向量召回 Top-N → Rerank Top-M，默认开启）
ENABLE_RERANK: bool = os.getenv("FINANCIAL_POC_ENABLE_RERANK", "true").lower() in (
    "true",
    "1",
    "yes",
)
RERANK_MODEL: str = os.getenv(
    "FINANCIAL_POC_RERANK_MODEL", "BAAI/bge-reranker-base"
)
RETRIEVAL_TOP_K: int = int(os.getenv("FINANCIAL_POC_RETRIEVAL_TOP_K", "20"))
RERANK_TOP_K: int = int(os.getenv("FINANCIAL_POC_RERANK_TOP_K", "5"))

# Hybrid Retrieval（Embedding + BM25 + RRF）
HYBRID_EMBEDDING_WEIGHT: float = float(
    os.getenv("FINANCIAL_POC_HYBRID_EMBEDDING_WEIGHT", "0.5")
)
HYBRID_BM25_WEIGHT: float = float(os.getenv("FINANCIAL_POC_HYBRID_BM25_WEIGHT", "0.55"))
HYBRID_RRF_K: int = int(os.getenv("FINANCIAL_POC_HYBRID_RRF_K", "60"))
HYBRID_CANDIDATE_K: int = int(os.getenv("FINANCIAL_POC_HYBRID_CANDIDATE_K", "50"))

# 对话记忆
CHAT_MEMORY_MAX_TURNS: int = int(os.getenv("FINANCIAL_POC_CHAT_MEMORY_MAX_TURNS", "10"))
CHAT_HISTORY_MAX_TOKENS: int = int(
    os.getenv("FINANCIAL_POC_CHAT_HISTORY_MAX_TOKENS", "2000")
)

# 金融信息处理 Agent（日报流水线）
DAILY_REPORT_DIR: Path = DOCS_DIR / "daily"
AGENT_SCHEDULE_TIME: str = os.getenv("FINANCIAL_POC_AGENT_SCHEDULE_TIME", "08:00")
AGENT_NEWS_DAYS: int = int(os.getenv("FINANCIAL_POC_AGENT_NEWS_DAYS", "1"))
AGENT_RETRIEVAL_TOP_K: int = int(os.getenv("FINANCIAL_POC_AGENT_RETRIEVAL_TOP_K", "8"))
AGENT_ENABLE_LLM: bool = os.getenv("FINANCIAL_POC_AGENT_ENABLE_LLM", "true").lower() in (
    "true",
    "1",
    "yes",
)

# 新闻采集
NEWS_FETCH_LIMIT: int = int(os.getenv("FINANCIAL_POC_NEWS_FETCH_LIMIT", "30"))
NEWS_DEFAULT_SYMBOLS: tuple[str, ...] = tuple(
    item.strip()
    for item in os.getenv(
        "FINANCIAL_POC_NEWS_SYMBOLS",
        "600519.SH,600036.SH,601318.SH",
    ).split(",")
    if item.strip()
)
NEWS_RSS_FEEDS: tuple[str, ...] = tuple(
    item.strip()
    for item in os.getenv(
        "FINANCIAL_POC_RSS_FEEDS",
        "https://rss.sina.com.cn/finance/stock.xml,https://feedx.net/rss/finance.xml",
    ).split(",")
    if item.strip()
)
NEWS_JSON: Path = RAW_NEWS_DIR / "news.json"
NEWS_DEFAULT_DAYS: int = int(os.getenv("FINANCIAL_POC_NEWS_DEFAULT_DAYS", "7"))
AGENT_WATCHLIST: tuple[str, ...] = tuple(
    item.strip()
    for item in os.getenv(
        "FINANCIAL_POC_AGENT_WATCHLIST",
        os.getenv("FINANCIAL_POC_NEWS_SYMBOLS", "600519.SH,600036.SH,601318.SH"),
    ).split(",")
    if item.strip()
)

# 股票映射（简称 <-> 代码）
REFERENCE_DIR: Path = DATA_DIR / "reference"
STOCK_LIST_CSV: Path = REFERENCE_DIR / "stock_list.csv"
STOCK_REGISTRY_CACHE: Path = REFERENCE_DIR / "stock_registry.json"
ENABLE_AKSHARE_STOCK_SYNC: bool = os.getenv(
    "FINANCIAL_POC_ENABLE_AKSHARE_STOCK_SYNC", "true"
).lower() in ("true", "1", "yes")
ENABLE_TUSHARE_STOCK_SYNC: bool = os.getenv(
    "FINANCIAL_POC_ENABLE_TUSHARE_STOCK_SYNC", "false"
).lower() in ("true", "1", "yes")
TUSHARE_TOKEN: str = os.getenv("TUSHARE_TOKEN", "")

# 未知实体占位（自动识别失败时使用）
UNKNOWN_ENTITY_ID: str = "UNKNOWN"
UNKNOWN_ENTITY_NAME: str = "UNKNOWN"

# ---------------------------------------------------------------------------
# 业务常量
# ---------------------------------------------------------------------------
ENTITY_ID: str = "600519.SH"
ENTITY_NAME: str = "贵州茅台"
DEFAULT_ENCODING: str = "utf-8"

# ---------------------------------------------------------------------------
# 需要创建的目录列表
# ---------------------------------------------------------------------------
DATA_DIRS: tuple[Path, ...] = (
    RAW_PDF_DIR,
    RAW_NEWS_DIR,
    REFERENCE_DIR,
    CLEANED_DIR,
    CHUNKS_DIR,
    TOKENS_DIR,
    CHROMA_DIR,
    DAILY_REPORT_DIR,
)


def get_log_level() -> int:
    """将 LOG_LEVEL 字符串转为 logging 模块使用的整数等级。"""
    level_name = LOG_LEVEL.upper()
    if level_name not in logging._nameToLevel:
        return logging.INFO
    return logging._nameToLevel[level_name]


def _configure_stdio_utf8() -> None:
    """Windows 终端默认 GBK，强制 stdout/stderr 使用 UTF-8 减少中文乱码。"""
    if sys.platform != "win32":
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def setup_logging(name: str | None = None) -> logging.Logger:
    """
    初始化并返回 logger。

    在 scripts 或 src 模块开头调用一次即可：
        from config import setup_logging
        logger = setup_logging(__name__)
    """
    _configure_stdio_utf8()
    logging.basicConfig(
        level=get_log_level(),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        force=True,
    )
    return logging.getLogger(name or "financial_poc")


def ensure_dirs() -> None:
    """创建项目所需的全部数据目录（已存在则跳过）。"""
    for directory in DATA_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def add_project_root_to_path() -> Path:
    """
    将项目根目录加入 sys.path，便于 scripts 中 import config / src。

    返回 PROJECT_ROOT 便于链式调用。
    """
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "RAW_PDF_DIR",
    "RAW_NEWS_DIR",
    "PROCESSED_DIR",
    "CLEANED_DIR",
    "CHUNKS_DIR",
    "TOKENS_DIR",
    "CHROMA_DIR",
    "CLEANED_JSON",
    "CHUNKS_JSON",
    "TOKENS_JSON",
    "UNIFIED_CSV",
    "SRC_DIR",
    "SCRIPTS_DIR",
    "DOCS_DIR",
    "COLLECTORS_DIR",
    "PROCESSORS_DIR",
    "EMBEDDINGS_DIR",
    "VECTORSTORE_DIR",
    "UTILS_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "LOG_DATE_FORMAT",
    "EMBEDDING_MODEL",
    "COLLECTION_NAME",
    "TOP_K",
    "ENABLE_HYBRID",
    "ENABLE_RERANK",
    "RERANK_MODEL",
    "RETRIEVAL_TOP_K",
    "RERANK_TOP_K",
    "HYBRID_EMBEDDING_WEIGHT",
    "HYBRID_BM25_WEIGHT",
    "HYBRID_RRF_K",
    "HYBRID_CANDIDATE_K",
    "CHAT_MEMORY_MAX_TURNS",
    "CHAT_HISTORY_MAX_TOKENS",
    "DAILY_REPORT_DIR",
    "AGENT_SCHEDULE_TIME",
    "AGENT_NEWS_DAYS",
    "AGENT_RETRIEVAL_TOP_K",
    "AGENT_ENABLE_LLM",
    "AGENT_WATCHLIST",
    "NEWS_FETCH_LIMIT",
    "NEWS_DEFAULT_SYMBOLS",
    "NEWS_RSS_FEEDS",
    "NEWS_JSON",
    "NEWS_DEFAULT_DAYS",
    "REFERENCE_DIR",
    "STOCK_LIST_CSV",
    "STOCK_REGISTRY_CACHE",
    "ENABLE_AKSHARE_STOCK_SYNC",
    "ENABLE_TUSHARE_STOCK_SYNC",
    "TUSHARE_TOKEN",
    "UNKNOWN_ENTITY_ID",
    "UNKNOWN_ENTITY_NAME",
    "ENTITY_ID",
    "ENTITY_NAME",
    "DEFAULT_ENCODING",
    "DATA_DIRS",
    "get_log_level",
    "setup_logging",
    "ensure_dirs",
    "add_project_root_to_path",
]
