"""分析运行时静默日志与进度条，仅保留用户可见结果。"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager


@contextmanager
def quiet_analysis():
    """临时关闭 INFO 日志与 HF 进度输出。"""
    root = logging.getLogger()
    old_root_level = root.level
    old_handlers = {h: h.level for h in root.handlers}

    env_keys = (
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "TRANSFORMERS_VERBOSITY",
        "TOKENIZERS_PARALLELISM",
        "HF_HUB_DISABLE_TELEMETRY",
    )
    saved_env = {k: os.environ.get(k) for k in env_keys}

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    root.setLevel(logging.ERROR)
    for handler in root.handlers:
        handler.setLevel(logging.ERROR)
    for name in (
        "financial_poc",
        "sentence_transformers",
        "transformers",
        "httpx",
        "httpcore",
        "urllib3",
        "chromadb",
        "huggingface_hub",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    # sentence-transformers / tqdm
    old_stderr = sys.stderr
    try:
        yield
    finally:
        root.setLevel(old_root_level)
        for handler, level in old_handlers.items():
            handler.setLevel(level)
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
