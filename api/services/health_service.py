"""健康检查服务。"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import COLLECTION_NAME, EMBEDDING_MODEL, add_project_root_to_path
from src.vectorstore.chroma_store import ChromaStore

add_project_root_to_path()


def get_health(store: ChromaStore) -> dict[str, object]:
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "document_count": store.count(),
        "embedding_model": EMBEDDING_MODEL,
    }
