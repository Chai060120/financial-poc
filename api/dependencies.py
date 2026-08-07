"""FastAPI 依赖与应用状态。"""

from __future__ import annotations



import sys

import uuid

from dataclasses import dataclass, field

from pathlib import Path



from fastapi import Request



_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(_PROJECT_ROOT))



from config import (

    CHAT_HISTORY_MAX_TOKENS,

    CHAT_MEMORY_MAX_TURNS,

    CHROMA_DIR,

    COLLECTION_NAME,

    EMBEDDING_MODEL,

    add_project_root_to_path,

    setup_logging,

)

from src.chat.history import ConversationHistory

from src.embeddings.text_embedding import TextEmbedder

from src.vectorstore.chroma_store import ChromaStore

from src.vectorstore.unified_retrieval import UnifiedRetrievalEngine



add_project_root_to_path()

logger = setup_logging(__name__)





@dataclass

class AppState:

    """应用级单例与连续对话会话存储。"""



    retrieval_engine: UnifiedRetrievalEngine | None = None

    chroma_store: ChromaStore | None = None

    embedder: TextEmbedder | None = None

    chat_sessions: dict[str, ConversationHistory] = field(default_factory=dict)



    def get_chroma_store(self) -> ChromaStore:

        if self.chroma_store is None:

            self.chroma_store = ChromaStore(

                persist_directory=CHROMA_DIR,

                collection_name=COLLECTION_NAME,

            )

        return self.chroma_store



    def get_embedder(self) -> TextEmbedder:

        if self.embedder is None:

            self.embedder = TextEmbedder.get_instance(model_name=EMBEDDING_MODEL)

        return self.embedder



    def get_retrieval_engine(self) -> UnifiedRetrievalEngine:

        if self.retrieval_engine is None:

            self.retrieval_engine = UnifiedRetrievalEngine(

                store=self.get_chroma_store(),

                embedder=self.get_embedder(),

            )

        return self.retrieval_engine



    def get_or_create_session(

        self,

        session_id: str | None,

        *,

        reset: bool = False,

    ) -> tuple[str, ConversationHistory]:

        if session_id and session_id in self.chat_sessions:

            history = self.chat_sessions[session_id]

            if reset:

                history.reset()

            return session_id, history



        new_id = session_id or uuid.uuid4().hex

        history = ConversationHistory(

            max_turns=CHAT_MEMORY_MAX_TURNS,

            max_tokens=CHAT_HISTORY_MAX_TOKENS,

        )

        if reset:

            history.reset()

        self.chat_sessions[new_id] = history

        return new_id, history





def get_app_state(request: Request) -> AppState:

    state = getattr(request.app.state, "app_state", None)

    if not isinstance(state, AppState):

        state = AppState()

        request.app.state.app_state = state

    return state

