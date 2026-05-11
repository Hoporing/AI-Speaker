import logging
from datetime import datetime
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger(__name__)

_RELEVANCE_THRESHOLD = 0.8


class MemoryStore:
    def __init__(self, db_path: str = "./memory_db"):
        logger.info("메모리 데이터베이스 로딩 중")
        ef = DefaultEmbeddingFunction()
        self._client = chromadb.PersistentClient(path=db_path)
        self._col = self._client.get_or_create_collection(
            name="conversations",
            embedding_function=ef,
        )
        logger.info(f"메모리 로드 완료 (저장된 대화: {self._col.count()}개)")

    def save(self, user_input: str, response: str):
        text = f"사용자: {user_input}\n어시스턴트: {response}"
        doc_id = datetime.now().isoformat()
        self._col.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[{"timestamp": doc_id}],
        )
        logger.info(f"대화 저장 완료 (총 {self._col.count()}개)")

    def search(self, query: str, n_results: int = 3) -> str:
        total = self._col.count()
        if total == 0:
            return ""

        results = self._col.query(
            query_texts=[query],
            n_results=min(n_results, total),
        )

        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        relevant = [
            doc for doc, dist in zip(docs, distances)
            if dist < _RELEVANCE_THRESHOLD
        ]

        if relevant:
            logger.info(f"관련 과거 대화 {len(relevant)}개 검색됨")

        return "\n\n".join(relevant)
