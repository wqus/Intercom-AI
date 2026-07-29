from app.services.llm_client import YandexLLMClient
from app.services.knowledge_base import KnowledgeBase
from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.session import SessionLocal

_llm_client: YandexLLMClient | None = None
_knowledge_base: KnowledgeBase | None = None


def init_llm_client(client: YandexLLMClient) -> None:
    global _llm_client
    _llm_client = client


def init_knowledge_base(kb: KnowledgeBase) -> None:
    global _knowledge_base
    _knowledge_base = kb


def get_llm_client() -> YandexLLMClient:
    if _llm_client is None:
        raise RuntimeError("LLM client not initialized")
    return _llm_client


def get_knowledge_base() -> KnowledgeBase:
    if _knowledge_base is None:
        raise RuntimeError("Knowledge base not initialized")
    return _knowledge_base

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session