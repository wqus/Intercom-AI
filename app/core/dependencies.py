from app.services.llm_client import YandexLLMClient
from app.services.knowledge_base import KnowledgeBase

_llm_client: YandexLLMClient | None = None
_knowledge_base: KnowledgeBase | None = None


def init_llm_client(client: YandexLLMClient) -> None:
    """Инициализация LLM клиента (вызывается из lifespan)"""
    global _llm_client
    _llm_client = client


def init_knowledge_base(kb: KnowledgeBase) -> None:
    """Инициализация Knowledge Base (вызывается из lifespan)"""
    global _knowledge_base
    _knowledge_base = kb


def get_llm_client() -> YandexLLMClient:
    """Dependency for Yandex LLM client"""
    if _llm_client is None:
        raise RuntimeError("LLM client not initialized")
    return _llm_client


def get_knowledge_base() -> KnowledgeBase:
    """Dependency for knowledge base"""
    if _knowledge_base is None:
        raise RuntimeError("Knowledge base not initialized")
    return _knowledge_base