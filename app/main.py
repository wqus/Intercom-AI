import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.services.llm_client import YandexLLMClient
from app.services.knowledge_base import KnowledgeBase
from app.api.webhook import router as webhook_router
from app.api.process import router as process_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("main")

llm_client: YandexLLMClient | None = None
knowledge_base: KnowledgeBase | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, knowledge_base

    logger.info("Starting up...")
    logger.info("DRY_RUN=%s", settings.DRY_RUN)

    llm_client = YandexLLMClient(
        api_key=settings.YANDEX_API_KEY,
        folder_id=settings.YANDEX_FOLDER_ID,
        prompt_id=settings.YANDEX_PROMPT_ID,
    )

    knowledge_base = KnowledgeBase(path="data/knowledge_base.json")

    logger.info("LLM client and knowledge base initialized")

    yield  # приложение работает

    logger.info("Shutting down...")


app = FastAPI(
    title="Intercom AI Support",
    lifespan=lifespan,
)

app.include_router(webhook_router)
app.include_router(process_router)


def get_llm_client() -> YandexLLMClient:
    return llm_client


def get_knowledge_base() -> KnowledgeBase:
    return knowledge_base
