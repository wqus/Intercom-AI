import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.services.llm_client import YandexLLMClient
from app.services.knowledge_base import KnowledgeBase
from app.api.webhook import router as webhook_router
from app.api.process import router as process_router
from app.core.dependencies import init_llm_client, init_knowledge_base
from app.database.base import Base
from app.database.engine import engine

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    logger.info("DRY_RUN=%s", settings.DRY_RUN)

    logger.info("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")

    llm_client = YandexLLMClient(
        api_key=settings.YANDEX_API_KEY,
        folder_id=settings.YANDEX_FOLDER_ID,
        prompt_id=settings.YANDEX_PROMPT_ID,
    )

    knowledge_base = KnowledgeBase(path="data/knowledge_base.json")

    init_llm_client(llm_client)
    init_knowledge_base(knowledge_base)

    logger.info("LLM client and knowledge base initialized")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Intercom AI Support",
    lifespan=lifespan,
)

app.include_router(webhook_router)
app.include_router(process_router)