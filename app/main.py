import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.database.base import Base
from app.database.engine import engine
from app.infrastructure.persistence import models as _persistence_models
from app.infrastructure.ai.knowledge_base import KnowledgeBase
from app.infrastructure.ai.llm_client import YandexLLMClient
from app.infrastructure.intercom.client import IntercomGateway
from app.application.decision_pipeline import DecisionPipeline
from app.application.support_service import SupportApplication
from app.application.event_worker import EventWorker
from app.api.webhook import router as webhook_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    kb=KnowledgeBase(path=settings.KB_PATH, device=settings.EMBEDDING_DEVICE, use_reranker=settings.USE_RERANKER)
    llm=YandexLLMClient(
        api_key=settings.YANDEX_API_KEY,
        folder_id=settings.YANDEX_FOLDER_ID,
        prompt_id=settings.YANDEX_PROMPT_ID,
    )
    client=httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=20,max_connections=50),
        headers={"Authorization":f"Bearer {settings.INTERCOM_TOKEN}","Accept":"application/json","Content-Type":"application/json","Intercom-Version":"2.15"},
    )
    intercom=IntercomGateway(client)
    application=SupportApplication(DecisionPipeline(kb,llm),intercom)
    worker=EventWorker(application)
    app.state.application=application
    await worker.start()

    yield

    await worker.stop()
    await client.aclose()
    await engine.dispose()

app=FastAPI(title="Terra Intercom AI Support v15 Conservative", lifespan=lifespan)
app.include_router(webhook_router)

