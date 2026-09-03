import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.database.base import Base
from app.database.engine import engine
from app.database.schema import ensure_schema
from app.infrastructure.persistence import models as _persistence_models
from app.infrastructure.ai.knowledge_base import KnowledgeBase
from app.infrastructure.ai.llm_client import PolzaLLMClient
from app.infrastructure.intercom.client import IntercomGateway
from app.infrastructure.telegram.notifier import TelegramNotifier
from app.application.decision_pipeline import DecisionPipeline
from app.application.support_service import SupportApplication
from app.application.event_worker import EventWorker
from app.application.outbox_worker import OutboxWorker
from app.api.webhook import router as webhook_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.DRY_RUN:
        missing = [
            name for name, value in {
                "INTERCOM_TOKEN": settings.INTERCOM_TOKEN,
                "INTERCOM_ADMIN_ID": settings.INTERCOM_ADMIN_ID,
                "INTERCOM_CLIENT_SECRET": settings.INTERCOM_CLIENT_SECRET,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Live mode requires Intercom configuration: " + ", ".join(missing)
            )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await ensure_schema(engine)

    kb=KnowledgeBase(path=settings.KB_PATH, device=settings.EMBEDDING_DEVICE, use_reranker=settings.USE_RERANKER)
    llm = None
    if settings.LLM_ENABLED:
        if settings.POLZA_API_KEY:
            llm = PolzaLLMClient(
                api_key=settings.POLZA_API_KEY,
                model=settings.POLZA_MODEL,
                temperature=settings.POLZA_TEMPERATURE,
            )
        else:
            logging.getLogger("main").warning(
                "LLM_ENABLED=true but POLZA_API_KEY is empty; middle-band routing "
                "will fail closed instead of preventing application startup."
            )
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        headers={
            "Authorization": f"Bearer {settings.INTERCOM_TOKEN}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Intercom-Version": "2.16",
        },
    )
    telegram_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))
    intercom = IntercomGateway(client)
    telegram = TelegramNotifier(telegram_client)
    application = SupportApplication(DecisionPipeline(kb, llm), intercom, telegram)
    worker=EventWorker(application)
    outbox_worker=OutboxWorker(intercom, telegram)
    app.state.application=application
    app.state.outbox_worker=outbox_worker
    await worker.start()
    await outbox_worker.start()

    yield

    await worker.stop()
    await outbox_worker.stop()
    await client.aclose()
    await telegram_client.aclose()
    await engine.dispose()

app=FastAPI(title="Terra Intercom AI Support v22.11 Production", lifespan=lifespan)
app.include_router(webhook_router)

