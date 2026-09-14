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
# httpx emits full request URLs by default; suppress those debug lines so
# Telegram bot tokens and signed provider URLs cannot leak into production logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.DRY_RUN:
        required = {
            "INTERCOM_TOKEN": settings.INTERCOM_TOKEN,
            "INTERCOM_ADMIN_ID": settings.INTERCOM_ADMIN_ID,
            "INTERCOM_CLIENT_SECRET": settings.INTERCOM_CLIENT_SECRET,
        }
        if settings.LLM_ENABLED:
            required["POLZA_API_KEY"] = settings.POLZA_API_KEY
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                "Live mode requires Intercom configuration: " + ", ".join(missing)
            )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await ensure_schema(engine)

    router_mode = str(settings.LLM_ROUTER_MODE or "taxonomy").strip().lower()
    if router_mode not in {"taxonomy", "legacy"}:
        raise RuntimeError("LLM_ROUTER_MODE must be either 'taxonomy' or 'legacy'")
    if router_mode == "taxonomy" and not settings.LLM_ENABLED:
        raise RuntimeError("LLM_ENABLED must remain true when LLM_ROUTER_MODE=taxonomy")
    kb=KnowledgeBase(
        path=settings.KB_PATH,
        device=settings.EMBEDDING_DEVICE,
        use_reranker=settings.USE_RERANKER,
        enable_retrieval=(router_mode == "legacy"),
    )
    llm = None
    if settings.LLM_ENABLED:
        llm = PolzaLLMClient(
            api_key=settings.POLZA_API_KEY,
            model=settings.POLZA_MODEL,
            temperature=settings.POLZA_TEMPERATURE,
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
    app.state.event_worker=worker
    app.state.outbox_worker=outbox_worker
    await worker.start()
    await outbox_worker.start()

    yield

    await worker.stop()
    await outbox_worker.stop()
    await client.aclose()
    await telegram_client.aclose()
    await engine.dispose()

app=FastAPI(title="Terra Intercom AI Support V28 Production", lifespan=lifespan)
app.include_router(webhook_router)

