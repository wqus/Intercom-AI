import asyncio
import logging
from app.core.config import settings
from app.database.session import SessionLocal
from app.infrastructure.persistence.models import InboundEvent
from app.infrastructure.persistence.repository import EventRepository, EscalatedConversationRepository
from app.infrastructure.intercom.parser import parse
from app.application.support_service import SupportApplication

logger = logging.getLogger("event_worker")


class EventWorker:
    def __init__(self, application: SupportApplication):
        self.application = application
        self._stop = asyncio.Event()
        self._task = None

    async def start(self):
        self._task = asyncio.create_task(self._run(), name="intercom-event-worker")
        logger.info("Event worker started")

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task

    async def _run(self):
        while not self._stop.is_set():
            processed = False
            try:
                async with SessionLocal() as session:
                    repo = EventRepository(session)
                    event = await repo.claim_one()
                    if event:
                        processed = True
                        try:
                            # Database blocklist is checked before parsing/pipeline work.
                            # A webhook race cannot cause the bot to answer after hand-off.
                            if await EscalatedConversationRepository(session).is_escalated(event.entity_id):
                                logger.info(
                                    "Dropping queued event for escalated entity=%s event=%s",
                                    event.entity_id,
                                    event.id,
                                )
                                await repo.mark_done(event)
                                continue

                            message = parse(event.payload)
                            if message is None:
                                await repo.mark_done(event)
                                continue

                            await self.application.handle(session, event, message)
                            await repo.mark_done(event)
                        except Exception as exc:
                            logger.exception("Event processing failed event=%s", event.id)
                            # Never let partially written conversation state or an
                            # escalation block leak into the retry transaction.
                            await session.rollback()
                            await repo.mark_failed(event, str(exc), retry=True)
            except Exception:
                logger.exception("Worker loop failure")
            if not processed:
                await asyncio.sleep(settings.WORKER_POLL_SECONDS)
