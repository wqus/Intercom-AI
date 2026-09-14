import json
import logging
from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import text
from app.core.config import settings
from app.database.session import SessionLocal
from app.infrastructure.intercom.parser import verify_signature, parse, event_order_at as _event_order_at, CONVERSATION_TOPICS, TICKET_TOPICS
from app.infrastructure.persistence.repository import EventRepository, EscalatedConversationRepository

router = APIRouter()
logger = logging.getLogger("webhook")


def _redact_payload(value):
    """Return a JSON-safe payload copy with credentials/secrets masked.

    Raw webhook logging is a debug feature; customer message bodies are intentionally
    retained so an operator can diagnose parser/entity mismatches.
    """
    secret_keys = {
        "authorization", "token", "access_token", "api_key", "client_secret",
        "x-hub-signature", "secret",
    }
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out[key] = "[REDACTED]" if str(key).lower() in secret_keys else _redact_payload(item)
        return out
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    return value


@router.api_route("/webhook/intercom", methods=["POST", "HEAD"])
async def intercom_webhook(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)

    body = await request.body()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400, "invalid json")

    if settings.INTERCOM_CLIENT_SECRET and not verify_signature(
        body,
        request.headers.get("X-Hub-Signature", ""),
    ):
        raise HTTPException(401, "invalid signature")

    topic = payload.get("topic", "")
    if settings.LOG_WEBHOOK_PAYLOAD:
        logger.info("webhook_payload | topic=%s | payload=%s", topic, json.dumps(_redact_payload(payload), ensure_ascii=False, separators=(",", ":")))
    if topic not in CONVERSATION_TOPICS | TICKET_TOPICS:
        return {"status": "ignored", "topic": topic}

    # A customer-created ticket's `_default_description_` is the initial request.
    # In this workflow Intercom does not emit a separate `ticket.contact.replied`
    # for that first request; the latter is reserved for subsequent customer parts.
    # Therefore ticket.created is a real inbound turn when the parser finds content.
    # Empty/system-only ticket.created events are rejected by parse() and remain no-op.

    message = parse(payload)
    if message is None:
        return {"status": "skipped", "reason": "no_message"}

    async with SessionLocal() as session:
        # Fast path: once an operator owns the thread, do not even enqueue future
        # messages. EventWorker performs the same check as a race-safe defense.
        if await EscalatedConversationRepository(session).is_escalated(message.entity_id):
            logger.info("Ignoring webhook for escalated entity=%s", message.entity_id)
            return {"status": "ignored", "reason": "escalated", "entity_id": message.entity_id}

        inserted = await EventRepository(session).enqueue(
            event_key=message.event_key,
            provider_event_id=message.event_id,
            topic=message.topic,
            entity_id=message.entity_id,
            payload=payload,
            event_order_at=_event_order_at(payload),
        )

    logger.info(
        "Webhook accepted topic=%s entity=%s queued=%s",
        topic,
        message.entity_id,
        inserted,
    )
    return {"status": "accepted" if inserted else "duplicate", "event_key": message.event_key}


@router.get("/health")
async def health():
    """Liveness probe: the process is alive even if dependencies are degraded."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    """Readiness probe for load balancers/orchestrators.

    A process is ready only while both background workers are alive and the DB
    connection can execute a trivial query. This avoids routing traffic to a
    process that can accept webhooks but cannot drain the durable queues.
    """
    outbox_worker = getattr(request.app.state, "outbox_worker", None)
    event_worker = getattr(request.app.state, "event_worker", None)
    workers_ok = bool(
        event_worker is not None
        and outbox_worker is not None
        and getattr(getattr(event_worker, "_task", None), "done", lambda: True)() is False
        and getattr(getattr(outbox_worker, "_task", None), "done", lambda: True)() is False
    )
    db_ok = False
    try:
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        db_ok = False
    if not (workers_ok and db_ok):
        raise HTTPException(503, {"status": "not_ready", "workers": workers_ok, "db": db_ok})
    return {"status": "ready", "workers": True, "db": True}
