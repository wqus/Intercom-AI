import json
import logging
from fastapi import APIRouter, HTTPException, Request, Response
from app.core.config import settings
from app.database.session import SessionLocal
from app.infrastructure.intercom.parser import verify_signature, parse, CONVERSATION_TOPICS, TICKET_TOPICS
from app.infrastructure.persistence.repository import EventRepository

router=APIRouter()
logger=logging.getLogger("webhook")

@router.api_route("/webhook/intercom", methods=["POST","HEAD"])
async def intercom_webhook(request: Request):
    if request.method=="HEAD":
        return Response(status_code=200)

    body=await request.body()
    try:
        payload=json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400,"invalid json")

    if settings.INTERCOM_CLIENT_SECRET and not verify_signature(body,request.headers.get("X-Hub-Signature","")):
        raise HTTPException(401,"invalid signature")

    topic=payload.get("topic","")
    if topic not in CONVERSATION_TOPICS | TICKET_TOPICS:
        return {"status":"ignored","topic":topic}

    message=parse(payload)
    if message is None:
        return {"status":"skipped","reason":"no_message"}

    async with SessionLocal() as session:
        inserted=await EventRepository(session).enqueue(
            event_key=message.event_key,
            provider_event_id=message.event_id,
            topic=message.topic,
            entity_id=message.entity_id,
            payload=payload,
        )

    logger.info("Webhook accepted topic=%s entity=%s queued=%s",topic,message.entity_id,inserted)
    return {"status":"accepted" if inserted else "duplicate","event_key":message.event_key}

@router.get("/health")
async def health():
    return {"status":"ok"}

